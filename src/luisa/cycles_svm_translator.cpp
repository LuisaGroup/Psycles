/* SPDX-FileCopyrightText: 2026 Psycles Contributors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/luisa/cycles_svm_translator.h>

#include "cycles_svm_internal.h"
#include "cycles_svm_volume.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

// DirectJitTranslator crossover: direct structured dispatch wins decisively up
// through ~80 active non-empty shaders per specialized bucket/domain (covering
// Monster 19, BMW27 23, Lone Monk 34, Classroom 76, and 4-bucket Barbershop ~70),
// while unpartitioned mega-tables (> 80 shaders) or extreme procedural graphs
// (> 9000 JIT weight, which blow up GPU PSO compilation) stay compact in
// BytecodeVmTranslator.
inline constexpr std::size_t default_auto_direct_jit_max_shaders = 80u;
inline constexpr std::size_t default_auto_direct_jit_max_weight = 9000u;

[[nodiscard]] std::size_t configured_direct_jit_max_shaders() noexcept {
  const char *raw = std::getenv("PSYCLES_SVM_JIT_MAX_SHADERS");
  if (raw != nullptr && *raw != '\0') {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end != raw && *end == '\0' && parsed > 0ul) {
      return static_cast<std::size_t>(parsed);
    }
  }
  return default_auto_direct_jit_max_shaders;
}

[[nodiscard]] std::size_t configured_direct_jit_max_weight() noexcept {
  const char *raw = std::getenv("PSYCLES_SVM_JIT_MAX_WEIGHT");
  if (raw != nullptr && *raw != '\0') {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end != raw && *end == '\0' && parsed > 0ul) {
      return static_cast<std::size_t>(parsed);
    }
  }
  return default_auto_direct_jit_max_weight;
}

[[nodiscard]] SvmBackendStrategy env_backend_override() noexcept {
  const char *raw = std::getenv("PSYCLES_SVM_BACKEND");
  if (raw == nullptr || *raw == '\0') {
    return SvmBackendStrategy::auto_select;
  }
  const std::string_view value{raw};
  if (value == "vm" || value == "bytecode" || value == "0") {
    return SvmBackendStrategy::bytecode_vm;
  }
  if (value == "jit" || value == "direct" || value == "flat" || value == "1") {
    return SvmBackendStrategy::direct_jit;
  }
  return SvmBackendStrategy::auto_select;
}

[[nodiscard]] bool can_direct_jit(const SvmTranslationRequest &request) noexcept {
  return request.table.valid && request.table.ir.valid &&
         request.table.ir.shaders.size() ==
             static_cast<std::size_t>(request.table.shader_count);
}

[[nodiscard]] bool is_op_enabled_for_mask(
    const SvmOpIr &op, ShaderType shader_type, std::uint32_t node_feature_mask,
    bool shadow_eval_only) noexcept {
  const auto opcode = op.instruction.opcode;
  switch (opcode) {
    case NODE_END:
    case NODE_SHADER_JUMP:
    case NODE_JUMP_IF_ZERO:
    case NODE_JUMP_IF_ONE:
    case NODE_NONE:
    case NODE_PAD1:
      return false;
    case NODE_CLOSURE_EMISSION:
    case NODE_CLOSURE_BACKGROUND:
      return !shadow_eval_only &&
             (node_feature_mask & kernel_feature_node_emission) != 0u;
    case NODE_CLOSURE_HOLDOUT:
      return !shadow_eval_only;
    case NODE_CLOSURE_VOLUME:
    case NODE_VOLUME_COEFFICIENTS:
    case NODE_PRINCIPLED_VOLUME:
      return !shadow_eval_only &&
             (node_feature_mask & kernel_feature_node_volume) != 0u &&
             shader_type == SHADER_TYPE_VOLUME;
    case NODE_CLOSURE_SET_NORMAL:
      return (node_feature_mask & kernel_feature_node_bump) != 0u;
    case NODE_ENTER_BUMP_EVAL:
    case NODE_LEAVE_BUMP_EVAL:
      return (node_feature_mask & kernel_feature_node_bump_state) != 0u;
    case NODE_AOV_START:
    case NODE_AOV_COLOR:
    case NODE_AOV_VALUE:
      return !shadow_eval_only &&
             (node_feature_mask & kernel_feature_node_aov) != 0u;
    case NODE_AMBIENT_OCCLUSION:
    case NODE_BEVEL:
      return !shadow_eval_only &&
             (node_feature_mask & kernel_feature_node_raytrace) != 0u;
    default:
      return true;
  }
}

template<typename T>
[[nodiscard]] std::optional<T> decode_op_payload(
    const SvmInstructionIr &inst,
    std::span<const std::uint32_t> words,
    std::size_t extra_word_offset = 0u) noexcept {
  static_assert(sizeof(T) % sizeof(std::uint32_t) == 0u);
  constexpr std::size_t need_words = sizeof(T) / sizeof(std::uint32_t);
  if (words.empty() ||
      inst.payload_word_count < extra_word_offset + need_words) {
    return std::nullopt;
  }
  const std::size_t start =
      static_cast<std::size_t>(inst.payload_offset) + extra_word_offset;
  if (start + need_words > words.size()) {
    return std::nullopt;
  }
  T out{};
  std::memcpy(&out, words.data() + start, sizeof(T));
  return out;
}

[[nodiscard]] bool is_const_float_ge(SVMInputFloat in,
                                     float threshold) noexcept {
  if ((in.bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    return false;
  }
  float value = 0.0f;
  std::memcpy(&value, &in.bits, sizeof(value));
  return !std::isnan(value) && value >= threshold;
}

[[nodiscard]] bool is_const_float_le(SVMInputFloat in,
                                     float threshold) noexcept {
  if ((in.bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    return false;
  }
  float value = 0.0f;
  std::memcpy(&value, &in.bits, sizeof(value));
  return !std::isnan(value) && value <= threshold;
}

struct SvmLivenessState {
  std::array<bool, SVM_STACK_SIZE> stack{};
  bool closure_weight{false};
  bool normal{false};

  void mark_all() noexcept {
    stack.fill(true);
    closure_weight = true;
    normal = true;
  }

  void merge_from(const SvmLivenessState &other) noexcept {
    for (std::size_t i = 0u; i < SVM_STACK_SIZE; ++i) {
      stack[i] = stack[i] || other.stack[i];
    }
    closure_weight = closure_weight || other.closure_weight;
    normal = normal || other.normal;
  }

  [[nodiscard]] bool any_slot(SVMStackOffset off,
                              std::uint32_t count = 1u) const noexcept {
    if (off >= SVM_STACK_INVALID || count == 0u) {
      return false;
    }
    const std::size_t begin = off;
    const std::size_t end =
        std::min<std::size_t>(begin + count, SVM_STACK_INVALID);
    for (std::size_t i = begin; i < end; ++i) {
      if (stack[i]) {
        return true;
      }
    }
    return false;
  }

  void read_slot(SVMStackOffset off, std::uint32_t count = 1u) noexcept {
    if (off >= SVM_STACK_INVALID || count == 0u) {
      return;
    }
    const std::size_t begin = off;
    const std::size_t end =
        std::min<std::size_t>(begin + count, SVM_STACK_INVALID);
    for (std::size_t i = begin; i < end; ++i) {
      stack[i] = true;
    }
  }

  void kill_slot(SVMStackOffset off, std::uint32_t count = 1u) noexcept {
    if (off >= SVM_STACK_INVALID || count == 0u) {
      return;
    }
    const std::size_t begin = off;
    const std::size_t end =
        std::min<std::size_t>(begin + count, SVM_STACK_INVALID);
    for (std::size_t i = begin; i < end; ++i) {
      stack[i] = false;
    }
  }

  void read_input(SVMInputFloat in, std::uint32_t count = 1u) noexcept {
    if ((in.bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      read_slot(static_cast<SVMStackOffset>(in.bits & 0xffu), count);
    }
  }

  void read_input3(SVMInputFloat3 in, std::uint32_t count = 3u) noexcept {
    if ((in.x.bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      read_slot(static_cast<SVMStackOffset>(in.x.bits & 0xffu), count);
    }
  }
};

void filter_enabled_block(const SvmBlockIr &in_block, ShaderType shader_type,
                          std::uint32_t node_feature_mask,
                          bool shadow_eval_only,
                          SvmBlockIr &out_block) noexcept {
  out_block.clear();
  out_block.reserve(in_block.size());
  for (const auto &op : in_block) {
    if (op.is_conditional) {
      SvmOpIr filtered = op;
      filtered.children.clear();
      filter_enabled_block(op.children, shader_type, node_feature_mask,
                           shadow_eval_only, filtered.children);
      if (!filtered.children.empty()) {
        out_block.emplace_back(std::move(filtered));
      }
      continue;
    }
    if (is_op_enabled_for_mask(op, shader_type, node_feature_mask,
                               shadow_eval_only)) {
      out_block.push_back(op);
    }
  }
}

[[nodiscard]] bool is_principled_shadow_thin_wall_inactive(
    const SVMNodePrincipledBsdfData &data) noexcept {
  if (data.thin_wall.offset == SVM_STACK_INVALID && data.thin_wall.value == 0) {
    return true;
  }
  return is_const_float_le(data.transmission_weight, 0.0f);
}

[[nodiscard]] bool is_principled_shadow_opaque(
    const SVMNodePrincipledBsdfData &data) noexcept {
  return is_principled_shadow_thin_wall_inactive(data) &&
         is_const_float_ge(data.alpha, 1.0f);
}

void run_shadow_dce_block(const SvmBlockIr &in_block,
                          std::span<const std::uint32_t> words,
                          SvmLivenessState &live,
                          SvmBlockIr &out_block) noexcept {
  luisa::vector<SvmOpIr> reversed;
  reversed.reserve(in_block.size());

  for (auto it = in_block.rbegin(); it != in_block.rend(); ++it) {
    const auto &op = *it;
    if (op.is_conditional) {
      SvmLivenessState child_live = live;
      SvmBlockIr pruned_children;
      run_shadow_dce_block(op.children, words, child_live, pruned_children);
      if (!pruned_children.empty()) {
        SvmOpIr kept = op;
        kept.children = std::move(pruned_children);
        live.merge_from(child_live);
        live.read_slot(op.cond_stack_offset, 1u);
        reversed.emplace_back(std::move(kept));
      }
      continue;
    }

    const auto &inst = op.instruction;
    const auto keep_unknown = [&]() noexcept {
      live.mark_all();
      reversed.push_back(op);
    };

    switch (inst.opcode) {
      case NODE_END:
      case NODE_SHADER_JUMP:
      case NODE_JUMP_IF_ZERO:
      case NODE_JUMP_IF_ONE:
      case NODE_NONE:
      case NODE_PAD1:
      case NODE_CLOSURE_EMISSION:
      case NODE_CLOSURE_BACKGROUND:
      case NODE_CLOSURE_HOLDOUT:
      case NODE_CLOSURE_VOLUME:
      case NODE_VOLUME_COEFFICIENTS:
      case NODE_PRINCIPLED_VOLUME:
      case NODE_AOV_START:
      case NODE_AOV_COLOR:
      case NODE_AOV_VALUE:
      case NODE_BEVEL:
        break;

      case NODE_CLOSURE_BSDF: {
        const auto bsdf = decode_op_payload<SVMNodeClosureBsdf>(inst, words);
        if (!bsdf) {
          keep_unknown();
          break;
        }
        constexpr std::size_t hdr_words =
            sizeof(SVMNodeClosureBsdf) / sizeof(std::uint32_t);
        const auto ct = static_cast<ClosureType>(bsdf->closure_type);
        if (ct == CLOSURE_BSDF_TRANSPARENT_ID) {
          live.read_slot(bsdf->mix_weight_offset, 1u);
          live.closure_weight = true;
          reversed.push_back(op);
        } else if (ct == CLOSURE_BSDF_RAY_PORTAL_ID) {
          const auto portal = decode_op_payload<SVMNodeRayPortalBsdfData>(
              inst, words, hdr_words);
          if (!portal) {
            keep_unknown();
            break;
          }
          live.read_slot(bsdf->mix_weight_offset, 1u);
          live.closure_weight = true;
          live.read_input3(portal->direction, 3u);
          live.read_slot(portal->position_offset, 3u);
          live.normal = true;
          reversed.push_back(op);
        } else if (ct == CLOSURE_BSDF_PRINCIPLED_ID) {
          const auto p = decode_op_payload<SVMNodePrincipledBsdfData>(
              inst, words, hdr_words);
          if (!p) {
            keep_unknown();
            break;
          }
          if (!is_principled_shadow_thin_wall_inactive(*p)) {
            keep_unknown();
          } else if (!is_const_float_ge(p->alpha, 1.0f)) {
            live.read_slot(bsdf->mix_weight_offset, 1u);
            live.read_input(p->alpha, 1u);
            reversed.push_back(op);
          }
        } else {
          switch (ct) {
            case CLOSURE_BSDF_DIFFUSE_ID:
            case CLOSURE_BSDF_TRANSLUCENT_ID:
            case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
            case CLOSURE_BSDF_SHEEN_ID:
            case CLOSURE_BSDF_PHYSICAL_CONDUCTOR:
            case CLOSURE_BSDF_F82_CONDUCTOR:
            case CLOSURE_BSDF_MICROFACET_GGX_ID:
            case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
            case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
            case CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID:
            case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
            case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
            case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
            case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
            case CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID:
            case CLOSURE_BSDF_GLOSSY_TOON_ID:
            case CLOSURE_BSDF_DIFFUSE_TOON_ID:
            case CLOSURE_BSDF_HAIR_REFLECTION_ID:
            case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
            case CLOSURE_BSDF_HAIR_CHIANG_ID:
            case CLOSURE_BSDF_HAIR_HUANG_ID:
            case CLOSURE_BSSRDF_BURLEY_ID:
            case CLOSURE_BSSRDF_RANDOM_WALK_ID:
            case CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID:
            case CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID:
              break;
            default:
              keep_unknown();
              break;
          }
        }
        break;
      }

      case NODE_CLOSURE_SET_WEIGHT: {
        if (live.closure_weight) {
          live.closure_weight = false;
          reversed.push_back(op);
        }
        break;
      }
      case NODE_CLOSURE_WEIGHT: {
        const auto n = decode_op_payload<SVMNodeClosureWeight>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.closure_weight) {
          live.closure_weight = false;
          live.read_slot(n->weight_offset, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_EMISSION_WEIGHT: {
        const auto n = decode_op_payload<SVMNodeEmissionWeight>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.closure_weight) {
          live.closure_weight = false;
          live.read_input3(n->color, 3u);
          live.read_input(n->strength, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MIX_CLOSURE: {
        const auto n = decode_op_payload<SVMNodeMixClosure>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->weight1_offset, 1u) ||
            live.any_slot(n->weight2_offset, 1u)) {
          live.kill_slot(n->weight1_offset, 1u);
          live.kill_slot(n->weight2_offset, 1u);
          live.read_input(n->fac, 1u);
          live.read_slot(n->in_weight_offset, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_CLOSURE_SET_NORMAL: {
        const auto n = decode_op_payload<SVMNodeClosureSetNormal>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.normal || live.any_slot(n->normal_offset, 3u)) {
          live.normal = false;
          live.kill_slot(n->normal_offset, 3u);
          live.read_slot(n->direction_offset, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_SET_BUMP: {
        const auto n = decode_op_payload<SVMNodeSetBump>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 3u)) {
          live.kill_slot(n->out_offset, 3u);
          live.read_input(n->scale, 1u);
          live.read_input(n->strength, 1u);
          live.read_slot(n->normal_offset, 3u);
          if (n->normal_offset == SVM_STACK_INVALID) {
            live.normal = true;
          }
          live.read_slot(n->center_offset, 1u);
          live.read_slot(n->dx_offset, 1u);
          live.read_slot(n->dy_offset, 1u);
          if (n->bump_state_offset != SVM_STACK_INVALID) {
            live.read_slot(
                static_cast<SVMStackOffset>(
                    std::min<std::uint32_t>(n->bump_state_offset + 4u, 254u)),
                6u);
          }
          reversed.push_back(op);
        }
        break;
      }
      case NODE_ENTER_BUMP_EVAL: {
        const auto n = decode_op_payload<SVMNodeEnterBumpEval>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        live.kill_slot(n->state_offset, SVM_BUMP_EVAL_STATE_SIZE);
        live.normal = true;
        reversed.push_back(op);
        break;
      }
      case NODE_LEAVE_BUMP_EVAL: {
        const auto n = decode_op_payload<SVMNodeLeaveBumpEval>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        live.read_slot(n->state_offset, SVM_BUMP_EVAL_STATE_SIZE);
        reversed.push_back(op);
        break;
      }
      case NODE_MATH: {
        const auto n = decode_op_payload<SVMNodeMath>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 1u)) {
          live.kill_slot(n->result_offset, 1u);
          live.read_input(n->value1, 1u);
          live.read_input(n->value2, 1u);
          live.read_input(n->value3, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_CLAMP: {
        const auto n = decode_op_payload<SVMNodeClamp>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 1u)) {
          live.kill_slot(n->result_offset, 1u);
          live.read_input(n->min, 1u);
          live.read_input(n->max, 1u);
          live.read_input(n->value, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MAPPING:
      case NODE_MAPPING_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeMapping>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t s =
            inst.opcode == NODE_MAPPING_DERIVATIVE ? 9u : 3u;
        if (live.any_slot(n->result_offset, s)) {
          live.kill_slot(n->result_offset, s);
          live.read_input3(n->vector, s);
          live.read_input3(n->location, s);
          live.read_input3(n->rotation, s);
          live.read_input3(n->scale, s);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEXTURE_MAPPING:
      case NODE_TEXTURE_MAPPING_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeTextureMapping>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t s =
            inst.opcode == NODE_TEXTURE_MAPPING_DERIVATIVE ? 9u : 3u;
        if (live.any_slot(n->out_offset, s)) {
          live.kill_slot(n->out_offset, s);
          live.read_slot(n->vec_offset, s);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MIN_MAX: {
        const auto n = decode_op_payload<SVMNodeMinMax>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 3u)) {
          live.kill_slot(n->out_offset, 3u);
          live.read_slot(n->vec_offset, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_NORMAL_MAP: {
        const auto n = decode_op_payload<SVMNodeNormalMap>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->normal_offset, 3u)) {
          live.kill_slot(n->normal_offset, 3u);
          live.read_input3(n->color, 3u);
          live.read_input(n->strength, 1u);
          live.normal = true;
          reversed.push_back(op);
        }
        break;
      }
      case NODE_VALUE_F:
      case NODE_VALUE_F_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeValueF>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t s =
            inst.opcode == NODE_VALUE_F_DERIVATIVE ? 3u : 1u;
        if (live.any_slot(n->out_offset, s)) {
          live.kill_slot(n->out_offset, s);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_VALUE_V:
      case NODE_VALUE_V_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeValueV>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t s =
            inst.opcode == NODE_VALUE_V_DERIVATIVE ? 9u : 3u;
        if (live.any_slot(n->out_offset, s)) {
          live.kill_slot(n->out_offset, s);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_CONVERT:
      case NODE_CONVERT_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeConvert>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const bool deriv = inst.opcode == NODE_CONVERT_DERIVATIVE;
        std::uint32_t in_slots = 1u;
        std::uint32_t out_slots = 1u;
        switch (n->convert_type) {
          case NODE_CONVERT_FV:
            in_slots = deriv ? 3u : 1u;
            out_slots = deriv ? 9u : 3u;
            break;
          case NODE_CONVERT_IV:
            in_slots = 1u;
            out_slots = deriv ? 9u : 3u;
            break;
          case NODE_CONVERT_CF:
          case NODE_CONVERT_VF:
            in_slots = deriv ? 9u : 3u;
            out_slots = deriv ? 3u : 1u;
            break;
          case NODE_CONVERT_CI:
          case NODE_CONVERT_VI:
            in_slots = 3u;
            out_slots = 1u;
            break;
          case NODE_CONVERT_IF:
            in_slots = 1u;
            out_slots = deriv ? 3u : 1u;
            break;
          default:
            in_slots = 1u;
            out_slots = 1u;
            break;
        }
        if (live.any_slot(n->to_offset, out_slots)) {
          live.kill_slot(n->to_offset, out_slots);
          live.read_slot(n->from_offset, in_slots);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_GEOMETRY:
      case NODE_GEOMETRY_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeGeometry>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t s =
            (inst.opcode == NODE_GEOMETRY_DERIVATIVE &&
             n->store_derivatives != 0u)
                ? 9u
                : 3u;
        if (live.any_slot(n->out_offset, s)) {
          live.kill_slot(n->out_offset, s);
          if (n->geom_type == NODE_GEOM_N || n->geom_type == NODE_GEOM_T ||
              n->geom_type == NODE_GEOM_I) {
            live.normal = true;
          }
          reversed.push_back(op);
        }
        break;
      }
      case NODE_OBJECT_INFO:
      case NODE_PARTICLE_INFO:
      case NODE_HAIR_INFO:
      case NODE_POINT_INFO:
      case NODE_LIGHT_PATH: {
        static_assert(sizeof(SVMNodeObjectInfo) == sizeof(SVMNodeParticleInfo) &&
                      sizeof(SVMNodeObjectInfo) == sizeof(SVMNodeHairInfo) &&
                      sizeof(SVMNodeObjectInfo) == sizeof(SVMNodePointInfo) &&
                      sizeof(SVMNodeObjectInfo) == sizeof(SVMNodeLightPath));
        const auto n = decode_op_payload<SVMNodeObjectInfo>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        std::uint32_t slots = 1u;
        if (inst.opcode == NODE_OBJECT_INFO) {
          if (n->info_type == NODE_INFO_OB_LOCATION ||
              n->info_type == NODE_INFO_OB_COLOR) {
            slots = 3u;
          }
        } else if (inst.opcode == NODE_PARTICLE_INFO) {
          const auto pt = static_cast<NodeParticleInfo>(n->info_type);
          if (pt == NODE_INFO_PAR_LOCATION || pt == NODE_INFO_PAR_VELOCITY ||
              pt == NODE_INFO_PAR_ANGULAR_VELOCITY) {
            slots = 3u;
          }
        } else if (inst.opcode == NODE_HAIR_INFO) {
          if (static_cast<NodeHairInfo>(n->info_type) ==
              NODE_INFO_CURVE_TANGENT_NORMAL) {
            slots = 3u;
          }
        } else if (inst.opcode == NODE_POINT_INFO) {
          if (static_cast<NodePointInfo>(n->info_type) ==
              NODE_INFO_POINT_POSITION) {
            slots = 3u;
          }
        }
        if (live.any_slot(n->out_offset, slots)) {
          live.kill_slot(n->out_offset, slots);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_ATTR:
      case NODE_ATTR_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeAttr>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const bool deriv =
            inst.opcode == NODE_ATTR_DERIVATIVE && n->store_derivatives != 0u;
        const std::uint32_t s =
            (n->output_type == NODE_ATTR_OUTPUT_FLOAT3 ? 3u : 1u) *
            (deriv ? 3u : 1u);
        if (live.any_slot(n->out_offset, s)) {
          live.kill_slot(n->out_offset, s);
          live.normal = true;
          reversed.push_back(op);
        }
        break;
      }
      case NODE_VERTEX_COLOR:
      case NODE_VERTEX_COLOR_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeVertexColor>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u) ||
            live.any_slot(n->alpha_offset, 1u)) {
          live.kill_slot(n->color_offset, 3u);
          live.kill_slot(n->alpha_offset, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_COORD:
      case NODE_TEX_COORD_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeTexCoord>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t s =
            (inst.opcode == NODE_TEX_COORD_DERIVATIVE &&
             n->store_derivatives != 0u)
                ? 9u
                : 3u;
        if (live.any_slot(n->out_offset, s)) {
          live.kill_slot(n->out_offset, s);
          if (n->texco_type == NODE_TEXCO_NORMAL ||
              n->texco_type == NODE_TEXCO_REFLECTION) {
            live.normal = true;
          }
          reversed.push_back(op);
        }
        break;
      }
      case NODE_GAMMA: {
        const auto n = decode_op_payload<SVMNodeGamma>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 3u)) {
          live.kill_slot(n->out_offset, 3u);
          live.read_input3(n->color, 3u);
          live.read_input(n->gamma, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_BRIGHTCONTRAST: {
        const auto n = decode_op_payload<SVMNodeBrightContrast>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 3u)) {
          live.kill_slot(n->out_offset, 3u);
          live.read_input3(n->color, 3u);
          live.read_input(n->bright, 1u);
          live.read_input(n->contrast, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_INVERT: {
        const auto n = decode_op_payload<SVMNodeInvert>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 3u)) {
          live.kill_slot(n->out_offset, 3u);
          live.read_input3(n->color, 3u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_FRESNEL: {
        const auto n = decode_op_payload<SVMNodeFresnel>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 1u)) {
          live.kill_slot(n->out_offset, 1u);
          live.read_input(n->ior, 1u);
          live.read_slot(n->normal_offset, 3u);
          if (n->normal_offset == SVM_STACK_INVALID) {
            live.normal = true;
          }
          reversed.push_back(op);
        }
        break;
      }
      case NODE_LAYER_WEIGHT: {
        const auto n = decode_op_payload<SVMNodeLayerWeight>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 1u)) {
          live.kill_slot(n->out_offset, 1u);
          live.read_input(n->blend, 1u);
          live.read_slot(n->normal_offset, 3u);
          if (n->normal_offset == SVM_STACK_INVALID) {
            live.normal = true;
          }
          reversed.push_back(op);
        }
        break;
      }
      case NODE_HSV: {
        const auto n = decode_op_payload<SVMNodeHSV>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_color_offset, 3u)) {
          live.kill_slot(n->out_color_offset, 3u);
          live.read_input3(n->color, 3u);
          live.read_input(n->hue, 1u);
          live.read_input(n->sat, 1u);
          live.read_input(n->val, 1u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_CAMERA: {
        const auto n = decode_op_payload<SVMNodeCamera>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->vector_offset, 3u) ||
            live.any_slot(n->zdepth_offset, 1u) ||
            live.any_slot(n->distance_offset, 1u)) {
          live.kill_slot(n->vector_offset, 3u);
          live.kill_slot(n->zdepth_offset, 1u);
          live.kill_slot(n->distance_offset, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_BLACKBODY: {
        const auto n = decode_op_payload<SVMNodeBlackbody>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u)) {
          live.kill_slot(n->color_offset, 3u);
          live.read_input(n->temperature, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_WAVELENGTH: {
        const auto n = decode_op_payload<SVMNodeWavelength>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u)) {
          live.kill_slot(n->color_offset, 3u);
          live.read_input(n->wavelength, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_VECTOR_MATH:
      case NODE_VECTOR_MATH_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeVectorMath>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const bool deriv = inst.opcode == NODE_VECTOR_MATH_DERIVATIVE;
        const std::uint32_t f1 = deriv ? 3u : 1u;
        const std::uint32_t f3 = deriv ? 9u : 3u;
        if (live.any_slot(n->value_offset, f1) ||
            live.any_slot(n->vector_offset, f3)) {
          live.kill_slot(n->value_offset, f1);
          live.kill_slot(n->vector_offset, f3);
          live.read_input3(n->a, f3);
          live.read_input3(n->b, f3);
          live.read_input3(n->c, f3);
          live.read_input(n->param1, f1);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_VECTOR_ROTATE: {
        const auto n = decode_op_payload<SVMNodeVectorRotate>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 3u)) {
          live.kill_slot(n->result_offset, 3u);
          live.read_input3(n->vector, 3u);
          live.read_input3(n->center, 3u);
          live.read_input3(n->axis, 3u);
          live.read_input3(n->rotation, 3u);
          live.read_input(n->angle, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_VECTOR_TRANSFORM: {
        const auto n = decode_op_payload<SVMNodeVectorTransform>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->vector_out_offset, 3u)) {
          live.kill_slot(n->vector_out_offset, 3u);
          live.read_input3(n->vector_in, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_SEPARATE_VECTOR:
      case NODE_SEPARATE_VECTOR_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeSeparateVector>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const bool deriv = inst.opcode == NODE_SEPARATE_VECTOR_DERIVATIVE;
        const std::uint32_t f1 = deriv ? 3u : 1u;
        const std::uint32_t f3 = deriv ? 9u : 3u;
        if (live.any_slot(n->out_offset, f1)) {
          live.kill_slot(n->out_offset, f1);
          live.read_input3(n->vector, f3);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_COMBINE_VECTOR:
      case NODE_COMBINE_VECTOR_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeCombineVector>(inst, words);
        if (!n || n->vector_index >= 3u) {
          keep_unknown();
          break;
        }
        const bool deriv = inst.opcode == NODE_COMBINE_VECTOR_DERIVATIVE;
        if (n->out_offset < SVM_STACK_INVALID) {
          const auto s0 = static_cast<SVMStackOffset>(
              std::min<std::uint32_t>(n->out_offset + n->vector_index, 255u));
          bool used = live.any_slot(s0, 1u);
          if (deriv) {
            const auto s3 = static_cast<SVMStackOffset>(
                std::min<std::uint32_t>(s0 + 3u, 255u));
            const auto s6 = static_cast<SVMStackOffset>(
                std::min<std::uint32_t>(s0 + 6u, 255u));
            used = used || live.any_slot(s3, 1u) || live.any_slot(s6, 1u);
          }
          if (used) {
            live.kill_slot(s0, 1u);
            if (deriv) {
              if (s0 + 3u < SVM_STACK_INVALID) {
                live.kill_slot(static_cast<SVMStackOffset>(s0 + 3u), 1u);
              }
              if (s0 + 6u < SVM_STACK_INVALID) {
                live.kill_slot(static_cast<SVMStackOffset>(s0 + 6u), 1u);
              }
            }
            live.read_input(n->in, deriv ? 3u : 1u);
            reversed.push_back(op);
          }
        }
        break;
      }
      case NODE_SEPARATE_COLOR: {
        const auto n = decode_op_payload<SVMNodeSeparateColor>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->red_offset, 1u) ||
            live.any_slot(n->green_offset, 1u) ||
            live.any_slot(n->blue_offset, 1u)) {
          live.kill_slot(n->red_offset, 1u);
          live.kill_slot(n->green_offset, 1u);
          live.kill_slot(n->blue_offset, 1u);
          live.read_input3(n->color, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_COMBINE_COLOR: {
        const auto n = decode_op_payload<SVMNodeCombineColor>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u)) {
          live.kill_slot(n->color_offset, 3u);
          live.read_input(n->red, 1u);
          live.read_input(n->green, 1u);
          live.read_input(n->blue, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MIX_COLOR: {
        const auto n = decode_op_payload<SVMNodeMixColor>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 3u)) {
          live.kill_slot(n->result_offset, 3u);
          live.read_input3(n->a, 3u);
          live.read_input3(n->b, 3u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MIX_FLOAT: {
        const auto n = decode_op_payload<SVMNodeMixFloat>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 1u)) {
          live.kill_slot(n->result_offset, 1u);
          live.read_input(n->fac, 1u);
          live.read_input(n->a, 1u);
          live.read_input(n->b, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MIX: {
        const auto n = decode_op_payload<SVMNodeMix>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 3u)) {
          live.kill_slot(n->result_offset, 3u);
          live.read_input3(n->c1, 3u);
          live.read_input3(n->c2, 3u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MIX_VECTOR: {
        const auto n = decode_op_payload<SVMNodeMixVector>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 3u)) {
          live.kill_slot(n->result_offset, 3u);
          live.read_input3(n->a, 3u);
          live.read_input3(n->b, 3u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MIX_VECTOR_NON_UNIFORM: {
        const auto n =
            decode_op_payload<SVMNodeMixVectorNonUniform>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 3u)) {
          live.kill_slot(n->result_offset, 3u);
          live.read_input3(n->a, 3u);
          live.read_input3(n->b, 3u);
          live.read_input3(n->fac, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_MAP_RANGE: {
        const auto n = decode_op_payload<SVMNodeMapRange>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 1u)) {
          live.kill_slot(n->result_offset, 1u);
          live.read_input(n->value, 1u);
          live.read_input(n->from_min, 1u);
          live.read_input(n->from_max, 1u);
          live.read_input(n->to_min, 1u);
          live.read_input(n->to_max, 1u);
          live.read_input(n->steps, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_VECTOR_MAP_RANGE: {
        const auto n = decode_op_payload<SVMNodeVectorMapRange>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->result_offset, 3u)) {
          live.kill_slot(n->result_offset, 3u);
          live.read_input3(n->value, 3u);
          live.read_input3(n->from_min, 3u);
          live.read_input3(n->from_max, 3u);
          live.read_input3(n->to_min, 3u);
          live.read_input3(n->to_max, 3u);
          live.read_input3(n->steps, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_NORMAL: {
        const auto n = decode_op_payload<SVMNodeNormal>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_normal_offset, 3u) ||
            live.any_slot(n->out_dot_offset, 1u)) {
          live.kill_slot(n->out_normal_offset, 3u);
          live.kill_slot(n->out_dot_offset, 1u);
          live.read_input3(n->in_normal, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_NOISE: {
        const auto n = decode_op_payload<SVMNodeTexNoise>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->value_offset, 1u) ||
            live.any_slot(n->color_offset, 3u)) {
          live.kill_slot(n->value_offset, 1u);
          live.kill_slot(n->color_offset, 3u);
          live.read_input(n->w, 1u);
          live.read_input(n->scale, 1u);
          live.read_input(n->detail, 1u);
          live.read_input(n->roughness, 1u);
          live.read_input(n->lacunarity, 1u);
          live.read_input(n->offset, 1u);
          live.read_input(n->gain, 1u);
          live.read_input(n->distortion, 1u);
          live.read_slot(n->vector, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_VORONOI: {
        const auto n = decode_op_payload<SVMNodeTexVoronoi>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->distance_offset, 1u) ||
            live.any_slot(n->color_offset, 3u) ||
            live.any_slot(n->position_offset, 3u) ||
            live.any_slot(n->w_out_offset, 1u) ||
            live.any_slot(n->radius_offset, 1u)) {
          live.kill_slot(n->distance_offset, 1u);
          live.kill_slot(n->color_offset, 3u);
          live.kill_slot(n->position_offset, 3u);
          live.kill_slot(n->w_out_offset, 1u);
          live.kill_slot(n->radius_offset, 1u);
          live.read_input(n->w, 1u);
          live.read_input(n->scale, 1u);
          live.read_input(n->detail, 1u);
          live.read_input(n->roughness, 1u);
          live.read_input(n->lacunarity, 1u);
          live.read_input(n->smoothness, 1u);
          live.read_input(n->exponent, 1u);
          live.read_input(n->randomness, 1u);
          live.read_slot(n->coord, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_GABOR: {
        const auto n = decode_op_payload<SVMNodeTexGabor>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->value_offset, 1u) ||
            live.any_slot(n->phase_offset, 1u) ||
            live.any_slot(n->intensity_offset, 1u)) {
          live.kill_slot(n->value_offset, 1u);
          live.kill_slot(n->phase_offset, 1u);
          live.kill_slot(n->intensity_offset, 1u);
          live.read_input3(n->orientation_3d, 3u);
          live.read_input(n->scale, 1u);
          live.read_input(n->frequency, 1u);
          live.read_input(n->anisotropy, 1u);
          live.read_input(n->orientation_2d, 1u);
          live.read_slot(n->coordinates, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_WAVE: {
        const auto n = decode_op_payload<SVMNodeTexWave>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u) ||
            live.any_slot(n->fac_offset, 1u)) {
          live.kill_slot(n->color_offset, 3u);
          live.kill_slot(n->fac_offset, 1u);
          live.read_input(n->scale, 1u);
          live.read_input(n->distortion, 1u);
          live.read_input(n->detail, 1u);
          live.read_input(n->dscale, 1u);
          live.read_input(n->droughness, 1u);
          live.read_input(n->phase, 1u);
          live.read_slot(n->co, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_MAGIC: {
        const auto n = decode_op_payload<SVMNodeTexMagic>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u) ||
            live.any_slot(n->fac_offset, 1u)) {
          live.kill_slot(n->color_offset, 3u);
          live.kill_slot(n->fac_offset, 1u);
          live.read_input(n->scale, 1u);
          live.read_input(n->distortion, 1u);
          live.read_slot(n->co, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_CHECKER: {
        const auto n = decode_op_payload<SVMNodeTexChecker>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u) ||
            live.any_slot(n->fac_offset, 1u)) {
          live.kill_slot(n->color_offset, 3u);
          live.kill_slot(n->fac_offset, 1u);
          live.read_input3(n->color1, 3u);
          live.read_input3(n->color2, 3u);
          live.read_input(n->scale, 1u);
          live.read_slot(n->co, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_BRICK: {
        const auto n = decode_op_payload<SVMNodeTexBrick>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u) ||
            live.any_slot(n->fac_offset, 1u)) {
          live.kill_slot(n->color_offset, 3u);
          live.kill_slot(n->fac_offset, 1u);
          live.read_input3(n->color1, 3u);
          live.read_input3(n->color2, 3u);
          live.read_input3(n->mortar, 3u);
          live.read_input(n->scale, 1u);
          live.read_input(n->mortar_size, 1u);
          live.read_input(n->bias, 1u);
          live.read_input(n->brick_width, 1u);
          live.read_input(n->row_height, 1u);
          live.read_input(n->mortar_smooth, 1u);
          live.read_slot(n->co, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_WHITE_NOISE: {
        const auto n = decode_op_payload<SVMNodeTexWhiteNoise>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->value_offset, 1u) ||
            live.any_slot(n->color_offset, 3u)) {
          live.kill_slot(n->value_offset, 1u);
          live.kill_slot(n->color_offset, 3u);
          live.read_input3(n->vector, 3u);
          live.read_input(n->w, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_GRADIENT: {
        const auto n = decode_op_payload<SVMNodeTexGradient>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->fac_offset, 1u) ||
            live.any_slot(n->color_offset, 3u)) {
          live.kill_slot(n->fac_offset, 1u);
          live.kill_slot(n->color_offset, 3u);
          live.read_slot(n->co, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_IMAGE:
      case NODE_TEX_IMAGE_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeTexImage>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t in_s =
            inst.opcode == NODE_TEX_IMAGE_DERIVATIVE ? 9u : 3u;
        if (live.any_slot(n->out_offset, 3u) ||
            live.any_slot(n->alpha_offset, 1u)) {
          live.kill_slot(n->out_offset, 3u);
          live.kill_slot(n->alpha_offset, 1u);
          live.read_slot(n->co, in_s);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_IMAGE_BOX:
      case NODE_TEX_IMAGE_BOX_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeTexImageBox>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t in_s =
            inst.opcode == NODE_TEX_IMAGE_BOX_DERIVATIVE ? 9u : 3u;
        if (live.any_slot(n->out_offset, 3u) ||
            live.any_slot(n->alpha_offset, 1u)) {
          live.kill_slot(n->out_offset, 3u);
          live.kill_slot(n->alpha_offset, 1u);
          live.read_slot(n->co, in_s);
          live.normal = true;
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_ENVIRONMENT:
      case NODE_TEX_ENVIRONMENT_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeTexEnvironment>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t in_s =
            inst.opcode == NODE_TEX_ENVIRONMENT_DERIVATIVE ? 9u : 3u;
        if (live.any_slot(n->out_offset, 3u) ||
            live.any_slot(n->alpha_offset, 1u)) {
          live.kill_slot(n->out_offset, 3u);
          live.kill_slot(n->alpha_offset, 1u);
          live.read_slot(n->co, in_s);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TEX_SKY: {
        const auto n = decode_op_payload<SVMNodeTexSky>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 3u)) {
          live.kill_slot(n->out_offset, 3u);
          live.read_slot(n->dir_offset, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_RGB_RAMP: {
        const auto n = decode_op_payload<SVMNodeRGBRamp>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->color_offset, 3u) ||
            live.any_slot(n->alpha_offset, 1u)) {
          live.kill_slot(n->color_offset, 3u);
          live.kill_slot(n->alpha_offset, 1u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_CURVES: {
        const auto n = decode_op_payload<SVMNodeCurves>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 3u)) {
          live.kill_slot(n->out_offset, 3u);
          live.read_input3(n->color, 3u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_FLOAT_CURVE: {
        const auto n = decode_op_payload<SVMNodeFloatCurve>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 1u)) {
          live.kill_slot(n->out_offset, 1u);
          live.read_input(n->value_in, 1u);
          live.read_input(n->fac, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_TANGENT:
      case NODE_TANGENT_DERIVATIVE: {
        const auto n = decode_op_payload<SVMNodeTangent>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        const std::uint32_t s =
            inst.opcode == NODE_TANGENT_DERIVATIVE ? 9u : 3u;
        if (live.any_slot(n->tangent_offset, s)) {
          live.kill_slot(n->tangent_offset, s);
          live.normal = true;
          reversed.push_back(op);
        }
        break;
      }
      case NODE_WIREFRAME: {
        const auto n = decode_op_payload<SVMNodeWireframe>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_fac_offset, 1u)) {
          live.kill_slot(n->out_fac_offset, 1u);
          live.read_input(n->in_size, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_IES: {
        const auto n = decode_op_payload<SVMNodeIES>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->fac_offset, 1u)) {
          live.kill_slot(n->fac_offset, 1u);
          live.read_input(n->strength, 1u);
          live.read_slot(n->vector_offset, 3u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_LIGHT_FALLOFF: {
        const auto n = decode_op_payload<SVMNodeLightFalloff>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_offset, 1u)) {
          live.kill_slot(n->out_offset, 1u);
          live.read_input(n->strength, 1u);
          live.read_input(n->smooth, 1u);
          reversed.push_back(op);
        }
        break;
      }
      case NODE_AMBIENT_OCCLUSION: {
        const auto n = decode_op_payload<SVMNodeAmbientOcclusion>(inst, words);
        if (!n) {
          keep_unknown();
          break;
        }
        if (live.any_slot(n->out_ao_offset, 1u) ||
            live.any_slot(n->out_color_offset, 3u)) {
          live.kill_slot(n->out_ao_offset, 1u);
          live.kill_slot(n->out_color_offset, 3u);
          live.read_input3(n->color, 3u);
          live.read_input(n->dist, 1u);
          live.read_slot(n->normal_offset, 3u);
          if (n->normal_offset == SVM_STACK_INVALID) {
            live.normal = true;
          }
          reversed.push_back(op);
        }
        break;
      }
      default:
        keep_unknown();
        break;
    }
  }

  out_block.assign(reversed.rbegin(), reversed.rend());
}

[[nodiscard]] bool strip_empty_bump_eval_pairs(
    SvmBlockIr &block, std::span<const std::uint32_t> words) noexcept {
  bool removed_any = false;
  for (auto &op : block) {
    if (op.is_conditional) {
      removed_any =
          strip_empty_bump_eval_pairs(op.children, words) || removed_any;
    }
  }
  SvmBlockIr cleaned;
  cleaned.reserve(block.size());
  for (std::size_t i = 0u; i < block.size(); ++i) {
    if (i + 1u < block.size() && !block[i].is_conditional &&
        !block[i + 1u].is_conditional &&
        block[i].instruction.opcode == NODE_ENTER_BUMP_EVAL &&
        block[i + 1u].instruction.opcode == NODE_LEAVE_BUMP_EVAL) {
      const auto enter = decode_op_payload<SVMNodeEnterBumpEval>(
          block[i].instruction, words);
      const auto leave = decode_op_payload<SVMNodeLeaveBumpEval>(
          block[i + 1u].instruction, words);
      if (enter && leave && enter->state_offset == leave->state_offset) {
        removed_any = true;
        ++i;
        continue;
      }
    }
    cleaned.emplace_back(std::move(block[i]));
  }
  block = std::move(cleaned);
  return removed_any;
}

[[nodiscard]] SvmBlockIr prune_svm_block_for_eval(
    const SvmBlockIr &block, std::span<const std::uint32_t> words,
    ShaderType shader_type, std::uint32_t node_feature_mask,
    bool shadow_eval_only) noexcept {
  SvmBlockIr enabled;
  filter_enabled_block(block, shader_type, node_feature_mask, shadow_eval_only,
                       enabled);
  if (!shadow_eval_only || shader_type != SHADER_TYPE_SURFACE ||
      enabled.empty()) {
    return enabled;
  }
  SvmBlockIr pruned;
  SvmLivenessState live{};
  run_shadow_dce_block(enabled, words, live, pruned);
  bool stripped = false;
  while (strip_empty_bump_eval_pairs(pruned, words)) {
    stripped = true;
  }
  if (stripped && !pruned.empty()) {
    SvmBlockIr rerun;
    SvmLivenessState live2{};
    run_shadow_dce_block(pruned, words, live2, rerun);
    return rerun;
  }
  return pruned;
}

[[nodiscard]] std::size_t svm_block_jit_weight_impl(
    const SvmBlockIr &block, std::uint32_t node_feature_mask,
    bool shadow_eval_only,
    std::span<const std::uint32_t> words = {}) noexcept {
  std::size_t w = 0u;
  for (const auto &op : block) {
    if (op.is_conditional) {
      const auto cw = svm_block_jit_weight_impl(
          op.children, node_feature_mask, shadow_eval_only, words);
      if (cw > 0u) {
        w += 2u + cw;
      }
      continue;
    }
    switch (op.instruction.opcode) {
      case NODE_END:
      case NODE_SHADER_JUMP:
      case NODE_JUMP_IF_ZERO:
      case NODE_JUMP_IF_ONE:
      case NODE_NONE:
      case NODE_PAD1:
        break;
      case NODE_CLOSURE_EMISSION:
      case NODE_CLOSURE_BACKGROUND:
        if (!shadow_eval_only &&
            (node_feature_mask & kernel_feature_node_emission) != 0u) {
          w += 1u;
        }
        break;
      case NODE_CLOSURE_HOLDOUT:
        if (!shadow_eval_only) {
          w += 1u;
        }
        break;
      case NODE_CLOSURE_VOLUME:
      case NODE_VOLUME_COEFFICIENTS:
        if (!shadow_eval_only &&
            (node_feature_mask & kernel_feature_node_volume) != 0u) {
          w += 1u;
        }
        break;
      case NODE_PRINCIPLED_VOLUME:
        if (!shadow_eval_only &&
            (node_feature_mask & kernel_feature_node_volume) != 0u) {
          w += 20u;
        }
        break;
      case NODE_AMBIENT_OCCLUSION:
        if ((node_feature_mask & kernel_feature_node_raytrace) != 0u ||
            node_feature_mask == ~0u) {
          w += 20u;
        }
        break;
      case NODE_BEVEL:
        if ((node_feature_mask & kernel_feature_node_raytrace) != 0u ||
            node_feature_mask == ~0u) {
          w += 24u;
        }
        break;
      case NODE_TEX_VORONOI: w += 36u; break;
      case NODE_TEX_GABOR: w += 32u; break;
      case NODE_RADIAL_TILING: w += 16u; break;
      case NODE_TEX_SKY: w += 16u; break;
      case NODE_CLOSURE_BSDF: {
        if (shadow_eval_only) {
          w += 2u;
        } else {
          const auto bsdf =
              decode_op_payload<SVMNodeClosureBsdf>(op.instruction, words);
          w += (bsdf &&
                static_cast<ClosureType>(bsdf->closure_type) ==
                    CLOSURE_BSDF_PRINCIPLED_ID)
                   ? 80u
                   : 14u;
        }
        break;
      }
      case NODE_TEX_NOISE: w += 12u; break;
      case NODE_TEX_IMAGE_BOX:
      case NODE_TEX_IMAGE_BOX_DERIVATIVE: w += 12u; break;
      case NODE_TEX_BRICK:
      case NODE_TEX_WAVE: w += 10u; break;
      case NODE_TEX_IMAGE:
      case NODE_TEX_IMAGE_DERIVATIVE:
      case NODE_TEX_ENVIRONMENT:
      case NODE_TEX_ENVIRONMENT_DERIVATIVE: w += 5u; break;
      case NODE_NORMAL_MAP:
      case NODE_VECTOR_MATH_DERIVATIVE:
      case NODE_MAPPING_DERIVATIVE: w += 4u; break;
      case NODE_OSL_SCRIPT: w += 40u; break;
      default: w += 1u; break;
    }
  }
  return w;
}

[[nodiscard]] std::size_t count_active_nonempty_shaders(
    const SvmTranslationRequest &request) noexcept {
  if (!can_direct_jit(request)) {
    return 0u;
  }
  const auto shader_count =
      static_cast<std::size_t>(request.table.shader_count);
  const bool has_mask = request.active_shader_mask.size() == shader_count;
  const auto host_words = std::span<const std::uint32_t>{request.table.words};
  std::size_t nonempty = 0u;
  for (std::size_t i = 0u; i < shader_count; ++i) {
    if (has_mask && request.active_shader_mask[i] == 0u) {
      continue;
    }
    const auto &raw =
        request.table.ir.shaders[i].routine_for(request.shader_type);
    if (raw.empty()) {
      continue;
    }
    if (!prune_svm_block_for_eval(raw, host_words, request.shader_type,
                                  request.node_feature_mask,
                                  request.shadow_eval_only)
             .empty()) {
      ++nonempty;
    }
  }
  return nonempty;
}

class InstructionScopedKernelGlobals final
    : public detail::DelegatingKernelGlobals {
  const SvmInstructionSpecialization &_spec;
  std::span<const compiler::cycles_svm::OslScriptBinding> _table_osl_scripts;

public:
  InstructionScopedKernelGlobals(
      const KernelGlobals &inner,
      const SvmInstructionSpecialization &spec,
      std::span<const compiler::cycles_svm::OslScriptBinding>
          table_osl_scripts) noexcept
      : DelegatingKernelGlobals{inner},
        _spec{spec},
        _table_osl_scripts{table_osl_scripts} {}

  [[nodiscard]] std::span<const compiler::cycles_svm::OslScriptBinding>
  osl_scripts() const noexcept override {
    if (!_table_osl_scripts.empty()) {
      return _table_osl_scripts;
    }
    return _inner.osl_scripts();
  }

  [[nodiscard]] bool may_have_thin_film() const noexcept override {
    return _inner.may_have_thin_film() && _spec.may_have_thin_film;
  }

  [[nodiscard]] ClosureUsage closure_usage() const noexcept override {
    const auto base = _inner.closure_usage();
    return {
        .svm_bsdf_node_mask =
            base.svm_bsdf_node_mask & _spec.closure_usage.svm_bsdf_node_mask,
        .surface_closure_mask = base.surface_closure_mask &
                                _spec.closure_usage.surface_closure_mask,
        .microfacet_fresnel_mask = base.microfacet_fresnel_mask &
                                   _spec.closure_usage.microfacet_fresnel_mask,
        .principled_mask =
            base.principled_mask & _spec.closure_usage.principled_mask,
        .diffuse_node_mask =
            base.diffuse_node_mask & _spec.closure_usage.diffuse_node_mask};
  }

  [[nodiscard]] NormalMapUsage normal_map_usage() const noexcept override {
    const auto base = _inner.normal_map_usage();
    return {.space_mask = base.space_mask & _spec.normal_map_space_mask,
            .uses_original_base =
                base.uses_original_base && _spec.normal_map_uses_original_base};
  }

  [[nodiscard]] std::uint32_t mix_blend_mask() const noexcept override {
    return _inner.mix_blend_mask() & _spec.mix_blend_mask;
  }

  [[nodiscard]] NoiseStaticOptions
  noise_static_options() const noexcept override {
    const auto base = _inner.noise_static_options();
    return {.may_have_distortion = base.may_have_distortion &&
                                   _spec.noise_may_have_distortion,
            .may_need_color =
                base.may_need_color && _spec.noise_may_need_color};
  }

  [[nodiscard]] std::uint32_t tex_coord_mask() const noexcept override {
    return _inner.tex_coord_mask() & _spec.tex_coord_mask;
  }

  [[nodiscard]] bool bump_uses_object_space() const noexcept override {
    return _inner.bump_uses_object_space() && _spec.bump_uses_object_space;
  }

  [[nodiscard]] bool curves_may_extrapolate() const noexcept override {
    return _inner.curves_may_extrapolate() && _spec.curves_may_extrapolate;
  }

  [[nodiscard]] std::uint64_t math_type_mask() const noexcept override {
    return _inner.math_type_mask() & _spec.math_type_mask;
  }

  [[nodiscard]] std::uint64_t vector_math_type_mask() const noexcept override {
    return _inner.vector_math_type_mask() & _spec.vector_math_type_mask;
  }

  [[nodiscard]] MappingStaticOptions
  mapping_static_options() const noexcept override {
    const auto base = _inner.mapping_static_options();
    return {.type_mask = base.type_mask & _spec.mapping_type_mask,
            .may_rotate = base.may_rotate && _spec.mapping_may_rotate};
  }

  [[nodiscard]] std::uint32_t map_range_type_mask() const noexcept override {
    return _inner.map_range_type_mask() & _spec.map_range_type_mask;
  }

  [[nodiscard]] VoronoiStaticOptions
  voronoi_static_options() const noexcept override {
    const auto base = _inner.voronoi_static_options();
    return {.dimension_mask =
                base.dimension_mask & _spec.voronoi_dimension_mask,
            .feature_mask = base.feature_mask & _spec.voronoi_feature_mask,
            .metric_mask = base.metric_mask & _spec.voronoi_metric_mask,
            .may_have_detail =
                base.may_have_detail && _spec.voronoi_may_have_detail,
            .may_normalize =
                base.may_normalize && _spec.voronoi_may_normalize,
            .may_need_color =
                base.may_need_color && _spec.voronoi_may_need_color,
            .may_need_position =
                base.may_need_position && _spec.voronoi_may_need_position,
            .may_need_radius =
                base.may_need_radius && _spec.voronoi_may_need_radius};
  }

  [[nodiscard]] WaveStaticOptions
  wave_static_options() const noexcept override {
    const auto base = _inner.wave_static_options();
    return {.type_mask = base.type_mask & _spec.wave_type_mask,
            .profile_mask = base.profile_mask & _spec.wave_profile_mask,
            .may_have_distortion = base.may_have_distortion &&
                                   _spec.wave_may_have_distortion};
  }
};

void emit_jit_instruction(
    const SvmTranslationRequest &req, const SvmInstructionIr &inst,
    detail::Stack &stack, Float3 &closure_weight,
    detail::SpectralClosureWeight *spectral_weight = nullptr) noexcept {
  const auto host_words = std::span<const std::uint32_t>{req.table.words};
  LUISA_ASSERT(
      static_cast<std::size_t>(inst.payload_offset) + inst.payload_word_count <=
          host_words.size(),
      "DirectJitTranslator instruction payload [{}, +{}) exceeds word table {}.",
      inst.payload_offset, inst.payload_word_count, host_words.size());
  const auto payload_span = host_words.subspan(
      static_cast<std::size_t>(inst.payload_offset),
      static_cast<std::size_t>(inst.payload_word_count));
  detail::Cursor cursor{req.words, payload_span, inst.payload_offset};
  const InstructionScopedKernelGlobals kg{req.kernel_globals,
                                          inst.specialization,
                                          req.table.osl_scripts};
  const detail::EvaluationTransition transition{nullptr};
  const auto kernel_features = req.kernel_features;
  const auto node_feature_mask = req.node_feature_mask;
  const bool is_volume_mask =
      (node_feature_mask & kernel_feature_node_volume) != 0u;
  const bool object_motion =
      (kernel_features & kernel_feature_object_motion) != 0u;

  switch (inst.opcode) {
    case NODE_END:
    case NODE_SHADER_JUMP:
    case NODE_JUMP_IF_ZERO:
    case NODE_JUMP_IF_ONE:
      return;
    case NODE_CLOSURE_BSDF:
      if (req.shadow_eval_only) {
        detail::node_closure_bsdf_shadow_only(
            kg, cursor, stack, closure_weight, req.shader_type,
            kernel_features, node_feature_mask, req.shader_data, req.path_state,
            transition);
      } else {
        detail::node_closure_bsdf(kg, cursor, stack, closure_weight,
                                  req.shader_type, kernel_features,
                                  node_feature_mask, req.shader_data,
                                  req.path_state, transition, spectral_weight);
      }
      return;
    case NODE_CLOSURE_EMISSION:
      if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
        detail::node_closure_emission(kg, cursor, stack, closure_weight,
                                      req.shader_data, transition,
                                      spectral_weight);
      }
      return;
    case NODE_CLOSURE_VOLUME:
      if (is_volume_mask && req.shader_type == SHADER_TYPE_VOLUME &&
          (kernel_features & kernel_feature_volume) != 0u) {
        detail::node_closure_volume(kg, cursor, stack, closure_weight,
                                    req.shader_data, transition,
                                    spectral_weight);
      }
      return;
    case NODE_VOLUME_COEFFICIENTS:
      if (is_volume_mask && req.shader_type == SHADER_TYPE_VOLUME &&
          (kernel_features & kernel_feature_volume) != 0u) {
        detail::node_volume_coefficients(kg, cursor, stack, closure_weight,
                                         req.shader_data, req.path_state,
                                         transition, spectral_weight);
      }
      return;
    case NODE_PRINCIPLED_VOLUME:
      if (is_volume_mask && req.shader_type == SHADER_TYPE_VOLUME &&
          (kernel_features & kernel_feature_volume) != 0u) {
        detail::node_principled_volume(kg, cursor, stack, closure_weight,
                                       req.shader_data, req.path_state,
                                       transition, spectral_weight);
      }
      return;
    case NODE_CLOSURE_HOLDOUT:
      detail::node_closure_holdout(cursor, stack, closure_weight,
                                   req.shader_data);
      return;
    case NODE_CLOSURE_BACKGROUND:
      if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
        detail::node_closure_background(cursor, stack, closure_weight,
                                        req.shader_data, spectral_weight);
      }
      return;
    case NODE_CLOSURE_SET_WEIGHT:
      detail::node_closure_set_weight(cursor, closure_weight, spectral_weight);
      return;
    case NODE_CLOSURE_WEIGHT:
      detail::node_closure_weight(cursor, stack, closure_weight,
                                  spectral_weight);
      return;
    case NODE_EMISSION_WEIGHT:
      if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
        detail::node_emission_weight(cursor, stack, closure_weight,
                                     spectral_weight);
      }
      return;
    case NODE_MIX_CLOSURE:
      detail::node_mix_closure(cursor, stack);
      return;
    case NODE_GEOMETRY:
      detail::node_geometry(cursor, stack, kg, req.shader_data, false);
      return;
    case NODE_GEOMETRY_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_geometry(cursor, stack, kg, req.shader_data, true);
      }
      return;
    case NODE_CAMERA:
      detail::node_camera(cursor, stack, req.transform_state, req.shader_data);
      return;
    case NODE_FRESNEL:
      detail::node_fresnel(cursor, stack, req.shader_data);
      return;
    case NODE_LAYER_WEIGHT:
      detail::node_layer_weight(cursor, stack, req.shader_data);
      return;
    case NODE_TEX_COORD:
      detail::node_tex_coord(cursor, stack, kg, req.transform_state,
                             req.shader_data, req.path_state, false,
                             is_volume_mask, object_motion,
                             kg.tex_coord_mask());
      return;
    case NODE_TEX_COORD_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_tex_coord(cursor, stack, kg, req.transform_state,
                               req.shader_data, req.path_state, true, false,
                               object_motion, kg.tex_coord_mask());
      }
      return;
    case NODE_TEX_IMAGE:
      detail::node_tex_image(cursor, stack, kg, req.shader_data, req.path_state,
                             false);
      return;
    case NODE_TEX_IMAGE_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_tex_image(cursor, stack, kg, req.shader_data,
                               req.path_state, true);
      }
      return;
    case NODE_TEX_IMAGE_BOX:
      detail::node_tex_image_box(cursor, stack, kg, req.transform_state,
                                 req.shader_data, req.path_state, false,
                                 object_motion);
      return;
    case NODE_TEX_IMAGE_BOX_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_tex_image_box(cursor, stack, kg, req.transform_state,
                                   req.shader_data, req.path_state, true,
                                   object_motion);
      }
      return;
    case NODE_TEX_NOISE: {
      NoiseUsage scoped_noise{.shape_mask = req.noise_usage.shape_mask &
                                            inst.specialization.noise_usage
                                                .shape_mask};
      detail::node_tex_noise(cursor, stack, scoped_noise,
                             kg.noise_static_options());
      return;
    }
    case NODE_TEX_WHITE_NOISE:
      detail::node_tex_white_noise(cursor, stack);
      return;
    case NODE_TEX_GRADIENT:
      detail::node_tex_gradient(cursor, stack);
      return;
    case NODE_TEX_VORONOI:
      detail::node_tex_voronoi(
          cursor, stack,
          (node_feature_mask & kernel_feature_node_voronoi_extra) != 0u,
          kg.voronoi_static_options());
      return;
    case NODE_TEX_GABOR:
      detail::node_tex_gabor(cursor, stack);
      return;
    case NODE_RADIAL_TILING:
      detail::node_radial_tiling(cursor, stack);
      return;
    case NODE_TEX_WAVE:
      detail::node_tex_wave(cursor, stack, kg.wave_static_options());
      return;
    case NODE_TEX_MAGIC:
      detail::node_tex_magic(cursor, stack);
      return;
    case NODE_TEX_CHECKER:
      detail::node_tex_checker(cursor, stack);
      return;
    case NODE_TEX_BRICK:
      detail::node_tex_brick(cursor, stack);
      return;
    case NODE_RGB_RAMP:
      detail::node_rgb_ramp(cursor, stack);
      return;
    case NODE_CURVES:
      detail::node_curves(cursor, stack, kg.curves_may_extrapolate());
      return;
    case NODE_FLOAT_CURVE:
      detail::node_float_curve(cursor, stack);
      return;
    case NODE_TEX_ENVIRONMENT:
      detail::node_tex_environment(cursor, stack, kg, req.shader_data, false);
      return;
    case NODE_TEX_ENVIRONMENT_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_tex_environment(cursor, stack, kg, req.shader_data, true);
      }
      return;
    case NODE_TEX_SKY:
      detail::node_tex_sky(cursor, stack, kg, req.shader_data, req.path_state);
      return;
    case NODE_ATTR:
      if (is_volume_mask) {
        detail::node_attr_volume(cursor, stack, kg, req.shader_data);
      } else {
        detail::node_attr_surface(cursor, stack, kg, req.shader_data);
      }
      return;
    case NODE_ATTR_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_attr_derivative(cursor, stack, kg, req.shader_data);
      }
      return;
    case NODE_VERTEX_COLOR:
      detail::node_vertex_color(cursor, stack, kg, req.shader_data);
      return;
    case NODE_VERTEX_COLOR_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_vertex_color_derivative(cursor, stack, kg,
                                             req.shader_data);
      }
      return;
    case NODE_CONVERT:
      detail::node_convert(cursor, stack, kg, false);
      return;
    case NODE_CONVERT_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_convert(cursor, stack, kg, true);
      }
      return;
    case NODE_VALUE_F:
      detail::node_value_f(cursor, stack, false);
      return;
    case NODE_VALUE_F_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_value_f(cursor, stack, true);
      }
      return;
    case NODE_VALUE_V:
      detail::node_value_v(cursor, stack, false);
      return;
    case NODE_VALUE_V_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_value_v(cursor, stack, true);
      }
      return;
    case NODE_MAPPING:
      detail::node_mapping(cursor, stack, false, kg.mapping_static_options());
      return;
    case NODE_MAPPING_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_mapping(cursor, stack, true, kg.mapping_static_options());
      }
      return;
    case NODE_TEXTURE_MAPPING:
      detail::node_texture_mapping(cursor, stack, false);
      return;
    case NODE_TEXTURE_MAPPING_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_texture_mapping(cursor, stack, true);
      }
      return;
    case NODE_MIN_MAX:
      detail::node_min_max(cursor, stack);
      return;
    case NODE_VECTOR_MATH:
      detail::node_vector_math(cursor, stack, false, kg.vector_math_type_mask());
      return;
    case NODE_VECTOR_MATH_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_vector_math(cursor, stack, true,
                                 kg.vector_math_type_mask());
      }
      return;
    case NODE_SET_BUMP:
      detail::node_set_bump(
          cursor, stack, req.transform_state, req.shader_data,
          (node_feature_mask & kernel_feature_node_bump) != 0u, object_motion,
          (node_feature_mask & kernel_feature_node_bump_state) != 0u,
          kg.bump_uses_object_space());
      return;
    case NODE_CLOSURE_SET_NORMAL:
      if ((node_feature_mask & kernel_feature_node_bump) != 0u) {
        detail::node_set_normal(cursor, stack, req.shader_data);
      }
      return;
    case NODE_ENTER_BUMP_EVAL:
      if ((node_feature_mask & kernel_feature_node_bump_state) != 0u) {
        detail::node_enter_bump_eval(
            cursor, stack, kg, req.transform_state, req.shader_data,
            object_motion,
            kg.surface_attribute_summary().has_undisplaced_position);
      }
      return;
    case NODE_LEAVE_BUMP_EVAL:
      if ((node_feature_mask & kernel_feature_node_bump_state) != 0u) {
        detail::node_leave_bump_eval(cursor, stack, req.shader_data);
      }
      return;
    case NODE_SET_DISPLACEMENT:
      detail::node_set_displacement(
          cursor, stack, req.shader_data,
          (node_feature_mask & kernel_feature_node_bump) != 0u);
      return;
    case NODE_DISPLACEMENT:
      detail::node_displacement(
          cursor, stack, req.transform_state, req.shader_data,
          (node_feature_mask & kernel_feature_node_bump) != 0u, object_motion);
      return;
    case NODE_VECTOR_DISPLACEMENT:
      detail::node_vector_displacement(
          cursor, stack, kg, req.transform_state, req.shader_data,
          (node_feature_mask & kernel_feature_node_bump) != 0u, object_motion);
      return;
    case NODE_HSV:
      detail::node_hsv(cursor, stack);
      return;
    case NODE_MATH:
      detail::node_math(cursor, stack, kg.math_type_mask());
      return;
    case NODE_GAMMA:
      detail::node_gamma(cursor, stack);
      return;
    case NODE_BRIGHTCONTRAST:
      detail::node_brightness(cursor, stack);
      return;
    case NODE_WAVELENGTH:
      detail::node_wavelength(cursor, stack, kg, &req.shader_data);
      return;
    case NODE_BLACKBODY:
      detail::node_blackbody(cursor, stack, kg, &req.shader_data);
      return;
    case NODE_LIGHT_PATH:
      detail::node_light_path(cursor, stack, req.shader_data, req.path_state,
                              node_feature_mask);
      return;
    case NODE_OBJECT_INFO:
      if (const auto *services = kg.info_services()) {
        detail::node_object_info(cursor, stack, *services, req.shader_data);
      } else {
        transition.unsupported();
      }
      return;
    case NODE_PARTICLE_INFO:
      if (const auto *services = kg.info_services()) {
        detail::node_particle_info(cursor, stack, *services, req.shader_data);
      } else {
        transition.unsupported();
      }
      return;
    case NODE_HAIR_INFO:
      if ((kernel_features & kernel_feature_hair) != 0u) {
        detail::node_hair_info(cursor, stack, kg.info_services(),
                               req.shader_data, transition);
      }
      return;
    case NODE_POINT_INFO:
      if ((kernel_features & kernel_feature_pointcloud) != 0u) {
        if (const auto *services = kg.info_services()) {
          detail::node_point_info(cursor, stack, *services, req.shader_data);
        } else {
          transition.unsupported();
        }
      }
      return;
    case NODE_AMBIENT_OCCLUSION:
      if ((kernel_features & kernel_feature_node_raytrace) != 0u &&
          (req.node_feature_mask & kernel_feature_node_raytrace) != 0u) {
        if (const auto *services = kg.ambient_occlusion_services()) {
          detail::node_ambient_occlusion(cursor, stack, *services,
                                         req.shader_data);
        } else {
          transition.unsupported();
        }
      }
      return;
    case NODE_BEVEL:
      if ((kernel_features & kernel_feature_node_raytrace) != 0u &&
          (req.node_feature_mask & kernel_feature_node_raytrace) != 0u) {
        if (const auto *services = kg.ambient_occlusion_services()) {
          detail::node_bevel(cursor, stack, kg, req.transform_state, *services,
                             req.shader_data, object_motion);
        } else {
          transition.unsupported();
        }
      }
      return;
    case NODE_INVERT:
      detail::node_invert(cursor, stack);
      return;
    case NODE_MIX:
      detail::node_mix(cursor, stack, kg.mix_blend_mask());
      return;
    case NODE_SEPARATE_COLOR:
      detail::node_separate_color(cursor, stack);
      return;
    case NODE_COMBINE_COLOR:
      detail::node_combine_color(cursor, stack);
      return;
    case NODE_SEPARATE_VECTOR:
      detail::node_separate_vector(cursor, stack, false);
      return;
    case NODE_SEPARATE_VECTOR_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_separate_vector(cursor, stack, true);
      }
      return;
    case NODE_COMBINE_VECTOR:
      detail::node_combine_vector(cursor, stack, false);
      return;
    case NODE_COMBINE_VECTOR_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_combine_vector(cursor, stack, true);
      }
      return;
    case NODE_VECTOR_ROTATE:
      detail::node_vector_rotate(cursor, stack);
      return;
    case NODE_VECTOR_TRANSFORM:
      detail::node_vector_transform(cursor, stack, req.transform_state,
                                    req.shader_data, object_motion);
      return;
    case NODE_NORMAL:
      detail::node_normal(cursor, stack);
      return;
    case NODE_NORMAL_MAP:
      detail::node_normal_map(cursor, stack, kg, req.transform_state,
                              req.shader_data, object_motion);
      return;
    case NODE_TANGENT:
      detail::node_tangent(cursor, stack, kg, req.transform_state,
                           req.shader_data, false, object_motion);
      return;
    case NODE_TANGENT_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_tangent(cursor, stack, kg, req.transform_state,
                             req.shader_data, true, object_motion);
      }
      return;
    case NODE_LIGHT_FALLOFF:
      detail::node_light_falloff(cursor, stack, req.shader_data);
      return;
    case NODE_IES:
      detail::node_ies(cursor, stack, kg);
      return;
    case NODE_WIREFRAME:
      detail::node_wireframe(
          cursor, stack, kg, req.transform_state, req.shader_data,
          (kernel_features &
           (kernel_feature_hair | kernel_feature_pointcloud)) != 0u,
          object_motion);
      return;
    case NODE_MAP_RANGE:
      detail::node_map_range(cursor, stack, kg.map_range_type_mask());
      return;
    case NODE_VECTOR_MAP_RANGE:
      detail::node_vector_map_range(cursor, stack, kg.map_range_type_mask());
      return;
    case NODE_CLAMP:
      detail::node_clamp(cursor, stack);
      return;
    case NODE_MIX_COLOR:
      detail::node_mix_color(cursor, stack, kg.mix_blend_mask());
      return;
    case NODE_MIX_FLOAT:
      detail::node_mix_float(cursor, stack);
      return;
    case NODE_MIX_VECTOR:
      detail::node_mix_vector(cursor, stack);
      return;
    case NODE_MIX_VECTOR_NON_UNIFORM:
      detail::node_mix_vector_non_uniform(cursor, stack);
      return;
    case NODE_OSL_SCRIPT:
      detail::node_osl_script(cursor, stack, kg, req.transform_state,
                              req.shader_data, req.path_state, req.shader_type,
                              kernel_features, node_feature_mask, transition);
      return;
    default:
      dsl::unreachable(
          "invalid node in DirectJitTranslator Cycles SVM stream");
      return;
  }
}

[[nodiscard]] bool block_has_spectral_source(const SvmBlockIr &block) noexcept {
  for (const auto &op : block) {
    if (op.is_conditional) {
      if (block_has_spectral_source(op.children)) {
        return true;
      }
    } else if (op.instruction.opcode == NODE_BLACKBODY ||
               op.instruction.opcode == NODE_WAVELENGTH) {
      return true;
    }
  }
  return false;
}

void emit_jit_block(
    const SvmTranslationRequest &req, const SvmBlockIr &block,
    detail::Stack &stack, Float3 &closure_weight,
    detail::SpectralClosureWeight *spectral_weight = nullptr) noexcept {
  for (const auto &op : block) {
    if (op.is_conditional) {
      if (op.children.empty()) {
        continue;
      }
      const auto cond_off = static_cast<std::uint32_t>(op.cond_stack_offset);
      if (const auto c_cond = stack.try_const_slot(cond_off)) {
        const bool take =
            op.jump_if_one ? !(*c_cond >= 1.0f) : !(*c_cond <= 0.0f);
        if (take) {
          emit_jit_block(req, op.children, stack, closure_weight,
                         spectral_weight);
        }
        continue;
      }
      const auto cond_val = detail::stack_load_float(stack, cond_off);
      const Bool take_children =
          op.jump_if_one ? !(cond_val >= 1.0f) : !(cond_val <= 0.0f);
      $if(take_children) {
        emit_jit_block(req, op.children, stack, closure_weight,
                       spectral_weight);
      };
    } else {
      emit_jit_instruction(req, op.instruction, stack, closure_weight,
                           spectral_weight);
    }
  }
}

} // namespace

std::size_t svm_block_jit_weight(
    const SvmBlockIr &block, std::uint32_t node_feature_mask) noexcept {
  return svm_block_jit_weight_impl(block, node_feature_mask, false);
}

std::size_t svm_block_jit_weight(
    const SvmBlockIr &block, std::span<const std::uint32_t> words,
    ShaderType shader_type, std::uint32_t node_feature_mask,
    bool shadow_eval_only) noexcept {
  const auto pruned = prune_svm_block_for_eval(
      block, words, shader_type, node_feature_mask, shadow_eval_only);
  return svm_block_jit_weight_impl(pruned, node_feature_mask, shadow_eval_only,
                                   words);
}

std::size_t direct_jit_max_shaders() noexcept {
  return configured_direct_jit_max_shaders();
}

std::size_t direct_jit_max_weight() noexcept {
  return configured_direct_jit_max_weight();
}

void BytecodeVmTranslator::translate(
    const SvmTranslationRequest &request) const noexcept {
  struct OslTableKernelGlobals final : detail::DelegatingKernelGlobals {
    std::span<const OslScriptBinding> _scripts;
    OslTableKernelGlobals(const KernelGlobals &inner,
                          std::span<const OslScriptBinding> scripts) noexcept
        : DelegatingKernelGlobals{inner}, _scripts{scripts} {}
    [[nodiscard]] std::span<const OslScriptBinding>
    osl_scripts() const noexcept override {
      return _scripts;
    }
  };
  if (!request.table.osl_scripts.empty()) {
    const OslTableKernelGlobals scoped_kg{request.kernel_globals,
                                          request.table.osl_scripts};
    eval_nodes_assume_valid(scoped_kg, request.words, request.shader_type,
                            request.kernel_features, request.node_feature_mask,
                            request.node_types_used, request.transform_state,
                            request.shader_data, request.path_state,
                            request.stack_size, request.noise_usage);
    return;
  }
  eval_nodes_assume_valid(request.kernel_globals, request.words,
                          request.shader_type, request.kernel_features,
                          request.node_feature_mask, request.node_types_used,
                          request.transform_state, request.shader_data,
                          request.path_state, request.stack_size,
                          request.noise_usage);
}

void DirectJitTranslator::translate(
    const SvmTranslationRequest &request) const noexcept {
  if (request.shader_type == SHADER_TYPE_BUMP) {
    return;
  }
  if (!can_direct_jit(request)) {
    BytecodeVmTranslator{}.translate(request);
    return;
  }
  LUISA_ASSERT(request.stack_size != 0u && request.stack_size <= SVM_STACK_SIZE,
               "Cycles SVM stack extent must be in [1, {}], got {}.",
               SVM_STACK_SIZE, request.stack_size);

  struct ShaderGroup {
    luisa::vector<std::uint32_t> shader_ids;
    const SvmBlockIr *routine{nullptr};
  };

  const auto shader_count =
      static_cast<std::size_t>(request.table.shader_count);
  const bool has_mask = request.active_shader_mask.size() == shader_count;
  const auto host_words = std::span<const std::uint32_t>{request.table.words};

  luisa::vector<SvmBlockIr> pruned_routines(shader_count);
  luisa::vector<ShaderGroup> groups;
  std::size_t active_empty_count = 0u;
  for (std::size_t i = 0u; i < shader_count; ++i) {
    if (has_mask && request.active_shader_mask[i] == 0u) {
      continue;
    }
    pruned_routines[i] = prune_svm_block_for_eval(
        request.table.ir.shaders[i].routine_for(request.shader_type),
        host_words, request.shader_type, request.node_feature_mask,
        request.shadow_eval_only);
    const auto &routine = pruned_routines[i];
    if (routine.empty()) {
      ++active_empty_count;
      continue;
    }
    bool merged = false;
    for (auto &group : groups) {
      if (same_linked_svm_block(host_words, *group.routine, routine)) {
        group.shader_ids.emplace_back(static_cast<std::uint32_t>(i));
        merged = true;
        break;
      }
    }
    if (!merged) {
      ShaderGroup g;
      g.shader_ids.emplace_back(static_cast<std::uint32_t>(i));
      g.routine = &routine;
      groups.emplace_back(std::move(g));
    }
  }

  std::size_t total_jit_weight = 0u;
  for (const auto &g : groups) {
    total_jit_weight += svm_block_jit_weight_impl(
        *g.routine, request.node_feature_mask, request.shadow_eval_only,
        host_words);
  }

  const bool explicit_chunk_env = []() noexcept {
    const char *raw = std::getenv("PSYCLES_SVM_OUTLINE_CHUNK_GROUPS");
    return raw != nullptr && *raw != '\0';
  }();
  const auto configured_chunk_groups = [&]() noexcept -> std::size_t {
    if (const char *raw = std::getenv("PSYCLES_SVM_OUTLINE_CHUNK_GROUPS");
        raw != nullptr && *raw != '\0') {
      char *end = nullptr;
      const unsigned long parsed = std::strtoul(raw, &end, 10);
      if (end != raw && *end == '\0') {
        return static_cast<std::size_t>(parsed);
      }
    }
    if (request.max_groups_per_outlined_chunk > 16u &&
        (groups.size() > 16u || total_jit_weight > 600u)) {
      return std::size_t{16u};
    }
    return request.max_groups_per_outlined_chunk;
  }();
  const std::size_t target_chunk_weight_budget = [&]() noexcept -> std::size_t {
    if (const char *raw = std::getenv("PSYCLES_SVM_OUTLINE_CHUNK_WEIGHT");
        raw != nullptr && *raw != '\0') {
      char *end = nullptr;
      const unsigned long parsed = std::strtoul(raw, &end, 10);
      if (end != raw && *end == '\0' && parsed > 0ul) {
        return static_cast<std::size_t>(parsed);
      }
    }
    return request.max_chunk_jit_weight > 0u ? request.max_chunk_jit_weight
                                             : std::size_t{900u};
  }();
  const bool should_outline_chunks =
      configured_chunk_groups > 0u &&
      (groups.size() > configured_chunk_groups ||
       (!explicit_chunk_env && groups.size() > 1u &&
        total_jit_weight > target_chunk_weight_budget));
  if (should_outline_chunks) {
    struct OutlinedChunk {
      std::uint32_t max_shader_id{};
      std::size_t weight{};
      luisa::vector<ShaderGroup> groups;
    };
    luisa::vector<OutlinedChunk> chunks;
    OutlinedChunk current;
    for (std::size_t i = 0u; i < shader_count; ++i) {
      if (has_mask && request.active_shader_mask[i] == 0u) {
        continue;
      }
      const auto &routine = pruned_routines[i];
      if (routine.empty()) {
        continue;
      }
      const auto sid = static_cast<std::uint32_t>(i);
      bool merged_in_chunk = false;
      for (auto &group : current.groups) {
        if (same_linked_svm_block(host_words, *group.routine, routine)) {
          group.shader_ids.emplace_back(sid);
          current.max_shader_id = sid;
          merged_in_chunk = true;
          break;
        }
      }
      if (merged_in_chunk) {
        continue;
      }
      const std::size_t routine_weight = svm_block_jit_weight_impl(
          routine, request.node_feature_mask, request.shadow_eval_only,
          host_words);
      const bool exceeds_weight =
          !explicit_chunk_env && !current.groups.empty() &&
          (current.weight + routine_weight > target_chunk_weight_budget);
      if (current.groups.size() >= configured_chunk_groups || exceeds_weight) {
        chunks.emplace_back(std::move(current));
        current = OutlinedChunk{};
      }
      ShaderGroup g;
      g.shader_ids.emplace_back(sid);
      g.routine = &routine;
      current.max_shader_id = sid;
      current.weight += routine_weight;
      current.groups.emplace_back(std::move(g));
    }
    if (!current.groups.empty()) {
      chunks.emplace_back(std::move(current));
    }
    if (chunks.size() > 1u) {
      const auto emit_dispatcher = [&]() noexcept {
        const UInt shader_index = request.shader_data.shader & shader_mask;
        const auto emit_range =
            [&](auto &&self, std::size_t lo, std::size_t hi) noexcept -> void {
          if (hi - lo == 1u) {
            const auto &chunk = chunks[lo];
            $outline_when_with_name(should_outline_chunks,
                                    "svm_eval_nodes_chunk") {
              if (chunk.groups.size() == 1u &&
                  chunk.groups.front().shader_ids.size() == 1u) {
                using ::operator==;
                $if(shader_index ==
                    def<luisa::uint>(chunk.groups.front().shader_ids.front())) {
                  detail::DirectJitCaseCacheGuard attr_cache_guard;
                  const bool has_spectral =
                      request.kernel_globals.is_spectral() &&
                      request.shader_data.spectral.has_value() &&
                      block_has_spectral_source(*chunk.groups.front().routine);
                  auto group_stack = detail::Stack::direct_jit(
                      request.stack_size, has_spectral);
                  Float3 closure_weight = make_float3(0.0f);
                  std::optional<detail::SpectralClosureWeight> spectral_weight;
                  if (has_spectral) {
                    spectral_weight.emplace();
                  }
                  emit_jit_block(request, *chunk.groups.front().routine,
                                 group_stack, closure_weight,
                                 spectral_weight ? &*spectral_weight : nullptr);
                };
              } else {
                $switch(shader_index) {
                  for (const auto &group : chunk.groups) {
                    luisa::compute::detail::SwitchCaseStmtBuilder{
                        luisa::span<const std::uint32_t>{group.shader_ids}} %
                        [&]() noexcept {
                          detail::DirectJitCaseCacheGuard attr_cache_guard;
                          const bool has_spectral =
                              request.kernel_globals.is_spectral() &&
                              request.shader_data.spectral.has_value() &&
                              block_has_spectral_source(*group.routine);
                          auto case_stack = detail::Stack::direct_jit(
                              request.stack_size, has_spectral);
                          Float3 closure_weight = make_float3(0.0f);
                          std::optional<detail::SpectralClosureWeight>
                              spectral_weight;
                          if (has_spectral) {
                            spectral_weight.emplace();
                          }
                          emit_jit_block(
                              request, *group.routine, case_stack,
                              closure_weight,
                              spectral_weight ? &*spectral_weight : nullptr);
                        };
                  }
                  $default {};
                };
              }
            };
            return;
          }
          const std::size_t mid = lo + (hi - lo) / 2u;
          $if(shader_index <= def<luisa::uint>(chunks[mid - 1u].max_shader_id)) {
            self(self, lo, mid);
          }
          $else {
            self(self, mid, hi);
          };
        };
        emit_range(emit_range, 0u, chunks.size());
      };
      if (request.outline_chunk_dispatcher) {
        $outline_when_with_name(should_outline_chunks, "svm_eval_nodes") {
          emit_dispatcher();
        };
      } else {
        emit_dispatcher();
      }
      return;
    }
  }

  $outline_with_name("svm_eval_nodes") {
    if (groups.empty()) {
      return;
    }
    if (groups.size() == 1u && active_empty_count == 0u) {
      detail::DirectJitCaseCacheGuard attr_cache_guard;
      const bool has_spectral =
          request.kernel_globals.is_spectral() &&
          request.shader_data.spectral.has_value() &&
          block_has_spectral_source(*groups.front().routine);
      auto case_stack =
          detail::Stack::direct_jit(request.stack_size, has_spectral);
      Float3 closure_weight = make_float3(0.0f);
      std::optional<detail::SpectralClosureWeight> spectral_weight;
      if (has_spectral) {
        spectral_weight.emplace();
      }
      emit_jit_block(request, *groups.front().routine, case_stack,
                     closure_weight,
                     spectral_weight ? &*spectral_weight : nullptr);
      return;
    }
    const UInt shader_index = request.shader_data.shader & shader_mask;
    $switch(shader_index) {
      for (const auto &group : groups) {
        luisa::compute::detail::SwitchCaseStmtBuilder{
            luisa::span<const std::uint32_t>{group.shader_ids}} %
            [&]() noexcept {
              detail::DirectJitCaseCacheGuard attr_cache_guard;
              const bool has_spectral =
                  request.kernel_globals.is_spectral() &&
                  request.shader_data.spectral.has_value() &&
                  block_has_spectral_source(*group.routine);
              auto case_stack =
                  detail::Stack::direct_jit(request.stack_size, has_spectral);
              Float3 closure_weight = make_float3(0.0f);
              std::optional<detail::SpectralClosureWeight> spectral_weight;
              if (has_spectral) {
                spectral_weight.emplace();
              }
              emit_jit_block(request, *group.routine, case_stack,
                             closure_weight,
                             spectral_weight ? &*spectral_weight : nullptr);
            };
      }
      $default {};
    };
  };
}

SvmBackendStrategy resolve_svm_backend_strategy(
    const SvmTranslationRequest &request,
    SvmBackendStrategy preferred) noexcept {
  const auto override_strategy = env_backend_override();
  const auto desired = override_strategy != SvmBackendStrategy::auto_select
                           ? override_strategy
                           : preferred;
  if (desired == SvmBackendStrategy::bytecode_vm) {
    return SvmBackendStrategy::bytecode_vm;
  }
  if (!can_direct_jit(request)) {
    return SvmBackendStrategy::bytecode_vm;
  }
  if (desired == SvmBackendStrategy::direct_jit) {
    return SvmBackendStrategy::direct_jit;
  }
  const auto sc = static_cast<std::size_t>(request.table.shader_count);
  const bool has_mask = request.active_shader_mask.size() == sc;
  const auto host_words = std::span<const std::uint32_t>{request.table.words};
  luisa::vector<SvmBlockIr> pruned_routines;
  pruned_routines.reserve(sc);
  luisa::vector<const SvmBlockIr *> unique_routines;
  std::size_t active_nonempty = 0u;
  std::size_t unique_jit_weight = 0u;
  for (std::size_t i = 0u; i < sc; ++i) {
    if (has_mask && request.active_shader_mask[i] == 0u) {
      continue;
    }
    auto pruned = prune_svm_block_for_eval(
        request.table.ir.shaders[i].routine_for(request.shader_type),
        host_words, request.shader_type, request.node_feature_mask,
        request.shadow_eval_only);
    if (pruned.empty()) {
      continue;
    }
    ++active_nonempty;
    bool merged = false;
    for (const SvmBlockIr *existing : unique_routines) {
      if (same_linked_svm_block(host_words, *existing, pruned)) {
        merged = true;
        break;
      }
    }
    if (!merged) {
      unique_jit_weight += svm_block_jit_weight_impl(
          pruned, request.node_feature_mask, request.shadow_eval_only,
          host_words);
      pruned_routines.emplace_back(std::move(pruned));
      unique_routines.push_back(&pruned_routines.back());
    }
  }
  if (active_nonempty > configured_direct_jit_max_shaders()) {
    return SvmBackendStrategy::bytecode_vm;
  }
  const auto max_weight = configured_direct_jit_max_weight();
  if (unique_jit_weight > max_weight) {
    LUISA_INFO("Cycles SVM auto_select: falling back to BytecodeVm "
               "(active_nonempty={}, unique_jit_weight={}, limit={}).",
               active_nonempty, unique_jit_weight, max_weight);
    return SvmBackendStrategy::bytecode_vm;
  }
  return SvmBackendStrategy::direct_jit;
}

const SvmTranslator &svm_translator_for(SvmBackendStrategy strategy) noexcept {
  static const BytecodeVmTranslator vm_translator{};
  static const DirectJitTranslator jit_translator{};
  switch (strategy) {
    case SvmBackendStrategy::bytecode_vm:
      return vm_translator;
    case SvmBackendStrategy::direct_jit:
    case SvmBackendStrategy::auto_select:
      return jit_translator;
  }
  return jit_translator;
}

void translate_svm_nodes_assume_valid(const SvmTranslationRequest &request,
                                      SvmBackendStrategy preferred) noexcept {
  const auto resolved = resolve_svm_backend_strategy(request, preferred);
  svm_translator_for(resolved).translate(request);
}

} // namespace psycles::luisa_backend::cycles_svm
