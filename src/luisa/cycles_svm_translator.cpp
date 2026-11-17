/* SPDX-FileCopyrightText: 2026 Psycles Contributors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/luisa/cycles_svm_translator.h>

#include "cycles_svm_internal.h"
#include "cycles_svm_volume.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

// Empirical Apple M5 Pro L1 I-cache / VGPR crossover: DirectJitTranslator wins
// decisively up through ~34-48 active non-empty shaders (Monster 19, Pabellon
// 22, BMW27 23, Lone Monk 34, plus Classroom's background/light/shadow domains),
// whereas 70+ material surface tables (Classroom surface_only = 76) stay more
// compact and faster in BytecodeVmTranslator.
inline constexpr std::size_t default_auto_direct_jit_max_shaders = 48u;

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

[[nodiscard]] std::size_t count_active_nonempty_shaders(
    const SvmTranslationRequest &request) noexcept {
  if (!can_direct_jit(request)) {
    return 0u;
  }
  const auto shader_count =
      static_cast<std::size_t>(request.table.shader_count);
  const bool has_mask = request.active_shader_mask.size() == shader_count;
  std::size_t nonempty = 0u;
  for (std::size_t i = 0u; i < shader_count; ++i) {
    if (has_mask && request.active_shader_mask[i] == 0u) {
      continue;
    }
    if (!request.table.ir.shaders[i].routine_for(request.shader_type).empty()) {
      ++nonempty;
    }
  }
  return nonempty;
}

class InstructionScopedKernelGlobals final
    : public detail::DelegatingKernelGlobals {
  const SvmInstructionSpecialization &_spec;

public:
  InstructionScopedKernelGlobals(
      const KernelGlobals &inner,
      const SvmInstructionSpecialization &spec) noexcept
      : DelegatingKernelGlobals{inner}, _spec{spec} {}

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

void emit_jit_instruction(const SvmTranslationRequest &req,
                          const SvmInstructionIr &inst, detail::Stack &stack,
                          Float3 &closure_weight) noexcept {
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
                                          inst.specialization};
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
      detail::node_closure_bsdf(kg, cursor, stack, closure_weight,
                                req.shader_type, kernel_features,
                                node_feature_mask, req.shader_data,
                                req.path_state, transition);
      return;
    case NODE_CLOSURE_EMISSION:
      if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
        detail::node_closure_emission(kg, cursor, stack, closure_weight,
                                      req.shader_data, transition);
      }
      return;
    case NODE_CLOSURE_VOLUME:
      if (is_volume_mask && req.shader_type == SHADER_TYPE_VOLUME &&
          (kernel_features & kernel_feature_volume) != 0u) {
        detail::node_closure_volume(kg, cursor, stack, closure_weight,
                                    req.shader_data, transition);
      }
      return;
    case NODE_VOLUME_COEFFICIENTS:
      if (is_volume_mask && req.shader_type == SHADER_TYPE_VOLUME &&
          (kernel_features & kernel_feature_volume) != 0u) {
        detail::node_volume_coefficients(kg, cursor, stack, closure_weight,
                                         req.shader_data, req.path_state,
                                         transition);
      }
      return;
    case NODE_PRINCIPLED_VOLUME:
      if (is_volume_mask && req.shader_type == SHADER_TYPE_VOLUME &&
          (kernel_features & kernel_feature_volume) != 0u) {
        detail::node_principled_volume(kg, cursor, stack, closure_weight,
                                       req.shader_data, req.path_state,
                                       transition);
      }
      return;
    case NODE_CLOSURE_HOLDOUT:
      detail::node_closure_holdout(cursor, stack, closure_weight,
                                   req.shader_data);
      return;
    case NODE_CLOSURE_BACKGROUND:
      if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
        detail::node_closure_background(cursor, stack, closure_weight,
                                        req.shader_data);
      }
      return;
    case NODE_CLOSURE_SET_WEIGHT:
      detail::node_closure_set_weight(cursor, closure_weight);
      return;
    case NODE_CLOSURE_WEIGHT:
      detail::node_closure_weight(cursor, stack, closure_weight);
      return;
    case NODE_EMISSION_WEIGHT:
      if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
        detail::node_emission_weight(cursor, stack, closure_weight);
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
      detail::node_tex_image(cursor, stack, kg, req.shader_data, false);
      return;
    case NODE_TEX_IMAGE_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_tex_image(cursor, stack, kg, req.shader_data, true);
      }
      return;
    case NODE_TEX_IMAGE_BOX:
      detail::node_tex_image_box(cursor, stack, kg, req.transform_state,
                                 req.shader_data, false, object_motion);
      return;
    case NODE_TEX_IMAGE_BOX_DERIVATIVE:
      if (!is_volume_mask) {
        detail::node_tex_image_box(cursor, stack, kg, req.transform_state,
                                   req.shader_data, true, object_motion);
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
      detail::node_wavelength(cursor, stack, kg);
      return;
    case NODE_BLACKBODY:
      detail::node_blackbody(cursor, stack, kg);
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
    default:
      dsl::unreachable(
          "invalid node in DirectJitTranslator Cycles SVM stream");
      return;
  }
}

void emit_jit_block(const SvmTranslationRequest &req, const SvmBlockIr &block,
                    detail::Stack &stack, Float3 &closure_weight) noexcept {
  for (const auto &op : block) {
    if (op.is_conditional) {
      if (op.children.empty()) {
        continue;
      }
      const auto cond_val = detail::stack_load_float(
          stack, static_cast<std::uint32_t>(op.cond_stack_offset));
      const Bool take_children =
          op.jump_if_one ? !(cond_val >= 1.0f) : !(cond_val <= 0.0f);
      $if(take_children) {
        emit_jit_block(req, op.children, stack, closure_weight);
      };
    } else {
      emit_jit_instruction(req, op.instruction, stack, closure_weight);
    }
  }
}

} // namespace

void BytecodeVmTranslator::translate(
    const SvmTranslationRequest &request) const noexcept {
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

  luisa::vector<ShaderGroup> groups;
  std::size_t active_empty_count = 0u;
  for (std::size_t i = 0u; i < shader_count; ++i) {
    if (has_mask && request.active_shader_mask[i] == 0u) {
      continue;
    }
    const auto &routine =
        request.table.ir.shaders[i].routine_for(request.shader_type);
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

  $outline_with_name("svm_eval_nodes") {
    detail::Stack stack{request.stack_size};
    static_cast<void>(stack);
    if (groups.empty()) {
      return;
    }
    if (groups.size() == 1u && active_empty_count == 0u) {
      Float3 closure_weight = make_float3(0.0f);
      emit_jit_block(request, *groups.front().routine, stack, closure_weight);
      return;
    }
    const UInt shader_index = request.shader_data.shader & shader_mask;
    $switch(shader_index) {
      for (const auto &group : groups) {
        luisa::compute::detail::SwitchCaseStmtBuilder{
            luisa::span<const std::uint32_t>{group.shader_ids}} %
            [&]() noexcept {
              Float3 closure_weight = make_float3(0.0f);
              emit_jit_block(request, *group.routine, stack, closure_weight);
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
  const auto active_nonempty = count_active_nonempty_shaders(request);
  return active_nonempty <= configured_direct_jit_max_shaders()
             ? SvmBackendStrategy::direct_jit
             : SvmBackendStrategy::bytecode_vm;
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
