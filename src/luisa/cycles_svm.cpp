/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

ShaderData::ShaderData(Expr<luisa::float3> position,
                       Expr<luisa::float3> normal,
                       Expr<luisa::float3> geometric_normal,
                       Expr<luisa::float3> incoming,
                       Expr<std::uint32_t> shader_id,
                       Expr<std::uint32_t> shader_flags,
                       Expr<float> parametric_u,
                       Expr<float> parametric_v,
                       Expr<float> length) noexcept
    : P{position},
      N{normal},
      Ng{geometric_normal},
      wi{incoming},
      shader{shader_id},
      flag{shader_flags},
      u{parametric_u},
      v{parametric_v},
      ray_length{length},
      closure_emission_background{make_float3(0.0f)},
      closure_transparent_extinction{make_float3(0.0f)} {}

PathState::PathState(Expr<std::uint32_t> path_visibility,
                     Expr<std::uint32_t> path_flag,
                     Expr<std::uint32_t> ray_bounce,
                     Expr<std::uint32_t> ray_transparent,
                     Expr<std::uint32_t> ray_diffuse,
                     Expr<std::uint32_t> ray_glossy,
                     Expr<std::uint32_t> ray_transmission,
                     Expr<std::uint32_t> ray_portal) noexcept
    : visibility{path_visibility},
      flag{path_flag},
      bounce{ray_bounce},
      transparent_bounce{ray_transparent},
      diffuse_bounce{ray_diffuse},
      glossy_bounce{ray_glossy},
      transmission_bounce{ray_transmission},
      portal_bounce{ray_portal} {}

EvaluationResult::EvaluationResult() noexcept
    : final_offset{0u},
      status{static_cast<std::uint32_t>(EvaluationStatus::running)},
      closure_weight{make_float3(0.0f)} {}

void eval_nodes(
    const BufferUInt &words,
    ShaderType shader_type,
    std::uint32_t node_feature_mask,
    const std::array<bool, NODE_NUM> &node_types_used,
    ShaderData &shader_data,
    const PathState &path_state,
    EvaluationResult &result) noexcept {
  detail::Stack stack;
  Float3 closure_weight = make_float3(0.0f);
  UInt offset = (shader_data.shader & shader_mask) *
                (1u + static_cast<std::uint32_t>(
                          sizeof(SVMNodeShaderJump) /
                          sizeof(std::uint32_t)));
  Bool active = true;
  result.status = static_cast<std::uint32_t>(EvaluationStatus::running);

  $while (active) {
    const auto node_type = words.read(offset);
    offset += 1u;
    detail::Cursor cursor{words, offset};
    Bool transition_supported = true;

    $switch (node_type) {
      if (node_types_used[NODE_END]) {
        PSYCLES_SVM_CASE(NODE_END) {
          result.status =
              static_cast<std::uint32_t>(EvaluationStatus::ended);
          active = false;
        };
      }
      if (node_types_used[NODE_SHADER_JUMP]) {
        PSYCLES_SVM_CASE(NODE_SHADER_JUMP) {
          const auto offset_surface = cursor.word();
          const auto offset_volume = cursor.word();
          const auto offset_displacement = cursor.word();
          switch (shader_type) {
            case SHADER_TYPE_SURFACE:
              offset = offset_surface;
              break;
            case SHADER_TYPE_VOLUME:
              offset = offset_volume;
              break;
            case SHADER_TYPE_DISPLACEMENT:
              offset = offset_displacement;
              break;
            case SHADER_TYPE_BUMP:
              result.status =
                  static_cast<std::uint32_t>(EvaluationStatus::ended);
              active = false;
              break;
          }
        };
      }
      if (node_types_used[NODE_CLOSURE_BSDF]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_BSDF) {
          const auto closure_type = cursor.word();
          const auto packed = cursor.word();
          const auto mix_weight_offset = cursor.byte(packed, 0u);
          const auto mix_weight = detail::stack_load_float_default(
              stack, mix_weight_offset, 1.0f);
          Bool needs_unported_transition = false;

          if (shader_type == SHADER_TYPE_SURFACE) {
            if ((node_feature_mask & kernel_feature_node_bsdf) != 0u) {
              $if (mix_weight != 0.0f) {
                needs_unported_transition = true;
              };
            } else if ((node_feature_mask &
                        kernel_feature_node_emission) != 0u) {
              $if ((mix_weight != 0.0f) &
                   (closure_type == static_cast<std::uint32_t>(
                                        CLOSURE_BSDF_PRINCIPLED_ID))) {
                needs_unported_transition = true;
              };
            }
          }
          detail::node_closure_bsdf_skip(cursor, closure_type);
          $if (needs_unported_transition) { transition_supported = false; };
        };
      }
      if (node_types_used[NODE_CLOSURE_EMISSION]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_EMISSION) {
          if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
            detail::node_closure_emission(cursor, stack, closure_weight,
                                          shader_data,
                                          transition_supported);
          }
        };
      }
      if (node_types_used[NODE_CLOSURE_SET_WEIGHT]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_SET_WEIGHT) {
          detail::node_closure_set_weight(cursor, closure_weight);
        };
      }
      if (node_types_used[NODE_CLOSURE_WEIGHT]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_WEIGHT) {
          detail::node_closure_weight(cursor, stack, closure_weight);
        };
      }
      if (node_types_used[NODE_EMISSION_WEIGHT]) {
        PSYCLES_SVM_CASE(NODE_EMISSION_WEIGHT) {
          if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
            detail::node_emission_weight(cursor, stack, closure_weight);
          }
        };
      }
      if (node_types_used[NODE_MIX_CLOSURE]) {
        PSYCLES_SVM_CASE(NODE_MIX_CLOSURE) {
          detail::node_mix_closure(cursor, stack);
        };
      }
      if (node_types_used[NODE_JUMP_IF_ZERO]) {
        PSYCLES_SVM_CASE(NODE_JUMP_IF_ZERO) {
          const auto jump_offset = cursor.word();
          const auto packed = cursor.word();
          const auto stack_offset = cursor.byte(packed, 0u);
          $if (detail::stack_load_float(stack, stack_offset) <= 0.0f) {
            offset += jump_offset;
          };
        };
      }
      if (node_types_used[NODE_JUMP_IF_ONE]) {
        PSYCLES_SVM_CASE(NODE_JUMP_IF_ONE) {
          const auto jump_offset = cursor.word();
          const auto packed = cursor.word();
          const auto stack_offset = cursor.byte(packed, 0u);
          $if (detail::stack_load_float(stack, stack_offset) >= 1.0f) {
            offset += jump_offset;
          };
        };
      }
      if (node_types_used[NODE_GEOMETRY]) {
        PSYCLES_SVM_CASE(NODE_GEOMETRY) {
          detail::node_geometry(cursor, stack, shader_data,
                                transition_supported);
        };
      }
      if (node_types_used[NODE_VALUE_F]) {
        PSYCLES_SVM_CASE(NODE_VALUE_F) { detail::node_value_f(cursor, stack); };
      }
      if (node_types_used[NODE_VALUE_V]) {
        PSYCLES_SVM_CASE(NODE_VALUE_V) { detail::node_value_v(cursor, stack); };
      }
      if (node_types_used[NODE_HSV]) {
        PSYCLES_SVM_CASE(NODE_HSV) { detail::node_hsv(cursor, stack); };
      }
      if (node_types_used[NODE_MATH]) {
        PSYCLES_SVM_CASE(NODE_MATH) { detail::node_math(cursor, stack); };
      }
      if (node_types_used[NODE_GAMMA]) {
        PSYCLES_SVM_CASE(NODE_GAMMA) { detail::node_gamma(cursor, stack); };
      }
      if (node_types_used[NODE_BRIGHTCONTRAST]) {
        PSYCLES_SVM_CASE(NODE_BRIGHTCONTRAST) {
          detail::node_brightness(cursor, stack);
        };
      }
      if (node_types_used[NODE_LIGHT_PATH]) {
        PSYCLES_SVM_CASE(NODE_LIGHT_PATH) {
          detail::node_light_path(cursor, stack, shader_data, path_state,
                                  node_feature_mask);
        };
      }
      if (node_types_used[NODE_INVERT]) {
        PSYCLES_SVM_CASE(NODE_INVERT) { detail::node_invert(cursor, stack); };
      }
      if (node_types_used[NODE_CLAMP]) {
        PSYCLES_SVM_CASE(NODE_CLAMP) { detail::node_clamp(cursor, stack); };
      }
      $default {
        result.status =
            static_cast<std::uint32_t>(EvaluationStatus::invalid_node);
        active = false;
      };
    };

    $if (!transition_supported) {
      result.status =
          static_cast<std::uint32_t>(EvaluationStatus::unsupported_node);
      active = false;
    };
  };

  result.final_offset = offset;
  result.closure_weight = closure_weight;
}

} // namespace psycles::luisa_backend::cycles_svm

#undef PSYCLES_SVM_CASE
