/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "cycles_svm_volume.h"

#include <optional>

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) $case(static_cast<std::uint32_t>(node))
#define PSYCLES_SVM_OUTLINE_NODE(name, ...)                                      \
  do {                                                                           \
    __VA_ARGS__;                                                                 \
  } while (false)

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

TransformState::TransformState(
    Expr<luisa::float4x4> camera_to_world_transform,
    Expr<luisa::float4x4> world_to_camera_transform,
    Expr<luisa::float4x4> object_to_world_transform,
    Expr<luisa::float4x4> world_to_object_transform) noexcept
    : camera_to_world{camera_to_world_transform},
      world_to_camera{world_to_camera_transform},
      object_to_world{object_to_world_transform},
      world_to_object{world_to_object_transform} {}

ShaderData::ShaderData(
    Expr<luisa::float3> position, Expr<luisa::float3> normal,
    Expr<luisa::float3> geometric_normal, Expr<luisa::float3> incoming,
    Expr<std::uint32_t> primitive_type, Expr<std::uint32_t> shader_id,
    Expr<std::uint32_t> shader_flags, Expr<std::uint32_t> object_flags,
    Expr<std::uint32_t> primitive_id, Expr<float> parametric_u,
    Expr<float> parametric_v, Expr<std::uint32_t> object_id,
    Expr<float> motion_time, Expr<float> length,
    Expr<float> position_differential, Expr<float> incoming_differential,
    Expr<float> parametric_u_dx, Expr<float> parametric_u_dy,
    Expr<float> parametric_v_dx, Expr<float> parametric_v_dy,
    Expr<luisa::float3> position_u_derivative,
    Expr<luisa::float3> position_v_derivative,
    Expr<luisa::float4x4> motion_object_to_world,
    Expr<luisa::float4x4> motion_world_to_object,
    Expr<std::uint32_t> random_state, ClosurePool *closure_pool) noexcept
    : P{position}, ray_P{position}, N{normal}, Ng{geometric_normal},
      wi{incoming}, type{primitive_type}, shader{shader_id}, flag{shader_flags},
      object_flag{object_flags}, prim{primitive_id}, u{parametric_u},
      v{parametric_v}, object{object_id}, time{motion_time}, ray_length{length},
      dP{position_differential}, dI{incoming_differential},
      du{parametric_u_dx, parametric_u_dy},
      dv{parametric_v_dx, parametric_v_dy}, dPdu{position_u_derivative},
      dPdv{position_v_derivative}, ob_tfm_motion{motion_object_to_world},
      ob_itfm_motion{motion_world_to_object}, lcg_state{random_state},
      closure_emission_background{make_float3(0.0f)},
      closure_transparent_extinction{make_float3(0.0f)},
      closure{closure_pool} {}

PathState::PathState(Expr<std::uint32_t> path_visibility,
                     Expr<std::uint32_t> path_flag,
                     Expr<std::uint32_t> ray_bounce,
                     Expr<std::uint32_t> ray_transparent,
                     Expr<std::uint32_t> ray_diffuse,
                     Expr<std::uint32_t> ray_glossy,
                     Expr<std::uint32_t> ray_transmission,
                     Expr<std::uint32_t> ray_portal) noexcept
    : visibility{path_visibility}, flag{path_flag}, bounce{ray_bounce},
      transparent_bounce{ray_transparent}, diffuse_bounce{ray_diffuse},
      glossy_bounce{ray_glossy}, transmission_bounce{ray_transmission},
      portal_bounce{ray_portal} {}

EvaluationResult::EvaluationResult() noexcept
    : final_offset{0u},
      status{static_cast<std::uint32_t>(EvaluationStatus::running)},
      closure_weight{make_float3(0.0f)} {}

template<bool diagnose_failures>
void eval_nodes_impl(
    const KernelGlobals &kernel_globals, Expr<Buffer<luisa::uint>> words,
    ShaderType shader_type, std::uint32_t kernel_features,
    std::uint32_t node_feature_mask,
    const std::array<bool, NODE_NUM> &node_types_used,
    const TransformState &transform_state, ShaderData &shader_data,
    const PathState &path_state, EvaluationResult *result,
    std::size_t stack_size, NoiseUsage noise_usage) noexcept {
  LUISA_ASSERT(stack_size != 0u && stack_size <= SVM_STACK_SIZE,
               "Cycles SVM stack extent must be in [1, {}], got {}.",
               SVM_STACK_SIZE, stack_size);
  detail::Stack stack{stack_size};
  Float3 closure_weight = make_float3(0.0f);
  UInt offset = (shader_data.shader & shader_mask) *
                (1u + static_cast<std::uint32_t>(sizeof(SVMNodeShaderJump) /
                                                 sizeof(std::uint32_t)));
  std::optional<Bool> diagnostic_active;
  if constexpr (diagnose_failures) {
    diagnostic_active.emplace(true);
    result->status = static_cast<std::uint32_t>(EvaluationStatus::running);
  }

  const auto evaluate_one_node = [&]() noexcept {
    const auto node_type = words.read(offset);
    offset += 1u;
    detail::Cursor cursor{words, offset};
    std::optional<Bool> diagnostic_supported;
    if constexpr (diagnose_failures) {
      diagnostic_supported.emplace(true);
    }
    const detail::EvaluationTransition transition{
        diagnose_failures ? &*diagnostic_supported : nullptr};

    $switch(node_type) {
      if (node_types_used[NODE_END]) {
        PSYCLES_SVM_CASE(NODE_END) {
          if constexpr (diagnose_failures) {
            result->status =
                static_cast<std::uint32_t>(EvaluationStatus::ended);
            *diagnostic_active = false;
          } else {
            $return();
          }
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
            if constexpr (diagnose_failures) {
              result->status =
                  static_cast<std::uint32_t>(EvaluationStatus::ended);
              *diagnostic_active = false;
            } else {
              $return();
            }
            break;
          }
        };
      }
      if (node_types_used[NODE_CLOSURE_BSDF]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_BSDF) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_closure_bsdf",
              detail::node_closure_bsdf(
                  kernel_globals, cursor, stack, closure_weight, shader_type,
                  kernel_features, node_feature_mask, shader_data, path_state,
                  transition));
        };
      }
      if (node_types_used[NODE_CLOSURE_EMISSION]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_EMISSION) {
          if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_closure_emission",
                detail::node_closure_emission(kernel_globals, cursor, stack,
                                              closure_weight, shader_data,
                                              transition));
          }
        };
      }
      if (node_types_used[NODE_CLOSURE_VOLUME]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_VOLUME) {
          if ((node_feature_mask & kernel_feature_node_volume) != 0u) {
            if (shader_type == SHADER_TYPE_VOLUME &&
                (kernel_features & kernel_feature_volume) != 0u) {
              detail::node_closure_volume(kernel_globals, cursor, stack,
                                           closure_weight, shader_data, transition);
            } else {
              cursor.advance(static_cast<unsigned>(sizeof(SVMNodeClosureVolume) / sizeof(std::uint32_t)));
            }
          }
        };
      }
      if (node_types_used[NODE_VOLUME_COEFFICIENTS]) {
        PSYCLES_SVM_CASE(NODE_VOLUME_COEFFICIENTS) {
          if ((node_feature_mask & kernel_feature_node_volume) != 0u) {
            if (shader_type == SHADER_TYPE_VOLUME &&
                (kernel_features & kernel_feature_volume) != 0u) {
              detail::node_volume_coefficients(kernel_globals, cursor, stack,
                                                closure_weight, shader_data, path_state, transition);
            } else {
              cursor.advance(static_cast<unsigned>(sizeof(SVMNodeVolumeCoefficients) / sizeof(std::uint32_t)));
            }
          }
        };
      }
      if (node_types_used[NODE_PRINCIPLED_VOLUME]) {
        PSYCLES_SVM_CASE(NODE_PRINCIPLED_VOLUME) {
          if ((node_feature_mask & kernel_feature_node_volume) != 0u) {
            if (shader_type == SHADER_TYPE_VOLUME &&
                (kernel_features & kernel_feature_volume) != 0u) {
              detail::node_principled_volume(kernel_globals, cursor, stack,
                                             closure_weight, shader_data, path_state, transition);
            } else {
              cursor.advance(static_cast<unsigned>(sizeof(SVMNodePrincipledVolume) / sizeof(std::uint32_t)));
            }
          }
        };
      }
      if (node_types_used[NODE_CLOSURE_HOLDOUT]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_HOLDOUT) {
          detail::node_closure_holdout(cursor, stack, closure_weight, shader_data);
        };
      }
      if (node_types_used[NODE_CLOSURE_BACKGROUND]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_BACKGROUND) {
          if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_closure_background",
                detail::node_closure_background(cursor, stack, closure_weight,
                                                shader_data));
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
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_mix_closure",
              detail::node_mix_closure(cursor, stack));
        };
      }
      if (node_types_used[NODE_JUMP_IF_ZERO]) {
        PSYCLES_SVM_CASE(NODE_JUMP_IF_ZERO) {
          const auto jump_offset = cursor.word();
          const auto packed = cursor.word();
          const auto stack_offset = cursor.byte(packed, 0u);
          $if(detail::stack_load_float(stack, stack_offset) <= 0.0f) {
            offset += jump_offset;
          };
        };
      }
      if (node_types_used[NODE_JUMP_IF_ONE]) {
        PSYCLES_SVM_CASE(NODE_JUMP_IF_ONE) {
          const auto jump_offset = cursor.word();
          const auto packed = cursor.word();
          const auto stack_offset = cursor.byte(packed, 0u);
          $if(detail::stack_load_float(stack, stack_offset) >= 1.0f) {
            offset += jump_offset;
          };
        };
      }
      if (node_types_used[NODE_GEOMETRY]) {
        PSYCLES_SVM_CASE(NODE_GEOMETRY) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_geometry",
              detail::node_geometry(cursor, stack, kernel_globals,
                                    shader_data, false));
        };
      }
      if (node_types_used[NODE_GEOMETRY_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_GEOMETRY_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_geometry_derivative",
                detail::node_geometry(cursor, stack, kernel_globals,
                                      shader_data, true));
          }
        };
      }
      if (node_types_used[NODE_CAMERA]) {
        PSYCLES_SVM_CASE(NODE_CAMERA) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_camera",
              detail::node_camera(cursor, stack, transform_state,
                                  shader_data));
        };
      }
      if (node_types_used[NODE_FRESNEL]) {
        PSYCLES_SVM_CASE(NODE_FRESNEL) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_fresnel",
              detail::node_fresnel(cursor, stack, shader_data));
        };
      }
      if (node_types_used[NODE_LAYER_WEIGHT]) {
        PSYCLES_SVM_CASE(NODE_LAYER_WEIGHT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_layer_weight",
              detail::node_layer_weight(cursor, stack, shader_data));
        };
      }
      if (node_types_used[NODE_TEX_COORD]) {
        PSYCLES_SVM_CASE(NODE_TEX_COORD) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_coord",
              detail::node_tex_coord(
                  cursor, stack, kernel_globals, transform_state, shader_data,
                  path_state, false,
                  (node_feature_mask & kernel_feature_node_volume) != 0u,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_TEX_COORD_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_COORD_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_tex_coord_derivative",
                detail::node_tex_coord(
                    cursor, stack, kernel_globals, transform_state,
                    shader_data, path_state, true, false,
                    (kernel_features & kernel_feature_object_motion) != 0u));
          }
        };
      }
      if (node_types_used[NODE_TEX_IMAGE]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_image",
              detail::node_tex_image(cursor, stack, kernel_globals,
                                     shader_data, false));
        };
      }
      if (node_types_used[NODE_TEX_IMAGE_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_tex_image_derivative",
                detail::node_tex_image(cursor, stack, kernel_globals,
                                       shader_data, true));
          }
        };
      }
      if (node_types_used[NODE_TEX_IMAGE_BOX]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE_BOX) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_image_box",
              detail::node_tex_image_box(
                  cursor, stack, kernel_globals, transform_state, shader_data,
                  false,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_TEX_IMAGE_BOX_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE_BOX_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_tex_image_box_derivative",
                detail::node_tex_image_box(
                    cursor, stack, kernel_globals, transform_state, shader_data,
                    true,
                    (kernel_features & kernel_feature_object_motion) != 0u));
          }
        };
      }
      if (node_types_used[NODE_TEX_NOISE]) {
        PSYCLES_SVM_CASE(NODE_TEX_NOISE) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_noise",
              detail::node_tex_noise(cursor, stack, noise_usage));
        };
      }
      if (node_types_used[NODE_TEX_WHITE_NOISE]) {
        PSYCLES_SVM_CASE(NODE_TEX_WHITE_NOISE) {
          detail::node_tex_white_noise(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_GRADIENT]) {
        PSYCLES_SVM_CASE(NODE_TEX_GRADIENT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_gradient",
              detail::node_tex_gradient(cursor, stack));
        };
      }
      if (node_types_used[NODE_TEX_VORONOI]) {
        PSYCLES_SVM_CASE(NODE_TEX_VORONOI) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_voronoi",
              detail::node_tex_voronoi(
                  cursor, stack,
                  (node_feature_mask & kernel_feature_node_voronoi_extra) !=
                      0u));
        };
      }
      if (node_types_used[NODE_TEX_GABOR]) {
        PSYCLES_SVM_CASE(NODE_TEX_GABOR) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_gabor",
              detail::node_tex_gabor(cursor, stack));
        };
      }
      if (node_types_used[NODE_TEX_WAVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_WAVE) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_wave",
              detail::node_tex_wave(cursor, stack));
        };
      }
      if (node_types_used[NODE_TEX_MAGIC]) {
        PSYCLES_SVM_CASE(NODE_TEX_MAGIC) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_magic",
              detail::node_tex_magic(cursor, stack));
        };
      }
      if (node_types_used[NODE_TEX_CHECKER]) {
        PSYCLES_SVM_CASE(NODE_TEX_CHECKER) {
          detail::node_tex_checker(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_BRICK]) {
        PSYCLES_SVM_CASE(NODE_TEX_BRICK) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_brick", detail::node_tex_brick(cursor, stack));
        };
      }
      if (node_types_used[NODE_RGB_RAMP]) {
        PSYCLES_SVM_CASE(NODE_RGB_RAMP) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_rgb_ramp", detail::node_rgb_ramp(cursor, stack));
        };
      }
      if (node_types_used[NODE_CURVES]) {
        PSYCLES_SVM_CASE(NODE_CURVES) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_curves", detail::node_curves(cursor, stack));
        };
      }
      if (node_types_used[NODE_FLOAT_CURVE]) {
        PSYCLES_SVM_CASE(NODE_FLOAT_CURVE) {
          detail::node_float_curve(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_ENVIRONMENT]) {
        PSYCLES_SVM_CASE(NODE_TEX_ENVIRONMENT) {
          detail::node_tex_environment(cursor, stack, kernel_globals,
                                       shader_data, false);
        };
      }
      if (node_types_used[NODE_TEX_ENVIRONMENT_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_ENVIRONMENT_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_tex_environment(cursor, stack, kernel_globals,
                                         shader_data, true);
          }
        };
      }
      if (node_types_used[NODE_TEX_SKY]) {
        PSYCLES_SVM_CASE(NODE_TEX_SKY) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tex_sky",
              detail::node_tex_sky(cursor, stack, kernel_globals, shader_data,
                                   path_state));
        };
      }
      if (node_types_used[NODE_ATTR]) {
        PSYCLES_SVM_CASE(NODE_ATTR) {
          if ((node_feature_mask & kernel_feature_node_volume) != 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_attr_volume",
                detail::node_attr_volume(cursor, stack, kernel_globals,
                                         shader_data));
          } else {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_attr_surface",
                detail::node_attr_surface(cursor, stack, kernel_globals,
                                          shader_data));
          }
        };
      }
      if (node_types_used[NODE_ATTR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_ATTR_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_attr_derivative",
                detail::node_attr_derivative(cursor, stack, kernel_globals,
                                             shader_data));
          }
        };
      }
      if (node_types_used[NODE_VERTEX_COLOR]) {
        PSYCLES_SVM_CASE(NODE_VERTEX_COLOR) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_vertex_color",
              detail::node_vertex_color(cursor, stack, kernel_globals,
                                        shader_data));
        };
      }
      if (node_types_used[NODE_VERTEX_COLOR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_VERTEX_COLOR_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_vertex_color_derivative",
                detail::node_vertex_color_derivative(
                    cursor, stack, kernel_globals, shader_data));
          }
        };
      }
      if (node_types_used[NODE_CONVERT]) {
        PSYCLES_SVM_CASE(NODE_CONVERT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_convert_float3",
              detail::node_convert(cursor, stack, kernel_globals, false));
        };
      }
      if (node_types_used[NODE_CONVERT_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_CONVERT_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_convert(cursor, stack, kernel_globals, true);
          }
        };
      }
      if (node_types_used[NODE_VALUE_F]) {
        PSYCLES_SVM_CASE(NODE_VALUE_F) {
          detail::node_value_f(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_VALUE_F_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_VALUE_F_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_value_f(cursor, stack, true);
          }
        };
      }
      if (node_types_used[NODE_VALUE_V]) {
        PSYCLES_SVM_CASE(NODE_VALUE_V) {
          detail::node_value_v(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_VALUE_V_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_VALUE_V_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_value_v(cursor, stack, true);
          }
        };
      }
      if (node_types_used[NODE_MAPPING]) {
        PSYCLES_SVM_CASE(NODE_MAPPING) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_mapping_float3",
              detail::node_mapping(cursor, stack, false));
        };
      }
      if (node_types_used[NODE_MAPPING_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_MAPPING_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_mapping_derivative",
                detail::node_mapping(cursor, stack, true));
          }
        };
      }
      if (node_types_used[NODE_TEXTURE_MAPPING]) {
        PSYCLES_SVM_CASE(NODE_TEXTURE_MAPPING) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_texture_mapping_float3",
              detail::node_texture_mapping(cursor, stack, false));
        };
      }
      if (node_types_used[NODE_TEXTURE_MAPPING_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEXTURE_MAPPING_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_texture_mapping_derivative",
                detail::node_texture_mapping(cursor, stack, true));
          }
        };
      }
      if (node_types_used[NODE_MIN_MAX]) {
        PSYCLES_SVM_CASE(NODE_MIN_MAX) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_min_max", detail::node_min_max(cursor, stack));
        };
      }
      if (node_types_used[NODE_VECTOR_MATH]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_MATH) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_vector_math_float3",
              detail::node_vector_math(cursor, stack, false));
        };
      }
      if (node_types_used[NODE_VECTOR_MATH_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_MATH_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_vector_math_derivative",
                detail::node_vector_math(cursor, stack, true));
          }
        };
      }
      if (node_types_used[NODE_SET_BUMP]) {
        PSYCLES_SVM_CASE(NODE_SET_BUMP) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_set_bump",
              detail::node_set_bump(
                  cursor, stack, transform_state, shader_data,
                  (node_feature_mask & kernel_feature_node_bump) != 0u,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_CLOSURE_SET_NORMAL]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_SET_NORMAL) {
          if ((node_feature_mask & kernel_feature_node_bump) != 0u) {
            detail::node_set_normal(cursor, stack, shader_data);
          }
        };
      }
      if (node_types_used[NODE_ENTER_BUMP_EVAL]) {
        PSYCLES_SVM_CASE(NODE_ENTER_BUMP_EVAL) {
          if ((node_feature_mask & kernel_feature_node_bump_state) != 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_enter_bump_eval",
                detail::node_enter_bump_eval(
                    cursor, stack, kernel_globals, transform_state,
                    shader_data,
                    (kernel_features & kernel_feature_object_motion) != 0u));
          }
        };
      }
      if (node_types_used[NODE_LEAVE_BUMP_EVAL]) {
        PSYCLES_SVM_CASE(NODE_LEAVE_BUMP_EVAL) {
          if ((node_feature_mask & kernel_feature_node_bump_state) != 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_leave_bump_eval",
                detail::node_leave_bump_eval(cursor, stack, shader_data));
          }
        };
      }
      if (node_types_used[NODE_SET_DISPLACEMENT]) {
        PSYCLES_SVM_CASE(NODE_SET_DISPLACEMENT) {
          detail::node_set_displacement(
              cursor, stack, shader_data,
              (node_feature_mask & kernel_feature_node_bump) != 0u);
        };
      }
      if (node_types_used[NODE_DISPLACEMENT]) {
        PSYCLES_SVM_CASE(NODE_DISPLACEMENT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_displacement",
              detail::node_displacement(
                  cursor, stack, transform_state, shader_data,
                  (node_feature_mask & kernel_feature_node_bump) != 0u,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_VECTOR_DISPLACEMENT]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_DISPLACEMENT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_vector_displacement",
              detail::node_vector_displacement(
                  cursor, stack, kernel_globals, transform_state, shader_data,
                  (node_feature_mask & kernel_feature_node_bump) != 0u,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_HSV]) {
        PSYCLES_SVM_CASE(NODE_HSV) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_hsv", detail::node_hsv(cursor, stack));
        };
      }
      if (node_types_used[NODE_MATH]) {
        PSYCLES_SVM_CASE(NODE_MATH) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_math", detail::node_math(cursor, stack));
        };
      }
      if (node_types_used[NODE_GAMMA]) {
        PSYCLES_SVM_CASE(NODE_GAMMA) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_gamma", detail::node_gamma(cursor, stack));
        };
      }
      if (node_types_used[NODE_BRIGHTCONTRAST]) {
        PSYCLES_SVM_CASE(NODE_BRIGHTCONTRAST) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_brightness",
              detail::node_brightness(cursor, stack));
        };
      }
      if (node_types_used[NODE_WAVELENGTH]) {
        PSYCLES_SVM_CASE(NODE_WAVELENGTH) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_wavelength",
              detail::node_wavelength(cursor, stack, kernel_globals));
        };
      }
      if (node_types_used[NODE_BLACKBODY]) {
        PSYCLES_SVM_CASE(NODE_BLACKBODY) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_blackbody",
              detail::node_blackbody(cursor, stack, kernel_globals));
        };
      }
      if (node_types_used[NODE_LIGHT_PATH]) {
        PSYCLES_SVM_CASE(NODE_LIGHT_PATH) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_light_path",
              detail::node_light_path(cursor, stack, shader_data, path_state,
                                      node_feature_mask));
        };
      }
      if (node_types_used[NODE_OBJECT_INFO]) {
        PSYCLES_SVM_CASE(NODE_OBJECT_INFO) {
          if (const auto *services = kernel_globals.info_services()) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_object_info",
                detail::node_object_info(cursor, stack, *services,
                                         shader_data));
          } else {
            cursor.advance(2u);
            transition.unsupported();
          }
        };
      }
      if (node_types_used[NODE_PARTICLE_INFO]) {
        PSYCLES_SVM_CASE(NODE_PARTICLE_INFO) {
          if (const auto *services = kernel_globals.info_services()) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_particle_info",
                detail::node_particle_info(cursor, stack, *services,
                                           shader_data));
          } else {
            cursor.advance(2u);
            transition.unsupported();
          }
        };
      }
      if (node_types_used[NODE_HAIR_INFO] &&
          (kernel_features & kernel_feature_hair) != 0u) {
        PSYCLES_SVM_CASE(NODE_HAIR_INFO) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_hair_info",
              detail::node_hair_info(cursor, stack,
                                     kernel_globals.info_services(),
                                     shader_data, transition));
        };
      }
      if (node_types_used[NODE_POINT_INFO] &&
          (kernel_features & kernel_feature_pointcloud) != 0u) {
        PSYCLES_SVM_CASE(NODE_POINT_INFO) {
          if (const auto *services = kernel_globals.info_services()) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_point_info",
                detail::node_point_info(cursor, stack, *services,
                                        shader_data));
          } else {
            cursor.advance(2u);
            transition.unsupported();
          }
        };
      }
      if (node_types_used[NODE_INVERT]) {
        PSYCLES_SVM_CASE(NODE_INVERT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_invert", detail::node_invert(cursor, stack));
        };
      }
      if (node_types_used[NODE_MIX]) {
        PSYCLES_SVM_CASE(NODE_MIX) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_mix", detail::node_mix(cursor, stack));
        };
      }
      if (node_types_used[NODE_SEPARATE_COLOR]) {
        PSYCLES_SVM_CASE(NODE_SEPARATE_COLOR) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_separate_color",
              detail::node_separate_color(cursor, stack));
        };
      }
      if (node_types_used[NODE_COMBINE_COLOR]) {
        PSYCLES_SVM_CASE(NODE_COMBINE_COLOR) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_combine_color",
              detail::node_combine_color(cursor, stack));
        };
      }
      if (node_types_used[NODE_SEPARATE_VECTOR]) {
        PSYCLES_SVM_CASE(NODE_SEPARATE_VECTOR) {
          detail::node_separate_vector(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_SEPARATE_VECTOR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_SEPARATE_VECTOR_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_separate_vector(cursor, stack, true);
          }
        };
      }
      if (node_types_used[NODE_COMBINE_VECTOR]) {
        PSYCLES_SVM_CASE(NODE_COMBINE_VECTOR) {
          detail::node_combine_vector(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_COMBINE_VECTOR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_COMBINE_VECTOR_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_combine_vector(cursor, stack, true);
          }
        };
      }
      if (node_types_used[NODE_VECTOR_ROTATE]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_ROTATE) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_vector_rotate",
              detail::node_vector_rotate(cursor, stack));
        };
      }
      if (node_types_used[NODE_VECTOR_TRANSFORM]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_TRANSFORM) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_vector_transform",
              detail::node_vector_transform(
                  cursor, stack, transform_state, shader_data,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_NORMAL]) {
        PSYCLES_SVM_CASE(NODE_NORMAL) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_normal", detail::node_normal(cursor, stack));
        };
      }
      if (node_types_used[NODE_NORMAL_MAP]) {
        PSYCLES_SVM_CASE(NODE_NORMAL_MAP) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_normal_map",
              detail::node_normal_map(
                  cursor, stack, kernel_globals, transform_state, shader_data,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_TANGENT]) {
        PSYCLES_SVM_CASE(NODE_TANGENT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_tangent",
              detail::node_tangent(
                  cursor, stack, kernel_globals, transform_state, shader_data,
                  false,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_TANGENT_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TANGENT_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            PSYCLES_SVM_OUTLINE_NODE(
                "svm_node_tangent_derivative",
                detail::node_tangent(
                    cursor, stack, kernel_globals, transform_state, shader_data,
                    true,
                    (kernel_features & kernel_feature_object_motion) != 0u));
          }
        };
      }
      if (node_types_used[NODE_LIGHT_FALLOFF]) {
        PSYCLES_SVM_CASE(NODE_LIGHT_FALLOFF) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_light_falloff",
              detail::node_light_falloff(cursor, stack, shader_data));
        };
      }
      if (node_types_used[NODE_IES]) {
        PSYCLES_SVM_CASE(NODE_IES) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_ies",
              detail::node_ies(cursor, stack, kernel_globals));
        };
      }
      if (node_types_used[NODE_WIREFRAME]) {
        PSYCLES_SVM_CASE(NODE_WIREFRAME) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_wireframe",
              detail::node_wireframe(
                  cursor, stack, kernel_globals, transform_state, shader_data,
                  (kernel_features &
                   (kernel_feature_hair | kernel_feature_pointcloud)) != 0u,
                  (kernel_features & kernel_feature_object_motion) != 0u));
        };
      }
      if (node_types_used[NODE_MAP_RANGE]) {
        PSYCLES_SVM_CASE(NODE_MAP_RANGE) {
          detail::node_map_range(cursor, stack);
        };
      }
      if (node_types_used[NODE_VECTOR_MAP_RANGE]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_MAP_RANGE) {
          detail::node_vector_map_range(cursor, stack);
        };
      }
      if (node_types_used[NODE_CLAMP]) {
        PSYCLES_SVM_CASE(NODE_CLAMP) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_clamp", detail::node_clamp(cursor, stack));
        };
      }
      if (node_types_used[NODE_MIX_COLOR]) {
        PSYCLES_SVM_CASE(NODE_MIX_COLOR) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_mix_color", detail::node_mix_color(cursor, stack));
        };
      }
      if (node_types_used[NODE_MIX_FLOAT]) {
        PSYCLES_SVM_CASE(NODE_MIX_FLOAT) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_mix_float",
              detail::node_mix_float(cursor, stack));
        };
      }
      if (node_types_used[NODE_MIX_VECTOR]) {
        PSYCLES_SVM_CASE(NODE_MIX_VECTOR) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_mix_vector",
              detail::node_mix_vector(cursor, stack));
        };
      }
      if (node_types_used[NODE_MIX_VECTOR_NON_UNIFORM]) {
        PSYCLES_SVM_CASE(NODE_MIX_VECTOR_NON_UNIFORM) {
          PSYCLES_SVM_OUTLINE_NODE(
              "svm_node_mix_vector_non_uniform",
              detail::node_mix_vector_non_uniform(cursor, stack));
        };
      }
      $default {
        if constexpr (diagnose_failures) {
          result->status =
              static_cast<std::uint32_t>(EvaluationStatus::invalid_node);
          *diagnostic_active = false;
        } else {
          dsl::unreachable(
              "invalid node in compiler-validated Cycles SVM stream");
        }
      };
    };

    if constexpr (diagnose_failures) {
      $if(!*diagnostic_supported) {
        result->status =
            static_cast<std::uint32_t>(EvaluationStatus::unsupported_node);
        *diagnostic_active = false;
      };
    }
  };

  if constexpr (diagnose_failures) {
    $while(*diagnostic_active) { evaluate_one_node(); };
  } else {
    $loop { evaluate_one_node(); };
  };

  if constexpr (diagnose_failures) {
    result->final_offset = offset;
    result->closure_weight = closure_weight;
  }
}

void eval_nodes(const KernelGlobals &kernel_globals,
                Expr<Buffer<luisa::uint>> words,
                ShaderType shader_type, std::uint32_t kernel_features,
                std::uint32_t node_feature_mask,
                const std::array<bool, NODE_NUM> &node_types_used,
                const TransformState &transform_state, ShaderData &shader_data,
                const PathState &path_state,
                EvaluationResult &result, std::size_t stack_size,
                NoiseUsage noise_usage) noexcept {
  eval_nodes_impl<true>(kernel_globals, words, shader_type, kernel_features,
                        node_feature_mask, node_types_used, transform_state,
                        shader_data, path_state, &result, stack_size,
                        noise_usage);
}

void eval_nodes_assume_valid(
    const KernelGlobals &kernel_globals, Expr<Buffer<luisa::uint>> words,
    ShaderType shader_type, std::uint32_t kernel_features,
    std::uint32_t node_feature_mask,
    const std::array<bool, NODE_NUM> &node_types_used,
    const TransformState &transform_state, ShaderData &shader_data,
    const PathState &path_state, std::size_t stack_size,
    NoiseUsage noise_usage) noexcept {
  $outline_with_name("svm_eval_nodes") {
    eval_nodes_impl<false>(kernel_globals, words, shader_type, kernel_features,
                           node_feature_mask, node_types_used, transform_state,
                           shader_data, path_state, nullptr, stack_size,
                           noise_usage);
  };
}

} // namespace psycles::luisa_backend::cycles_svm

#undef PSYCLES_SVM_CASE
#undef PSYCLES_SVM_OUTLINE_NODE
