/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) $case(static_cast<std::uint32_t>(node))

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
      closure_transparent_extinction{make_float3(0.0f)}, closure{closure_pool} {
}

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

void eval_nodes(const KernelGlobals &kernel_globals,
                Expr<Buffer<luisa::uint>> words,
                ShaderType shader_type, std::uint32_t kernel_features,
                std::uint32_t node_feature_mask,
                const std::array<bool, NODE_NUM> &node_types_used,
                const TransformState &transform_state, ShaderData &shader_data,
                const PathState &path_state,
                EvaluationResult &result) noexcept {
  detail::Stack stack;
  Float3 closure_weight = make_float3(0.0f);
  UInt offset = (shader_data.shader & shader_mask) *
                (1u + static_cast<std::uint32_t>(sizeof(SVMNodeShaderJump) /
                                                 sizeof(std::uint32_t)));
  Bool active = true;
  result.status = static_cast<std::uint32_t>(EvaluationStatus::running);

  $while(active) {
    const auto node_type = words.read(offset);
    offset += 1u;
    detail::Cursor cursor{words, offset};
    Bool transition_supported = true;

    $switch(node_type) {
      if (node_types_used[NODE_END]) {
        PSYCLES_SVM_CASE(NODE_END) {
          result.status = static_cast<std::uint32_t>(EvaluationStatus::ended);
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
            result.status = static_cast<std::uint32_t>(EvaluationStatus::ended);
            active = false;
            break;
          }
        };
      }
      if (node_types_used[NODE_CLOSURE_BSDF]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_BSDF) {
          detail::node_closure_bsdf(kernel_globals, cursor, stack,
                                    closure_weight, shader_type,
                                    node_feature_mask, shader_data, path_state,
                                    transition_supported);
        };
      }
      if (node_types_used[NODE_CLOSURE_EMISSION]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_EMISSION) {
          if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
            detail::node_closure_emission(cursor, stack, closure_weight,
                                          shader_data, transition_supported);
          }
        };
      }
      if (node_types_used[NODE_CLOSURE_BACKGROUND]) {
        PSYCLES_SVM_CASE(NODE_CLOSURE_BACKGROUND) {
          detail::node_closure_background(cursor, stack, closure_weight,
                                          shader_data);
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
          detail::node_geometry(cursor, stack, kernel_globals, shader_data,
                                false);
        };
      }
      if (node_types_used[NODE_GEOMETRY_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_GEOMETRY_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_geometry(cursor, stack, kernel_globals, shader_data,
                                  true);
          }
        };
      }
      if (node_types_used[NODE_CAMERA]) {
        PSYCLES_SVM_CASE(NODE_CAMERA) {
          detail::node_camera(cursor, stack, transform_state, shader_data);
        };
      }
      if (node_types_used[NODE_FRESNEL]) {
        PSYCLES_SVM_CASE(NODE_FRESNEL) {
          detail::node_fresnel(cursor, stack, shader_data);
        };
      }
      if (node_types_used[NODE_LAYER_WEIGHT]) {
        PSYCLES_SVM_CASE(NODE_LAYER_WEIGHT) {
          detail::node_layer_weight(cursor, stack, shader_data);
        };
      }
      if (node_types_used[NODE_TEX_COORD]) {
        PSYCLES_SVM_CASE(NODE_TEX_COORD) {
          detail::node_tex_coord(
              cursor, stack, kernel_globals, transform_state, shader_data,
              path_state, false,
              (node_feature_mask & kernel_feature_node_volume) != 0u,
              (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_TEX_COORD_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_COORD_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_tex_coord(
                cursor, stack, kernel_globals, transform_state, shader_data,
                path_state, true, false,
                (kernel_features & kernel_feature_object_motion) != 0u);
          }
        };
      }
      if (node_types_used[NODE_TEX_IMAGE]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE) {
          detail::node_tex_image(cursor, stack, kernel_globals, shader_data,
                                 false);
        };
      }
      if (node_types_used[NODE_TEX_IMAGE_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE_DERIVATIVE) {
          detail::node_tex_image(cursor, stack, kernel_globals, shader_data,
                                 true);
        };
      }
      if (node_types_used[NODE_TEX_IMAGE_BOX]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE_BOX) {
          detail::node_tex_image_box(
              cursor, stack, kernel_globals, transform_state, shader_data,
              false, (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_TEX_IMAGE_BOX_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_IMAGE_BOX_DERIVATIVE) {
          detail::node_tex_image_box(
              cursor, stack, kernel_globals, transform_state, shader_data, true,
              (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_TEX_NOISE]) {
        PSYCLES_SVM_CASE(NODE_TEX_NOISE) {
          detail::node_tex_noise(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_WHITE_NOISE]) {
        PSYCLES_SVM_CASE(NODE_TEX_WHITE_NOISE) {
          detail::node_tex_white_noise(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_GRADIENT]) {
        PSYCLES_SVM_CASE(NODE_TEX_GRADIENT) {
          detail::node_tex_gradient(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_VORONOI]) {
        PSYCLES_SVM_CASE(NODE_TEX_VORONOI) {
          detail::node_tex_voronoi(
              cursor, stack,
              (node_feature_mask & kernel_feature_node_voronoi_extra) != 0u);
        };
      }
      if (node_types_used[NODE_TEX_GABOR]) {
        PSYCLES_SVM_CASE(NODE_TEX_GABOR) {
          detail::node_tex_gabor(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_WAVE]) {
        PSYCLES_SVM_CASE(NODE_TEX_WAVE) {
          detail::node_tex_wave(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_MAGIC]) {
        PSYCLES_SVM_CASE(NODE_TEX_MAGIC) {
          detail::node_tex_magic(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_CHECKER]) {
        PSYCLES_SVM_CASE(NODE_TEX_CHECKER) {
          detail::node_tex_checker(cursor, stack);
        };
      }
      if (node_types_used[NODE_TEX_BRICK]) {
        PSYCLES_SVM_CASE(NODE_TEX_BRICK) {
          detail::node_tex_brick(cursor, stack);
        };
      }
      if (node_types_used[NODE_RGB_RAMP]) {
        PSYCLES_SVM_CASE(NODE_RGB_RAMP) {
          detail::node_rgb_ramp(cursor, stack);
        };
      }
      if (node_types_used[NODE_CURVES]) {
        PSYCLES_SVM_CASE(NODE_CURVES) { detail::node_curves(cursor, stack); };
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
          detail::node_tex_environment(cursor, stack, kernel_globals,
                                       shader_data, true);
        };
      }
      if (node_types_used[NODE_TEX_SKY]) {
        PSYCLES_SVM_CASE(NODE_TEX_SKY) {
          detail::node_tex_sky(cursor, stack, kernel_globals, shader_data,
                               path_state);
        };
      }
      if (node_types_used[NODE_ATTR]) {
        PSYCLES_SVM_CASE(NODE_ATTR) {
          if ((node_feature_mask & kernel_feature_node_volume) != 0u) {
            detail::node_attr_volume(cursor, stack, kernel_globals,
                                     shader_data);
          } else {
            detail::node_attr_surface(cursor, stack, kernel_globals,
                                      shader_data);
          }
        };
      }
      if (node_types_used[NODE_ATTR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_ATTR_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_attr_derivative(cursor, stack, kernel_globals,
                                         shader_data);
          }
        };
      }
      if (node_types_used[NODE_VERTEX_COLOR]) {
        PSYCLES_SVM_CASE(NODE_VERTEX_COLOR) {
          detail::node_vertex_color(cursor, stack, kernel_globals, shader_data);
        };
      }
      if (node_types_used[NODE_VERTEX_COLOR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_VERTEX_COLOR_DERIVATIVE) {
          if ((node_feature_mask & kernel_feature_node_volume) == 0u) {
            detail::node_vertex_color_derivative(cursor, stack, kernel_globals,
                                                 shader_data);
          }
        };
      }
      if (node_types_used[NODE_CONVERT]) {
        PSYCLES_SVM_CASE(NODE_CONVERT) {
          detail::node_convert(cursor, stack, kernel_globals, false);
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
        PSYCLES_SVM_CASE(NODE_VALUE_F) { detail::node_value_f(cursor, stack); };
      }
      if (node_types_used[NODE_VALUE_V]) {
        PSYCLES_SVM_CASE(NODE_VALUE_V) { detail::node_value_v(cursor, stack); };
      }
      if (node_types_used[NODE_MAPPING]) {
        PSYCLES_SVM_CASE(NODE_MAPPING) {
          detail::node_mapping(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_MAPPING_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_MAPPING_DERIVATIVE) {
          detail::node_mapping(cursor, stack, true);
        };
      }
      if (node_types_used[NODE_TEXTURE_MAPPING]) {
        PSYCLES_SVM_CASE(NODE_TEXTURE_MAPPING) {
          detail::node_texture_mapping(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_TEXTURE_MAPPING_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TEXTURE_MAPPING_DERIVATIVE) {
          detail::node_texture_mapping(cursor, stack, true);
        };
      }
      if (node_types_used[NODE_MIN_MAX]) {
        PSYCLES_SVM_CASE(NODE_MIN_MAX) { detail::node_min_max(cursor, stack); };
      }
      if (node_types_used[NODE_VECTOR_MATH]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_MATH) {
          detail::node_vector_math(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_VECTOR_MATH_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_MATH_DERIVATIVE) {
          detail::node_vector_math(cursor, stack, true);
        };
      }
      if (node_types_used[NODE_SET_BUMP]) {
        PSYCLES_SVM_CASE(NODE_SET_BUMP) {
          detail::node_set_bump(
              cursor, stack, transform_state, shader_data,
              (node_feature_mask & kernel_feature_node_bump) != 0u,
              (kernel_features & kernel_feature_object_motion) != 0u);
        };
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
      if (node_types_used[NODE_WAVELENGTH]) {
        PSYCLES_SVM_CASE(NODE_WAVELENGTH) {
          detail::node_wavelength(cursor, stack, kernel_globals);
        };
      }
      if (node_types_used[NODE_BLACKBODY]) {
        PSYCLES_SVM_CASE(NODE_BLACKBODY) {
          detail::node_blackbody(cursor, stack, kernel_globals);
        };
      }
      if (node_types_used[NODE_LIGHT_PATH]) {
        PSYCLES_SVM_CASE(NODE_LIGHT_PATH) {
          detail::node_light_path(cursor, stack, shader_data, path_state,
                                  node_feature_mask);
        };
      }
      if (node_types_used[NODE_OBJECT_INFO]) {
        PSYCLES_SVM_CASE(NODE_OBJECT_INFO) {
          if (const auto *services = kernel_globals.info_services()) {
            detail::node_object_info(cursor, stack, *services, shader_data);
          } else {
            cursor.advance(2u);
            transition_supported = false;
          }
        };
      }
      if (node_types_used[NODE_PARTICLE_INFO]) {
        PSYCLES_SVM_CASE(NODE_PARTICLE_INFO) {
          if (const auto *services = kernel_globals.info_services()) {
            detail::node_particle_info(cursor, stack, *services, shader_data);
          } else {
            cursor.advance(2u);
            transition_supported = false;
          }
        };
      }
      if (node_types_used[NODE_HAIR_INFO] &&
          (kernel_features & kernel_feature_hair) != 0u) {
        PSYCLES_SVM_CASE(NODE_HAIR_INFO) {
          detail::node_hair_info(cursor, stack, kernel_globals.info_services(),
                                 shader_data, transition_supported);
        };
      }
      if (node_types_used[NODE_POINT_INFO] &&
          (kernel_features & kernel_feature_pointcloud) != 0u) {
        PSYCLES_SVM_CASE(NODE_POINT_INFO) {
          if (const auto *services = kernel_globals.info_services()) {
            detail::node_point_info(cursor, stack, *services, shader_data);
          } else {
            cursor.advance(2u);
            transition_supported = false;
          }
        };
      }
      if (node_types_used[NODE_INVERT]) {
        PSYCLES_SVM_CASE(NODE_INVERT) { detail::node_invert(cursor, stack); };
      }
      if (node_types_used[NODE_MIX]) {
        PSYCLES_SVM_CASE(NODE_MIX) { detail::node_mix(cursor, stack); };
      }
      if (node_types_used[NODE_SEPARATE_COLOR]) {
        PSYCLES_SVM_CASE(NODE_SEPARATE_COLOR) {
          detail::node_separate_color(cursor, stack);
        };
      }
      if (node_types_used[NODE_COMBINE_COLOR]) {
        PSYCLES_SVM_CASE(NODE_COMBINE_COLOR) {
          detail::node_combine_color(cursor, stack);
        };
      }
      if (node_types_used[NODE_SEPARATE_VECTOR]) {
        PSYCLES_SVM_CASE(NODE_SEPARATE_VECTOR) {
          detail::node_separate_vector(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_SEPARATE_VECTOR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_SEPARATE_VECTOR_DERIVATIVE) {
          detail::node_separate_vector(cursor, stack, true);
        };
      }
      if (node_types_used[NODE_COMBINE_VECTOR]) {
        PSYCLES_SVM_CASE(NODE_COMBINE_VECTOR) {
          detail::node_combine_vector(cursor, stack, false);
        };
      }
      if (node_types_used[NODE_COMBINE_VECTOR_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_COMBINE_VECTOR_DERIVATIVE) {
          detail::node_combine_vector(cursor, stack, true);
        };
      }
      if (node_types_used[NODE_VECTOR_ROTATE]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_ROTATE) {
          detail::node_vector_rotate(cursor, stack);
        };
      }
      if (node_types_used[NODE_VECTOR_TRANSFORM]) {
        PSYCLES_SVM_CASE(NODE_VECTOR_TRANSFORM) {
          detail::node_vector_transform(
              cursor, stack, transform_state, shader_data,
              (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_NORMAL]) {
        PSYCLES_SVM_CASE(NODE_NORMAL) { detail::node_normal(cursor, stack); };
      }
      if (node_types_used[NODE_NORMAL_MAP]) {
        PSYCLES_SVM_CASE(NODE_NORMAL_MAP) {
          detail::node_normal_map(
              cursor, stack, kernel_globals, transform_state, shader_data,
              (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_TANGENT]) {
        PSYCLES_SVM_CASE(NODE_TANGENT) {
          detail::node_tangent(
              cursor, stack, kernel_globals, transform_state, shader_data,
              false, (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_TANGENT_DERIVATIVE]) {
        PSYCLES_SVM_CASE(NODE_TANGENT_DERIVATIVE) {
          detail::node_tangent(
              cursor, stack, kernel_globals, transform_state, shader_data, true,
              (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_LIGHT_FALLOFF]) {
        PSYCLES_SVM_CASE(NODE_LIGHT_FALLOFF) {
          detail::node_light_falloff(cursor, stack, shader_data);
        };
      }
      if (node_types_used[NODE_IES]) {
        PSYCLES_SVM_CASE(NODE_IES) {
          detail::node_ies(cursor, stack, kernel_globals);
        };
      }
      if (node_types_used[NODE_WIREFRAME]) {
        PSYCLES_SVM_CASE(NODE_WIREFRAME) {
          detail::node_wireframe(
              cursor, stack, kernel_globals, transform_state, shader_data,
              (kernel_features &
               (kernel_feature_hair | kernel_feature_pointcloud)) != 0u,
              (kernel_features & kernel_feature_object_motion) != 0u);
        };
      }
      if (node_types_used[NODE_CLAMP]) {
        PSYCLES_SVM_CASE(NODE_CLAMP) { detail::node_clamp(cursor, stack); };
      }
      if (node_types_used[NODE_MIX_COLOR]) {
        PSYCLES_SVM_CASE(NODE_MIX_COLOR) {
          detail::node_mix_color(cursor, stack);
        };
      }
      if (node_types_used[NODE_MIX_FLOAT]) {
        PSYCLES_SVM_CASE(NODE_MIX_FLOAT) {
          detail::node_mix_float(cursor, stack);
        };
      }
      if (node_types_used[NODE_MIX_VECTOR]) {
        PSYCLES_SVM_CASE(NODE_MIX_VECTOR) {
          detail::node_mix_vector(cursor, stack);
        };
      }
      if (node_types_used[NODE_MIX_VECTOR_NON_UNIFORM]) {
        PSYCLES_SVM_CASE(NODE_MIX_VECTOR_NON_UNIFORM) {
          detail::node_mix_vector_non_uniform(cursor, stack);
        };
      }
      $default {
        result.status =
            static_cast<std::uint32_t>(EvaluationStatus::invalid_node);
        active = false;
      };
    };

    $if(!transition_supported) {
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
