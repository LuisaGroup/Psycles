/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_object_info(Cursor &cursor, Stack &stack,
                      const InfoServices &services,
                      const ShaderData &shader_data) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const auto emit_one = [&](std::uint32_t t) noexcept {
    switch (t) {
      case NODE_INFO_OB_LOCATION:
        stack_store_float3(
            stack, output_offset, services.object_location(shader_data));
        break;
      case NODE_INFO_OB_COLOR:
        stack_store_float3(
            stack, output_offset, services.object_color(shader_data.object));
        break;
      case NODE_INFO_OB_ALPHA:
        stack_store_float(
            stack, output_offset, services.object_alpha(shader_data.object));
        break;
      case NODE_INFO_OB_INDEX:
        stack_store_float(
            stack, output_offset, services.object_pass_id(shader_data.object));
        break;
      case NODE_INFO_MAT_INDEX:
        stack_store_float(
            stack, output_offset, services.shader_pass_id(shader_data));
        break;
      case NODE_INFO_OB_RANDOM:
        stack_store_float(
            stack, output_offset,
            services.object_random_number(shader_data.object));
        break;
      default:
        stack_store_float(stack, output_offset, 0.0f);
        break;
    }
  };
  if (const auto ct = try_const_uint(info_type)) {
    emit_one(*ct);
    return;
  }

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_OB_LOCATION) { emit_one(NODE_INFO_OB_LOCATION); };
    PSYCLES_SVM_CASE(NODE_INFO_OB_COLOR) { emit_one(NODE_INFO_OB_COLOR); };
    PSYCLES_SVM_CASE(NODE_INFO_OB_ALPHA) { emit_one(NODE_INFO_OB_ALPHA); };
    PSYCLES_SVM_CASE(NODE_INFO_OB_INDEX) { emit_one(NODE_INFO_OB_INDEX); };
    PSYCLES_SVM_CASE(NODE_INFO_MAT_INDEX) { emit_one(NODE_INFO_MAT_INDEX); };
    PSYCLES_SVM_CASE(NODE_INFO_OB_RANDOM) { emit_one(NODE_INFO_OB_RANDOM); };
    $default { stack_store_float(stack, output_offset, 0.0f); };
  };
}

void node_particle_info(Cursor &cursor, Stack &stack,
                        const InfoServices &services,
                        const ShaderData &shader_data) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const auto emit_one = [&](std::uint32_t t) noexcept {
    switch (t) {
      case NODE_INFO_PAR_INDEX: {
        const auto particle = services.object_particle_id(shader_data.object);
        stack_store_float(
            stack, output_offset,
            services.particle_index(particle).cast<float>());
        break;
      }
      case NODE_INFO_PAR_RANDOM: {
        const auto particle = services.object_particle_id(shader_data.object);
        const auto index = services.particle_index(particle);
        stack_store_float(
            stack, output_offset,
            cycles_noise::uint_to_float_inclusive(
                cycles_noise::hash_uint2(index, 0u)));
        break;
      }
      case NODE_INFO_PAR_AGE: {
        const auto particle = services.object_particle_id(shader_data.object);
        stack_store_float(
            stack, output_offset, services.particle_age(particle));
        break;
      }
      case NODE_INFO_PAR_LIFETIME: {
        const auto particle = services.object_particle_id(shader_data.object);
        stack_store_float(
            stack, output_offset, services.particle_lifetime(particle));
        break;
      }
      case NODE_INFO_PAR_LOCATION: {
        const auto particle = services.object_particle_id(shader_data.object);
        stack_store_float3(
            stack, output_offset, services.particle_location(particle));
        break;
      }
      case NODE_INFO_PAR_SIZE: {
        const auto particle = services.object_particle_id(shader_data.object);
        stack_store_float(
            stack, output_offset, services.particle_size(particle));
        break;
      }
      case NODE_INFO_PAR_VELOCITY: {
        const auto particle = services.object_particle_id(shader_data.object);
        stack_store_float3(
            stack, output_offset, services.particle_velocity(particle));
        break;
      }
      case NODE_INFO_PAR_ANGULAR_VELOCITY: {
        const auto particle = services.object_particle_id(shader_data.object);
        stack_store_float3(
            stack, output_offset,
            services.particle_angular_velocity(particle));
        break;
      }
      default:
        break;
    }
  };
  if (const auto ct = try_const_uint(info_type)) {
    emit_one(*ct);
    return;
  }

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_PAR_INDEX) { emit_one(NODE_INFO_PAR_INDEX); };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_RANDOM) { emit_one(NODE_INFO_PAR_RANDOM); };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_AGE) { emit_one(NODE_INFO_PAR_AGE); };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_LIFETIME) {
      emit_one(NODE_INFO_PAR_LIFETIME);
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_LOCATION) {
      emit_one(NODE_INFO_PAR_LOCATION);
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_SIZE) { emit_one(NODE_INFO_PAR_SIZE); };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_VELOCITY) {
      emit_one(NODE_INFO_PAR_VELOCITY);
    };
    PSYCLES_SVM_CASE(NODE_INFO_PAR_ANGULAR_VELOCITY) {
      emit_one(NODE_INFO_PAR_ANGULAR_VELOCITY);
    };
    $default {};
  };
}

void node_hair_info(Cursor &cursor, Stack &stack,
                    const InfoServices *services,
                    const ShaderData &shader_data,
                    const EvaluationTransition &transition) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const auto emit_one = [&](std::uint32_t t) noexcept {
    switch (t) {
      case NODE_INFO_CURVE_IS_STRAND:
        stack_store_float(
            stack, output_offset,
            select(0.0f, 1.0f,
                   (shader_data.type & primitive_curve) != 0u));
        break;
      case NODE_INFO_CURVE_INTERCEPT:
      case NODE_INFO_CURVE_LENGTH:
      case NODE_INFO_CURVE_RANDOM:
        break;
      case NODE_INFO_CURVE_THICKNESS:
        if (services != nullptr) {
          stack_store_float(
              stack, output_offset, services->curve_thickness(shader_data));
        } else {
          transition.unsupported();
        }
        break;
      case NODE_INFO_CURVE_TANGENT_NORMAL: {
        Float3 tangent_normal = make_float3(0.0f);
        $if ((shader_data.type & primitive_curve) != 0u) {
          const auto incoming = -shader_data.wi;
          tangent_normal =
              -(incoming - shader_data.dPdu *
                               (dot(shader_data.dPdu, incoming) /
                                dot(shader_data.dPdu, shader_data.dPdu)));
          tangent_normal = normalize_cycles(tangent_normal);
        };
        stack_store_float3(stack, output_offset, tangent_normal);
        break;
      }
      default:
        break;
    }
  };
  if (const auto ct = try_const_uint(info_type)) {
    emit_one(*ct);
    return;
  }

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_IS_STRAND) {
      emit_one(NODE_INFO_CURVE_IS_STRAND);
    };
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_INTERCEPT) {};
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_LENGTH) {};
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_RANDOM) {};
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_THICKNESS) {
      emit_one(NODE_INFO_CURVE_THICKNESS);
    };
    PSYCLES_SVM_CASE(NODE_INFO_CURVE_TANGENT_NORMAL) {
      emit_one(NODE_INFO_CURVE_TANGENT_NORMAL);
    };
    $default {};
  };
}

void node_point_info(Cursor &cursor, Stack &stack,
                     const InfoServices &services,
                     const ShaderData &shader_data) noexcept {
  const auto info_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const auto emit_one = [&](std::uint32_t t) noexcept {
    switch (t) {
      case NODE_INFO_POINT_POSITION:
        stack_store_float3(
            stack, output_offset, services.point_position(shader_data));
        break;
      case NODE_INFO_POINT_RADIUS:
        stack_store_float(
            stack, output_offset, services.point_radius(shader_data));
        break;
      case NODE_INFO_POINT_RANDOM:
      default:
        break;
    }
  };
  if (const auto ct = try_const_uint(info_type)) {
    emit_one(*ct);
    return;
  }

  $switch (info_type) {
    PSYCLES_SVM_CASE(NODE_INFO_POINT_POSITION) {
      emit_one(NODE_INFO_POINT_POSITION);
    };
    PSYCLES_SVM_CASE(NODE_INFO_POINT_RADIUS) {
      emit_one(NODE_INFO_POINT_RADIUS);
    };
    PSYCLES_SVM_CASE(NODE_INFO_POINT_RANDOM) {};
    $default {};
  };
}

void node_ambient_occlusion(Cursor &cursor, Stack &stack,
                            const AmbientOcclusionServices &services,
                            ShaderData &shader_data) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto dist_word = cursor.word();
  const auto packed4 = cursor.word();
  const auto packed5 = cursor.word();

  const auto flags = cursor.byte(packed4, 0u);
  const auto samples = cursor.byte(packed4, 1u);
  const auto normal_offset = cursor.byte(packed4, 2u);
  const auto out_ao_offset = cursor.byte(packed4, 3u);
  const auto out_color_offset = cursor.byte(packed5, 0u);

  Float max_dist = stack_load_input_float(stack, dist_word);
  $if((flags & static_cast<std::uint32_t>(NODE_AO_GLOBAL_RADIUS)) != 0u) {
    max_dist = services.global_ao_distance(shader_data);
  };

  const auto raw_normal =
      stack_load_float3_default(stack, normal_offset, shader_data.N);
  const auto n_len2 = dot(raw_normal, raw_normal);
  Float3 normal = select(shader_data.N, raw_normal * rsqrt(n_len2), n_len2 > 0.0f);
  normal = select(normal, -normal,
                  (flags & static_cast<std::uint32_t>(NODE_AO_INSIDE)) != 0u);
  const Bool only_local =
      (flags & static_cast<std::uint32_t>(NODE_AO_ONLY_LOCAL)) != 0u;

  const auto lcg_step_float = [](UInt &state) noexcept {
    state = 1103515245u * state + 12345u;
    return state.cast<float>() * (1.0f / 4294967295.0f);
  };

  Float ao = 1.0f;
  $if((samples > 0u) & (max_dist > 0.0f)) {
    const auto inv_samples = 1.0f / samples.cast<float>();
    UInt unoccluded = 0u;
    UInt sample = 0u;
    $while(sample < samples) {
      const auto u = lcg_step_float(shader_data.lcg_state);
      const auto v = lcg_step_float(shader_data.lcg_state);
      const auto randu = (sample.cast<float>() + u) * inv_samples;
      const auto dir = cycles_sample_mapping::sample_cosine_hemisphere(
                           normal, make_float2(randu, v))
                           .direction;
      const auto ray = make_ray(shader_data.P, dir, 0.0f, max_dist);
      $if(!services.ambient_occluded(shader_data, ray, only_local)) {
        unoccluded += 1u;
      };
      sample += 1u;
    };
    ao = unoccluded.cast<float>() * inv_samples;
  };

  stack_store_float_if_valid(stack, out_ao_offset, ao);
  if (const auto out_color = try_const_uint(out_color_offset);
      !out_color.has_value() || *out_color != SVM_STACK_INVALID) {
    $if(out_color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      const auto color =
          stack_load_input_float3(stack, color_x, color_y, color_z);
      stack_store_float3(stack, out_color_offset, color * ao);
    };
  }
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE
