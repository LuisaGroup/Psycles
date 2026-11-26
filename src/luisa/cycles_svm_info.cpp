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

namespace {

[[nodiscard]] Float bevel_cubic_pdf(Expr<float> radius,
                                    Expr<float> r) noexcept {
  const auto rm5 = (radius * radius) * (radius * radius) * radius;
  const auto f = radius - r;
  const auto num = f * f * f;
  return select(0.0f, (10.0f * num) / (rm5 * cycles_sample_mapping::pi),
                (r < radius) & (rm5 > 0.0f));
}

[[nodiscard]] Float bevel_cubic_quintic_root_find(Expr<float> xi) noexcept {
  constexpr float tolerance = 1.0e-6f;
  Float x = 0.25f;
  UInt iteration = 0u;
  $while(iteration < 10u) {
    const auto x2 = x * x;
    const auto x3 = x2 * x;
    const auto nx = 1.0f - x;
    const auto f =
        10.0f * x2 - 20.0f * x3 + 15.0f * x2 * x2 - 4.0f * x2 * x3 - xi;
    const auto df = 20.0f * (x * nx) * (nx * nx);
    $if((abs(f) < tolerance) | (df == 0.0f)) { $break; };
    x = clamp(x - f / df, 0.0f, 1.0f);
    iteration += 1u;
  };
  return x;
}

} // namespace

void node_bevel(Cursor &cursor, Stack &stack,
                const KernelGlobals &kernel_globals,
                const TransformState &transform_state,
                const AmbientOcclusionServices &services,
                const ShaderData &shader_data,
                bool object_motion_enabled) noexcept {
  const auto radius_word = cursor.word();
  const auto packed = cursor.word();

  const auto num_samples = cursor.byte(packed, 0u);
  const auto normal_offset = cursor.byte(packed, 1u);
  const auto out_offset = cursor.byte(packed, 2u);

  const auto radius = stack_load_input_float(stack, radius_word);
  Float3 bevel_N = shader_data.N;

  $if((radius > 0.0f) & (num_samples >= 1u) &
      (shader_data.object != object_none) &
      ((shader_data.type & primitive_triangle) != 0u) &
      (services.bevel_min_ray_pdf() >= 8.0f)) {
    UInt lcg_state = services.bevel_lcg_state_init();
    Float3 sum_N = make_float3(0.0f);

    const auto object_flag = shader_data.object_flag;
    const auto transform_applied =
        (object_flag & shader_data_object_transform_applied) != 0u;
    const auto negative_scale_applied =
        transform_applied &
        ((object_flag &
          static_cast<std::uint32_t>(
              compiler::cycles_svm::SD_OBJECT_NEGATIVE_SCALE)) != 0u);
    constexpr float two_pi = 2.0f * cycles_sample_mapping::pi;

    UInt sample = 0u;
    $while(sample < num_samples) {
      const auto rand_disk = services.bevel_sample_disk(sample, num_samples);

      Float3 disk_N = shader_data.Ng;
      const auto basis = cycles_sample_mapping::make_orthonormals(disk_N);
      Float3 disk_T = basis.tangent;
      Float3 disk_B = basis.bitangent;
      Float pick_pdf_N = 0.5f;
      Float pick_pdf_T = 0.25f;
      Float pick_pdf_B = 0.25f;
      Float disk_u = rand_disk.x;

      $if(disk_u < 0.5f) {
        disk_u *= 2.0f;
      }
      $elif(disk_u < 0.75f) {
        const auto tmp = disk_N;
        disk_N = disk_T;
        disk_T = tmp;
        pick_pdf_N = 0.25f;
        pick_pdf_T = 0.5f;
        pick_pdf_B = 0.25f;
        disk_u = (disk_u - 0.5f) * 4.0f;
      }
      $else {
        const auto tmp = disk_N;
        disk_N = disk_B;
        disk_B = tmp;
        pick_pdf_N = 0.25f;
        pick_pdf_T = 0.25f;
        pick_pdf_B = 0.5f;
        disk_u = (disk_u - 0.75f) * 4.0f;
      };

      const auto phi = two_pi * disk_u;
      const auto disk_r = bevel_cubic_quintic_root_find(rand_disk.y) * radius;
      const auto disk_height = sqrt(max(radius * radius - disk_r * disk_r, 0.0f));
      const auto disk_P =
          disk_T * (disk_r * cos(phi)) + disk_B * (disk_r * sin(phi));

      const auto ray =
          make_ray(shader_data.P + disk_N * disk_height + disk_P, -disk_N,
                   0.0f, 2.0f * disk_height);
      const auto reservoir =
          services.trace_bevel_local(shader_data, ray, lcg_state);
      const auto num_hits = reservoir.num_hits;
      const auto num_eval_hits = min(num_hits, 4u);
      const auto disk_pdf = bevel_cubic_pdf(radius, disk_r);

      UInt hit_index = 0u;
      $while(hit_index < num_eval_hits) {
        UInt hit_prim = reservoir.prim[0u];
        Float hit_u = reservoir.u[0u];
        Float hit_v = reservoir.v[0u];
        for (std::size_t slot = 1u; slot < 4u; ++slot) {
          $if(hit_index == static_cast<std::uint32_t>(slot)) {
            hit_prim = reservoir.prim[slot];
            hit_u = reservoir.u[slot];
            hit_v = reservoir.v[slot];
          };
        }

        TriangleVertices hit_verts;
        if (object_motion_enabled) {
          $if((shader_data.type & primitive_motion) != 0u) {
            hit_verts = kernel_globals.motion_triangle_vertices(
                shader_data.object, hit_prim, shader_data.time);
          }
          $else {
            hit_verts =
                kernel_globals.triangle_vertices(shader_data.object, hit_prim);
          };
        } else {
          hit_verts =
              kernel_globals.triangle_vertices(shader_data.object, hit_prim);
        }

        const auto dPdu = hit_verts.v1 - hit_verts.v0;
        const auto dPdv = hit_verts.v2 - hit_verts.v0;
        Float3 hit_P = hit_verts.v0 + hit_u * dPdu + hit_v * dPdv;
        Float3 hit_Ng = normalize_cycles(
            select(cross(dPdu, dPdv), cross(dPdv, dPdu), negative_scale_applied));

        Float3 N = hit_Ng;
        const auto hit_shader = kernel_globals.triangle_shader(hit_prim);
        $if((hit_shader & shader_smooth_normal) != 0u) {
          ShaderData hit_sd = shader_data;
          hit_sd.prim = hit_prim;
          hit_sd.u = hit_u;
          hit_sd.v = hit_v;
          hit_sd.shader = hit_shader;
          const auto normals = triangle_normals(kernel_globals, hit_sd);
          const auto smooth = safe_normalize_cycles(
              (1.0f - hit_u - hit_v) * normals.n0 + hit_u * normals.n1 +
              hit_v * normals.n2);
          N = select(smooth, hit_Ng, all(smooth == 0.0f));
        };

        $if(!transform_applied) {
          object_position_transform(hit_P, transform_state, shader_data,
                                    object_motion_enabled);
          object_normal_transform(N, transform_state, shader_data,
                                  object_motion_enabled);
          object_normal_transform(hit_Ng, transform_state, shader_data,
                                  object_motion_enabled);
        };

        const auto pdf_N = pick_pdf_N * abs(dot(disk_N, hit_Ng));
        const auto pdf_T = pick_pdf_T * abs(dot(disk_T, hit_Ng));
        const auto pdf_B = pick_pdf_B * abs(dot(disk_B, hit_Ng));
        const auto denom = pdf_N * pdf_N + pdf_T * pdf_T + pdf_B * pdf_B;
        Float w = select(0.0f, pdf_N / denom, denom > 0.0f);
        $if(num_hits > 4u) { w *= num_hits.cast<float>() * 0.25f; };
        const auto r = length(hit_P - shader_data.P);
        const auto pdf = bevel_cubic_pdf(radius, r);
        w *= select(0.0f, pdf / disk_pdf, disk_pdf > 0.0f);
        sum_N += w * N;
        hit_index += 1u;
      };
      sample += 1u;
    };

    const auto norm_sum = safe_normalize_cycles(sum_N);
    const auto has_bevel = any(norm_sum != 0.0f);
    const auto oriented =
        select(norm_sum, -norm_sum,
               (shader_data.flag & shader_data_backfacing) != 0u);
    bevel_N = select(shader_data.N, oriented, has_bevel);
  };

  if (const auto normal_slot = try_const_uint(normal_offset);
      !normal_slot.has_value() || *normal_slot != SVM_STACK_INVALID) {
    $if(normal_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      const auto ref_N = stack_load_float3(stack, normal_offset);
      const auto shifted =
          safe_normalize_cycles(ref_N + (bevel_N - shader_data.N));
      bevel_N = select(shifted, shader_data.N, all(shifted == 0.0f));
    };
  }

  stack_store_float3_if_valid(stack, out_offset, bevel_N);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE
