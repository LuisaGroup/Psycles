/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_principled_hair.h"

#include "cycles_svm_microfacet.h"
#include "cycles_svm_simple_closure.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace sample_mapping =
    ::psycles::luisa_backend::cycles_sample_mapping;

namespace {

struct PrincipledHairNode {
  UInt parametrization;
  UInt color_x;
  UInt color_y;
  UInt color_z;
  UInt tint_x;
  UInt tint_y;
  UInt tint_z;
  UInt absorption_x;
  UInt absorption_y;
  UInt absorption_z;
  UInt roughness;
  UInt random_roughness;
  UInt offset;
  UInt ior;
  UInt random;
  UInt melanin;
  UInt melanin_redness;
  UInt coat;
  UInt aspect_ratio;
  UInt radial_roughness;
  UInt random_color;
  UInt R;
  UInt TT;
  UInt TRT;
  UInt attr_random;
  UInt attr_normal;
};

[[nodiscard]] PrincipledHairNode
read_principled_hair_node(Cursor &cursor) noexcept {
  /* This is the named, field-for-field projection of the 104-byte payload.
   * Reading it up front preserves svm_node_get<T>() cursor semantics even if
   * the weight is cut off or closure allocation later fails. */
  return {.parametrization = cursor.word(),
          .color_x = cursor.word(),
          .color_y = cursor.word(),
          .color_z = cursor.word(),
          .tint_x = cursor.word(),
          .tint_y = cursor.word(),
          .tint_z = cursor.word(),
          .absorption_x = cursor.word(),
          .absorption_y = cursor.word(),
          .absorption_z = cursor.word(),
          .roughness = cursor.word(),
          .random_roughness = cursor.word(),
          .offset = cursor.word(),
          .ior = cursor.word(),
          .random = cursor.word(),
          .melanin = cursor.word(),
          .melanin_redness = cursor.word(),
          .coat = cursor.word(),
          .aspect_ratio = cursor.word(),
          .radial_roughness = cursor.word(),
          .random_color = cursor.word(),
          .R = cursor.word(),
          .TT = cursor.word(),
          .TRT = cursor.word(),
          .attr_random = cursor.word(),
          .attr_normal = cursor.word()};
}

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] Bool is_zero(Expr<luisa::float3> value) noexcept {
  return (value.x == 0.0f) & (value.y == 0.0f) & (value.z == 0.0f);
}

[[nodiscard]] Bool is_finite(Expr<luisa::float3> value) noexcept {
  return !any(dsl::isnan(value)) & !any(dsl::isinf(value));
}

[[nodiscard]] Float3 to_local(Expr<luisa::float3> value,
                              Expr<luisa::float3> X,
                              Expr<luisa::float3> Y,
                              Expr<luisa::float3> Z) noexcept {
  return make_float3(dot(value, X), dot(value, Y), dot(value, Z));
}

[[nodiscard]] Float pow20(Expr<float> value) noexcept {
  /* Cycles util/math_base.h::pow20; preserve the multiplication tree rather
   * than lowering this integer power through a generic transcendental. */
  return square(square(square(square(value)) * value));
}

[[nodiscard]] Float pow22(Expr<float> value) noexcept {
  /* Cycles util/math_base.h::pow22. */
  return square(value * square(square(square(value)) * value));
}

[[nodiscard]] Float3 sigma_from_node(const PrincipledHairNode &node,
                                     Stack &stack, Expr<float> random,
                                     Expr<float> radial_roughness) noexcept {
  Float3 sigma;
  $switch(node.parametrization) {
    $case(static_cast<std::uint32_t>(NODE_PRINCIPLED_HAIR_DIRECT_ABSORPTION)) {
      /* RGB builds use an identity rgb_to_spectrum projection. */
      sigma = stack_load_input_float3(stack, node.absorption_x,
                                      node.absorption_y, node.absorption_z);
    };
    $case(static_cast<std::uint32_t>(
        NODE_PRINCIPLED_HAIR_PIGMENT_CONCENTRATION)) {
      Float melanin = stack_load_input_float(stack, node.melanin);
      const auto melanin_redness =
          stack_load_input_float(stack, node.melanin_redness);
      const auto random_color =
          clamp(stack_load_input_float(stack, node.random_color), 0.0f, 1.0f);
      const auto factor_random_color =
          1.0f + 2.0f * (random - 0.5f) * random_color;
      melanin *= factor_random_color;
      melanin = -log(max(1.0f - melanin, 0.0001f));
      const auto eumelanin = melanin * (1.0f - melanin_redness);
      const auto pheomelanin = melanin * melanin_redness;
      const auto melanin_sigma =
          bsdf_principled_hair_sigma_from_concentration(eumelanin, pheomelanin);
      const auto tint =
          stack_load_input_float3(stack, node.tint_x, node.tint_y, node.tint_z);
      sigma = melanin_sigma + bsdf_principled_hair_sigma_from_reflectance(
                                  tint, radial_roughness);
    };
    $case(static_cast<std::uint32_t>(NODE_PRINCIPLED_HAIR_REFLECTANCE)) {
      const auto color = stack_load_input_float3(stack, node.color_x,
                                                 node.color_y, node.color_z);
      sigma =
          bsdf_principled_hair_sigma_from_reflectance(color, radial_roughness);
    };
    $default {
      /* Cycles' release fallback for an invalid parametrization. */
      sigma = bsdf_principled_hair_sigma_from_concentration(0.0f, 0.8054375f);
    };
  };
  return sigma;
}

void chiang_setup(ShaderData &shader_data,
                  const ClosurePool::Allocation &allocation,
                  Expr<luisa::float3> sigma, Expr<float> roughness,
                  Expr<float> radial_roughness, Expr<float> coat,
                  Expr<float> alpha, Expr<float> eta) noexcept {
  auto &pool = *shader_data.closure;
  Float v = clamp(roughness, 0.001f, 1.0f);
  Float s = clamp(radial_roughness, 0.001f, 1.0f);
  Float m0_roughness =
      clamp((1.0f - clamp(coat, 0.0f, 1.0f)) * v, 0.001f, 1.0f);

  v = square(0.726f * v + 0.812f * square(v) + 3.700f * pow20(v));
  s = (0.265f * s + 1.194f * square(s) + 5.372f * pow22(s)) *
      0.6266570686577501f;
  m0_roughness = square(0.726f * m0_roughness + 0.812f * square(m0_roughness) +
                        3.700f * pow20(m0_roughness));

  const auto X = safe_normalize_cycles(shader_data.dPdu);
  const auto Y = safe_normalize_cycles(cross(X, shader_data.wi));
  const auto Z = safe_normalize_cycles(cross(X, Y));
  const auto ribbon =
      (shader_data.type & primitive_curve) == primitive_curve_ribbon;
  const auto h =
      select(dot(cross(shader_data.Ng, X), Z), -shader_data.v, ribbon);

  pool.set_type(allocation.index, cycles_closure::type_hair_chiang);
  pool.set_normal(allocation.index, Y);
  pool.set_chiang_hair_param(allocation.index, {.sigma = sigma,
                                                .v = v,
                                                .s = s,
                                                .alpha = -alpha,
                                                .eta = eta,
                                                .m0_roughness = m0_roughness,
                                                .h = h});
  shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval |
                      shader_data_bsdf_has_transmission;
}

void huang_setup(const KernelGlobals &kernel_globals, ShaderData &shader_data,
                 const PathState &path_state,
                 const ClosurePool::Allocation &allocation,
                 Expr<luisa::float3> sigma, Expr<float> roughness,
                 Expr<float> alpha, Expr<float> eta, Expr<float> aspect_ratio,
                 Expr<float> R, Expr<float> TT, Expr<float> TRT,
                 Expr<std::uint32_t> normal_attribute) noexcept {
  auto &pool = *shader_data.closure;

  Float pixel_coverage = 1.0f;
  const Bool camera_curve =
      ((path_state.visibility & path_ray_visibility_camera) != 0u) &
      ((shader_data.type & primitive_curve) != 0u);
  $if(camera_curve) {
    const auto curve = kernel_globals.curve(shader_data.prim);
    const Int k0 =
        curve.first_key +
        (shader_data.type >> primitive_num_bits).cast<std::int32_t>();
    const Int position_offset =
        kernel_globals.object_position_offset(shader_data.object);
    const auto radius0 = kernel_globals.curve_key(position_offset + k0).w;
    const auto radius1 = kernel_globals.curve_key(position_offset + k0 + 1).w;
    const auto radius = radius0 + shader_data.u * (radius1 - radius0);
    pixel_coverage = 0.5f * shader_data.dP / radius;
  };

  Float3 N = shader_data.N;
  $if(aspect_ratio != 1.0f) {
    const auto normal_descriptor = find_attribute(
        kernel_globals, shader_data, normal_attribute.cast<luisa::ulong>());
    N = primitive_surface_attribute_float3(kernel_globals, shader_data,
                                           normal_descriptor);
  };

  const auto setup_roughness = clamp(roughness, 0.001f, 1.0f);
  Float setup_aspect_ratio = aspect_ratio;
  const auto tilt = -alpha;
  const auto Y = safe_normalize_cycles(shader_data.dPdu);
  const auto X = safe_normalize_cycles(cross(shader_data.dPdu, shader_data.wi));
  const auto ribbon =
      (shader_data.type & primitive_curve) == primitive_curve_ribbon;
  const auto h = select(-dot(X, shader_data.N), -shader_data.v, ribbon);

  $if((setup_aspect_ratio != 1.0f) &
      ((shader_data.type & primitive_curve) != 0u)) {
    N = safe_normalize_cycles(
        cross(shader_data.dPdu,
              safe_normalize_cycles(cross(N, shader_data.dPdu))));
    $if(setup_aspect_ratio > 1.0f) {
      setup_aspect_ratio = 1.0f / setup_aspect_ratio;
      const auto minor_axis =
          safe_normalize_cycles(cross(shader_data.dPdu, N));
      N = safe_normalize_cycles(cross(minor_axis, shader_data.dPdu));
    };
  }
  $else { N = X; };

  Float3 Z;
  $if(is_zero(N) | !is_finite(N)) {
    const auto basis = sample_mapping::make_orthonormals(Y);
    Z = basis.tangent;
    N = basis.bitangent;
  }
  $else { Z = safe_normalize_cycles(cross(N, shader_data.dPdu)); };

  const auto wi = to_local(shader_data.wi, N, Y, Z);
  const auto e2 = 1.0f - square(setup_aspect_ratio);
  Float radius;
  $if(e2 == 0.0f) { radius = 1.0f; }
  $else {
    radius = sqrt(1.0f - e2 * square(wi.x) /
                             (square(wi.x) + square(wi.z)));
  };

  $if(abs(h) >= radius) {
    pool.rollback_with_extra(allocation, 1u);
    transparent_setup(shader_data, path_state,
                      pool.common(allocation.index).weight);
  }
  $else {
    pool.set_type(allocation.index, cycles_closure::type_hair_huang);
    pool.set_normal(allocation.index, N);
    pool.set_huang_hair(
        allocation.index,
        {.sigma = sigma,
         .roughness = setup_roughness,
         .tilt = tilt,
         .eta = eta,
         .aspect_ratio = setup_aspect_ratio,
         .h = h},
        {.R = max(0.0f, R),
         .TT = max(0.0f, TT),
         .TRT = max(0.0f, TRT),
         .Y = Y,
         .Z = Z,
         .wi = wi,
         .radius = radius,
         .e2 = e2,
         .pixel_coverage = pixel_coverage});
    shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval |
                        shader_data_bsdf_has_transmission;
  };
}

} // namespace

Float3 bsdf_principled_hair_sigma_from_reflectance(
    Expr<luisa::float3> color, Expr<float> azimuthal_roughness) noexcept {
  const auto x = azimuthal_roughness;
  const auto scale =
      (((((0.245f * x) + 5.574f) * x - 10.73f) * x + 2.532f) * x - 0.215f) * x +
      5.969f;
  const auto sigma = log(max(color, make_float3(0.0f))) / scale;
  return sigma * sigma;
}

Float3 bsdf_principled_hair_sigma_from_concentration(
    Expr<float> eumelanin, Expr<float> pheomelanin) noexcept {
  return eumelanin * make_float3(0.506f, 0.841f, 1.653f) +
         pheomelanin * make_float3(0.343f, 0.733f, 1.924f);
}

void node_principled_hair(const KernelGlobals &kernel_globals, Cursor &cursor,
                          Stack &stack, Expr<std::uint32_t> closure_type,
                          Expr<luisa::float3> closure_weight,
                          Expr<float> mix_weight, ShaderData &shader_data,
                          const PathState &path_state) noexcept {
  const auto node = read_principled_hair_node(cursor);
  const auto weight = closure_weight * mix_weight;
  const auto alpha = stack_load_input_float(stack, node.offset);
  const auto ior = stack_load_input_float(stack, node.ior);

  const auto random_descriptor = find_attribute(
      kernel_globals, shader_data, node.attr_random.cast<luisa::ulong>());
  Float random;
  $if(random_descriptor.offset !=
      static_cast<std::int32_t>(ATTR_STD_NOT_FOUND)) {
    random = primitive_surface_attribute_float(kernel_globals, shader_data,
                                               random_descriptor);
  }
  $else { random = stack_load_input_float(stack, node.random); };

  const auto random_roughness =
      stack_load_input_float(stack, node.random_roughness);
  const auto factor_random_roughness =
      1.0f + 2.0f * (random - 0.5f) * random_roughness;
  const auto roughness =
      stack_load_input_float(stack, node.roughness) * factor_random_roughness;
  Float radial_roughness = roughness;
  const Bool is_chiang =
      closure_type == static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_CHIANG_ID);
  $if(is_chiang) {
    radial_roughness = stack_load_input_float(stack, node.radial_roughness) *
                       factor_random_roughness;
  };
  const auto sigma = sigma_from_node(node, stack, random, radial_roughness);

  $if(is_chiang) {
    const auto allocation = bsdf_allocate(shader_data, weight);
    $if(allocation.valid) {
      chiang_setup(shader_data, allocation, sigma, roughness, radial_roughness,
                   stack_load_input_float(stack, node.coat), alpha, ior);
    };
  }
  $else {
    const auto R = stack_load_input_float(stack, node.R);
    const auto TT = stack_load_input_float(stack, node.TT);
    const auto TRT = stack_load_input_float(stack, node.TRT);
    $if(!((R <= 0.0f) & (TT <= 0.0f) & (TRT <= 0.0f))) {
      const auto allocation = bsdf_allocate(shader_data, weight);
      const auto extra_allocated =
          shader_data.closure->allocate_extra(
              allocation, 1u, ClosurePool::ExtraPayload::huang_hair);
      $if(extra_allocated) {
        huang_setup(
            kernel_globals, shader_data, path_state, allocation, sigma,
            roughness, alpha, ior,
            stack_load_input_float(stack, node.aspect_ratio), R, TT, TRT,
            node.attr_normal);
      };
    };
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail
