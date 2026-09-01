/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_microfacet.h"

#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace table_detail = ::psycles::luisa_backend::detail;

namespace {

inline constexpr float two_pi = 6.28318530717958647692f;

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
    return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] Float safe_sqrt(Expr<float> value) noexcept {
    return sqrt(max(value, 0.0f));
}

[[nodiscard]] Float f0_from_ior(Expr<float> ior) noexcept {
    const auto r = (ior - 1.0f) / (ior + 1.0f);
    return r * r;
}

[[nodiscard]] Float fresnel_dielectric_fss(Expr<float> ior) noexcept {
    Float result;
    $if (ior < 1.0f) {
        result = 0.997118f + ior * (0.1014f - ior * (0.965241f + ior * 0.130607f));
    }
    $else { result = (ior - 1.0f) / (4.08567f + 1.00071f * ior); };
    return result;
}

[[nodiscard]] Float3 rotate_around_axis(Expr<luisa::float3> point,
                                        Expr<luisa::float3> axis,
                                        Expr<float> angle) noexcept {
    const auto cosine = cos(angle);
    const auto sine = sin(angle);
    const auto one_minus_cosine = 1.0f - cosine;
    Float3 result;
    result.x =
        (cosine + one_minus_cosine * axis.x * axis.x) * point.x +
        (one_minus_cosine * axis.x * axis.y - axis.z * sine) * point.y +
        (one_minus_cosine * axis.x * axis.z + axis.y * sine) * point.z;
    result.y =
        (one_minus_cosine * axis.x * axis.y + axis.z * sine) * point.x +
        (cosine + one_minus_cosine * axis.y * axis.y) * point.y +
        (one_minus_cosine * axis.y * axis.z - axis.x * sine) * point.z;
    result.z =
        (one_minus_cosine * axis.x * axis.z - axis.y * sine) * point.x +
        (one_minus_cosine * axis.y * axis.z + axis.x * sine) * point.y +
        (cosine + one_minus_cosine * axis.z * axis.z) * point.z;
    return result;
}

[[nodiscard]] Float3
ensure_valid_specular_reflection(Expr<luisa::float3> geometric_normal,
                                 Expr<luisa::float3> incident,
                                 Expr<luisa::float3> normal) noexcept {
    Float3 result = normal;
    const auto reflection = 2.0f * dot(normal, incident) * normal - incident;
    const auto incident_z = dot(incident, geometric_normal);
    const auto threshold = min(0.9f * incident_z, 0.01f);
    $if (dot(geometric_normal, reflection) < threshold) {
        const auto x_axis = native_vector_math::safe_normalize_nonzero_or(
            normal - dot(normal, geometric_normal) * geometric_normal, normal);
        const auto incident_x = dot(incident, x_axis);
        const auto a = square(incident_x) + square(incident_z);
        const auto b = 2.0f * (a + incident_z * threshold);
        const auto c = square(threshold + incident_z);
        const auto root = safe_sqrt(square(b) - 4.0f * a * c);
        Float normal_z_squared;
        $if (incident_x < 0.0f) { normal_z_squared = 0.25f * (b + root) / a; }
        $else { normal_z_squared = 0.25f * (b - root) / a; };
        const auto normal_x = safe_sqrt(1.0f - normal_z_squared);
        const auto normal_z = safe_sqrt(normal_z_squared);
        result = normal_x * x_axis + normal_z * geometric_normal;
    };
    return result;
}

[[nodiscard]] Float3 generalized_schlick_albedo(
    const KernelGlobals &kernel_globals, Expr<luisa::float3> incoming,
    Expr<luisa::float3> normal, const MicrofacetParam &microfacet,
    const FresnelGeneralizedSchlick &fresnel) noexcept {
    const auto cosine_incoming = dot(incoming, normal);
    Float3 reflectance;
    $if (fresnel.thin_film.thickness > table_detail::thin_film_thickness_cutoff) {
        reflectance =
            table_detail::thin_film_dielectric_fresnel(
                kernel_globals, fresnel.thin_film.thickness, fresnel.thin_film.ior,
                microfacet.ior, fresnel.f0, cosine_incoming)
                .reflectance;
    }
    $else {
        const auto table_roughness =
            sqrt(sqrt(microfacet.alpha_x * microfacet.alpha_y));
        const auto z = sqrt(abs((microfacet.ior - 1.0f) / (microfacet.ior + 1.0f)));
        const auto interpolation = table_detail::cycles_table_3d(
            kernel_globals, table_roughness, cosine_incoming, z,
            UInt{cycles45_tables::ggx_gen_schlick_ior_s_offset}, 16u, 16u, 16u);
        reflectance = lerp(fresnel.f0, fresnel.f90, interpolation);
    };
    return reflectance * fresnel.reflection_tint +
           (make_float3(1.0f) - reflectance) * fresnel.transmission_tint;
}

void apply_multi_ggx_energy(ClosurePool &pool,
                            const ClosurePool::Allocation &allocation,
                            MicrofacetParam &microfacet, Expr<float> energy,
                            Expr<float> average_energy,
                            Expr<luisa::float3> average_fresnel) noexcept {
    const auto missing_factor = (1.0f - energy) / energy;
    microfacet.energy_scale = 1.0f + missing_factor;
    $if (!all(average_fresnel == make_float3(1.0f))) {
        const auto multiple_scatter_fresnel =
            average_fresnel * average_energy /
            (make_float3(1.0f) - average_fresnel * (1.0f - average_energy));
        const auto darkening =
            (make_float3(1.0f) + multiple_scatter_fresnel * missing_factor) /
            microfacet.energy_scale;
        const auto common = pool.common(allocation.index);
        pool.set_weight(allocation.index, common.weight * darkening);
        pool.set_sample_weight(allocation.index,
                               common.sample_weight * average(darkening));
    };
}

void preserve_multi_ggx_glass_energy(
    const KernelGlobals &kernel_globals, ClosurePool &pool,
    const ClosurePool::Allocation &allocation, Expr<luisa::float3> incoming,
    Expr<luisa::float3> normal, MicrofacetParam &microfacet,
    const FresnelGeneralizedSchlick &fresnel) noexcept {
    const auto mu = dot(incoming, normal);
    const auto table_roughness =
        sqrt(sqrt(microfacet.alpha_x * microfacet.alpha_y));
    Float lookup_ior = microfacet.ior;
    UInt energy_offset = cycles45_tables::ggx_glass_e_offset;
    UInt average_offset = cycles45_tables::ggx_glass_eavg_offset;
    $if (lookup_ior < 1.0f) {
        lookup_ior = 1.0f / lookup_ior;
        energy_offset = cycles45_tables::ggx_glass_inv_e_offset;
        average_offset = cycles45_tables::ggx_glass_inv_eavg_offset;
    };
    const auto z = sqrt(abs((lookup_ior - 1.0f) / (lookup_ior + 1.0f)));
    const auto energy = table_detail::cycles_table_3d(
        kernel_globals, table_roughness, mu, z, energy_offset, 16u, 16u, 16u);
    const auto average_energy = table_detail::cycles_table_2d(
        kernel_globals, table_roughness, z, average_offset, 16u, 16u);

    Float3 average_fresnel = make_float3(1.0f);
    $if (all(fresnel.transmission_tint == make_float3(0.0f))) {
        Float interpolation;
        $if (fresnel.exponent < 0.0f) {
            const auto real_f0 = f0_from_ior(microfacet.ior);
            const auto real_fss = fresnel_dielectric_fss(microfacet.ior);
            interpolation =
                clamp((real_fss - real_f0) / (1.0f - real_f0), 0.0f, 1.0f);
        }
        $else {
            interpolation =
                2.0f / ((fresnel.exponent + 3.0f) * fresnel.exponent + 2.0f);
        };
        average_fresnel =
            fresnel.reflection_tint * lerp(fresnel.f0, fresnel.f90, interpolation);
    }
    $else { average_fresnel = fresnel.transmission_tint; };

    apply_multi_ggx_energy(pool, allocation, microfacet, energy,
                           average_energy, average_fresnel);
}

void preserve_multi_ggx_reflection_energy(
    const KernelGlobals &kernel_globals, ClosurePool &pool,
    const ClosurePool::Allocation &allocation, Expr<luisa::float3> incoming,
    Expr<luisa::float3> normal, MicrofacetParam &microfacet,
    Expr<luisa::float3> color) noexcept {
    const auto mu = dot(incoming, normal);
    const auto table_roughness =
        sqrt(sqrt(microfacet.alpha_x * microfacet.alpha_y));
    const auto energy = table_detail::cycles_table_2d(
        kernel_globals, table_roughness, mu,
        UInt{cycles45_tables::ggx_e_offset}, 32u, 32u);
    const auto average_energy = table_detail::cycles_table_1d(
        kernel_globals, table_roughness,
        UInt{cycles45_tables::ggx_eavg_offset}, 32u);
    apply_multi_ggx_energy(pool, allocation, microfacet, energy,
                           average_energy, color);
}

}// namespace

ClosurePool::Allocation
bsdf_allocate(ShaderData &shader_data,
              Expr<luisa::float3> input_weight) noexcept {
    auto &pool = *shader_data.closure;
    const Float3 weight = max(input_weight, make_float3(0.0f));
    const Float sample_weight = abs(average(weight));
    const Bool survives_cutoff =
        (sample_weight >= CLOSURE_WEIGHT_CUTOFF) |
        ((shader_data.flag & shader_data_is_volume_shader_eval) != 0u);
    ClosurePool::Allocation result{.index = 0u, .valid = false};
    $if (survives_cutoff & (sample_weight > 0.0f)) {
        const auto allocated =
            pool.allocate(static_cast<std::uint32_t>(CLOSURE_NONE_ID), weight);
        result.index = allocated.index;
        result.valid = allocated.valid;
        $if (allocated.valid) {
            pool.set_sample_weight(allocated.index, sample_weight);
        };
    };
    return result;
}

Float3
maybe_ensure_valid_specular_reflection(const ShaderData &shader_data,
                                       Expr<luisa::float3> normal) noexcept {
    Float3 result = normal;
    $if ((shader_data.flag & shader_data_use_bump_map_correction) != 0u) {
        const auto is_curve =
            (shader_data.type & static_cast<std::uint32_t>(PRIMITIVE_CURVE)) != 0u;
        $if (!is_curve & !all(shader_data.Ng == normal)) {
            result = ensure_valid_specular_reflection(shader_data.Ng, shader_data.wi,
                                                      normal);
        };
    };
    return result;
}

void glass_setup(const KernelGlobals &kernel_globals, ShaderData &shader_data,
                 const PathState &path_state, Expr<std::uint32_t> input_type,
                 Expr<float> mix_weight, Expr<luisa::float3> normal,
                 Expr<luisa::float3> color, Expr<float> roughness,
                 Expr<float> ior, Expr<float> thin_film_thickness,
                 Expr<float> thin_film_ior) noexcept {
    const auto diffuse_visibility =
        (path_state.visibility & path_ray_visibility_diffuse) != 0u;
    const Bool reflective_caustics =
        kernel_globals.caustics_reflective() | !diffuse_visibility;
    const Bool refractive_caustics =
        kernel_globals.caustics_refractive() | !diffuse_visibility;

    $if (reflective_caustics | refractive_caustics) {
        auto &pool = *shader_data.closure;
        const auto allocated = bsdf_allocate(shader_data, make_float3(mix_weight));
        const auto extra_allocated = pool.allocate_extra(allocated, 1u);
        $if (extra_allocated) {
            const auto original_ior = max(ior, 1.0e-5f);
            const auto backfacing = (shader_data.flag & shader_data_backfacing) != 0u;
            const auto adjusted_ior =
                select(original_ior, 1.0f / original_ior, backfacing);
            const auto alpha = square(clamp(roughness, 0.0f, 1.0f));
            const auto output_type =
                select(UInt{static_cast<std::uint32_t>(
                           CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID)},
                       UInt{static_cast<std::uint32_t>(
                           CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)},
                       input_type == static_cast<std::uint32_t>(
                                         CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID));
            const auto valid_normal =
                maybe_ensure_valid_specular_reflection(shader_data, normal);

            FresnelGeneralizedSchlick fresnel{
                .thin_film = {.thickness = thin_film_thickness,
                              .ior =
                                  select(max(thin_film_ior, 1.0e-5f),
                                         max(thin_film_ior, 1.0e-5f) / original_ior,
                                         backfacing)},
                .reflection_tint =
                    select(make_float3(0.0f), max(color, make_float3(0.0f)),
                           reflective_caustics),
                .transmission_tint =
                    select(make_float3(0.0f), max(color, make_float3(0.0f)),
                           refractive_caustics),
                .f0 = clamp(make_float3(f0_from_ior(original_ior)), make_float3(0.0f),
                            make_float3(1.0f)),
                .f90 = make_float3(1.0f),
                .exponent = -original_ior};
            MicrofacetParam microfacet{.alpha_x = alpha,
                                       .alpha_y = alpha,
                                       .ior = adjusted_ior,
                                       .energy_scale = 1.0f,
                                       .fresnel_type = static_cast<std::uint32_t>(
                                           MicrofacetFresnel::generalized_schlick),
                                       .T = make_float3(0.0f)};

            pool.set_normal(allocated.index, valid_normal);
            pool.set_type(allocated.index, output_type);
            pool.set_generalized_schlick(allocated.index, fresnel);

            const auto albedo = generalized_schlick_albedo(
                kernel_globals, shader_data.wi, valid_normal, microfacet, fresnel);
            const auto common = pool.common(allocated.index);
            pool.set_sample_weight(allocated.index,
                                   common.sample_weight * average(albedo));

            $if (input_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID)) {
                preserve_multi_ggx_glass_energy(
                    kernel_globals, pool, allocated, shader_data.wi,
                    valid_normal, microfacet, fresnel);
            };
            pool.set_microfacet_param(allocated.index, microfacet);

            UInt flags = shader_data_bsdf | shader_data_bsdf_has_transmission;
            $if ((microfacet.alpha_x * microfacet.alpha_y) > 2.0e-10f) {
                flags |= shader_data_bsdf_has_eval;
            };
            shader_data.flag |= flags;
        };
    };
}

void glossy_setup(const KernelGlobals &kernel_globals, ShaderData &shader_data,
                  const PathState &path_state,
                  Expr<std::uint32_t> input_type, Expr<float> mix_weight,
                  Expr<luisa::float3> closure_weight,
                  Expr<luisa::float3> normal, Expr<luisa::float3> color,
                  Expr<float> roughness, Expr<float> anisotropy,
                  Expr<float> rotation, Expr<luisa::float3> tangent,
                  Expr<bool> tangent_valid) noexcept {
    const auto diffuse_visibility =
        (path_state.visibility & path_ray_visibility_diffuse) != 0u;
    const Bool reflective_caustics =
        kernel_globals.caustics_reflective() | !diffuse_visibility;

    $if (reflective_caustics) {
        auto &pool = *shader_data.closure;
        const auto allocated =
            bsdf_allocate(shader_data, closure_weight * mix_weight);
        $if (allocated.valid) {
            const auto valid_normal =
                maybe_ensure_valid_specular_reflection(shader_data, normal);
            const auto alpha = square(clamp(roughness, 0.0f, 1.0f));
            const auto clamped_anisotropy =
                clamp(anisotropy, -0.99f, 0.99f);

            MicrofacetParam microfacet{
                .alpha_x = alpha,
                .alpha_y = alpha,
                .ior = 1.0f,
                .energy_scale = 1.0f,
                .fresnel_type =
                    static_cast<std::uint32_t>(MicrofacetFresnel::none),
                .T = make_float3(0.0f)};
            $if (tangent_valid & (abs(clamped_anisotropy) > 1.0e-4f)) {
                microfacet.T = tangent;
                $if (rotation != 0.0f) {
                    microfacet.T = rotate_around_axis(
                        microfacet.T, valid_normal, rotation * two_pi);
                };
                $if (clamped_anisotropy < 0.0f) {
                    microfacet.alpha_x =
                        alpha / (1.0f + clamped_anisotropy);
                    microfacet.alpha_y =
                        alpha * (1.0f + clamped_anisotropy);
                }
                $else {
                    microfacet.alpha_x =
                        alpha * (1.0f - clamped_anisotropy);
                    microfacet.alpha_y =
                        alpha / (1.0f - clamped_anisotropy);
                };
            };

            UInt output_type = static_cast<std::uint32_t>(
                CLOSURE_BSDF_MICROFACET_GGX_ID);
            Bool always_has_eval = false;
            $if (input_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_BECKMANN_ID)) {
                microfacet.alpha_x = clamp(microfacet.alpha_x, 0.0f, 1.0f);
                microfacet.alpha_y = clamp(microfacet.alpha_y, 0.0f, 1.0f);
                output_type = static_cast<std::uint32_t>(
                    CLOSURE_BSDF_MICROFACET_BECKMANN_ID);
            }
            $elif (input_type == static_cast<std::uint32_t>(
                                      CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID)) {
                microfacet.alpha_x =
                    clamp(microfacet.alpha_x, 1.0e-4f, 1.0f);
                microfacet.alpha_y =
                    clamp(microfacet.alpha_y, 1.0e-4f, 1.0f);
                output_type = static_cast<std::uint32_t>(
                    CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID);
                always_has_eval = true;
            }
            $else {
                microfacet.alpha_x = clamp(microfacet.alpha_x, 0.0f, 1.0f);
                microfacet.alpha_y = clamp(microfacet.alpha_y, 0.0f, 1.0f);
            };

            pool.set_normal(allocated.index, valid_normal);
            pool.set_type(allocated.index, output_type);
            $if (input_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID)) {
                preserve_multi_ggx_reflection_energy(
                    kernel_globals, pool, allocated, shader_data.wi,
                    valid_normal, microfacet,
                    max(color, make_float3(0.0f)));
            };
            pool.set_microfacet_param(allocated.index, microfacet);

            UInt flags = shader_data_bsdf;
            $if (always_has_eval |
                 ((microfacet.alpha_x * microfacet.alpha_y) > 2.0e-10f)) {
                flags |= shader_data_bsdf_has_eval;
            };
            shader_data.flag |= flags;
        };
    };
}

void refraction_setup(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const PathState &path_state, Expr<std::uint32_t> input_type,
    Expr<float> mix_weight, Expr<luisa::float3> closure_weight,
    Expr<luisa::float3> normal, Expr<float> roughness,
    Expr<float> ior) noexcept {
    const auto diffuse_visibility =
        (path_state.visibility & path_ray_visibility_diffuse) != 0u;
    const Bool refractive_caustics =
        kernel_globals.caustics_refractive() | !diffuse_visibility;

    $if (refractive_caustics) {
        auto &pool = *shader_data.closure;
        const auto allocated =
            bsdf_allocate(shader_data, closure_weight * mix_weight);
        $if (allocated.valid) {
            const auto backfacing =
                (shader_data.flag & shader_data_backfacing) != 0u;
            const auto original_ior = max(ior, 1.0e-5f);
            const auto adjusted_ior =
                select(original_ior, 1.0f / original_ior, backfacing);
            const auto alpha = clamp(square(roughness), 0.0f, 1.0f);
            const auto beckmann =
                input_type == static_cast<std::uint32_t>(
                                  CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID);
            const auto output_type =
                select(UInt{static_cast<std::uint32_t>(
                           CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID)},
                       UInt{static_cast<std::uint32_t>(
                           CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID)},
                       beckmann);
            const auto valid_normal =
                maybe_ensure_valid_specular_reflection(shader_data, normal);
            const MicrofacetParam microfacet{
                .alpha_x = alpha,
                .alpha_y = alpha,
                .ior = adjusted_ior,
                .energy_scale = 1.0f,
                .fresnel_type =
                    static_cast<std::uint32_t>(MicrofacetFresnel::none),
                .T = make_float3(0.0f)};
            pool.set_normal(allocated.index, valid_normal);
            pool.set_type(allocated.index, output_type);
            pool.set_microfacet_param(allocated.index, microfacet);

            UInt flags = shader_data_bsdf | shader_data_bsdf_has_transmission;
            $if ((microfacet.alpha_x * microfacet.alpha_y) > 2.0e-10f) {
                flags |= shader_data_bsdf_has_eval;
            };
            shader_data.flag |= flags;
        };
    };
}

}// namespace psycles::luisa_backend::cycles_svm::detail
