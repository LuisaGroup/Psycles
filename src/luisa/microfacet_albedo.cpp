#include "microfacet_albedo.h"

#include "graph_surface_internal.h"
#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_bsdf_tables.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

using namespace luisa::compute;

Float3 microfacet_generalized_schlick_albedo(
    const CyclesBsdfTableReader &tables,
    Float3 incoming,
    Float3 normal,
    Float alpha_x,
    Float alpha_y,
    Float ior,
    Float thin_film_thickness,
    Float thin_film_ior,
    Float3 reflection_tint,
    Float3 transmission_tint,
    Float3 f0,
    Float3 f90,
    Float exponent,
    Bool reflection,
    Bool transmission,
    bool may_have_thin_film) noexcept {
    const auto reflectance =
        microfacet_generalized_schlick_reflectance(
            tables,
            incoming,
            normal,
            alpha_x,
            alpha_y,
            ior,
            thin_film_thickness,
            thin_film_ior,
            f0,
            f90,
            exponent,
            may_have_thin_film);
    return reflectance * reflection_tint *
               select(0.0f, 1.0f, reflection) +
           (make_float3(1.0f) - reflectance) * transmission_tint *
               select(0.0f, 1.0f, transmission);
}

Float3 microfacet_generalized_schlick_reflectance(
    const CyclesBsdfTableReader &tables,
    Float3 incoming,
    Float3 normal,
    Float alpha_x,
    Float alpha_y,
    Float ior,
    Float thin_film_thickness,
    Float thin_film_ior,
    Float3 f0,
    Float3 f90,
    Float exponent,
    bool may_have_thin_film) noexcept {
    const auto cosine_incoming = dot(incoming, normal);
    Float3 reflectance;
    const auto table_reflectance = [&] noexcept {
        const auto table_roughness =
            sqrt(sqrt(alpha_x * alpha_y));
        Float z;
        UInt table_offset;
        $if(exponent < 0.0f) {
            z = sqrt(abs((ior - 1.0f) / (ior + 1.0f)));
            table_offset =
                cycles45_tables::ggx_gen_schlick_ior_s_offset;
        }
        $else {
            z = 1.0f / (0.2f * exponent + 1.0f);
            table_offset = cycles45_tables::ggx_gen_schlick_s_offset;
        };
        const auto interpolation = cycles_table_3d(
            tables,
            table_roughness,
            cosine_incoming,
            z,
            table_offset,
            16u,
            16u,
            16u);
        return lerp(f0, f90, interpolation);
    };
    if (may_have_thin_film) {
        $if(thin_film_thickness > thin_film_thickness_cutoff) {
            reflectance = thin_film_dielectric_fresnel(
                              tables,
                              thin_film_thickness,
                              thin_film_ior,
                              ior,
                              f0,
                              cosine_incoming)
                              .reflectance;
        }
        $else {
            reflectance = table_reflectance();
        };
    } else {
        reflectance = table_reflectance();
    }
    return reflectance;
}

Float microfacet_dielectric_reflection_albedo(
    const CyclesBsdfTableReader &tables,
    Float3 incoming,
    Float3 normal,
    Float alpha_x,
    Float alpha_y,
    Float ior) noexcept {
    Float result = 0.0f;
    // Cycles uses this LUT specialization only above eta=1. The caller keeps
    // the exact directional Fresnel fallback for eta<=1.
    $if(ior > 1.0f) {
        const auto table_roughness =
            sqrt(sqrt(alpha_x * alpha_y));
        const auto cosine_incoming = dot(incoming, normal);
        const auto z =
            sqrt(abs((ior - 1.0f) / (ior + 1.0f)));
        const auto interpolation = cycles_table_3d(
            tables,
            table_roughness,
            cosine_incoming,
            z,
            UInt{cycles45_tables::ggx_gen_schlick_ior_s_offset},
            16u,
            16u,
            16u);
        result = lerp(f0_from_ior(ior), 1.0f, interpolation);
    };
    return result;
}

Float3 microfacet_f82_tint_albedo(
    const CyclesBsdfTableReader &tables,
    Float3 incoming,
    Float3 normal,
    Float alpha_x,
    Float alpha_y,
    Float thin_film_thickness,
    Float thin_film_ior,
    Float3 f0,
    Float3 b,
    bool may_have_thin_film) noexcept {
    const auto cosine_incoming = dot(incoming, normal);
    Float3 result;
    const auto table_result = [&] noexcept {
        const auto table_roughness =
            sqrt(sqrt(alpha_x * alpha_y));
        const auto interpolation = cycles_table_3d(
            tables,
            table_roughness,
            cosine_incoming,
            0.5f,
            UInt{cycles45_tables::ggx_gen_schlick_s_offset},
            16u,
            16u,
            16u);
        return lerp(f0, make_float3(1.0f), interpolation);
    };
    if (may_have_thin_film) {
        $if(thin_film_thickness > thin_film_thickness_cutoff) {
            result = thin_film_f82_fresnel(
                tables,
                thin_film_thickness,
                thin_film_ior,
                f0,
                b,
                cosine_incoming);
        }
        $else {
            result = table_result();
        };
    } else {
        result = table_result();
    }
    return result;
}

Float3 microfacet_conductor_albedo(
    const CyclesBsdfTableReader &tables,
    Float3 incoming,
    Float3 normal,
    Float thin_film_thickness,
    Float thin_film_ior,
    Float3 ior,
    Float3 extinction,
    bool may_have_thin_film) noexcept {
    const auto cosine_incoming = dot(incoming, normal);
    Float3 result;
    if (may_have_thin_film) {
        $if(thin_film_thickness > thin_film_thickness_cutoff) {
            result = thin_film_conductor_fresnel(
                tables,
                thin_film_thickness,
                thin_film_ior,
                ior,
                extinction,
                cosine_incoming);
        }
        $else {
            result = fresnel_conductor(cosine_incoming, ior, extinction);
        };
    } else {
        result = fresnel_conductor(cosine_incoming, ior, extinction);
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
