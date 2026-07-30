#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_volume.h> through the Psycles::luisa target."
#endif

#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_color_nodes.h>
#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::cycles_volume {

[[nodiscard]] inline Float sample_weight(Float3 value) noexcept {
    return abs((value.x + value.y + value.z) / 3.0f);
}

template<typename ScalarReader, typename VectorReader>
void accumulate_coefficients(
    const compiler::VolumeInstruction &volume,
    Float mix_weight,
    const ShaderServices &services,
    const SurfacePoint &point,
    const VolumeQuery &query,
    ScalarReader &&scalar,
    VectorReader &&vector,
    VolumeCoefficients &result) noexcept {
    const auto add_scatter =
        [&](Float3 weight, Bool active) noexcept {
            // Cycles closure_sample_weight() clamps negative volume closure
            // weights, but deliberately applies no 1e-5 surface cutoff to a
            // volume shader.
            const auto allocated =
                max(weight, make_float3(0.0f));
            const auto allocated_active =
                active &
                (sample_weight(allocated) > 0.0f);
            result.sigma_s += select(
                make_float3(0.0f),
                allocated,
                allocated_active);
            result.has_scatter =
                result.has_scatter | allocated_active;
        };
    const auto add_extinction =
        [&](Float3 weight, Bool active) noexcept {
            result.sigma_t += select(
                make_float3(0.0f),
                weight,
                active);
            result.has_extinction =
                result.has_extinction | active;
        };
    const auto add_emission =
        [&](Float3 weight, Bool active) noexcept {
            const auto enabled =
                active & query.evaluate_emission;
            result.emission += select(
                make_float3(0.0f),
                weight,
                enabled);
            result.has_emission =
                result.has_emission | enabled;
        };

    const auto active = mix_weight != 0.0f;
    switch (volume.operation) {
        case compiler::VolumeOperation::absorption: {
            const auto density =
                mix_weight *
                max(scalar(volume.density), 0.0f) *
                query.object_density;
            const auto extinction =
                (make_float3(1.0f) -
                 vector(volume.color)) *
                density;
            add_extinction(extinction, active);
            return;
        }
        case compiler::VolumeOperation::scatter: {
            const auto density =
                mix_weight *
                max(scalar(volume.density), 0.0f) *
                query.object_density;
            const auto scatter =
                vector(volume.color) * density;
            add_scatter(scatter, active);
            add_extinction(scatter, active);
            return;
        }
        case compiler::VolumeOperation::coefficients: {
            const auto weight =
                mix_weight * query.object_density;
            const auto scatter =
                vector(volume.scatter_coefficients) *
                weight;
            const auto absorption =
                vector(volume.absorption_coefficients) *
                weight;
            add_scatter(scatter, active);
            add_extinction(
                scatter + absorption, active);
            const auto emission =
                vector(volume.emission_coefficients);
            add_emission(
                emission * weight,
                active &
                    any(emission != make_float3(0.0f)));
            return;
        }
        case compiler::VolumeOperation::principled: {
            const auto object_weight =
                mix_weight * query.object_density;
            auto density =
                object_weight *
                max(scalar(volume.density), 0.0f);
            const auto density_attribute =
                services.attribute(
                    contract::attribute_id("density"),
                    point);
            density = select(
                density,
                max(
                    density *
                        density_attribute.value.x,
                    0.0f),
                (density > 0.0f) &
                    density_attribute.found);
            const auto density_active =
                density > 0.0f;
            const auto color = vector(volume.color);
            const auto scatter = color * density;
            add_scatter(scatter, density_active);
            const auto absorption_color =
                max(
                    sqrt(vector(volume.absorption_color)),
                    make_float3(0.0f));
            const auto absorption =
                max(
                    make_float3(1.0f) - color,
                    make_float3(0.0f)) *
                max(
                    make_float3(1.0f) -
                        absorption_color,
                    make_float3(0.0f));
            add_extinction(
                (color + absorption) * density,
                density_active);

            const auto emission_strength =
                scalar(volume.emission_strength);
            add_emission(
                emission_strength *
                    vector(volume.emission_color) *
                    object_weight,
                emission_strength > 0.0f);

            const auto blackbody =
                scalar(volume.blackbody_intensity);
            auto temperature =
                scalar(volume.temperature);
            const auto temperature_attribute =
                services.attribute(
                    contract::attribute_id("temperature"),
                    point);
            temperature = select(
                temperature,
                temperature *
                    max(
                        temperature_attribute.value.x,
                        0.0f),
                temperature_attribute.found);
            temperature = max(temperature, 0.0f);
            const auto temperature2 =
                temperature * temperature;
            const auto temperature4 =
                temperature2 * temperature2;
            constexpr auto pi =
                3.14159265358979323846f;
            constexpr auto sigma =
                5.670373e-8f * 1.0e-6f / pi;
            const auto intensity =
                sigma *
                lerp(
                    1.0f,
                    temperature4,
                    blackbody);
            const auto blackbody_active =
                (blackbody > 0.0f) &
                (intensity > 0.0f);
            const auto blackbody_color =
                vector(volume.blackbody_tint) *
                intensity *
                services.rec709_to_rgb(
                    cycles_color_nodes::
                        blackbody_rec709(
                            temperature));
            add_emission(
                blackbody_color * object_weight,
                blackbody_active);
            return;
        }
        case compiler::VolumeOperation::null_volume:
        case compiler::VolumeOperation::add:
        case compiler::VolumeOperation::mix:
            return;
    }
}

}// namespace psycles::luisa_backend::cycles_volume
