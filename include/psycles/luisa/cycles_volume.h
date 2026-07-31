#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_volume.h> through the Psycles::luisa target."
#endif

#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_color_nodes.h>
#include <psycles/luisa/cycles_volume_phase.h>
#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::cycles_volume {

[[nodiscard]] inline Float sample_weight(Float3 value) noexcept {
    return abs((value.x + value.y + value.z) / 3.0f);
}

struct LeafCoefficients {
    Float3 sigma_t;
    Float3 sigma_s;
    Float3 emission;
    Bool has_extinction;
    Bool has_scatter;
    Bool has_emission;

    [[nodiscard]] static LeafCoefficients zero() noexcept {
        return {
            .sigma_t = make_float3(0.0f),
            .sigma_s = make_float3(0.0f),
            .emission = make_float3(0.0f),
            .has_extinction = false,
            .has_scatter = false,
            .has_emission = false};
    }
};

template<typename ScalarReader, typename VectorReader>
[[nodiscard]] LeafCoefficients evaluate_leaf(
    const compiler::VolumeInstruction &volume,
    Float mix_weight,
    const ShaderServices &services,
    const SurfacePoint &point,
    const VolumeQuery &query,
    ScalarReader &&scalar,
    VectorReader &&vector) noexcept {
    auto result = LeafCoefficients::zero();
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
            return result;
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
            return result;
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
            return result;
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
                active &
                    (emission_strength > 0.0f));

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
                active & blackbody_active);
            return result;
        }
        case compiler::VolumeOperation::null_volume:
        case compiler::VolumeOperation::add:
        case compiler::VolumeOperation::mix:
            return result;
    }
    return result;
}

inline void accumulate_coefficients(
    const LeafCoefficients &leaf,
    VolumeCoefficients &result) noexcept {
    result.sigma_t += leaf.sigma_t;
    result.sigma_s += leaf.sigma_s;
    result.emission += leaf.emission;
    result.has_extinction =
        result.has_extinction | leaf.has_extinction;
    result.has_scatter =
        result.has_scatter | leaf.has_scatter;
    result.has_emission =
        result.has_emission | leaf.has_emission;
}

template<typename ScalarReader, typename Visitor>
void emit_phase_closures(
    const compiler::VolumeInstruction &volume,
    const LeafCoefficients &leaf,
    ScalarReader &&scalar,
    Visitor &&visitor) noexcept {
    if (volume.operation !=
            compiler::VolumeOperation::scatter &&
        volume.operation !=
            compiler::VolumeOperation::coefficients &&
        volume.operation !=
            compiler::VolumeOperation::principled) {
        return;
    }
    const auto add =
        [&](cycles_volume_phase::Closure phase,
            Float3 weight) noexcept {
            visitor(phase, weight);
        };
    if (volume.operation ==
        compiler::VolumeOperation::principled) {
        add(
            cycles_volume_phase::henyey_greenstein(
                scalar(volume.anisotropy)),
            leaf.sigma_s);
        return;
    }
    switch (volume.phase) {
        case compiler::VolumePhase::
            henyey_greenstein:
            add(
                cycles_volume_phase::
                    henyey_greenstein(
                        scalar(volume.anisotropy)),
                leaf.sigma_s);
            return;
        case compiler::VolumePhase::fournier_forand:
            add(
                cycles_volume_phase::fournier_forand(
                    scalar(volume.backscatter),
                    scalar(volume.ior)),
                leaf.sigma_s);
            return;
        case compiler::VolumePhase::draine:
            add(
                cycles_volume_phase::draine(
                    scalar(volume.anisotropy),
                    scalar(volume.alpha)),
                leaf.sigma_s);
            return;
        case compiler::VolumePhase::rayleigh:
            add(
                cycles_volume_phase::rayleigh(),
                leaf.sigma_s);
            return;
        case compiler::VolumePhase::mie: {
            const auto parameters =
                cycles_volume_phase::mie_parameters(
                    scalar(volume.diameter));
            add(
                cycles_volume_phase::
                    henyey_greenstein(
                        parameters
                            .henyey_greenstein_g),
                leaf.sigma_s *
                    (1.0f -
                     parameters.draine_weight));
            add(
                cycles_volume_phase::draine(
                    parameters.draine_g,
                    parameters.draine_alpha),
                leaf.sigma_s *
                    parameters.draine_weight);
            return;
        }
    }
}

}// namespace psycles::luisa_backend::cycles_volume
