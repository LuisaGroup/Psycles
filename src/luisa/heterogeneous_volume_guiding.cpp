#include <psycles/luisa/heterogeneous_volume_guiding.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

HeterogeneousVolumeGuidingSample
HeterogeneousVolumeScatterProbability::evaluate(
    const HeterogeneousVolumeGuidingState
        &state) const noexcept {
    Float scatter_probability = 1.0f;
    Float majorant_scale = 1.0f;

    $if(state.enabled) {
        const auto scattered =
            state.scattered_radiance.x +
            state.scattered_radiance.y +
            state.scattered_radiance.z;
        const auto transmitted =
            state.transmitted_radiance.x +
            state.transmitted_radiance.y +
            state.transmitted_radiance.z;
        const auto volume = scattered + transmitted;

        $if(volume == 0.0f) {
            scatter_probability = 0.5f;
        }
        $else {
            // The exponential tail gives transmission non-zero probability,
            // so Cycles caps the requested scatter probability below one.
            scatter_probability =
                min(
                    scattered / volume,
                    0.9999f);
        };

        $if(state.majorant_optical_depth ==
            0.0f) {
            majorant_scale = 1.0f;
        }
        $else {
            majorant_scale =
                -log(
                    1.0f -
                    scatter_probability) /
                state.majorant_optical_depth;
        };

        $if(majorant_scale < 1.0f) {
            majorant_scale = 1.0f;
            const auto attenuation =
                1.0f -
                exp(
                    -state
                         .majorant_optical_depth);
            scatter_probability =
                scatter_probability /
                select(
                    1.0f,
                    attenuation,
                    attenuation != 0.0f);
        }
        $else {
            scatter_probability = 1.0f;
        };
    };

    return {
        .scatter_probability =
            scatter_probability,
        .majorant_scale =
            majorant_scale,
        .enabled = state.enabled};
}

}// namespace psycles::luisa_backend
