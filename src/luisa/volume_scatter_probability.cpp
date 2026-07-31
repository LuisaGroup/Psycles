#include <psycles/luisa/volume_scatter_probability.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

Float3 HomogeneousVolumeScatterProbability::_safe_divide(
    Float3 numerator,
    Float3 denominator) noexcept {
    const auto nonzero =
        denominator != make_float3(0.0f);
    const auto safe_denominator =
        select(
            make_float3(1.0f),
            denominator,
            nonzero);
    return select(
        make_float3(0.0f),
        numerator / safe_denominator,
        nonzero);
}

Float3 HomogeneousVolumeScatterProbability::evaluate(
    const VolumeCoefficients &coefficients,
    Float distance,
    Bool terminate,
    const VolumeScatterProbabilityGuidingState &guiding) const noexcept {
    const auto attenuation_only =
        terminate |
        !coefficients.has_scatter |
        all(
            coefficients.sigma_s ==
            make_float3(0.0f));
    const auto attenuation =
        make_float3(1.0f) -
        exp(
            -coefficients.sigma_t *
            distance);

    const auto volume_radiance =
        guiding.transmitted_radiance +
        guiding.scattered_radiance;
    const auto history_empty =
        all(
            volume_radiance ==
            make_float3(0.0f));
    const auto empty_history_probability =
        select(
            make_float3(0.0f),
            make_float3(0.5f),
            coefficients.sigma_t >
                make_float3(0.0f));

    // The guide represents the whole primary volume ray, while this
    // estimator may cover only one homogeneous segment. Cycles scales the
    // contribution ratio by the segment attenuation relative to the
    // accumulated majorant optical depth before defensive sampling.
    const auto optical_attenuation =
        1.0f -
        exp(
            -guiding
                 .majorant_optical_depth);
    const auto safe_optical_attenuation =
        select(
            1.0f,
            optical_attenuation,
            optical_attenuation !=
                0.0f);
    const auto scale =
        luisa::compute::max(
            attenuation.x,
            luisa::compute::max(
                attenuation.y,
                attenuation.z)) /
        safe_optical_attenuation;
    const auto history_probability =
        clamp(
            _safe_divide(
                guiding.scattered_radiance,
                volume_radiance) *
                scale,
            make_float3(0.0f),
            make_float3(1.0f));
    const auto guided_probability =
        select(
            history_probability,
            empty_history_probability,
            history_empty);
    const auto defensive_probability =
        attenuation * 0.25f +
        guided_probability * 0.75f;
    const auto probability =
        select(
            attenuation,
            defensive_probability,
            guiding.enabled);
    return select(
        probability,
        make_float3(0.0f),
        attenuation_only);
}

}// namespace psycles::luisa_backend
