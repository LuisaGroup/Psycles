#include <psycles/luisa/heterogeneous_volume_tracking.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

Float3 HeterogeneousVolumeTracking::_safe_divide_color(
    Float3 numerator,
    Float3 denominator,
    float fallback) noexcept {
    const auto nonzero =
        denominator != make_float3(0.0f);
    return select(
        make_float3(fallback),
        numerator /
            select(
                make_float3(1.0f),
                denominator,
                nonzero),
        nonzero);
}

Float3 HeterogeneousVolumeTracking::_safe_divide_scalar(
    Float3 numerator,
    Float denominator) noexcept {
    const auto nonzero = denominator != 0.0f;
    return select(
        make_float3(0.0f),
        numerator /
            select(1.0f, denominator, nonzero),
        nonzero);
}

Float HeterogeneousVolumeTracking::candidate_distance(
    Float random,
    Float majorant) const noexcept {
    const auto active = majorant != 0.0f;
    return select(
        0.0f,
        -log(1.0f - random) /
            select(1.0f, majorant, active),
        active);
}

VolumeNullEventCoefficients
HeterogeneousVolumeTracking::null_event_coefficients(
    Float3 sigma_t,
    Float sampled_majorant) const noexcept {
    const auto actual_maximum =
        max(sigma_t.x, max(sigma_t.y, sigma_t.z));
    const auto majorant =
        max(actual_maximum, sampled_majorant);
    return {
        .sigma_n =
            make_float3(majorant) - sigma_t,
        .majorant = majorant,
        .majorant_exceeded =
            actual_maximum > sampled_majorant};
}

HeterogeneousVolumeCollision
HeterogeneousVolumeTracking::evaluate_collision(
    const VolumeCoefficients &coefficients,
    Float sampled_majorant,
    Float candidate_distance_value,
    Float3 throughput,
    Float transmittance) const noexcept {
    const auto null_event =
        null_event_coefficients(
            coefficients.sigma_t,
            sampled_majorant);
    const auto active = sampled_majorant != 0.0f;

    // With a sound majorant, null_event.majorant == sampled_majorant and
    // this is exactly throughput / majorant. The exponential branch matches
    // Cycles' defensive correction while surfacing the contract violation.
    const auto corrected =
        throughput *
        exp(
            (sampled_majorant -
             null_event.majorant) *
            candidate_distance_value);
    const auto defensive_normalization =
        _safe_divide_scalar(
            corrected,
            sampled_majorant);
    const auto ordinary_normalization =
        _safe_divide_scalar(
            throughput,
            null_event.majorant);
    const auto normalized_throughput =
        select(
            ordinary_normalization,
            defensive_normalization,
            null_event.majorant !=
                sampled_majorant);
    const auto emission =
        normalized_throughput *
        coefficients.emission;

    const auto absorption_only =
        coefficients.sigma_s.x +
            coefficients.sigma_s.y +
            coefficients.sigma_s.z ==
        0.0f;
    Float3 channel_pdf = make_float3(0.0f);
    Float scatter_probability = 0.0f;
    // Cycles returns before constructing a channel measure for an
    // absorption-only candidate. Preserve that control-flow domain so dead
    // 0/0 spectral quotients cannot leak NaNs into the result.
    $if(!absorption_only) {
        const auto sigma_c =
            coefficients.sigma_s +
            null_event.sigma_n;
        const auto albedo =
            _safe_divide_color(
                coefficients.sigma_s,
                coefficients.sigma_t,
                1.0f);
        const auto channel_weights =
            abs(normalized_throughput * albedo);
        const auto weight_sum =
            channel_weights.x +
            channel_weights.y +
            channel_weights.z;
        // Cycles uses this form to reject denormals that could normalize to
        // NaN.
        const auto usable_weights =
            (1.0f - weight_sum) < 1.0f;
        channel_pdf =
            select(
                make_float3(1.0f / 3.0f),
                channel_weights /
                    select(
                        1.0f,
                        weight_sum,
                        usable_weights),
                usable_weights);

        // Cycles evaluates this quotient directly inside the scattering
        // domain.
        const auto scatter_ratio =
            coefficients.sigma_s / sigma_c;
        scatter_probability =
            dot(scatter_ratio, channel_pdf);
    };
    const auto null_probability =
        1.0f - scatter_probability;
    const auto scatter_throughput =
        _safe_divide_scalar(
            normalized_throughput *
                coefficients.sigma_s,
            scatter_probability);
    const auto stochastic_null_throughput =
        _safe_divide_scalar(
            normalized_throughput *
                null_event.sigma_n,
            null_probability);
    const auto null_throughput =
        select(
            stochastic_null_throughput,
            normalized_throughput *
                null_event.sigma_n,
            absorption_only);
    const auto null_transmittance =
        select(
            transmittance * null_probability,
            transmittance,
            absorption_only);

    return {
        .null_event = null_event,
        .normalized_throughput =
            normalized_throughput,
        .emission = emission,
        .channel_pdf = channel_pdf,
        .scatter_probability =
            scatter_probability,
        .null_probability =
            null_probability,
        .scatter_throughput =
            scatter_throughput,
        .null_throughput =
            null_throughput,
        .null_transmittance =
            null_transmittance,
        .absorption_only =
            absorption_only,
        .active = active};
}

HeterogeneousVolumeEventSelection
HeterogeneousVolumeTracking::select_event(
    Float random,
    Float scatter_probability) const noexcept {
    const auto scattered =
        random <= scatter_probability;
    const auto null_probability =
        1.0f - scatter_probability;
    const auto scatter_random =
        random /
        select(
            1.0f,
            scatter_probability,
            scattered);
    const auto null_random =
        (random - scatter_probability) /
        select(
            1.0f,
            null_probability,
            !scattered);
    return {
        .random =
            clamp(
                select(
                    null_random,
                    scatter_random,
                    scattered),
                0.0f,
                1.0f),
        .scattered = scattered};
}

}// namespace psycles::luisa_backend
