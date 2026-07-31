#include <psycles/luisa/homogeneous_volume_transport.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

Float3 HomogeneousVolumeTransport::_safe_divide(
    Float3 numerator,
    Float3 denominator,
    float fallback) noexcept {
    const auto nonzero = denominator != make_float3(0.0f);
    const auto safe_denominator =
        select(make_float3(1.0f), denominator, nonzero);
    return select(
        make_float3(fallback),
        numerator / safe_denominator,
        nonzero);
}

Float3 HomogeneousVolumeTransport::_safe_divide(
    Float3 numerator,
    Float denominator) noexcept {
    const auto nonzero = denominator != 0.0f;
    return select(
        make_float3(0.0f),
        numerator / select(1.0f, denominator, nonzero),
        nonzero);
}

Float3 HomogeneousVolumeTransport::transmittance(
    Float3 sigma_t,
    Float distance) const noexcept {
    return exp(-sigma_t * distance);
}

Float3 HomogeneousVolumeTransport::emission_integral(
    const VolumeCoefficients &coefficients,
    Float distance) const noexcept {
    const auto optical_depth =
        coefficients.sigma_t * distance;
    const auto analytic =
        optical_depth > make_float3(1.0e-5f);
    const auto safe_sigma_t = select(
        make_float3(1.0f),
        coefficients.sigma_t,
        analytic);
    const auto attenuated = select(
        distance *
            (make_float3(1.0f) -
             0.5f * optical_depth),
        (make_float3(1.0f) -
         exp(-optical_depth)) /
            safe_sigma_t,
        analytic);
    return coefficients.emission *
           select(
               make_float3(distance),
               attenuated,
               coefficients.has_extinction);
}

Float3 HomogeneousVolumeTransport::channel_pdf(
    Float3 albedo,
    Float3 throughput) const noexcept {
    const auto weights =
        abs(throughput * albedo);
    const auto sum =
        weights.x + weights.y + weights.z;
    // This is intentionally the Cycles `(1 - sum) < 1` test rather than
    // `sum > 0`: the former rejects denormals that could produce NaNs.
    const auto usable = (1.0f - sum) < 1.0f;
    const auto normalized =
        weights / select(1.0f, sum, usable);
    return select(
        make_float3(1.0f / 3.0f),
        normalized,
        usable);
}

VolumeChannelSample
HomogeneousVolumeTransport::sample_channel(
    Float3 albedo,
    Float3 throughput,
    Float random) const noexcept {
    const auto pdf = channel_pdf(
        albedo, throughput);
    const auto first = random < pdf.x;
    const auto second =
        !first & (random < pdf.x + pdf.y);
    const auto third =
        !first & !second &
        (random < pdf.x + pdf.y + pdf.z);
    const auto matched =
        first | second | third;
    const auto channel =
        select(2u, select(1u, 0u, first), first | second);
    const auto lower =
        select(
            pdf.x + pdf.y,
            select(pdf.x, 0.0f, first),
            first | second);
    const auto selected_pdf =
        select(
            pdf.z,
            select(pdf.y, pdf.x, first),
            first | second);
    const auto rescaled =
        (random - lower) /
        select(1.0f, selected_pdf, matched);
    return {
        .channel = channel,
        .pdf = pdf,
        .random =
            select(random, rescaled, matched),
        .matched = matched};
}

Float HomogeneousVolumeTransport::
    bounded_exponential_sample(
        Float random,
        Float rate,
        Float minimum,
        Float maximum) const noexcept {
    const auto attenuation =
        1.0f -
        exp(rate * (minimum - maximum));
    const auto safe_rate =
        select(1.0f, rate, rate != 0.0f);
    return clamp(
        minimum -
            log(
                1.0f -
                random * attenuation) /
                safe_rate,
        minimum,
        maximum);
}

Float3 HomogeneousVolumeTransport::
    bounded_exponential_pdf(
        Float distance,
        Float3 rate,
        Float minimum,
        Float maximum) const noexcept {
    const auto attenuation =
        exp(-rate * minimum) -
        exp(-rate * maximum);
    const auto numerator =
        rate *
        exp(
            -rate *
            clamp(
                make_float3(distance),
                make_float3(minimum),
                make_float3(maximum)));
    return _safe_divide(
        numerator, attenuation);
}

Float3 HomogeneousVolumeTransport::
    unguided_scatter_probability(
        const VolumeCoefficients &coefficients,
        Float distance,
        Bool terminate) const noexcept {
    const auto has_scatter =
        coefficients.has_scatter &
        any(
            coefficients.sigma_s !=
            make_float3(0.0f));
    return select(
        make_float3(0.0f),
        make_float3(1.0f) -
            transmittance(
                coefficients.sigma_t,
                distance),
        !terminate & has_scatter);
}

HomogeneousVolumeSample
HomogeneousVolumeTransport::sample(
    const VolumeCoefficients &coefficients,
    Float distance,
    Float3 throughput,
    Float scatter_random,
    Float channel_random,
    Bool terminate) const noexcept {
    return sample_with_probability(
        coefficients,
        distance,
        throughput,
        scatter_random,
        channel_random,
        unguided_scatter_probability(
            coefficients,
            distance,
            terminate),
        terminate);
}

HomogeneousVolumeSample
HomogeneousVolumeTransport::sample_with_probability(
    const VolumeCoefficients &coefficients,
    Float distance,
    Float3 throughput,
    Float scatter_random,
    Float channel_random,
    Float3 scatter_probability,
    Bool terminate) const noexcept {
    const auto segment_transmittance =
        transmittance(
            coefficients.sigma_t,
            distance);
    const auto segment_emission =
        throughput *
        emission_integral(
            coefficients, distance);
    const auto has_scatter =
        coefficients.has_scatter &
        any(
            coefficients.sigma_s !=
            make_float3(0.0f));
    const auto eligible =
        !terminate &
        has_scatter &
        (distance > 0.0f);
    scatter_probability = select(
        make_float3(0.0f),
        scatter_probability,
        eligible);

    const auto albedo = _safe_divide(
        coefficients.sigma_s,
        coefficients.sigma_t);
    const auto multiple_scattering_albedo =
        albedo *
        (make_float3(1.0f) -
         segment_transmittance) *
        throughput;
    const auto channel_sample =
        sample_channel(
            multiple_scattering_albedo +
                segment_transmittance,
            throughput,
            channel_random);
    const auto channel =
        channel_sample.channel;
    const auto channel_scatter_probability =
        select(
            scatter_probability.z,
            select(
                scatter_probability.y,
                scatter_probability.x,
                channel == 0u),
            channel < 2u);
    const auto scattered =
        eligible &
        (scatter_random <
         channel_scatter_probability);
    const auto rescaled_scatter =
        scatter_random /
        select(
            1.0f,
            channel_scatter_probability,
            scattered);
    const auto rate =
        select(
            coefficients.sigma_t.z,
            select(
                coefficients.sigma_t.y,
                coefficients.sigma_t.x,
                channel == 0u),
            channel < 2u);
    const auto sampled_distance =
        bounded_exponential_sample(
            rescaled_scatter,
            rate,
            0.0f,
            distance);
    const auto scatter_pdf =
        dot(
            bounded_exponential_pdf(
                sampled_distance,
                coefficients.sigma_t,
                0.0f,
                distance) *
                scatter_probability,
            channel_sample.pdf);
    const auto scatter_throughput =
        throughput *
        _safe_divide(
            coefficients.sigma_s *
                transmittance(
                    coefficients.sigma_t,
                    sampled_distance),
            scatter_pdf);

    const auto transmit_pdf =
        dot(
            make_float3(1.0f) -
                scatter_probability,
            channel_sample.pdf);
    const auto transmit_throughput =
        throughput *
        _safe_divide(
            segment_transmittance,
            transmit_pdf);
    const auto transmit_rescaled =
        (scatter_random -
         channel_scatter_probability) /
        select(
            1.0f,
            1.0f -
                channel_scatter_probability,
            !scattered & eligible);
    const auto active =
        coefficients.has_extinction |
        coefficients.has_scatter |
        coefficients.has_emission;
    return {
        .transmittance =
            segment_transmittance,
        .emission = segment_emission,
        .throughput = select(
            transmit_throughput,
            scatter_throughput,
            scattered),
        .scatter_probability =
            scatter_probability,
        .channel_pdf =
            channel_sample.pdf,
        .distance = select(
            distance,
            sampled_distance,
            scattered),
        .event_pdf = select(
            transmit_pdf,
            scatter_pdf,
            scattered),
        .scatter_random = select(
            transmit_rescaled,
            rescaled_scatter,
            scattered),
        .reservoir_random =
            channel_sample.random,
        .channel = channel,
        .scattered = scattered,
        .active = active};
}

HomogeneousVolumeDirectSample
HomogeneousVolumeTransport::
    sample_direct_distance(
        const VolumeCoefficients &coefficients,
        Float distance,
        Float3 throughput,
        Float scatter_random,
        Float reservoir_random,
        Bool enabled) const noexcept {
    const VolumeDirectSampling sampling;
    const auto state =
        sampling.prepare(
            volume_sample_distance,
            scatter_random,
            enabled);
    return sample_direct(
        coefficients,
        distance,
        throughput,
        scatter_random,
        reservoir_random,
        state,
        make_float3(0.0f),
        make_float3(
            0.0f, 0.0f, 1.0f),
        {.light_position =
             make_float3(
                 1.0f, 0.0f, 0.0f),
         .interval =
             {.minimum = 0.0f,
              .maximum = distance}});
}

HomogeneousVolumeDirectSample
HomogeneousVolumeTransport::sample_direct(
    const VolumeCoefficients &coefficients,
    Float distance,
    Float3 throughput,
    Float scatter_random,
    Float reservoir_random,
    const VolumeDirectSamplingState &sampling,
    Float3 ray_origin,
    Float3 ray_direction,
    const VolumeEquiangularCoefficients
        &equiangular) const noexcept {
    const auto segment_transmittance =
        transmittance(
            coefficients.sigma_t,
            distance);
    const auto has_scatter =
        coefficients.has_scatter &
        any(
            coefficients.sigma_s !=
            make_float3(0.0f));
    const auto scattered =
        sampling.enabled &
        has_scatter &
        (distance > 0.0f) &
        (equiangular.interval.minimum <
         equiangular.interval.maximum);

    const auto albedo = _safe_divide(
        coefficients.sigma_s,
        coefficients.sigma_t);
    // volume_integrate_homogeneous stores this throughput-weighted quantity
    // in vstate.albedo before reusing the reservoir dimension for the direct
    // channel selection. Preserve that apparently redundant second
    // throughput factor in volume_sample_channel_pdf: it is part of the
    // Cycles estimator measure.
    const auto volume_albedo =
        albedo *
        (make_float3(1.0f) -
         segment_transmittance) *
        throughput;
    const auto channel_sample =
        sample_channel(
            volume_albedo,
            throughput,
            reservoir_random);
    const auto channel =
        channel_sample.channel;
    const auto rate =
        select(
            coefficients.sigma_t.z,
            select(
                coefficients.sigma_t.y,
                coefficients.sigma_t.x,
                channel == 0u),
            channel < 2u);
    const auto distance_sample =
        bounded_exponential_sample(
            scatter_random,
            rate,
            equiangular.interval.minimum,
            equiangular.interval.maximum);
    const auto distance_sample_pdf =
        dot(
            bounded_exponential_pdf(
                distance_sample,
                coefficients.sigma_t,
                equiangular.interval.minimum,
                equiangular.interval.maximum),
            channel_sample.pdf);
    const VolumeDirectSampling
        direct_sampling;
    const auto equiangular_sample =
        direct_sampling.sample_equiangular(
            ray_origin,
            ray_direction,
            equiangular,
            sampling.random);
    const auto select_equiangular =
        sampling.method ==
        volume_sample_equiangular;
    const auto sampled_distance =
        select(
            distance_sample,
            equiangular_sample.distance,
            select_equiangular);
    const auto distance_pdf =
        select(
            distance_sample_pdf,
            dot(
                bounded_exponential_pdf(
                    equiangular_sample.distance,
                    coefficients.sigma_t,
                    equiangular.interval.minimum,
                    equiangular.interval.maximum),
                channel_sample.pdf),
            select_equiangular);
    const auto equiangular_pdf =
        select(
            direct_sampling.equiangular_pdf(
                ray_origin,
                ray_direction,
                equiangular,
                distance_sample),
            equiangular_sample.pdf,
            select_equiangular);
    const auto selected_pdf =
        select(
            distance_pdf,
            equiangular_pdf,
            select_equiangular);
    const auto competing_pdf =
        select(
            equiangular_pdf,
            distance_pdf,
            select_equiangular);
    const auto mis_weight =
        select(
            1.0f,
            2.0f *
                direct_sampling
                    .power_heuristic(
                        selected_pdf,
                        competing_pdf),
            sampling.use_mis);
    const auto valid =
        scattered &
        (!select_equiangular |
         equiangular_sample.valid) &
        (selected_pdf > 0.0f);
    const auto direct_throughput =
        throughput *
        _safe_divide(
            coefficients.sigma_s *
                transmittance(
                    coefficients.sigma_t,
                    sampled_distance),
            selected_pdf) *
        mis_weight;
    return {
        .transmittance =
            select(
                make_float3(1.0f),
                transmittance(
                    coefficients.sigma_t,
                    sampled_distance),
                valid),
        .throughput =
            select(
                make_float3(0.0f),
                direct_throughput,
                valid),
        .channel_pdf =
            channel_sample.pdf,
        .distance =
            select(
                0.0f,
                sampled_distance,
                valid),
        .distance_pdf =
            select(
                0.0f,
                distance_pdf,
                valid),
        .equiangular_pdf =
            select(
                0.0f,
                equiangular_pdf,
                valid),
        .mis_weight =
            select(
                0.0f,
                mis_weight,
                valid),
        .channel = channel,
        .sample_method =
            sampling.method,
        .use_mis =
            sampling.use_mis,
        .scattered = valid};
}

}// namespace psycles::luisa_backend
