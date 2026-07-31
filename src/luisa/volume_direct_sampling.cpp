#include <psycles/luisa/volume_direct_sampling.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

VolumeDirectSamplingState
VolumeDirectSampling::prepare(
    UInt requested_method,
    Float random,
    Bool enabled) const noexcept {
    const auto use_mis =
        requested_method ==
        volume_sample_mis;
    const auto choose_distance =
        random < 0.5f;
    const auto mis_method =
        select(
            volume_sample_equiangular,
            volume_sample_distance,
            choose_distance);
    const auto mis_random =
        select(
            (random - 0.5f) * 2.0f,
            random * 2.0f,
            choose_distance);
    const auto method =
        select(
            requested_method,
            mis_method,
            use_mis);
    const auto prepared_random =
        select(
            random,
            mis_random,
            use_mis);
    const auto active =
        enabled &
        (requested_method !=
         volume_sample_none);
    return {
        .method = select(
            volume_sample_none,
            method,
            active),
        .random = prepared_random,
        .use_mis = active & use_mis,
        .enabled = active};
}

namespace {

struct EquiangularGeometry {
    Float delta;
    Float perpendicular_distance;
    Float theta_minimum;
    Float theta_maximum;
    Float theta_range;
    Bool valid;
    Bool uniform;
};

[[nodiscard]] EquiangularGeometry
equiangular_geometry(
    Float3 ray_origin,
    Float3 ray_direction,
    const VolumeEquiangularCoefficients
        &coefficients) noexcept {
    const auto light_offset =
        coefficients.light_position -
        ray_origin;
    const auto delta =
        dot(light_offset, ray_direction);
    const auto perpendicular =
        light_offset -
        ray_direction * delta;
    const auto perpendicular_distance =
        length(perpendicular);
    const auto interval_valid =
        coefficients.interval.minimum <
        coefficients.interval.maximum;
    const auto valid =
        interval_valid &
        (perpendicular_distance != 0.0f);
    const auto safe_distance =
        select(
            1.0f,
            perpendicular_distance,
            valid);
    const auto theta_minimum =
        atan2(
            coefficients.interval.minimum -
                delta,
            safe_distance);
    const auto theta_maximum =
        atan2(
            coefficients.interval.maximum -
                delta,
            safe_distance);
    const auto theta_range =
        theta_maximum -
        theta_minimum;
    return {
        .delta = delta,
        .perpendicular_distance =
            perpendicular_distance,
        .theta_minimum = theta_minimum,
        .theta_maximum = theta_maximum,
        .theta_range = theta_range,
        .valid = valid,
        .uniform =
            valid &
            (theta_range < 1.0e-6f)};
}

[[nodiscard]] Float safe_uniform_pdf(
    const VolumeDirectSampleInterval
        &interval) noexcept {
    const auto length =
        interval.maximum -
        interval.minimum;
    const auto valid = length != 0.0f;
    const auto safe_length =
        select(1.0f, length, valid);
    return select(
        0.0f,
        1.0f / safe_length,
        valid);
}

}// namespace

VolumeEquiangularSample
VolumeDirectSampling::sample_equiangular(
    Float3 ray_origin,
    Float3 ray_direction,
    const VolumeEquiangularCoefficients
        &coefficients,
    Float random) const noexcept {
    const auto geometry =
        equiangular_geometry(
            ray_origin,
            ray_direction,
            coefficients);
    const auto uniform_distance =
        coefficients.interval.minimum +
        random *
            (coefficients.interval.maximum -
             coefficients.interval.minimum);
    const auto angle =
        random *
            geometry.theta_maximum +
        (1.0f - random) *
            geometry.theta_minimum;
    const auto offset =
        geometry.perpendicular_distance *
        tan(angle);
    const auto sampled_distance =
        clamp(
            geometry.delta + offset,
            coefficients.interval.minimum,
            coefficients.interval.maximum);
    const auto analytic_denominator =
        geometry.theta_range *
        (geometry.perpendicular_distance *
             geometry.perpendicular_distance +
         offset * offset);
    const auto safe_analytic_denominator =
        select(
            1.0f,
            analytic_denominator,
            analytic_denominator != 0.0f);
    const auto analytic_pdf =
        geometry.perpendicular_distance /
        safe_analytic_denominator;
    const auto pdf =
        select(
            analytic_pdf,
            safe_uniform_pdf(
                coefficients.interval),
            geometry.uniform);
    return {
        .distance = select(
            0.0f,
            select(
                sampled_distance,
                uniform_distance,
                geometry.uniform),
            geometry.valid),
        .pdf =
            select(
                0.0f,
                pdf,
                geometry.valid),
        .valid = geometry.valid};
}

Float VolumeDirectSampling::equiangular_pdf(
    Float3 ray_origin,
    Float3 ray_direction,
    const VolumeEquiangularCoefficients
        &coefficients,
    Float distance) const noexcept {
    const auto geometry =
        equiangular_geometry(
            ray_origin,
            ray_direction,
            coefficients);
    const auto offset =
        distance -
        geometry.delta;
    const auto analytic_denominator =
        geometry.theta_range *
        (geometry.perpendicular_distance *
             geometry.perpendicular_distance +
         offset * offset);
    const auto safe_analytic_denominator =
        select(
            1.0f,
            analytic_denominator,
            analytic_denominator != 0.0f);
    const auto analytic_pdf =
        geometry.perpendicular_distance /
        safe_analytic_denominator;
    const auto pdf =
        select(
            analytic_pdf,
            safe_uniform_pdf(
                coefficients.interval),
            geometry.uniform);
    return select(
        0.0f,
        pdf,
        geometry.valid);
}

Float VolumeDirectSampling::power_heuristic(
    Float selected_pdf,
    Float competing_pdf) const noexcept {
    const auto selected_squared =
        selected_pdf * selected_pdf;
    const auto competing_squared =
        competing_pdf * competing_pdf;
    const auto sum =
        selected_squared +
        competing_squared;
    const auto valid = sum > 0.0f;
    const auto safe_sum =
        select(1.0f, sum, valid);
    return select(
        0.0f,
        selected_squared / safe_sum,
        valid);
}

}// namespace psycles::luisa_backend
