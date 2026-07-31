#include <psycles/luisa/volume_analytic_light_sampling.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

namespace sampling =
    analytic_light_sampling;

sampling::FiniteLightSample
VolumeAnalyticLightSampling::point(
    const VolumePointLightSampleInput
        &input) const noexcept {
    return sampling::sample_point_light(
        input.reference,
        make_float3(0.0f),
        true,
        input.center,
        input.radius,
        input.sphere,
        input.axis_x,
        input.axis_y,
        input.axis_z,
        input.axis_scale,
        input.random,
        input.normalize_power);
}

sampling::FiniteLightSample
VolumeAnalyticLightSampling::
    spot_from_segment(
        const VolumeSpotLightSampleInput
            &input) const noexcept {
    const auto &point = input.point;
    const auto center_offset =
        point.reference -
        point.center;
    const auto center_distance_squared =
        dot(center_offset, center_offset);
    const auto radius_squared =
        point.radius * point.radius;
    const auto light_normal =
        sampling::safe_normalize(
            center_offset,
            make_float3(
                0.0f, 0.0f, -1.0f));
    const auto sphere_cap =
        cycles_sample_mapping::
            sin_squared_to_one_minus_cosine(
                radius_squared /
                max(
                    center_distance_squared,
                    1.0e-30f));

    sampling::FiniteLightGeometrySample
        geometry{
            .valid = false,
            .direction =
                make_float3(0.0f),
            .position = point.center,
            .normal =
                make_float3(0.0f),
            .distance = 0.0f,
            .conditional_pdf = 0.0f};
    const auto finite_sphere =
        point.sphere &
        (point.radius > 0.0f);
    $if(finite_sphere) {
        // spot_light_sample<true> always samples the visible sphere cap when
        // outside. The surface-only spread-cone alternative is not part of
        // the segment proposal measure.
        geometry =
            sampling::
                sample_sphere_geometry(
                    point.reference,
                    make_float3(0.0f),
                    true,
                    point.center,
                    point.radius,
                    -light_normal,
                    sphere_cap,
                    false,
                    point.random);
    }
    $else {
        geometry =
            sampling::
                sample_disk_geometry(
                    point.reference,
                    point.center,
                    point.radius,
                    point.random);
    };

    const auto transform =
        sampling::
            light_linear_transform(
                point.axis_x,
                point.axis_y,
                point.axis_z,
                point.axis_scale);
    const auto local_ray =
        sampling::spot_light_to_local(
            -geometry.direction,
            transform);
    const auto attenuation =
        sampling::
            spot_light_attenuation(
                local_ray,
                input.spot_angle,
                input.spot_smooth);
    const auto use_attenuation =
        !finite_sphere |
        (center_distance_squared >
         radius_squared);
    const auto evaluation_factor =
        sampling::point_eval_factor(
            point.radius,
            point.normalize_power) *
        select(
            1.0f,
            attenuation,
            use_attenuation);

    return {
        // Cycles keeps an otherwise valid proposal even when attenuation is
        // zero. The clipped segment determines whether a direct collision
        // can receive energy; final sampling performs the radiometric reject.
        .valid = geometry.valid,
        .direction = geometry.direction,
        .position = geometry.position,
        .normal = geometry.normal,
        .uv =
            sampling::spot_light_uv(
                local_ray,
                input.spot_angle),
        .distance = geometry.distance,
        .conditional_pdf =
            geometry.conditional_pdf,
        .evaluation_factor =
            evaluation_factor};
}

sampling::FiniteLightSample
VolumeAnalyticLightSampling::
    spot_from_position(
        const VolumeSpotLightSampleInput
            &input) const noexcept {
    const auto &point = input.point;
    return sampling::sample_spot_light(
        point.reference,
        make_float3(0.0f),
        true,
        point.center,
        point.radius,
        point.sphere,
        point.axis_x,
        point.axis_y,
        point.axis_z,
        point.axis_scale,
        input.spot_angle,
        input.spot_smooth,
        point.random,
        point.normalize_power);
}

}// namespace psycles::luisa_backend
