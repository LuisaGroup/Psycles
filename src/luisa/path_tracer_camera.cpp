#include "path_tracer_camera.h"

#include <psycles/luisa/camera_sampling.h>
#include <psycles/luisa/pixel_filter.h>

#include <utility>

namespace psycles::luisa_backend::detail {

CameraDimensionSample sample_camera_dimensions(
    const BufferFloat4 &sobol_table,
    const Var<RenderKernelParameters> &parameters,
    UInt full_x,
    UInt full_y,
    UInt sample_index) noexcept {
    UInt cycles_y = camera_sampling::cycles_pixel_y(
        full_y, parameters.full_height);
    UInt rng_hash = cycles_sampler::pixel_hash(
        full_x,
        cycles_y,
        parameters.seed);
    Float2 filter_sample =
        cycles_sampler::sample_2d(
            sobol_table,
            parameters.sobol_sequence_size,
            sample_index,
            rng_hash,
            UInt{
                tabulated_sobol::
                    camera_filter_dimension});
    filter_sample = select(
        filter_sample,
        make_float2(0.5f),
        sample_index == 0u);
    Float3 lens_time_sample =
        cycles_sampler::sample_3d(
            sobol_table,
            parameters.sobol_sequence_size,
            sample_index,
            rng_hash,
            UInt{
                tabulated_sobol::
                    camera_lens_time_dimension});
    return {
        .cycles_y = cycles_y,
        .rng_hash = rng_hash,
        .filter_sample = filter_sample,
        .lens_time_sample = lens_time_sample};
}

CameraRaySample construct_camera_ray(
    const BufferFloat &filter_table,
    const Var<RenderKernelParameters> &parameters,
    UInt full_x,
    UInt full_y,
    const CameraDimensionSample &sample,
    CameraProjection projection,
    bool depth_of_field,
    std::uint32_t aperture_blades,
    float aperture_rotation,
    const SafeNormalizeCallable &safe_normalize) noexcept {
    const Float jitter_x =
        pixel_filter::sample(
            filter_table,
            sample.filter_sample.x);
    const Float jitter_y =
        camera_sampling::output_filter_y(
            pixel_filter::sample(
                filter_table,
                sample.filter_sample.y));
    const Float width =
        cast<float>(parameters.full_width);
    const Float height =
        cast<float>(parameters.full_height);
    const Float screen_x =
        2.0f *
            (cast<float>(full_x) + jitter_x) /
            width -
        1.0f +
        2.0f * parameters.camera_shift_x;
    const Float screen_y =
        1.0f -
        2.0f *
            (cast<float>(full_y) + jitter_y) /
            height +
        2.0f * parameters.camera_shift_y;
    const Float aspect = width / height;

    Float3 local_origin = make_float3(0.0f);
    Float3 local_direction =
        make_float3(0.0f, 0.0f, -1.0f);
    Float camera_clip_cosine = 1.0f;
    Float3 local_direction_dx = local_direction;
    Float3 local_direction_dy = local_direction;
    Float differential_position = 0.0f;
    Float differential_direction = 0.0f;
    if (projection == CameraProjection::perspective) {
        local_direction = normalize(make_float3(
            screen_x *
                parameters.camera_horizontal_tangent,
            screen_y *
                parameters.camera_vertical_tangent,
            -1.0f));
        camera_clip_cosine = -local_direction.z;
        local_direction_dx =
            normalize(make_float3(
                (screen_x + 2.0f / width) *
                    parameters.camera_horizontal_tangent,
                screen_y *
                    parameters.camera_vertical_tangent,
                -1.0f));
        local_direction_dy =
            normalize(make_float3(
                screen_x *
                    parameters.camera_horizontal_tangent,
                (screen_y - 2.0f / height) *
                    parameters.camera_vertical_tangent,
                -1.0f));
        differential_direction =
            0.5f *
            (length(
                 local_direction_dx -
                 local_direction) +
             length(
                 local_direction_dy -
                 local_direction));
    } else if (
        projection == CameraProjection::orthographic) {
        local_origin = make_float3(
            screen_x *
                parameters.camera_ortho_vertical_span *
                aspect * 0.5f,
            screen_y *
                parameters.camera_ortho_vertical_span *
                0.5f,
            0.0f);
        differential_position =
            0.5f *
            (parameters.camera_ortho_vertical_span *
                 aspect / width +
             parameters.camera_ortho_vertical_span /
                 height);
    } else {
        const Float longitude = screen_x * pi;
        const Float latitude =
            screen_y * pi * 0.5f;
        const Float cosine_latitude =
            cos(latitude);
        local_direction = make_float3(
            cosine_latitude * sin(longitude),
            sin(latitude),
            -cosine_latitude * cos(longitude));
        const Float longitude_dx =
            (screen_x + 2.0f / width) * pi;
        const Float latitude_dy =
            (screen_y - 2.0f / height) *
            pi * 0.5f;
        local_direction_dx = make_float3(
            cosine_latitude * sin(longitude_dx),
            sin(latitude),
            -cosine_latitude * cos(longitude_dx));
        const Float cosine_latitude_dy =
            cos(latitude_dy);
        local_direction_dy = make_float3(
            cosine_latitude_dy * sin(longitude),
            sin(latitude_dy),
            -cosine_latitude_dy * cos(longitude));
        differential_direction =
            0.5f *
            (length(
                 local_direction_dx -
                 local_direction) +
             length(
                 local_direction_dy -
                 local_direction));
    }

    if (depth_of_field) {
        Float2 lens_position =
            camera_sampling::sample_aperture(
                sample.lens_time_sample.yz(),
                aperture_blades,
                aperture_rotation) *
            parameters.camera_aperture_radius;
        lens_position.x /=
            max(
                parameters.camera_aperture_ratio,
                1.0e-5f);
        const Float focus_scale =
            parameters.camera_focal_distance /
            max(-local_direction.z, 1.0e-6f);
        const Float focus_scale_dx =
            parameters.camera_focal_distance /
            max(-local_direction_dx.z, 1.0e-6f);
        const Float focus_scale_dy =
            parameters.camera_focal_distance /
            max(-local_direction_dy.z, 1.0e-6f);
        const Float3 focus_position =
            local_direction * focus_scale;
        const Float3 focus_position_dx =
            local_direction_dx * focus_scale_dx;
        const Float3 focus_position_dy =
            local_direction_dy * focus_scale_dy;
        local_origin = make_float3(
            lens_position.x,
            lens_position.y,
            0.0f);
        local_direction = normalize(
            focus_position - local_origin);
        local_direction_dx = normalize(
            focus_position_dx - local_origin);
        local_direction_dy = normalize(
            focus_position_dy - local_origin);
    }

    Float3 ray_origin =
        (parameters.camera_transform *
         make_float4(local_origin, 1.0f))
            .xyz();
    const Float3 ray_direction = safe_normalize(
        (parameters.camera_transform *
         make_float4(local_direction, 0.0f))
            .xyz(),
        make_float3(0.0f, 0.0f, -1.0f));
    const auto camera_clip =
        camera_sampling::camera_clip_range(
            parameters.camera_near,
            parameters.camera_far,
            camera_clip_cosine);
    ray_origin += ray_direction * camera_clip.x;
    Var<luisa::compute::Ray> ray = make_ray(
        ray_origin,
        ray_direction,
        0.0f,
        camera_clip.y);
    return {
        .ray = std::move(ray),
        .differential_position =
            differential_position,
        .differential_direction =
            differential_direction};
}

}// namespace psycles::luisa_backend::detail
