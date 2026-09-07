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
    const Float jitter_x = pixel_filter::sample(filter_table, sample.filter_sample.x);
    const Float jitter_y = pixel_filter::sample(filter_table, sample.filter_sample.y);
    Float3 origin = make_float3(0.0f);
    Float3 direction = make_float3(0.0f, 0.0f, 1.0f);
    Float dP = 0.0f, dD = 0.0f;
    Float tmax = parameters.camera_far - parameters.camera_near;

    if (projection != CameraProjection::panorama) {
        // Cycles uses bottom-left raster coordinates directly, then the
        // host-prepared projective transform. Keep Pcamera unnormalized for
        // focus-plane construction and for the independent ray differentials.
        const auto raster = make_float3(
            cast<float>(full_x) + jitter_x,
            cast<float>(sample.cycles_y) + jitter_y, 0.0f);
        const auto Pcamera = cycles_transform::perspective(
            parameters.camera_raster_to_camera, raster);
        const auto t = parameters.camera_transform;
        const auto camera_to_world = make_float4x4(t[0u], t[1u], -t[2u], t[3u]);
        Float2 lens = make_float2(0.0f);
        if (depth_of_field) {
            lens = camera_sampling::sample_aperture(
                sample.lens_time_sample.yz(), aperture_blades < 3u ? 0u : aperture_blades,
                aperture_rotation);
            lens.x *= parameters.camera_inv_aperture_ratio;
            lens *= parameters.camera_aperture_radius;
        }
        if (projection == CameraProjection::perspective) {
            direction = Pcamera;
            if (depth_of_field) {
                const auto focus = direction * (parameters.camera_focal_distance / direction.z);
                origin = make_float3(lens, 0.0f);
                direction = normalize(focus - origin);
            }
            origin = cycles_transform::point(camera_to_world, origin);
            direction = normalize(cycles_transform::direction(camera_to_world, direction));
            const auto center = cycles_transform::direction(camera_to_world, Pcamera);
            const auto normalized_center = normalize(center);
            dD = 0.5f * (length(normalize(center + parameters.camera_dx) - normalized_center) +
                         length(normalize(center + parameters.camera_dy) - normalized_center));
            const auto z_inv = 1.0f / normalize(Pcamera).z;
            const auto near = parameters.camera_near * z_inv;
            origin += near * direction;
            dP += near * dD;
            tmax *= z_inv;
        } else {
            if (depth_of_field) {
                const auto focus = direction * parameters.camera_focal_distance;
                const auto lens_position = make_float3(lens, 0.0f);
                direction = normalize(focus - lens_position);
                origin = Pcamera + lens_position +
                         direction * (parameters.camera_near / direction.z);
            } else {
                origin = Pcamera + make_float3(0.0f, 0.0f, parameters.camera_near);
            }
            origin = cycles_transform::point(camera_to_world, origin);
            direction = normalize(cycles_transform::direction(camera_to_world, direction));
            dP = 0.5f * (length(parameters.camera_dx) + length(parameters.camera_dy));
        }
    } else {
        // Existing equirectangular path; stereo/motion/panorama variants are
        // outside the static rectilinear projection contract above.
        const Float width = cast<float>(parameters.full_width);
        const Float height = cast<float>(parameters.full_height);
        const Float screen_x = 2.0f * (cast<float>(full_x) + jitter_x) / width -
                               1.0f + 2.0f * parameters.camera_shift_x;
        const Float screen_y = 1.0f -
            2.0f * (cast<float>(full_y) + camera_sampling::output_filter_y(jitter_y)) / height +
            2.0f * parameters.camera_shift_y;
        const auto longitude = screen_x * pi;
        const auto latitude = screen_y * pi * 0.5f;
        const auto cosine_latitude = cos(latitude);
        const auto local = make_float3(cosine_latitude * sin(longitude), sin(latitude),
                                       -cosine_latitude * cos(longitude));
        const auto longitude_dx = (screen_x + 2.0f / width) * pi;
        const auto latitude_dy = (screen_y - 2.0f / height) * pi * 0.5f;
        const auto dx = make_float3(cosine_latitude * sin(longitude_dx), sin(latitude),
                                    -cosine_latitude * cos(longitude_dx));
        const auto dy = make_float3(cos(latitude_dy) * sin(longitude), sin(latitude_dy),
                                    -cos(latitude_dy) * cos(longitude));
        dD = 0.5f * (length(dx - local) + length(dy - local));
        const auto ray = camera_sampling::camera_to_world_ray(
            parameters.camera_transform, make_float3(0.0f), local);
        direction = safe_normalize(ray.direction, make_float3(0.0f, 0.0f, -1.0f));
        origin = ray.origin + parameters.camera_near * direction;
        dP = parameters.camera_near * dD;
    }
    auto ray = make_ray(origin, direction, 0.0f, tmax);
    return {std::move(ray), dP, dD};
}

}// namespace psycles::luisa_backend::detail
