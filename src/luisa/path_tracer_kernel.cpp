#include "path_tracer_internal.h"
#include "cycles_filter_glossy.h"
#include "cycles_integrator_limits.h"
#include "path_tracer_environment.h"
#include "path_tracer_geometry.h"
#include "path_tracer_light_distribution.h"
#include "path_tracer_lighting.h"
#include "path_tracer_surfaces.h"

#include <psycles/luisa/analytic_light_sampling.h>
#include <psycles/luisa/background_sampling.h>
#include <psycles/luisa/camera_sampling.h>
#include <psycles/luisa/pixel_filter.h>
#include <psycles/luisa/spherical_geometry.h>
#include <psycles/luisa/surface_ray.h>

#include <psycles/sampling/light_distribution.h>

namespace psycles::luisa_backend::detail {

void LuisaRenderSession::initialize(const RenderSettings &settings) {
    _settings = settings;
    _total_aa_samples = 0u;
    _path_trace_delivered = false;
    _window = effective_window(settings);
    const auto count = std::max<std::size_t>(pixel_count(), 1u);
    _combined =
        _scene->device.create_buffer<luisa::float4>(count);
    _normal =
        _scene->device.create_buffer<luisa::float4>(count);
    _albedo =
        _scene->device.create_buffer<luisa::float4>(count);
    _light_passes =
        _scene->device.create_buffer<luisa::float4>(
            count * light_pass_buffer_count);
    _sample_count =
        _scene->device.create_buffer<luisa::uint>(count);
    _path_trace =
        _scene->device.create_buffer<luisa::float4>(
            _options.path_trace
                ? path_trace_schema::slot_count
                : 1u);
    const auto generated_filter_table =
        sampling::make_pixel_filter_table(
            settings.pixel_filter,
            settings.filter_width);
    luisa::vector<float> filter_table;
    filter_table.reserve(
        generated_filter_table.size());
    for (const auto value :
         generated_filter_table) {
        filter_table.emplace_back(value);
    }
    _pixel_filter_table =
        _scene->device.create_buffer<float>(
            filter_table.size());

    luisa::vector<luisa::float4> zeros_float(count);
    luisa::vector<luisa::float4> zeros_light_passes(
        count * light_pass_buffer_count);
    luisa::vector<luisa::uint> zeros_uint(count);
    luisa::vector<luisa::float4> zeros_path_trace(
        _options.path_trace
            ? path_trace_schema::slot_count
            : 1u);
    _stream << _combined.copy_from(luisa::span{zeros_float})
            << _normal.copy_from(luisa::span{zeros_float})
            << _albedo.copy_from(luisa::span{zeros_float})
            << _light_passes.copy_from(
                   luisa::span{zeros_light_passes})
            << _sample_count.copy_from(luisa::span{zeros_uint})
            << _path_trace.copy_from(
                   luisa::span{zeros_path_trace})
            << _pixel_filter_table.copy_from(
                   luisa::span{filter_table})
            << synchronize();

    auto scene = _scene;
    const auto render_settings = _settings;
    const auto render_window = _window;
    const auto integrator = render_settings.integrator;
    const auto bounce_limits =
        cycles_kernel_bounce_limits({
            .maximum = integrator.max_bounces,
            .minimum = integrator.min_bounces,
            .maximum_diffuse =
                integrator.diffuse_bounces,
            .maximum_glossy =
                integrator.glossy_bounces,
            .maximum_transmission =
                integrator.transmission_bounces,
            .transparent_minimum =
                integrator.transparent_min_bounces,
            .transparent_maximum =
                integrator.transparent_max_bounces});
    const auto max_bounces = bounce_limits.maximum;
    const auto min_bounces = bounce_limits.minimum;
    const auto max_diffuse_bounces =
        bounce_limits.maximum_diffuse;
    const auto max_glossy_bounces =
        bounce_limits.maximum_glossy;
    const auto max_transmission_bounces =
        bounce_limits.maximum_transmission;
    const auto transparent_min_bounces =
        bounce_limits.transparent_minimum;
    const auto transparent_max_bounces =
        bounce_limits.transparent_maximum;
    // Blender exposes clamp per RGB channel. Cycles' device kernel
    // compares the sum of absolute RGB components, so scene sync
    // multiplies non-zero UI values by three.
    const auto sample_clamp_direct =
        integrator.sample_clamp_direct > 0.0f
            ? integrator.sample_clamp_direct * 3.0f
            : 0.0f;
    const auto sample_clamp_indirect =
        integrator.sample_clamp_indirect > 0.0f
            ? integrator.sample_clamp_indirect * 3.0f
            : 0.0f;
    const auto filter_glossy =
        cycles_filter_glossy_device_scale(
            integrator.filter_glossy);
    const auto light_inv_rr_threshold =
        !integrator.use_light_tree &&
                integrator.light_sampling_threshold > 0.0f
            ? integrator.film_exposure /
                  integrator.light_sampling_threshold
            : 0.0f;
    const auto reflective_caustics =
        integrator.reflective_caustics;
    const auto refractive_caustics =
        integrator.refractive_caustics;
    const auto max_path_steps =
        bounce_limits.maximum_path_steps;
    const auto next_event_estimation =
        _options.next_event_estimation &&
        integrator.direct_light_sampling !=
            contract::DirectLightSampling::
                forward_path_tracing;
    const auto direct_light_sampling =
        next_event_estimation
            ? integrator.direct_light_sampling
            : contract::DirectLightSampling::
                  forward_path_tracing;
    const auto camera_transform = to_luisa(scene->camera.transform);
    const auto camera_projection = scene->camera.projection;
    const auto camera_aspect =
        static_cast<float>(std::max(
            render_settings.full_extent.width, 1u)) /
        static_cast<float>(std::max(
            render_settings.full_extent.height, 1u));
    const auto camera_horizontal_fit =
        scene->camera.sensor_fit ==
            CameraSensorFit::horizontal ||
        (scene->camera.sensor_fit ==
             CameraSensorFit::automatic &&
         camera_aspect >= 1.0f);
    const auto camera_vertical_tangent =
        camera_horizontal_fit
            ? std::tan(
                  scene->camera.horizontal_field_of_view *
                  0.5f) /
                  camera_aspect
            : std::tan(
                  scene->camera.field_of_view * 0.5f);
    const auto camera_horizontal_tangent =
        camera_vertical_tangent * camera_aspect;
    const auto camera_ortho_scale =
        scene->camera.orthographic_scale;
    const auto camera_shift_x =
        scene->camera.lens_shift_x *
        (camera_horizontal_fit
             ? 1.0f
             : 1.0f / camera_aspect);
    const auto camera_shift_y =
        scene->camera.lens_shift_y *
        (camera_horizontal_fit
             ? camera_aspect
             : 1.0f);
    const auto camera_near = scene->camera.near_clip;
    const auto camera_far = scene->camera.far_clip;
    const auto camera_aperture_radius =
        scene->camera.aperture_radius;
    const auto camera_focal_distance =
        scene->camera.focal_distance;
    const auto camera_aperture_ratio =
        scene->camera.aperture_ratio;
    const auto camera_aperture_blades = scene->camera.aperture_blades;
    const auto camera_aperture_rotation =
        scene->camera.aperture_rotation;
    const auto background = scene->background;
    const auto camera_depth_of_field =
        camera_projection == CameraProjection::perspective &&
        camera_aperture_radius > 0.0f &&
        camera_focal_distance > 0.0f;
    const auto pass_alpha_threshold = std::clamp(
        render_settings.pass_alpha_threshold,
        0.0f,
        1.0f);
    _kernel_parameters = RenderKernelParameters{
        .window_x = render_window.x,
        .window_y = render_window.y,
        .window_width =
            std::max(render_window.width, 1u),
        .full_width = std::max(
            render_settings.full_extent.width, 1u),
        .full_height = std::max(
            render_settings.full_extent.height, 1u),
        .seed = render_settings.seed,
        .sobol_sequence_size = 0u,
        .max_bounces = max_bounces,
        .min_bounces = min_bounces,
        .max_diffuse_bounces = max_diffuse_bounces,
        .max_glossy_bounces = max_glossy_bounces,
        .max_transmission_bounces =
            max_transmission_bounces,
        .transparent_min_bounces =
            transparent_min_bounces,
        .transparent_max_bounces =
            transparent_max_bounces,
        .max_path_steps = max_path_steps,
        .transparent_background =
            render_settings.transparent_background ? 1u : 0u,
        .path_trace_enabled =
            _options.path_trace ? 1u : 0u,
        .path_trace_pixel_x =
            _options.path_trace
                ? _options.path_trace->pixel_x
                : 0u,
        .path_trace_pixel_y =
            _options.path_trace
                ? _options.path_trace->pixel_y
                : 0u,
        .path_trace_sample =
            _options.path_trace
                ? _options.path_trace->sample
                : 0u,
        .sample_clamp_direct = sample_clamp_direct,
        .sample_clamp_indirect = sample_clamp_indirect,
        .filter_glossy = filter_glossy,
        .light_inv_rr_threshold =
            light_inv_rr_threshold,
        .camera_horizontal_tangent =
            camera_horizontal_tangent,
        .camera_vertical_tangent =
            camera_vertical_tangent,
        .camera_ortho_scale = camera_ortho_scale,
        .camera_shift_x = camera_shift_x,
        .camera_shift_y = camera_shift_y,
        .camera_near = camera_near,
        .camera_far = camera_far,
        .camera_aperture_radius =
            camera_aperture_radius,
        .camera_focal_distance =
            camera_focal_distance,
        .camera_aperture_ratio =
            camera_aperture_ratio,
        .pass_alpha_threshold =
            pass_alpha_threshold,
        .background = background,
        .camera_transform = camera_transform};
    auto light_transport =
        make_light_transport_callables(
            direct_light_sampling);
    auto safe_normalize =
        light_transport.safe_normalize;
    auto forward_light_weight =
        light_transport.forward_light_weight;
    auto nee_light_weight =
        light_transport.nee_light_weight;
    auto clamp_light_contribution =
        light_transport.clamp_light_contribution;
    auto light_sample_roulette_weight =
        light_transport.light_sample_roulette_weight;
    auto light_component_ratio =
        light_transport.light_component_ratio;
    auto split_scattered_light =
        light_transport.split_scattered_light;
    auto split_nee_light =
        light_transport.split_nee_light;
    auto emissive_triangle_pdf_callable =
        make_emissive_triangle_pdf_callable(scene);
    auto light_distribution_sample_callable =
        make_light_distribution_sample_callable(scene);

    auto surface_callables =
        make_surface_callables(scene);
    auto shared_surface_evaluate =
        surface_callables.evaluate;
    auto shared_surface_emission =
        surface_callables.emission;
    auto shared_surface_sample =
        surface_callables.sample;
    auto shared_surface_aov =
        surface_callables.aov;
    auto shared_surface_shading_normal =
        surface_callables.shading_normal;
    auto environment_callables =
        make_environment_callables(
            scene,
            safe_normalize,
            shared_surface_emission);
    auto environment_base_callable =
        environment_callables.base;
    auto environment_sun_callables =
        environment_callables.suns;
    auto nishita_sun_callable =
        environment_callables.nishita_sun;
    auto trace_shadow_callable =
        make_trace_shadow_callable(
            scene, safe_normalize);
    const auto path_trace_enabled =
        _options.path_trace.has_value();

    Kernel1D kernel = [
                          =,
                          &shared_surface_evaluate,
                          &shared_surface_emission,
                          &shared_surface_sample,
                          &shared_surface_aov,
                          &shared_surface_shading_normal,
                          &light_component_ratio,
                          &split_scattered_light,
                          &split_nee_light](
                          BufferFloat4 combined,
                          BufferFloat4 normal,
                          BufferFloat4 albedo,
                          BufferFloat4 light_passes,
                          BufferUInt sample_count,
                          BufferFloat4 path_trace,
                          UInt sample_first,
                          UInt samples,
                          BufferFloat4 sobol_table,
                          BufferFloat filter_table,
                          Var<RenderKernelParameters>
                              kernel_parameters) noexcept {
        UInt pixel = dispatch_x();
        UInt local_x =
            pixel % kernel_parameters.window_width;
        UInt local_y =
            pixel / kernel_parameters.window_width;
        UInt full_x =
            local_x + kernel_parameters.window_x;
        UInt full_y =
            local_y + kernel_parameters.window_y;
        auto clamp_contribution =
            [&](Float3 contribution,
                UInt depth) noexcept {
                return clamp_light_contribution(
                    contribution,
                    depth,
                    kernel_parameters.sample_clamp_direct,
                    kernel_parameters
                        .sample_clamp_indirect);
            };
        auto sample_light_roulette =
            [&](Float3 unshadowed_contribution,
                Float random) noexcept {
                return light_sample_roulette_weight(
                    unshadowed_contribution,
                    random,
                    kernel_parameters
                        .light_inv_rr_threshold);
            };
        Float4 combined_sum = combined.read(pixel);
        Float4 normal_sum = normal.read(pixel);
        Float4 albedo_sum = albedo.read(pixel);
        const auto light_pass_base =
            pixel * light_pass_buffer_count;
        Float4 diffuse_direct_sum = light_passes.read(
            light_pass_base +
            light_pass_index(
                LightPassBuffer::diffuse_direct));
        Float4 diffuse_indirect_sum = light_passes.read(
            light_pass_base +
            light_pass_index(
                LightPassBuffer::diffuse_indirect));
        Float4 glossy_direct_sum = light_passes.read(
            light_pass_base +
            light_pass_index(
                LightPassBuffer::glossy_direct));
        Float4 glossy_indirect_sum = light_passes.read(
            light_pass_base +
            light_pass_index(
                LightPassBuffer::glossy_indirect));
        Float4 transmission_direct_sum = light_passes.read(
            light_pass_base +
            light_pass_index(
                LightPassBuffer::transmission_direct));
        Float4 transmission_indirect_sum = light_passes.read(
            light_pass_base +
            light_pass_index(
                LightPassBuffer::transmission_indirect));
        Float4 emission_sum = light_passes.read(
            light_pass_base +
            light_pass_index(LightPassBuffer::emission));
        Float4 environment_sum = light_passes.read(
            light_pass_base +
            light_pass_index(LightPassBuffer::environment));
        Float4 glossy_color_sum = light_passes.read(
            light_pass_base +
            light_pass_index(LightPassBuffer::glossy_color));
        Float4 transmission_color_sum = light_passes.read(
            light_pass_base +
            light_pass_index(
                LightPassBuffer::transmission_color));
        UInt completed = sample_count.read(pixel);

        auto evaluate_surface =
            [&](UInt surface_tag,
                const SurfacePoint &point,
                Float3 outgoing,
                const SurfaceQuery &query) noexcept {
                return unpack_surface_evaluation(
                    shared_surface_evaluate(
                        scene->parameter_buffer,
                        scene->cycles_bsdf_table_buffer,
                        scene->texture_heap,
                        scene->heap,
                        surface_tag,
                        pack_surface_point(point),
                        outgoing,
                        query.lobe_mask,
                        query.transport_mode,
                        query.glossy_filter_roughness));
            };
        auto surface_emission =
            [&](UInt surface_tag,
                const SurfacePoint &point,
                Float3 outgoing) noexcept {
                return shared_surface_emission(
                    scene->parameter_buffer,
                    scene->cycles_bsdf_table_buffer,
                    scene->texture_heap,
                    scene->heap,
                    surface_tag,
                    pack_surface_point(point),
                    outgoing);
            };
        auto sample_surface =
            [&](UInt surface_tag,
                const SurfacePoint &point,
                Float u_lobe,
                Float2 u_direction,
                const SurfaceQuery &query) noexcept {
                return unpack_surface_sample(
                    shared_surface_sample(
                        scene->parameter_buffer,
                        scene->cycles_bsdf_table_buffer,
                        scene->texture_heap,
                        scene->heap,
                        surface_tag,
                        pack_surface_point(point),
                        u_lobe,
                        u_direction,
                        query.lobe_mask,
                        query.transport_mode,
                        query.glossy_filter_roughness));
            };
        auto surface_aov =
            [&](UInt surface_tag,
                const SurfacePoint &point) noexcept {
                return unpack_surface_aov(
                    shared_surface_aov(
                        scene->parameter_buffer,
                        scene->cycles_bsdf_table_buffer,
                        scene->texture_heap,
                        scene->heap,
                        surface_tag,
                        pack_surface_point(point)));
            };
        auto surface_shading_normal =
            [&](UInt surface_tag,
                const SurfacePoint &point) noexcept {
                return shared_surface_shading_normal(
                    scene->parameter_buffer,
                    scene->cycles_bsdf_table_buffer,
                    scene->texture_heap,
                    scene->heap,
                    surface_tag,
                    pack_surface_point(point));
            };
        SurfaceQuery surface_query{
            .lobe_mask =
                static_cast<std::uint32_t>(
                    contract::event_diffuse |
                    contract::event_glossy |
                    contract::event_singular |
                    contract::event_reflection |
                    contract::event_transmission |
                    contract::event_transparent),
            .transport_mode =
                static_cast<std::uint32_t>(
                    contract::TransportMode::radiance),
            .glossy_filter_roughness = 0.0f};
        auto evaluate_environment =
            [&](Float3 direction,
                UInt visibility) noexcept {
                Float3 result =
                    environment_base_callable(
                        direction,
                        kernel_parameters.background,
                        visibility);
                for (const auto &sun :
                     environment_sun_callables) {
                    result += sun(direction);
                }
                result +=
                    nishita_sun_callable(direction);
                return result;
            };
        auto trace_shadow =
            trace_shadow_callable;
        auto emissive_triangle_pdf =
            emissive_triangle_pdf_callable;

        $for (sample_offset, samples) {
            UInt sample_index =
                sample_first + sample_offset;
            UInt cycles_y = camera_sampling::cycles_pixel_y(
                full_y,
                kernel_parameters.full_height);
            UInt rng_hash = cycles_sampler::pixel_hash(
                full_x,
                cycles_y,
                kernel_parameters.seed);
            Float2 filter_sample =
                cycles_sampler::sample_2d(
                    sobol_table,
                    kernel_parameters.sobol_sequence_size,
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
                    kernel_parameters.sobol_sequence_size,
                    sample_index,
                    rng_hash,
                    UInt{
                        tabulated_sobol::
                            camera_lens_time_dimension});
            Bool path_trace_active = false;
            if (path_trace_enabled) {
                path_trace_active =
                    (kernel_parameters.path_trace_enabled !=
                     0u) &
                    (full_x ==
                     kernel_parameters.path_trace_pixel_x) &
                    (cycles_y ==
                     kernel_parameters.path_trace_pixel_y) &
                    (sample_index ==
                     kernel_parameters.path_trace_sample);
            }
            auto trace_uint32 =
                [](UInt value) noexcept {
                    return make_float3(
                        cast<float>(value & 0xffffu),
                        cast<float>(value >> 16u),
                        0.0f);
                };
            auto trace_write =
                [&](UInt slot, Float3 value) noexcept {
                    if (path_trace_enabled) {
                        $if (path_trace_active) {
                            path_trace.write(
                                slot,
                                make_float4(value, 1.0f));
                        };
                    }
                };
            auto trace_write_global =
                [&](path_trace_schema::GlobalSlot slot,
                    Float3 value) noexcept {
                    trace_write(
                        UInt{path_trace_schema::index(slot)},
                        value);
                };
            auto trace_write_event =
                [&](UInt event,
                    path_trace_schema::EventSlot slot,
                    Float3 value) noexcept {
                    $if (
                        event <
                        path_trace_schema::max_events) {
                        trace_write(
                            path_trace_schema::
                                global_slot_count +
                                event *
                                    path_trace_schema::
                                        event_slot_count +
                                static_cast<std::uint32_t>(
                                    slot),
                            value);
                    };
                };
            Float jitter_x =
                pixel_filter::sample(
                    filter_table,
                    filter_sample.x);
            Float jitter_y = camera_sampling::output_filter_y(
                pixel_filter::sample(
                    filter_table,
                    filter_sample.y));
            Float width =
                cast<float>(kernel_parameters.full_width);
            Float height =
                cast<float>(kernel_parameters.full_height);
            Float screen_x =
                2.0f *
                    (cast<float>(full_x) + jitter_x) /
                    width -
                1.0f +
                2.0f *
                    kernel_parameters.camera_shift_x;
            Float screen_y =
                1.0f -
                2.0f *
                    (cast<float>(full_y) + jitter_y) /
                    height +
                2.0f *
                    kernel_parameters.camera_shift_y;
            Float aspect = width / height;

            Float3 local_origin = make_float3(0.0f);
            Float3 local_direction =
                make_float3(0.0f, 0.0f, -1.0f);
            Float camera_clip_cosine = 1.0f;
            Float3 local_direction_dx = local_direction;
            Float3 local_direction_dy = local_direction;
            Float ray_dP = 0.0f;
            Float ray_dD = 0.0f;
            if (camera_projection ==
                CameraProjection::perspective) {
                local_direction = normalize(make_float3(
                    screen_x *
                        kernel_parameters
                            .camera_horizontal_tangent,
                    screen_y *
                        kernel_parameters
                            .camera_vertical_tangent,
                    -1.0f));
                camera_clip_cosine = -local_direction.z;
                local_direction_dx =
                    normalize(make_float3(
                        (screen_x + 2.0f / width) *
                            kernel_parameters
                                .camera_horizontal_tangent,
                        screen_y *
                            kernel_parameters
                                .camera_vertical_tangent,
                        -1.0f));
                local_direction_dy =
                    normalize(make_float3(
                        screen_x *
                            kernel_parameters
                                .camera_horizontal_tangent,
                        (screen_y - 2.0f / height) *
                            kernel_parameters
                                .camera_vertical_tangent,
                        -1.0f));
                ray_dD =
                    0.5f *
                    (length(
                         local_direction_dx -
                         local_direction) +
                     length(
                         local_direction_dy -
                         local_direction));
            } else if (
                camera_projection ==
                CameraProjection::orthographic) {
                local_origin = make_float3(
                    screen_x *
                        kernel_parameters
                            .camera_ortho_scale *
                        aspect * 0.5f,
                    screen_y *
                        kernel_parameters
                            .camera_ortho_scale *
                        0.5f,
                    0.0f);
                ray_dP =
                    0.5f *
                    (kernel_parameters.camera_ortho_scale *
                         aspect / width +
                     kernel_parameters.camera_ortho_scale /
                         height);
            } else {
                Float longitude = screen_x * pi;
                Float latitude = screen_y * pi * 0.5f;
                Float cosine_latitude = cos(latitude);
                local_direction = make_float3(
                    cosine_latitude * sin(longitude),
                    sin(latitude),
                    -cosine_latitude * cos(longitude));
                Float longitude_dx =
                    (screen_x + 2.0f / width) * pi;
                Float latitude_dy =
                    (screen_y - 2.0f / height) *
                    pi * 0.5f;
                local_direction_dx =
                    make_float3(
                        cosine_latitude *
                            sin(longitude_dx),
                        sin(latitude),
                        -cosine_latitude *
                            cos(longitude_dx));
                Float cosine_latitude_dy =
                    cos(latitude_dy);
                local_direction_dy =
                    make_float3(
                        cosine_latitude_dy *
                            sin(longitude),
                        sin(latitude_dy),
                        -cosine_latitude_dy *
                            cos(longitude));
                ray_dD =
                    0.5f *
                    (length(
                         local_direction_dx -
                         local_direction) +
                     length(
                         local_direction_dy -
                         local_direction));
            }
            if (camera_depth_of_field) {
                Float2 lens_position =
                    camera_sampling::sample_aperture(
                        lens_time_sample.yz(),
                        camera_aperture_blades,
                        camera_aperture_rotation) *
                    kernel_parameters
                        .camera_aperture_radius;
                lens_position.x /=
                    max(
                        kernel_parameters
                            .camera_aperture_ratio,
                        1.0e-5f);
                Float focus_scale =
                    kernel_parameters
                        .camera_focal_distance /
                    max(-local_direction.z, 1.0e-6f);
                Float3 focus_position =
                    local_direction * focus_scale;
                Float focus_scale_dx =
                    kernel_parameters
                        .camera_focal_distance /
                    max(-local_direction_dx.z, 1.0e-6f);
                Float focus_scale_dy =
                    kernel_parameters
                        .camera_focal_distance /
                    max(-local_direction_dy.z, 1.0e-6f);
                Float3 focus_position_dx =
                    local_direction_dx * focus_scale_dx;
                Float3 focus_position_dy =
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
                (kernel_parameters.camera_transform *
                 make_float4(local_origin, 1.0f))
                    .xyz();
            Float3 ray_direction = safe_normalize(
                (kernel_parameters.camera_transform *
                 make_float4(local_direction, 0.0f))
                    .xyz(),
                make_float3(0.0f, 0.0f, -1.0f));
            auto camera_clip =
                camera_sampling::camera_clip_range(
                    kernel_parameters.camera_near,
                    kernel_parameters.camera_far,
                    camera_clip_cosine);
            ray_origin += ray_direction * camera_clip.x;
            Var<luisa::compute::Ray> ray = make_ray(
                ray_origin,
                ray_direction,
                0.0f,
                camera_clip.y);
            if (path_trace_enabled) {
                trace_write_global(
                    path_trace_schema::GlobalSlot::header,
                    make_float3(
                        static_cast<float>(
                            path_trace_schema::version),
                        cast<float>(full_x),
                        cast<float>(cycles_y)));
                trace_write_global(
                    path_trace_schema::GlobalSlot::rng,
                    make_float3(
                        cast<float>(sample_index),
                        cast<float>(rng_hash & 0xffffu),
                        cast<float>(rng_hash >> 16u)));
                trace_write_global(
                    path_trace_schema::GlobalSlot::filter,
                    make_float3(filter_sample, 0.0f));
                trace_write_global(
                    path_trace_schema::GlobalSlot::lens_time,
                    camera_depth_of_field
                        ? lens_time_sample
                        : make_float3(0.0f));
                trace_write_global(
                    path_trace_schema::GlobalSlot::ray_p,
                    ray->origin());
                trace_write_global(
                    path_trace_schema::GlobalSlot::ray_d,
                    ray->direction());
                trace_write_global(
                    path_trace_schema::GlobalSlot::ray_range,
                    make_float3(
                        ray->t_min(),
                        ray->t_max(),
                        0.5f));
            }
            UInt ray_source_instance =
                surface_ray::invalid_primitive;
            UInt ray_source_primitive =
                surface_ray::invalid_primitive;
            UInt ray_visibility = camera_visibility;

            Float3 radiance = make_float3(0.0f);
            Float3 throughput = make_float3(1.0f);
            Float3 sample_normal = make_float3(0.0f);
            Float3 sample_albedo = make_float3(0.0f);
            Float3 sample_glossy_color =
                make_float3(0.0f);
            Float3 sample_transmission_color =
                make_float3(0.0f);
            Float3 sample_diffuse_direct =
                make_float3(0.0f);
            Float3 sample_diffuse_indirect =
                make_float3(0.0f);
            Float3 sample_glossy_direct =
                make_float3(0.0f);
            Float3 sample_glossy_indirect =
                make_float3(0.0f);
            Float3 sample_transmission_direct =
                make_float3(0.0f);
            Float3 sample_transmission_indirect =
                make_float3(0.0f);
            Float3 sample_emission = make_float3(0.0f);
            Float3 sample_environment =
                make_float3(0.0f);
            Float sample_alpha = select(
                1.0f,
                0.0f,
                kernel_parameters.transparent_background !=
                    0u);
            Bool primary_recorded = false;
            Float previous_bsdf_pdf = 0.0f;
            Float minimum_bsdf_pdf =
                std::numeric_limits<float>::max();
            Bool previous_delta = true;
            Float continuation_probability = 1.0f;
            Float3 path_diffuse_weight =
                make_float3(0.0f);
            Float3 path_glossy_weight =
                make_float3(0.0f);
            UInt ray_events = 0u;
            UInt diffuse_depth = 0u;
            UInt glossy_depth = 0u;
            UInt transparent_depth = 0u;
            UInt transmission_depth = 0u;
            UInt path_depth = 0u;
            // By the time Cycles records a surface event, its primary data
            // pass has marked PATH_RAY_SINGLE_PASS_DONE.
            UInt path_flags =
                (1u << 7u) | (1u << 8u) | (1u << 9u);
            Bool terminate_after_transparent = false;
            Bool terminate_on_next_surface = false;
            auto accumulate_light_pass =
                [&](Var<LightPassContributionCall>
                        contribution) noexcept {
                    sample_diffuse_direct +=
                        contribution.diffuse_direct;
                    sample_diffuse_indirect +=
                        contribution.diffuse_indirect;
                    sample_glossy_direct +=
                        contribution.glossy_direct;
                    sample_glossy_indirect +=
                        contribution.glossy_indirect;
                    sample_transmission_direct +=
                        contribution.transmission_direct;
                    sample_transmission_indirect +=
                        contribution.transmission_indirect;
                };

            $for (
                path_step,
                kernel_parameters.max_path_steps) {
                continuation_probability = 1.0f;
                const auto terminate_sample =
                    cycles_sampler::sample_1d(
                        sobol_table,
                        kernel_parameters
                            .sobol_sequence_size,
                        sample_index,
                        rng_hash,
                        cycles_sampler::path_dimension(
                            path_step,
                            tabulated_sobol::
                                terminate_dimension));
                const auto light_sample =
                    cycles_sampler::sample_3d(
                        sobol_table,
                        kernel_parameters
                            .sobol_sequence_size,
                        sample_index,
                        rng_hash,
                        cycles_sampler::path_dimension(
                            path_step,
                            tabulated_sobol::
                                light_dimension));
                Var<LightDistributionGpu> selected_light =
                    light_distribution_sample_callable(
                        light_sample.z);
                const auto light_terminate_sample =
                    cycles_sampler::sample_1d(
                        sobol_table,
                        kernel_parameters
                            .sobol_sequence_size,
                        sample_index,
                        rng_hash,
                        cycles_sampler::path_dimension(
                            path_step,
                            tabulated_sobol::
                                light_terminate_dimension));
                const auto bsdf_sample =
                    cycles_sampler::sample_3d(
                        sobol_table,
                        kernel_parameters
                            .sobol_sequence_size,
                        sample_index,
                        rng_hash,
                        cycles_sampler::path_dimension(
                            path_step,
                            tabulated_sobol::
                                surface_bsdf_dimension));
                // Match Cycles' RaySelfPrimitives contract: the previous
                // committed primitive is rejected by identity during
                // traversal. This is independent of origin offset and remains
                // active for both transparent and non-transparent bounces.
                Var<luisa::compute::CommittedHit> hit =
                    scene->accel
                        ->traverse(
                            ray,
                            {.visibility_mask =
                                 ray_visibility})
                        .on_surface_candidate(
                            [&](luisa::compute::
                                    SurfaceCandidate
                                        &candidate) noexcept {
                                auto candidate_hit =
                                    candidate.hit();
                                $if (!surface_ray::
                                          same_primitive(
                                              candidate_hit
                                                  ->inst,
                                              candidate_hit
                                                  ->prim,
                                              ray_source_instance,
                                              ray_source_primitive)) {
                                    candidate.commit();
                                };
                            })
                        .on_procedural_candidate(
                            [](luisa::compute::
                                   ProceduralCandidate &) noexcept {
                            })
                        .trace();
                $if (hit->miss()) {
                    Bool competing =
                        (path_depth > 0u) & (!previous_delta);
                    const auto environment_selection_pdf =
                        scene
                                ->environment_in_light_distribution
                            ? scene->light_selection_pdf
                            : 0.0f;
                    Float background_pdf =
                        background_sampling::pdf(
                            scene
                                ->background_conditional_cdf,
                            scene
                                ->background_marginal_cdf,
                            scene->background_map_width,
                            scene->background_map_height,
                            scene->background_map_weight,
                            scene
                                ->background_guided_sun_weight,
                            make_float3(
                                scene
                                    ->background_guided_sun_axis),
                            scene
                                ->background_guided_sun_radius,
                            ray->direction());
                    Float environment_pdf =
                        environment_selection_pdf *
                        background_pdf;
                    Float environment_weight =
                        forward_light_weight(
                            previous_bsdf_pdf,
                            environment_pdf,
                            competing,
                            environment_pdf > 0.0f);
                    Float3 environment_contribution =
                        clamp_contribution(
                            throughput *
                                evaluate_environment(
                                    ray->direction(),
                                    ray_visibility) *
                                environment_weight,
                            path_depth);
                    radiance += environment_contribution;
                    auto directly_visible_environment =
                        path_depth == 0u;
                    sample_environment += select(
                        make_float3(0.0f),
                        environment_contribution,
                        directly_visible_environment);
                    accumulate_light_pass(
                        split_scattered_light(
                        select(
                            environment_contribution,
                            make_float3(0.0f),
                            directly_visible_environment),
                        path_diffuse_weight,
                        path_glossy_weight,
                        path_depth == 1u));
                    $break;
                };

                Var<InstanceGpu> instance =
                    scene->instance_buffer->read(hit->inst);
                Var<GeometryGpu> geometry =
                    scene->geometry_buffer->read(
                        instance.geometry_index);
                Var<Triangle> triangle =
                    scene->heap
                        ->buffer<Triangle>(
                            geometry.bindless_base)
                        .read(hit->prim);
                Float3 p0 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 1u)
                        .read(triangle.i0);
                Float3 p1 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 1u)
                        .read(triangle.i1);
                Float3 p2 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 1u)
                        .read(triangle.i2);
                Float3 n0 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 2u)
                        .read(triangle.i0);
                Float3 n1 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 2u)
                        .read(triangle.i1);
                Float3 n2 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 2u)
                        .read(triangle.i2);
                Float2 uv0 =
                    scene->heap
                        ->buffer<luisa::float2>(
                            geometry.bindless_base + 3u)
                        .read(triangle.i0);
                Float2 uv1 =
                    scene->heap
                        ->buffer<luisa::float2>(
                            geometry.bindless_base + 3u)
                        .read(triangle.i1);
                Float2 uv2 =
                    scene->heap
                        ->buffer<luisa::float2>(
                            geometry.bindless_base + 3u)
                        .read(triangle.i2);
                Float4 tangent0 =
                    scene->heap
                        ->buffer<luisa::float4>(
                            geometry.bindless_base + 7u)
                        .read(triangle.i0);
                Float4 tangent1 =
                    scene->heap
                        ->buffer<luisa::float4>(
                            geometry.bindless_base + 7u)
                        .read(triangle.i1);
                Float4 tangent2 =
                    scene->heap
                        ->buffer<luisa::float4>(
                            geometry.bindless_base + 7u)
                        .read(triangle.i2);
                Float3 generated0 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 5u)
                        .read(triangle.i0);
                Float3 generated1 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 5u)
                        .read(triangle.i1);
                Float3 generated2 =
                    scene->heap
                        ->buffer<luisa::float3>(
                            geometry.bindless_base + 5u)
                        .read(triangle.i2);
                Float random_per_island =
                    scene->heap
                        ->buffer<float>(
                            geometry.bindless_base + 6u)
                        .read(hit->prim);
                UInt material_slot =
                    scene->heap
                        ->buffer<luisa::uint>(
                            geometry.bindless_base + 4u)
                        .read(hit->prim);
                Bool triangle_smooth =
                    scene->heap
                        ->buffer<luisa::uint>(
                            geometry.bindless_base + 8u)
                        .read(hit->prim) != 0u;

                auto object_to_world =
                    scene->accel
                        ->instance_transform(hit->inst);
                auto world_to_object =
                    inverse(object_to_world);
                auto normal_to_world =
                    transpose(world_to_object);
                Float3 wp0 =
                    (object_to_world *
                     make_float4(p0, 1.0f))
                        .xyz();
                Float3 wp1 =
                    (object_to_world *
                     make_float4(p1, 1.0f))
                        .xyz();
                Float3 wp2 =
                    (object_to_world *
                     make_float4(p2, 1.0f))
                        .xyz();
                Float3 object_geometric_normal =
                    safe_normalize(
                        cross(p1 - p0, p2 - p0),
                        make_float3(0.0f, 0.0f, 1.0f));
                Float3 geometric_normal = safe_normalize(
                    (normal_to_world *
                     make_float4(
                         object_geometric_normal, 0.0f))
                        .xyz(),
                    -ray->direction());
                Float3 object_shading_normal =
                    triangle_interpolate(
                        hit->bary, n0, n1, n2);
                Float4 object_tangent =
                    triangle_interpolate(
                        hit->bary,
                        tangent0,
                        tangent1,
                        tangent2);
                Float3 shading_normal = safe_normalize(
                    (normal_to_world *
                     make_float4(
                         object_shading_normal,
                         0.0f))
                        .xyz(),
                    geometric_normal);
                Bool back_facing =
                    dot(geometric_normal, -ray->direction()) <
                    0.0f;
                geometric_normal = select(
                    geometric_normal,
                    -geometric_normal,
                    back_facing);
                shading_normal = select(
                    shading_normal,
                    -shading_normal,
                    back_facing);
                shading_normal = select(
                    shading_normal,
                    -shading_normal,
                    dot(shading_normal, geometric_normal) <
                        0.0f);
                // Reconstruct static-triangle shading points from the
                // committed barycentrics. This is both more accurate than
                // origin + t * direction at large world coordinates and
                // provides the same geometric point used by Cycles before
                // spawning secondary and shadow rays.
                Float3 object_shading_position =
                    p0 +
                    hit->bary.x * (p1 - p0) +
                    hit->bary.y * (p2 - p0);
                Float3 hit_position =
                    (object_to_world *
                     make_float4(
                         object_shading_position,
                         1.0f))
                        .xyz();
                Float3 object_hit_position =
                    (world_to_object *
                     make_float4(hit_position, 1.0f))
                        .xyz();
                auto make_surface_ray_origin =
                    [&](Float3 direction) noexcept {
                        Float3 object_direction =
                            (world_to_object *
                             make_float4(direction, 0.0f))
                                .xyz();
                        return surface_ray::
                            origin_with_explicit_self_exclusion(
                                hit_position,
                                geometric_normal,
                                object_hit_position,
                                object_direction,
                                p0,
                                p1,
                                p2);
                    };
                Float3 tangent = safe_normalize(
                    (object_to_world *
                     make_float4(
                         object_tangent.xyz(), 0.0f))
                        .xyz(),
                    safe_normalize(
                        (wp1 - wp0) -
                            geometric_normal *
                                dot(
                                    wp1 - wp0,
                                    geometric_normal),
                        make_float3(
                            1.0f, 0.0f, 0.0f)));
                Float surface_radius =
                    ray_dP +
                    hit->committed_ray_t * ray_dD;
                // Cycles stores a compact scalar ray differential.
                // ShaderData reconstructs its two surface directions
                // with make_orthonormals(sd->Ng), rather than retaining
                // the full camera-ray differential vectors.
                auto normal_components_differ =
                    (geometric_normal.x != geometric_normal.y) |
                    (geometric_normal.x != geometric_normal.z);
                Float3 compact_dx = select(
                    make_float3(
                        geometric_normal.z -
                            geometric_normal.y,
                        geometric_normal.x +
                            geometric_normal.z,
                        -geometric_normal.y -
                            geometric_normal.x),
                    make_float3(
                        geometric_normal.z -
                            geometric_normal.y,
                        geometric_normal.x -
                            geometric_normal.z,
                        geometric_normal.y -
                            geometric_normal.x),
                    normal_components_differ);
                compact_dx = safe_normalize(
                    compact_dx,
                    tangent);
                Float3 compact_dy =
                    cross(geometric_normal, compact_dx);
                Float3 dPdx =
                    compact_dx * surface_radius;
                Float3 dPdy =
                    compact_dy * surface_radius;
                Float differential_radius = surface_radius;
                Float3 edge1 = wp1 - wp0;
                Float3 edge2 = wp2 - wp0;
                Float gram00 = dot(edge1, edge1);
                Float gram01 = dot(edge1, edge2);
                Float gram11 = dot(edge2, edge2);
                Float gram_determinant =
                    gram00 * gram11 -
                    gram01 * gram01;
                Bool valid_gram =
                    abs(gram_determinant) > 1.0e-20f;
                Float safe_gram_determinant = select(
                    1.0f,
                    gram_determinant,
                    valid_gram);
                auto barycentric_differential =
                    [&](Float3 differential) noexcept {
                        Float projected1 =
                            dot(differential, edge1);
                        Float projected2 =
                            dot(differential, edge2);
                        Float2 delta = make_float2(
                            (projected1 * gram11 -
                             projected2 * gram01) /
                                safe_gram_determinant,
                            (projected2 * gram00 -
                             projected1 * gram01) /
                                safe_gram_determinant);
                        delta = select(
                            make_float2(0.0f),
                            delta,
                            valid_gram);
                        // Cycles flips dPdu/dPdv when ShaderData is
                        // oriented to a backfacing hit. Solving the
                        // compact surface differential in that basis
                        // therefore reverses arbitrary attribute
                        // derivatives while leaving world P unchanged.
                        return select(
                            delta, -delta, back_facing);
                    };
                Float2 barycentric_dx =
                    barycentric_differential(dPdx);
                Float2 barycentric_dy =
                    barycentric_differential(dPdy);
                Float3 object_dPdx =
                    (world_to_object *
                     make_float4(dPdx, 0.0f))
                        .xyz();
                Float3 object_dPdy =
                    (world_to_object *
                     make_float4(dPdy, 0.0f))
                        .xyz();
                Float3 generated_dx =
                    (generated1 - generated0) *
                        barycentric_dx.x +
                    (generated2 - generated0) *
                        barycentric_dx.y;
                Float3 generated_dy =
                    (generated1 - generated0) *
                        barycentric_dy.x +
                    (generated2 - generated0) *
                        barycentric_dy.y;
                Float2 uv = triangle_interpolate(
                    hit->bary, uv0, uv1, uv2);
                Float2 uv_dx =
                    (uv1 - uv0) * barycentric_dx.x +
                    (uv2 - uv0) * barycentric_dx.y;
                Float2 uv_dy =
                    (uv1 - uv0) * barycentric_dy.x +
                    (uv2 - uv0) * barycentric_dy.y;

                UInt2 material_binding =
                    scene->geometry_material_buffer->read(
                        geometry.material_offset +
                        min(
                            material_slot,
                            max(
                                geometry.material_count,
                                1u) -
                                1u));
                $if (material_slot <
                     instance.override_count) {
                    material_binding =
                        scene->override_material_buffer->read(
                            instance.override_offset +
                            material_slot);
                };
                UInt surface_tag = material_binding.x;

                SurfacePoint point{
                    .position = hit_position,
                    .object_position =
                        object_shading_position,
                    .object_location =
                        (object_to_world *
                         make_float4(
                             0.0f, 0.0f, 0.0f, 1.0f))
                            .xyz(),
                    .generated =
                        triangle_interpolate(
                            hit->bary,
                            generated0,
                            generated1,
                            generated2),
                    .geometric_normal = geometric_normal,
                    .shading_normal = shading_normal,
                    .object_shading_normal =
                        object_shading_normal,
                    .object_tangent =
                        object_tangent.xyz(),
                    .tangent_sign = object_tangent.w,
                    .normal_to_world_x =
                        (normal_to_world *
                         make_float4(
                             1.0f,
                             0.0f,
                             0.0f,
                             0.0f))
                            .xyz(),
                    .normal_to_world_y =
                        (normal_to_world *
                         make_float4(
                             0.0f,
                             1.0f,
                             0.0f,
                             0.0f))
                            .xyz(),
                    .normal_to_world_z =
                        (normal_to_world *
                         make_float4(
                             0.0f,
                             0.0f,
                             1.0f,
                             0.0f))
                            .xyz(),
                    .dpdu = tangent,
                    .dpdv = cross(
                        shading_normal, tangent),
                    .dPdx = dPdx,
                    .dPdy = dPdy,
                    .object_dPdx = object_dPdx,
                    .object_dPdy = object_dPdy,
                    .generated_dx = generated_dx,
                    .generated_dy = generated_dy,
                    .incoming = -ray->direction(),
                    .uv = uv,
                    .uv_dx = uv_dx,
                    .uv_dy = uv_dy,
                    .geometry_index =
                        instance.geometry_index,
                    .barycentric = hit->bary,
                    .barycentric_dx =
                        barycentric_dx,
                    .barycentric_dy =
                        barycentric_dy,
                    .instance_id = hit->inst,
                    .primitive_id = hit->prim,
                    .parameter_block =
                        material_binding.y,
                    .object_random =
                        instance.object_random,
                    .particle_index =
                        instance.particle_index,
                    .random_per_island =
                        random_per_island,
                    .ray_visibility = ray_visibility,
                    .ray_events = ray_events,
                    .ray_depth = path_depth,
                    .diffuse_depth = diffuse_depth,
                    .glossy_depth = glossy_depth,
                    .transparent_depth =
                        transparent_depth,
                    .transmission_depth =
                        transmission_depth,
                    .ray_length =
                        hit->committed_ray_t,
                    .time = 0.0f,
                    .back_facing = back_facing};
                Float3 shadow_shading_normal =
                    shading_normal;
                $if (triangle_smooth &
                     (instance
                          .shadow_terminator_geometry_offset >
                      0.0f)) {
                    shadow_shading_normal =
                        surface_shading_normal(
                            surface_tag, point);
                };
                auto make_surface_shadow_origin =
                    [&](Float3 direction) noexcept {
                        // Cycles' direct-light shadow rays use the
                        // shadow-terminator construction directly. They do
                        // not apply integrate_surface_ray_offset(): exact
                        // source identity is handled by the ray query.
                        return surface_ray::
                            shadow_terminator_origin(
                                hit_position,
                                shadow_shading_normal,
                                geometric_normal,
                                direction,
                                instance
                                    .shadow_terminator_geometry_offset,
                                triangle_smooth,
                                object_to_world,
                                hit->bary,
                                p0,
                                p1,
                                p2,
                                n0,
                                n1,
                                n2);
                    };
                UInt path_lobe_mask =
                    surface_query.lobe_mask;
                Bool previous_ray_was_diffuse =
                    (ray_events &
                     static_cast<std::uint32_t>(
                         contract::event_diffuse)) != 0u;
                if (!reflective_caustics) {
                    path_lobe_mask = select(
                        path_lobe_mask,
                        path_lobe_mask &
                            ~static_cast<std::uint32_t>(
                                contract::event_glossy),
                        previous_ray_was_diffuse);
                }
                if (!refractive_caustics) {
                    path_lobe_mask = select(
                        path_lobe_mask,
                        path_lobe_mask &
                            ~static_cast<std::uint32_t>(
                                contract::event_transmission),
                        previous_ray_was_diffuse);
                }
                // Cycles' PATH_RAY_TERMINATE_AFTER_TRANSPARENT
                // evaluates emission but allocates only transparent
                // closures. Filtering the query here also renormalizes
                // mixed transparent/opaque closure selection instead of
                // probabilistically losing the transparent branch.
                path_lobe_mask = select(
                    path_lobe_mask,
                    static_cast<std::uint32_t>(
                        contract::event_transparent),
                    terminate_after_transparent);
                SurfaceQuery path_surface_query{
                    .lobe_mask = path_lobe_mask,
                    .transport_mode =
                        surface_query.transport_mode,
                    .glossy_filter_roughness = 0.0f};
                auto blur_pdf =
                    kernel_parameters.filter_glossy *
                    minimum_bsdf_pdf;
                auto filter_glossy_enabled =
                    kernel_parameters.filter_glossy <
                    std::numeric_limits<float>::max();
                path_surface_query.glossy_filter_roughness =
                    select(
                        0.0f,
                        sqrt(max(1.0f - blur_pdf, 0.0f)) *
                            0.5f,
                        filter_glossy_enabled &
                            (blur_pdf < 1.0f));

                Float3 emitted = surface_emission(
                    surface_tag,
                    point,
                    point.incoming);
                Float emission_weight = 1.0f;
                if (next_event_estimation &&
                    scene->emissive_triangle_count > 0u) {
                    Bool competing =
                        (path_depth > 0u) & (!previous_delta);
                    Float light_pdf =
                        emissive_triangle_pdf(
                            hit->inst,
                            hit->prim,
                            ray->origin(),
                            hit_position,
                            wp0,
                            wp1,
                            wp2);
                    emission_weight =
                        forward_light_weight(
                            previous_bsdf_pdf,
                            light_pdf,
                            competing,
                            light_pdf > 0.0f);
                }
                Float3 emission_contribution =
                    clamp_contribution(
                        throughput * emitted *
                            emission_weight,
                        path_depth);
                radiance += emission_contribution;
                auto directly_visible_emission =
                    path_depth == 0u;
                sample_emission += select(
                    make_float3(0.0f),
                    emission_contribution,
                    directly_visible_emission);
                accumulate_light_pass(
                    split_scattered_light(
                    select(
                        emission_contribution,
                        make_float3(0.0f),
                        directly_visible_emission),
                    path_diffuse_weight,
                    path_glossy_weight,
                    path_depth == 1u));

                // PATH_RAY_TERMINATE_ON_NEXT_SURFACE still records
                // surface emission, then stops before data passes, direct
                // lighting, or another closure sample.
                $if (terminate_on_next_surface) {
                    $break;
                };

                // Cycles performs continuation roulette only after the
                // next ray is known to hit a surface. Background and
                // surface-emission contributions above are therefore
                // retained even when the path does not continue
                // scattering.
                Bool arrived_through_transparency =
                    (ray_events &
                     static_cast<std::uint32_t>(
                         contract::event_transparent)) != 0u;
                // Kernel parameters already contain Cycles' scene-sync
                // representation: opaque minimums include the first
                // direct-light bounce.
                Bool use_roulette = select(
                    path_depth >
                        kernel_parameters.min_bounces,
                    transparent_depth >
                        kernel_parameters
                            .transparent_min_bounces,
                    arrived_through_transparency);
                $if (use_roulette) {
                    Float survival = min(
                        sqrt(
                            max(
                                abs(throughput.x),
                                max(
                                    abs(throughput.y),
                                    abs(throughput.z)))),
                        1.0f);
                    continuation_probability = survival;
                    $if (survival <= 0.0f) {
                        $break;
                    };
                    $if (terminate_sample >= survival) {
                        $break;
                    };
                    throughput /= survival;
                };

                if (path_trace_enabled) {
                    UInt cycles_visibility = 0u;
                    cycles_visibility |= select(
                        0u,
                        1u << 0u,
                        (ray_visibility &
                         camera_visibility) != 0u);
                    cycles_visibility |= select(
                        0u,
                        1u << 1u,
                        (ray_visibility &
                         transmission_visibility) != 0u);
                    cycles_visibility |= select(
                        0u,
                        1u << 2u,
                        (ray_visibility &
                         diffuse_visibility) != 0u);
                    cycles_visibility |= select(
                        0u,
                        1u << 3u,
                        (ray_visibility &
                         glossy_visibility) != 0u);
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::state_depth,
                        make_float3(
                            cast<float>(path_step),
                            cast<float>(
                                (path_step + 1u) *
                                tabulated_sobol::
                                    bounce_dimension_count),
                            cast<float>(path_depth)));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::state_lobes,
                        make_float3(
                            cast<float>(transparent_depth),
                            cast<float>(diffuse_depth),
                            cast<float>(glossy_depth)));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::state_visibility,
                        make_float3(
                            cast<float>(transmission_depth),
                            trace_uint32(
                                cycles_visibility)
                                .xy()));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::state_flags,
                        make_float3(
                            trace_uint32(path_flags).xy(),
                            continuation_probability));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::throughput,
                        throughput);
                    trace_write_event(
                        path_step,
                        path_trace_schema::EventSlot::ray_p,
                        ray->origin());
                    trace_write_event(
                        path_step,
                        path_trace_schema::EventSlot::ray_d,
                        ray->direction());
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::ray_range,
                        make_float3(
                            ray->t_min(),
                            ray->t_max(),
                            0.5f));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::isect_coord,
                        make_float3(
                            hit->committed_ray_t,
                            hit->bary));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::isect_id,
                        make_float3(
                            cast<float>(hit->inst),
                            cast<float>(hit->prim),
                            1.0f));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::surface_meta,
                        make_float3(
                            trace_uint32(surface_tag).xy(),
                            0.0f));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::surface_p,
                        point.position);
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::surface_ng,
                        point.geometric_normal);
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::surface_n,
                        point.shading_normal);
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::random_scalars,
                        make_float3(
                            terminate_sample,
                            select(
                                0.0f,
                                light_terminate_sample,
                                kernel_parameters
                                        .light_inv_rr_threshold >
                                    0.0f),
                            0.0f));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::random_light,
                        light_sample);
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::random_bsdf,
                        bsdf_sample);
                }

                // Cycles writes camera data passes only along the
                // transparent-background chain. Diffuse Color is
                // throughput-weighted at every surface in that chain;
                // Normal is captured once, after skipping transparent
                // surfaces below the View Layer alpha threshold.
                $if (path_depth == 0u) {
                    auto aov =
                        surface_aov(surface_tag, point);
                    sample_albedo +=
                        throughput * aov.albedo;
                    sample_glossy_color +=
                        throughput * aov.glossy_albedo;
                    sample_transmission_color +=
                        throughput *
                        aov.transmission_albedo;
                    auto surface_alpha =
                        clamp(
                            make_float3(1.0f) -
                                aov.transparency,
                            make_float3(0.0f),
                            make_float3(1.0f));
                    auto average_alpha =
                        (surface_alpha.x +
                         surface_alpha.y +
                         surface_alpha.z) *
                        (1.0f / 3.0f);
                    auto writes_normal =
                        (!primary_recorded) &
                        ((kernel_parameters
                                  .pass_alpha_threshold ==
                          0.0f) |
                         (average_alpha >=
                          kernel_parameters
                              .pass_alpha_threshold));
                    sample_normal = select(
                        sample_normal,
                        aov.normal,
                        writes_normal);
                    primary_recorded =
                        primary_recorded |
                        writes_normal;
                    sample_alpha = select(
                        sample_alpha,
                        1.0f,
                        average_alpha > 0.0f);
                };

                if (next_event_estimation) {
                    $if (selected_light.kind ==
                         static_cast<std::uint32_t>(
                             sampling::
                                 LightDistributionEmitterKind::
                                     environment)) {
                        const auto background_sample =
                            background_sampling::sample(
                                scene
                                    ->background_conditional_cdf,
                                scene
                                    ->background_marginal_cdf,
                                scene
                                    ->background_map_width,
                                scene
                                    ->background_map_height,
                                scene
                                    ->background_map_weight,
                                scene
                                    ->background_guided_sun_weight,
                                make_float3(
                                    scene
                                        ->background_guided_sun_axis),
                                scene
                                    ->background_guided_sun_radius,
                                light_sample.xy());
                        Float3 wi =
                            background_sample.direction;
                        Float light_pdf =
                            background_sample.pdf *
                            selected_light.selection_pdf;
                        $if (light_pdf > 0.0f) {
                            const auto shadow =
                                make_surface_shadow_origin(
                                    wi);
                            Var<luisa::compute::Ray>
                                environment_shadow_ray =
                                    make_ray(
                                        shadow.position,
                                        wi,
                                        0.0f,
                                        ray_maximum);
                            Float3 shadow_transmittance =
                                trace_shadow(
                                    environment_shadow_ray,
                                    select(
                                        surface_ray::
                                            invalid_primitive,
                                        hit->inst,
                                        shadow.skip_self),
                                    select(
                                        surface_ray::
                                            invalid_primitive,
                                        hit->prim,
                                        shadow.skip_self),
                                    surface_ray::
                                        invalid_primitive,
                                    surface_ray::
                                        invalid_primitive);
                            $if (any(
                                shadow_transmittance >
                                0.0f)) {
                                auto evaluation =
                                    evaluate_surface(
                                        surface_tag,
                                        point,
                                        wi,
                                        path_surface_query);
                                Float mis_weight =
                                    nee_light_weight(
                                        light_pdf,
                                        evaluation.pdf);
                                Float3
                                    unshadowed_contribution =
                                        evaluation.f *
                                        evaluate_environment(
                                            wi,
                                            ray_visibility) *
                                        (mis_weight /
                                         light_pdf);
                                Float roulette_weight =
                                    sample_light_roulette(
                                        unshadowed_contribution,
                                        light_terminate_sample);
                                Float3 contribution =
                                    clamp_contribution(
                                        throughput *
                                            unshadowed_contribution *
                                            shadow_transmittance *
                                            roulette_weight,
                                        path_depth);
                                radiance += contribution;
                                accumulate_light_pass(
                                    split_nee_light(
                                    contribution,
                                    evaluation.f,
                                    evaluation.diffuse_f,
                                    path_diffuse_weight,
                                    path_glossy_weight,
                                    path_depth));
                            };
                        };
                    };
                    $if (selected_light.kind ==
                         static_cast<std::uint32_t>(
                             sampling::
                                 LightDistributionEmitterKind::
                                     emissive_triangle)) {
                        Var<EmissiveTriangleGpu> emitter =
                            scene->emissive_triangle_buffer->read(
                                selected_light.index);
                        Var<GeometryGpu> light_geometry =
                            scene->geometry_buffer->read(
                                emitter.geometry_index);
                        Var<Triangle> light_triangle =
                            scene->heap
                                ->buffer<Triangle>(
                                    light_geometry.bindless_base)
                                .read(emitter.primitive_index);
                        Float3 lp0 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    1u)
                                .read(light_triangle.i0);
                        Float3 lp1 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    1u)
                                .read(light_triangle.i1);
                        Float3 lp2 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    1u)
                                .read(light_triangle.i2);
                        Float3 ln0 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    2u)
                                .read(light_triangle.i0);
                        Float3 ln1 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    2u)
                                .read(light_triangle.i1);
                        Float3 ln2 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    2u)
                                .read(light_triangle.i2);
                        Float2 luv0 =
                            scene->heap
                                ->buffer<luisa::float2>(
                                    light_geometry.bindless_base +
                                    3u)
                                .read(light_triangle.i0);
                        Float2 luv1 =
                            scene->heap
                                ->buffer<luisa::float2>(
                                    light_geometry.bindless_base +
                                    3u)
                                .read(light_triangle.i1);
                        Float2 luv2 =
                            scene->heap
                                ->buffer<luisa::float2>(
                                    light_geometry.bindless_base +
                                    3u)
                                .read(light_triangle.i2);
                        Float4 light_tangent0 =
                            scene->heap
                                ->buffer<luisa::float4>(
                                    light_geometry.bindless_base +
                                    7u)
                                .read(light_triangle.i0);
                        Float4 light_tangent1 =
                            scene->heap
                                ->buffer<luisa::float4>(
                                    light_geometry.bindless_base +
                                    7u)
                                .read(light_triangle.i1);
                        Float4 light_tangent2 =
                            scene->heap
                                ->buffer<luisa::float4>(
                                    light_geometry.bindless_base +
                                    7u)
                                .read(light_triangle.i2);
                        Float3 light_generated0 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    5u)
                                .read(light_triangle.i0);
                        Float3 light_generated1 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    5u)
                                .read(light_triangle.i1);
                        Float3 light_generated2 =
                            scene->heap
                                ->buffer<luisa::float3>(
                                    light_geometry.bindless_base +
                                    5u)
                                .read(light_triangle.i2);
                        Float light_random_per_island =
                            scene->heap
                                ->buffer<float>(
                                    light_geometry.bindless_base +
                                    6u)
                                .read(
                                    emitter.primitive_index);
                        Var<InstanceGpu> light_instance =
                            scene->instance_buffer->read(
                                emitter.instance_index);
                        Float3 local_lp0 = lp0;
                        Float3 local_lp1 = lp1;
                        Float3 local_lp2 = lp2;
                        auto light_object_to_world =
                            scene->accel->instance_transform(
                                emitter.instance_index);
                        auto light_normal_to_world =
                            transpose(inverse(
                                light_object_to_world));
                        lp0 =
                            (light_object_to_world *
                             make_float4(lp0, 1.0f))
                                .xyz();
                        lp1 =
                            (light_object_to_world *
                             make_float4(lp1, 1.0f))
                                .xyz();
                        lp2 =
                            (light_object_to_world *
                             make_float4(lp2, 1.0f))
                                .xyz();
                        auto triangle_sample =
                            spherical_geometry::
                                sample_triangle(
                                    hit_position,
                                    lp0,
                                    lp1,
                                    lp2,
                                    light_sample.xy());
                        Float2 light_barycentric =
                            triangle_sample.barycentric;
                        Float3 light_position =
                            triangle_sample.position;
                        Float3
                            light_object_geometric_normal =
                                safe_normalize(
                                    cross(
                                        local_lp1 - local_lp0,
                                        local_lp2 - local_lp0),
                                    make_float3(
                                        0.0f,
                                        0.0f,
                                        1.0f));
                        Float3 light_geometric_normal =
                            safe_normalize(
                                (light_normal_to_world *
                                 make_float4(
                                     light_object_geometric_normal,
                                     0.0f))
                                    .xyz(),
                                make_float3(
                                    0.0f, 0.0f, 1.0f));
                        Float3 light_object_shading_normal =
                            triangle_interpolate(
                                light_barycentric,
                                ln0,
                                ln1,
                                ln2);
                        Float4 light_object_tangent =
                            triangle_interpolate(
                                light_barycentric,
                                light_tangent0,
                                light_tangent1,
                                light_tangent2);
                        Float3 light_shading_normal =
                            safe_normalize(
                                (light_normal_to_world *
                                 make_float4(
                                     light_object_shading_normal,
                                     0.0f))
                                    .xyz(),
                                light_geometric_normal);
                        Float light_distance = triangle_sample.distance;
                        Float3 wi = triangle_sample.direction;
                        Bool light_back_facing =
                            dot(
                                light_geometric_normal,
                                -wi) < 0.0f;
                        light_geometric_normal = select(
                            light_geometric_normal,
                            -light_geometric_normal,
                            light_back_facing);
                        light_shading_normal = select(
                            light_shading_normal,
                            -light_shading_normal,
                            light_back_facing);
                        light_shading_normal = select(
                            light_shading_normal,
                            -light_shading_normal,
                            dot(
                                light_shading_normal,
                                light_geometric_normal) <
                                0.0f);
                        Float3 light_tangent =
                            safe_normalize(
                                (light_object_to_world *
                                 make_float4(
                                     light_object_tangent.xyz(),
                                     0.0f))
                                    .xyz(),
                                safe_normalize(
                                    (lp1 - lp0) -
                                        light_geometric_normal *
                                            dot(
                                                lp1 - lp0,
                                                light_geometric_normal),
                                    make_float3(
                                        1.0f,
                                        0.0f,
                                        0.0f)));
                        SurfacePoint light_point{
                            .position = light_position,
                            .object_position =
                                triangle_interpolate(
                                    light_barycentric,
                                    local_lp0,
                                    local_lp1,
                                    local_lp2),
                            .object_location =
                                (light_object_to_world *
                                 make_float4(
                                     0.0f,
                                     0.0f,
                                     0.0f,
                                     1.0f))
                                    .xyz(),
                            .generated =
                                triangle_interpolate(
                                    light_barycentric,
                                    light_generated0,
                                    light_generated1,
                                    light_generated2),
                            .geometric_normal =
                                light_geometric_normal,
                            .shading_normal =
                                light_shading_normal,
                            .object_shading_normal =
                                light_object_shading_normal,
                            .object_tangent =
                                light_object_tangent.xyz(),
                            .tangent_sign =
                                light_object_tangent.w,
                            .normal_to_world_x =
                                (light_normal_to_world *
                                 make_float4(
                                     1.0f,
                                     0.0f,
                                     0.0f,
                                     0.0f))
                                    .xyz(),
                            .normal_to_world_y =
                                (light_normal_to_world *
                                 make_float4(
                                     0.0f,
                                     1.0f,
                                     0.0f,
                                     0.0f))
                                    .xyz(),
                            .normal_to_world_z =
                                (light_normal_to_world *
                                 make_float4(
                                     0.0f,
                                     0.0f,
                                     1.0f,
                                     0.0f))
                                    .xyz(),
                            .dpdu = light_tangent,
                            .dpdv = cross(
                                light_shading_normal,
                                light_tangent),
                            .dPdx = make_float3(0.0f),
                            .dPdy = make_float3(0.0f),
                            .object_dPdx =
                                make_float3(0.0f),
                            .object_dPdy =
                                make_float3(0.0f),
                            .generated_dx =
                                make_float3(0.0f),
                            .generated_dy =
                                make_float3(0.0f),
                            .incoming = -wi,
                            .uv = triangle_interpolate(
                                light_barycentric,
                                luv0,
                                luv1,
                                luv2),
                            .uv_dx = make_float2(0.0f),
                            .uv_dy = make_float2(0.0f),
                            .geometry_index =
                                emitter.geometry_index,
                            .barycentric =
                                light_barycentric,
                            .barycentric_dx =
                                make_float2(0.0f),
                            .barycentric_dy =
                                make_float2(0.0f),
                            .instance_id =
                                emitter.instance_index,
                            .primitive_id =
                                emitter.primitive_index,
                            .parameter_block =
                                emitter.parameter_block,
                            .object_random =
                                light_instance.object_random,
                            .particle_index =
                                light_instance.particle_index,
                            .random_per_island =
                                light_random_per_island,
                            .ray_visibility =
                                shadow_visibility,
                            .ray_events = 0u,
                            .ray_depth = path_depth,
                            .diffuse_depth =
                                diffuse_depth,
                            .glossy_depth =
                                glossy_depth,
                            .transparent_depth =
                                transparent_depth,
                            .transmission_depth =
                                transmission_depth,
                            .ray_length =
                                light_distance,
                    .time = 0.5f,
                            .back_facing =
                                light_back_facing};
                        Float3 light_radiance =
                            surface_emission(
                                emitter.surface_tag,
                                light_point,
                                -wi);
                        Float light_pdf =
                            emissive_triangle_pdf(
                                emitter.instance_index,
                                emitter.primitive_index,
                                hit_position,
                                light_position,
                                lp0,
                                lp1,
                                lp2);
                        $if (triangle_sample.valid &
                             (light_pdf > 0.0f) &
                             any(light_radiance > 0.0f)) {
                            const auto shadow =
                                make_surface_shadow_origin(
                                    wi);
                            const auto shadow_offset =
                                light_position -
                                shadow.position;
                            const auto shadow_distance =
                                sqrt(max(
                                    length_squared(
                                        shadow_offset),
                                    1.0e-20f));
                            const auto shadow_direction =
                                shadow_offset /
                                shadow_distance;
                            Var<luisa::compute::Ray>
                                mesh_light_shadow_ray =
                                    make_ray(
                                        shadow.position,
                                        shadow_direction,
                                        0.0f,
                                        shadow_distance);
                            Float3 shadow_transmittance =
                                trace_shadow(
                                    mesh_light_shadow_ray,
                                    select(
                                        surface_ray::
                                            invalid_primitive,
                                        hit->inst,
                                        shadow.skip_self),
                                    select(
                                        surface_ray::
                                            invalid_primitive,
                                        hit->prim,
                                        shadow.skip_self),
                                    emitter.instance_index,
                                    emitter.primitive_index);
                            $if (any(
                                shadow_transmittance > 0.0f)) {
                                auto evaluation =
                                    evaluate_surface(
                                        surface_tag,
                                        point,
                                        wi,
                                        path_surface_query);
                                Float mis_weight =
                                    nee_light_weight(
                                        light_pdf,
                                        evaluation.pdf);
                                Float3 unshadowed_contribution =
                                    evaluation.f *
                                    light_radiance *
                                    (mis_weight / light_pdf);
                                Float roulette_weight =
                                    sample_light_roulette(
                                        unshadowed_contribution,
                                        light_terminate_sample);
                                Float3 contribution =
                                    clamp_contribution(
                                        throughput *
                                            unshadowed_contribution *
                                            shadow_transmittance *
                                            roulette_weight,
                                        path_depth);
                                radiance += contribution;
                                accumulate_light_pass(
                                    split_nee_light(
                                    contribution,
                                    evaluation.f,
                                    evaluation.diffuse_f,
                                    path_diffuse_weight,
                                    path_glossy_weight,
                                    path_depth));
                            };
                        };
                    };
                    $if (selected_light.kind ==
                         static_cast<std::uint32_t>(
                             sampling::
                                 LightDistributionEmitterKind::
                                     analytic_light)) {
                        UInt light_index =
                            selected_light.index;
                        Var<LightGpu> light =
                            scene->light_buffer->read(
                                light_index);
                        Float3 wi = make_float3(0.0f);
                        Float3 light_radiance =
                            make_float3(0.0f);
                        Float3 light_position =
                            light.position;
                        Float3 light_normal =
                            make_float3(
                                0.0f, 0.0f, 1.0f);
                        Float2 light_uv =
                            make_float2(0.5f);
                        Float light_distance = ray_maximum;
                        Float light_pdf = 0.0f;
                        Float light_eval_factor = 0.0f;
                        Bool light_valid = false;

                        $if (light.type ==
                             static_cast<std::uint32_t>(
                                 LightType::area)) {
                            Float sample_x =
                                light_sample.x;
                            Float sample_y =
                                light_sample.y;
                            Bool ellipse =
                                (light.flags &
                                 light_flag_ellipse) != 0u;
                            Float disk_radius =
                                0.5f * sqrt(sample_x);
                            Float disk_angle =
                                2.0f * pi * sample_y;
                            Float u = select(
                                sample_x - 0.5f,
                                disk_radius *
                                    cos(disk_angle),
                                ellipse);
                            Float v = select(
                                sample_y - 0.5f,
                                disk_radius *
                                    sin(disk_angle),
                                ellipse);
                            light_position =
                                light.position +
                                light.axis_x *
                                    (u * light.size_u) +
                                light.axis_y *
                                    (v * light.size_v);
                            Float3 offset =
                                light_position - hit_position;
                            Float distance2 =
                                length_squared(offset);
                            light_distance = sqrt(max(
                                distance2, 1.0e-20f));
                            wi = offset / light_distance;
                            Float cosine = max(
                                dot(
                                    -light.axis_z,
                                    -wi),
                                0.0f);
                            Float area =
                                light.size_u *
                                light.size_v;
                            area *= select(
                                1.0f,
                                0.25f * pi,
                                ellipse);
                            area = max(area, 1.0e-12f);
                            light_pdf =
                                distance2 /
                                max(cosine * area, 1.0e-20f);
                            Bool normalize_power =
                                (light.flags &
                                 light_flag_normalize) != 0u;
                            Float inverse_area = select(
                                1.0f,
                                1.0f / area,
                                normalize_power);
                            Float spread_attenuation = 1.0f;
                            $if (light.spread <
                                 pi - 1.0e-6f) {
                                Float half_spread =
                                    0.5f *
                                    max(light.spread, 0.0f);
                                Float sine_angle = sqrt(max(
                                    1.0f -
                                        cosine * cosine,
                                    0.0f));
                                Float tangent_angle =
                                    sine_angle /
                                    max(cosine, 1.0e-20f);
                                $if (half_spread <= 0.0f) {
                                    spread_attenuation =
                                        select(
                                            pi,
                                            0.0f,
                                            tangent_angle >
                                                1.0e-5f);
                                }
                                $else {
                                    Float tangent_spread =
                                        tan(half_spread);
                                    Float normalization =
                                        select(
                                            3.0f /
                                                max(
                                                    half_spread *
                                                        half_spread *
                                                        half_spread,
                                                    1.0e-20f),
                                            1.0f /
                                                max(
                                                    tangent_spread -
                                                        half_spread,
                                                    1.0e-20f),
                                            half_spread >
                                                0.05f);
                                    spread_attenuation =
                                        max(
                                            (tangent_spread -
                                             tangent_angle) *
                                                normalization,
                                            0.0f);
                                };
                            };
                            light_radiance =
                                light.color *
                                (light.power *
                                 inverse_area *
                                 (1.0f / pi) *
                                 spread_attenuation);
                            light_normal =
                                -light.axis_z;
                            light_uv = make_float2(
                                u + 0.5f,
                                v + 0.5f);
                            light_valid =
                                (distance2 > 1.0e-12f) &
                                (cosine > 0.0f) &
                                (spread_attenuation > 0.0f);
                        }
                        $elif (
                            light.type ==
                                static_cast<std::uint32_t>(
                                    LightType::distant)) {
                            Float half_angle =
                                0.5f *
                                max(light.angle, 0.0f);
                            Bool finite_sun =
                                half_angle > 0.0f;
                            Float cap_height =
                                spherical_geometry::
                                    unit_cap_height(
                                        half_angle);
                            Float cosine_max =
                                1.0f - cap_height;
                            Float3 sun_axis = light.axis_z;
                            Float3 basis_reference = select(
                                make_float3(
                                    0.0f, 0.0f, 1.0f),
                                make_float3(
                                    0.0f, 1.0f, 0.0f),
                                abs(sun_axis.z) > 0.999f);
                            Float3 sun_tangent =
                                safe_normalize(
                                    cross(
                                        basis_reference,
                                        sun_axis),
                                    make_float3(
                                        1.0f,
                                        0.0f,
                                        0.0f));
                            Float3 sun_bitangent =
                                cross(
                                    sun_axis,
                                    sun_tangent);
                            Float cosine_theta =
                                1.0f -
                                light_sample.x * cap_height;
                            Float sine_theta = sqrt(max(
                                1.0f -
                                    cosine_theta *
                                        cosine_theta,
                                0.0f));
                            Float phi =
                                2.0f * pi *
                                light_sample.y;
                            Float3 cone_direction =
                                sun_tangent *
                                    (cos(phi) *
                                     sine_theta) +
                                sun_bitangent *
                                    (sin(phi) *
                                     sine_theta) +
                                sun_axis * cosine_theta;
                            wi = select(
                                sun_axis,
                                cone_direction,
                                finite_sun);
                            Float solid_angle =
                                spherical_geometry::two_pi *
                                cap_height;
                            light_pdf = select(
                                1.0f,
                                1.0f /
                                    max(
                                        solid_angle,
                                        1.0e-20f),
                                finite_sun);
                            Bool normalize_power =
                                (light.flags &
                                 light_flag_normalize) != 0u;
                            Float disk_area =
                                pi *
                                sin(half_angle) *
                                sin(half_angle);
                            Float eval_factor = select(
                                1.0f,
                                1.0f /
                                    max(
                                        disk_area,
                                        1.0e-20f),
                                normalize_power &
                                    finite_sun);
                            light_radiance =
                                light.color *
                                (light.power *
                                 eval_factor);
                            light_position = wi;
                            light_normal = -wi;
                            light_valid = true;
                        }
                        $else {
                            Float3 offset =
                                light.position - hit_position;
                            Float distance2 =
                                length_squared(offset);
                            Float center_distance = sqrt(max(
                                distance2, 1.0e-20f));
                            Float3 center_direction =
                                offset / center_distance;
                            Bool normalize_power =
                                (light.flags &
                                 light_flag_normalize) != 0u;
                            light_valid =
                                distance2 > 1.0e-12f;
                            $if (light.radius > 0.0f) {
                                Float sample_x =
                                    light_sample.x;
                                Float sample_y =
                                    light_sample.y;
                                Bool sphere =
                                    (light.flags &
                                     light_flag_sphere) != 0u;
                                $if (sphere) {
                                    Float sine2 = min(
                                        light.radius *
                                            light.radius /
                                            max(
                                                distance2,
                                                1.0e-20f),
                                        1.0f);
                                    Float cosine_max =
                                        sqrt(max(
                                            1.0f - sine2,
                                            0.0f));
                                    Float3 basis_reference =
                                        select(
                                            make_float3(
                                                0.0f,
                                                0.0f,
                                                1.0f),
                                            make_float3(
                                                0.0f,
                                                1.0f,
                                                0.0f),
                                            abs(
                                                center_direction
                                                    .z) >
                                                0.999f);
                                    Float3 tangent =
                                        safe_normalize(
                                            cross(
                                                basis_reference,
                                                center_direction),
                                            make_float3(
                                                1.0f,
                                                0.0f,
                                                0.0f));
                                    Float3 bitangent = cross(
                                        center_direction,
                                        tangent);
                                    Float cosine_theta =
                                        1.0f -
                                        sample_x *
                                            (1.0f -
                                             cosine_max);
                                    Float sine_theta =
                                        sqrt(max(
                                            1.0f -
                                                cosine_theta *
                                                    cosine_theta,
                                            0.0f));
                                    Float phi =
                                        2.0f * pi *
                                        sample_y;
                                    wi =
                                        tangent *
                                            (cos(phi) *
                                             sine_theta) +
                                        bitangent *
                                            (sin(phi) *
                                             sine_theta) +
                                        center_direction *
                                            cosine_theta;
                                    light_pdf =
                                        1.0f /
                                        max(
                                            2.0f * pi *
                                                (1.0f -
                                                 cosine_max),
                                            1.0e-20f);
                                    Float root = sqrt(max(
                                        light.radius *
                                                light.radius -
                                            distance2 +
                                            distance2 *
                                                cosine_theta *
                                                cosine_theta,
                                        0.0f));
                                    light_distance =
                                        center_distance *
                                            cosine_theta -
                                        root;
                                    light_position =
                                        hit_position +
                                        wi *
                                            light_distance;
                                    light_normal =
                                        safe_normalize(
                                            light_position -
                                                light.position,
                                            -wi);
                                    light_valid =
                                        light_valid &
                                        (distance2 >
                                         light.radius *
                                             light.radius);
                                }
                                $else {
                                    Float3 disk_normal =
                                        -center_direction;
                                    Float3 basis_reference =
                                        select(
                                            make_float3(
                                                0.0f,
                                                0.0f,
                                                1.0f),
                                            make_float3(
                                                0.0f,
                                                1.0f,
                                                0.0f),
                                            abs(disk_normal.z) >
                                                0.999f);
                                    Float3 tangent =
                                        safe_normalize(
                                            cross(
                                                basis_reference,
                                                disk_normal),
                                            make_float3(
                                                1.0f,
                                                0.0f,
                                                0.0f));
                                    Float3 bitangent = cross(
                                        disk_normal,
                                        tangent);
                                    Float disk_radius =
                                        light.radius *
                                        sqrt(sample_x);
                                    Float disk_angle =
                                        2.0f * pi *
                                        sample_y;
                                    light_position =
                                        light.position +
                                        tangent *
                                            (disk_radius *
                                             cos(
                                                 disk_angle)) +
                                        bitangent *
                                            (disk_radius *
                                             sin(
                                                 disk_angle));
                                    Float3 light_offset =
                                        light_position -
                                        hit_position;
                                    Float sampled_distance2 =
                                        length_squared(
                                            light_offset);
                                    light_distance = sqrt(max(
                                        sampled_distance2,
                                        1.0e-20f));
                                    wi =
                                        light_offset /
                                        light_distance;
                                    Float light_cosine = abs(
                                        dot(
                                            disk_normal,
                                            -wi));
                                    light_pdf =
                                        sampled_distance2 /
                                        max(
                                            pi *
                                                light.radius *
                                                light.radius *
                                                light_cosine,
                                            1.0e-20f);
                                    light_normal =
                                        disk_normal;
                                    light_valid =
                                        light_valid &
                                        (light_cosine >
                                         0.0f);
                                };
                                Float point_eval_factor =
                                    analytic_light_sampling::
                                        point_eval_factor(
                                            light.radius,
                                            normalize_power);
                                light_eval_factor =
                                    point_eval_factor;
                                light_radiance =
                                    light.color *
                                    (light.power *
                                     point_eval_factor);
                            }
                            $else {
                                wi = center_direction;
                                light_distance =
                                    center_distance;
                                Float point_eval_factor =
                                    analytic_light_sampling::
                                        point_eval_factor(
                                            light.radius,
                                            normalize_power);
                                light_eval_factor =
                                    point_eval_factor;
                                light_radiance =
                                    light.color *
                                    (light.power *
                                     point_eval_factor);
                                light_pdf =
                                    analytic_light_sampling::
                                        point_disk_pdf(
                                            distance2,
                                            1.0f,
                                            light.radius);
                                light_position =
                                    light.position;
                                light_normal = -wi;
                            };
                            $if (light.type ==
                                 static_cast<std::uint32_t>(
                                     LightType::spot)) {
                                Float cone =
                                    dot(-light.axis_z, -wi);
                                Float cone_minimum =
                                    cos(
                                        max(
                                            light.spot_angle,
                                            0.0f) *
                                        0.5f);
                                Float blend_width =
                                    (1.0f - cone_minimum) *
                                    max(
                                        light.spot_smooth,
                                        0.0f);
                                Float attenuation =
                                    select(
                                        select(
                                            0.0f,
                                            1.0f,
                                            cone >=
                                                cone_minimum),
                                        clamp(
                                            (cone -
                                             cone_minimum) /
                                                max(
                                                    blend_width,
                                                    1.0e-20f),
                                            0.0f,
                                            1.0f),
                                        blend_width >
                                            0.0f);
                                attenuation =
                                    attenuation *
                                    attenuation *
                                    (3.0f -
                                     2.0f * attenuation);
                                light_radiance *= attenuation;
                                light_eval_factor *=
                                    attenuation;
                                light_valid =
                                    light_valid &
                                    (attenuation > 0.0f);
                            };
                        };

                        $if (light.surface_tag !=
                             ~std::uint32_t{0u}) {
                            Float3 relative_position =
                                light_position -
                                light.position;
                            Float3 object_position =
                                make_float3(
                                    dot(
                                        relative_position,
                                        light.axis_x),
                                    dot(
                                        relative_position,
                                        light.axis_y),
                                    dot(
                                        relative_position,
                                        light.axis_z));
                            SurfacePoint light_point{
                                .position =
                                    light_position,
                                .object_position =
                                    object_position,
                                .object_location =
                                    light.position,
                                .generated =
                                    object_position,
                                .geometric_normal =
                                    light_normal,
                                .shading_normal =
                                    light_normal,
                                .object_shading_normal =
                                    light_normal,
                                .object_tangent =
                                    light.axis_x,
                                .tangent_sign = 1.0f,
                                .normal_to_world_x =
                                    light.axis_x,
                                .normal_to_world_y =
                                    light.axis_y,
                                .normal_to_world_z =
                                    light.axis_z,
                                .dpdu =
                                    light.axis_x *
                                    light.size_u,
                                .dpdv =
                                    light.axis_y *
                                    light.size_v,
                                .dPdx =
                                    make_float3(0.0f),
                                .dPdy =
                                    make_float3(0.0f),
                                .object_dPdx =
                                    make_float3(0.0f),
                                .object_dPdy =
                                    make_float3(0.0f),
                                .generated_dx =
                                    make_float3(0.0f),
                                .generated_dy =
                                    make_float3(0.0f),
                                .incoming = -wi,
                                .uv = light_uv,
                                .uv_dx =
                                    make_float2(0.0f),
                                .uv_dy =
                                    make_float2(0.0f),
                                .geometry_index = ~0u,
                                .barycentric =
                                    make_float2(0.0f),
                                .barycentric_dx =
                                    make_float2(0.0f),
                                .barycentric_dy =
                                    make_float2(0.0f),
                                .instance_id = 0u,
                                .primitive_id =
                                    light_index,
                                .parameter_block =
                                    light.parameter_block,
                                .object_random = 0.0f,
                                .particle_index = 0u,
                                .random_per_island = 0.0f,
                                .ray_visibility =
                                    ray_visibility,
                                .ray_events =
                                    ray_events,
                                .ray_depth =
                                    path_depth,
                                .diffuse_depth =
                                    diffuse_depth,
                                .glossy_depth =
                                    glossy_depth,
                                .transparent_depth =
                                    transparent_depth,
                                .transmission_depth =
                                    transmission_depth,
                                .ray_length =
                                    light_distance,
                                .time = 0.0f,
                                .back_facing = false};
                            light_radiance *=
                                surface_emission(
                                    light.surface_tag,
                                    light_point,
                                    -wi);
                        };

                        light_pdf *=
                            selected_light.selection_pdf;
                        if (path_trace_enabled) {
                            const auto delta_point =
                                (light.type ==
                                 static_cast<
                                     std::uint32_t>(
                                     LightType::point)) &
                                (light.radius == 0.0f);
                            $if (
                                path_trace_active &
                                delta_point &
                                light_valid &
                                (light_pdf > 0.0f)) {
                                auto trace_evaluation =
                                    evaluate_surface(
                                        surface_tag,
                                        point,
                                        wi,
                                        path_surface_query);
                                const auto use_mis =
                                    (light.flags &
                                     light_flag_use_mis) !=
                                    0u;
                                const auto has_competing =
                                    analytic_light_sampling::
                                        point_has_competing_bsdf_technique(
                                            light.radius,
                                            use_mis);
                                const auto trace_bsdf_pdf =
                                    select(
                                        0.0f,
                                        trace_evaluation.pdf,
                                        has_competing);
                                const auto trace_mis_weight =
                                    nee_light_weight(
                                        light_pdf,
                                        trace_bsdf_pdf);
                                trace_write_event(
                                    path_step,
                                    path_trace_schema::
                                        EventSlot::
                                            light_meta,
                                    make_float3(
                                        0.0f,
                                        cast<float>(
                                            selected_light
                                                .emitter_id),
                                        cast<float>(
                                            light_index)));
                                trace_write_event(
                                    path_step,
                                    path_trace_schema::
                                        EventSlot::
                                            light_pdf,
                                    make_float3(
                                        light_pdf,
                                        selected_light
                                            .selection_pdf,
                                        light_eval_factor));
                                trace_write_event(
                                    path_step,
                                    path_trace_schema::
                                        EventSlot::light_d,
                                    wi);
                                trace_write_event(
                                    path_step,
                                    path_trace_schema::
                                        EventSlot::light_p,
                                    light_position);
                                trace_write_event(
                                    path_step,
                                    path_trace_schema::
                                        EventSlot::light_ng,
                                    light_normal);
                                trace_write_event(
                                    path_step,
                                    path_trace_schema::
                                        EventSlot::
                                            light_eval,
                                    make_float3(
                                        light_distance,
                                        trace_bsdf_pdf,
                                        trace_mis_weight));
                            };
                        }
                        $if (light_valid &
                             (light_pdf > 0.0f)) {
                            const auto shadow =
                                make_surface_shadow_origin(
                                    wi);
                            const auto finite_offset =
                                light_position -
                                shadow.position;
                            const auto finite_distance =
                                sqrt(max(
                                    length_squared(
                                        finite_offset),
                                    1.0e-20f));
                            const auto finite_direction =
                                finite_offset /
                                finite_distance;
                            const auto distant =
                                light.type ==
                                static_cast<std::uint32_t>(
                                    LightType::distant);
                            const auto shadow_direction =
                                select(
                                    finite_direction,
                                    wi,
                                    distant);
                            const auto shadow_maximum =
                                select(
                                    finite_distance,
                                    ray_maximum,
                                    distant);
                            Var<luisa::compute::Ray>
                                shadow_ray = make_ray(
                                    shadow.position,
                                    shadow_direction,
                                    0.0f,
                                    shadow_maximum);
                            Float3 shadow_transmittance =
                                trace_shadow(
                                    shadow_ray,
                                    select(
                                        surface_ray::
                                            invalid_primitive,
                                        hit->inst,
                                        shadow.skip_self),
                                    select(
                                        surface_ray::
                                            invalid_primitive,
                                        hit->prim,
                                        shadow.skip_self),
                                    surface_ray::
                                        invalid_primitive,
                                    surface_ray::
                                        invalid_primitive);
                            $if (any(
                                shadow_transmittance > 0.0f)) {
                                auto evaluation =
                                    evaluate_surface(
                                        surface_tag,
                                        point,
                                        wi,
                                        path_surface_query);
                                // Analytic lights are not part of the
                                // Psycles acceleration structure yet, so
                                // forward BSDF rays cannot hit them. Their
                                // NEE estimator has no competing forward
                                // technique and must carry full weight.
                                Float mis_weight = 1.0f;
                                Float3 unshadowed_contribution =
                                    evaluation.f *
                                    light_radiance *
                                    (mis_weight /
                                     max(
                                         light_pdf,
                                         1.0e-20f));
                                Float roulette_weight =
                                    sample_light_roulette(
                                        unshadowed_contribution,
                                        light_terminate_sample);
                                Float3 contribution =
                                    clamp_contribution(
                                        throughput *
                                            unshadowed_contribution *
                                            shadow_transmittance *
                                            roulette_weight,
                                        path_depth);
                                radiance += contribution;
                                accumulate_light_pass(
                                    split_nee_light(
                                    contribution,
                                    evaluation.f,
                                    evaluation.diffuse_f,
                                    path_diffuse_weight,
                                    path_glossy_weight,
                                    path_depth));
                            };
                        };
                    };
                }

                auto surface_sample = sample_surface(
                    surface_tag,
                    point,
                    bsdf_sample.z,
                    bsdf_sample.xy(),
                    path_surface_query);
                $if (!surface_sample.valid |
                     (surface_sample.evaluation.pdf <=
                      0.0f)) {
                    $break;
                };

                Bool transparent =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_transparent)) != 0u;
                Bool transmission =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_transmission)) != 0u;
                Bool glossy =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_glossy)) != 0u;
                Bool diffuse =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_diffuse)) != 0u;
                Bool singular =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_singular)) != 0u;
                Bool reflection =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_reflection)) != 0u;
                Bool diffuse_reflection =
                    diffuse & reflection & (!transparent);
                Bool glossy_reflection =
                    glossy & reflection & (!transparent);
                Bool material_transmission =
                    transmission & (!transparent);

                auto sampled_diffuse_weight =
                    light_component_ratio(
                        surface_sample.evaluation.diffuse_f,
                        surface_sample.evaluation.f);
                auto sampled_glossy_weight =
                    light_component_ratio(
                        surface_sample.evaluation.f -
                            surface_sample.evaluation.diffuse_f,
                        surface_sample.evaluation.f);
                auto records_first_surface =
                    (path_depth == 0u) & (!transparent);
                path_diffuse_weight = select(
                    path_diffuse_weight,
                    sampled_diffuse_weight,
                    records_first_surface);
                path_glossy_weight = select(
                    path_glossy_weight,
                    sampled_glossy_weight,
                    records_first_surface);

                throughput *=
                    surface_sample.evaluation.f /
                    surface_sample.evaluation.pdf;
                $if (any(luisa::compute::dsl::isnan(
                         throughput)) |
                     any(throughput < 0.0f)) {
                    $break;
                };

                UInt scattered_visibility = select(
                    diffuse_visibility,
                    glossy_visibility,
                    glossy);
                scattered_visibility = select(
                    scattered_visibility,
                    transmission_visibility,
                    transmission);
                ray_visibility = select(
                    scattered_visibility,
                    ray_visibility,
                    transparent);
                ray_events = select(
                    surface_sample.evaluation.events,
                    ray_events |
                        surface_sample.evaluation.events,
                    transparent);
                diffuse_depth += select(
                    0u, 1u, diffuse_reflection);
                glossy_depth += select(
                    0u, 1u, glossy_reflection);
                transmission_depth += select(
                    0u, 1u, material_transmission);
                transparent_depth += select(
                    0u, 1u, transparent);
                path_depth += select(
                    0u, 1u, !transparent);
                terminate_on_next_surface |=
                    transparent &
                    (transparent_depth >=
                     kernel_parameters
                         .transparent_max_bounces);
                terminate_after_transparent |=
                    (!transparent) &
                    ((path_depth >=
                      kernel_parameters.max_bounces) |
                     (diffuse_reflection &
                      (diffuse_depth >=
                       kernel_parameters
                           .max_diffuse_bounces)) |
                     (glossy_reflection &
                      (glossy_depth >=
                       kernel_parameters
                           .max_glossy_bounces)) |
                     (material_transmission &
                      (transmission_depth >=
                       kernel_parameters
                           .max_transmission_bounces)));
                // Cycles does not update forward-MIS state for a
                // transparent bounce. The next emitter remains paired
                // with the most recent non-transparent BSDF technique.
                previous_bsdf_pdf = select(
                    surface_sample.evaluation.pdf,
                    previous_bsdf_pdf,
                    transparent);
                minimum_bsdf_pdf = select(
                    min(
                        minimum_bsdf_pdf,
                        surface_sample.evaluation.pdf),
                    minimum_bsdf_pdf,
                    transparent);
                previous_delta = select(
                    singular,
                    previous_delta,
                    transparent);
                // A transparent Cycles bounce keeps the complete ray line and
                // advances only tmin to the next representable float after
                // the committed distance. A regular surface bounce starts a
                // new ray and uses the conditional triangle-origin policy.
                Float3 next_origin = select(
                    make_surface_ray_origin(
                        surface_sample.wi),
                    ray->origin(),
                    transparent);
                Float3 next_direction = select(
                    surface_sample.wi,
                    ray->direction(),
                    transparent);
                Float next_minimum = select(
                    0.0f,
                    surface_ray::intersection_t_offset(
                        hit->committed_ray_t),
                    transparent);
                Float next_maximum = select(
                    ray_maximum,
                    ray->t_max(),
                    transparent);
                ray_source_instance = hit->inst;
                ray_source_primitive = hit->prim;
                ray_dP = differential_radius;
                ray = make_ray(
                    next_origin,
                    next_direction,
                    next_minimum,
                    next_maximum);
            };

            radiance = select(
                radiance,
                make_float3(0.0f),
                any(luisa::compute::dsl::isnan(radiance)));
            combined_sum +=
                make_float4(radiance, sample_alpha);
            normal_sum +=
                make_float4(sample_normal, 1.0f);
            albedo_sum +=
                make_float4(sample_albedo, 1.0f);
            glossy_color_sum +=
                make_float4(sample_glossy_color, 1.0f);
            transmission_color_sum +=
                make_float4(
                    sample_transmission_color, 1.0f);
            diffuse_direct_sum +=
                make_float4(sample_diffuse_direct, 1.0f);
            diffuse_indirect_sum +=
                make_float4(sample_diffuse_indirect, 1.0f);
            glossy_direct_sum +=
                make_float4(sample_glossy_direct, 1.0f);
            glossy_indirect_sum +=
                make_float4(sample_glossy_indirect, 1.0f);
            transmission_direct_sum +=
                make_float4(
                    sample_transmission_direct, 1.0f);
            transmission_indirect_sum +=
                make_float4(
                    sample_transmission_indirect, 1.0f);
            emission_sum +=
                make_float4(sample_emission, 1.0f);
            environment_sum +=
                make_float4(sample_environment, 1.0f);
            completed += 1u;
        };
        combined.write(pixel, combined_sum);
        normal.write(pixel, normal_sum);
        albedo.write(pixel, albedo_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::diffuse_direct),
            diffuse_direct_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::diffuse_indirect),
            diffuse_indirect_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::glossy_direct),
            glossy_direct_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::glossy_indirect),
            glossy_indirect_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::transmission_direct),
            transmission_direct_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::transmission_indirect),
            transmission_indirect_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(LightPassBuffer::emission),
            emission_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::environment),
            environment_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::glossy_color),
            glossy_color_sum);
        light_passes.write(
            light_pass_base +
                light_pass_index(
                    LightPassBuffer::transmission_color),
            transmission_color_sum);
        sample_count.write(pixel, completed);
    };

    _render_shader = _scene->device.compile(
        kernel,
        luisa::compute::ShaderOption{
            .enable_cache = true,
            .enable_fast_math = false});
}

}// namespace psycles::luisa_backend::detail
