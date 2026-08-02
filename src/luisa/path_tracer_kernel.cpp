#include "path_tracer_internal.h"
#include "cycles_filter_glossy.h"
#include "cycles_integrator_limits.h"
#include "path_kernel_builder.h"

#include <psycles/luisa/pixel_filter.h>
#include <psycles/sampling/light_distribution.h>

#include <cstdlib>
#include <string_view>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool
render_shader_cache_enabled() noexcept {
    const auto *disabled =
        std::getenv(
            "PSYCLES_DISABLE_SHADER_CACHE");
    return disabled == nullptr ||
           std::string_view{disabled} != "1";
}

}// namespace

void LuisaRenderSession::initialize(const RenderSettings &settings) {
    _settings = settings;
    _total_aa_samples = 0u;
    _rendered_samples = 0u;
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
    _volume_guiding_raw =
        _scene->device.create_buffer<luisa::float4>(
            count *
            volume_guiding::raw_pixel_stride);
    _volume_guiding_denoised =
        _scene->device.create_buffer<luisa::uint>(
            count *
            volume_guiding::
                denoised_pixel_stride);
    _volume_guiding_intermediate =
        _scene->device.create_buffer<luisa::uint>(
            count *
            volume_guiding::
                denoised_pixel_stride);
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
    luisa::vector<luisa::float4>
        zeros_volume_guiding_raw(
            count *
            volume_guiding::
                raw_pixel_stride);
    luisa::vector<luisa::uint>
        zeros_volume_guiding_denoised(
            count *
            volume_guiding::
                denoised_pixel_stride);
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
            << _volume_guiding_raw.copy_from(
                   luisa::span{
                       zeros_volume_guiding_raw})
            << _volume_guiding_denoised.copy_from(
                   luisa::span{
                       zeros_volume_guiding_denoised})
            << _volume_guiding_intermediate.copy_from(
                   luisa::span{
                       zeros_volume_guiding_denoised})
            << _path_trace.copy_from(
                   luisa::span{zeros_path_trace})
            << _pixel_filter_table.copy_from(
                   luisa::span{filter_table})
            << synchronize();
    _volume_guiding_filter =
        _scene->volume_metadata.has_volumes()
            ? std::make_unique<
                  VolumeGuidingFilter>(
                  _scene->device)
            : nullptr;

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
            .maximum_volume =
                integrator.volume_bounces,
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
    const auto max_volume_bounces =
        bounce_limits.maximum_volume;
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
    // Cycles' flat-distribution LightManager sets use_direct_light only
    // when the distribution has positive total weight. This is a host-stage
    // capability gate: no distribution means NEE was never attempted,
    // whereas a selected emitter whose position sample fails is an attempted
    // proposal and has a distinct diagnostic trace state.
    const auto direct_light_available =
        scene->light_distribution_count > 0u;
    const auto next_event_estimation =
        _options.next_event_estimation &&
        integrator.direct_light_sampling !=
            contract::DirectLightSampling::
                forward_path_tracing &&
        direct_light_available;
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
    const auto camera_may_be_inside_volume =
        VolumeSceneMetadataComponent{}
            .camera_may_be_inside_volume(
                scene->volume_metadata,
                CameraVolumeBoundsQuery{
                    .projection =
                        camera_projection,
                    .transform =
                        scene->camera.transform,
                    .aspect = camera_aspect,
                    .horizontal_tangent =
                        camera_horizontal_tangent,
                    .vertical_tangent =
                        camera_vertical_tangent,
                    .orthographic_scale =
                        camera_ortho_scale,
                    .shift_x = camera_shift_x,
                    .shift_y = camera_shift_y,
                    .near_clip = camera_near,
                    .aperture_radius =
                        camera_aperture_radius,
                    .focal_distance =
                        camera_focal_distance,
                    .aperture_ratio =
                        camera_aperture_ratio});
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
        .max_volume_bounces =
            max_volume_bounces,
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
    auto light_distribution_sample_callable =
        make_light_distribution_sample_callable(scene);

    auto surface_callables =
        make_surface_callables(scene);
    auto environment_callables =
        make_environment_callables(
            scene,
            light_transport.safe_normalize,
            surface_callables.constant_emission,
            surface_callables.emission);
    auto trace_shadow_callable =
        make_trace_shadow_callable(
            scene, light_transport.safe_normalize);
    const auto path_trace_enabled =
        _options.path_trace.has_value();
    std::shared_ptr<
        const PathVolumeStateComponent>
        volume_state;
    if (scene->volume_metadata.stack_size != 0u) {
        volume_state =
            make_path_volume_state_component();
    }

    PathKernelConfig kernel_config{
        .scene = scene,
        .camera_projection = camera_projection,
        .camera_depth_of_field = camera_depth_of_field,
        .camera_aperture_blades = camera_aperture_blades,
        .camera_aperture_rotation =
            camera_aperture_rotation,
        .next_event_estimation = next_event_estimation,
        .reflective_caustics = reflective_caustics,
        .refractive_caustics = refractive_caustics,
        .path_trace_enabled = path_trace_enabled,
        .volume_stack_size =
            scene->volume_metadata.stack_size,
        .camera_may_be_inside_volume =
            camera_may_be_inside_volume,
        .volume_state = std::move(volume_state),
        .light_transport = std::move(light_transport),
        .light_distribution_sample =
            std::move(light_distribution_sample_callable),
        .surfaces = std::move(surface_callables),
        .environment = std::move(environment_callables),
        .trace_shadow = std::move(trace_shadow_callable)};
    auto kernel = build_path_kernel(kernel_config);

    _render_shader = _scene->device.compile(
        kernel,
        luisa::compute::ShaderOption{
            .enable_cache =
                render_shader_cache_enabled(),
            .enable_fast_math = false});
}

}// namespace psycles::luisa_backend::detail
