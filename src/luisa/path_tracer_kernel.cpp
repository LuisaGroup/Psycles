#include "cycles_filter_glossy.h"
#include "cycles_integrator_limits.h"
#include "path_kernel_builder.h"
#include "path_kernel_executor.h"
#include "sample_dispatch_partition.h"
#include "path_tracer_internal.h"
#include "path_tracer_volume_majorant_scene.h"
#include "virtual_texture_cache.h"
#include "virtual_texture_preflight.h"

#include <psycles/compiler/cycles_transform.h>
#include <psycles/compiler/cycles_camera.h>
#include <psycles/luisa/camera_sampling.h>
#include <psycles/luisa/pixel_filter.h>
#include <psycles/sampling/light_distribution.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool render_shader_cache_enabled() noexcept {
  const auto *disabled = std::getenv("PSYCLES_DISABLE_SHADER_CACHE");
  return disabled == nullptr || std::string_view{disabled} != "1";
}

[[nodiscard]] bool render_shader_driver_optimization_enabled(
    const luisa::compute::Device &device,
    LuisaPathScheduler scheduler) noexcept {
    if (device.backend_name() != "vk") {
        return true;
    }
    if (const auto *enabled =
            std::getenv("PSYCLES_VULKAN_ENABLE_DRIVER_OPTIMIZATION");
        enabled != nullptr && *enabled != '\0') {
        const std::string_view value{enabled};
        return value == "1" || value == "true" || value == "on";
    }
    // The monolithic megakernel retains a runtime-indexed Cycles closure array
    // whose full RADV optimization can consume unbounded host memory on a cold
    // compile. Wavefront schedulers decompose shading into bounded per-stage
    // kernels and benefit directly from full Vulkan driver optimization.
    return scheduler == LuisaPathScheduler::wavefront ||
           scheduler == LuisaPathScheduler::wavefront_staged ||
           scheduler == LuisaPathScheduler::wavefront_graph;
}

}// namespace

void LuisaRenderSession::clear_film_buffers(std::size_t count) {
    const auto diagnostic_layout = path_diagnostic_buffer_layout(_options);
    const auto clear_count = std::max<std::size_t>(
        count, diagnostic_layout.allocation_slot_count);
    _stream << _scene->film_pass_accessor
                   ->clear_film(
                       _combined,
                       _normal,
                       _albedo,
                       _light_passes,
                       _sample_count,
                       _volume_guiding_raw,
                       _volume_guiding_denoised,
                       _volume_guiding_intermediate,
                       _path_trace,
                       static_cast<std::uint32_t>(count),
                       static_cast<std::uint32_t>(
                           diagnostic_layout.allocation_slot_count))
                   .dispatch(clear_count)
            << synchronize();
}

void LuisaRenderSession::prepare_film_buffers(
    std::size_t count,
    const RenderSettings &settings,
    bool clear_now) {
    const auto diagnostic_layout = path_diagnostic_buffer_layout(_options);
    if (count != _allocated_pixel_count ||
        !_combined || !_normal || !_albedo || !_light_passes ||
        !_sample_count || !_volume_guiding_raw ||
        !_volume_guiding_denoised || !_volume_guiding_intermediate) {
        _combined = _scene->device.create_buffer<luisa::float4>(count);
        _normal = _scene->device.create_buffer<luisa::float4>(count);
        _albedo = _scene->device.create_buffer<luisa::float4>(count);
        _light_passes = _scene->device.create_buffer<luisa::float4>(
            count * light_pass_buffer_count);
        _sample_count = _scene->device.create_buffer<luisa::uint>(count);
        _volume_guiding_raw = _scene->device.create_buffer<luisa::float4>(
            count * volume_guiding::raw_pixel_stride);
        _volume_guiding_denoised = _scene->device.create_buffer<luisa::uint>(
            count * volume_guiding::denoised_pixel_stride);
        _volume_guiding_intermediate = _scene->device.create_buffer<luisa::uint>(
            count * volume_guiding::denoised_pixel_stride);
        _allocated_pixel_count = count;
    }
    if (diagnostic_layout.allocation_slot_count != _allocated_path_trace_slots ||
        !_path_trace) {
        _path_trace = _scene->device.create_buffer<luisa::float4>(
            diagnostic_layout.allocation_slot_count);
        _allocated_path_trace_slots = diagnostic_layout.allocation_slot_count;
    }

    const bool rebuild_filter =
        !_pixel_filter_table ||
        settings.pixel_filter != _allocated_filter ||
        settings.filter_width != _allocated_filter_width;
    if (rebuild_filter) {
        const auto generated_filter_table = sampling::make_pixel_filter_table(
            settings.pixel_filter, settings.filter_width);
        luisa::vector<float> filter_table;
        filter_table.reserve(generated_filter_table.size());
        for (const auto value : generated_filter_table) {
            filter_table.emplace_back(value);
        }
        _pixel_filter_table =
            _scene->device.create_buffer<float>(filter_table.size());
        _stream << _pixel_filter_table.copy_from(luisa::span{filter_table});
        _allocated_filter = settings.pixel_filter;
        _allocated_filter_width = settings.filter_width;
    }

    prepare_film_pass_accessor();
    if (clear_now) {
        clear_film_buffers(count);
    }
    if (_scene->volume_metadata.has_volumes()) {
        if (!_volume_guiding_filter) {
            _volume_guiding_filter =
                std::make_unique<VolumeGuidingFilter>(_scene->device);
        }
    } else {
        _volume_guiding_filter.reset();
    }
}

SessionKernelSignature
LuisaRenderSession::update_kernel_parameters(const RenderSettings &settings) {
    auto scene = _scene;
    const auto render_window = _window;
    const auto count = std::max<std::size_t>(pixel_count(), 1u);
    const auto integrator = settings.integrator;
    const auto bounce_limits = cycles_kernel_bounce_limits(
        {.maximum = integrator.max_bounces,
         .minimum = integrator.min_bounces,
         .maximum_diffuse = integrator.diffuse_bounces,
         .maximum_glossy = integrator.glossy_bounces,
         .maximum_transmission = integrator.transmission_bounces,
         .maximum_volume = integrator.volume_bounces,
         .transparent_minimum = integrator.transparent_min_bounces,
         .transparent_maximum = integrator.transparent_max_bounces});
    const auto max_bounces = bounce_limits.maximum;
    const auto min_bounces = bounce_limits.minimum;
    const auto max_diffuse_bounces = bounce_limits.maximum_diffuse;
    const auto max_glossy_bounces = bounce_limits.maximum_glossy;
    const auto max_transmission_bounces = bounce_limits.maximum_transmission;
    const auto max_volume_bounces = bounce_limits.maximum_volume;
    const auto transparent_min_bounces = bounce_limits.transparent_minimum;
    const auto transparent_max_bounces = bounce_limits.transparent_maximum;
    // Blender exposes clamp per RGB channel. Cycles' device kernel
    // compares the sum of absolute RGB components, so scene sync
    // multiplies non-zero UI values by three.
    const auto sample_clamp_direct = integrator.sample_clamp_direct > 0.0f
                                         ? integrator.sample_clamp_direct * 3.0f
                                         : 0.0f;
    const auto sample_clamp_indirect =
        integrator.sample_clamp_indirect > 0.0f
            ? integrator.sample_clamp_indirect * 3.0f
            : 0.0f;
    const auto filter_glossy =
        cycles_filter_glossy_device_scale(integrator.filter_glossy);
    const auto light_inv_rr_threshold =
        !integrator.use_light_tree && integrator.light_sampling_threshold > 0.0f
            ? integrator.film_exposure / integrator.light_sampling_threshold
            : 0.0f;
    const auto reflective_caustics = integrator.reflective_caustics;
    const auto refractive_caustics = integrator.refractive_caustics;
    // Cycles' flat-distribution LightManager sets use_direct_light only
    // when the distribution has positive total weight. This is a host-stage
    // capability gate: no distribution means NEE was never attempted,
    // whereas a selected emitter whose position sample fails is an attempted
    // proposal and has a distinct diagnostic trace state.
    const auto use_light_tree = integrator.use_light_tree &&
                                scene->light_tree_node_count > 0u &&
                                scene->light_tree_emitter_count > 0u;
    const auto direct_light_available =
        use_light_tree ? scene->light_tree_emitter_count > 0u
                       : scene->light_distribution_count > 0u;
    const auto next_event_estimation =
        _options.next_event_estimation &&
        integrator.direct_light_sampling !=
            contract::DirectLightSampling::forward_path_tracing &&
        direct_light_available;
    const auto direct_light_sampling =
        next_event_estimation
            ? integrator.direct_light_sampling
            : contract::DirectLightSampling::forward_path_tracing;
    const auto camera_transform = to_luisa(_camera.transform);
    const auto camera_projection = _camera.projection;
    const auto camera_setup = compiler::make_cycles_camera_projection(
        _camera, settings.full_extent.width, settings.full_extent.height);
    const auto camera_aspect = camera_setup.aspect;
    const auto camera_vertical_tangent = camera_setup.vertical_tangent;
    const auto camera_horizontal_tangent = camera_setup.horizontal_tangent;
    const auto camera_ortho_vertical_span =
        camera_setup.orthographic_vertical_span;
    const auto camera_shift_x = camera_setup.shift_x;
    const auto camera_shift_y = camera_setup.shift_y;
    const auto camera_near = _camera.near_clip;
    const auto camera_far = _camera.far_clip;
    const auto camera_aperture_radius = _camera.aperture_radius;
    const auto camera_focal_distance = _camera.focal_distance;
    const auto camera_aperture_ratio = _camera.aperture_ratio;
    const auto camera_aperture_blades =
        _camera.aperture_blades < 3u ? 0u : _camera.aperture_blades;
    const auto camera_aperture_rotation = _camera.aperture_rotation;
    const auto background = scene->background;
    const auto camera_depth_of_field =
        camera_projection != CameraProjection::panorama &&
        camera_aperture_radius > 0.0f && camera_focal_distance > 0.0f;
    const auto camera_may_be_inside_volume =
        VolumeSceneMetadataComponent{}.camera_may_be_inside_volume(
            scene->volume_metadata,
            CameraVolumeBoundsQuery{
                .projection = camera_projection,
                .transform = _camera.transform,
                .aspect = camera_aspect,
                .horizontal_tangent = camera_horizontal_tangent,
                .vertical_tangent = camera_vertical_tangent,
                .orthographic_vertical_span = camera_ortho_vertical_span,
                .shift_x = camera_shift_x,
                .shift_y = camera_shift_y,
                .near_clip = camera_near,
                .aperture_radius = camera_aperture_radius,
                .focal_distance = camera_focal_distance,
                .aperture_ratio = camera_aperture_ratio});
    const auto pass_alpha_threshold =
        std::clamp(settings.pass_alpha_threshold, 0.0f, 1.0f);
    const auto wavefront_capacity_plan =
        WavefrontFrameCapacityPlan::make(
            count, _options.max_samples_per_dispatch,
            _options.max_pixel_samples_per_dispatch,
            _options.wavefront_frame_capacity);
    // A zero frame request is permitted for schedulers that do not allocate
    // coroutine frames. Keep their otherwise-unused ABI value valid without
    // weakening the nonzero validation of every wavefront scheduler.
    const auto wavefront_frame_capacity =
        wavefront_capacity_plan
            ? wavefront_capacity_plan->effective_capacity
            : 1u;
    _kernel_parameters = RenderKernelParameters{
        .window_x = render_window.x,
        .window_y = render_window.y,
        .window_width = std::max(render_window.width, 1u),
        .full_width = std::max(settings.full_extent.width, 1u),
        .full_height = std::max(settings.full_extent.height, 1u),
        .seed = settings.seed,
        .sobol_sequence_size = _sobol_sequence_size,
        .max_bounces = max_bounces,
        .min_bounces = min_bounces,
        .max_diffuse_bounces = max_diffuse_bounces,
        .max_glossy_bounces = max_glossy_bounces,
        .max_transmission_bounces = max_transmission_bounces,
        .max_volume_bounces = max_volume_bounces,
        .transparent_min_bounces = transparent_min_bounces,
        .transparent_max_bounces = transparent_max_bounces,
        .transparent_background = settings.transparent_background ? 1u : 0u,
        .path_trace_enabled = _options.path_trace ? 1u : 0u,
        .path_trace_pixel_x =
            _options.path_trace ? _options.path_trace->pixel_x : 0u,
        .path_trace_pixel_y =
            _options.path_trace ? _options.path_trace->pixel_y : 0u,
        .path_trace_sample =
            _options.path_trace ? _options.path_trace->sample : 0u,
        .analytic_light_count = scene->light_count,
        .portal_count = scene->portal_count,
        .ambient_occlusion_object_distance_count =
            scene->ambient_occlusion_object_distance_count,
        .wavefront_frame_capacity = wavefront_frame_capacity,
        .shadow_storage_capacity = 1u,
        .shadow_storage_block_size = 1u,
        .ambient_occlusion_bounces = integrator.ambient_occlusion_bounces,
        .ambient_occlusion_factor = integrator.ambient_occlusion_factor,
        .ambient_occlusion_distance = integrator.ambient_occlusion_distance,
        .sample_clamp_direct = sample_clamp_direct,
        .sample_clamp_indirect = sample_clamp_indirect,
        .filter_glossy = filter_glossy,
        .light_inv_rr_threshold = light_inv_rr_threshold,
        .camera_horizontal_tangent = camera_horizontal_tangent,
        .camera_vertical_tangent = camera_vertical_tangent,
        .camera_ortho_vertical_span = camera_ortho_vertical_span,
        .camera_shift_x = camera_shift_x,
        .camera_shift_y = camera_shift_y,
        .camera_near = camera_near,
        .camera_far = camera_far,
        .camera_aperture_radius = camera_aperture_radius,
        .camera_focal_distance = camera_focal_distance,
        .camera_aperture_ratio = camera_aperture_ratio,
        .pass_alpha_threshold = pass_alpha_threshold,
        .background = background,
        .camera_transform = camera_transform,
        .camera_inverse_transform = to_luisa(
            compiler::cycles_inverse_affine_transform(_camera.transform)),
        .camera_raster_to_camera = to_luisa(camera_setup.raster_to_camera),
        .camera_world_to_ndc = to_luisa(camera_setup.world_to_ndc),
        .camera_dx = to_luisa(camera_setup.dx),
        .camera_dy = to_luisa(camera_setup.dy),
        .camera_inv_aperture_ratio = 1.0f / camera_aperture_ratio};

    const auto spectral_mode = contract::effective_spectral_mode(integrator);
    const bool spectral_rendering =
        spectral_mode != contract::SpectralMode::none;
    const bool xyz3_force_collapse = [&]() noexcept {
        if (spectral_mode != contract::SpectralMode::xyz3) {
            return false;
        }
        const char *raw = std::getenv("PSYCLES_XYZ3_FORCE_COLLAPSE");
        return raw != nullptr && std::string_view{raw} == "1";
    }();
    return SessionKernelSignature{
        .camera_projection = camera_projection,
        .camera_depth_of_field = camera_depth_of_field,
        .camera_aperture_blades = camera_aperture_blades,
        .camera_aperture_rotation = camera_aperture_rotation,
        .camera_may_be_inside_volume = camera_may_be_inside_volume,
        .next_event_estimation = next_event_estimation,
        .use_light_tree = use_light_tree,
        .direct_light_sampling = direct_light_sampling,
        .reflective_caustics = reflective_caustics,
        .refractive_caustics = refractive_caustics,
        .has_filter_glossy = integrator.filter_glossy > 0.0f,
        .ambient_occlusion_bounce_approximation =
            integrator.ambient_occlusion_bounces != 0u,
        .transparent_background = settings.transparent_background,
        .transparent_max_bounces = transparent_max_bounces,
        .wavefront_frame_capacity = wavefront_frame_capacity,
        .spectral_rendering = spectral_rendering,
        .spectral_mode = spectral_mode,
        .xyz3_force_collapse = xyz3_force_collapse,
        .caustics_method = integrator.caustics_method,
        .sms_unbiased = integrator.sms_unbiased,
        .sms_max_trials = integrator.sms_max_trials,
        .valid = true};
}

void LuisaRenderSession::initialize(const RenderSettings &settings) {
    const bool tl_active =
        _scene && _scene->device &&
        _scene->device.backend_name() == "vk" &&
        _scene->device.impl()->query("vk_timeline_active") == "1";
    if (tl_active) {
        static_cast<void>(
            _scene->device.impl()->query("vk_timeline_mark:session_init_begin"));
    }
    _settings = settings;
    _total_aa_samples = 0u;
    _rendered_samples = 0u;
    _path_trace_delivered = false;
    _last_render_timings.reset();
    _window = effective_window(settings);
    const auto count = std::max<std::size_t>(pixel_count(), 1u);
    prepare_film_buffers(count, settings, /*clear_now=*/false);

    const auto signature = update_kernel_parameters(settings);
    if (signature.spectral_rendering && !_scene->spectral_tables) {
        _scene->spectral_tables = cycles_spectrum::upload_spectral_tables(
            _scene->device, _stream, _scene->shader_color_space);
    }
    _background_sampling = build_background_sampling_distribution(
        _scene, _stream, _kernel_parameters);
    if (_scene->volume_majorant_plan) {
        auto majorants = VolumeMajorantSceneComponent{}.build(
            _scene, _stream, *_scene->volume_majorant_plan, _kernel_parameters);
        LUISA_ASSERT(majorants.ok(), "Native Cycles volume density bake failed: {}",
                     majorants.diagnostic);
        _volume_majorants = std::move(majorants.runtime);
    } else {
        _volume_majorants.reset();
    }

    auto light_transport = make_light_transport_callables(
        signature.direct_light_sampling);
    auto light_distribution_sample_callable =
        make_light_distribution_sample_callable(_scene);
    auto light_tree_callables = make_light_tree_callables(
        _scene, _scene->volume_metadata.stack_size != 0u);
    auto surface_callables = make_surface_callables(
        _scene, signature.spectral_rendering, signature.spectral_mode);
    const bool needs_initial_shadow_trace =
        _options.scheduler == LuisaPathScheduler::megakernel ||
        _options.scheduler == LuisaPathScheduler::megakernel_per_sample;
    using NullFnBuilder =
        luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>;
    auto shadow_trace_callables =
        needs_initial_shadow_trace
            ? make_shadow_trace_callables(_scene, light_transport.safe_normalize)
            : ShadowTraceCallables{
                  .intersect = nullptr,
                  .shade_surface = EvaluateShadowSurfaceCallable{NullFnBuilder{}},
                  .trace = TraceShadowCallable{NullFnBuilder{}}};
    std::shared_ptr<const PathVolumeStateComponent> volume_state;
    if (_scene->volume_metadata.stack_size != 0u) {
        volume_state = make_path_volume_state_component();
    }
    const auto diagnostic_layout = path_diagnostic_buffer_layout(_options);
    PathKernelConfig kernel_config{
        .scene = _scene,
        .camera_projection = signature.camera_projection,
        .camera_depth_of_field = signature.camera_depth_of_field,
        .camera_aperture_blades = signature.camera_aperture_blades,
        .camera_aperture_rotation = signature.camera_aperture_rotation,
        .next_event_estimation = signature.next_event_estimation,
        .use_light_tree = signature.use_light_tree,
        .reflective_caustics = signature.reflective_caustics,
        .refractive_caustics = signature.refractive_caustics,
        .has_filter_glossy = signature.has_filter_glossy,
        .has_subsurface = _scene->has_subsurface,
        .ambient_occlusion_bounce_approximation =
            signature.ambient_occlusion_bounce_approximation,
        .path_trace_enabled = _options.path_trace.has_value(),
        .surface_closure_count_histogram_enabled =
            static_cast<bool>(_options.surface_closure_count_histogram),
        .surface_closure_count_histogram_base = static_cast<std::uint32_t>(
            diagnostic_layout.surface_closure_count_histogram_base),
        .staged_surface_sorting = _options.staged_surface_sorting,
        .volume_stack_size = _scene->volume_metadata.stack_size,
        .transparent_max_bounces = signature.transparent_max_bounces,
        .camera_may_be_inside_volume = signature.camera_may_be_inside_volume,
        .volume_state = std::move(volume_state),
        .light_transport = std::move(light_transport),
        .light_distribution_sample =
            std::move(light_distribution_sample_callable),
        .light_tree = std::move(light_tree_callables),
        .surfaces = std::move(surface_callables),
        .background_sampling = _background_sampling,
        .volume_majorants = _volume_majorants,
        .intersect_shadow = std::move(shadow_trace_callables.intersect),
        .shade_shadow_surface =
            std::move(shadow_trace_callables.shade_surface),
        .trace_shadow = std::move(shadow_trace_callables.trace),
        .enable_virtual_texture_cache =
            _options.virtual_texture_cache.enabled,
        .spectral_rendering = signature.spectral_rendering,
        .spectral_mode = signature.spectral_mode,
        .xyz3_force_collapse = signature.xyz3_force_collapse,
        .caustics_method = signature.caustics_method,
        .sms_unbiased = signature.sms_unbiased,
        .sms_max_trials = signature.sms_max_trials,
        .transparent_background = signature.transparent_background};

    PathKernelExecutorConfig executor_config{
        .scheduler = _options.scheduler,
        .render_pixel_count = static_cast<std::uint32_t>(count),
        .wavefront_frame_capacity = signature.wavefront_frame_capacity,
        .wavefront_execution_block_size = _options.wavefront_execution_block_size,
        .wavefront_graph_worker_count = _options.wavefront_graph_worker_count,
        .wavefront_graph_selective_scheduling =
            _options.wavefront_graph_selective_scheduling,
        .wavefront_graph_refill_threshold =
            _options.wavefront_graph_refill_threshold,
        .wavefront_counter_readback_batch_size =
            _options.wavefront_counter_readback_batch_size,
        .wavefront_counter_readback_pipeline_depth =
            _options.wavefront_counter_readback_pipeline_depth,
        .wavefront_tail_megakernel_threshold =
            _options.wavefront_tail_megakernel_threshold,
        .staged_direct_light_queue = _options.staged_direct_light_queue,
        .persistent_worker_count = _options.persistent_worker_count,
        .persistent_block_size = _options.persistent_block_size,
        .persistent_fetch_size = _options.persistent_fetch_size,
        .persistent_shared_memory_soa = _options.persistent_shared_memory_soa,
        .persistent_global_memory_extension =
            _options.persistent_global_memory_extension,
        .shader_option = {
            .enable_cache = render_shader_cache_enabled(),
            .enable_fast_math = _options.enable_fast_math,
            .assume_no_packed_textures = !_scene->may_have_packed_textures,
            .enable_driver_optimization =
                render_shader_driver_optimization_enabled(
                    _scene->device, _options.scheduler)}};

    PreflightOverlapHandle preflight_handle;
    const bool vt_sched =
        (_options.scheduler == LuisaPathScheduler::wavefront ||
         _options.scheduler == LuisaPathScheduler::wavefront_staged);
    const bool run_preflight =
        vt_sched && _options.virtual_texture_cache.enable_preflight &&
        _scene && _scene->virtual_texture_cache &&
        _scene->virtual_texture_cache->has_virtual_images() &&
        !_scene->virtual_texture_cache->eager_fallback_active();
    if (run_preflight) {
        if (tl_active) {
            static_cast<void>(
                _scene->device.impl()->query("vk_timeline_mark:preflight_begin"));
        }
        preflight_handle = start_vt_preflight(
            _scene, signature, _pixel_filter_table, _kernel_parameters,
            kernel_config.light_transport.safe_normalize,
            VtPreflightConfig{
                .overlap_pso = _options.virtual_texture_cache.preflight_overlap_pso,
                .stride = _options.virtual_texture_cache.preflight_stride,
                .bounces = _options.virtual_texture_cache.preflight_bounces,
                .width = _window.width,
                .height = _window.height,
                .shader_option = executor_config.shader_option});
        executor_config.on_before_register_extensions = [&preflight_handle]() {
            preflight_handle.join();
        };
    }

    bool film_cleared = false;
    executor_config.on_after_coroutine_build = [this, count, &film_cleared]() {
        // Clear film after coroutine lowering (while GPU is still in P3) so
        // the subsequent shader-compile phase fits within the P3->P5 timeout.
        clear_film_buffers(count);
        film_cleared = true;
    };

    const auto pso_t0 = std::chrono::steady_clock::now();
    _render_executor = build_path_kernel_executor(
        _scene->device, kernel_config, executor_config);
    const auto pso_t1 = std::chrono::steady_clock::now();
    preflight_handle.join();
    if (!film_cleared) {
        clear_film_buffers(count);
    }
    if (tl_active) {
        static_cast<void>(
            _scene->device.impl()->query("vk_timeline_mark:pso_compile_end"));
    }
    if (_scene && _scene->virtual_texture_cache) {
        const double pso_ms =
            std::chrono::duration<double, std::milli>(pso_t1 - pso_t0).count();
        _scene->virtual_texture_cache->set_pso_compile_ms(pso_ms);
    }
    _kernel_signature = signature;
}

bool LuisaRenderSession::update_camera(
    const CameraDesc &camera,
    const RenderSettings &settings) {
    _camera = camera;
    _cancelled.store(false);
    const auto saved_settings = _settings;
    const auto saved_window = _window;
    const auto saved_parameters = _kernel_parameters;
    _settings = settings;
    _window = effective_window(settings);
    const auto count = std::max<std::size_t>(pixel_count(), 1u);
    const auto signature = update_kernel_parameters(settings);
    if (_kernel_signature.valid && signature == _kernel_signature) {
        _total_aa_samples = 0u;
        _rendered_samples = 0u;
        _path_trace_delivered = false;
        _last_render_timings.reset();
        prepare_film_buffers(count, settings);
        return true;
    }
    _settings = saved_settings;
    _window = saved_window;
    _kernel_parameters = saved_parameters;
    initialize(settings);
    return true;
}

}// namespace psycles::luisa_backend::detail
