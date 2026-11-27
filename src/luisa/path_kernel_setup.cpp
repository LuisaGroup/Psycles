#include "path_kernel_builder.h"
#include "path_tracer_cycles_svm_emission.h"

#include <psycles/luisa/analytic_light_sampling.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>

#include <utility>

namespace psycles::luisa_backend::detail {

PathKernelInvocation
begin_path_kernel(const PathKernelConfig &config,
                  PathFilmAccumulation film_accumulation,
                  UInt pixel,
                  const BufferFloat4 &combined,
                  const BufferFloat4 &normal,
                  const BufferFloat4 &albedo,
                  const BufferFloat4 &light_passes,
                  const BufferUInt &sample_count,
                  const BufferFloat4 &volume_guiding_raw,
                  const BufferUInt &volume_guiding_denoised,
                  const BufferFloat4 &path_trace,
                  const UInt &sample_first,
                  const BufferFloat4 &sobol_table,
                  const BufferFloat &filter_table,
                  const Var<RenderKernelParameters> &parameters) noexcept {
    const auto zero_init_film =
        film_accumulation == PathFilmAccumulation::atomic ||
        film_accumulation == PathFilmAccumulation::warp_reduce;
    const auto film_value = [zero_init_film](
                                const BufferFloat4 &buffer,
                                const UInt &index) noexcept {
        return zero_init_film
                   ? make_float4(0.0f)
                   : buffer.read(index);
    };
    UInt local_x = pixel % parameters.window_width;
    UInt local_y = pixel / parameters.window_width;
    UInt full_x = local_x + parameters.window_x;
    UInt full_y = local_y + parameters.window_y;
    Float4 combined_sum = film_value(combined, pixel);
    Float4 normal_sum = film_value(normal, pixel);
    Float4 albedo_sum = film_value(albedo, pixel);
    UInt light_pass_base = pixel * light_pass_buffer_count;
    Float4 diffuse_direct_sum = film_value(light_passes,
        light_pass_base + light_pass_index(LightPassBuffer::diffuse_direct));
    Float4 diffuse_indirect_sum = film_value(light_passes,
        light_pass_base + light_pass_index(LightPassBuffer::diffuse_indirect));
    Float4 glossy_direct_sum = film_value(light_passes,
        light_pass_base + light_pass_index(LightPassBuffer::glossy_direct));
    Float4 glossy_indirect_sum = film_value(light_passes,
        light_pass_base + light_pass_index(LightPassBuffer::glossy_indirect));
    Float4 transmission_direct_sum = film_value(light_passes,
        light_pass_base +
        light_pass_index(LightPassBuffer::transmission_direct));
    Float4 transmission_indirect_sum = film_value(light_passes,
        light_pass_base +
        light_pass_index(LightPassBuffer::transmission_indirect));
    Float4 volume_direct_sum = film_value(light_passes,
        light_pass_base +
        light_pass_index(
            LightPassBuffer::volume_direct));
    Float4 volume_indirect_sum = film_value(light_passes,
        light_pass_base +
        light_pass_index(
            LightPassBuffer::volume_indirect));
    Float4 emission_sum = film_value(light_passes,
        light_pass_base + light_pass_index(LightPassBuffer::emission));
    Float4 environment_sum = film_value(light_passes,
        light_pass_base + light_pass_index(LightPassBuffer::environment));
    Float4 glossy_color_sum = film_value(light_passes,
        light_pass_base + light_pass_index(LightPassBuffer::glossy_color));
    Float4 transmission_color_sum = film_value(light_passes,
        light_pass_base +
        light_pass_index(LightPassBuffer::transmission_color));
    UInt completed = zero_init_film
                         ? 0u
                         : sample_count.read(pixel);
    UInt volume_guiding_raw_base = 0u;
    Float4 volume_guiding_scatter_sum =
        make_float4(0.0f);
    Float4 volume_guiding_transmit_sum =
        make_float4(0.0f);
    Float4 volume_guiding_optical_depth_sum =
        make_float4(0.0f);
    Float3 volume_guiding_scattered_radiance =
        make_float3(0.0f);
    Float3 volume_guiding_transmitted_radiance =
        make_float3(0.0f);
    if (config.volume_state) {
        volume_guiding_raw_base =
            pixel *
            volume_guiding::raw_pixel_stride;
        volume_guiding_scatter_sum =
            film_value(volume_guiding_raw,
                volume_guiding_raw_base +
                volume_guiding::
                    raw_scatter_slot);
        volume_guiding_transmit_sum =
            film_value(volume_guiding_raw,
                volume_guiding_raw_base +
                volume_guiding::
                    raw_transmit_slot);
        volume_guiding_optical_depth_sum =
            film_value(volume_guiding_raw,
                volume_guiding_raw_base +
                volume_guiding::
                    optical_depth_slot);
        const auto denoised_base =
            pixel *
            volume_guiding::
                denoised_pixel_stride;
        volume_guiding_scattered_radiance =
            volume_guiding::decode_rgbe(
                volume_guiding_denoised.read(
                    denoised_base +
                    volume_guiding::
                        denoised_scatter_slot));
        volume_guiding_transmitted_radiance =
            volume_guiding::decode_rgbe(
                volume_guiding_denoised.read(
                    denoised_base +
                    volume_guiding::
                        denoised_transmit_slot));
    }
    SurfaceQuery surface_query{
        .lobe_mask = static_cast<std::uint32_t>(
            contract::event_diffuse | contract::event_glossy |
            contract::event_singular | contract::event_reflection |
            contract::event_transmission | contract::event_transparent),
        .transport_mode =
            static_cast<std::uint32_t>(contract::TransportMode::radiance),
        .glossy_filter_roughness = 0.0f,
        .reflective_caustics = true,
        .refractive_caustics = true};
    return {config,
            film_accumulation,
            combined,
            normal,
            albedo,
            light_passes,
            sample_count,
            volume_guiding_raw,
            volume_guiding_denoised,
            path_trace,
            sample_first,
            sobol_table,
            filter_table,
            parameters,
            std::move(pixel),
            std::move(local_x),
            std::move(local_y),
            std::move(full_x),
            std::move(full_y),
            std::move(combined_sum),
            std::move(normal_sum),
            std::move(albedo_sum),
            std::move(light_pass_base),
            std::move(diffuse_direct_sum),
            std::move(diffuse_indirect_sum),
            std::move(glossy_direct_sum),
            std::move(glossy_indirect_sum),
            std::move(transmission_direct_sum),
            std::move(transmission_indirect_sum),
            std::move(volume_direct_sum),
            std::move(volume_indirect_sum),
            std::move(emission_sum),
            std::move(environment_sum),
            std::move(glossy_color_sum),
            std::move(transmission_color_sum),
            std::move(completed),
            std::move(volume_guiding_raw_base),
            std::move(volume_guiding_scatter_sum),
            std::move(volume_guiding_transmit_sum),
            std::move(
                volume_guiding_optical_depth_sum),
            std::move(
                volume_guiding_scattered_radiance),
            std::move(
                volume_guiding_transmitted_radiance),
            std::move(surface_query)};
}

Float3 PathKernelInvocation::clamp_contribution(Float3 contribution,
                                                UInt depth) const noexcept {
    return config.light_transport.clamp_light_contribution(
        contribution,
        depth,
        parameters.sample_clamp_direct,
        parameters.sample_clamp_indirect);
}

Float3 PathKernelInvocation::
clamp_emission_contribution(
    Float3 contribution,
    UInt path_depth) const noexcept {
    // Cycles passes `path.bounce - 1` to film_clamp_light for forward
    // emitters (surface, lamp, background, and volume). Only its `> 0`
    // classification matters; avoid unsigned underflow at the camera.
    const auto clamp_depth =
        select(0u, 1u, path_depth > 1u);
    return clamp_contribution(
        contribution, clamp_depth);
}

Float PathKernelInvocation::sample_light_roulette(
    Float3 unshadowed_contribution, Float random) const noexcept {
    return config.light_transport.light_sample_roulette_weight(
        unshadowed_contribution, random, parameters.light_inv_rr_threshold);
}

Float PathKernelInvocation::
volume_guiding_majorant_optical_depth()
    const noexcept {
    // Cycles GPU reads this raw running statistic from the shared render
    // buffer. A per-sample invocation must do the same: its local field holds
    // only the contribution that will later be atomically added, not the
    // history. The serial pixel owner instead carries the history locally so
    // later iterations of its sample loop observe earlier samples exactly.
    auto history = volume_guiding_optical_depth_sum;
    if (film_accumulation == PathFilmAccumulation::atomic) {
        history = volume_guiding_raw.read(
            volume_guiding_raw_base +
            volume_guiding::optical_depth_slot);
    }
    return select(
        std::numeric_limits<float>::max(),
        history.x /
            max(
                history.y,
                1.0f),
        history.y >
            0.0f);
}

SurfaceEvaluation SurfaceShadingState::evaluate_light(
    PathKernelInvocation &,
    UInt,
    const SurfacePoint &,
    Float3 outgoing,
    const SurfaceQuery &query,
    UInt shader_flags) const noexcept {
    LUISA_ASSERT(populated_surface != nullptr,
                 "Native surface evaluation requires populated surface state.");
    return populated_surface->evaluate_light(
        Expr<luisa::float3>{outgoing.expression()},
        {.surface = query, .shader_flags = shader_flags});
}

SurfaceSample SurfaceShadingState::sample(
    PathKernelInvocation &,
    UInt,
    const SurfacePoint &,
    Float u_lobe,
    Float2 u_direction,
    const SurfaceQuery &query) const noexcept {
    LUISA_ASSERT(populated_surface != nullptr,
                 "Native surface sampling requires populated surface state.");
    return populated_surface->sample(
        Expr<float>{u_lobe.expression()},
        Expr<luisa::float2>{u_direction.expression()},
        query);
}

SurfaceSampleTrace SurfaceShadingState::sample_trace(
    PathKernelInvocation &,
    UInt,
    const SurfacePoint &,
    Float u_lobe,
    Float2 u_direction,
    const SurfaceQuery &query) const noexcept {
    LUISA_ASSERT(populated_surface != nullptr,
                 "Native surface sample trace requires populated surface state.");
    return populated_surface->sample_trace(
        Expr<float>{u_lobe.expression()},
        Expr<luisa::float2>{u_direction.expression()},
        query);
}

Float3 PathKernelInvocation::constant_environment() const noexcept {
    return constant_environment_emission(*config.scene, parameters.background);
}

Float4 PathKernelInvocation::constant_environment_spectral(
    const cycles_spectrum::SpectralContext &spectral) const noexcept {
    return constant_environment_emission_spectral(
        *config.scene, parameters.background, spectral);
}

UInt PathSampleContext::traversal_ray_visibility() const noexcept {
    return cycles_path_state::to_contract_traversal_visibility(
        cycles_path_visibility);
}

UInt PathSampleContext::shader_ray_visibility() const noexcept {
    return cycles_path_state::to_contract_shader_visibility(
        cycles_path_visibility);
}

Bool PathSampleContext::single_pass_recorded() const noexcept {
    return (path_flags & cycles_path_state::flag_single_pass_done) != 0u;
}

Bool PathSampleContext::mis_competition_skipped() const noexcept {
    return (path_flags & cycles_path_state::flag_mis_skip) != 0u;
}

Bool PathSampleContext::terminate_after_transparent_requested() const noexcept {
    return (path_flags &
            cycles_path_state::flag_terminate_after_transparent) != 0u;
}

Bool PathSampleContext::terminate_on_next_surface_requested() const noexcept {
    return (path_flags & cycles_path_state::flag_terminate_on_next_surface) !=
           0u;
}

Float PathSampleContext::continuation_terminate_sample() const noexcept {
    const auto value = cycles_sampler::sample_1d(
        invocation.sobol_table,
        invocation.parameters.sobol_sequence_size,
        sample_index,
        rng_hash,
        cycles_sampler::path_state_dimension(
            cycles_rng_offset,
            sampling::tabulated_sobol::terminate_dimension));
    value.set_name("path_continuation_terminate_sample");
    return value;
}

Float3 PathSampleContext::surface_bsdf_sample() const noexcept {
    return cycles_sampler::sample_3d(
        invocation.sobol_table, invocation.parameters.sobol_sequence_size,
        sample_index, rng_hash,
        cycles_sampler::path_state_dimension(
            cycles_rng_offset, sampling::tabulated_sobol::surface_bsdf_dimension));
}

Float3 PathSampleContext::trace_uint32(UInt value) const noexcept {
    return make_float3(
        cast<float>(value & 0xffffu), cast<float>(value >> 16u), 0.0f);
}

void PathSampleContext::trace_write(UInt slot, Float3 value) const noexcept {
    if (invocation.config.path_trace_enabled) {
        $if(path_trace_active) {
            invocation.path_trace.write(slot, make_float4(value, 1.0f));
        };
    }
}

void PathSampleContext::trace_write_global(path_trace_schema::GlobalSlot slot,
                                           Float3 value) const noexcept {
    trace_write(UInt{path_trace_schema::index(slot)}, value);
}

void PathSampleContext::trace_write_event(UInt event,
                                          path_trace_schema::EventSlot slot,
                                          Float3 value) const noexcept {
    $if(event < path_trace_schema::max_events) {
        trace_write(path_trace_schema::global_slot_count +
                        event * path_trace_schema::event_slot_count +
                        static_cast<std::uint32_t>(slot),
                    value);
    };
}

void PathSampleContext::trace_write_shadow_event(
    UInt event,
    path_trace_schema::ShadowEventSlot slot,
    Float3 value) const noexcept {
    $if(event < path_trace_schema::max_events) {
        trace_write(path_trace_schema::shadow_event_base +
                        event * path_trace_schema::shadow_event_slot_count +
                        static_cast<std::uint32_t>(slot),
                    value);
    };
}

void PathSampleContext::trace_write_forward_event(
    UInt event,
    path_trace_schema::ForwardEventSlot slot,
    Float3 value) const noexcept {
    $if(event < path_trace_schema::max_events) {
        trace_write(path_trace_schema::forward_event_base +
                        event * path_trace_schema::forward_event_slot_count +
                        static_cast<std::uint32_t>(slot),
                    value);
    };
}

void PathSampleContext::trace_write_closure(UInt event,
                                            std::uint32_t closure,
                                            std::uint32_t field,
                                            Float3 value) const noexcept {
    $if(event < path_trace_schema::max_events) {
        trace_write(path_trace_schema::global_slot_count +
                        event * path_trace_schema::event_slot_count +
                        static_cast<std::uint32_t>(
                            path_trace_schema::EventSlot::closure_base) +
                        closure * 3u + field,
                    value);
    };
}

void PathSampleContext::record_surface_closure_count(
    UInt count) const noexcept {
    static_assert(
        luisa_surface_closure_count_histogram_max ==
        maximum_surface_closure_capacity);
    const auto &config = invocation.config;
    if (!config.surface_closure_count_histogram_enabled) {
        return;
    }

    // Four independent float lanes preserve exact unit increments up to four
    // times the IEEE-754 consecutive-integer range per bin. The lane map is a
    // deterministic projection of the logical (pixel, sample) identity; host
    // chunking and scheduler order can only permute atomic arrival order and
    // therefore cannot change the lane sums.
    const auto bin = min(
        count, luisa_surface_closure_count_histogram_max);
    const auto lane = (invocation.pixel + sample_index) & 3u;
    auto destination = invocation.path_trace.atomic(
        config.surface_closure_count_histogram_base + bin);
    $if (lane == 0u) {
        destination.x.fetch_add(1.0f);
    }
    $elif (lane == 1u) {
        destination.y.fetch_add(1.0f);
    }
    $elif (lane == 2u) {
        destination.z.fetch_add(1.0f);
    }
    $else {
        destination.w.fetch_add(1.0f);
    };
}


PathSampleContext begin_path_sample(PathKernelInvocation &invocation,
                                    const UInt &sample_offset) noexcept {
    const auto &sample_first = invocation.sample_first;
    const auto &sobol_table = invocation.sobol_table;
    const auto &filter_table = invocation.filter_table;
    const auto &kernel_parameters = invocation.parameters;
    const auto &full_x = invocation.full_x;
    const auto &full_y = invocation.full_y;
    const auto &path_trace = invocation.path_trace;
    const auto &config = invocation.config;
    const auto camera_projection = config.camera_projection;
    const auto camera_depth_of_field = config.camera_depth_of_field;
    const auto camera_aperture_blades = config.camera_aperture_blades;
    const auto camera_aperture_rotation = config.camera_aperture_rotation;
    const auto path_trace_enabled = config.path_trace_enabled;
    const auto &safe_normalize = config.light_transport.safe_normalize;
    UInt sample_index = sample_first + sample_offset;
    const auto camera_dimensions = sample_camera_dimensions(
        sobol_table, kernel_parameters, full_x, full_y, sample_index);
    UInt cycles_y = camera_dimensions.cycles_y;
    UInt rng_hash = camera_dimensions.rng_hash;
    Float2 filter_sample = camera_dimensions.filter_sample;
    Float3 lens_time_sample = camera_dimensions.lens_time_sample;
    Bool path_trace_active = false;
    if (path_trace_enabled) {
        path_trace_active =
            (kernel_parameters.path_trace_enabled != 0u) &
            (full_x == kernel_parameters.path_trace_pixel_x) &
            (cycles_y == kernel_parameters.path_trace_pixel_y) &
            (sample_index == kernel_parameters.path_trace_sample);
    }
    auto trace_write = [&](UInt slot, Float3 value) noexcept {
        if (path_trace_enabled) {
            $if(path_trace_active) {
                path_trace.write(slot, make_float4(value, 1.0f));
            };
        }
    };
    auto trace_write_global = [&](path_trace_schema::GlobalSlot slot,
                                  Float3 value) noexcept {
        trace_write(UInt{path_trace_schema::index(slot)}, value);
    };
    auto camera_ray = construct_camera_ray(filter_table,
                                           kernel_parameters,
                                           full_x,
                                           full_y,
                                           camera_dimensions,
                                           camera_projection,
                                           camera_depth_of_field,
                                           camera_aperture_blades,
                                           camera_aperture_rotation,
                                           safe_normalize);
    Var<luisa::compute::Ray> ray = std::move(camera_ray.ray);
    Float ray_dP = camera_ray.differential_position;
    Float ray_dD = camera_ray.differential_direction;
    if (path_trace_enabled) {
        trace_write_global(
            path_trace_schema::GlobalSlot::header,
            make_float3(static_cast<float>(path_trace_schema::version),
                        cast<float>(full_x),
                        cast<float>(cycles_y)));
        trace_write_global(path_trace_schema::GlobalSlot::rng,
                           make_float3(cast<float>(sample_index),
                                       cast<float>(rng_hash & 0xffffu),
                                       cast<float>(rng_hash >> 16u)));
        trace_write_global(path_trace_schema::GlobalSlot::filter,
                           make_float3(filter_sample, 0.0f));
        trace_write_global(path_trace_schema::GlobalSlot::lens_time,
                           camera_depth_of_field ? lens_time_sample
                                                 : make_float3(0.0f));
        trace_write_global(path_trace_schema::GlobalSlot::ray_p, ray->origin());
        trace_write_global(path_trace_schema::GlobalSlot::ray_d,
                           ray->direction());
        trace_write_global(path_trace_schema::GlobalSlot::ray_range,
                           make_float3(ray->t_min(), ray->t_max(), 0.5f));
    }
    UInt ray_source_object = surface_ray::invalid_primitive;
    UInt ray_source_primitive = surface_ray::invalid_primitive;

    Float3 radiance = make_float3(0.0f);
    const auto mode = config.effective_spectral_mode();
    auto throughput = cycles_spectrum::Spectrum::one(mode);
    std::optional<Float> spectral_u_lambda;
    std::optional<Float3> spectral_pdf_ratio;
    if (mode != contract::SpectralMode::none) {
        spectral_u_lambda.emplace(cycles_sampler::sample_1d(
            sobol_table, kernel_parameters.sobol_sequence_size,
            sample_index, rng_hash ^ 0x51a7b9d3u, 0u));
    }
    if (config.xyz3_rough_dispersion()) {
        spectral_pdf_ratio.emplace(make_float3(1.0f));
    }
    Float3 sample_normal = make_float3(0.0f);
    Float3 sample_albedo = make_float3(0.0f);
    Float3 sample_glossy_color = make_float3(0.0f);
    Float3 sample_transmission_color = make_float3(0.0f);
    Float3 sample_diffuse_direct = make_float3(0.0f);
    Float3 sample_diffuse_indirect = make_float3(0.0f);
    Float3 sample_glossy_direct = make_float3(0.0f);
    Float3 sample_glossy_indirect = make_float3(0.0f);
    Float3 sample_transmission_direct = make_float3(0.0f);
    Float3 sample_transmission_indirect = make_float3(0.0f);
    Float3 sample_volume_direct = make_float3(0.0f);
    Float3 sample_volume_indirect = make_float3(0.0f);
    Float3 sample_emission = make_float3(0.0f);
    Float3 sample_environment = make_float3(0.0f);
    Float3 volume_guiding_scatter =
        make_float3(0.0f);
    Float3 volume_guiding_transmit =
        make_float3(0.0f);
    Float sample_transparency = 0.0f;
    Float previous_bsdf_pdf = 0.0f;
    Float3 previous_mis_origin_normal = make_float3(0.0f);
    Float previous_light_tree_dt = 0.0f;
    Float minimum_bsdf_pdf = std::numeric_limits<float>::max();
    Float continuation_probability = 1.0f;
    Float3 path_diffuse_weight = make_float3(0.0f);
    Float3 path_glossy_weight = make_float3(0.0f);
    UInt ray_events = 0u;
    UInt diffuse_depth = 0u;
    UInt glossy_depth = 0u;
    UInt transparent_depth = 0u;
    UInt transmission_depth = 0u;
    UInt portal_depth = 0u;
    UInt path_depth = 0u;
    // PATH_RAY_SINGLE_PASS_DONE is not an initial camera-path property.
    // Cycles sets it only when a surface actually passes the film alpha
    // threshold and writes the first single-value data passes.
    const auto initial_cycles_path_state = cycles_path_state::initial_state();
    UInt path_flags = initial_cycles_path_state.flag;
    if (mode == contract::SpectralMode::xyz3 && config.xyz3_force_collapse) {
        LUISA_ASSERT(config.scene && config.scene->spectral_tables &&
                         spectral_u_lambda.has_value(),
                     "xyz3_force_collapse requires SpectralTablesGpu.");
        const auto decoded = cycles_spectrum::xyz3_decode_u(*spectral_u_lambda);
        const auto lambdas = cycles_spectrum::sample_xyz3_lambdas(
            *config.scene->spectral_tables, decoded.u);
        const auto h_lambda =
            cycles_spectrum::xyz3_hero(lambdas, decoded.hero_index);
        throughput = cycles_spectrum::Spectrum::from_xyz3(
            cycles_spectrum::xyz3_collapse_weight(
                *config.scene->spectral_tables, h_lambda));
        path_flags |= cycles_path_state::flag_spectral_collapsed;
    }
    UInt cycles_path_visibility = initial_cycles_path_state.visibility;
    UInt cycles_rng_offset = initial_cycles_path_state.rng_offset;
    auto volume =
        make_disabled_path_volume_state();
    if (config.volume_state) {
        volume =
            config.volume_state->initialize(
                config.scene,
                ray->origin(),
                camera_visibility,
                config.volume_stack_size,
                config
                    .camera_may_be_inside_volume);
    }
    UInt volume_bounce = 0u;
    UInt volume_bounds_bounce = 0u;
    Float optical_depth = 0.0f;
    Bool pending_subsurface_exit = false;
    PendingSubsurfaceHit pending_subsurface_hit{
        .instance = surface_ray::invalid_primitive,
        .primitive = surface_ray::invalid_primitive,
        .barycentric = make_float2(0.0f),
        .committed_ray_t = 0.0f};
    return {invocation,
            std::move(sample_index),
            std::move(cycles_y),
            std::move(rng_hash),
            std::move(filter_sample),
            std::move(lens_time_sample),
            std::move(path_trace_active),
            std::move(ray),
            std::move(ray_dP),
            std::move(ray_dD),
            std::move(ray_source_object),
            std::move(ray_source_primitive),
            std::move(radiance),
            std::move(throughput),
            std::move(spectral_u_lambda),
            std::move(spectral_pdf_ratio),
            std::move(sample_normal),
            std::move(sample_albedo),
            std::move(sample_glossy_color),
            std::move(sample_transmission_color),
            std::move(sample_diffuse_direct),
            std::move(sample_diffuse_indirect),
            std::move(sample_glossy_direct),
            std::move(sample_glossy_indirect),
            std::move(sample_transmission_direct),
            std::move(sample_transmission_indirect),
            std::move(sample_volume_direct),
            std::move(sample_volume_indirect),
            std::move(sample_emission),
            std::move(sample_environment),
            std::move(volume_guiding_scatter),
            std::move(volume_guiding_transmit),
            std::move(sample_transparency),
            std::move(previous_bsdf_pdf),
            std::move(previous_mis_origin_normal),
            std::move(previous_light_tree_dt),
            std::move(minimum_bsdf_pdf),
            std::move(continuation_probability),
            std::move(path_diffuse_weight),
            std::move(path_glossy_weight),
            std::move(ray_events),
            std::move(diffuse_depth),
            std::move(glossy_depth),
            std::move(transparent_depth),
            std::move(transmission_depth),
            std::move(portal_depth),
            std::move(path_depth),
            std::move(path_flags),
            std::move(cycles_path_visibility),
            std::move(cycles_rng_offset),
            std::move(volume),
            std::move(volume_bounce),
            std::move(volume_bounds_bounce),
            std::move(optical_depth),
            std::move(pending_subsurface_exit),
            std::move(pending_subsurface_hit)};
}

void reset_path_sample(PathSampleContext &sample,
                       const UInt &sample_offset) noexcept {
    auto &invocation = sample.invocation;
    const auto &sample_first = invocation.sample_first;
    const auto &sobol_table = invocation.sobol_table;
    const auto &filter_table = invocation.filter_table;
    const auto &kernel_parameters = invocation.parameters;
    const auto &full_x = invocation.full_x;
    const auto &full_y = invocation.full_y;
    const auto &path_trace = invocation.path_trace;
    const auto &config = invocation.config;
    const auto camera_projection = config.camera_projection;
    const auto camera_depth_of_field = config.camera_depth_of_field;
    const auto camera_aperture_blades = config.camera_aperture_blades;
    const auto camera_aperture_rotation = config.camera_aperture_rotation;
    const auto path_trace_enabled = config.path_trace_enabled;
    const auto &safe_normalize = config.light_transport.safe_normalize;

    sample.sample_index = sample_first + sample_offset;
    const auto camera_dimensions = sample_camera_dimensions(
        sobol_table, kernel_parameters, full_x, full_y, sample.sample_index);
    sample.cycles_y = camera_dimensions.cycles_y;
    sample.rng_hash = camera_dimensions.rng_hash;
    sample.filter_sample = camera_dimensions.filter_sample;
    sample.lens_time_sample = camera_dimensions.lens_time_sample;
    sample.path_trace_active = false;
    if (path_trace_enabled) {
        sample.path_trace_active =
            (kernel_parameters.path_trace_enabled != 0u) &
            (full_x == kernel_parameters.path_trace_pixel_x) &
            (sample.cycles_y == kernel_parameters.path_trace_pixel_y) &
            (sample.sample_index == kernel_parameters.path_trace_sample);
    }
    auto camera_ray = construct_camera_ray(filter_table,
                                           kernel_parameters,
                                           full_x,
                                           full_y,
                                           camera_dimensions,
                                           camera_projection,
                                           camera_depth_of_field,
                                           camera_aperture_blades,
                                           camera_aperture_rotation,
                                           safe_normalize);
    sample.ray = std::move(camera_ray.ray);
    sample.ray_dP = camera_ray.differential_position;
    sample.ray_dD = camera_ray.differential_direction;
    if (path_trace_enabled) {
        auto trace_write = [&](UInt slot, Float3 value) noexcept {
            $if(sample.path_trace_active) {
                path_trace.write(slot, make_float4(value, 1.0f));
            };
        };
        auto trace_write_global = [&](path_trace_schema::GlobalSlot slot,
                                      Float3 value) noexcept {
            trace_write(UInt{path_trace_schema::index(slot)}, value);
        };
        trace_write_global(
            path_trace_schema::GlobalSlot::header,
            make_float3(static_cast<float>(path_trace_schema::version),
                        cast<float>(full_x),
                        cast<float>(sample.cycles_y)));
        trace_write_global(path_trace_schema::GlobalSlot::rng,
                           make_float3(cast<float>(sample.sample_index),
                                       cast<float>(sample.rng_hash & 0xffffu),
                                       cast<float>(sample.rng_hash >> 16u)));
        trace_write_global(path_trace_schema::GlobalSlot::filter,
                           make_float3(sample.filter_sample, 0.0f));
        trace_write_global(path_trace_schema::GlobalSlot::lens_time,
                           camera_depth_of_field ? sample.lens_time_sample
                                                 : make_float3(0.0f));
        trace_write_global(path_trace_schema::GlobalSlot::ray_p,
                           sample.ray->origin());
        trace_write_global(path_trace_schema::GlobalSlot::ray_d,
                           sample.ray->direction());
        trace_write_global(path_trace_schema::GlobalSlot::ray_range,
                           make_float3(sample.ray->t_min(),
                                       sample.ray->t_max(), 0.5f));
    }
    sample.ray_source_object = surface_ray::invalid_primitive;
    sample.ray_source_primitive = surface_ray::invalid_primitive;
    sample.radiance = make_float3(0.0f);
    const auto mode = config.effective_spectral_mode();
    sample.throughput = cycles_spectrum::Spectrum::one(mode);
    if (mode != contract::SpectralMode::none) {
        sample.spectral_u_lambda = cycles_sampler::sample_1d(
            sobol_table, kernel_parameters.sobol_sequence_size,
            sample.sample_index, sample.rng_hash ^ 0x51a7b9d3u, 0u);
    } else {
        sample.spectral_u_lambda.reset();
    }
    if (config.xyz3_rough_dispersion()) {
        sample.spectral_pdf_ratio = make_float3(1.0f);
    } else {
        sample.spectral_pdf_ratio.reset();
    }
    sample.sample_normal = make_float3(0.0f);
    sample.sample_albedo = make_float3(0.0f);
    sample.sample_glossy_color = make_float3(0.0f);
    sample.sample_transmission_color = make_float3(0.0f);
    sample.sample_diffuse_direct = make_float3(0.0f);
    sample.sample_diffuse_indirect = make_float3(0.0f);
    sample.sample_glossy_direct = make_float3(0.0f);
    sample.sample_glossy_indirect = make_float3(0.0f);
    sample.sample_transmission_direct = make_float3(0.0f);
    sample.sample_transmission_indirect = make_float3(0.0f);
    sample.sample_volume_direct = make_float3(0.0f);
    sample.sample_volume_indirect = make_float3(0.0f);
    sample.sample_emission = make_float3(0.0f);
    sample.sample_environment = make_float3(0.0f);
    sample.volume_guiding_scatter = make_float3(0.0f);
    sample.volume_guiding_transmit = make_float3(0.0f);
    sample.sample_transparency = 0.0f;
    sample.previous_bsdf_pdf = 0.0f;
    sample.previous_mis_origin_normal = make_float3(0.0f);
    sample.previous_light_tree_dt = 0.0f;
    sample.minimum_bsdf_pdf = std::numeric_limits<float>::max();
    sample.continuation_probability = 1.0f;
    sample.path_diffuse_weight = make_float3(0.0f);
    sample.path_glossy_weight = make_float3(0.0f);
    sample.ray_events = 0u;
    sample.diffuse_depth = 0u;
    sample.glossy_depth = 0u;
    sample.transparent_depth = 0u;
    sample.transmission_depth = 0u;
    sample.portal_depth = 0u;
    sample.path_depth = 0u;
    const auto initial_cycles_path_state = cycles_path_state::initial_state();
    sample.path_flags = initial_cycles_path_state.flag;
    if (mode == contract::SpectralMode::xyz3 && config.xyz3_force_collapse) {
        LUISA_ASSERT(config.scene && config.scene->spectral_tables &&
                         sample.spectral_u_lambda.has_value(),
                     "xyz3_force_collapse requires SpectralTablesGpu.");
        const auto decoded =
            cycles_spectrum::xyz3_decode_u(*sample.spectral_u_lambda);
        const auto lambdas = cycles_spectrum::sample_xyz3_lambdas(
            *config.scene->spectral_tables, decoded.u);
        const auto h_lambda =
            cycles_spectrum::xyz3_hero(lambdas, decoded.hero_index);
        sample.throughput = cycles_spectrum::Spectrum::from_xyz3(
            cycles_spectrum::xyz3_collapse_weight(
                *config.scene->spectral_tables, h_lambda));
        sample.path_flags |= cycles_path_state::flag_spectral_collapsed;
    }
    sample.cycles_path_visibility = initial_cycles_path_state.visibility;
    sample.cycles_rng_offset = initial_cycles_path_state.rng_offset;
    if (config.volume_state) {
        config.volume_state->reset(
            sample.volume,
            config.scene,
            sample.ray->origin(),
            camera_visibility,
            config.camera_may_be_inside_volume);
    }
    sample.volume_bounce = 0u;
    sample.volume_bounds_bounce = 0u;
    sample.optical_depth = 0.0f;
    sample.pending_subsurface_exit = false;
    sample.pending_subsurface_hit.instance = surface_ray::invalid_primitive;
    sample.pending_subsurface_hit.primitive = surface_ray::invalid_primitive;
    sample.pending_subsurface_hit.barycentric = make_float2(0.0f);
    sample.pending_subsurface_hit.committed_ray_t = 0.0f;
}

contract::SpectralMode PathSampleContext::spectral_mode() const noexcept {
    return invocation.config.effective_spectral_mode();
}

bool PathSampleContext::is_spectral() const noexcept {
    return spectral_mode() != contract::SpectralMode::none;
}

bool PathSampleContext::is_hero4() const noexcept {
    return spectral_mode() == contract::SpectralMode::hero4;
}

bool PathSampleContext::is_xyz3() const noexcept {
    return spectral_mode() == contract::SpectralMode::xyz3;
}

Bool PathSampleContext::spectral_secondary_terminated() const noexcept {
    if (!is_spectral()) {
        return false;
    }
    return (path_flags &
            cycles_path_state::flag_spectral_secondary_terminated) != 0u;
}

Bool PathSampleContext::spectral_lane_terminated() const noexcept {
    if (!is_hero4()) {
        return false;
    }
    return spectral_secondary_terminated();
}

Expr<luisa::float4> PathSampleContext::throughput_eval4() const noexcept {
    if (is_xyz3()) {
        const auto ctx = spectral_context();
        return Expr<luisa::float4>{cycles_spectrum::xyz3_rotate_eval(
            throughput.xyz3(), *ctx->hero_index)};
    }
    return Expr<luisa::float4>{throughput.lanes()};
}

std::optional<cycles_spectrum::SampledWavelengths>
PathSampleContext::sampled_wavelengths() const noexcept {
    const auto mode = spectral_mode();
    if (mode == contract::SpectralMode::none || !spectral_u_lambda.has_value()) {
        return std::nullopt;
    }
    if (mode == contract::SpectralMode::xyz3) {
        return spectral_context()->wavelengths;
    }
    return cycles_spectrum::sample_visible_wavelengths(*spectral_u_lambda);
}

std::optional<cycles_spectrum::SpectralContext>
PathSampleContext::spectral_context() const noexcept {
    const auto mode = spectral_mode();
    if (mode == contract::SpectralMode::none) {
        return std::nullopt;
    }
    const auto &scene = invocation.config.scene;
    LUISA_ASSERT(scene && scene->spectral_tables &&
                     spectral_u_lambda.has_value(),
                 "Spectral path requires uploaded SpectralTablesGpu and u_lambda.");
    if (mode == contract::SpectralMode::hero4) {
        return cycles_spectrum::SpectralContext{
            .tables = scene->spectral_tables.get(),
            .wavelengths =
                cycles_spectrum::sample_visible_wavelengths(*spectral_u_lambda),
            .secondary_terminated = spectral_secondary_terminated()};
    }
    return cycles_spectrum::make_spectral_context(
        scene->spectral_tables.get(), mode, *spectral_u_lambda,
        spectral_secondary_terminated());
}

Float3 PathSampleContext::spectrum_to_rgb(
    const cycles_spectrum::Spectrum &value) const noexcept {
    if (!value.is_spectral()) {
        return value.rgb();
    }
    const auto ctx = spectral_context();
    return cycles_spectrum::to_rgb(value, *ctx);
}

Float3 PathSampleContext::reflectance_to_rgb(
    const cycles_spectrum::Spectrum &reflectance) const noexcept {
    if (!reflectance.is_spectral()) {
        return reflectance.rgb();
    }
    const auto ctx = spectral_context();
    return cycles_spectrum::reflectance_to_rgb(reflectance, *ctx);
}

Float PathSampleContext::throughput_average() const noexcept {
    return cycles_spectrum::average(throughput,
                                    spectral_secondary_terminated());
}

cycles_spectrum::Spectrum PathSampleContext::spectrum_from_eval(
    Expr<luisa::float4> eval4) const noexcept {
    if (is_hero4()) {
        return cycles_spectrum::Spectrum::from_lanes(eval4);
    }
    if (!is_spectral()) {
        return cycles_spectrum::Spectrum::from_rgb_raw(eval4.xyz());
    }
    const auto ctx = spectral_context();
    return cycles_spectrum::Spectrum::from_eval4(&*ctx, eval4);
}

Float PathSampleContext::xyz3_mis_pdf(
    Expr<float> scalar_pdf,
    const std::optional<Float4> &lane_pdf) const noexcept {
    if (!is_xyz3() || !spectral_pdf_ratio.has_value() ||
        !lane_pdf.has_value()) {
        return scalar_pdf;
    }
    const auto ctx = spectral_context();
    const auto h = *ctx->hero_index;
    const Float3 p_xyz =
        max(cycles_spectrum::xyz3_unrotate_eval(*lane_pdf, h),
            make_float3(0.0f));
    const Float3 &r = *spectral_pdf_ratio;
    const Float r_sum = max(r.x + r.y + r.z, 1.0e-30f);
    return select(dot(r, p_xyz) / r_sum, scalar_pdf,
                  spectral_secondary_terminated());
}

Float PathSampleContext::apply_xyz3_bsdf_weight(
    Expr<luisa::float4> f_eval, Expr<float> scalar_pdf,
    const std::optional<Float4> &lane_pdf) noexcept {
    LUISA_ASSERT(is_xyz3(), "apply_xyz3_bsdf_weight requires SpectralMode::xyz3.");
    const auto ctx = spectral_context();
    const auto h = *ctx->hero_index;
    const Float3 f_xyz = cycles_spectrum::xyz3_unrotate_eval(f_eval, h);
    if (!spectral_pdf_ratio.has_value() || !lane_pdf.has_value()) {
        throughput *= cycles_spectrum::Spectrum::from_xyz3(f_xyz / scalar_pdf);
        return scalar_pdf;
    }
    const Float3 p_xyz =
        max(cycles_spectrum::xyz3_unrotate_eval(*lane_pdf, h),
            make_float3(0.0f));
    Float3 &r = *spectral_pdf_ratio;
    const Float r_sum = max(r.x + r.y + r.z, 1.0e-30f);
    const Float p_bar_raw = dot(r, p_xyz) / r_sum;
    const Float p_bar = select(p_bar_raw, scalar_pdf,
                               spectral_secondary_terminated());
    throughput *= cycles_spectrum::Spectrum::from_xyz3(
        f_xyz / max(p_bar, 1.0e-30f));
    $if(!spectral_secondary_terminated()) {
        const Float3 next_r = r * p_xyz;
        const Float next_sum = next_r.x + next_r.y + next_r.z;
        r = select(make_float3(1.0f), next_r * (3.0f / max(next_sum, 1.0e-30f)),
                   next_sum > 1.0e-30f);
    };
    return p_bar;
}

void PathSampleContext::collapse_xyz3(Expr<float> hero_bsdf_weight) noexcept {
    LUISA_ASSERT(is_xyz3(), "collapse_xyz3 requires SpectralMode::xyz3.");
    $if(spectral_secondary_terminated()) {
        throughput *= hero_bsdf_weight;
    }
    $else {
        const auto ctx = spectral_context();
        const auto h = *ctx->hero_index;
        const auto h_lambda =
            cycles_spectrum::xyz3_hero(*ctx->lambdas_xyz, h);
        const auto w_raw = cycles_spectrum::xyz3_collapse_weight(
            *ctx->tables, h_lambda);
        const Float t_h = cycles_spectrum::xyz3_hero(throughput.xyz3(), h);
        Float cf = 1.0f;
        if (spectral_pdf_ratio.has_value()) {
            const Float3 &r = *spectral_pdf_ratio;
            const Float r_h = cycles_spectrum::xyz3_hero(r, h);
            const Float r_mean = (r.x + r.y + r.z) * (1.0f / 3.0f);
            cf = select(0.0f, r_mean / max(r_h, 1.0e-30f), r_h > 0.0f);
        }
        throughput = cycles_spectrum::Spectrum::from_xyz3(
            (t_h * cf * hero_bsdf_weight) * w_raw);
    };
    path_flags |= cycles_path_state::flag_spectral_collapsed;
    if (spectral_pdf_ratio.has_value()) {
        *spectral_pdf_ratio = make_float3(1.0f);
    }
}

Float PathSampleContext::xyz3_hero_caustic_factor() const noexcept {
    if (!is_xyz3() || !spectral_pdf_ratio.has_value()) {
        return 1.0f;
    }
    const auto ctx = spectral_context();
    const auto h = *ctx->hero_index;
    const Float3 &r = *spectral_pdf_ratio;
    const Float r_h = cycles_spectrum::xyz3_hero(r, h);
    const Float r_mean = (r.x + r.y + r.z) * (1.0f / 3.0f);
    return select(1.0f, select(0.0f, r_mean / max(r_h, 1.0e-30f), r_h > 0.0f),
                  !spectral_secondary_terminated());
}

Float3
PathSampleContext::analytic_light_constant_shader(
    Var<LightGpu> light) const noexcept {
    Float3 result = make_float3(1.0f);
    $if(light.surface_tag != ~std::uint32_t{0u}) {
        const auto constant = cycles_svm_constant_emission(
            *invocation.config.scene, light.cycles_shader_id, result);
        assume(constant);
    };
    return result;
}

Float3
PathSampleContext::analytic_light_shader(Var<LightGpu> light,
                                         UInt light_index,
                                         Float3 light_position,
                                         Float3 light_normal,
                                         Float2 light_uv,
                                         Float3 incoming,
                                         Float light_distance) const noexcept {
    Float3 result = make_float3(1.0f);
    $if(light.surface_tag != ~std::uint32_t{0u}) {
        const cycles_svm::PathState state{
            0u, cycles_svm::path_ray_emission, path_depth, transparent_depth,
            diffuse_depth, glossy_depth, transmission_depth, portal_depth};
        result = evaluate_cycles_svm_lamp_emission(
            invocation.config.scene, invocation.parameters,
            light.cycles_shader_id, light.cycles_object_index, light_index,
            light_position, light_normal, incoming, light_uv, light_distance,
            0.0f, state, vt_miss_page);
    };
    return result;
}

cycles_spectrum::Spectrum
PathSampleContext::analytic_light_constant_shader_spectrum(
    Var<LightGpu> light) const noexcept {
    if (!is_spectral()) {
        return cycles_spectrum::Spectrum::from_rgb_raw(
            analytic_light_constant_shader(light));
    }
    const auto ctx = spectral_context();
    Float3 result = make_float3(1.0f);
    $if(light.surface_tag != ~std::uint32_t{0u}) {
        const auto constant = cycles_svm_constant_emission(
            *invocation.config.scene, light.cycles_shader_id, result);
        assume(constant);
    };
    return cycles_spectrum::Spectrum::from_eval4(
        &*ctx, cycles_spectrum::rgb_to_spectrum_illuminant(
                   *ctx->tables, result, ctx->wavelengths.lambdas));
}

cycles_spectrum::Spectrum
PathSampleContext::analytic_light_shader_spectrum(
    Var<LightGpu> light, UInt light_index, Float3 light_position,
    Float3 light_normal, Float2 light_uv, Float3 incoming,
    Float light_distance) const noexcept {
    if (!is_spectral()) {
        return cycles_spectrum::Spectrum::from_rgb_raw(
            analytic_light_shader(light, light_index, light_position,
                                  light_normal, light_uv, incoming,
                                  light_distance));
    }
    const auto ctx = spectral_context();
    Float4 result = cycles_spectrum::rgb_to_spectrum_illuminant(
        *ctx->tables, make_float3(1.0f), ctx->wavelengths.lambdas);
    $if(light.surface_tag != ~std::uint32_t{0u}) {
        const cycles_svm::PathState state{
            0u, cycles_svm::path_ray_emission, path_depth, transparent_depth,
            diffuse_depth, glossy_depth, transmission_depth, portal_depth};
        result = evaluate_cycles_svm_lamp_emission_spectral(
            invocation.config.scene, invocation.parameters,
            light.cycles_shader_id, light.cycles_object_index, light_index,
            light_position, light_normal, incoming, light_uv, light_distance,
            0.0f, state, *ctx, vt_miss_page);
    };
    return cycles_spectrum::Spectrum::from_eval4(&*ctx, result);
}

} // namespace psycles::luisa_backend::detail
