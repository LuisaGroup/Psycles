#include "path_kernel_builder.h"
#include "subsurface_exit_closure_component.h"

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
    const auto atomic_film =
        film_accumulation == PathFilmAccumulation::atomic;
    const auto film_value = [atomic_film](
                                const BufferFloat4 &buffer,
                                const UInt &index) noexcept {
        return atomic_film
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
    UInt completed = atomic_film
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

SurfaceEvaluation PathKernelInvocation::evaluate_light_surface(
    UInt surface_tag,
    const SurfacePoint &point,
    Float3 outgoing,
    const SurfaceQuery &query,
    UInt shader_flags) const noexcept {
    auto result = unpack_surface_evaluation(
        config.surfaces.evaluate_light(
            config.scene->scalar_parameter_buffer,
            config.scene->vector_parameter_buffer,
            config.scene->cycles_bsdf_table_buffer,
            config.scene->texture_heap,
            config.scene->heap,
            surface_tag,
            pack_surface_point(point),
            outgoing,
            query.lobe_mask,
            query.transport_mode,
            query.glossy_filter_roughness,
            query.reflective_caustics,
            query.refractive_caustics,
            shader_flags));
    $if(query.subsurface_exit) {
        result = SubsurfaceExitClosureComponent{}.evaluate_light(
            point, outgoing, query, shader_flags);
    };
    return result;
}

SurfaceEvaluation SurfaceShadingState::evaluate_light(
    PathKernelInvocation &invocation,
    UInt surface_tag,
    const SurfacePoint &point,
    Float3 outgoing,
    const SurfaceQuery &query,
    UInt shader_flags) const noexcept {
    if (populated_surface) {
        return populated_surface->evaluate_light(
            Expr<luisa::float3>{outgoing.expression()},
            {.surface = query, .shader_flags = shader_flags});
    }
    return invocation.evaluate_light_surface(
        std::move(surface_tag),
        point,
        std::move(outgoing),
        query,
        std::move(shader_flags));
}

SurfacePreparation PathKernelInvocation::prepare_surface(
    UInt surface_tag,
    const SurfacePoint &point,
    Float3 outgoing,
    const SurfaceQuery &query,
    Bool include_runtime_flags,
    Bool include_aov) const noexcept {
    const auto preparation_query = surface_preparation_query(
        point,
        std::move(outgoing),
        query,
        std::move(include_runtime_flags),
        std::move(include_aov));
    return unpack_surface_preparation(
        config.surfaces.preparation(
            config.scene->scalar_parameter_buffer,
            config.scene->vector_parameter_buffer,
            config.scene->cycles_bsdf_table_buffer,
            config.scene->texture_heap,
            config.scene->heap,
            surface_tag,
            pack_surface_point(point),
            pack_surface_preparation_query(preparation_query)));
}

SurfacePreparationQuery PathKernelInvocation::surface_preparation_query(
    const SurfacePoint &point,
    Float3 outgoing,
    const SurfaceQuery &query,
    Bool include_runtime_flags,
    Bool include_aov) const noexcept {
    const auto emission_reflective_caustics =
        Bool{config.reflective_caustics} |
        ((point.ray_visibility &
             contract::visibility_bit(
                 contract::RayVisibility::diffuse)) ==
            0u);
    return SurfacePreparationQuery{
        .outgoing = std::move(outgoing),
        .glossy_filter_roughness =
            query.glossy_filter_roughness,
        .emission_reflective_caustics =
            emission_reflective_caustics,
        .reflective_caustics = query.reflective_caustics,
        .refractive_caustics = query.refractive_caustics,
        .include_runtime_flags = include_runtime_flags,
        .include_aov = include_aov};
}

Float3 PathKernelInvocation::surface_emission(UInt surface_tag,
                                              const SurfacePoint &point,
                                              Float3 outgoing) const noexcept {
    const auto reflective_caustics =
        Bool{config.reflective_caustics} |
        ((point.ray_visibility &
             contract::visibility_bit(
                 contract::RayVisibility::diffuse)) ==
            0u);
    return config.surfaces.emission(config.scene->scalar_parameter_buffer,
                                    config.scene->vector_parameter_buffer,
                                    config.scene->cycles_bsdf_table_buffer,
                                    config.scene->texture_heap,
                                    config.scene->heap,
                                    surface_tag,
                                    pack_surface_point(point),
                                    outgoing,
                                    reflective_caustics);
}

Float3 PathKernelInvocation::constant_surface_emission(
    UInt surface_tag,
    UInt parameter_block) const noexcept {
    return config.surfaces.constant_emission(
        config.scene->scalar_parameter_buffer,
        config.scene->vector_parameter_buffer,
        surface_tag,
        parameter_block);
}

SurfaceSample
PathKernelInvocation::sample_surface(UInt surface_tag,
                                     const SurfacePoint &point,
                                     Float u_lobe,
                                     Float2 u_direction,
                                     const SurfaceQuery &query) const noexcept {
    auto result = unpack_surface_sample(
        config.surfaces.sample(config.scene->scalar_parameter_buffer,
                               config.scene->vector_parameter_buffer,
                               config.scene->cycles_bsdf_table_buffer,
                               config.scene->texture_heap,
                               config.scene->heap,
                               surface_tag,
                               pack_surface_point(point),
                               u_lobe,
                               u_direction,
                               query.lobe_mask,
                               query.transport_mode,
                               query.glossy_filter_roughness,
                               query.reflective_caustics,
                               query.refractive_caustics));
    $if(query.subsurface_exit) {
        result = SubsurfaceExitClosureComponent{}.sample(
            point, u_direction, query);
    };
    return result;
}

SurfaceSample SurfaceShadingState::sample(
    PathKernelInvocation &invocation,
    UInt surface_tag,
    const SurfacePoint &point,
    Float u_lobe,
    Float2 u_direction,
    const SurfaceQuery &query) const noexcept {
    if (populated_surface) {
        return populated_surface->sample(
            Expr<float>{u_lobe.expression()},
            Expr<luisa::float2>{u_direction.expression()},
            query);
    }
    return invocation.sample_surface(
        std::move(surface_tag),
        point,
        std::move(u_lobe),
        std::move(u_direction),
        query);
}

SurfaceClosureTrace PathKernelInvocation::trace_surface_closure(
    UInt surface_tag,
    const SurfacePoint &point,
    UInt requested_index,
    Bool reflective_caustics,
    Bool refractive_caustics) const noexcept {
    return unpack_surface_closure_trace(
        config.surfaces.closure_trace(config.scene->scalar_parameter_buffer,
                                      config.scene->vector_parameter_buffer,
                                      config.scene->cycles_bsdf_table_buffer,
                                      config.scene->texture_heap,
                                      config.scene->heap,
                                      surface_tag,
                                      pack_surface_point(point),
                                      requested_index,
                                      reflective_caustics,
                                      refractive_caustics));
}

SurfaceSampleTrace PathKernelInvocation::trace_sample_surface(
    UInt surface_tag,
    const SurfacePoint &point,
    Float u_lobe,
    Float2 u_direction,
    const SurfaceQuery &query) const noexcept {
    auto result = unpack_surface_sample_trace(
        config.surfaces.sample_trace(config.scene->scalar_parameter_buffer,
                                     config.scene->vector_parameter_buffer,
                                     config.scene->cycles_bsdf_table_buffer,
                                     config.scene->texture_heap,
                                     config.scene->heap,
                                     surface_tag,
                                     pack_surface_point(point),
                                     u_lobe,
                                     u_direction,
                                     query.lobe_mask,
                                     query.transport_mode,
                                     query.glossy_filter_roughness,
                                     query.reflective_caustics,
                                     query.refractive_caustics));
    $if(query.subsurface_exit) {
        result = SubsurfaceExitClosureComponent{}.sample_trace(
            point, u_direction, query);
    };
    return result;
}

SurfaceSampleTrace SurfaceShadingState::sample_trace(
    PathKernelInvocation &invocation,
    UInt surface_tag,
    const SurfacePoint &point,
    Float u_lobe,
    Float2 u_direction,
    const SurfaceQuery &query) const noexcept {
    if (populated_surface) {
        return populated_surface->sample_trace(
            Expr<float>{u_lobe.expression()},
            Expr<luisa::float2>{u_direction.expression()},
            query);
    }
    return invocation.trace_sample_surface(
        std::move(surface_tag),
        point,
        std::move(u_lobe),
        std::move(u_direction),
        query);
}

Float3 PathKernelInvocation::surface_bssrdf_normal(
    UInt surface_tag,
    const SurfacePoint &point,
    Bool has_bssrdf_bump,
    Bool reflective_caustics,
    Bool refractive_caustics) const noexcept {
    return config.surfaces.bssrdf_normal(
        config.scene->scalar_parameter_buffer,
        config.scene->vector_parameter_buffer,
        config.scene->cycles_bsdf_table_buffer,
        config.scene->texture_heap,
        config.scene->heap,
        surface_tag,
        pack_surface_point(point),
        has_bssrdf_bump,
        reflective_caustics,
        refractive_caustics);
}

Float3 PathKernelInvocation::evaluate_environment(
    Float3 direction,
    const cycles_path_state::ShaderEvaluationState &shader_state)
    const noexcept {
    Float3 result =
        config.environment.base(direction,
                                parameters.background,
                                pack_shader_evaluation_state(shader_state));
    for (const auto &sun : config.environment.suns) {
        result += sun(direction);
    }
    result += config.environment.nishita_sun(direction);
    return result;
}

Float3 PathKernelInvocation::constant_environment() const noexcept {
    return config.environment.constant(
        parameters.background);
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

void PathSampleContext::record_surface_program_execution(
    UInt surface_tag) const noexcept {
  const auto &config = invocation.config;
  if (!config.surface_program_execution_histogram_enabled) {
    return;
  }

  static_assert(
      (surface_program_execution_histogram_lanes_per_topology &
       (surface_program_execution_histogram_lanes_per_topology - 1u)) == 0u);
  $if(surface_tag < config.surface_program_execution_histogram_topology_count) {
    // A checked counter is a commutative multiset projection, so arrival
    // order is irrelevant. Mix the logical event identity only to spread
    // one topology over independent exact-float shards; the host rejects
    // a shard that reaches the 2^24 consecutive-integer boundary.
    const auto lane =
        ((invocation.pixel * 0x9e3779b9u) ^ (sample_index * 0x85ebca6bu) ^
         (path_depth * 0xc2b2ae35u)) &
        static_cast<std::uint32_t>(
            surface_program_execution_histogram_lanes_per_topology - 1u);
    const auto slot =
        config.surface_program_execution_histogram_base +
        surface_tag *
            static_cast<std::uint32_t>(
                surface_program_execution_histogram_shards_per_topology) +
        lane / 4u;
    auto destination = invocation.path_trace.atomic(slot);
    $if((lane & 3u) == 0u) { destination.x.fetch_add(1.0f); }
    $elif((lane & 3u) == 1u) { destination.y.fetch_add(1.0f); }
    $elif((lane & 3u) == 2u) { destination.z.fetch_add(1.0f); }
    $else { destination.w.fetch_add(1.0f); };
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
    Float3 throughput = make_float3(1.0f);
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
    UInt path_depth = 0u;
    // PATH_RAY_SINGLE_PASS_DONE is not an initial camera-path property.
    // Cycles sets it only when a surface actually passes the film alpha
    // threshold and writes the first single-value data passes.
    const auto initial_cycles_path_state = cycles_path_state::initial_state();
    UInt path_flags = initial_cycles_path_state.flag;
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

Float3
PathSampleContext::analytic_light_constant_shader(
    Var<LightGpu> light) const noexcept {
    Float3 result = make_float3(1.0f);
    $if(light.surface_tag != ~std::uint32_t{0u}) {
        result = invocation.constant_surface_emission(
            light.surface_tag,
            light.parameter_block);
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
        Float3 relative_position = light_position - light.position;
        const auto light_transform =
            analytic_light_sampling::light_linear_transform(
                light.axis_x, light.axis_y, light.axis_z, light.axis_scale);
        Float3 object_position =
            analytic_light_sampling::world_to_light_direction(relative_position,
                                                              light_transform);
        Float3 object_normal = analytic_light_sampling::world_to_light_normal(
            light_normal, light_transform);
        SurfacePoint light_point{
            .position = light_position,
            .object_position = object_position,
            .object_location = light.position,
            .generated = object_position,
            .geometric_normal = light_normal,
            .shading_normal = light_normal,
            .object_shading_normal = object_normal,
            .object_tangent = light.axis_x,
            .tangent_sign = 1.0f,
            .undisplaced_position = light_position,
            .undisplaced_object_position = object_position,
            .undisplaced_shading_normal = light_normal,
            .undisplaced_object_shading_normal = object_normal,
            .undisplaced_object_tangent = light.axis_x,
            .undisplaced_tangent_sign = 1.0f,
            .normal_to_world_x = light_transform.inverse_row_x,
            .normal_to_world_y = light_transform.inverse_row_y,
            .normal_to_world_z = light_transform.inverse_row_z,
            .dpdu = make_float3(0.0f),
            .dpdv = make_float3(0.0f),
            .dPdx = make_float3(0.0f),
            .dPdy = make_float3(0.0f),
            .object_dPdx = make_float3(0.0f),
            .object_dPdy = make_float3(0.0f),
            .undisplaced_dPdx = make_float3(0.0f),
            .undisplaced_dPdy = make_float3(0.0f),
            .undisplaced_object_dPdx = make_float3(0.0f),
            .undisplaced_object_dPdy = make_float3(0.0f),
            .generated_dx = make_float3(0.0f),
            .generated_dy = make_float3(0.0f),
            .incoming = incoming,
            .uv = light_uv,
            .uv_dx = make_float2(0.0f),
            .uv_dy = make_float2(0.0f),
            .geometry_index = ~0u,
            .barycentric = make_float2(0.0f),
            .barycentric_dx = make_float2(0.0f),
            .barycentric_dy = make_float2(0.0f),
            .instance_id = 0u,
            .primitive_id = light_index,
            .parameter_block = light.parameter_block,
            .object_random = 0.0f,
            .particle_index = 0u,
            .random_per_island = 0.0f,
            .triangle_smooth = false,
            .is_curve = false,
            .curve_intercept = 0.0f,
            .curve_length = 0.0f,
            .curve_thickness = 0.0f,
            .curve_tangent_normal = make_float3(0.0f),
            .curve_random = 0.0f,
            .ray_visibility = shader_ray_visibility(),
            .ray_events = ray_events,
            .ray_depth = path_depth,
            .diffuse_depth = diffuse_depth,
            .glossy_depth = glossy_depth,
            .transparent_depth = transparent_depth,
            .transmission_depth = transmission_depth,
            .ray_length = light_distance,
            .time = 0.0f,
            .use_bump_map_correction = false,
            .back_facing = false};
        cycles_path_state::apply_shader_state(
            light_point,
            cycles_path_state::light_emission_shader_state(path_depth,
                                                           diffuse_depth,
                                                           glossy_depth,
                                                           transparent_depth,
                                                           transmission_depth));
        result = invocation.surface_emission(
            light.surface_tag, light_point, incoming);
    };
    return result;
}

} // namespace psycles::luisa_backend::detail
