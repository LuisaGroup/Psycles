// Closure sampling, path-state transition, ray continuation, and film writes.
// Included by path_tracer_kernel.cpp inside LuisaRenderSession::initialize.

                auto surface_sample =
                    SurfaceSample::zero();
                if (path_trace_enabled) {
                    const auto sample_trace =
                        trace_sample_surface(
                            surface_tag,
                            point,
                            bsdf_sample.z,
                            bsdf_sample.xy(),
                            path_surface_query);
                    surface_sample =
                        sample_trace.sample;
                    $if (sample_trace.closure_valid) {
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::
                                    closure_pick,
                            make_float3(
                                cast<float>(
                                    sample_trace
                                        .closure_index),
                                cast<float>(
                                    sample_trace
                                        .closure_type),
                                sample_trace
                                    .closure_sample_weight));
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::
                                    closure_random,
                            make_float3(
                                bsdf_sample.xy(),
                                sample_trace
                                    .selection_rescaled));
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::
                                    closure_weight,
                            sample_trace
                                .closure_weight);
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::closure_n,
                            sample_trace
                                .closure_normal);
                        const auto cycles_label =
                            cycles_closure::
                                label_from_events(
                                    sample_trace
                                        .sample
                                        .evaluation
                                        .events);
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::bsdf_meta,
                            make_float3(
                                sample_trace
                                    .sample
                                    .evaluation
                                    .pdf,
                                sample_trace
                                    .sample
                                    .evaluation
                                    .pdf,
                                cast<float>(
                                    cycles_label)));
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::bsdf_wo,
                            sample_trace.sample.wi);
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::bsdf_eval,
                            sample_trace
                                .sample
                                .evaluation.f);
                        trace_write_event(
                            path_step,
                            path_trace_schema::
                                EventSlot::
                                    bsdf_roughness_eta,
                            make_float3(
                                sample_trace
                                    .sample.roughness,
                                sample_trace
                                    .sample.eta));
                    };
                } else {
                    surface_sample = sample_surface(
                        surface_tag,
                        point,
                        bsdf_sample.z,
                        bsdf_sample.xy(),
                        path_surface_query);
                }
                cycles_surface_runtime_flags =
                    surface_sample.runtime_flags;
                if (path_trace_enabled) {
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::surface_flags,
                        make_float3(
                            trace_uint32(
                                cycles_surface_runtime_flags)
                                .xy(),
                            0.0f));
                }
                $if (!surface_sample.valid |
                     (surface_sample.evaluation.pdf <=
                      0.0f)) {
                    $break;
                };

                Bool transparent =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_transparent)) != 0u;
                Bool singular =
                    (surface_sample.evaluation.events &
                     static_cast<std::uint32_t>(
                         contract::event_singular)) != 0u;
                const auto cycles_label =
                    cycles_closure::label_from_events(
                        surface_sample.evaluation.events);

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

                ray_events = select(
                    surface_sample.evaluation.events,
                    ray_events |
                        surface_sample.evaluation.events,
                    transparent);
                const auto next_cycles_path_state =
                    cycles_path_state::next_surface(
                        {
                            .flag = path_flags,
                            .visibility =
                                cycles_path_visibility,
                            .bounce = path_depth,
                            .diffuse_bounce =
                                diffuse_depth,
                            .glossy_bounce =
                                glossy_depth,
                            .transmission_bounce =
                                transmission_depth,
                            .transparent_bounce =
                                transparent_depth,
                            .rng_offset =
                                cycles_rng_offset},
                        cycles_label,
                        cycles_surface_runtime_flags,
                        {
                            .maximum =
                                kernel_parameters
                                    .max_bounces,
                            .maximum_diffuse =
                                kernel_parameters
                                    .max_diffuse_bounces,
                            .maximum_glossy =
                                kernel_parameters
                                    .max_glossy_bounces,
                            .maximum_transmission =
                                kernel_parameters
                                    .max_transmission_bounces,
                            .maximum_transparent =
                                kernel_parameters
                                    .transparent_max_bounces});
                path_flags =
                    next_cycles_path_state.flag;
                cycles_path_visibility =
                    next_cycles_path_state.visibility;
                path_depth =
                    next_cycles_path_state.bounce;
                diffuse_depth =
                    next_cycles_path_state.diffuse_bounce;
                glossy_depth =
                    next_cycles_path_state.glossy_bounce;
                transmission_depth =
                    next_cycles_path_state
                        .transmission_bounce;
                transparent_depth =
                    next_cycles_path_state
                        .transparent_bounce;
                cycles_rng_offset =
                    next_cycles_path_state.rng_offset;
                ray_visibility =
                    cycles_path_state::
                        contract_visibility(
                            cycles_path_visibility);
                terminate_on_next_surface =
                    (path_flags &
                     cycles_path_state::
                         flag_terminate_on_next_surface) !=
                    0u;
                terminate_after_transparent =
                    (path_flags &
                     cycles_path_state::
                         flag_terminate_after_transparent) !=
                    0u;
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
                previous_mis_origin_normal = select(
                    point.shading_normal,
                    previous_mis_origin_normal,
                    transparent);
                // A transparent Cycles bounce keeps the complete ray line and
                // advances only tmin to the next representable float after
                // the committed distance. A regular surface bounce starts a
                // new ray and uses the conditional triangle-origin policy.
                const auto normalized_surface_direction =
                    normalize(surface_sample.wi);
                Float3 next_origin = select(
                    make_surface_ray_origin(
                        normalized_surface_direction),
                    ray->origin(),
                    transparent);
                Float3 next_direction = select(
                    normalized_surface_direction,
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
                if (path_trace_enabled) {
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::post_depth,
                        make_float3(
                            cast<float>(path_depth),
                            cast<float>(
                                transparent_depth),
                            cast<float>(
                                cycles_rng_offset)));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::post_throughput,
                        throughput);
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::post_ray_p,
                        ray->origin());
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::post_ray_d,
                        ray->direction());
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::post_flags,
                        make_float3(
                            trace_uint32(path_flags).xy(),
                            0.0f));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::post_mis,
                        make_float3(
                            previous_bsdf_pdf,
                            minimum_bsdf_pdf,
                            continuation_probability));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::post_visibility,
                        make_float3(
                            trace_uint32(
                                cycles_path_visibility)
                                .xy(),
                            0.0f));
                }
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
