// Surface emission, continuation roulette, trace records, and data passes.
// Included by path_tracer_kernel.cpp inside LuisaRenderSession::initialize.

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

                UInt cycles_surface_runtime_flags = 0u;
                if (path_trace_enabled) {
                    const auto closure_summary =
                        trace_surface_closure(
                            surface_tag,
                            point,
                            0u);
                    cycles_surface_runtime_flags =
                        closure_summary.runtime_flags;
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::state_depth,
                        make_float3(
                            cast<float>(path_step),
                            cast<float>(
                                cycles_rng_offset),
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
                                cycles_path_visibility)
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
                            cast<float>(
                                cycles_object_index),
                            cast<float>(hit->prim),
                            1.0f));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::surface_meta,
                        make_float3(
                            trace_uint32(
                                cycles_surface_shader)
                                .xy(),
                            cast<float>(
                                closure_summary.count)));
                    trace_write_event(
                        path_step,
                        path_trace_schema::
                            EventSlot::surface_flags,
                        make_float3(
                            trace_uint32(
                                cycles_surface_runtime_flags)
                                .xy(),
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
                    for (auto closure_index = 0u;
                         closure_index <
                         path_trace_schema::max_closures;
                         ++closure_index) {
                        const auto closure =
                            trace_surface_closure(
                                surface_tag,
                                point,
                                closure_index);
                        $if (closure.valid) {
                            trace_write_closure(
                                path_step,
                                closure_index,
                                0u,
                                make_float3(
                                    cast<float>(
                                        closure.index),
                                    cast<float>(
                                        closure.type),
                                    closure
                                        .sample_weight));
                            trace_write_closure(
                                path_step,
                                closure_index,
                                1u,
                                closure.weight);
                            trace_write_closure(
                                path_step,
                                closure_index,
                                2u,
                                closure.normal);
                        };
                    }
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
