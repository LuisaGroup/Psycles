// Environment next-event estimation.
// Included by path_tracer_kernel.cpp inside LuisaRenderSession::initialize.

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
                                        invalid_primitive,
                                    kernel_parameters
                                        .transparent_max_bounces,
                                    pack_shader_evaluation_state(
                                        cycles_path_state::
                                            shadow_shader_state(
                                                path_depth,
                                                diffuse_depth,
                                                glossy_depth,
                                                transparent_depth,
                                                transmission_depth)));
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
                                            cycles_path_state::
                                                light_emission_shader_state(
                                                    path_depth,
                                                    diffuse_depth,
                                                    glossy_depth,
                                                    transparent_depth,
                                                    transmission_depth)) *
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
