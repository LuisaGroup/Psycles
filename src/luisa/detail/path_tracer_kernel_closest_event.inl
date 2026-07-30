// Path RNG dimensions, mesh traversal, forward-light hits, and environment miss.
// Included by path_tracer_kernel.cpp inside LuisaRenderSession::initialize.

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
                        cycles_sampler::
                            path_state_dimension(
                            cycles_rng_offset,
                            tabulated_sobol::
                                terminate_dimension));
                const auto light_sample =
                    cycles_sampler::sample_3d(
                        sobol_table,
                        kernel_parameters
                            .sobol_sequence_size,
                        sample_index,
                        rng_hash,
                        cycles_sampler::
                            path_state_dimension(
                            cycles_rng_offset,
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
                        cycles_sampler::
                            path_state_dimension(
                            cycles_rng_offset,
                            tabulated_sobol::
                                light_terminate_dimension));
                const auto bsdf_sample =
                    cycles_sampler::sample_3d(
                        sobol_table,
                        kernel_parameters
                            .sobol_sequence_size,
                        sample_index,
                        rng_hash,
                        cycles_sampler::
                            path_state_dimension(
                            cycles_rng_offset,
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
                Float closest_surface_distance =
                    ray->t_max();
                $if (!hit->miss()) {
                    closest_surface_distance =
                        hit->committed_ray_t;
                };

                // Cycles treats finite analytic lights as transparent
                // primitives in the same closest-event ordering as mesh
                // geometry. Accumulate every light before the closest
                // surface, advance only tmin, and preserve the path/RNG
                // state. The immediately preceding lamp is excluded exactly
                // like RaySelfPrimitives; transparent-bounce depth bounds the
                // sequence.
                UInt previous_analytic_light =
                    surface_ray::invalid_primitive;
                Bool search_analytic_lights = true;
                Bool analytic_light_terminated = false;
                $while (
                    search_analytic_lights &
                    !analytic_light_terminated) {
                    Bool light_hit = false;
                    Float light_hit_distance =
                        closest_surface_distance;
                    UInt light_hit_index =
                        surface_ray::invalid_primitive;
                    Float3 light_hit_position =
                        make_float3(0.0f);
                    Float3 light_hit_normal =
                        make_float3(0.0f);
                    Float2 light_hit_uv =
                        make_float2(0.0f);
                    Float light_hit_pdf = 0.0f;
                    Float light_hit_eval_factor = 0.0f;

                    $for (
                        light_index,
                        scene->light_count) {
                        Var<LightGpu> light =
                            scene->light_buffer->read(
                                light_index);
                        const auto camera_path =
                            (ray_visibility &
                             camera_visibility) != 0u;
                        const auto use_mis =
                            (light.flags &
                             light_flag_use_mis) !=
                            0u;
                        const auto visible =
                            (light.visibility_mask &
                             ray_visibility) != 0u;
                        const auto eligible =
                            visible &
                            (camera_path | use_mis) &
                            (light_index !=
                             previous_analytic_light);
                        $if (
                            eligible &
                            (light.type ==
                             static_cast<
                                 std::uint32_t>(
                                 LightType::area))) {
                            const auto ellipse =
                                (light.flags &
                                 light_flag_ellipse) !=
                                0u;
                            const auto full_spread =
                                (light.flags &
                                 light_flag_full_spread) !=
                                0u;
                            const auto normalize_power =
                                (light.flags &
                                 light_flag_normalize) !=
                                0u;
                            const auto candidate =
                                analytic_light_intersection::
                                    intersect_area(
                                        ray->origin(),
                                        ray->direction(),
                                        ray->t_min(),
                                        light_hit_distance,
                                        light.position,
                                        light.axis_x,
                                        light.size_u,
                                        light.axis_y,
                                        light.size_v,
                                        light.axis_z,
                                        ellipse,
                                        full_spread,
                                        light.spread,
                                        normalize_power);
                            $if (candidate.valid) {
                                light_hit = true;
                                light_hit_distance =
                                    candidate.distance;
                                light_hit_index =
                                    light_index;
                                light_hit_position =
                                    candidate.position;
                                light_hit_normal =
                                    candidate.normal;
                                light_hit_uv =
                                    candidate.uv;
                                light_hit_pdf =
                                    candidate
                                        .conditional_pdf;
                                light_hit_eval_factor =
                                    candidate
                                        .evaluation_factor;
                            };
                        };
                        const auto point_type =
                            light.type ==
                            static_cast<
                                std::uint32_t>(
                                LightType::point);
                        const auto spot_type =
                            light.type ==
                            static_cast<
                                std::uint32_t>(
                                LightType::spot);
                        $if (
                            eligible &
                            (point_type | spot_type)) {
                            const auto sphere =
                                (light.flags &
                                 light_flag_sphere) !=
                                0u;
                            const auto normalize_power =
                                (light.flags &
                                 light_flag_normalize) !=
                                0u;
                            const auto
                                had_transmission =
                                    (path_flags &
                                     cycles_path_state::
                                         flag_mis_had_transmission) !=
                                    0u;
                            analytic_light_intersection::
                                PointIntersection candidate{
                                    .valid = false,
                                    .distance = 0.0f,
                                    .position =
                                        make_float3(
                                            0.0f),
                                    .normal =
                                        make_float3(
                                            0.0f),
                                    .uv =
                                        make_float2(
                                            0.0f),
                                    .conditional_pdf =
                                        0.0f,
                                    .evaluation_factor =
                                        0.0f};
                            $if (spot_type) {
                                candidate =
                                    analytic_light_intersection::
                                        intersect_spot(
                                            ray->origin(),
                                            ray->direction(),
                                            ray->t_min(),
                                            light_hit_distance,
                                            light.position,
                                            light.radius,
                                            sphere,
                                            light.axis_x,
                                            light.axis_y,
                                            light.axis_z,
                                            light.axis_scale,
                                            light.spot_angle,
                                            light.spot_smooth,
                                            normalize_power,
                                            previous_mis_origin_normal,
                                            had_transmission);
                            }
                            $else {
                                candidate =
                                    analytic_light_intersection::
                                        intersect_point(
                                            ray->origin(),
                                            ray->direction(),
                                            ray->t_min(),
                                            light_hit_distance,
                                            light.position,
                                            light.radius,
                                            sphere,
                                            light.axis_x,
                                            light.axis_y,
                                            light.axis_z,
                                            light.axis_scale,
                                            normalize_power,
                                            previous_mis_origin_normal,
                                            had_transmission);
                            };
                            $if (candidate.valid) {
                                light_hit = true;
                                light_hit_distance =
                                    candidate.distance;
                                light_hit_index =
                                    light_index;
                                light_hit_position =
                                    candidate.position;
                                light_hit_normal =
                                    candidate.normal;
                                light_hit_uv =
                                    candidate.uv;
                                light_hit_pdf =
                                    candidate
                                        .conditional_pdf;
                                light_hit_eval_factor =
                                    candidate
                                        .evaluation_factor;
                            };
                        };
                    };

                    $if (light_hit) {
                        Var<LightGpu> light =
                            scene->light_buffer->read(
                                light_hit_index);
                        const auto light_pdf =
                            light_hit_pdf *
                            scene->light_selection_pdf;
                        const auto competing =
                            (path_depth > 0u) &
                            (!previous_delta);
                        const auto mis_weight =
                            forward_light_weight(
                                previous_bsdf_pdf,
                                light_pdf,
                                competing,
                                light_pdf > 0.0f);
                        auto light_radiance =
                            light.color *
                            (light.power *
                             light_hit_eval_factor);
                        light_radiance *=
                            analytic_light_shader(
                                light,
                                light_hit_index,
                                light_hit_position,
                                light_hit_normal,
                                light_hit_uv,
                                -ray->direction(),
                                light_hit_distance);
                        const auto contribution =
                            clamp_contribution(
                                throughput *
                                    light_radiance *
                                    mis_weight,
                                path_depth);
                        radiance += contribution;
                        const auto directly_visible =
                            path_depth == 0u;
                        sample_emission += select(
                            make_float3(0.0f),
                            contribution,
                            directly_visible);
                        accumulate_light_pass(
                            split_scattered_light(
                            select(
                                contribution,
                                make_float3(0.0f),
                                directly_visible),
                            path_diffuse_weight,
                            path_glossy_weight,
                            path_depth == 1u));

                        previous_analytic_light =
                            light_hit_index;
                        transparent_depth += 1u;
                        analytic_light_terminated =
                            transparent_depth >=
                            kernel_parameters
                                .transparent_max_bounces;
                        ray = make_ray(
                            ray->origin(),
                            ray->direction(),
                            surface_ray::
                                intersection_t_offset(
                                    light_hit_distance),
                            ray->t_max());
                    }
                    $else {
                        search_analytic_lights = false;
                    };
                };
                $if (analytic_light_terminated) {
                    $break;
                };
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
                                    cycles_path_state::
                                        background_emission_shader_state(
                                            ray_visibility,
                                            ray_events,
                                            path_depth,
                                            diffuse_depth,
                                            glossy_depth,
                                            transparent_depth,
                                            transmission_depth)) *
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
