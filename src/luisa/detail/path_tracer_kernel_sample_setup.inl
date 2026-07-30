// Kernel ABI, film state, callable adapters, and camera/sample initialization.
// Included by path_tracer_kernel.cpp inside LuisaRenderSession::initialize.

    Kernel1D kernel = [
                          =,
                          &shared_surface_evaluate,
                          &shared_surface_emission,
                          &shared_surface_sample,
                          &shared_surface_closure_trace,
                          &shared_surface_sample_trace,
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
        auto trace_surface_closure =
            [&](UInt surface_tag,
                const SurfacePoint &point,
                UInt requested_index) noexcept {
                return unpack_surface_closure_trace(
                    shared_surface_closure_trace(
                        scene->parameter_buffer,
                        scene->cycles_bsdf_table_buffer,
                        scene->texture_heap,
                        scene->heap,
                        surface_tag,
                        pack_surface_point(point),
                        requested_index));
            };
        auto trace_sample_surface =
            [&](UInt surface_tag,
                const SurfacePoint &point,
                Float u_lobe,
                Float2 u_direction,
                const SurfaceQuery &query) noexcept {
                return unpack_surface_sample_trace(
                    shared_surface_sample_trace(
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
                const cycles_path_state::
                    ShaderEvaluationState
                        &shader_state) noexcept {
                Float3 result =
                    environment_base_callable(
                        direction,
                        kernel_parameters.background,
                        pack_shader_evaluation_state(
                            shader_state));
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
            const auto camera_dimensions =
                sample_camera_dimensions(
                    sobol_table,
                    kernel_parameters,
                    full_x,
                    full_y,
                    sample_index);
            const UInt cycles_y =
                camera_dimensions.cycles_y;
            const UInt rng_hash =
                camera_dimensions.rng_hash;
            const Float2 filter_sample =
                camera_dimensions.filter_sample;
            const Float3 lens_time_sample =
                camera_dimensions.lens_time_sample;
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
            auto trace_write_closure =
                [&](UInt event,
                    std::uint32_t closure,
                    std::uint32_t field,
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
                                    path_trace_schema::
                                        EventSlot::
                                            closure_base) +
                                closure * 3u + field,
                            value);
                    };
                };
            auto camera_ray = construct_camera_ray(
                filter_table,
                kernel_parameters,
                full_x,
                full_y,
                camera_dimensions,
                camera_projection,
                camera_depth_of_field,
                camera_aperture_blades,
                camera_aperture_rotation,
                safe_normalize);
            Var<luisa::compute::Ray> ray =
                std::move(camera_ray.ray);
            Float ray_dP =
                camera_ray.differential_position;
            Float ray_dD =
                camera_ray.differential_direction;
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
            Float3 previous_mis_origin_normal =
                make_float3(0.0f);
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
            const auto initial_cycles_path_state =
                cycles_path_state::initial_state();
            UInt path_flags =
                initial_cycles_path_state.flag;
            UInt cycles_path_visibility =
                initial_cycles_path_state.visibility;
            UInt cycles_rng_offset =
                initial_cycles_path_state.rng_offset;
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
            auto analytic_light_shader =
                [&](Var<LightGpu> light,
                    UInt light_index,
                    Float3 light_position,
                    Float3 light_normal,
                    Float2 light_uv,
                    Float3 incoming,
                    Float light_distance) noexcept {
                    Float3 result = make_float3(1.0f);
                    $if (light.surface_tag !=
                         ~std::uint32_t{0u}) {
                        Float3 relative_position =
                            light_position -
                            light.position;
                        const auto light_transform =
                            analytic_light_sampling::
                                light_linear_transform(
                                    light.axis_x,
                                    light.axis_y,
                                    light.axis_z,
                                    light.axis_scale);
                        Float3 object_position =
                            analytic_light_sampling::
                                world_to_light_direction(
                                    relative_position,
                                    light_transform);
                        Float3 object_normal =
                            analytic_light_sampling::
                                world_to_light_normal(
                                    light_normal,
                                    light_transform);
                        SurfacePoint light_point{
                            .position = light_position,
                            .object_position =
                                object_position,
                            .object_location =
                                light.position,
                            .generated = object_position,
                            .geometric_normal =
                                light_normal,
                            .shading_normal =
                                light_normal,
                            .object_shading_normal =
                                object_normal,
                            .object_tangent =
                                light.axis_x,
                            .tangent_sign = 1.0f,
                            .normal_to_world_x =
                                light_transform
                                    .inverse_row_x,
                            .normal_to_world_y =
                                light_transform
                                    .inverse_row_y,
                            .normal_to_world_z =
                                light_transform
                                    .inverse_row_z,
                            .dpdu =
                                make_float3(0.0f),
                            .dpdv =
                                make_float3(0.0f),
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
                            .incoming = incoming,
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
                            .ray_events = ray_events,
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
                            .time = 0.0f,
                            .back_facing = false};
                        cycles_path_state::
                            apply_shader_state(
                                light_point,
                                cycles_path_state::
                                    light_emission_shader_state(
                                        path_depth,
                                        diffuse_depth,
                                        glossy_depth,
                                        transparent_depth,
                                        transmission_depth));
                        result =
                            surface_emission(
                                light.surface_tag,
                                light_point,
                                incoming);
                    };
                    return result;
                };
