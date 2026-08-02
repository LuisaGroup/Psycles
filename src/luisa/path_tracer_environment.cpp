#include "path_tracer_environment.h"

#include "path_tracer_shader_services.h"

#include <psycles/luisa/background_sampling.h>
#include <psycles/luisa/cycles_nishita.h>
#include <psycles/sampling/background_distribution.h>

namespace psycles::luisa_backend::detail {

namespace {

[[nodiscard]] Vec3f normalized_or_z(
    Vec3f direction) noexcept {
    const auto length_squared =
        direction.x * direction.x +
        direction.y * direction.y +
        direction.z * direction.z;
    if (!(length_squared > 1.0e-20f) ||
        !std::isfinite(length_squared)) {
        return {0.0f, 0.0f, 1.0f};
    }
    const auto inverse_length =
        1.0f / std::sqrt(length_squared);
    return {
        direction.x * inverse_length,
        direction.y * inverse_length,
        direction.z * inverse_length};
}

}// namespace

EnvironmentCallables make_environment_callables(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize,
    const SurfaceEmissionCallable &surface_emission) {
    EnvironmentBaseCallable base =
        [scene, surface_emission](
            Float3 direction,
            Float3 background,
            Var<ShaderEvaluationStateCall>
                shader_state_call) noexcept {
            const auto shader_state =
                unpack_shader_evaluation_state(
                    shader_state_call);
            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto evaluate_world_graph =
                [&](Float3 world_direction) noexcept {
                    Float3 world = background;
                    if (scene->world_surface) {
                        SurfacePoint world_point{
                            .position = make_float3(0.0f),
                            .object_position =
                                make_float3(0.0f),
                            .object_location =
                                make_float3(0.0f),
                            .generated = world_direction,
                            .geometric_normal =
                                -world_direction,
                            .shading_normal =
                                -world_direction,
                            .dpdu = make_float3(
                                1.0f, 0.0f, 0.0f),
                            .dpdv = make_float3(
                                0.0f, 1.0f, 0.0f),
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
                            .incoming = -world_direction,
                            .uv = make_float2(0.0f),
                            .uv_dx = make_float2(0.0f),
                            .uv_dy = make_float2(0.0f),
                            .geometry_index = ~0u,
                            .barycentric =
                                make_float2(0.0f),
                            .barycentric_dx =
                                make_float2(0.0f),
                            .barycentric_dy =
                                make_float2(0.0f),
                            .instance_id = 0u,
                            .primitive_id = 0u,
                            .parameter_block =
                                scene->world_surface
                                    ->parameter_block,
                            .object_random = 0.0f,
                            .particle_index = 0u,
                            .random_per_island = 0.0f,
                            .ray_visibility = 0u,
                            .ray_events = 0u,
                            .ray_depth = 0u,
                            .diffuse_depth = 0u,
                            .glossy_depth = 0u,
                            .transparent_depth = 0u,
                            .transmission_depth = 0u,
                            .ray_length =
                                std::numeric_limits<
                                    float>::max(),
                            .time = 0.0f,
                            .back_facing = false};
                        cycles_path_state::
                            apply_shader_state(
                                world_point,
                                shader_state);
                        world += surface_emission(
                            scene->parameter_buffer,
                            scene->cycles_bsdf_table_buffer,
                            scene->texture_heap,
                            scene->heap,
                            UInt{
                                scene->world_surface
                                    ->surface_tag},
                            pack_surface_point(world_point),
                            -world_direction);
                    }
                    return world;
                };
            if (scene->environment_texture_slot) {
                if (scene->nishita_environment) {
                    const auto &sky =
                        scene->nishita_environment
                            ->parameters;
                    return max(
                               services.xyz_to_rgb(
                                   cycles_nishita::
                                       sky_radiance_xyz(
                                           scene->texture_heap
                                               ->tex2d(
                                                   *scene
                                                        ->environment_texture_slot),
                                           direction,
                                           sky.sun_rotation)),
                               make_float3(0.0f)) *
                           sky.background_strength;
                }
                auto u = fract(
                    (pi - atan2(direction.y, direction.x)) /
                    (2.0f * pi));
                auto half_texel_y =
                    0.5f /
                    static_cast<float>(std::max(
                        scene->environment_height, 1u));
                auto v = clamp(
                    acos(clamp(direction.z, -1.0f, 1.0f)) /
                        pi,
                    half_texel_y,
                    1.0f - half_texel_y);
                return scene->texture_heap
                    ->tex2d(*scene->environment_texture_slot)
                    .sample(make_float2(u, v))
                    .xyz();
            }
            return evaluate_world_graph(direction);
        };

    std::vector<EnvironmentSunCallable> suns;
    suns.reserve(scene->environment_suns.size());
    for (const auto &sun : scene->environment_suns) {
        EnvironmentSunCallable evaluate =
            [safe_normalize, sun](
                Float3 direction) noexcept {
                Float3 axis = safe_normalize(
                    to_luisa(sun.direction),
                    make_float3(0.0f, 0.0f, 1.0f));
                auto cosine = clamp(
                    dot(direction, axis), -1.0f, 1.0f);
                auto radius = std::max(
                    sun.angular_radius, 1.0e-7f);
                auto radial_distance =
                    acos(cosine) / radius;
                auto limb =
                    0.4f +
                    0.6f *
                        sqrt(max(
                            1.0f -
                                radial_distance *
                                    radial_distance,
                            0.0f));
                auto inside =
                    cosine >= std::cos(sun.angular_radius);
                return select(
                    make_float3(0.0f),
                    to_luisa(sun.radiance) *
                        (limb / 0.8f),
                    inside);
            };
        suns.emplace_back(std::move(evaluate));
    }

    EnvironmentSunCallable nishita_sun =
        [scene](Float3 direction) noexcept {
            if (!scene->nishita_environment ||
                scene->nishita_environment
                        ->angular_radius <=
                    0.0f) {
                return Float3{make_float3(0.0f)};
            }
            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            const auto &sun =
                *scene->nishita_environment;
            const auto &sky = sun.parameters;
            return max(
                       services.xyz_to_rgb(
                           cycles_nishita::
                               sun_disc_radiance_xyz(
                                   direction,
                                   make_float3(
                                       sun.sun_direction),
                                   make_float3(
                                       sun.pixel_bottom_xyz),
                                   make_float3(
                                       sun.pixel_top_xyz),
                                   sky.sun_elevation,
                                   sky.angular_diameter,
                                   sky.sun_intensity)),
                       make_float3(0.0f)) *
                   sky.background_strength;
        };

    return {
        std::move(base),
        std::move(suns),
        std::move(nishita_sun)};
}

void configure_background_sampling(
    LuisaSceneData &data,
    const SceneSnapshot &snapshot,
    bool include_environment) noexcept {
    data.background_map_width = 1u;
    data.background_map_height = 1u;
    data.background_map_weight =
        include_environment ? 1.0f : 0.0f;
    data.background_guided_sun_weight = 0.0f;
    data.background_guided_sun_axis =
        luisa::make_float3(0.0f, 0.0f, 1.0f);
    data.background_guided_sun_radius = 0.0f;

    if (!include_environment) {
        return;
    }

    // Match Cycles' single-Sun guidance contract. Multiple solar discs stay
    // in the importance map because one analytic cone cannot represent their
    // support without changing the estimator.
    if (data.nishita_environment &&
        data.nishita_environment->angular_radius > 0.0f) {
        data.background_guided_sun_weight = 4.0f;
        data.background_guided_sun_axis =
            data.nishita_environment->sun_direction;
        data.background_guided_sun_radius =
            data.nishita_environment->angular_radius;
    } else if (data.environment_suns.size() == 1u &&
               data.environment_suns.front()
                       .angular_radius >
                   0.0f) {
        const auto &sun = data.environment_suns.front();
        const auto axis = normalized_or_z(sun.direction);
        data.background_guided_sun_weight = 4.0f;
        data.background_guided_sun_axis =
            luisa::make_float3(axis.x, axis.y, axis.z);
        data.background_guided_sun_radius =
            sun.angular_radius;
    }

    if (snapshot.world_sampling ==
        contract::WorldSampling::manual) {
        data.background_map_width =
            std::max(
                snapshot.world_sample_map_resolution,
                2u);
        data.background_map_height =
            std::max(
                data.background_map_width / 2u,
                1u);
        return;
    }

    if (data.nishita_environment &&
        data.background_guided_sun_weight > 0.0f) {
        // Cycles raises an automatically sized guided Nishita map to this
        // resolution even though the atmosphere LUT itself is smaller.
        data.background_map_width = 512u;
        data.background_map_height = 256u;
    } else if (
        snapshot.environment &&
        snapshot.environment->width > 0u &&
        snapshot.environment->height > 0u) {
        data.background_map_width =
            snapshot.environment->width;
        data.background_map_height =
            snapshot.environment->height;
    } else {
        data.background_map_width = 1024u;
        data.background_map_height = 512u;
    }
}

void build_background_sampling_distribution(
    const std::shared_ptr<LuisaSceneData> &data,
    Stream &stream) {
    std::vector<Vec3f> radiance;
    if (data->background_map_weight > 0.0f) {
        const auto pixel_count =
            static_cast<std::size_t>(
                data->background_map_width) *
            static_cast<std::size_t>(
                data->background_map_height);
        auto radiance_buffer =
            data->device.create_buffer<luisa::float4>(
                pixel_count);
        luisa::vector<luisa::float4> readback(
            pixel_count);

        SafeNormalizeCallable safe_normalize =
            [](Float3 value,
               Float3 fallback) noexcept {
                const auto length_squared =
                    dot(value, value);
                return select(
                    fallback,
                    value /
                        sqrt(max(
                            length_squared,
                            1.0e-20f)),
                    length_squared > 1.0e-20f);
            };
        auto surface_callables =
            make_surface_callables(data);
        auto surface_emission =
            surface_callables.emission;
        auto environment_callables =
            make_environment_callables(
                data,
                safe_normalize,
                surface_emission);
        auto environment_base =
            environment_callables.base;
        auto environment_suns =
            environment_callables.suns;
        auto nishita_sun =
            environment_callables.nishita_sun;
        const auto width =
            data->background_map_width;
        const auto height =
            data->background_map_height;
        const auto include_discrete_suns =
            data->background_guided_sun_weight <=
            0.0f;
        const auto background = data->background;

        Kernel2D evaluate_importance = [
            =,
            &surface_emission](
            BufferFloat4 output) noexcept {
            set_block_size(8u, 8u, 1u);
            const auto coordinate =
                dispatch_id().xy();
            const auto u =
                (cast<float>(coordinate.x) + 0.5f) /
                static_cast<float>(width);
            const auto v =
                (cast<float>(coordinate.y) + 0.5f) /
                static_cast<float>(height);
            const auto direction =
                background_sampling::
                    equirectangular_to_direction(
                        u, v);
            Float3 value = environment_base(
                direction,
                make_float3(background),
                pack_shader_evaluation_state(
                    cycles_path_state::
                        light_emission_shader_state(
                            0u, 0u, 0u, 0u, 0u)));
            if (include_discrete_suns) {
                for (const auto &sun :
                     environment_suns) {
                    value += sun(direction);
                }
                value += nishita_sun(direction);
            }
            output.write(
                coordinate.y * width +
                    coordinate.x,
                make_float4(value, 1.0f));
        };
        auto importance_shader =
            data->device.compile(
                evaluate_importance);
        stream
            << importance_shader(
                   radiance_buffer)
                   .dispatch(width, height)
            << radiance_buffer.copy_to(
                   luisa::span{readback})
            << synchronize();

        radiance.reserve(readback.size());
        for (const auto value : readback) {
            const auto finite_or_zero =
                [](float component) noexcept {
                    return std::isfinite(component)
                               ? component
                               : 0.0f;
                };
            radiance.emplace_back(
                finite_or_zero(value.x),
                finite_or_zero(value.y),
                finite_or_zero(value.z));
        }
    } else {
        radiance.emplace_back(1.0f, 1.0f, 1.0f);
    }

    const auto distribution =
        sampling::
            build_cycles_background_map_distribution(
                radiance,
                data->background_map_width,
                data->background_map_height);
    luisa::vector<luisa::float2> conditional;
    conditional.reserve(
        distribution.conditional.size());
    for (const auto entry :
         distribution.conditional) {
        conditional.emplace_back(
            entry.function,
            entry.cumulative);
    }
    luisa::vector<luisa::float2> marginal;
    marginal.reserve(
        distribution.marginal.size());
    for (const auto entry :
         distribution.marginal) {
        marginal.emplace_back(
            entry.function,
            entry.cumulative);
    }
    data->background_conditional_cdf =
        data->device.create_buffer<luisa::float2>(
            conditional.size());
    data->background_marginal_cdf =
        data->device.create_buffer<luisa::float2>(
            marginal.size());
    stream
        << data->background_conditional_cdf
               .copy_from(luisa::span{conditional})
        << data->background_marginal_cdf
               .copy_from(luisa::span{marginal})
        << synchronize();
}

}// namespace psycles::luisa_backend::detail
