#include "path_tracer_environment.h"

#include "path_tracer_cycles_svm_emission.h"

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

Float3 constant_environment_emission(
    const LuisaSceneData &scene, Float3 background) noexcept {
    Float3 emission = make_float3(0.0f);
    if (scene.cycles_background_shader_id != ~0u) {
        const auto constant = cycles_svm_constant_emission(
            scene, scene.cycles_background_shader_id, emission);
        assume(constant);
    }
    return background + emission;
}

Float3 evaluate_environment_emission(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    Float3 origin, Float3 direction, Float differential, Float time,
    const cycles_svm::PathState &state, UInt lcg_state,
    CyclesSvmBackgroundEvaluation evaluation) noexcept {
    if (scene->cycles_background_shader_id != ~0u) {
        // The authored world graph is authoritative, including environment
        // textures and Nishita solar discs. Sampling metadata must never
        // replace it or add a second copy of its sun.
        return parameters.background + evaluate_cycles_svm_background_emission(
            scene, parameters, origin, direction, differential, time, state,
            lcg_state, evaluation);
    }

    // Standalone contract panoramas have no world shader graph. Retain this
    // explicitly authored image input without a SurfaceProgram interpreter.
    Float3 emission = parameters.background;
    const auto xyz_to_rgb = [&](Float3 xyz) {
        const auto &c = scene->shader_color_space;
        return make_float3(dot(xyz, to_luisa(c.xyz_to_r)),
                           dot(xyz, to_luisa(c.xyz_to_g)),
                           dot(xyz, to_luisa(c.xyz_to_b)));
    };
    if (scene->environment_texture_slot) {
        if (scene->nishita_environment) {
            const auto &sky = scene->nishita_environment->parameters;
            emission = max(xyz_to_rgb(cycles_nishita::sky_radiance_xyz(
                scene->texture_heap->tex2d(*scene->environment_texture_slot),
                direction, sky.sun_rotation)), make_float3(0.0f)) * sky.background_strength;
        } else {
            const auto u = fract((pi - atan2(direction.y, direction.x)) / (2.0f * pi));
            const auto half_texel_y = 0.5f / std::max(scene->environment_height, 1u);
            const auto v = clamp(acos(clamp(direction.z, -1.0f, 1.0f)) / pi,
                                 half_texel_y, 1.0f - half_texel_y);
            emission = scene->texture_heap->tex2d(*scene->environment_texture_slot)
                           .sample(make_float2(u, v)).xyz();
        }
    }
    const auto include_suns = evaluation != CyclesSvmBackgroundEvaluation::importance_bake ||
                              scene->background_guided_sun_weight <= 0.0f;
    if (include_suns) {
        for (const auto &sun : scene->environment_suns) {
            const auto axis = normalized_or_z(sun.direction);
            const auto cosine = clamp(dot(direction, to_luisa(axis)), -1.0f, 1.0f);
            const auto radial_distance = acos(cosine) / std::max(sun.angular_radius, 1.0e-7f);
            const auto limb = 0.4f + 0.6f * sqrt(max(1.0f - radial_distance * radial_distance, 0.0f));
            emission += select(make_float3(0.0f), to_luisa(sun.radiance) * (limb / 0.8f),
                               cosine >= std::cos(sun.angular_radius));
        }
        if (scene->nishita_environment && scene->nishita_environment->angular_radius > 0.0f) {
            const auto &sun = *scene->nishita_environment;
            const auto &sky = sun.parameters;
            emission += max(xyz_to_rgb(cycles_nishita::sun_disc_radiance_xyz(
                direction, make_float3(sun.sun_direction),
                make_float3(sun.pixel_bottom_xyz), make_float3(sun.pixel_top_xyz),
                sky.sun_elevation, sky.angular_diameter, sky.sun_intensity)),
                make_float3(0.0f)) * sky.background_strength;
        }
    }
    return emission;
}

Float3 evaluate_background_importance(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters, Float u, Float v) noexcept {
    const auto ray = cycles_svm_background_bake_ray(
        u, v, scene->background_map_width, scene->background_map_height);
    const cycles_svm::PathState state{
        0u, cycles_svm::path_ray_emission | cycles_svm::path_ray_importance_bake};
    const auto value = evaluate_environment_emission(
        scene, parameters, make_float3(0.0f), ray.direction, ray.differential,
        0.5f, state, 0u, CyclesSvmBackgroundEvaluation::importance_bake);
    const auto not_nan = select(value, make_float3(0.0f), luisa::compute::dsl::isnan(value));
    return select(not_nan, make_float3(0.0f), luisa::compute::dsl::isinf(value));
}

void configure_background_sampling(
    LuisaSceneData &data,
    const SceneSnapshot &snapshot,
    bool include_environment) noexcept {
    data.background_map_width = 1u;
    data.background_map_height = 1u;
    data.background_portal_weight =
        data.portal_count > 0u ? 1.0f : 0.0f;
    data.background_map_weight =
        include_environment &&
                snapshot.world_sampling != contract::WorldSampling::none
            ? 1.0f
            : 0.0f;
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

std::shared_ptr<const BackgroundSamplingDistribution>
build_background_sampling_distribution(
    const std::shared_ptr<LuisaSceneData> &data,
    Stream &stream, const RenderKernelParameters &parameters) {
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

        const auto width = data->background_map_width;
        const auto height = data->background_map_height;
        Kernel2D evaluate_importance = [data, width, height](
            BufferFloat4 output, Var<RenderKernelParameters> parameters) noexcept {
            set_block_size(8u, 8u, 1u);
            const auto coordinate = dispatch_id().xy();
            const auto u = (cast<float>(coordinate.x) + 0.5f) / float(width);
            const auto v = (cast<float>(coordinate.y) + 0.5f) / float(height);
            const auto value = evaluate_background_importance(data, parameters, u, v);
            output.write(coordinate.y * width + coordinate.x, make_float4(value, 1.0f));
        };
        auto importance_shader =
            data->device.compile(
                evaluate_importance);
        stream
            << importance_shader(
                   radiance_buffer, parameters)
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
                data->background_map_weight > 0.0f ? data->background_map_width : 1u,
                data->background_map_weight > 0.0f ? data->background_map_height : 1u);
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
    auto result = std::make_shared<BackgroundSamplingDistribution>();
    result->conditional =
        data->device.create_buffer<luisa::float2>(
            conditional.size());
    result->marginal =
        data->device.create_buffer<luisa::float2>(
            marginal.size());
    stream
        << result->conditional
               .copy_from(luisa::span{conditional})
        << result->marginal
               .copy_from(luisa::span{marginal})
        << synchronize();
    return result;
}

}// namespace psycles::luisa_backend::detail
