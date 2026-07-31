#include "path_kernel_volume_majorant_provider.h"

#include "path_tracer_volume_capabilities.h"

#include <limits>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;

class SceneVolumeMajorantEntryProvider final
    : public VolumeMajorantEntryProvider {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<
        const VolumeStackEntryPointProvider>
        _points;
    const ShaderServices &_services;
    const VolumeShadingState &_state;
    bool _evaluate_emission;

    [[nodiscard]] UInt _surface_flags(
        const VolumeStackEntry &entry)
        const noexcept {
        UInt flags = 0u;
        $if(entry.surface_tag <
            _scene->volume_surface_flag_count) {
            flags =
                _scene->volume_surface_flag_buffer
                    ->read(entry.surface_tag);
        };
        return flags;
    }

  public:
    SceneVolumeMajorantEntryProvider(
        std::shared_ptr<LuisaSceneData> scene,
        std::shared_ptr<
            const VolumeStackEntryPointProvider>
            points,
        const ShaderServices &services,
        const VolumeShadingState &state,
        bool evaluate_emission) noexcept
        : _scene{std::move(scene)},
          _points{std::move(points)},
          _services{services},
          _state{state},
          _evaluate_emission{
              evaluate_emission} {}

    VolumeMajorantEntrySpace entry_space(
        const VolumeStackEntry &entry,
        Float3 world_ray_origin,
        Float3 world_ray_direction)
        const noexcept override {
        Float3 ray_origin =
            world_ray_origin;
        Float3 ray_direction =
            world_ray_direction;
        const auto object_entry =
            entry.instance_id !=
            invalid_volume_identity;
        // A World entry has no TLAS instance. Keep the transform query in the
        // object branch rather than selecting a nominal index zero.
        $if(object_entry) {
            const auto object_to_world =
                _scene->accel->instance_transform(
                    entry.instance_id);
            const auto world_to_object =
                inverse(object_to_world);
            ray_origin =
                (world_to_object *
                 make_float4(
                     world_ray_origin, 1.0f))
                    .xyz();
            ray_direction =
                (world_to_object *
                 make_float4(
                     world_ray_direction,
                     0.0f))
                    .xyz();
        };
        return {
            .ray_origin =
                std::move(ray_origin),
            .ray_direction =
                std::move(ray_direction),
            .object_density =
                _points->object_density(entry)};
    }

    VolumeMajorantRuntimeExtrema extrema(
        const VolumeStackEntry &entry,
        const VolumeMajorantLeaf &leaf,
        Float object_density,
        Float shade_offset,
        Float3 world_ray_origin,
        Float3 world_ray_direction)
        const noexcept override {
        auto result =
            VolumeMajorantEntryProvider::extrema(
                entry,
                leaf,
                object_density,
                shade_offset,
                world_ray_origin,
                world_ray_direction);
        const auto flags =
            _surface_flags(entry);
        const auto has_light_path =
            (flags &
             volume_surface_flag_light_path) !=
            0u;
        const auto camera_ray =
            (_state.ray_visibility &
             camera_visibility) != 0u;
        $if(has_light_path & !camera_ray) {
            const auto heterogeneous =
                (flags &
                 volume_surface_flag_heterogeneous) !=
                0u;
            const auto samples =
                select(1u, 4u, heterogeneous);
            const auto sample_offset =
                select(
                    0.5f,
                    shade_offset,
                    heterogeneous);
            const auto step =
                (leaf.maximum - leaf.minimum) /
                cast<float>(samples);
            Float minimum =
                std::numeric_limits<float>::max();
            Float maximum =
                -std::numeric_limits<float>::max();
            UInt index = 0u;
            $while(index < samples) {
                const auto shade_t =
                    leaf.minimum +
                    (sample_offset +
                     cast<float>(index)) *
                        step;
                const VolumeShadingState
                    sample_state{
                        .position =
                            world_ray_origin +
                            world_ray_direction *
                                shade_t,
                        .incoming =
                            _state.incoming,
                        .ray_visibility =
                            _state.ray_visibility,
                        .ray_events =
                            _state.ray_events,
                        .ray_depth =
                            _state.ray_depth,
                        .diffuse_depth =
                            _state.diffuse_depth,
                        .glossy_depth =
                            _state.glossy_depth,
                        .transparent_depth =
                            _state
                                .transparent_depth,
                        .transmission_depth =
                            _state
                                .transmission_depth,
                        .ray_length =
                            _state.ray_length,
                        .time =
                            _state.time};
                const auto shading =
                    _points->emit(
                        entry,
                        sample_state);
                const auto coefficients =
                    _scene->surfaces
                        .volume_coefficients(
                            entry.surface_tag,
                            _services,
                            shading.point,
                            VolumeQuery{
                                .object_density =
                                    shading
                                        .object_density,
                                .evaluate_emission =
                                    _evaluate_emission});
                const auto extinction =
                    max(
                        coefficients.sigma_t.x,
                        max(
                            coefficients.sigma_t.y,
                            coefficients
                                .sigma_t.z));
                const auto emission =
                    max(
                        coefficients.emission.x,
                        max(
                            coefficients.emission.y,
                            coefficients
                                .emission.z));
                const auto sigma =
                    max(
                        extinction,
                        emission);
                minimum =
                    min(minimum, sigma);
                maximum =
                    max(maximum, sigma);
                index += 1u;
            };
            result.minimum = minimum;
            result.maximum =
                select(
                    maximum,
                    max(0.5f, maximum * 1.5f),
                    heterogeneous);
        };
        return result;
    }
};

}// namespace

std::unique_ptr<VolumeMajorantEntryProvider>
make_scene_volume_majorant_entry_provider(
    std::shared_ptr<LuisaSceneData> scene,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    const ShaderServices &services,
    const VolumeShadingState &state,
    bool evaluate_emission) {
    return std::make_unique<
        SceneVolumeMajorantEntryProvider>(
        std::move(scene),
        std::move(points),
        services,
        state,
        evaluate_emission);
}

}// namespace psycles::luisa_backend::detail
