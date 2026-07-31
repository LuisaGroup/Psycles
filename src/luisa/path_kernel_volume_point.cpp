#include "path_kernel_volume_point.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class SceneVolumeStackEntryPointProvider final
    : public VolumeStackEntryPointProvider {

  private:
    std::shared_ptr<LuisaSceneData> _scene;

  public:
    explicit SceneVolumeStackEntryPointProvider(
        std::shared_ptr<LuisaSceneData> scene) noexcept
        : _scene{std::move(scene)} {}

    VolumeStackEntryShading emit(
        const VolumeStackEntry &entry,
        const VolumeShadingState &state)
        const noexcept override {
        const auto object_entry =
            entry.instance_id !=
            invalid_volume_identity;

        Float3 object_position = state.position;
        Float3 object_location =
            make_float3(0.0f);
        Float3 generated = state.position;
        Float3 object_shading_normal =
            state.incoming;
        Float3 normal_to_world_x =
            make_float3(1.0f, 0.0f, 0.0f);
        Float3 normal_to_world_y =
            make_float3(0.0f, 1.0f, 0.0f);
        Float3 normal_to_world_z =
            make_float3(0.0f, 0.0f, 1.0f);
        Float object_random = 0.0f;
        UInt particle_index = 0u;

        // Keep the acceleration-structure transform query inside the device
        // branch. A world-only scene has no TLAS instance at index zero, so
        // selecting a nominal "safe" index would still be invalid.
        $if(object_entry) {
            const auto instance =
                _scene->instance_buffer->read(
                    entry.instance_id);
            const auto geometry =
                _scene->geometry_buffer->read(
                    instance.geometry_index);
            const auto object_to_world =
                _scene->accel->instance_transform(
                    entry.instance_id);
            const auto world_to_object =
                inverse(object_to_world);
            const auto normal_to_world =
                transpose(world_to_object);

            object_position =
                (world_to_object *
                 make_float4(
                     state.position, 1.0f))
                    .xyz();
            object_location =
                (object_to_world *
                 make_float4(
                     0.0f, 0.0f, 0.0f, 1.0f))
                    .xyz();
            generated =
                (geometry.generated_transform *
                 make_float4(
                     object_position, 1.0f))
                    .xyz();
            object_shading_normal =
                normalize(
                    (transpose(object_to_world) *
                     make_float4(
                         state.incoming, 0.0f))
                        .xyz());
            normal_to_world_x =
                (normal_to_world *
                 make_float4(
                     1.0f, 0.0f, 0.0f, 0.0f))
                    .xyz();
            normal_to_world_y =
                (normal_to_world *
                 make_float4(
                     0.0f, 1.0f, 0.0f, 0.0f))
                    .xyz();
            normal_to_world_z =
                (normal_to_world *
                 make_float4(
                     0.0f, 0.0f, 1.0f, 0.0f))
                    .xyz();
            object_random =
                instance.object_random;
            particle_index =
                instance.particle_index;
        };

        SurfacePoint point{
            .position = state.position,
            .object_position =
                std::move(object_position),
            .object_location =
                std::move(object_location),
            .generated = std::move(generated),
            .geometric_normal = state.incoming,
            .shading_normal = state.incoming,
            .object_shading_normal =
                std::move(object_shading_normal),
            .object_tangent =
                make_float3(0.0f),
            .tangent_sign = 0.0f,
            .normal_to_world_x =
                std::move(normal_to_world_x),
            .normal_to_world_y =
                std::move(normal_to_world_y),
            .normal_to_world_z =
                std::move(normal_to_world_z),
            .dpdu = make_float3(0.0f),
            .dpdv = make_float3(0.0f),
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
            .incoming = state.incoming,
            .uv = make_float2(0.0f),
            .uv_dx = make_float2(0.0f),
            .uv_dy = make_float2(0.0f),
            // A volume point has PRIM_NONE. Passing a surface geometry index
            // would let named-attribute code dereference an unrelated
            // triangle. Native grid attributes will get a dedicated volume
            // service rather than overloading triangle interpolation.
            .geometry_index =
                invalid_volume_identity,
            .barycentric =
                make_float2(0.0f),
            .barycentric_dx =
                make_float2(0.0f),
            .barycentric_dy =
                make_float2(0.0f),
            .instance_id = entry.instance_id,
            .primitive_id =
                invalid_volume_identity,
            .parameter_block =
                entry.parameter_block,
            .object_random =
                std::move(object_random),
            .particle_index =
                std::move(particle_index),
            .random_per_island = 0.0f,
            .ray_visibility =
                state.ray_visibility,
            .ray_events = state.ray_events,
            .ray_depth = state.ray_depth,
            .diffuse_depth =
                state.diffuse_depth,
            .glossy_depth =
                state.glossy_depth,
            .transparent_depth =
                state.transparent_depth,
            .transmission_depth =
                state.transmission_depth,
            .ray_length = state.ray_length,
            .time = state.time,
            .back_facing = false};
        return {
            .point = std::move(point),
            // The scene contract currently represents mesh-boundary
            // volumes. Cycles scales those by one regardless of instance
            // transform; object-space VDB volumes will carry their explicit
            // object_volume_density in a future geometry kind.
            .object_density = 1.0f};
    }
};

}// namespace

std::shared_ptr<const VolumeStackEntryPointProvider>
make_scene_volume_stack_entry_point_provider(
    std::shared_ptr<LuisaSceneData> scene) {
    return std::make_shared<
        SceneVolumeStackEntryPointProvider>(
        std::move(scene));
}

}// namespace psycles::luisa_backend::detail
