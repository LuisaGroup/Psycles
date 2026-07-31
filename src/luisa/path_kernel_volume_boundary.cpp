#include "path_kernel_volume_boundary.h"

#include "cycles_shader_identity.h"

#include <psycles/luisa/surface_ray.h>

#include <limits>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class TriangleVolumeBoundaryComponentImpl final
    : public TriangleVolumeBoundaryComponent {

  private:
    std::shared_ptr<const TrianglePrimitiveComponent>
        _primitive;

  public:
    explicit TriangleVolumeBoundaryComponentImpl(
        std::shared_ptr<
            const TrianglePrimitiveComponent>
            primitive) noexcept
        : _primitive{std::move(primitive)} {}

    Bool has_volume(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_id)
        const noexcept override {
        return _primitive
            ->emit(
                scene,
                instance_id,
                primitive_id)
            .has_volume;
    }

    TriangleVolumeBoundary resolve(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<std::uint32_t> instance_id,
        Expr<std::uint32_t> primitive_id,
        Expr<luisa::float3> ray_direction)
        const noexcept override {
        auto primitive = _primitive->emit(
            scene,
            instance_id,
            primitive_id);
        const auto positions =
            scene->heap->buffer<luisa::float3>(
                primitive.geometry.bindless_base +
                1u);
        const auto p0 =
            positions.read(primitive.triangle.i0);
        const auto p1 =
            positions.read(primitive.triangle.i1);
        const auto p2 =
            positions.read(primitive.triangle.i2);
        const auto object_geometric_normal =
            normalize(
                cross(
                    p1 - p0,
                    p2 - p0));
        const auto normal_to_world =
            transpose(
                inverse(
                    scene->accel
                        ->instance_transform(
                            instance_id)));
        Float3 geometric_normal =
            normalize(
                (normal_to_world *
                 make_float4(
                     object_geometric_normal,
                     0.0f))
                    .xyz());
        Bool back_facing =
            dot(
                geometric_normal,
                -ray_direction) < 0.0f;
        return {
            .primitive =
                std::move(primitive),
            .geometric_normal =
                std::move(geometric_normal),
            .back_facing =
                std::move(back_facing)};
    }
};

class CameraVolumeStackComponentImpl final
    : public CameraVolumeStackComponent {

  private:
    std::shared_ptr<
        const TriangleVolumeBoundaryComponent>
        _boundary;

    void _initialize_background(
        const std::shared_ptr<LuisaSceneData> &scene,
        VolumeStack &stack) const noexcept {
        if (!scene->world_surface) {
            return;
        }
        const auto &binding =
            *scene->world_surface;
        const auto has_volume =
            (binding.flags &
             material_flag_has_volume) != 0u;
        const auto has_object_identity =
            scene->cycles_background_object_index !=
            cycles_shader_identity::invalid_index;
        const auto base_shader =
            binding.cycles_shader_index !=
                    cycles_shader_identity::
                        invalid_index
                ? binding.cycles_shader_index
                : binding.material_identity;
        const auto has_shader_identity =
            base_shader !=
            cycles_shader_identity::invalid_index;
        if (!has_volume ||
            !has_object_identity ||
            !has_shader_identity) {
            return;
        }
        stack.initialize_background(
            VolumeStackEntry{
                .object =
                    scene
                        ->cycles_background_object_index,
                .shader =
                    cycles_shader_identity::surface(
                        base_shader,
                        false),
                .surface_tag =
                    binding.surface_tag,
                .parameter_block =
                    binding.parameter_block,
                .instance_id =
                    invalid_volume_identity,
                .valid = true},
            true);
    }

  public:
    explicit CameraVolumeStackComponentImpl(
        std::shared_ptr<
            const TriangleVolumeBoundaryComponent>
            boundary) noexcept
        : _boundary{std::move(boundary)} {}

    void initialize_background(
        const std::shared_ptr<LuisaSceneData> &scene,
        VolumeStack &stack) const noexcept override {
        _initialize_background(scene, stack);
    }

    CameraVolumeStackInitialization
    initialize(
        const std::shared_ptr<LuisaSceneData> &scene,
        Expr<luisa::float3> camera_origin,
        Expr<std::uint32_t> visibility,
        VolumeStack &stack) const noexcept override {
        stack.clear();
        _initialize_background(
            scene, stack);
        VolumeStackCameraInitializer initializer{
            stack};
        UInt intersection_count = 0u;
        UInt source_instance =
            surface_ray::invalid_primitive;
        UInt source_primitive =
            surface_ray::invalid_primitive;
        Bool active = true;
        Var<luisa::compute::Ray> probe =
            make_ray(
                camera_origin,
                make_float3(
                    camera_volume_probe_direction),
                0.0f,
                std::numeric_limits<float>::
                    max());
        $while(
            active &
            initializer.can_continue(
                intersection_count)) {
            auto hit =
                scene->accel
                    ->traverse(
                        probe,
                        {.visibility_mask =
                             visibility})
                    .on_surface_candidate(
                        [&](luisa::compute::
                                SurfaceCandidate
                                    &candidate) noexcept {
                            const auto candidate_hit =
                                candidate.hit();
                            const auto distinct =
                                !surface_ray::
                                    same_primitive(
                                        candidate_hit
                                            ->inst,
                                        candidate_hit
                                            ->prim,
                                        source_instance,
                                        source_primitive);
                            const auto has_volume =
                                _boundary
                                    ->has_volume(
                                        scene,
                                        candidate_hit
                                            ->inst,
                                        candidate_hit
                                            ->prim);
                            $if(
                                distinct &
                                has_volume) {
                                candidate.commit();
                            };
                        })
                    .on_procedural_candidate(
                        [](luisa::compute::
                               ProceduralCandidate
                                   &) noexcept {})
                    .trace();
            $if(hit->miss()) {
                active = false;
            }
            $else {
                auto boundary =
                    _boundary->resolve(
                        scene,
                        hit->inst,
                        hit->prim,
                        probe->direction());
                initializer.observe(
                    boundary.primitive
                        .volume_stack_entry(),
                    boundary.back_facing,
                    boundary.primitive
                        .has_volume);
                source_instance =
                    hit->inst;
                source_primitive =
                    hit->prim;
                probe = make_ray(
                    probe->origin(),
                    probe->direction(),
                    surface_ray::
                        intersection_t_offset(
                            hit
                                ->committed_ray_t),
                    probe->t_max());
                intersection_count += 1u;
            };
        };
        return {
            .intersection_count =
                std::move(intersection_count),
            .enclosed_count =
                initializer.enclosed_count()};
    }
};

}// namespace

std::shared_ptr<const TriangleVolumeBoundaryComponent>
make_triangle_volume_boundary_component(
    std::shared_ptr<const TrianglePrimitiveComponent>
        primitive) {
    if (!primitive) {
        primitive =
            make_triangle_primitive_component();
    }
    return std::make_shared<
        TriangleVolumeBoundaryComponentImpl>(
        std::move(primitive));
}

std::shared_ptr<const CameraVolumeStackComponent>
make_camera_volume_stack_component() {
    return std::make_shared<
        CameraVolumeStackComponentImpl>(
        make_triangle_volume_boundary_component());
}

}// namespace psycles::luisa_backend::detail
