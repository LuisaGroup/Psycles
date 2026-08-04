#include "path_kernel_scene_traversal.h"

#include "curve_ribbon_component.h"
#include "path_kernel_curve_primitive.h"
#include "path_kernel_primitive_material.h"

#include <psycles/luisa/surface_ray.h>

namespace psycles::luisa_backend::detail {

ScenePrimitiveIdentity ScenePrimitiveIdentity::invalid() noexcept {
  return {.object = surface_ray::invalid_primitive,
          .primitive = surface_ray::invalid_primitive};
}

Bool ScenePrimitiveIdentity::matches(
    Expr<std::uint32_t> candidate_object,
    Expr<std::uint32_t> candidate_primitive) const noexcept {
  return (object == candidate_object) & (primitive == candidate_primitive);
}

namespace {

class UnifiedSceneTraversalComponent final : public SceneTraversalComponent {

private:
  std::shared_ptr<const CurvePrimitiveComponent> _curves;
  std::shared_ptr<const CurveRibbonComponent> _ribbons;
  std::shared_ptr<const PrimitiveMaterialComponent> _materials;

  [[nodiscard]] Var<luisa::compute::CommittedHit>
  trace(const std::shared_ptr<LuisaSceneData> &scene,
        const Var<luisa::compute::Ray> &ray,
        Expr<std::uint32_t> visibility_mask,
        const ScenePrimitiveIdentity &source,
        const ScenePrimitiveIdentity &light) const noexcept {
    return scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
        .on_surface_candidate(
            [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
              const auto hit = candidate.hit();
              const auto instance = scene->instance_buffer->read(hit->inst);
              const auto geometry =
                  scene->geometry_buffer->read(instance.geometry_index);
              const auto object =
                  _materials->cycles_object_index(hit->inst, instance);
              const auto primitive =
                  geometry.cycles_primitive_offset + hit->prim;
              const auto excluded = source.matches(object, primitive) |
                                    light.matches(object, primitive);
              $if(!excluded) { candidate.commit(); };
            })
        .on_procedural_candidate(
            [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
              const auto hit = candidate.hit();
              const auto curve =
                  _curves->emit_segment(scene, hit->inst, hit->prim);
              $if(curve.geometry.primitive_kind == geometry_kind_curve) {
                const auto object =
                    _materials->cycles_object_index(hit->inst, curve.instance);
                const auto primitive = curve.segment.cycles_curve_index;
                const auto excluded = source.matches(object, primitive) |
                                      light.matches(object, primitive);
                const auto world_to_object =
                    inverse(scene->accel->instance_transform(hit->inst));
                const auto candidate_ray = candidate.ray();
                const auto object_ray =
                    make_ray((world_to_object *
                              make_float4(candidate_ray->origin(), 1.0f))
                                 .xyz(),
                             (world_to_object *
                              make_float4(candidate_ray->direction(), 0.0f))
                                 .xyz(),
                             candidate_ray->t_min(), candidate_ray->t_max());
                const auto intersection =
                    _ribbons->intersect(object_ray, curve.control_points,
                                        curve.geometry.curve_subdivision_level);
                $if(!excluded & intersection.valid) {
                  candidate.commit(intersection.distance);
                };
              };
            })
        .trace();
  }

public:
  UnifiedSceneTraversalComponent()
      : _curves{make_curve_primitive_component()},
        _ribbons{make_curve_ribbon_component()},
        _materials{make_primitive_material_component()} {}

  Var<luisa::compute::CommittedHit>
  closest(const std::shared_ptr<LuisaSceneData> &scene,
          const Var<luisa::compute::Ray> &ray,
          Expr<std::uint32_t> visibility_mask,
          const ScenePrimitiveIdentity &source) const noexcept override {
    return trace(scene, ray, visibility_mask, source,
                 ScenePrimitiveIdentity::invalid());
  }

  Var<luisa::compute::CommittedHit>
  closest_shadow(const std::shared_ptr<LuisaSceneData> &scene,
                 const Var<luisa::compute::Ray> &ray,
                 Expr<std::uint32_t> visibility_mask,
                 const ScenePrimitiveIdentity &source,
                 const ScenePrimitiveIdentity &light) const noexcept override {
    return trace(scene, ray, visibility_mask, source, light);
  }
};

} // namespace

std::shared_ptr<const SceneTraversalComponent>
make_scene_traversal_component() {
  return std::make_shared<UnifiedSceneTraversalComponent>();
}

} // namespace psycles::luisa_backend::detail
