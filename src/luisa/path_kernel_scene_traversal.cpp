#include "path_kernel_scene_traversal.h"

#include "curve_ribbon_component.h"
#include "cycles_triangle_intersection_component.h"
#include "path_kernel_curve_primitive.h"
#include "path_kernel_primitive_material.h"

#include <psycles/luisa/surface_ray.h>

#include <limits>

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

// Cycles sorts equal-centroid BVH references by object, primitive, and
// primitive type. Its closest-hit traversal accepts t == tmax and visits the
// later child last, so coincident references resolve to the lexicographically
// greatest stable Cycles identity. Backend acceleration structures may expose
// equal-distance candidates in a different order after every rebuild. Keep
// the same total order explicitly instead of inheriting backend traversal
// order; repeated spatial-split references then become idempotent as well.
class CyclesClosestHitOrder {

private:
  Bool _has_hit{false};
  Float _distance{0.0f};
  UInt _object{0u};
  UInt _primitive{0u};
  UInt _type_order{0u};

public:
  [[nodiscard]] Bool accepts(Expr<float> distance,
                             Expr<std::uint32_t> object,
                             Expr<std::uint32_t> primitive,
                             Expr<std::uint32_t> type_order) const noexcept {
    const auto identity_is_later =
        (object > _object) |
        ((object == _object) &
         ((primitive > _primitive) |
          ((primitive == _primitive) &
           (type_order > _type_order))));
    return !_has_hit | (distance < _distance) |
           ((distance == _distance) & identity_is_later);
  }

  void select(Expr<float> distance,
              Expr<std::uint32_t> object,
              Expr<std::uint32_t> primitive,
              Expr<std::uint32_t> type_order) noexcept {
    _has_hit = true;
    _distance = distance;
    _object = object;
    _primitive = primitive;
    _type_order = type_order;
  }
};

[[nodiscard]] Float distance_upper_neighbor(Expr<float> distance) noexcept {
  // Surface distances are non-negative. Incrementing the IEEE-754 encoding is
  // therefore nextafter(distance, +infinity); use the smallest normal at zero
  // so FTZ execution cannot collapse the widened interval back to zero.
  const auto successor = as<float>(as<uint>(distance) + 1u);
  return select(successor, std::numeric_limits<float>::min(),
                distance == 0.0f);
}

[[nodiscard]] Float distance_lower_neighbor(Expr<float> distance) noexcept {
  // Cycles path distances are non-negative. Decrementing the IEEE-754 encoding
  // is therefore nextafter(distance, -infinity). At zero, use the negative
  // smallest normal: a negative subnormal can be flushed back to -0, which
  // would leave Vulkan's strict triangle lower bound unchanged.
  const auto predecessor = as<float>(as<uint>(distance) - 1u);
  return select(predecessor, -std::numeric_limits<float>::min(),
                distance == 0.0f);
}

class UnifiedSceneTraversalComponent final : public SceneTraversalComponent {

private:
  std::shared_ptr<const CurvePrimitiveComponent> _curves;
  std::shared_ptr<const CurveRibbonComponent> _ribbons;
  std::shared_ptr<const PrimitiveMaterialComponent> _materials;
  std::shared_ptr<const CyclesTriangleIntersectionComponent>
      _triangle_intersection;

  [[nodiscard]] Var<luisa::compute::CommittedHit>
  trace(const std::shared_ptr<LuisaSceneData> &scene,
        const Var<luisa::compute::Ray> &ray,
        Expr<std::uint32_t> visibility_mask,
        const ScenePrimitiveIdentity &source,
        const ScenePrimitiveIdentity &light) const noexcept {
    CyclesClosestHitOrder closest;
    Var<luisa::compute::CommittedHit> resolved;
    resolved->inst = surface_ray::invalid_primitive;
    resolved->prim = surface_ray::invalid_primitive;
    resolved->bary = make_float2(0.0f);
    resolved->hit_type =
        static_cast<std::uint32_t>(luisa::compute::HitType::Miss);
    resolved->committed_ray_t = ray->t_max();

    // Cycles' triangle and curve intersections use the closed interval
    // [tmin, tmax]. Vulkan's built-in triangle intersection is specified on
    // the open interval (tmin, tmax), while the other Luisa backends accept
    // endpoints. Expand only the backend query bounds, then filter every
    // candidate against the original interval below. This is a backend-neutral
    // encoding of the Cycles contract rather than an epsilon heuristic.
    const auto query_ray = make_ray(
        ray->origin(), ray->direction(),
        distance_lower_neighbor(ray->t_min()),
        distance_upper_neighbor(ray->t_max()));

    auto handle_surface =
        [&](luisa::compute::SurfaceCandidate &candidate,
            bool commit_candidate) noexcept {
          const auto hit = candidate.hit();
          const auto instance = scene->instance_buffer->read(hit->inst);
          const UInt coincident_count =
              max(instance.coincident_count, 1u);
          $if(coincident_count > 1u) {
            CyclesClosestHitOrder group_closest;
            Bool group_has_hit = false;
            UInt group_instance = hit->inst;
            UInt group_object = 0u;
            UInt group_primitive = 0u;
            UInt group_type = geometry_kind_triangle;
            Float group_distance = ray->t_max();
            Float2 group_barycentric = hit->bary;
            UInt alias_index = hit->inst;
            UInt remaining = coincident_count;
            $while(remaining > 0u) {
              const auto alias =
                  scene->instance_buffer->read(alias_index);
              const auto alias_geometry =
                  scene->geometry_buffer->read(alias.geometry_index);
              const auto triangle =
                  scene->heap
                      ->buffer<Triangle>(alias_geometry.bindless_base)
                      .read(hit->prim);
              const auto positions =
                  scene->heap->buffer<luisa::float3>(
                      alias_geometry.bindless_base + 9u);
              const auto intersection = _triangle_intersection->intersect(
                  ray,
                  alias.cycles_world_to_object,
                  alias.cycles_transform_applied,
                  positions.read(triangle.i0),
                  positions.read(triangle.i1),
                  positions.read(triangle.i2));
              const auto object =
                  _materials->cycles_object_index(alias_index, alias);
              const auto primitive =
                  alias_geometry.cycles_primitive_offset + hit->prim;
              const auto excluded =
                  source.matches(object, primitive) |
                  light.matches(object, primitive);
              const auto visible =
                  (alias.visibility_mask & visibility_mask) != 0u;
              const auto accepted =
                  intersection.valid & visible & !excluded &
                  group_closest.accepts(
                      intersection.distance,
                      object,
                      primitive,
                      alias_geometry.primitive_kind);
              $if(accepted) {
                group_closest.select(
                    intersection.distance,
                    object,
                    primitive,
                    alias_geometry.primitive_kind);
                group_has_hit = true;
                group_instance = alias_index;
                group_object = object;
                group_primitive = primitive;
                group_type = alias_geometry.primitive_kind;
                group_distance = intersection.distance;
                group_barycentric = intersection.barycentric;
              };
              alias_index = alias.coincident_next;
              remaining -= 1u;
            };
            const auto accepted =
                group_has_hit &
                closest.accepts(
                    group_distance,
                    group_object,
                    group_primitive,
                    group_type);
            $if(accepted) {
              if (commit_candidate) {
                candidate.commit();
              }
              closest.select(
                  group_distance,
                  group_object,
                  group_primitive,
                  group_type);
              resolved->inst = group_instance;
              resolved->prim = hit->prim;
              resolved->bary = group_barycentric;
              resolved->hit_type = static_cast<std::uint32_t>(
                  luisa::compute::HitType::Surface);
              resolved->committed_ray_t = group_distance;
            };
          } $else {
            const auto geometry =
                scene->geometry_buffer->read(instance.geometry_index);
            const auto triangle =
                scene->heap
                    ->buffer<Triangle>(geometry.bindless_base)
                    .read(hit->prim);
            const auto positions =
                scene->heap->buffer<luisa::float3>(
                    geometry.bindless_base + 9u);
            // Hardware triangle queries are a broad-phase candidate source.
            // Resolve every candidate with Cycles' Pluecker predicate so a
            // backend-specific near-origin hit cannot become a shadow or
            // closest-surface event that Cycles itself would reject.
            const auto intersection = _triangle_intersection->intersect(
                ray,
                instance.cycles_world_to_object,
                instance.cycles_transform_applied,
                positions.read(triangle.i0),
                positions.read(triangle.i1),
                positions.read(triangle.i2));
            const auto object =
                _materials->cycles_object_index(hit->inst, instance);
            const auto primitive =
                geometry.cycles_primitive_offset + hit->prim;
            const auto excluded = source.matches(object, primitive) |
                                  light.matches(object, primitive);
            const auto visible =
                (instance.visibility_mask & visibility_mask) != 0u;
            const auto accepted =
                intersection.valid & visible &
                closest.accepts(intersection.distance, object, primitive,
                                geometry.primitive_kind);
            $if(!excluded & accepted) {
              if (commit_candidate) {
                candidate.commit();
              }
              closest.select(intersection.distance, object, primitive,
                             geometry.primitive_kind);
              resolved->inst = hit->inst;
              resolved->prim = hit->prim;
              resolved->bary = intersection.barycentric;
              resolved->hit_type = static_cast<std::uint32_t>(
                  luisa::compute::HitType::Surface);
              resolved->committed_ray_t = intersection.distance;
            };
          };
        };

    auto handle_procedural =
        [&](luisa::compute::ProceduralCandidate &candidate,
            bool commit_candidate) noexcept {
          const auto hit = candidate.hit();
          const auto curve =
              _curves->emit_segment(scene, hit->inst, hit->prim);
          $if(curve.geometry.primitive_kind == geometry_kind_curve) {
            const auto object =
                _materials->cycles_object_index(hit->inst, curve.instance);
            const auto primitive = curve.segment.cycles_curve_index;
            // Cycles packs the segment ordinal above its curve primitive-type
            // bits. cycles_segment_index is strictly monotonic with that
            // ordinal within a curve, so it is order-isomorphic to Cycles'
            // final BVH-reference key without reproducing the packed enum.
            const auto type_order = curve.segment.cycles_segment_index;
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
            const auto accepted = closest.accepts(
                intersection.distance, object, primitive,
                type_order);
            const auto within_cycles_interval =
                (intersection.distance >= ray->t_min()) &
                (intersection.distance <= ray->t_max());
            $if(!excluded & intersection.valid & within_cycles_interval &
                accepted) {
              if (commit_candidate) {
                candidate.commit(intersection.distance);
              }
              closest.select(intersection.distance, object, primitive,
                             type_order);
              resolved->inst = hit->inst;
              resolved->prim = hit->prim;
              resolved->bary = make_float2(0.0f);
              resolved->hit_type = static_cast<std::uint32_t>(
                  luisa::compute::HitType::Procedural);
              resolved->committed_ray_t = intersection.distance;
            };
          };
        };

    const auto backend_hit =
        scene->accel->traverse(query_ray,
                               {.visibility_mask = visibility_mask})
            .on_surface_candidate(
                [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
                  handle_surface(candidate, true);
                })
            .on_procedural_candidate(
                [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
                  handle_procedural(candidate, true);
                })
            .trace();

    // Some ray-query implementations stop reporting candidates at exactly the
    // committed distance. Re-traverse only the segment through the backend's
    // closest hit without committing, so every equal-distance identity can be
    // folded through the Cycles order above. This bounded pass does not scan
    // geometry behind the closest surface.
    $if(!backend_hit->miss()) {
      const auto tie_ray =
          make_ray(ray->origin(), ray->direction(),
                   distance_lower_neighbor(ray->t_min()),
                   distance_upper_neighbor(
                       max(backend_hit->committed_ray_t,
                           resolved->committed_ray_t)));
      const auto ignored =
          scene->accel
              ->traverse(tie_ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(
                  [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
                    handle_surface(candidate, false);
                  })
              .on_procedural_candidate(
                  [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
                    handle_procedural(candidate, false);
                  })
              .trace();
      static_cast<void>(ignored);
    };
    return resolved;
  }

public:
  UnifiedSceneTraversalComponent()
      : _curves{make_curve_primitive_component()},
        _ribbons{make_curve_ribbon_component()},
        _materials{make_primitive_material_component()},
        _triangle_intersection{
            make_cycles_triangle_intersection_component()} {}

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
