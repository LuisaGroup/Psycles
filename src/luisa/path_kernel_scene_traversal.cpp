#include "path_kernel_scene_traversal.h"

#include "curve_ribbon_component.h"
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
  UInt _kind{0u};

public:
  [[nodiscard]] Bool accepts(Expr<float> distance,
                             Expr<std::uint32_t> object,
                             Expr<std::uint32_t> primitive,
                             Expr<std::uint32_t> kind) const noexcept {
    const auto identity_is_later =
        (object > _object) |
        ((object == _object) &
         ((primitive > _primitive) |
          ((primitive == _primitive) & (kind > _kind))));
    return !_has_hit | (distance < _distance) |
           ((distance == _distance) & identity_is_later);
  }

  void select(Expr<float> distance,
              Expr<std::uint32_t> object,
              Expr<std::uint32_t> primitive,
              Expr<std::uint32_t> kind) noexcept {
    _has_hit = true;
    _distance = distance;
    _object = object;
    _primitive = primitive;
    _kind = kind;
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
          const auto geometry =
              scene->geometry_buffer->read(instance.geometry_index);
          const auto object =
              _materials->cycles_object_index(hit->inst, instance);
          const auto primitive =
              geometry.cycles_primitive_offset + hit->prim;
          const auto excluded = source.matches(object, primitive) |
                                light.matches(object, primitive);
          const auto within_cycles_interval =
              (hit->committed_ray_t >= ray->t_min()) &
              (hit->committed_ray_t <= ray->t_max());
          const auto accepted =
              within_cycles_interval &
              closest.accepts(hit->committed_ray_t, object, primitive,
                              geometry.primitive_kind);
          $if(!excluded & accepted) {
            if (commit_candidate) {
              candidate.commit();
            }
            closest.select(hit->committed_ray_t, object, primitive,
                           geometry.primitive_kind);
            resolved->inst = hit->inst;
            resolved->prim = hit->prim;
            resolved->bary = hit->bary;
            resolved->hit_type =
                static_cast<std::uint32_t>(luisa::compute::HitType::Surface);
            resolved->committed_ray_t = hit->committed_ray_t;
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
                curve.geometry.primitive_kind);
            const auto within_cycles_interval =
                (intersection.distance >= ray->t_min()) &
                (intersection.distance <= ray->t_max());
            $if(!excluded & intersection.valid & within_cycles_interval &
                accepted) {
              if (commit_candidate) {
                candidate.commit(intersection.distance);
              }
              closest.select(intersection.distance, object, primitive,
                             curve.geometry.primitive_kind);
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
                       backend_hit->committed_ray_t));
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
