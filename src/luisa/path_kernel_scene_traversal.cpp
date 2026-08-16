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

[[nodiscard]] Bool
shadow_batch_contains(const Var<ShadowIntersectionBatchCall> &batch,
                      const Var<ShadowIntersectionCall> &candidate) noexcept {
  Bool duplicate = false;
  for (auto index = std::size_t{0u}; index < shadow_intersection_batch_capacity;
       ++index) {
    const auto &stored = batch->hits[static_cast<luisa::uint>(index)];
    duplicate |= (static_cast<std::uint32_t>(index) < batch->count) &
                 (stored->instance == candidate->instance) &
                 (stored->primitive == candidate->primitive) &
                 (stored->hit_type == candidate->hit_type);
  }
  return duplicate;
}

// Maintains the following reduction invariant after every accepted candidate:
// hits[0:count) is sorted by nondecreasing t and contains exactly the nearest
// min(total, capacity) candidates seen so far. This makes the result
// independent of the backend's candidate enumeration order.
void insert_shadow_batch_hit(
    Var<ShadowIntersectionBatchCall> &batch,
    const Var<ShadowIntersectionCall> &candidate) noexcept {
  UInt insertion = batch->count;
  for (auto index = std::size_t{0u}; index < shadow_intersection_batch_capacity;
       ++index) {
    const auto &stored = batch->hits[static_cast<luisa::uint>(index)];
    const auto first_larger =
        (static_cast<std::uint32_t>(index) < batch->count) &
        (candidate->distance < stored->distance) & (insertion == batch->count);
    insertion =
        select(insertion, static_cast<std::uint32_t>(index), first_larger);
  }

  $if(insertion <
      static_cast<std::uint32_t>(shadow_intersection_batch_capacity)) {
    for (auto index = shadow_intersection_batch_capacity - 1u; index != 0u;
         --index) {
      const auto move = (insertion < static_cast<std::uint32_t>(index)) &
                        (static_cast<std::uint32_t>(index) <= batch->count);
      $if(move) {
        batch->hits[static_cast<luisa::uint>(index)] =
            batch->hits[static_cast<luisa::uint>(index - 1u)];
      };
    }
    batch->hits[insertion] = candidate;
    batch->count =
        min(batch->count + 1u,
            static_cast<std::uint32_t>(shadow_intersection_batch_capacity));
  };
}

template <typename Candidate>
void reduce_shadow_candidate(Candidate &query_candidate,
                             Var<ShadowIntersectionBatchCall> &batch,
                             const Var<ShadowIntersectionCall> &intersection,
                             Expr<bool> may_be_transparent,
                             Expr<std::uint32_t> transparent_maximum) noexcept {
  const auto duplicate = shadow_batch_contains(batch, intersection);
  $if(!duplicate) {
    $if(!may_be_transparent) {
      batch->blocked = 1u;
      query_candidate.terminate();
    }
    $else {
      batch->total += 1u;
      $if(batch->total > transparent_maximum) {
        batch->blocked = 1u;
        query_candidate.terminate();
      }
      $else { insert_shadow_batch_hit(batch, intersection); };
    };
  };
}

class UnifiedSceneTraversalComponent final : public SceneTraversalComponent {

private:
  SceneTraversalStagePlan _plan;
  std::shared_ptr<const CurvePrimitiveComponent> _curves;
  std::shared_ptr<const CurveRibbonComponent> _ribbons;
  std::shared_ptr<const PrimitiveMaterialComponent> _materials;

  [[nodiscard]] Var<luisa::compute::CommittedHit>
  trace(const std::shared_ptr<LuisaSceneData> &scene,
        const Var<luisa::compute::Ray> &ray,
        Expr<std::uint32_t> visibility_mask,
        const ScenePrimitiveIdentity &source,
        const ScenePrimitiveIdentity &light) const noexcept {
    if (_plan.primitives.empty()) {
      Var<luisa::compute::CommittedHit> miss;
      miss->inst = surface_ray::invalid_primitive;
      miss->prim = surface_ray::invalid_primitive;
      miss->bary = make_float2(0.0f);
      miss->hit_type =
          static_cast<std::uint32_t>(luisa::compute::HitType::Miss);
      miss->committed_ray_t = ray->t_max();
      return miss;
    }

    const auto handle_surface =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto instance = scene->instance_buffer->read(hit->inst);
          const auto object =
              _materials->cycles_object_index(hit->inst, instance);
          const auto primitive = instance.cycles_primitive_offset + hit->prim;
          const auto excluded = source.matches(object, primitive) |
                                light.matches(object, primitive);
          $if(!excluded) { candidate.commit(); };
        };
    const auto handle_procedural =
        [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto curve =
              _curves->emit_metadata(scene, hit->inst, hit->prim);
          $if(curve.geometry.primitive_kind == geometry_kind_curve) {
            const auto object =
                _materials->cycles_object_index(hit->inst, curve.instance);
            const auto primitive = curve.segment.cycles_curve_index;
            const auto excluded = source.matches(object, primitive) |
                                  light.matches(object, primitive);
            $if(!excluded) {
              // The backend already owns the candidate instance transform.
              // Consume its native object-space traversal ray directly; its
              // unnormalized direction preserves the world-ray parameter t.
              const auto object_ray = candidate.object_ray();
              const auto control_points =
                  _curves->emit_control_points(scene, curve);
              const auto intersection =
                  _ribbons->intersect(object_ray, control_points,
                                      curve.geometry.curve_subdivision_level);
              $if(intersection.valid) {
                candidate.commit(intersection.distance);
              };
            };
          };
        };

    if (_plan.primitives.mixed()) {
      return scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
          .on_surface_candidate(handle_surface)
          .on_procedural_candidate(handle_procedural)
          .trace();
    }
    if (_plan.primitives.triangles) {
      return scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
          .on_surface_candidate(handle_surface)
          .trace();
    }
    return scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
        .on_procedural_candidate(handle_procedural)
        .trace();
  }

  [[nodiscard]] Var<ShadowIntersectionBatchCall> collect_shadow_batch(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum) const noexcept {
    Var<ShadowIntersectionBatchCall> batch;
    // A DSL Var has storage but no implicit value initialization. Both the
    // duplicate check and sorted insertion evaluate fixed-capacity lanes
    // eagerly before masking them with `count`, so every lane must hold a
    // defined value even while logically inactive.
    batch->count = 0u;
    batch->total = 0u;
    batch->blocked = 0u;
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      auto &hit = batch->hits[static_cast<luisa::uint>(index)];
      hit->instance = surface_ray::invalid_primitive;
      hit->primitive = surface_ray::invalid_primitive;
      hit->hit_type = static_cast<std::uint32_t>(luisa::compute::HitType::Miss);
      hit->distance = ray->t_max();
      hit->barycentric = make_float2(0.0f);
    }
    if (_plan.primitives.empty()) {
      return batch;
    }

    const auto handle_surface =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto instance = scene->instance_buffer->read(hit->inst);
          const auto geometry =
              scene->geometry_buffer->read(instance.geometry_index);
          const auto object =
              _materials->cycles_object_index(hit->inst, instance);
          const auto primitive = geometry.cycles_primitive_offset + hit->prim;
          const auto excluded = source.matches(object, primitive) |
                                light.matches(object, primitive);
          $if(!excluded) {
            const auto material_slot =
                _materials->triangle_material_slot(scene, geometry, hit->prim);
            const auto binding = _materials->resolve_binding(
                scene, instance, geometry, material_slot);
            Var<ShadowIntersectionCall> intersection;
            intersection->instance = hit->inst;
            intersection->primitive = hit->prim;
            intersection->hit_type =
                static_cast<std::uint32_t>(luisa::compute::HitType::Surface);
            intersection->distance = hit->committed_ray_t;
            intersection->barycentric = hit->bary;
            reduce_shadow_candidate(
                candidate, batch, intersection,
                (binding.flags & material_flag_may_be_transparent) != 0u,
                transparent_maximum);
          };
        };
    const auto handle_procedural = [&](luisa::compute::ProceduralCandidate
                                           &candidate) noexcept {
      const auto hit = candidate.hit();
      const auto curve = _curves->emit_metadata(scene, hit->inst, hit->prim);
      $if(curve.geometry.primitive_kind == geometry_kind_curve) {
        const auto object =
            _materials->cycles_object_index(hit->inst, curve.instance);
        const auto primitive = curve.segment.cycles_curve_index;
        const auto excluded = source.matches(object, primitive) |
                              light.matches(object, primitive);
        $if(!excluded) {
          const auto object_ray = candidate.object_ray();
          const auto control_points =
              _curves->emit_control_points(scene, curve);
          const auto exact =
              _ribbons->intersect(object_ray, control_points,
                                  curve.geometry.curve_subdivision_level);
          $if(exact.valid) {
            const auto material_slot = _materials->curve_material_slot(
                scene, curve.geometry, curve.segment);
            const auto binding = _materials->resolve_binding(
                scene, curve.instance, curve.geometry, material_slot);
            Var<ShadowIntersectionCall> intersection;
            intersection->instance = hit->inst;
            intersection->primitive = hit->prim;
            intersection->hit_type =
                static_cast<std::uint32_t>(luisa::compute::HitType::Procedural);
            intersection->distance = exact.distance;
            intersection->barycentric = make_float2(exact.u, exact.v);
            reduce_shadow_candidate(
                candidate, batch, intersection,
                (binding.flags & material_flag_may_be_transparent) != 0u,
                transparent_maximum);
          };
        };
      };
    };

    if (_plan.primitives.mixed()) {
      const auto ignored =
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_surface)
              .on_procedural_candidate(handle_procedural)
              .trace();
      static_cast<void>(ignored);
    } else if (_plan.primitives.triangles) {
      const auto ignored =
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_surface)
              .trace();
      static_cast<void>(ignored);
    } else {
      const auto ignored =
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_procedural_candidate(handle_procedural)
              .trace();
      static_cast<void>(ignored);
    }
    return batch;
  }

public:
  explicit UnifiedSceneTraversalComponent(SceneTraversalStagePlan plan)
      : _plan{plan},
        _curves{plan.primitives.curves ? make_curve_primitive_component()
                                       : nullptr},
        _ribbons{plan.primitives.curves ? make_curve_ribbon_component()
                                        : nullptr},
        _materials{plan.primitives.empty()
                       ? nullptr
                       : make_primitive_material_component()} {}

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
  Var<ShadowIntersectionBatchCall> collect_shadow(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum) const noexcept override {
    return collect_shadow_batch(scene, ray, visibility_mask, source, light,
                                transparent_maximum);
  }
};

} // namespace

std::shared_ptr<const SceneTraversalComponent>
make_scene_traversal_component(SceneTraversalStagePlan plan) {
  return std::make_shared<UnifiedSceneTraversalComponent>(plan);
}

} // namespace psycles::luisa_backend::detail
