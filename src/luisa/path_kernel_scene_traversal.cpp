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

struct SourceAccelerationIdentity {
  Bool valid;
  UInt instance;
  UInt primitive;
};

struct CompletedTriangleHit {
  Bool valid;
  UInt instance;
  UInt coincident_next;
  UInt primitive;
  UInt object;
  UInt cycles_primitive;
  UInt type_order;
  Float distance;
  Float2 barycentric;
};

struct PrimitiveCompletionRange {
  Bool valid;
  UInt instance_offset;
  UInt instance_count;
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
  SceneTraversalStagePlan _plan;
  std::shared_ptr<const CurvePrimitiveComponent> _curves;
  std::shared_ptr<const CurveRibbonComponent> _ribbons;
  std::shared_ptr<const PrimitiveMaterialComponent> _materials;
  std::shared_ptr<const CyclesTriangleIntersectionComponent>
      _triangle_intersection;

  [[nodiscard]] SourceAccelerationIdentity
  source_acceleration_identity(
      const std::shared_ptr<LuisaSceneData> &scene,
      const ScenePrimitiveIdentity &source) const noexcept {
    Bool found = false;
    UInt instance_index = 0u;
    UInt primitive_index = 0u;
    if (scene->cycles_completion_source_dense_count != 0u) {
      $if((source.object != surface_ray::invalid_primitive) &
          (source.object < scene->cycles_completion_source_dense_count)) {
        const auto candidate =
            scene->cycles_completion_source_dense_buffer->read(source.object);
        $if(candidate != surface_ray::invalid_primitive) {
          found = true;
          instance_index = candidate;
        };
      };
    } else if (scene->cycles_completion_source_sparse_count != 0u) {
      $if(source.object != surface_ray::invalid_primitive) {
        UInt first = 0u;
        UInt last = scene->cycles_completion_source_sparse_count;
        $while(first < last) {
          const auto middle = first + (last - first) / 2u;
          const auto entry =
              scene->cycles_completion_source_sparse_buffer->read(middle);
          $if(entry.x < source.object) {
            first = middle + 1u;
          }
          $else {
            last = middle;
          };
        };
        $if(first < scene->cycles_completion_source_sparse_count) {
          const auto entry =
              scene->cycles_completion_source_sparse_buffer->read(first);
          $if(entry.x == source.object) {
            found = true;
            instance_index = entry.y;
          };
        };
      };
    }
    Bool valid = false;
    $if(found) {
      const auto instance = scene->instance_buffer->read(instance_index);
      const auto geometry =
          scene->geometry_buffer->read(instance.geometry_index);
      valid = (geometry.primitive_kind == geometry_kind_triangle) &
              (source.primitive >= geometry.cycles_primitive_offset);
      primitive_index = source.primitive - geometry.cycles_primitive_offset;
    };
    return {.valid = std::move(valid),
            .instance = std::move(instance_index),
            .primitive = std::move(primitive_index)};
  }

  [[nodiscard]] PrimitiveCompletionRange
  primitive_completion_range(
      const std::shared_ptr<LuisaSceneData> &scene,
      Expr<std::uint32_t> instance_index,
      Expr<std::uint32_t> local_primitive) const noexcept {
    const auto instance = scene->instance_buffer->read(instance_index);
    Bool valid = false;
    UInt instance_offset = 0u;
    UInt instance_count = 0u;
    UInt first = instance.primitive_completion_offset;
    const UInt end = first + instance.primitive_completion_count;
    UInt last = end;
    $while(first < last) {
      const auto middle = first + (last - first) / 2u;
      const auto record = scene->primitive_completion_buffer->read(middle);
      $if(record.local_primitive < local_primitive) {
        first = middle + 1u;
      }
      $else {
        last = middle;
      };
    };
    $if(first < end) {
      const auto record = scene->primitive_completion_buffer->read(first);
      $if(record.local_primitive == local_primitive) {
        valid = record.instance_count > 1u;
        instance_offset = record.instance_offset;
        instance_count = record.instance_count;
      };
    };
    return {.valid = std::move(valid),
            .instance_offset = std::move(instance_offset),
            .instance_count = std::move(instance_count)};
  }

  [[nodiscard]] CompletedTriangleHit
  resolve_triangle_instance(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray,
      Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source,
      const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> instance_index,
      Expr<std::uint32_t> local_primitive) const noexcept {
    const auto instance =
        scene->instance_buffer->read(instance_index);
    const auto geometry =
        scene->geometry_buffer->read(instance.geometry_index);
    const auto triangle =
        scene->heap->buffer<Triangle>(geometry.bindless_base)
            .read(local_primitive);
    const auto positions =
        scene->heap->buffer<luisa::float3>(geometry.bindless_base + 9u);
    const auto intersection = _triangle_intersection->intersect(
        ray,
        instance.cycles_world_to_object,
        instance.cycles_transform_applied,
        positions.read(triangle.i0),
        positions.read(triangle.i1),
        positions.read(triangle.i2));
    const auto object =
        _materials->cycles_object_index(instance_index, instance);
    const auto primitive =
        geometry.cycles_primitive_offset + local_primitive;
    const auto excluded = source.matches(object, primitive) |
                          light.matches(object, primitive);
    const auto visible =
        (instance.visibility_mask & visibility_mask) != 0u;
    return {
        .valid = intersection.valid & visible & !excluded,
        .instance = UInt{instance_index},
        .coincident_next = UInt{instance.coincident_next},
        .primitive = UInt{local_primitive},
        .object = UInt{object},
        .cycles_primitive = UInt{primitive},
        .type_order = UInt{geometry.primitive_kind},
        .distance = Float{intersection.distance},
        .barycentric = Float2{intersection.barycentric}};
  }

  [[nodiscard]] CompletedTriangleHit
  resolve_completed_triangle(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray,
      Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source,
      const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> seed_instance,
      Expr<std::uint32_t> local_primitive) const noexcept {
    if (!_plan.triangle_completion) {
      return resolve_triangle_instance(
          scene, ray, visibility_mask, source, light,
          seed_instance, local_primitive);
    }
    CyclesClosestHitOrder group_closest;
    Bool group_has_hit = false;
    UInt group_instance = seed_instance;
    UInt group_object = 0u;
    UInt group_primitive = 0u;
    UInt group_type = geometry_kind_triangle;
    Float group_distance = ray->t_max();
    Float2 group_barycentric = make_float2(0.0f);
    const auto seed = scene->instance_buffer->read(seed_instance);
    const auto completion = primitive_completion_range(
        scene, seed_instance, local_primitive);
    UInt alias_index = seed_instance;
    UInt completion_index = completion.instance_offset;
    UInt remaining = select(
        max(seed.coincident_count, 1u),
        completion.instance_count,
        completion.valid);
    $while(remaining > 0u) {
      $if(completion.valid) {
        alias_index =
            scene->primitive_completion_instance_buffer->read(
                completion_index);
      };
      const auto candidate = resolve_triangle_instance(
          scene, ray, visibility_mask, source, light,
          alias_index, local_primitive);
      const auto accepted =
          candidate.valid &
          group_closest.accepts(
              candidate.distance,
              candidate.object,
              candidate.cycles_primitive,
              candidate.type_order);
      $if(accepted) {
        group_closest.select(
            candidate.distance,
            candidate.object,
            candidate.cycles_primitive,
            candidate.type_order);
        group_has_hit = true;
        group_instance = candidate.instance;
        group_object = candidate.object;
        group_primitive = candidate.cycles_primitive;
        group_type = candidate.type_order;
        group_distance = candidate.distance;
        group_barycentric = candidate.barycentric;
      };
      $if(completion.valid) {
        completion_index += 1u;
      }
      $else {
        alias_index = candidate.coincident_next;
      };
      remaining -= 1u;
    };
    return {.valid = std::move(group_has_hit),
            .instance = std::move(group_instance),
            .coincident_next = 0u,
            .primitive = UInt{local_primitive},
            .object = std::move(group_object),
            .cycles_primitive = std::move(group_primitive),
            .type_order = std::move(group_type),
            .distance = std::move(group_distance),
            .barycentric = std::move(group_barycentric)};
  }

  [[nodiscard]] TriangleResolverCallable
  make_triangle_resolver(
      const std::shared_ptr<LuisaSceneData> &scene) const noexcept {
    TriangleResolverCallable resolver =
        [this, scene](Var<luisa::compute::Ray> ray,
                      Var<TriangleResolutionQueryCall> query) noexcept {
          const auto resolved = resolve_completed_triangle(
              scene,
              ray,
              query.visibility_mask,
              {.object = query.source_object,
               .primitive = query.source_primitive},
              {.object = query.light_object,
               .primitive = query.light_primitive},
              query.seed_instance,
              query.local_primitive);
          Var<TriangleResolutionCall> result;
          result.barycentric = resolved.barycentric;
          result.distance = resolved.distance;
          result.instance = resolved.instance;
          result.primitive = resolved.primitive;
          result.object = resolved.object;
          result.cycles_primitive = resolved.cycles_primitive;
          result.type_order = resolved.type_order;
          result.valid = select(0u, 1u, resolved.valid);
          return result;
        };
    resolver.set_name(
        _plan.triangle_completion
            ? "cycles_triangle_resolver_completion"
            : "cycles_triangle_resolver_singleton");
    return resolver;
  }

  [[nodiscard]] static CompletedTriangleHit
  invoke_triangle_resolver(
      const TriangleResolverCallable &resolver,
      const Var<luisa::compute::Ray> &ray,
      Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source,
      const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> seed_instance,
      Expr<std::uint32_t> local_primitive) noexcept {
    Var<TriangleResolutionQueryCall> query;
    query.visibility_mask = visibility_mask;
    query.source_object = source.object;
    query.source_primitive = source.primitive;
    query.light_object = light.object;
    query.light_primitive = light.primitive;
    query.seed_instance = seed_instance;
    query.local_primitive = local_primitive;
    const auto resolved = resolver(ray, query);
    return {
        .valid = resolved.valid != 0u,
        .instance = resolved.instance,
        .coincident_next = 0u,
        .primitive = resolved.primitive,
        .object = resolved.object,
        .cycles_primitive = resolved.cycles_primitive,
        .type_order = resolved.type_order,
        .distance = resolved.distance,
        .barycentric = resolved.barycentric};
  }

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
    if (_plan.primitives.empty()) {
      return resolved;
    }
    const auto triangle_resolver =
        _plan.primitives.triangles
            ? std::make_unique<TriangleResolverCallable>(
                  make_triangle_resolver(scene))
            : nullptr;

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
          // Hardware triangle queries are broad-phase candidate sources. The
          // Resolver handles a singleton, a whole-instance class, or a sparse
          // primitive completion list through the same Cycles predicate and
          // stable identity order.
          const auto group = invoke_triangle_resolver(
              *triangle_resolver, ray, visibility_mask, source, light,
              hit->inst, hit->prim);
          const auto accepted =
              group.valid &
              closest.accepts(group.distance, group.object,
                              group.cycles_primitive, group.type_order);
          $if(accepted) {
            if (commit_candidate) {
              candidate.commit();
            }
            closest.select(group.distance, group.object,
                           group.cycles_primitive, group.type_order);
            resolved->inst = group.instance;
            resolved->prim = group.primitive;
            resolved->bary = group.barycentric;
            resolved->hit_type = static_cast<std::uint32_t>(
                luisa::compute::HitType::Surface);
            resolved->committed_ray_t = group.distance;
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

    // Explicit self exclusion and geometric origin offset are independent
    // parts of the Cycles ray contract. If Cycles' exact source-triangle test
    // certifies the origin, it deliberately leaves the ray at the surface and
    // rejects only the source identity. Distinct instances with bit-identical
    // support, or corresponding primitives with overlapping closed world
    // bounds, therefore remain eligible at the closed t == 0 endpoint for
    // continuation as well as shadow rays. Hardware traversal is not required
    // to report that endpoint consistently, so complete its broad-phase
    // candidates from the source's finite completion relation. This is not a
    // topological forced hit: resolve_completed_triangle still applies the
    // actual ray, visibility, exclusions, and Cycles Pluecker predicate. A
    // geometrically offset origin therefore rejects candidates normally.
    if (_plan.triangle_completion) {
      const auto source_acceleration =
          source_acceleration_identity(scene, source);
      $if(source_acceleration.valid) {
        const auto source_instance =
            scene->instance_buffer->read(source_acceleration.instance);
        const auto completion = primitive_completion_range(
            scene, source_acceleration.instance,
            source_acceleration.primitive);
        $if((max(source_instance.coincident_count, 1u) > 1u) |
            completion.valid) {
          const auto group = invoke_triangle_resolver(
              *triangle_resolver, ray, visibility_mask, source, light,
              source_acceleration.instance,
              source_acceleration.primitive);
          const auto accepted =
              group.valid &
              closest.accepts(group.distance, group.object,
                              group.cycles_primitive, group.type_order);
          $if(accepted) {
            closest.select(group.distance, group.object,
                           group.cycles_primitive, group.type_order);
            resolved->inst = group.instance;
            resolved->prim = group.primitive;
            resolved->bary = group.barycentric;
            resolved->hit_type = static_cast<std::uint32_t>(
                luisa::compute::HitType::Surface);
            resolved->committed_ray_t = group.distance;
          };
        };
      };
    }

    const auto trace_backend =
        [&](const Var<luisa::compute::Ray> &candidate_ray,
            bool commit_candidate) noexcept {
          if (_plan.primitives.mixed()) {
            return scene->accel
                ->traverse(candidate_ray,
                           {.visibility_mask = visibility_mask})
                .on_surface_candidate(
                    [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
                      handle_surface(candidate, commit_candidate);
                    })
                .on_procedural_candidate(
                    [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
                      handle_procedural(candidate, commit_candidate);
                    })
                .trace();
          }
          if (_plan.primitives.triangles) {
            return scene->accel
                ->traverse(candidate_ray,
                           {.visibility_mask = visibility_mask})
                .on_surface_candidate(
                    [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
                      handle_surface(candidate, commit_candidate);
                    })
                .trace();
          }
          return scene->accel
              ->traverse(candidate_ray,
                         {.visibility_mask = visibility_mask})
              .on_procedural_candidate(
                  [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
                    handle_procedural(candidate, commit_candidate);
                  })
              .trace();
        };

    const auto backend_hit =
        trace_backend(query_ray, true);

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
          trace_backend(tie_ray, false);
      static_cast<void>(ignored);
    };
    return resolved;
  }

public:
  explicit UnifiedSceneTraversalComponent(
      SceneTraversalStagePlan plan)
      : _plan{plan},
        _curves{plan.primitives.curves
                    ? make_curve_primitive_component()
                    : nullptr},
        _ribbons{plan.primitives.curves
                     ? make_curve_ribbon_component()
                     : nullptr},
        _materials{plan.primitives.empty()
                       ? nullptr
                       : make_primitive_material_component()},
        _triangle_intersection{
            plan.primitives.triangles
                ? make_cycles_triangle_intersection_component()
                : nullptr} {}

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
make_scene_traversal_component(
    SceneTraversalStagePlan plan) {
  return std::make_shared<UnifiedSceneTraversalComponent>(
      plan.canonicalized());
}

} // namespace psycles::luisa_backend::detail
