#include "path_kernel_scene_traversal.h"

#include "curve_ribbon_component.h"
#include "path_kernel_shadow_storage.h"

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

void initialize_shadow_intersection(Var<ShadowIntersectionCall> &hit,
                                    Expr<float> miss_distance) noexcept {
  hit->instance = surface_ray::invalid_primitive;
  hit->primitive = surface_ray::invalid_primitive;
  hit->hit_type = static_cast<std::uint32_t>(luisa::compute::HitType::Miss);
  hit->distance = miss_distance;
  hit->barycentric = make_float2(0.0f);
}

// Host-side virtual dispatch selects the storage policy while recording the
// shader AST. No virtual call or weakly typed payload survives on the device.
class ShadowHitStorage {

public:
  virtual ~ShadowHitStorage() noexcept = default;
  [[nodiscard]] virtual Var<ShadowIntersectionCall>
  read(Expr<std::uint32_t> index) const noexcept = 0;
  virtual void write(Expr<std::uint32_t> index,
                     const Var<ShadowIntersectionCall> &hit) noexcept = 0;
  virtual void materialize(Var<ShadowIntersectionBatchCall> &batch,
                           Expr<std::uint32_t> count,
                           Expr<float> miss_distance) const noexcept = 0;
};

class LocalShadowHitStorage final : public ShadowHitStorage {

private:
  Var<ShadowIntersectionBatchCall> _batch;

public:
  explicit LocalShadowHitStorage(Expr<float> miss_distance) noexcept {
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      initialize_shadow_intersection(
          _batch->hits[static_cast<luisa::uint>(index)], miss_distance);
    }
  }

  [[nodiscard]] Var<ShadowIntersectionCall>
  read(Expr<std::uint32_t> index) const noexcept override {
    return _batch->hits[index];
  }

  void write(Expr<std::uint32_t> index,
             const Var<ShadowIntersectionCall> &hit) noexcept override {
    _batch->hits[index] = hit;
  }

  void materialize(Var<ShadowIntersectionBatchCall> &batch,
                   Expr<std::uint32_t>,
                   Expr<float>) const noexcept override {
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      batch->hits[static_cast<luisa::uint>(index)] =
          _batch->hits[static_cast<luisa::uint>(index)];
    }
  }
};

class SoAShadowHitStorage final : public ShadowHitStorage {

private:
  const ShadowIntersectionBatchStorage *_storage;
  UInt _invocation;
  UInt _runtime_capacity;

public:
  SoAShadowHitStorage(const ShadowIntersectionBatchStorage *storage,
                      Expr<std::uint32_t> invocation,
                      Expr<std::uint32_t> runtime_capacity) noexcept
      : _storage{storage},
        _invocation{invocation},
        _runtime_capacity{runtime_capacity} {}

  [[nodiscard]] Var<ShadowIntersectionCall>
  read(Expr<std::uint32_t> index) const noexcept override {
    return _storage->read(_invocation, index, _runtime_capacity);
  }

  void write(Expr<std::uint32_t> index,
             const Var<ShadowIntersectionCall> &hit) noexcept override {
    _storage->write(_invocation, index, hit, _runtime_capacity);
  }

  void materialize(Var<ShadowIntersectionBatchCall> &batch,
                   Expr<std::uint32_t> count,
                   Expr<float> miss_distance) const noexcept override {
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      auto &hit = batch->hits[static_cast<luisa::uint>(index)];
      initialize_shadow_intersection(hit, miss_distance);
      $if(static_cast<std::uint32_t>(index) < count) {
        hit = read(static_cast<std::uint32_t>(index));
      };
    }
  }
};

// A reduction over a candidate prefix is modeled by
// (R, n, m, i): R is the retained set, n the number of accepted candidates,
// m=max(t in R), and i=argmax(t in R). The inductive update either appends
// while |R|<k, rejects t>=m, or replaces R[i] and recomputes (m,i). Thus R is
// exactly the nearest min(n,k) candidates after every prefix, independently
// of candidate enumeration order (apart from unobservable equal-t ties).
class ShadowBatchReducer {

private:
  ShadowHitStorage &_hits;
  UInt _count{0u};
  UInt _total{0u};
  UInt _blocked{0u};
  UInt _replacement_index{0u};
  Float _max_record_distance;

private:
  void update_farthest() noexcept {
    _replacement_index = 0u;
    _max_record_distance = _hits.read(0u)->distance;
    for (auto index = std::size_t{1u};
         index < shadow_intersection_batch_capacity; ++index) {
      const auto distance =
          _hits.read(static_cast<std::uint32_t>(index))->distance;
      const auto farther = distance > _max_record_distance;
      _replacement_index = select(
          _replacement_index, static_cast<std::uint32_t>(index), farther);
      _max_record_distance =
          select(_max_record_distance, distance, farther);
    }
  }

  [[nodiscard]] Bool
  contains(const Var<ShadowIntersectionCall> &candidate) const noexcept {
    Bool duplicate = false;
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      $if(static_cast<std::uint32_t>(index) < _count) {
        const auto stored = _hits.read(static_cast<std::uint32_t>(index));
        duplicate |= (stored->instance == candidate->instance) &
                     (stored->primitive == candidate->primitive) &
                     (stored->hit_type == candidate->hit_type);
      };
    }
    return duplicate;
  }

  void insert(const Var<ShadowIntersectionCall> &candidate) noexcept {
    constexpr auto capacity =
        static_cast<std::uint32_t>(shadow_intersection_batch_capacity);
    $if(_count < capacity) {
      _hits.write(_count, candidate);
      _count += 1u;
      $if(_count == capacity) { update_farthest(); }
      $else { _replacement_index = _count; };
    }
    $else {
      // Strict comparison retains the first-seen representative for equal-t
      // ties, matching Cycles' bounded replacement rule.
      $if(candidate->distance < _max_record_distance) {
        _hits.write(_replacement_index, candidate);
        update_farthest();
      };
    };
  }

public:
  ShadowBatchReducer(ShadowHitStorage &hits,
                     Expr<float> miss_distance) noexcept
      : _hits{hits}, _max_record_distance{miss_distance} {}

  template <typename Candidate>
  void reduce(Candidate &query_candidate,
              const Var<ShadowIntersectionCall> &intersection,
              Expr<bool> has_transparent_shadow,
              Expr<std::uint32_t> transparent_maximum) noexcept {
    // Stored candidates in _hits[0.._count) are exclusively transparent-shadow
    // primitives, so an opaque candidate can never be a duplicate of any
    // retained hit. Terminate immediately without probing storage.
    $if(!has_transparent_shadow) {
      _blocked = 1u;
      query_candidate.terminate();
    }
    $else {
      const auto duplicate = contains(intersection);
      $if(!duplicate) {
        _total += 1u;
        $if(_total > transparent_maximum) {
          _blocked = 1u;
          query_candidate.terminate();
        }
        $else { insert(intersection); };
      };
    };
  }

  [[nodiscard]] Var<ShadowIntersectionBatchCall>
  materialize(Expr<float> miss_distance) const noexcept {
    Var<ShadowIntersectionBatchCall> batch;
    _hits.materialize(batch, _count, miss_distance);
    batch->count = _count;
    batch->total = _total;
    batch->blocked = _blocked;
    return batch;
  }

  [[nodiscard]] Var<ShadowIntersectionSummaryCall> summary() const noexcept {
    Var<ShadowIntersectionSummaryCall> result;
    result->count = _count;
    result->total = _total;
    result->blocked = _blocked;
    return result;
  }
};

struct TraversalPrimitiveMetadata {
  UInt instance_id;
  Var<SceneTraversalInstanceGpu> instance;
  UInt primitive_kind;
  UInt curve_subdivision_level;
};

struct TraversalCurveMetadata {
  TraversalPrimitiveMetadata primitive;
  Var<CurveSegmentGpu> segment;
};

struct ResolvedTriangleIdentity {
  UInt inst;
  UInt prim;
};

[[nodiscard]] ResolvedTriangleIdentity resolve_excluded_triangle_identity(
    const std::shared_ptr<LuisaSceneData> &scene,
    const ScenePrimitiveIdentity &identity) noexcept {
  UInt inst = surface_ray::invalid_primitive;
  UInt prim = surface_ray::invalid_primitive;
  $if((identity.primitive != surface_ray::invalid_primitive) &
      (identity.object < scene->traversal_object_instance_count)) {
    const auto entry =
        (*scene->traversal_object_instance_buffer)->read(identity.object);
    const auto valid = (entry.x != surface_ray::invalid_primitive) &
                       (identity.primitive >= entry.y);
    inst = select(surface_ray::invalid_primitive, entry.x, valid);
    prim = select(surface_ray::invalid_primitive, identity.primitive - entry.y,
                  valid);
  };
  return {.inst = std::move(inst), .prim = std::move(prim)};
}

class UnifiedSceneTraversalComponent final : public SceneTraversalComponent {

private:
  SceneTraversalStagePlan _plan;
  std::shared_ptr<const CurveRibbonComponent> _ribbons;

  [[nodiscard]] TraversalPrimitiveMetadata emit_primitive_metadata(
      const std::shared_ptr<LuisaSceneData> &scene,
      Expr<std::uint32_t> instance_id) const noexcept {
    UInt resolved_instance_id = instance_id;
    auto instance =
        scene->traversal_instance_buffer->read(resolved_instance_id);
    const auto packed = instance.primitive_kind_and_curve_subdivision;
    return {
        .instance_id = std::move(resolved_instance_id),
        .instance = std::move(instance),
        .primitive_kind = packed & scene_traversal_primitive_kind_mask,
        .curve_subdivision_level =
            packed >> scene_traversal_curve_subdivision_shift};
  }

  [[nodiscard]] TraversalCurveMetadata emit_curve_metadata(
      const std::shared_ptr<LuisaSceneData> &scene,
      Expr<std::uint32_t> instance_id,
      Expr<std::uint32_t> segment_id) const noexcept {
    auto primitive = emit_primitive_metadata(scene, instance_id);
    const auto segments = scene->heap->buffer<CurveSegmentGpu>(
        primitive.instance.bindless_base);
    return {.primitive = std::move(primitive),
            .segment = segments.read(segment_id)};
  }

  [[nodiscard]] CurveControlPoints emit_curve_control_points(
      const std::shared_ptr<LuisaSceneData> &scene,
      const TraversalCurveMetadata &curve) const noexcept {
    const auto keys = scene->heap->buffer<luisa::float4>(
        curve.primitive.instance.bindless_base + 1u);
    return {.before = keys.read(curve.segment.key_before),
            .begin = keys.read(curve.segment.key_begin),
            .end = keys.read(curve.segment.key_end),
            .after = keys.read(curve.segment.key_after)};
  }

  [[nodiscard]] UInt resolve_material_flags(
      const std::shared_ptr<LuisaSceneData> &scene,
      const TraversalPrimitiveMetadata &primitive,
      Expr<std::uint32_t> material_slot) const noexcept {
    UInt resolved_material_slot = material_slot;
    const auto &instance = primitive.instance;
    UInt flags = scene->traversal_material_flags_buffer->read(
        instance.geometry_material_offset +
        min(resolved_material_slot, instance.geometry_material_count - 1u));
    $if(resolved_material_slot < instance.override_material_count) {
      flags = scene->traversal_material_flags_buffer->read(
          instance.override_material_offset + resolved_material_slot);
    };
    return flags;
  }

  [[nodiscard]] Var<luisa::compute::CommittedHit>
  trace(const std::shared_ptr<LuisaSceneData> &scene,
        const Var<luisa::compute::Ray> &ray,
        Expr<std::uint32_t> visibility_mask,
        const ScenePrimitiveIdentity &source,
        const ScenePrimitiveIdentity &light,
        bool has_light_exclusion = true) const noexcept {
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

    const auto finish = [&](auto &&query) {
      auto hit = query.trace();
      // CommittedHit identifies a miss through hit_type; its other payload
      // fields are not a portable miss-distance contract (native SPIR-V
      // initializes them to zero). Cycles keeps the original traversal bound
      // when no primitive commits. Establish that scene-traversal invariant
      // here for both closest and closest-shadow consumers.
      hit->committed_ray_t =
          select(hit->committed_ray_t, ray->t_max(), hit->miss());
      return hit;
    };

    const auto handle_procedural =
        [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto curve = emit_curve_metadata(scene, hit->inst, hit->prim);
          $if(curve.primitive.primitive_kind == geometry_kind_curve) {
            const auto object =
                curve.primitive.instance.cycles_object_index;
            const auto primitive = curve.segment.cycles_curve_index;
            const auto excluded =
                has_light_exclusion
                    ? (source.matches(object, primitive) |
                       light.matches(object, primitive))
                    : source.matches(object, primitive);
            $if(!excluded) {
              // The backend already owns the candidate instance transform.
              // Consume its native object-space traversal ray directly; its
              // unnormalized direction preserves the world-ray parameter t.
              const auto object_ray = candidate.object_ray();
              const auto control_points = emit_curve_control_points(scene, curve);
              const auto intersection =
                  _ribbons->intersect(object_ray, control_points,
                                      curve.primitive.curve_subdivision_level);
              $if(intersection.valid) {
                candidate.commit(intersection.distance);
              };
            };
          };
        };

    if (_plan.primitives.triangles &&
        scene->traversal_object_instance_buffer.has_value()) {
      if (!has_light_exclusion) {
        const auto src = resolve_excluded_triangle_identity(scene, source);
        const auto handle_excluded_src =
            [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
              const auto hit = candidate.hit();
              $if(!((hit->inst == src.inst) & (hit->prim == src.prim))) {
                candidate.commit();
              };
            };
        if (_plan.primitives.curves) {
          return finish(
              scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
                  .on_surface_candidate(handle_excluded_src)
                  .on_procedural_candidate(handle_procedural));
        }
        return finish(
            scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_excluded_src));
      }
      const auto src = resolve_excluded_triangle_identity(scene, source);
      const auto dst = resolve_excluded_triangle_identity(scene, light);
      const auto handle_excluded_both =
          [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
            const auto hit = candidate.hit();
            const auto excluded =
                ((hit->inst == src.inst) & (hit->prim == src.prim)) |
                ((hit->inst == dst.inst) & (hit->prim == dst.prim));
            $if(!excluded) { candidate.commit(); };
          };
      if (_plan.primitives.curves) {
        return finish(
            scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_excluded_both)
                .on_procedural_candidate(handle_procedural));
      }
      return finish(
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_excluded_both));
    }

    const Bool has_any_exclusion =
        has_light_exclusion
            ? ((source.primitive != surface_ray::invalid_primitive) |
               (light.primitive != surface_ray::invalid_primitive))
            : (source.primitive != surface_ray::invalid_primitive);
    const auto handle_surface =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          $if(!has_any_exclusion) {
            candidate.commit();
          }
          $else {
            const auto hit = candidate.hit();
            const auto metadata = emit_primitive_metadata(scene, hit->inst);
            const auto object = metadata.instance.cycles_object_index;
            const auto primitive =
                metadata.instance.cycles_primitive_offset + hit->prim;
            const auto excluded =
                has_light_exclusion
                    ? (source.matches(object, primitive) |
                       light.matches(object, primitive))
                    : source.matches(object, primitive);
            $if(!excluded) { candidate.commit(); };
          };
        };

    if (_plan.primitives.mixed()) {
      return finish(
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_surface)
              .on_procedural_candidate(handle_procedural));
    }
    if (_plan.primitives.triangles) {
      return finish(
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_surface));
    }
    return finish(
        scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
            .on_procedural_candidate(handle_procedural));
  }

  [[nodiscard]] Var<ShadowIntersectionSummaryCall> reduce_shadow_candidates(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum,
      ShadowHitStorage &hit_storage) const noexcept {
    ShadowBatchReducer reducer{hit_storage, ray->t_max()};
    if (_plan.primitives.empty()) {
      return reducer.summary();
    }

    if (can_use_fast_triangle_shadow(scene)) {
      const auto src = resolve_excluded_triangle_identity(scene, source);
      const auto dst = resolve_excluded_triangle_identity(scene, light);
      if (!scene->has_transparent_shadow_materials) {
        const auto handle_opaque_shadow =
            [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
              const auto hit = candidate.hit();
              const auto excluded =
                  ((hit->inst == src.inst) & (hit->prim == src.prim)) |
                  ((hit->inst == dst.inst) & (hit->prim == dst.prim));
              $if(!excluded) {
                candidate.commit();
                candidate.terminate();
              };
            };
        const auto any_hit =
            scene->accel->traverse_any(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_opaque_shadow)
                .trace();
        Var<ShadowIntersectionSummaryCall> result;
        result->count = 0u;
        result->total = 0u;
        result->blocked = select(0u, 1u, !any_hit->miss());
        return result;
      }

      const auto opacity_words =
          Expr<Buffer<luisa::uint>>{*scene->traversal_shadow_opacity_buffer};
      const auto handle_fast_surface =
          [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
            const auto hit = candidate.hit();
            const auto excluded =
                ((hit->inst == src.inst) & (hit->prim == src.prim)) |
                ((hit->inst == dst.inst) & (hit->prim == dst.prim));
            $if(!excluded) {
              const auto code = opacity_words.read(hit->inst);
              Bool has_transparent = code != 0u;
              if (scene->has_mixed_shadow_opacity_instances) {
                has_transparent = code == ~0u;
                $if((code != 0u) & (code != ~0u)) {
                  const auto word =
                      opacity_words.read(code + (hit->prim >> 5u));
                  has_transparent = ((word >> (hit->prim & 31u)) & 1u) != 0u;
                };
              }
              Var<ShadowIntersectionCall> intersection;
              intersection->instance = hit->inst;
              intersection->primitive = hit->prim;
              intersection->hit_type =
                  static_cast<std::uint32_t>(luisa::compute::HitType::Surface);
              intersection->distance = hit->committed_ray_t;
              intersection->barycentric = hit->bary;
              reducer.reduce(candidate, intersection, has_transparent,
                             transparent_maximum);
            };
          };
      const auto ignored =
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_fast_surface)
              .trace();
      static_cast<void>(ignored);
      return reducer.summary();
    }

    const bool fast_surface_identity =
        _plan.primitives.triangles &&
        scene->traversal_object_instance_buffer.has_value();
    std::optional<ResolvedTriangleIdentity> fast_src;
    std::optional<ResolvedTriangleIdentity> fast_dst;
    if (fast_surface_identity) {
      fast_src.emplace(resolve_excluded_triangle_identity(scene, source));
      fast_dst.emplace(resolve_excluded_triangle_identity(scene, light));
    }

    const auto handle_surface =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          Bool excluded = false;
          std::optional<TraversalPrimitiveMetadata> fallback_metadata;
          if (fast_surface_identity) {
            excluded =
                ((hit->inst == fast_src->inst) & (hit->prim == fast_src->prim)) |
                ((hit->inst == fast_dst->inst) & (hit->prim == fast_dst->prim));
          } else {
            fallback_metadata.emplace(emit_primitive_metadata(scene, hit->inst));
            const auto object = fallback_metadata->instance.cycles_object_index;
            const auto primitive =
                fallback_metadata->instance.cycles_primitive_offset + hit->prim;
            excluded = source.matches(object, primitive) |
                       light.matches(object, primitive);
          }
          $if(!excluded) {
            Bool has_transparent = false;
            if (scene->traversal_shadow_opacity_buffer.has_value()) {
              if (scene->has_transparent_shadow_materials) {
                const auto opacity_words = Expr<Buffer<luisa::uint>>{
                    *scene->traversal_shadow_opacity_buffer};
                const auto code = opacity_words.read(hit->inst);
                has_transparent = code != 0u;
                if (scene->has_mixed_shadow_opacity_instances) {
                  has_transparent = code == ~0u;
                  $if((code != 0u) & (code != ~0u)) {
                    const auto word =
                        opacity_words.read(code + (hit->prim >> 5u));
                    has_transparent = ((word >> (hit->prim & 31u)) & 1u) != 0u;
                  };
                }
              }
            } else {
              const auto metadata = fallback_metadata.has_value()
                                        ? *fallback_metadata
                                        : emit_primitive_metadata(scene, hit->inst);
              const auto material_slot = scene->heap
                                             ->buffer<luisa::uint>(
                                                 metadata.instance.bindless_base +
                                                 4u)
                                             .read(hit->prim);
              const auto material_flags =
                  resolve_material_flags(scene, metadata, material_slot);
              has_transparent =
                  (material_flags & material_flag_has_transparent_shadow) != 0u;
            }
            Var<ShadowIntersectionCall> intersection;
            intersection->instance = hit->inst;
            intersection->primitive = hit->prim;
            intersection->hit_type =
                static_cast<std::uint32_t>(luisa::compute::HitType::Surface);
            intersection->distance = hit->committed_ray_t;
            intersection->barycentric = hit->bary;
            reducer.reduce(candidate, intersection, has_transparent,
                           transparent_maximum);
          };
        };
    const auto handle_procedural = [&](luisa::compute::ProceduralCandidate
                                           &candidate) noexcept {
      const auto hit = candidate.hit();
      const auto curve = emit_curve_metadata(scene, hit->inst, hit->prim);
      $if(curve.primitive.primitive_kind == geometry_kind_curve) {
        const auto object = curve.primitive.instance.cycles_object_index;
        const auto primitive = curve.segment.cycles_curve_index;
        const auto excluded = source.matches(object, primitive) |
                              light.matches(object, primitive);
        $if(!excluded) {
          const auto object_ray = candidate.object_ray();
          const auto control_points = emit_curve_control_points(scene, curve);
          const auto exact =
              _ribbons->intersect(object_ray, control_points,
                                  curve.primitive.curve_subdivision_level);
          $if(exact.valid) {
            const auto material_slot = scene->heap
                                           ->buffer<luisa::uint>(
                                               curve.primitive.instance
                                                       .bindless_base +
                                               4u)
                                           .read(curve.segment.curve_index);
            const auto material_flags = resolve_material_flags(
                scene, curve.primitive, material_slot);
            Var<ShadowIntersectionCall> intersection;
            intersection->instance = hit->inst;
            intersection->primitive = hit->prim;
            intersection->hit_type =
                static_cast<std::uint32_t>(luisa::compute::HitType::Procedural);
            intersection->distance = exact.distance;
            intersection->barycentric = make_float2(exact.u, exact.v);
            reducer.reduce(
                candidate, intersection,
                (material_flags & material_flag_has_transparent_shadow) != 0u,
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
    return reducer.summary();
  }

  [[nodiscard]] static bool is_cpu_scene(
      const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene || !static_cast<bool>(scene->device)) {
      return false;
    }
    const auto name = scene->device.backend_name();
    return name == "fallback";
  }

  [[nodiscard]] static bool is_fallback_scene(
      const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    return scene && static_cast<bool>(scene->device) &&
           scene->device.backend_name() == "fallback";
  }

  [[nodiscard]] bool can_use_fast_triangle_shadow(
      const std::shared_ptr<LuisaSceneData> &scene) const noexcept {
    return _plan.primitives.triangles && !_plan.primitives.curves &&
           scene->traversal_object_instance_buffer.has_value() &&
           scene->traversal_shadow_opacity_buffer.has_value();
  }

  [[nodiscard]] bool can_use_fast_shadow_collect(
      const std::shared_ptr<LuisaSceneData> &scene) const noexcept {
    return _plan.primitives.triangles &&
           (!_plan.primitives.curves || is_cpu_scene(scene)) &&
           scene->traversal_object_instance_buffer.has_value() &&
           scene->traversal_shadow_opacity_buffer.has_value();
  }

  [[nodiscard]] Var<CompactTriangleShadowState> reduce_triangle_shadow_fast(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum) const noexcept {
    const bool cpu_unordered_shadow = is_cpu_scene(scene);
    const bool cpu_occluded_shadow = is_cpu_scene(scene);
    const bool has_curves = _plan.primitives.curves;
    Var<CompactTriangleShadowState> state;
    state->packed_state = 0u;
    state->max_dist = ray->t_max();
    if (!scene->has_transparent_shadow_materials && !has_curves) {
      const auto src = resolve_excluded_triangle_identity(scene, source);
      const auto dst = resolve_excluded_triangle_identity(scene, light);
      const auto handle_opaque_shadow =
          [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
            const auto hit = candidate.hit();
            const auto excluded =
                ((hit->inst == src.inst) & (hit->prim == src.prim)) |
                ((hit->inst == dst.inst) & (hit->prim == dst.prim));
            $if(!excluded) {
              candidate.commit();
              candidate.terminate();
            };
          };
      if (cpu_unordered_shadow) {
        const Bool bypass_excl =
            (src.inst == surface_ray::invalid_primitive) &
            (dst.inst == surface_ray::invalid_primitive);
        Bool occluded = false;
        $if(bypass_excl & (ray->t_min() == 0.0f)) {
          occluded = scene->accel->intersect_any(
              ray, {.visibility_mask = visibility_mask});
        }
        $else {
          const auto any_hit =
              scene->accel
                  ->traverse_any(ray, {.visibility_mask = visibility_mask})
                  .on_surface_candidate(handle_opaque_shadow)
                  .trace();
          occluded = !any_hit->miss();
        };
        state->packed_state = select(0u, 1u << 5u, occluded);
        return state;
      }
      const auto any_hit =
          scene->accel->traverse_any(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_opaque_shadow)
              .trace();
      state->packed_state = select(0u, 1u << 5u, !any_hit->miss());
      return state;
    }
    const auto src = resolve_excluded_triangle_identity(scene, source);
    const auto dst = resolve_excluded_triangle_identity(scene, light);

    const UInt max_trans = transparent_maximum;
    const auto opacity_words =
        Expr<Buffer<luisa::uint>>{*scene->traversal_shadow_opacity_buffer};
    constexpr auto capacity =
        static_cast<std::uint32_t>(shadow_intersection_batch_capacity);
    const auto recompute_farthest = [&]() noexcept {
      UInt rep = 0u;
      Float farthest = state->dist[0u];
      for (auto i = std::size_t{1u}; i < capacity; ++i) {
        const auto d = state->dist[static_cast<std::uint32_t>(i)];
        const auto f = d > farthest;
        rep = select(rep, static_cast<std::uint32_t>(i), f);
        farthest = select(farthest, d, f);
      }
      state->max_dist = farthest;
      state->packed_state =
          (state->packed_state & ~(3u << 3u)) | (rep << 3u);
    };
    const auto record_transparent_hit =
        [&](auto &candidate, Expr<std::uint32_t> hit_inst,
            Expr<std::uint32_t> hit_prim, Expr<float> hit_dist,
            Expr<luisa::float2> hit_bary, bool is_procedural_hit) noexcept {
          const auto cur_count = state->packed_state & 7u;
          Bool duplicate = false;
          for (auto index = std::size_t{0u};
               index < shadow_intersection_batch_capacity; ++index) {
            const auto idx = static_cast<std::uint32_t>(index);
            duplicate |= (idx < cur_count) &
                         (state->inst[idx] == hit_inst) &
                         (state->prim[idx] == hit_prim);
          }
          $if(!duplicate) {
            const auto next_total =
                ((state->packed_state >> 6u) & 0x3fffffu) + 1u;
            state->packed_state =
                (state->packed_state & 0xf000003fu) | (next_total << 6u);
            $if(next_total > max_trans) {
              state->packed_state |= (1u << 5u);
              if (cpu_occluded_shadow) {
                if constexpr (std::is_same_v<
                                  std::remove_cvref_t<decltype(candidate)>,
                                  luisa::compute::ProceduralCandidate>) {
                  candidate.commit(hit_dist);
                } else {
                  candidate.commit();
                }
              }
              candidate.terminate();
            }
            $else {
              $if(cur_count < capacity) {
                state->inst[cur_count] = hit_inst;
                state->prim[cur_count] = hit_prim;
                state->dist[cur_count] = hit_dist;
                state->bx[cur_count] = hit_bary.x;
                state->by[cur_count] = hit_bary.y;
                if (has_curves) {
                  const UInt proc_bit = 1u << (28u + cur_count);
                  if (is_procedural_hit) {
                    state->packed_state |= proc_bit;
                  } else {
                    state->packed_state &= ~proc_bit;
                  }
                }
                const auto next_count = cur_count + 1u;
                state->packed_state =
                    (state->packed_state & ~7u) | next_count;
                $if(next_count == capacity) { recompute_farthest(); }
                $else {
                  state->packed_state =
                      (state->packed_state & ~(3u << 3u)) |
                      ((next_count & 3u) << 3u);
                };
              }
              $else {
                $if(hit_dist < state->max_dist) {
                  const auto rep = (state->packed_state >> 3u) & 3u;
                  state->inst[rep] = hit_inst;
                  state->prim[rep] = hit_prim;
                  state->dist[rep] = hit_dist;
                  state->bx[rep] = hit_bary.x;
                  state->by[rep] = hit_bary.y;
                  if (has_curves) {
                    const UInt proc_bit = 1u << (28u + rep);
                    if (is_procedural_hit) {
                      state->packed_state |= proc_bit;
                    } else {
                      state->packed_state &= ~proc_bit;
                    }
                  }
                  recompute_farthest();
                };
              };
            };
          };
        };

    const auto handle_transparent_shadow =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto excluded =
              ((hit->inst == src.inst) & (hit->prim == src.prim)) |
              ((hit->inst == dst.inst) & (hit->prim == dst.prim));
          $if(!excluded) {
            Bool has_transparent = false;
            if (scene->has_transparent_shadow_materials) {
              const auto code = opacity_words.read(hit->inst);
              has_transparent = code != 0u;
              if (scene->has_mixed_shadow_opacity_instances) {
                has_transparent = code == ~0u;
                $if((code != 0u) & (code != ~0u)) {
                  const auto word =
                      opacity_words.read(code + (hit->prim >> 5u));
                  has_transparent =
                      ((word >> (hit->prim & 31u)) & 1u) != 0u;
                };
              }
            }
            $if(!has_transparent) {
              state->packed_state |= (1u << 5u);
              if (cpu_occluded_shadow) {
                candidate.commit();
              }
              candidate.terminate();
            }
            $else {
              record_transparent_hit(candidate, hit->inst, hit->prim,
                                     hit->committed_ray_t, hit->bary, false);
            };
          };
        };

    if (has_curves) {
      const auto handle_procedural_shadow =
          [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
            const auto hit = candidate.hit();
            const auto curve = emit_curve_metadata(scene, hit->inst, hit->prim);
            $if(curve.primitive.primitive_kind == geometry_kind_curve) {
              const auto object = curve.primitive.instance.cycles_object_index;
              const auto primitive = curve.segment.cycles_curve_index;
              const auto excluded = source.matches(object, primitive) |
                                    light.matches(object, primitive);
              $if(!excluded) {
                const auto object_ray = candidate.object_ray();
                const auto control_points =
                    emit_curve_control_points(scene, curve);
                const auto exact =
                    _ribbons->intersect(object_ray, control_points,
                                        curve.primitive.curve_subdivision_level);
                $if(exact.valid) {
                  Bool has_transparent = false;
                  if (scene->has_transparent_shadow_materials) {
                    const auto material_slot =
                        scene->heap
                            ->buffer<luisa::uint>(
                                curve.primitive.instance.bindless_base + 4u)
                            .read(curve.segment.curve_index);
                    const auto material_flags = resolve_material_flags(
                        scene, curve.primitive, material_slot);
                    has_transparent =
                        (material_flags & material_flag_has_transparent_shadow) !=
                        0u;
                  }
                  $if(!has_transparent) {
                    state->packed_state |= (1u << 5u);
                    if (cpu_occluded_shadow) {
                      candidate.commit(exact.distance);
                    }
                    candidate.terminate();
                  }
                  $else {
                    record_transparent_hit(
                        candidate, hit->inst, hit->prim, exact.distance,
                        make_float2(exact.u, exact.v), true);
                  };
                };
              };
            };
          };
      if (cpu_occluded_shadow) {
        const auto ignored =
            scene->accel
                ->traverse_any(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_transparent_shadow)
                .on_procedural_candidate(handle_procedural_shadow)
                .trace();
        static_cast<void>(ignored);
      } else {
        const auto ignored =
            scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_transparent_shadow)
                .on_procedural_candidate(handle_procedural_shadow)
                .trace();
        static_cast<void>(ignored);
      }
      return state;
    }

    $if(max_trans == 0u) {
      const auto handle_zero_trans_shadow =
          [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
            const auto hit = candidate.hit();
            const auto excluded =
                ((hit->inst == src.inst) & (hit->prim == src.prim)) |
                ((hit->inst == dst.inst) & (hit->prim == dst.prim));
            $if(!excluded) {
              candidate.commit();
              candidate.terminate();
            };
          };
      const auto any_hit =
          scene->accel->traverse_any(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_zero_trans_shadow)
              .trace();
      state->packed_state = select(0u, 1u << 5u, !any_hit->miss());
    }
    $else {
      if (cpu_occluded_shadow) {
        const auto ignored =
            scene->accel
                ->traverse_any(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_transparent_shadow)
                .trace();
        static_cast<void>(ignored);
      } else {
        const auto ignored =
            scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_transparent_shadow)
                .trace();
        static_cast<void>(ignored);
      }
    };
    return state;
  }

  [[nodiscard]] Bool trace_ambient_occlusion(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray,
      const ScenePrimitiveIdentity &source,
      Expr<bool> only_local) const noexcept {
    if (_plan.primitives.triangles && !_plan.primitives.curves &&
        scene->traversal_object_instance_buffer.has_value()) {
      Bool occluded = false;
      const auto src = resolve_excluded_triangle_identity(scene, source);
      $if(only_local) {
        if (scene->ambient_occlusion_local_accel) {
          const auto handle_local_ao =
              [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
                const auto hit = candidate.hit();
                const auto primary_instance =
                    (*scene->ambient_occlusion_local_accel)
                        ->instance_user_id(hit->inst);
                const auto accepted =
                    (primary_instance == src.inst) & (hit->prim != src.prim);
                $if(accepted) {
                  candidate.commit();
                  candidate.terminate();
                };
              };
          const auto any_hit =
              (*scene->ambient_occlusion_local_accel)
                  ->traverse_any(ray, {.visibility_mask = 0xffu})
                  .on_surface_candidate(handle_local_ao)
                  .trace();
          occluded = !any_hit->miss();
        }
      }
      $else {
        const auto handle_global_ao =
            [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
              const auto hit = candidate.hit();
              $if(!((hit->inst == src.inst) & (hit->prim == src.prim))) {
                candidate.commit();
                candidate.terminate();
              };
            };
        const auto any_hit =
            scene->accel
                ->traverse_any(ray, {.visibility_mask = shadow_visibility})
                .on_surface_candidate(handle_global_ao)
                .trace();
        occluded = !any_hit->miss();
      };
      return occluded;
    }

    Bool occluded = false;
    const bool fast_ao_surface_identity =
        _plan.primitives.triangles &&
        scene->traversal_object_instance_buffer.has_value();
    std::optional<ResolvedTriangleIdentity> fast_ao_src;
    if (fast_ao_surface_identity) {
      fast_ao_src.emplace(resolve_excluded_triangle_identity(scene, source));
    }

    const auto handle_surface = [&](
        luisa::compute::SurfaceCandidate &candidate,
        Expr<std::uint32_t> primary_instance,
        Expr<bool> restrict_to_source_object) noexcept {
          const auto hit = candidate.hit();
          Bool accepted = false;
          if (fast_ao_surface_identity) {
            accepted = select(
                !((primary_instance == fast_ao_src->inst) &
                  (hit->prim == fast_ao_src->prim)),
                (primary_instance == fast_ao_src->inst) &
                    (hit->prim != fast_ao_src->prim),
                restrict_to_source_object);
          } else {
            const auto metadata =
                emit_primitive_metadata(scene, primary_instance);
            const auto object = metadata.instance.cycles_object_index;
            const auto primitive =
                metadata.instance.cycles_primitive_offset + hit->prim;
            accepted =
                (!source.matches(object, primitive)) &
                ((!restrict_to_source_object) |
                 (object == source.object));
          }
          $if(accepted) {
            occluded = true;
            candidate.terminate();
          };
        };
    const auto handle_global_surface =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          handle_surface(candidate, hit->inst, false);
        };
    const auto handle_local_surface =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto primary_instance =
              (*scene->ambient_occlusion_local_accel)
                  ->instance_user_id(hit->inst);
          handle_surface(candidate, primary_instance, true);
        };
    const auto handle_procedural =
        [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto curve = emit_curve_metadata(scene, hit->inst, hit->prim);
          $if((!only_local) &
              (curve.primitive.primitive_kind == geometry_kind_curve)) {
            const auto object =
                curve.primitive.instance.cycles_object_index;
            const auto primitive = curve.segment.cycles_curve_index;
            $if(!source.matches(object, primitive)) {
              const auto exact = _ribbons->intersect(
                  candidate.object_ray(),
                  emit_curve_control_points(scene, curve),
                  curve.primitive.curve_subdivision_level);
              $if(exact.valid) {
                occluded = true;
                candidate.terminate();
              };
            };
          };
        };

    $if(only_local) {
      if (scene->ambient_occlusion_local_accel) {
        if (is_cpu_scene(scene)) {
          const auto ignored =
              (*scene->ambient_occlusion_local_accel)
                  ->traverse_any(ray, {.visibility_mask = 0xffu})
                  .on_surface_candidate(handle_local_surface)
                  .trace();
          static_cast<void>(ignored);
        } else {
          const auto ignored =
              (*scene->ambient_occlusion_local_accel)
                  ->traverse(ray, {.visibility_mask = 0xffu})
                  .on_surface_candidate(handle_local_surface)
                  .trace();
          static_cast<void>(ignored);
        }
      }
    }
    $else {
      if (_plan.primitives.mixed()) {
        if (is_cpu_scene(scene)) {
          const auto ignored =
              scene->accel
                  ->traverse_any(ray, {.visibility_mask = shadow_visibility})
                  .on_surface_candidate(handle_global_surface)
                  .on_procedural_candidate(handle_procedural)
                  .trace();
          static_cast<void>(ignored);
        } else {
          const auto ignored =
              scene->accel
                  ->traverse(ray, {.visibility_mask = shadow_visibility})
                  .on_surface_candidate(handle_global_surface)
                  .on_procedural_candidate(handle_procedural)
                  .trace();
          static_cast<void>(ignored);
        }
      } else if (_plan.primitives.triangles) {
        if (is_cpu_scene(scene)) {
          const auto ignored =
              scene->accel
                  ->traverse_any(ray, {.visibility_mask = shadow_visibility})
                  .on_surface_candidate(handle_global_surface)
                  .trace();
          static_cast<void>(ignored);
        } else {
          const auto ignored =
              scene->accel
                  ->traverse(ray, {.visibility_mask = shadow_visibility})
                  .on_surface_candidate(handle_global_surface)
                  .trace();
          static_cast<void>(ignored);
        }
      } else if (_plan.primitives.curves) {
        if (is_cpu_scene(scene)) {
          const auto ignored =
              scene->accel
                  ->traverse_any(ray, {.visibility_mask = shadow_visibility})
                  .on_procedural_candidate(handle_procedural)
                  .trace();
          static_cast<void>(ignored);
        } else {
          const auto ignored =
              scene->accel
                  ->traverse(ray, {.visibility_mask = shadow_visibility})
                  .on_procedural_candidate(handle_procedural)
                  .trace();
          static_cast<void>(ignored);
        }
      }
    };
    return occluded;
  }

public:
  explicit UnifiedSceneTraversalComponent(SceneTraversalStagePlan plan)
      : _plan{plan},
        _ribbons{plan.primitives.curves ? make_curve_ribbon_component()
                                        : nullptr} {}

  Var<luisa::compute::CommittedHit>
  closest(const std::shared_ptr<LuisaSceneData> &scene,
          const Var<luisa::compute::Ray> &ray,
          Expr<std::uint32_t> visibility_mask,
          const ScenePrimitiveIdentity &source) const noexcept override {
    return trace(scene, ray, visibility_mask, source,
                 ScenePrimitiveIdentity::invalid(), false);
  }

  Var<luisa::compute::CommittedHit>
  closest_shadow(const std::shared_ptr<LuisaSceneData> &scene,
                 const Var<luisa::compute::Ray> &ray,
                 Expr<std::uint32_t> visibility_mask,
                 const ScenePrimitiveIdentity &source,
                 const ScenePrimitiveIdentity &light) const noexcept override {
    return trace(scene, ray, visibility_mask, source, light, true);
  }

  Bool ambient_occluded(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray,
      const ScenePrimitiveIdentity &source,
      Expr<bool> only_local) const noexcept override {
    return trace_ambient_occlusion(scene, ray, source, only_local);
  }

  Var<ShadowIntersectionBatchCall> collect_shadow(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum) const noexcept override {
    if (can_use_fast_shadow_collect(scene)) {
      const auto state = reduce_triangle_shadow_fast(
          scene, ray, visibility_mask, source, light, transparent_maximum);
      const auto count = state->packed_state & 7u;
      const auto blocked = (state->packed_state >> 5u) & 1u;
      const auto total = (state->packed_state >> 6u) & 0x3fffffu;
      Var<ShadowIntersectionBatchCall> batch;
      for (auto index = std::size_t{0u};
           index < shadow_intersection_batch_capacity; ++index) {
        const auto idx = static_cast<std::uint32_t>(index);
        auto &hit = batch->hits[idx];
        initialize_shadow_intersection(hit, ray->t_max());
        if (scene->has_transparent_shadow_materials) {
          $if(idx < count) {
            hit->instance = state->inst[idx];
            hit->primitive = state->prim[idx];
            if (_plan.primitives.curves) {
              const Bool is_proc =
                  ((state->packed_state >> (28u + idx)) & 1u) != 0u;
              hit->hit_type = select(
                  static_cast<std::uint32_t>(luisa::compute::HitType::Surface),
                  static_cast<std::uint32_t>(
                      luisa::compute::HitType::Procedural),
                  is_proc);
            } else {
              hit->hit_type =
                  static_cast<std::uint32_t>(luisa::compute::HitType::Surface);
            }
            hit->distance = state->dist[idx];
            hit->barycentric = make_float2(state->bx[idx], state->by[idx]);
          };
        }
      }
      batch->count = count;
      batch->total = total;
      batch->blocked = blocked;
      return batch;
    }
    LocalShadowHitStorage storage{ray->t_max()};
    const auto summary = reduce_shadow_candidates(
        scene, ray, visibility_mask, source, light, transparent_maximum,
        storage);
    Var<ShadowIntersectionBatchCall> batch;
    storage.materialize(batch, summary->count, ray->t_max());
    batch->count = summary->count;
    batch->total = summary->total;
    batch->blocked = summary->blocked;
    return batch;
  }

  Var<ShadowIntersectionSummaryCall> collect_shadow_summary(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum,
      const ShadowIntersectionBatchStorage &storage,
      Expr<std::uint32_t> storage_invocation,
      Expr<std::uint32_t> storage_capacity,
      std::uint32_t max_transparent_bounces = ~0u) const noexcept override {
    if (can_use_fast_triangle_shadow(scene)) {
      if (!scene->has_transparent_shadow_materials) {
        const auto state = reduce_triangle_shadow_fast(
            scene, ray, visibility_mask, source, light, transparent_maximum);
        Var<ShadowIntersectionSummaryCall> result;
        result->count = 0u;
        result->total = 0u;
        result->blocked = (state->packed_state >> 5u) & 1u;
        return result;
      }
      constexpr auto capacity =
          static_cast<std::uint32_t>(shadow_intersection_batch_capacity);
      const bool bounded_by_capacity = max_transparent_bounces <= capacity;
      const auto src = resolve_excluded_triangle_identity(scene, source);
      const auto dst = resolve_excluded_triangle_identity(scene, light);
      const UInt max_trans = transparent_maximum;
      const UInt inv = storage_invocation;
      const UInt cap = storage_capacity;
      UInt packed_state = 0u;
      const auto opacity_words =
          Expr<Buffer<luisa::uint>>{*scene->traversal_shadow_opacity_buffer};
      if (bounded_by_capacity) {
        const auto handle_bounded_summary =
            [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
              const auto hit = candidate.hit();
              const auto excluded =
                  ((hit->inst == src.inst) & (hit->prim == src.prim)) |
                  ((hit->inst == dst.inst) & (hit->prim == dst.prim));
              $if(!excluded) {
                $if(max_trans == 0u) {
                  candidate.commit();
                  candidate.terminate();
                }
                $else {
                  const auto code = opacity_words.read(hit->inst);
                  Bool has_transparent = code != 0u;
                  if (scene->has_mixed_shadow_opacity_instances) {
                    has_transparent = code == ~0u;
                    $if((code != 0u) & (code != ~0u)) {
                      const auto word =
                          opacity_words.read(code + (hit->prim >> 5u));
                      has_transparent =
                          ((word >> (hit->prim & 31u)) & 1u) != 0u;
                    };
                  }
                  $if(!has_transparent) {
                    candidate.commit();
                    candidate.terminate();
                  }
                  $else {
                    const auto cur_count = packed_state & 7u;
                    Bool duplicate = false;
                    $if(cur_count != 0u) {
                      const auto id0 = storage.read_identity(inv, 0u, cap);
                      duplicate = (id0.x == hit->inst) & (id0.y == hit->prim);
                      $if(cur_count > 1u) {
                        const auto id1 = storage.read_identity(inv, 1u, cap);
                        duplicate |=
                            (id1.x == hit->inst) & (id1.y == hit->prim);
                        $if(cur_count > 2u) {
                          const auto id2 = storage.read_identity(inv, 2u, cap);
                          duplicate |=
                              (id2.x == hit->inst) & (id2.y == hit->prim);
                          $if(cur_count > 3u) {
                            const auto id3 =
                                storage.read_identity(inv, 3u, cap);
                            duplicate |=
                                (id3.x == hit->inst) & (id3.y == hit->prim);
                          };
                        };
                      };
                    };
                    $if(!duplicate) {
                      const auto next_total = (packed_state >> 6u) + 1u;
                      $if(next_total > max_trans) {
                        candidate.commit();
                        candidate.terminate();
                      }
                      $else {
                        Var<ShadowIntersectionCall> candidate_rec;
                        candidate_rec->instance = hit->inst;
                        candidate_rec->primitive = hit->prim;
                        candidate_rec->hit_type = static_cast<std::uint32_t>(
                            luisa::compute::HitType::Surface);
                        candidate_rec->distance = hit->committed_ray_t;
                        candidate_rec->barycentric = hit->bary;
                        storage.write(inv, cur_count, candidate_rec, cap);
                        packed_state = (cur_count + 1u) | (next_total << 6u);
                      };
                    };
                  };
                };
              };
            };
        const auto blocker_hit =
            scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
                .on_surface_candidate(handle_bounded_summary)
                .trace();
        Var<ShadowIntersectionSummaryCall> result;
        result->count = packed_state & 7u;
        result->total = packed_state >> 6u;
        result->blocked = select(0u, 1u, !blocker_hit->miss());
        return result;
      }

      Float max_dist = ray->t_max();
      const auto handle_transparent_summary =
          [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
            const auto hit = candidate.hit();
            const auto excluded =
                ((hit->inst == src.inst) & (hit->prim == src.prim)) |
                ((hit->inst == dst.inst) & (hit->prim == dst.prim));
            $if(!excluded) {
              $if(max_trans == 0u) {
                candidate.commit();
                candidate.terminate();
              }
              $else {
                const auto code = opacity_words.read(hit->inst);
                Bool has_transparent = code != 0u;
                if (scene->has_mixed_shadow_opacity_instances) {
                  has_transparent = code == ~0u;
                  $if((code != 0u) & (code != ~0u)) {
                    const auto word =
                        opacity_words.read(code + (hit->prim >> 5u));
                    has_transparent =
                        ((word >> (hit->prim & 31u)) & 1u) != 0u;
                  };
                }
                $if(!has_transparent) {
                  candidate.commit();
                  candidate.terminate();
                }
                $else {
                  const auto cur_count = packed_state & 7u;
                  Bool duplicate = false;
                  $if(cur_count != 0u) {
                    const auto id0 = storage.read_identity(inv, 0u, cap);
                    duplicate = (id0.x == hit->inst) & (id0.y == hit->prim);
                    $if(cur_count > 1u) {
                      const auto id1 = storage.read_identity(inv, 1u, cap);
                      duplicate |= (id1.x == hit->inst) & (id1.y == hit->prim);
                      $if(cur_count > 2u) {
                        const auto id2 = storage.read_identity(inv, 2u, cap);
                        duplicate |=
                            (id2.x == hit->inst) & (id2.y == hit->prim);
                        $if(cur_count > 3u) {
                          const auto id3 = storage.read_identity(inv, 3u, cap);
                          duplicate |=
                              (id3.x == hit->inst) & (id3.y == hit->prim);
                        };
                      };
                    };
                  };
                  $if(!duplicate) {
                    const auto next_total = (packed_state >> 6u) + 1u;
                    $if(next_total > max_trans) {
                      candidate.commit();
                      candidate.terminate();
                    }
                    $else {
                      packed_state =
                          (packed_state & 0x3fu) | (next_total << 6u);
                      Var<ShadowIntersectionCall> candidate_rec;
                      candidate_rec->instance = hit->inst;
                      candidate_rec->primitive = hit->prim;
                      candidate_rec->hit_type = static_cast<std::uint32_t>(
                          luisa::compute::HitType::Surface);
                      candidate_rec->distance = hit->committed_ray_t;
                      candidate_rec->barycentric = hit->bary;
                      Bool need_recompute = false;
                      $if(cur_count < capacity) {
                        storage.write(inv, cur_count, candidate_rec, cap);
                        const auto next_count = cur_count + 1u;
                        packed_state = (packed_state & ~7u) | next_count;
                        need_recompute =
                            (next_count == capacity) & (max_trans > capacity);
                        $if(!need_recompute) {
                          packed_state = (packed_state & ~(3u << 3u)) |
                                         ((next_count & 3u) << 3u);
                        };
                      }
                      $else {
                        $if(hit->committed_ray_t < max_dist) {
                          const auto rep = (packed_state >> 3u) & 3u;
                          storage.write(inv, rep, candidate_rec, cap);
                          need_recompute = true;
                        };
                      };
                      $if(need_recompute) {
                        UInt rep = 0u;
                        Float farthest = storage.read_distance(inv, 0u, cap);
                        for (auto i = std::size_t{1u}; i < capacity; ++i) {
                          const auto d = storage.read_distance(
                              inv, static_cast<std::uint32_t>(i), cap);
                          const auto f = d > farthest;
                          rep = select(rep, static_cast<std::uint32_t>(i), f);
                          farthest = select(farthest, d, f);
                        }
                        max_dist = farthest;
                        packed_state =
                            (packed_state & ~(3u << 3u)) | (rep << 3u);
                      };
                    };
                  };
                };
              };
            };
          };
      const auto blocker_hit =
          scene->accel->traverse(ray, {.visibility_mask = visibility_mask})
              .on_surface_candidate(handle_transparent_summary)
              .trace();
      Var<ShadowIntersectionSummaryCall> result;
      result->count = packed_state & 7u;
      result->total = packed_state >> 6u;
      result->blocked = select(0u, 1u, !blocker_hit->miss());
      return result;
    }
    SoAShadowHitStorage hits{&storage, storage_invocation, storage_capacity};
    return reduce_shadow_candidates(scene, ray, visibility_mask, source, light,
                                    transparent_maximum, hits);
  }
};

} // namespace

std::shared_ptr<const SceneTraversalComponent>
make_scene_traversal_component(SceneTraversalStagePlan plan) {
  return std::make_shared<UnifiedSceneTraversalComponent>(plan);
}

} // namespace psycles::luisa_backend::detail
