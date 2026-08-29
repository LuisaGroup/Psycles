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
    const auto duplicate = contains(intersection);
    $if(!duplicate) {
      $if(!has_transparent_shadow) {
        _blocked = 1u;
        query_candidate.terminate();
      }
      $else {
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
          const auto metadata = emit_primitive_metadata(scene, hit->inst);
          const auto object = metadata.instance.cycles_object_index;
          const auto primitive =
              metadata.instance.cycles_primitive_offset + hit->prim;
          const auto excluded = source.matches(object, primitive) |
                                light.matches(object, primitive);
          $if(!excluded) { candidate.commit(); };
        };
    const auto handle_procedural =
        [&](luisa::compute::ProceduralCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto curve = emit_curve_metadata(scene, hit->inst, hit->prim);
          $if(curve.primitive.primitive_kind == geometry_kind_curve) {
            const auto object =
                curve.primitive.instance.cycles_object_index;
            const auto primitive = curve.segment.cycles_curve_index;
            const auto excluded = source.matches(object, primitive) |
                                  light.matches(object, primitive);
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

    const auto handle_surface =
        [&](luisa::compute::SurfaceCandidate &candidate) noexcept {
          const auto hit = candidate.hit();
          const auto metadata = emit_primitive_metadata(scene, hit->inst);
          const auto object = metadata.instance.cycles_object_index;
          const auto primitive =
              metadata.instance.cycles_primitive_offset + hit->prim;
          const auto excluded = source.matches(object, primitive) |
                                light.matches(object, primitive);
          $if(!excluded) {
            const auto material_slot = scene->heap
                                           ->buffer<luisa::uint>(
                                               metadata.instance.bindless_base +
                                               4u)
                                           .read(hit->prim);
            const auto material_flags =
                resolve_material_flags(scene, metadata, material_slot);
            Var<ShadowIntersectionCall> intersection;
            intersection->instance = hit->inst;
            intersection->primitive = hit->prim;
            intersection->hit_type =
                static_cast<std::uint32_t>(luisa::compute::HitType::Surface);
            intersection->distance = hit->committed_ray_t;
            intersection->barycentric = hit->bary;
            reducer.reduce(
                candidate, intersection,
                (material_flags & material_flag_has_transparent_shadow) != 0u,
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

  [[nodiscard]] Bool trace_ambient_occlusion(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray,
      const ScenePrimitiveIdentity &source,
      Expr<bool> only_local) const noexcept {
    Bool occluded = false;

    const auto handle_surface = [&](
        luisa::compute::SurfaceCandidate &candidate,
        Expr<std::uint32_t> primary_instance,
        Expr<bool> restrict_to_source_object) noexcept {
          const auto hit = candidate.hit();
          const auto metadata =
              emit_primitive_metadata(scene, primary_instance);
          const auto object = metadata.instance.cycles_object_index;
          const auto primitive =
              metadata.instance.cycles_primitive_offset + hit->prim;
          const auto accepted =
              (!source.matches(object, primitive)) &
              ((!restrict_to_source_object) |
               (object == source.object));
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
        const auto ignored = (*scene->ambient_occlusion_local_accel)->traverse(
            ray, {.visibility_mask = 0xffu})
                                 .on_surface_candidate(handle_local_surface)
                                 .trace();
        static_cast<void>(ignored);
      }
    }
    $else {
      if (_plan.primitives.mixed()) {
        const auto ignored =
            scene->accel->traverse(
                ray, {.visibility_mask = shadow_visibility})
                .on_surface_candidate(handle_global_surface)
                .on_procedural_candidate(handle_procedural)
                .trace();
        static_cast<void>(ignored);
      } else if (_plan.primitives.triangles) {
        const auto ignored =
            scene->accel->traverse(
                ray, {.visibility_mask = shadow_visibility})
                .on_surface_candidate(handle_global_surface)
                .trace();
        static_cast<void>(ignored);
      } else if (_plan.primitives.curves) {
        const auto ignored =
            scene->accel->traverse(
                ray, {.visibility_mask = shadow_visibility})
                .on_procedural_candidate(handle_procedural)
                .trace();
        static_cast<void>(ignored);
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
      Expr<std::uint32_t> storage_capacity) const noexcept override {
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
