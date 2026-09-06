#include "path_tracer_geometry.h"

#include "path_kernel_scene_traversal.h"
#include "path_kernel_shadow_storage.h"
#include "path_kernel_surface_primitive.h"
#include "path_tracer_cycles_svm_shadow.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {

Var<ShadowShaderContextCall> make_shadow_shader_context(
    const Var<ShaderEvaluationStateCall> &path, Expr<float> time,
    Expr<luisa::uint> sample, Expr<luisa::uint> rng_hash,
    Expr<luisa::uint> rng_offset,
    Expr<luisa::uint> volume_bounds_bounce) noexcept {
    Var<ShadowShaderContextCall> result;
    result.path = path;
    result.ray_time = time;
    result.sample_index = sample;
    result.rng_hash = rng_hash;
    result.rng_offset = rng_offset;
    result.volume_bounds_bounce = volume_bounds_bounce;
    return result;
}

Bool advance_shadow_surface_state(
    Var<ShadowShaderContextCall> &context, Float3 &throughput,
    const Var<ShadowSurfaceEvaluationCall> &surface) noexcept {
    // These three fields are uint16_t in Cycles' shadow_state_template.h.
    context.volume_bounds_bounce =
        (context.volume_bounds_bounce + surface.volume_boundary) & 0xffffu;
    const auto next = throughput * surface.transmittance;
    const auto nonzero = any(next != 0.0f);
    $if(nonzero) {
        throughput = next;
        context.path.transparent_depth =
            (context.path.transparent_depth + 1u) & 0xffffu;
        context.rng_offset =
            (context.rng_offset + cycles_path_state::bounce_dimension_count) &
            0xffffu;
    };
    return nonzero &
           (context.volume_bounds_bounce <= shadow_volume_bounds_max);
}

void sort_shadow_intersection_batch(
    Var<ShadowIntersectionBatchCall> &batch) noexcept {
    const auto compare_exchange =
        [&batch](luisa::uint left, luisa::uint right) noexcept {
            auto &a = batch->hits[left];
            auto &b = batch->hits[right];
            $if(a->distance > b->distance) {
                const auto temporary = def(a);
                a = b;
                b = temporary;
            };
        };
    // Optimal sorting network for four values. Inactive lanes are initialized
    // to ray.t_max, so the same network places them after every active hit.
    compare_exchange(0u, 1u);
    compare_exchange(2u, 3u);
    compare_exchange(0u, 2u);
    compare_exchange(1u, 3u);
    compare_exchange(1u, 2u);
}

ShadowIntersectionComponent::ShadowIntersectionComponent(
    LocalShadowIntersectionCallable intersect) noexcept
    : _intersect{std::move(intersect)} {}

ShadowIntersectionComponent::ShadowIntersectionComponent(
    std::shared_ptr<const ShadowIntersectionBatchStorage> storage,
    IntersectShadowCallable intersect) noexcept
    : _storage{std::move(storage)}, _intersect{std::move(intersect)} {
    LUISA_ASSERT(_storage != nullptr,
                 "Stored shadow intersection component requires storage.");
}

Var<ShadowIntersectionBatchCall> ShadowIntersectionComponent::collect(
    Var<luisa::compute::Ray> shadow_ray,
    Expr<std::uint32_t> source_object,
    Expr<std::uint32_t> source_primitive,
    Expr<std::uint32_t> light_object,
    Expr<std::uint32_t> light_primitive,
    Expr<std::uint32_t> transparent_maximum,
    Expr<std::uint32_t> storage_capacity,
    Expr<std::uint32_t> storage_block_size) const noexcept {
    if (!_storage) {
        return std::get<LocalShadowIntersectionCallable>(_intersect)(
            shadow_ray, source_object, source_primitive, light_object,
            light_primitive, transparent_maximum);
    }
    // The outer kernel owns launch geometry. Its explicit runtime stride makes
    // this map injective over physical lanes; the callable receives the
    // already-proved storage identity and cannot inspect launch metadata.
    const auto invocation = shadow_storage_invocation(storage_block_size);
    const auto summary = std::get<IntersectShadowCallable>(_intersect)(
        shadow_ray, source_object, source_primitive, light_object,
        light_primitive, transparent_maximum, invocation, storage_capacity);
    return _storage->materialize(invocation, summary, shadow_ray->t_max(),
                                 storage_capacity);
}

const IntersectShadowCallable &
ShadowIntersectionComponent::summary_callable() const noexcept {
    LUISA_ASSERT(_storage != nullptr,
                 "Local shadow traversal does not have a summary callable.");
    return std::get<IntersectShadowCallable>(_intersect);
}

namespace {

[[nodiscard]] EvaluateShadowSurfaceCallable
make_evaluate_shadow_surface_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept {
    if (scene->native_cycles_svm_surface &&
        (!scene->geometries.empty() || !scene->curve_geometries.empty())) {
        return make_cycles_svm_shadow_surface_callable(scene);
    }
    const auto primitive_plan =
        make_scene_primitive_stage_plan(
            scene->geometries.size(),
            scene->curve_geometries.size());
    const auto geometry =
        primitive_plan.empty()
            ? nullptr
            : make_surface_primitive_geometry_component(
                  primitive_plan);
    EvaluateShadowSurfaceCallable evaluate_shadow_surface =
        [scene, safe_normalize, geometry](
            Var<luisa::compute::Ray> candidate_ray,
            Var<ShadowIntersectionCall> intersection,
            Float ray_dP,
            Float ray_dD,
            Var<ShadowShaderContextCall> context,
            Var<RenderKernelParameters>) noexcept {
            Var<ShadowSurfaceEvaluationCall> result;
            if (!geometry) {
                result->transmittance =
                    make_float3(1.0f);
                result->object =
                    surface_ray::invalid_primitive;
                result->primitive =
                    surface_ray::invalid_primitive;
                result->kind =
                    surface_ray::invalid_primitive;
                return result;
            }
            Var<luisa::compute::CommittedHit> hit;
            hit->inst = intersection->instance;
            hit->prim = intersection->primitive;
            hit->bary = intersection->barycentric;
            hit->hit_type = intersection->hit_type;
            hit->committed_ray_t = intersection->distance;
            const auto shader_state =
                unpack_shader_evaluation_state(context.path);
            BufferShaderServices services{
                scene->scalar_parameter_buffer,
                scene->vector_parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto primitive = geometry->emit(
                scene,
                hit,
                candidate_ray,
                ray_dP,
                ray_dD,
                safe_normalize);
            auto point = std::move(primitive.point);
            point.time = context.ray_time;
            point.ray_visibility = shadow_visibility;
            cycles_path_state::apply_shader_state(point, shader_state);
            result->transmittance = clamp(
                scene->surfaces.transparent_extinction(
                    primitive.surface_tag, services, point),
                make_float3(0.0f),
                make_float3(1.0f));
            result->object = primitive.cycles_object_index;
            result->primitive = primitive.cycles_primitive_index;
            result->kind = select(
                geometry_kind_triangle,
                geometry_kind_curve,
                primitive.is_curve);
            return result;
        };
    return evaluate_shadow_surface;
}

} // namespace

ShadowTraceCallables make_shadow_trace_callables(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize,
    std::shared_ptr<const ShadowIntersectionBatchStorage> storage) noexcept {
    const auto evaluate_shadow_surface =
        make_evaluate_shadow_surface_callable(scene, safe_normalize);
    const auto traversal =
        make_scene_traversal_component(
            make_scene_traversal_stage_plan(
                scene->geometries.size(),
                scene->curve_geometries.size()));
    std::shared_ptr<const ShadowIntersectionComponent> intersection;
    if (storage) {
      auto captured_storage = storage;
      IntersectShadowCallable intersect_summary =
          [scene, traversal, captured_storage](
              Var<luisa::compute::Ray> shadow_ray,
              UInt source_object, UInt source_primitive,
              UInt light_object, UInt light_primitive,
              UInt transparent_maximum, UInt storage_invocation,
              UInt storage_capacity) noexcept {
            return traversal->collect_shadow_summary(
                scene, shadow_ray, shadow_visibility,
                {.object = source_object, .primitive = source_primitive},
                {.object = light_object, .primitive = light_primitive},
                transparent_maximum, *captured_storage, storage_invocation,
                storage_capacity);
          };
      intersection =
          std::make_shared<ShadowIntersectionComponent>(
              std::move(storage), std::move(intersect_summary));
    } else {
      auto intersect_local = LocalShadowIntersectionCallable{
          [scene, traversal](
              Var<luisa::compute::Ray> shadow_ray,
              UInt source_object, UInt source_primitive,
              UInt light_object, UInt light_primitive,
              UInt transparent_maximum) noexcept {
            return traversal->collect_shadow(
                scene, shadow_ray, shadow_visibility,
                {.object = source_object, .primitive = source_primitive},
                {.object = light_object, .primitive = light_primitive},
                transparent_maximum);
          }};
      intersection = std::make_shared<ShadowIntersectionComponent>(
          std::move(intersect_local));
    }
    auto trace_shadow =
        make_fused_shadow_trace_callable(intersection, evaluate_shadow_surface);
    return {.intersect = std::move(intersection),
            .shade_surface = std::move(evaluate_shadow_surface),
            .trace = std::move(trace_shadow)};
}

TraceShadowCallable make_fused_shadow_trace_callable(
    std::shared_ptr<const ShadowIntersectionComponent> intersection,
    EvaluateShadowSurfaceCallable evaluate_shadow_surface) noexcept {
  return [intersection, evaluate_shadow_surface](
             Var<luisa::compute::Ray> shadow_ray, Float ray_dP, Float ray_dD,
             UInt source_object, UInt source_primitive, UInt light_object,
             UInt light_primitive, UInt transparent_maximum,
             UInt storage_capacity, UInt storage_block_size, Float3 throughput,
             Var<ShadowShaderContextCall> context,
             Var<RenderKernelParameters> parameters) noexcept {
    Float3 transmittance = make_float3(1.0f);
    Bool active = true;
    Bool first_hit = false;
    UInt first_object = surface_ray::invalid_primitive;
    UInt first_primitive = surface_ray::invalid_primitive;
    UInt first_kind = surface_ray::invalid_primitive;
    Float first_distance = 0.0f;
    Float2 first_barycentric = make_float2(0.0f);

    // Candidate callbacks have no traversal-order contract. The
    // collector reduces one traversal to the nearest sorted batch;
    // repeat whenever the four slots are full, as in Cycles.
    $while(active) {
      const auto remaining =
          transparent_maximum -
          min(context.path.transparent_depth, transparent_maximum);
      auto batch = intersection->collect(
          shadow_ray, source_object, source_primitive, light_object,
          light_primitive, remaining, storage_capacity, storage_block_size);
      sort_shadow_intersection_batch(batch);
      $if(batch->blocked != 0u) {
        transmittance = make_float3(0.0f);
        throughput = make_float3(0.0f);
        active = false;
      }
      $else {
        Bool carries_light = true;
        Float last_distance = shadow_ray->t_min();
        for (auto index = std::size_t{0u};
             index < shadow_intersection_batch_capacity; ++index) {
          const auto shade = static_cast<std::uint32_t>(index) < batch->count;
          $if(shade & carries_light) {
            const auto &intersection =
                batch->hits[static_cast<luisa::uint>(index)];
            const auto surface = evaluate_shadow_surface(
                shadow_ray, intersection, ray_dP, ray_dD, context, parameters);
            $if(!first_hit) {
              first_hit = true;
              first_object = surface->object;
              first_primitive = surface->primitive;
              first_kind = surface->kind;
              first_distance = intersection->distance;
              first_barycentric = intersection->barycentric;
            };
            carries_light =
                advance_shadow_surface_state(context, throughput, surface);
            $if(carries_light) {
              transmittance *= surface->transmittance;
              last_distance = intersection->distance;
            }
            $else {
              transmittance = make_float3(0.0f);
              throughput = make_float3(0.0f);
            };
          };
        }
        $if(context.volume_bounds_bounce > shadow_volume_bounds_max) {
          carries_light = false;
          transmittance = make_float3(0.0f);
          throughput = make_float3(0.0f);
        };
        active = carries_light &
                 (batch->count ==
                  static_cast<luisa::uint>(shadow_intersection_batch_capacity));
        $if(active) {
          shadow_ray->set_t_min(
              surface_ray::intersection_t_offset(last_distance));
        };
      };
    };
    Var<ShadowTraceResultCall> result;
    result->transmittance = transmittance;
    result->throughput = throughput;
    result->first_hit = cast<uint>(first_hit);
    result->first_object = first_object;
    result->first_primitive = first_primitive;
    result->first_kind = first_kind;
    result->first_distance = first_distance;
    result->first_barycentric = first_barycentric;
    return result;
  };
}

} // namespace psycles::luisa_backend::detail
