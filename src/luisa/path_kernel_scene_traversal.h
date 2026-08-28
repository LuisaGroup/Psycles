#pragma once

#include "path_kernel_scene_geometry_plan.h"
#include "path_tracer_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

class ShadowIntersectionBatchStorage;

struct ScenePrimitiveIdentity {
  UInt object;
  UInt primitive;

  [[nodiscard]] static ScenePrimitiveIdentity invalid() noexcept;

  [[nodiscard]] Bool
  matches(Expr<std::uint32_t> candidate_object,
          Expr<std::uint32_t> candidate_primitive) const noexcept;
};

} // namespace psycles::luisa_backend::detail

namespace psycles::luisa_backend::detail {

// Emits one generic RayQuery for every scene surface. Candidate callbacks map
// hardware triangles and procedural Cycles ribbons into their shared
// (object, primitive) identity space and express source/light exclusion there.
// Backends see only ordinary RayQuery operations; candidate enumeration,
// interval details, and tie order remain backend-native implementation details.
class SceneTraversalComponent {

public:
  virtual ~SceneTraversalComponent() noexcept = default;

  [[nodiscard]] virtual Var<luisa::compute::CommittedHit>
  closest(const std::shared_ptr<LuisaSceneData> &scene,
          const Var<luisa::compute::Ray> &ray,
          Expr<std::uint32_t> visibility_mask,
          const ScenePrimitiveIdentity &source) const noexcept = 0;

  [[nodiscard]] virtual Var<luisa::compute::CommittedHit>
  closest_shadow(const std::shared_ptr<LuisaSceneData> &scene,
                 const Var<luisa::compute::Ray> &ray,
                 Expr<std::uint32_t> visibility_mask,
                 const ScenePrimitiveIdentity &source,
                 const ScenePrimitiveIdentity &light) const noexcept = 0;

  // Cycles' shader AO is an opaque any-hit query, not a closest-hit query.
  // Global AO observes the ordinary shadow-visible TLAS including curves;
  // Only Local observes the complete current-object triangle domain and
  // deliberately ignores normal instance visibility. Exact self identity is
  // rejected in either domain, so the ray begins at ShaderData::P with tmin 0.
  [[nodiscard]] virtual Bool ambient_occluded(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray,
      const ScenePrimitiveIdentity &source,
      Expr<bool> only_local) const noexcept = 0;

  // Enumerates one backend-native RayQuery and reduces its order-independent
  // candidate stream to the nearest fixed-capacity batch used by the shadow
  // scheduler. `transparent_maximum` is the remaining transparent-bounce
  // budget; exceeding it is an opaque terminal result.
  [[nodiscard]] virtual Var<ShadowIntersectionBatchCall> collect_shadow(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum) const noexcept = 0;

  // Same order-independent reduction, but the callback writes retained hits
  // directly to external SoA and returns only its compact private summary.
  // Keeping this a distinct virtual operation makes returning the full batch
  // from the RayQuery callable a type error rather than a performance
  // convention that a later refactor can accidentally violate.
  [[nodiscard]] virtual Var<ShadowIntersectionSummaryCall>
  collect_shadow_summary(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::Ray> &ray, Expr<std::uint32_t> visibility_mask,
      const ScenePrimitiveIdentity &source, const ScenePrimitiveIdentity &light,
      Expr<std::uint32_t> transparent_maximum,
      const ShadowIntersectionBatchStorage &storage,
      Expr<std::uint32_t> storage_invocation,
      Expr<std::uint32_t> storage_capacity) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const SceneTraversalComponent>
make_scene_traversal_component(
    SceneTraversalStagePlan plan);

} // namespace psycles::luisa_backend::detail
