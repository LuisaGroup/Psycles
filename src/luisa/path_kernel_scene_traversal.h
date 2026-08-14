#pragma once

#include "path_kernel_scene_geometry_plan.h"
#include "path_tracer_internal.h"

#include <memory>

namespace psycles::luisa_backend::detail {

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
};

[[nodiscard]] std::shared_ptr<const SceneTraversalComponent>
make_scene_traversal_component(
    SceneTraversalStagePlan plan);

} // namespace psycles::luisa_backend::detail
