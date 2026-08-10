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

struct TriangleResolutionQueryCall {
  luisa::uint visibility_mask{};
  luisa::uint source_object{};
  luisa::uint source_primitive{};
  luisa::uint light_object{};
  luisa::uint light_primitive{};
  luisa::uint seed_instance{};
  luisa::uint local_primitive{};
};

struct TriangleResolutionCall {
  luisa::float2 barycentric{};
  float distance{};
  luisa::uint instance{};
  luisa::uint primitive{};
  luisa::uint object{};
  luisa::uint cycles_primitive{};
  luisa::uint type_order{};
  luisa::uint valid{};
};

} // namespace psycles::luisa_backend::detail

LUISA_STRUCT(
    psycles::luisa_backend::detail::TriangleResolutionQueryCall,
    visibility_mask,
    source_object,
    source_primitive,
    light_object,
    light_primitive,
    seed_instance,
    local_primitive) {};

LUISA_STRUCT(
    psycles::luisa_backend::detail::TriangleResolutionCall,
    barycentric,
    distance,
    instance,
    primitive,
    object,
    cycles_primitive,
    type_order,
    valid) {};

namespace psycles::luisa_backend::detail {

using TriangleResolverCallable = Callable<
    TriangleResolutionCall(
        luisa::compute::Ray,
        TriangleResolutionQueryCall)>;

// Emits one order-independent closest-hit query for every scene surface.
// Hardware triangles and procedural Cycles ribbons share exact
// (object, primitive) self identities even though their acceleration
// primitives have different local address spaces.
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
