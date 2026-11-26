#pragma once

#include "path_tracer_internal.h"
#include "path_kernel_scene_traversal.h"
#include "cycles_svm_bsdf.h"

namespace psycles::luisa_backend::detail {

struct PathSurfaceAmbientOcclusionContext {
    std::shared_ptr<const SceneTraversalComponent> traversal;
    ScenePrimitiveIdentity source;
};

struct SurfacePopulationContext {
    Expr<std::uint32_t> surface_tag;
    const SurfacePoint &point;
    const SurfacePopulationQuery &query;
    Expr<std::uint32_t> cycles_surface_shader;
    Expr<std::uint32_t> cycles_object_index;
    Expr<std::uint32_t> cycles_primitive_index;
    Expr<std::uint32_t> primitive_type;
    Expr<luisa::float3> ray_origin;
    Expr<float> ray_position_differential;
    Expr<float> ray_direction_differential;
    Expr<std::uint32_t> path_flags;
    Expr<std::uint32_t> portal_depth = 0u;
    Expr<std::uint32_t> sample_index;
    Expr<std::uint32_t> rng_hash;
    Expr<std::uint32_t> rng_offset;
    const Var<RenderKernelParameters> &parameters;
    CameraProjection camera_projection;
    const PathSurfaceAmbientOcclusionContext *ambient_occlusion;
    std::optional<Expr<std::uint32_t>> cycles_shader_flags{};
    std::optional<Expr<std::uint32_t>> cycles_object_flags{};
    bool may_filter_glossy{true};
    UInt *vt_miss_page{nullptr};
};

// Host/JIT object whose device-local storage is populated exactly once for a
// path hit. Every method consumes retained native closures without replaying
// the material graph.
struct CyclesSvmSurfaceState {
    const cycles_svm::KernelGlobals &kernel_globals;
    cycles_svm::ShaderData &shader_data;
    cycles_svm::detail::ClosureTypeMask closure_types;
};

class PopulatedSurfaceShader {

  public:
    virtual ~PopulatedSurfaceShader() noexcept = default;

    [[nodiscard]] virtual std::optional<CyclesSvmSurfaceState>
    native_surface_state() const noexcept { return std::nullopt; }

    [[nodiscard]] virtual Expr<std::uint32_t>
    native_shader_data_flags() const noexcept {
      return Expr<std::uint32_t>{0u};
    }

    [[nodiscard]] virtual Expr<std::uint32_t>
    closure_count() const noexcept = 0;
    [[nodiscard]] virtual SurfacePreparation preparation() const noexcept = 0;
    [[nodiscard]] virtual SurfaceEvaluation evaluate_light(
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept = 0;
    [[nodiscard]] virtual SurfaceClosureTrace closure_trace(
        Expr<std::uint32_t> requested_index,
        const SurfaceQuery &query) const noexcept = 0;
    [[nodiscard]] virtual SurfaceSample sample(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept = 0;
    [[nodiscard]] virtual SurfaceSampleTrace sample_trace(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept = 0;
};

class SurfacePopulationComponent {

  public:
    virtual ~SurfacePopulationComponent() noexcept = default;

    [[nodiscard]] virtual std::shared_ptr<PopulatedSurfaceShader> populate(
        const SurfacePopulationContext &context) const noexcept = 0;
};

struct SurfaceCallables {
    std::shared_ptr<const SurfacePopulationComponent> population;
};

[[nodiscard]] SurfaceCallables make_surface_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

}// namespace psycles::luisa_backend::detail
