#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

struct SurfaceBumpConfiguration {
    bool invert{};
    bool normal_linked{};
    bool object_space{};
};

// Compact SVM keeps Cycles' NODE_SET_BUMP flags in the instruction record.
// These are device booleans rather than host/JIT specialization keys.
struct SurfaceBumpSvmConfiguration {
    Bool invert;
    Bool normal_linked;
    Bool object_space;
};

struct SurfaceBumpEvaluationDomain {
    Float filter_width;
    SurfacePoint point_x;
    SurfacePoint point_y;
};

[[nodiscard]] SurfaceBumpConfiguration
decode_surface_bump_configuration(
    std::uint64_t encoded) noexcept;

[[nodiscard]] SurfaceBumpEvaluationDomain
make_surface_bump_evaluation_domain(
    const SurfacePoint &point,
    Float filter_width) noexcept;

[[nodiscard]] Float3 evaluate_surface_bump(
    const ShaderServices &services,
    const SurfacePoint &point,
    SurfaceBumpConfiguration configuration,
    Float3 normal,
    const SurfaceBumpEvaluationDomain &domain,
    Float height_center,
    Float height_x,
    Float height_y,
    Float distance,
    Float strength) noexcept;

[[nodiscard]] Float3 evaluate_surface_bump(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceBumpSvmConfiguration &configuration,
    Float3 normal,
    Float filter_width,
    Float height_center,
    Float height_x,
    Float height_y,
    Float distance,
    Float strength) noexcept;

[[nodiscard]] Float3 bump_world_inline(
    const SurfaceBumpInput &input) noexcept;
[[nodiscard]] Float3 bump_object_inline(
    const SurfaceBumpInput &input) noexcept;

[[nodiscard]] Float3 bump_world(
    const ShaderServices &services,
    const SurfaceBumpInput &input) noexcept;
[[nodiscard]] Float3 bump_object(
    const ShaderServices &services,
    const SurfaceBumpInput &input) noexcept;

}// namespace psycles::luisa_backend::detail
