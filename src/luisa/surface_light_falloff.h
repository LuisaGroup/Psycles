#pragma once

#include <cstdint>
#include <span>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Cycles' Light Falloff node is a ShaderData operation, not ordinary graph
// algebra. In particular, FLT_MAX is a semantic distant-light sentinel that
// returns Strength before any distance product is evaluated.
[[nodiscard]] Float evaluate_surface_light_falloff(
    compiler::LightFalloffType type,
    Float strength,
    Float smooth,
    Float ray_length) noexcept;

// Compact SVM entry. The domain is the exact set of falloff outputs reachable
// in the scene and therefore the exact set of cases recorded in the callable.
[[nodiscard]] Float evaluate_surface_light_falloff_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float strength,
    Float smooth,
    Float ray_length) noexcept;

} // namespace psycles::luisa_backend::detail
