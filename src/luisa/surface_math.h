#pragma once

#include <cstdint>
#include <span>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float3 safe_normalize(
    Float3 value,
    Float3 fallback) noexcept;

// One statically selected Cycles Math operation. Both the topology-expanded
// graph and compact SVM handler record their Luisa AST through this function.
[[nodiscard]] Float evaluate_surface_math_operation(
    compiler::MathOperation operation,
    Float a,
    Float b,
    Float c) noexcept;

// Compact SVM entry. The mode is data in the typed instruction immediate;
// `immediate_domain` is the exact set reachable by this scene variant and
// therefore the exact set of cases recorded into the shared handler AST.
[[nodiscard]] Float evaluate_surface_math_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float a,
    Float b,
    Float c) noexcept;

} // namespace psycles::luisa_backend::detail
