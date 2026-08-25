#pragma once

#include <cstdint>
#include <span>

#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Static graph entries and compact SVM handlers share these implementations.
// The former selects a Cycles mode while recording the AST; the latter reads
// the same mode from the opcode-owned instruction immediate.
[[nodiscard]] Float evaluate_surface_clamp(compiler::ClampMode mode,
                                           Float value, Float minimum,
                                           Float maximum) noexcept;

[[nodiscard]] Float
evaluate_surface_clamp_svm(UInt immediate,
                           std::span<const std::uint16_t> immediate_domain,
                           Float value, Float minimum, Float maximum) noexcept;

[[nodiscard]] Float
evaluate_surface_map_range_float(compiler::MapRangeInterpolation interpolation,
                                 bool clamp_result, Float value, Float from_min,
                                 Float from_max, Float to_min, Float to_max,
                                 Float steps) noexcept;

[[nodiscard]] Float evaluate_surface_map_range_float_svm(
    UInt immediate, std::span<const std::uint16_t> immediate_domain,
    Float value, Float from_min, Float from_max, Float to_min, Float to_max,
    Float steps) noexcept;

[[nodiscard]] Float3 evaluate_surface_map_range_vector(
    compiler::MapRangeInterpolation interpolation, bool clamp_result,
    Float3 value, Float3 from_min, Float3 from_max, Float3 to_min,
    Float3 to_max, Float3 steps) noexcept;

[[nodiscard]] Float3 evaluate_surface_map_range_vector_svm(
    UInt immediate, std::span<const std::uint16_t> immediate_domain,
    Float3 value, Float3 from_min, Float3 from_max, Float3 to_min,
    Float3 to_max, Float3 steps) noexcept;

} // namespace psycles::luisa_backend::detail
