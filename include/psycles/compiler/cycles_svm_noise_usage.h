#pragma once

#include <cstdint>

namespace psycles::compiler::cycles_svm {

// Static shape facts for emitted NODE_TEX_NOISE payloads. The five Cycles
// noise types and four valid coordinate dimensions occupy one bit each.
struct NoiseUsage {
  static constexpr std::uint32_t shape_count = 20u;
  static constexpr std::uint32_t all_shapes = (1u << shape_count) - 1u;

  // Unknown external images conservatively retain every shape. Compiler
  // generated entries explicitly start from none() and add emitted payloads.
  std::uint32_t shape_mask{all_shapes};

  [[nodiscard]] static constexpr NoiseUsage none() noexcept { return {0u}; }

  [[nodiscard]] constexpr bool contains(std::uint32_t dimensions,
                                        std::uint32_t type) const noexcept {
    if (dimensions < 1u || dimensions > 4u || type >= 5u) {
      return true;
    }
    const auto bit = type * 4u + dimensions - 1u;
    return (shape_mask & (1u << bit)) != 0u;
  }

  constexpr void add(std::uint32_t dimensions,
                     std::uint32_t type) noexcept {
    if (dimensions < 1u || dimensions > 4u || type >= 5u) {
      shape_mask = all_shapes;
      return;
    }
    shape_mask |= 1u << (type * 4u + dimensions - 1u);
  }

  friend constexpr bool operator==(const NoiseUsage &, const NoiseUsage &) =
      default;
};

} // namespace psycles::compiler::cycles_svm
