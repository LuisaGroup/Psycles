#pragma once

#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace psycles::luisa_backend::detail {

enum class CyclesPrimitiveIntervalError : std::uint8_t {
    none,
    overlap,
    out_of_range
};

struct CyclesPrimitiveInterval {
    std::optional<std::uint32_t> offset;
    CyclesPrimitiveIntervalError error{
        CyclesPrimitiveIntervalError::none};
};

// Reconstructs Cycles GeometryManager's flattened primitive address space.
// Explicit BlenderSync prefixes and deterministic renderer-authored prefixes
// obey the same monotone, non-overlapping interval invariant.
class CyclesPrimitiveIntervalResolver {

  private:
    std::uint64_t _end{};

  public:
    [[nodiscard]] CyclesPrimitiveInterval resolve(
        std::size_t triangle_count,
        std::optional<std::uint32_t> explicit_offset =
            std::nullopt) noexcept;

    [[nodiscard]] std::uint64_t end() const noexcept {
        return _end;
    }
};

[[nodiscard]] float world_triangle_area(
    const Mat4f &transform,
    Vec3f p0,
    Vec3f p1,
    Vec3f p2) noexcept;

}// namespace psycles::luisa_backend::detail
