#pragma once

#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

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

// One entry per SceneSnapshot instance, in stable map iteration order. Exact
// coincident triangle supports form a circular list so a device ray query can
// recover source candidates that an acceleration backend legally coalesces.
struct CyclesInstanceIntersectionPlan {
    std::uint32_t coincident_next{};
    std::uint32_t coincident_count{1u};
    bool transform_applied{};
    Mat4f world_to_object;
};

[[nodiscard]] std::vector<CyclesInstanceIntersectionPlan>
build_cycles_instance_intersection_plan(
    const contract::SceneSnapshot &scene,
    const std::set<contract::MaterialId> &surface_bssrdf_materials);

// Host forms of Cycles' affine geometry operations. Static-transform vertices
// and object-space rays are uploaded as floats so every device backend sees
// the same two numerical representations used by Cycles' BVH builder.
[[nodiscard]] Vec3f cycles_transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept;

[[nodiscard]] Mat4f cycles_inverse_transform(
    const Mat4f &transform) noexcept;

}// namespace psycles::luisa_backend::detail
