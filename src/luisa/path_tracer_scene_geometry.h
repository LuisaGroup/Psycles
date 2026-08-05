#pragma once

#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
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

// A type-erased, non-owning view of the final float position array. Keeping
// this interface in terms of Psycles' host vector type lets the exact support
// planner remain independent of any particular compute runtime container.
struct CyclesPositionArrayView {
    const void *data{};
    std::size_t size{};
    Vec3f (*load)(const void *, std::size_t) noexcept{};

    [[nodiscard]] Vec3f operator[](
        std::size_t index) const noexcept {
        return load(data, index);
    }
};

template<typename Position>
[[nodiscard]] CyclesPositionArrayView
make_cycles_position_array_view(
    std::span<const Position> positions) noexcept {
    return {
        .data = positions.data(),
        .size = positions.size(),
        .load = [](const void *data, std::size_t index) noexcept {
            const auto &point =
                static_cast<const Position *>(data)[index];
            return Vec3f{point.x, point.y, point.z};
        }};
}

[[nodiscard]] std::vector<CyclesInstanceIntersectionPlan>
build_cycles_instance_intersection_plan(
    const contract::SceneSnapshot &scene,
    const std::set<contract::MaterialId> &surface_bssrdf_materials);

// Completes the plan only after every geometry mutation is finished. Geometry
// classes identify bitwise-equal local position and triangle-index arrays;
// instance classes additionally require bitwise-equal vertices after the
// authored affine transform. Keeping this phase separate prevents true
// displacement from invalidating an alias relation derived from the source
// mesh while admitting distinct transforms with the same finite float image.
[[nodiscard]] bool finalize_cycles_instance_intersection_plan(
    const contract::SceneSnapshot &scene,
    const std::map<contract::GeometryId, std::uint32_t>
        &final_triangle_support_classes,
    const std::map<contract::GeometryId, CyclesPositionArrayView>
        &final_positions,
    std::span<CyclesInstanceIntersectionPlan> plan);

// Host forms of Cycles' affine geometry operations. Static-transform vertices
// and object-space rays are uploaded as floats so every device backend sees
// the same two numerical representations used by Cycles' BVH builder.
[[nodiscard]] Vec3f cycles_transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept;

[[nodiscard]] Mat4f cycles_inverse_transform(
    const Mat4f &transform) noexcept;

}// namespace psycles::luisa_backend::detail
