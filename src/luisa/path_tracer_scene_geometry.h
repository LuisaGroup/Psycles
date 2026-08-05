#pragma once

#include <psycles/contract/scene.h>

#include <array>
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
    std::uint32_t coincident_primitive_offset{};
    std::uint32_t coincident_primitive_count{};
    bool transform_applied{};
    Mat4f world_to_object;
};

struct CyclesCoincidentPrimitiveRecord {
    std::uint32_t local_primitive{};
    std::uint32_t instance_offset{};
    std::uint32_t instance_count{};
};

struct CyclesPrimitiveIntersectionPlan {
    std::vector<CyclesCoincidentPrimitiveRecord> records;
    std::vector<std::uint32_t> instances;
};

// Type-erased, non-owning access to the final float triangle support. This
// keeps exact-support planning independent of Luisa's runtime containers while
// retaining the ordered triangle vertices that define Cycles primitive
// identity.
struct CyclesGeometrySupportView {
    const void *position_data{};
    std::size_t position_count{};
    Vec3f (*load_position)(const void *, std::size_t) noexcept{};
    const void *triangle_data{};
    std::size_t triangle_count{};
    std::array<std::uint32_t, 3u> (*load_triangle)(
        const void *, std::size_t) noexcept{};

    [[nodiscard]] Vec3f position(
        std::size_t index) const noexcept {
        return load_position(position_data, index);
    }

    [[nodiscard]] std::array<std::uint32_t, 3u> triangle(
        std::size_t index) const noexcept {
        return load_triangle(triangle_data, index);
    }
};

template<typename Position, typename Triangle>
[[nodiscard]] CyclesGeometrySupportView
make_cycles_geometry_support_view(
    std::span<const Position> positions,
    std::span<const Triangle> triangles) noexcept {
    return {
        .position_data = positions.data(),
        .position_count = positions.size(),
        .load_position = [](const void *data, std::size_t index) noexcept {
            const auto &point =
                static_cast<const Position *>(data)[index];
            return Vec3f{point.x, point.y, point.z};
        },
        .triangle_data = triangles.data(),
        .triangle_count = triangles.size(),
        .load_triangle = [](const void *data, std::size_t index) noexcept {
            const auto &triangle =
                static_cast<const Triangle *>(data)[index];
            if constexpr (requires {
                              triangle.i0;
                              triangle.i1;
                              triangle.i2;
                          }) {
                return std::array<std::uint32_t, 3u>{
                    triangle.i0, triangle.i1, triangle.i2};
            } else {
                return std::array<std::uint32_t, 3u>{
                    triangle[0u], triangle[1u], triangle[2u]};
            }
        }};
}

[[nodiscard]] std::vector<CyclesInstanceIntersectionPlan>
build_cycles_instance_intersection_plan(
    const contract::SceneSnapshot &scene,
    const std::set<contract::MaterialId> &surface_bssrdf_materials);

// Completes the plan only after every geometry mutation is finished. Geometry
// classes identify bitwise-equal local position and triangle-index arrays.
// Whole-instance classes require every transformed vertex to be bitwise equal;
// sparse primitive classes require the three ordered transformed vertices to
// be bitwise equal. Both are exact finite relations. Keeping this phase after
// geometry mutation prevents displacement from invalidating an alias relation
// derived from the source mesh while admitting distinct transforms with the
// same finite float image.
[[nodiscard]] bool finalize_cycles_instance_intersection_plan(
    const contract::SceneSnapshot &scene,
    const std::map<contract::GeometryId, std::uint32_t>
        &final_triangle_support_classes,
    const std::map<contract::GeometryId, CyclesGeometrySupportView>
        &final_supports,
    std::span<CyclesInstanceIntersectionPlan> plan,
    CyclesPrimitiveIntersectionPlan &primitive_plan);

// Host forms of Cycles' affine geometry operations. Static-transform vertices
// and object-space rays are uploaded as floats so every device backend sees
// the same two numerical representations used by Cycles' BVH builder.
[[nodiscard]] Vec3f cycles_transform_point(
    const Mat4f &transform,
    Vec3f point) noexcept;

[[nodiscard]] Mat4f cycles_inverse_transform(
    const Mat4f &transform) noexcept;

}// namespace psycles::luisa_backend::detail
