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

// Exact host image of every MaterialId dereference in one immutable scene.
// For each instanced geometry G, surface_by_geometry[G] is
// image(R_G,I o S_G) unioned over its instances I. surface_materials is the
// union of those images. shader_materials adds the only non-primitive roots:
// analytic-light shaders and the world shader. Consequently compiling exactly
// shader_materials is complete, while adding or editing a material outside
// this domain cannot affect any renderer observation.
struct SceneMaterialReachability {
    std::map<
        contract::GeometryId,
        std::set<contract::MaterialId>>
        surface_by_geometry;
    std::set<contract::MaterialId> surface_materials;
    std::set<contract::MaterialId> shader_materials;
};

[[nodiscard]] SceneMaterialReachability
build_scene_material_reachability(
    const contract::SceneSnapshot &scene);

// Computes the exact host-side image of material bindings that a surface
// primitive can produce. World and analytic-light shaders are intentionally
// excluded: they cannot enter surface transport. Primitive slot selection,
// instance overrides, and Cycles' last-slot clamp all mirror the device-side
// PrimitiveMaterialComponent contract.
[[nodiscard]] std::set<contract::MaterialId>
collect_reachable_surface_materials(
    const contract::SceneSnapshot &scene);

// Returns stable SceneSnapshot instance ordinals whose triangle support may
// enter BSSRDF transport. Material resolution is evaluated per primitive,
// including instance overrides and Cycles' last-slot clamp. Once any surface
// primitive selects a target material, the complete object is retained: a
// local BSSRDF ray may exit through a different primitive or material slot on
// that same object. Curves are excluded because Cycles' local intersection
// contract accepts triangle and motion-triangle objects only.
[[nodiscard]] std::vector<std::uint32_t>
collect_triangle_instances_with_surface_materials(
    const contract::SceneSnapshot &scene,
    const std::set<contract::MaterialId> &materials);

// One entry per SceneSnapshot instance, in stable map iteration order. This
// contains only the representation chosen for the actual accelerator; it does
// not manufacture traversal candidates outside the backend BVH result.
struct CyclesInstanceIntersectionPlan {
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
