#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_ray.h> through the Psycles::luisa target."
#endif

#include <cstdint>
#include <limits>

#include <luisa/dsl/rtx/accel.h>
#include <luisa/dsl/rtx/ray.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::surface_ray {

inline constexpr auto invalid_primitive =
    std::numeric_limits<std::uint32_t>::max();

struct ShadowOrigin {
    luisa::compute::Float3 position;
    luisa::compute::Bool skip_self;
};

// Cycles advances tmin for transparent continuation rays by one
// representable float while leaving the ray origin and direction unchanged.
// The zero case deliberately returns the smallest *normal* float instead of
// a denormal, matching intersection_t_offset in kernel/bvh/util.h.
// Precondition: distance is finite and non-negative.
[[nodiscard]] inline luisa::compute::Float
intersection_t_offset(
    luisa::compute::Float distance) noexcept {
    const auto next =
        luisa::compute::as<float>(
            luisa::compute::as<std::uint32_t>(distance) +
            1u);
    return luisa::compute::select(
        next,
        std::numeric_limits<float>::min(),
        distance == 0.0f);
}

// Classify whether a ray starting on a source triangle would intersect that
// exact triangle at its origin. The three Pluecker edge functions are an
// orientation-independent containment certificate: an interior origin gives
// all three the same strict sign. The scaled epsilon deliberately classifies
// edge/vertex ambiguity as "not certified", so the robust offset remains the
// conservative fallback for neighboring-triangle precision hazards.
[[nodiscard]] inline luisa::compute::Bool
source_triangle_covers_ray_origin(
    luisa::compute::Float3 object_origin,
    luisa::compute::Float3 object_direction,
    luisa::compute::Float3 p0,
    luisa::compute::Float3 p1,
    luisa::compute::Float3 p2) noexcept {
    const auto v0 = p0 - object_origin;
    const auto v1 = p1 - object_origin;
    const auto v2 = p2 - object_origin;
    const auto e0 = v2 - v0;
    const auto e1 = v0 - v1;
    const auto e2 = v1 - v2;
    const auto u =
        dot(cross(v2 + v0, e0), object_direction);
    const auto v =
        dot(cross(v0 + v1, e1), object_direction);
    const auto w =
        dot(cross(v1 + v2, e2), object_direction);
    const auto epsilon =
        std::numeric_limits<float>::epsilon() *
        abs(u + v + w);
    const auto minimum = min(u, min(v, w));
    const auto maximum = max(u, max(v, w));
    return (minimum >= epsilon) |
           (maximum <= -epsilon);
}

// A source primitive that can be rejected explicitly during traversal does
// not need its origin displaced when the exact triangle test above certifies
// self intersection. Preserving the hit point is important for dense,
// sub-millimeter geometry: an unconditional world-space ULP offset can cross
// nearby sheets and erase legitimate occlusion. Ambiguous edge cases retain
// Luisa's robust geometric-normal offset.
[[nodiscard]] inline luisa::compute::Float3
origin_with_explicit_self_exclusion(
    luisa::compute::Float3 world_origin,
    luisa::compute::Float3 world_geometric_normal,
    luisa::compute::Float3 object_origin,
    luisa::compute::Float3 object_direction,
    luisa::compute::Float3 p0,
    luisa::compute::Float3 p1,
    luisa::compute::Float3 p2) noexcept {
    const auto exact_origin =
        source_triangle_covers_ray_origin(
            object_origin,
            object_direction,
            p0,
            p1,
            p2);
    return select(
        luisa::compute::offset_ray_origin(
            world_origin,
            world_geometric_normal),
        world_origin,
        exact_origin);
}

// Cycles' parabolic smooth-surface construction from
// kernel/light/sample.h. Vertex positions and normals are in object space;
// the unnormalized interpolated normal is transformed as a direction so its
// scale carries the object-to-world distance conversion exactly as Cycles
// does.
[[nodiscard]] inline luisa::compute::Float3
smooth_surface_offset(
    luisa::compute::Float4x4 object_to_world,
    luisa::compute::Float2 barycentric,
    luisa::compute::Float3 p0,
    luisa::compute::Float3 p1,
    luisa::compute::Float3 p2,
    luisa::compute::Float3 n0,
    luisa::compute::Float3 n1,
    luisa::compute::Float3 n2,
    luisa::compute::Float3 world_geometric_normal) noexcept {
    const auto u =
        1.0f - barycentric.x - barycentric.y;
    const auto v = barycentric.x;
    const auto w = barycentric.y;
    const auto object_position =
        p0 * u + p1 * v + p2 * w;
    const auto object_normal =
        n0 * u + n1 * v + n2 * w;
    const auto world_normal =
        (object_to_world *
         luisa::compute::make_float4(
             object_normal, 0.0f))
            .xyz();

    const auto a = dot(n2 - n0, p0 - p2);
    const auto b = dot(n2 - n1, p1 - p2);
    const auto c = dot(n1 - n0, p1 - p0);
    const auto parabolic_height =
        a * u * (u - 1.0f) +
        (a + b + c) * u * v +
        b * v * (v - 1.0f);

    auto front_h0 = max(
        max(dot(p1 - p0, n0), dot(p2 - p0, n0)),
        0.0f);
    auto front_h1 = max(
        max(dot(p0 - p1, n1), dot(p2 - p1, n1)),
        0.0f);
    auto front_h2 = max(
        max(dot(p0 - p2, n2), dot(p1 - p2, n2)),
        0.0f);
    front_h0 = max(
        dot(p0 - object_position, n0) + front_h0,
        0.0f);
    front_h1 = max(
        dot(p1 - object_position, n1) + front_h1,
        0.0f);
    front_h2 = max(
        dot(p2 - object_position, n2) + front_h2,
        0.0f);
    const auto front_height = max(
        min(front_h0, min(front_h1, front_h2)),
        parabolic_height * 0.5f);

    auto back_h0 = max(
        max(dot(p0 - p1, n0), dot(p0 - p2, n0)),
        0.0f);
    auto back_h1 = max(
        max(dot(p1 - p0, n1), dot(p1 - p2, n1)),
        0.0f);
    auto back_h2 = max(
        max(dot(p2 - p0, n2), dot(p2 - p1, n2)),
        0.0f);
    back_h0 = max(
        dot(object_position - p0, n0) + back_h0,
        0.0f);
    back_h1 = max(
        dot(object_position - p1, n1) + back_h1,
        0.0f);
    back_h2 = max(
        dot(object_position - p2, n2) + back_h2,
        0.0f);
    const auto back_height = min(
        -min(back_h0, min(back_h1, back_h2)),
        parabolic_height * 0.5f);

    const auto height = luisa::compute::select(
        back_height,
        front_height,
        dot(world_normal, world_geometric_normal) >
            0.0f);
    return world_normal * height;
}

// Apply Cycles' shadow-terminator geometry offset and return whether the
// source primitive remains eligible for exact self exclusion. The cutoff is
// a dimensionless angular threshold; zero disables the construction.
[[nodiscard]] inline ShadowOrigin
shadow_terminator_origin(
    luisa::compute::Float3 world_position,
    luisa::compute::Float3 world_shading_normal,
    luisa::compute::Float3 world_geometric_normal,
    luisa::compute::Float3 light_direction,
    luisa::compute::Float geometry_offset,
    luisa::compute::Bool smooth_triangle,
    luisa::compute::Float4x4 object_to_world,
    luisa::compute::Float2 barycentric,
    luisa::compute::Float3 p0,
    luisa::compute::Float3 p1,
    luisa::compute::Float3 p2,
    luisa::compute::Float3 n0,
    luisa::compute::Float3 n1,
    luisa::compute::Float3 n2) noexcept {
    auto normal_light = dot(
        world_shading_normal, light_direction);
    const auto transmit = normal_light < 0.0f;
    normal_light = abs(normal_light);
    const auto offset_normal = luisa::compute::select(
        world_geometric_normal,
        -world_geometric_normal,
        transmit);
    const auto geometric_light =
        dot(offset_normal, light_direction);
    const auto safe_cutoff = max(
        geometry_offset, 1.0e-20f);
    const auto near_terminator = clamp(
        2.0f -
            (geometric_light + normal_light) /
                safe_cutoff,
        0.0f,
        1.0f);
    const auto regular = clamp(
        1.0f - geometric_light / safe_cutoff,
        0.0f,
        1.0f);
    const auto amount = luisa::compute::select(
        regular,
        near_terminator,
        normal_light < geometry_offset);
    const auto active =
        smooth_triangle &
        (geometry_offset > 0.0f) &
        (amount > 0.0f);
    const auto offset = smooth_surface_offset(
        object_to_world,
        barycentric,
        p0,
        p1,
        p2,
        n0,
        n1,
        n2,
        offset_normal);
    return {
        .position = luisa::compute::select(
            world_position,
            world_position + offset * amount,
            active),
        .skip_self = luisa::compute::select(
            luisa::compute::Bool{true},
            geometric_light > 0.0f,
            active)};
}

// Cycles constructs direct-light shadow origins in two formal stages:
// shadow_ray_offset may move a smooth surface and decide whether the source
// primitive remains explicitly excluded; when exclusion remains active,
// integrate_surface_ray_offset tests the exact source triangle and applies a
// robust geometric-normal ULP offset only if a neighboring triangle could be
// hit instead. Keeping the stages together prevents flat and disabled-smooth
// paths from accidentally skipping the neighboring-triangle certificate.
[[nodiscard]] inline ShadowOrigin surface_shadow_origin(
    luisa::compute::Float3 world_position,
    luisa::compute::Float3 world_shading_normal,
    luisa::compute::Float3 world_geometric_normal,
    luisa::compute::Float3 light_direction,
    luisa::compute::Float geometry_offset,
    luisa::compute::Bool smooth_triangle,
    luisa::compute::Float4x4 object_to_world,
    luisa::compute::Float4x4 world_to_object,
    luisa::compute::Float2 barycentric,
    luisa::compute::Float3 p0,
    luisa::compute::Float3 p1,
    luisa::compute::Float3 p2,
    luisa::compute::Float3 n0,
    luisa::compute::Float3 n1,
    luisa::compute::Float3 n2) noexcept {
    auto origin = shadow_terminator_origin(
        world_position,
        world_shading_normal,
        world_geometric_normal,
        light_direction,
        geometry_offset,
        smooth_triangle,
        object_to_world,
        barycentric,
        p0,
        p1,
        p2,
        n0,
        n1,
        n2);
    const auto object_origin =
        (world_to_object *
         luisa::compute::make_float4(
             origin.position, 1.0f))
            .xyz();
    const auto object_direction =
        (world_to_object *
         luisa::compute::make_float4(
             light_direction, 0.0f))
            .xyz();
    const auto offset_origin =
        origin_with_explicit_self_exclusion(
            origin.position,
            world_geometric_normal,
            object_origin,
            object_direction,
            p0,
            p1,
            p2);
    origin.position = luisa::compute::select(
        origin.position,
        offset_origin,
        origin.skip_self);
    return origin;
}

[[nodiscard]] inline luisa::compute::Bool
same_primitive(
    luisa::compute::Expr<luisa::uint> instance,
    luisa::compute::Expr<luisa::uint> primitive,
    luisa::compute::Expr<luisa::uint> other_instance,
    luisa::compute::Expr<luisa::uint> other_primitive) noexcept {
    return (instance == other_instance) &
           (primitive == other_primitive);
}

[[nodiscard]] inline luisa::compute::Bool
excluded_shadow_primitive(
    luisa::compute::Expr<luisa::uint> instance,
    luisa::compute::Expr<luisa::uint> primitive,
    luisa::compute::Expr<luisa::uint> source_instance,
    luisa::compute::Expr<luisa::uint> source_primitive,
    luisa::compute::Expr<luisa::uint> light_instance,
    luisa::compute::Expr<luisa::uint> light_primitive) noexcept {
    const auto source = same_primitive(
        instance,
        primitive,
        source_instance,
        source_primitive);
    const auto light = same_primitive(
        instance,
        primitive,
        light_instance,
        light_primitive);
    return source | light;
}

// Return the closest eligible shadow intersection. Ray-query candidate
// callbacks have no traversal-order contract; committing every eligible
// candidate lets the acceleration backend perform only the order-independent
// minimum-t reduction. Callers can iterate closest-to-farthest by advancing
// tmin with intersection_t_offset after each committed hit.
template<typename Accel>
[[nodiscard]] inline auto
closest_shadow_intersection(
    Accel &&accel,
    luisa::compute::Expr<luisa::compute::Ray> ray,
    luisa::compute::Expr<luisa::uint> source_instance,
    luisa::compute::Expr<luisa::uint> source_primitive,
    luisa::compute::Expr<luisa::uint> light_instance,
    luisa::compute::Expr<luisa::uint> light_primitive,
    luisa::compute::Expr<luisa::uint> visibility_mask) noexcept {
    return accel
        ->traverse(
            ray,
            {.visibility_mask = visibility_mask})
        .on_surface_candidate(
            [&](luisa::compute::SurfaceCandidate
                    &candidate) noexcept {
                const auto hit = candidate.hit();
                $if (!excluded_shadow_primitive(
                    hit->inst,
                    hit->prim,
                    source_instance,
                    source_primitive,
                    light_instance,
                    light_primitive)) {
                    candidate.commit();
                };
            })
        .on_procedural_candidate(
            [](luisa::compute::ProceduralCandidate &) noexcept {})
        .trace();
}

} // namespace psycles::luisa_backend::surface_ray
