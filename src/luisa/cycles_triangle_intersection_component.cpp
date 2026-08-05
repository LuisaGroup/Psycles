#include "cycles_triangle_intersection_component.h"

#include <psycles/luisa/cycles_transform.h>

#include <limits>

namespace psycles::luisa_backend::detail {
namespace {

// These fused operation trees are part of the intersection contract. Cycles
// uses the corresponding madd/msub ordering for its Embree-compatible path;
// leaving contraction or matrix-vector lowering implicit lets LLVM and SPIR-V
// disagree on the last bits and can change which coincident material wins.
[[nodiscard]] Float3 pluecker_cross(
    Expr<luisa::float3> a,
    Expr<luisa::float3> b) noexcept {
    return make_float3(
        luisa::compute::fma(a.y, b.z, -(a.z * b.y)),
        luisa::compute::fma(a.z, b.x, -(a.x * b.z)),
        luisa::compute::fma(a.x, b.y, -(a.y * b.x)));
}

[[nodiscard]] Float pluecker_dot(
    Expr<luisa::float3> a,
    Expr<luisa::float3> b) noexcept {
    return luisa::compute::fma(
        a.x, b.x,
        luisa::compute::fma(a.y, b.y, a.z * b.z));
}

[[nodiscard]] Float clamp_bvh_direction(Expr<float> value) noexcept {
    constexpr auto minimum_magnitude = 8.271806e-25f;
    return select(
        copysign(minimum_magnitude, value),
        value,
        abs(value) > minimum_magnitude);
}

class PlueckerTriangleIntersectionComponent final
    : public CyclesTriangleIntersectionComponent {

  public:
    CyclesTriangleIntersection intersect(
        const Var<luisa::compute::Ray> &world_ray,
        Expr<luisa::float4x4> world_to_object,
        Expr<std::uint32_t> transform_applied,
        Expr<luisa::float3> p0,
        Expr<luisa::float3> p1,
        Expr<luisa::float3> p2) const noexcept override {
        const auto object_origin = cycles_transform::point(
            world_to_object, world_ray->origin());
        const auto object_direction = cycles_transform::direction(
            world_to_object, world_ray->direction());
        const Bool positions_are_world = transform_applied != 0u;
        const Float3 origin = select(
            object_origin, world_ray->origin(), positions_are_world);
        const Float3 raw_direction = select(
            object_direction,
            world_ray->direction(),
            positions_are_world);
        const Float3 direction = make_float3(
            clamp_bvh_direction(raw_direction.x),
            clamp_bvh_direction(raw_direction.y),
            clamp_bvh_direction(raw_direction.z));

        const Float3 v0 = p0 - origin;
        const Float3 v1 = p1 - origin;
        const Float3 v2 = p2 - origin;
        const Float3 e0 = v2 - v0;
        const Float3 e1 = v0 - v1;
        const Float3 e2 = v1 - v2;
        const Float u = pluecker_dot(
            pluecker_cross(e0, v2 + v0), direction);
        const Float v = pluecker_dot(
            pluecker_cross(e1, v0 + v1), direction);
        const Float w = pluecker_dot(
            pluecker_cross(e2, v1 + v2), direction);
        const Float sum = u + v + w;
        const Float edge_tolerance =
            std::numeric_limits<float>::epsilon() * abs(sum);
        const Float minimum = min(u, min(v, w));
        const Float maximum = max(u, max(v, w));
        const Bool inside =
            (minimum >= -edge_tolerance) |
            (maximum <= edge_tolerance);

        const Float3 double_normal =
            2.0f * pluecker_cross(e1, e0);
        const Float denominator =
            pluecker_dot(double_normal, direction);
        const Float safe_denominator =
            select(1.0f, denominator, denominator != 0.0f);
        const Float distance =
            pluecker_dot(v0, double_normal) / safe_denominator;
        const Bool in_range =
            (distance >= world_ray->t_min()) &
            (distance <= world_ray->t_max());
        const Float reciprocal_sum = select(
            0.0f,
            1.0f / sum,
            abs(sum) >= 1.0e-18f);
        return {
            .valid = inside & (denominator != 0.0f) & in_range,
            .distance = distance,
            .barycentric = make_float2(
                min(u * reciprocal_sum, 1.0f),
                min(v * reciprocal_sum, 1.0f))};
    }
};

}// namespace

std::shared_ptr<const CyclesTriangleIntersectionComponent>
make_cycles_triangle_intersection_component() {
    return std::make_shared<
        PlueckerTriangleIntersectionComponent>();
}

}// namespace psycles::luisa_backend::detail
