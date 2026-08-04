#include "curve_ribbon_component.h"

#include <limits>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;

struct QuadIntersection {
    Bool valid;
    Float distance;
    Float u;
    Float v;
};

class CyclesCurveRibbonComponent final : public CurveRibbonComponent {

private:
    [[nodiscard]] QuadIntersection intersect_quad(
        Expr<float> t_min,
        Expr<float> t_max,
        Expr<luisa::float3> lower_begin,
        Expr<luisa::float3> lower_end,
        Expr<luisa::float3> upper_end,
        Expr<luisa::float3> upper_begin) const noexcept {
        constexpr auto ray_direction =
            luisa::make_float3(0.0f, 0.0f, 1.0f);

        const auto opposite_edge = lower_end - upper_begin;
        const auto winding =
            dot(cross(upper_begin, opposite_edge), ray_direction);
        const auto reversed = winding > 0.0f;
        const auto v0 = select(lower_begin, upper_end, reversed);
        const auto v1 = select(lower_end, upper_begin, reversed);
        const auto v2 = select(upper_begin, lower_end, reversed);
        const auto edge0 = v2 - v0;
        const auto edge1 = v0 - v1;
        const auto edge_u = dot(cross(v0, edge0), ray_direction);
        const auto edge_v = dot(cross(v1, edge1), ray_direction);
        const auto normal = cross(edge1, edge0);
        const auto denominator = dot(normal, ray_direction);
        const auto inverse_denominator = 1.0f / denominator;
        const auto distance = dot(v0, normal) * inverse_denominator;
        auto u = edge_u * inverse_denominator;
        auto v = edge_v * inverse_denominator;
        u = select(u, 1.0f - u, reversed);
        v = select(v, 1.0f - v, reversed);
        const auto valid =
            (luisa::compute::max(edge_u, edge_v) <= 0.0f) &
            (distance >= t_min) &
            (distance <= t_max) &
            (denominator != 0.0f);
        return {
            .valid = valid,
            .distance = distance,
            .u = u,
            .v = v};
    }

    [[nodiscard]] static Float maximum_absolute_component(
        Expr<luisa::float4> value) noexcept {
        return luisa::compute::max(
            luisa::compute::max(abs(value.x), abs(value.y)),
            luisa::compute::max(abs(value.z), abs(value.w)));
    }

public:
    Float4 evaluate(
        const CurveControlPoints &curve,
        Expr<float> u) const noexcept override {
        // Catmull-Rom in Horner form. Keeping the polynomial typed lets Luisa
        // fold constants and share expressions during kernel construction.
        const auto cubic =
            (-curve.before + 3.0f * curve.begin -
             3.0f * curve.end + curve.after) *
            0.5f;
        const auto quadratic =
            (2.0f * curve.before - 5.0f * curve.begin +
             4.0f * curve.end - curve.after) *
            0.5f;
        const auto linear =
            (-curve.before + curve.end) * 0.5f;
        return ((cubic * u + quadratic) * u + linear) * u +
               curve.begin;
    }

    Float4 derivative(
        const CurveControlPoints &curve,
        Expr<float> u) const noexcept override {
        const auto cubic =
            (-curve.before + 3.0f * curve.begin -
             3.0f * curve.end + curve.after) *
            0.5f;
        const auto quadratic =
            (2.0f * curve.before - 5.0f * curve.begin +
             4.0f * curve.end - curve.after) *
            0.5f;
        const auto linear =
            (-curve.before + curve.end) * 0.5f;
        return (3.0f * cubic * u + 2.0f * quadratic) * u +
               linear;
    }

    CurveRibbonIntersection intersect(
        const Var<Ray> &object_ray,
        const CurveControlPoints &object_curve,
        Expr<std::uint32_t> subdivision_level)
        const noexcept override {
        const auto ray_direction = object_ray->direction();
        const auto inverse_direction_length =
            1.0f / length(ray_direction);
        const auto unit_direction =
            ray_direction * inverse_direction_length;
        const auto basis_candidate_0 = make_float3(
            0.0f, unit_direction.z, -unit_direction.y);
        const auto basis_candidate_1 = make_float3(
            -unit_direction.z, 0.0f, unit_direction.x);
        const auto basis_x = normalize(select(
            basis_candidate_1,
            basis_candidate_0,
            dot(basis_candidate_0, basis_candidate_0) >
                dot(basis_candidate_1, basis_candidate_1)));
        const auto basis_y = normalize(cross(unit_direction, basis_x));
        // This basis vector converts object-space depth back to the original
        // ray parameter even when the direction is not unit length.
        const auto basis_z =
            unit_direction * inverse_direction_length;
        const auto ray_origin = object_ray->origin();
        const auto to_ray_space =
            [&](Expr<luisa::float4> key) noexcept {
                const auto relative = key.xyz() - ray_origin;
                return make_float4(
                    dot(basis_x, relative),
                    dot(basis_y, relative),
                    dot(basis_z, relative),
                    key.w);
            };
        const CurveControlPoints curve{
            .before = to_ray_space(object_curve.before),
            .begin = to_ray_space(object_curve.begin),
            .end = to_ray_space(object_curve.end),
            .after = to_ray_space(object_curve.after)};

        const auto scale = luisa::compute::max(
            luisa::compute::max(
                maximum_absolute_component(curve.before),
                maximum_absolute_component(curve.begin)),
            luisa::compute::max(
                maximum_absolute_component(curve.end),
                maximum_absolute_component(curve.after)));
        const auto derivative_epsilon =
            4.0f * std::numeric_limits<float>::epsilon() * scale;
        const UInt interval_count =
            1u << luisa::compute::min(subdivision_level, 4u);
        const Float interval_size =
            1.0f / cast<float>(interval_count);

        Float4 interval_begin = evaluate(curve, 0.0f);
        Float3 begin_derivative = derivative(curve, 0.0f).xyz();
        const auto first_probe = evaluate(curve, interval_size);
        const auto begin_degenerate =
            luisa::compute::max(
                luisa::compute::max(
                    abs(begin_derivative.x),
                    abs(begin_derivative.y)),
                abs(begin_derivative.z)) < derivative_epsilon;
        begin_derivative = select(
            begin_derivative,
            (first_probe - interval_begin).xyz(),
            begin_degenerate);
        Float3 begin_width =
            normalize(make_float3(
                begin_derivative.y,
                -begin_derivative.x,
                0.0f)) *
            interval_begin.w;

        Bool found = false;
        Float hit_distance = object_ray->t_max();
        Float hit_u = 0.0f;
        Float hit_v = 0.0f;
        for (std::uint32_t interval = 0u;
             interval < 16u;
             ++interval) {
            $if((interval < interval_count) & !found) {
                const auto interval_u =
                    cast<float>(interval) * interval_size;
                const auto interval_end =
                    evaluate(curve, interval_u + interval_size);
                auto end_derivative =
                    derivative(
                        curve,
                        interval_u + interval_size)
                        .xyz();
                const auto end_degenerate =
                    luisa::compute::max(
                        luisa::compute::max(
                            abs(end_derivative.x),
                            abs(end_derivative.y)),
                        abs(end_derivative.z)) < derivative_epsilon;
                end_derivative = select(
                    end_derivative,
                    (interval_end - interval_begin).xyz(),
                    end_degenerate);
                const auto end_width =
                    normalize(make_float3(
                        end_derivative.y,
                        -end_derivative.x,
                        0.0f)) *
                    interval_end.w;

                const auto projected_delta =
                    interval_end.xy() - interval_begin.xy();
                const auto culling_numerator =
                    projected_delta.x * interval_begin.y -
                    projected_delta.y * interval_begin.x;
                const auto culling_radius =
                    luisa::compute::max(
                        interval_begin.w, interval_end.w);
                const auto overlaps_ray =
                    culling_numerator * culling_numerator <=
                    culling_radius * culling_radius *
                        dot(projected_delta, projected_delta);

                const auto lower_begin =
                    interval_begin.xyz() + begin_width;
                const auto lower_end =
                    interval_end.xyz() + end_width;
                const auto upper_begin =
                    interval_begin.xyz() - begin_width;
                const auto upper_end =
                    interval_end.xyz() - end_width;
                const auto quad = intersect_quad(
                    object_ray->t_min(),
                    object_ray->t_max(),
                    lower_begin,
                    lower_end,
                    upper_end,
                    upper_begin);
                const auto radius =
                    interval_begin.w +
                    (interval_end.w - interval_begin.w) * quad.u;
                // Cycles' ribbon contract is a strict 2r/|D| exclusion. It
                // is independent of any floating-point origin epsilon.
                const auto avoids_self =
                    quad.distance >
                    2.0f * radius * inverse_direction_length;
                $if(overlaps_ray & quad.valid & avoids_self) {
                    found = true;
                    hit_distance = quad.distance;
                    hit_u = interval_u + quad.u * interval_size;
                    hit_v = 2.0f * quad.v - 1.0f;
                };
                interval_begin = interval_end;
                begin_width = end_width;
            };
        }
        return {
            .valid = found,
            .distance = hit_distance,
            .u = hit_u,
            .v = hit_v};
    }
};

}// namespace

std::shared_ptr<const CurveRibbonComponent>
make_curve_ribbon_component() {
    return std::make_shared<CyclesCurveRibbonComponent>();
}

}// namespace psycles::luisa_backend::detail
