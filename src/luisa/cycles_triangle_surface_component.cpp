#include "cycles_triangle_surface_component.h"

namespace psycles::luisa_backend::detail {
namespace {

class FinalSupportTriangleSurfaceComponent final
    : public CyclesTriangleSurfaceComponent {

public:
  CyclesTriangleSurface
  resolve(const CyclesTriangleSurfaceInput &input,
          const SafeNormalizeCallable &safe_normalize) const noexcept override {
    const auto p0 = select(input.p0, input.final_p0, input.transform_applied);
    const auto p1 = select(input.p1, input.final_p1, input.transform_applied);
    const auto p2 = select(input.p2, input.final_p2, input.transform_applied);

    const auto transformed_p0 =
        (input.object_to_world * make_float4(input.p0, 1.0f)).xyz();
    const auto transformed_p1 =
        (input.object_to_world * make_float4(input.p1, 1.0f)).xyz();
    const auto transformed_p2 =
        (input.object_to_world * make_float4(input.p2, 1.0f)).xyz();
    const auto world_p0 = select(transformed_p0, p0, input.transform_applied);
    const auto world_p1 = select(transformed_p1, p1, input.transform_applied);
    const auto world_p2 = select(transformed_p2, p2, input.transform_applied);

    const auto object_geometric_normal =
        safe_normalize(cross(input.p1 - input.p0, input.p2 - input.p0),
                       make_float3(0.0f, 0.0f, 1.0f));
    const auto transformed_geometric_normal = safe_normalize(
        (input.normal_to_world * make_float4(object_geometric_normal, 0.0f))
            .xyz(),
        -input.ray_direction);
    auto support_geometric_normal =
        safe_normalize(cross(p1 - p0, p2 - p0), transformed_geometric_normal);
    support_geometric_normal = select(
        support_geometric_normal, -support_geometric_normal,
        dot(support_geometric_normal, transformed_geometric_normal) < 0.0f);
    auto geometric_normal =
        select(transformed_geometric_normal, support_geometric_normal,
               input.transform_applied);

    const auto world_n0 = safe_normalize(
        (input.normal_to_world * make_float4(input.n0, 0.0f)).xyz(),
        geometric_normal);
    const auto world_n1 = safe_normalize(
        (input.normal_to_world * make_float4(input.n1, 0.0f)).xyz(),
        geometric_normal);
    const auto world_n2 = safe_normalize(
        (input.normal_to_world * make_float4(input.n2, 0.0f)).xyz(),
        geometric_normal);
    const auto n0 = select(input.n0, world_n0, input.transform_applied);
    const auto n1 = select(input.n1, world_n1, input.transform_applied);
    const auto n2 = select(input.n2, world_n2, input.transform_applied);

    const auto object_shading_normal = select(
        object_geometric_normal,
        triangle_interpolate(input.barycentric, input.n0, input.n1, input.n2),
        input.smooth);
    const auto transformed_shading_normal = safe_normalize(
        (input.normal_to_world * make_float4(object_shading_normal, 0.0f))
            .xyz(),
        geometric_normal);
    const auto support_shading_normal = safe_normalize(
        select(geometric_normal,
               triangle_interpolate(input.barycentric, n0, n1, n2),
               input.smooth),
        geometric_normal);
    auto shading_normal =
        select(transformed_shading_normal, support_shading_normal,
               input.transform_applied);

    const auto back_facing = dot(geometric_normal, -input.ray_direction) < 0.0f;
    geometric_normal = select(geometric_normal, -geometric_normal, back_facing);
    shading_normal = select(shading_normal, -shading_normal, back_facing);
    shading_normal = select(shading_normal, -shading_normal,
                            dot(shading_normal, geometric_normal) < 0.0f);

    const auto object_position = input.p0 +
                                 input.barycentric.x * (input.p1 - input.p0) +
                                 input.barycentric.y * (input.p2 - input.p0);
    const auto transformed_position =
        (input.object_to_world * make_float4(object_position, 1.0f)).xyz();
    const auto support_position =
        p0 + input.barycentric.x * (p1 - p0) + input.barycentric.y * (p2 - p0);
    const auto position =
        select(transformed_position, support_position, input.transform_applied);

    return {.p0 = p0,
            .p1 = p1,
            .p2 = p2,
            .n0 = n0,
            .n1 = n1,
            .n2 = n2,
            .world_p0 = world_p0,
            .world_p1 = world_p1,
            .world_p2 = world_p2,
            .object_position = object_position,
            .position = position,
            .object_geometric_normal = object_geometric_normal,
            .geometric_normal = geometric_normal,
            .object_shading_normal = object_shading_normal,
            .shading_normal = shading_normal,
            .back_facing = back_facing};
  }
};

} // namespace

std::shared_ptr<const CyclesTriangleSurfaceComponent>
make_cycles_triangle_surface_component() {
  return std::make_shared<FinalSupportTriangleSurfaceComponent>();
}

} // namespace psycles::luisa_backend::detail
