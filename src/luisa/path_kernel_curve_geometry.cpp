#include "path_kernel_curve_geometry.h"

#include "curve_ribbon_component.h"

#include <cmath>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class CyclesCurveGeometryComponent final : public CurveGeometryComponent {

private:
  std::shared_ptr<const CurvePrimitiveComponent> _primitive;
  std::shared_ptr<const CurveRibbonComponent> _ribbon;

public:
  CyclesCurveGeometryComponent()
      : _primitive{make_curve_primitive_component()},
        _ribbon{make_curve_ribbon_component()} {}

  CurveGeometryContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       Expr<std::uint32_t> instance_id, Expr<std::uint32_t> segment_id,
       const Var<luisa::compute::Ray> &world_ray,
       Expr<float> committed_distance) const noexcept override {
    auto primitive = _primitive->emit(scene, instance_id, segment_id);
    const auto object_to_world = scene->accel->instance_transform(instance_id);
    const auto world_to_object =
        primitive.curve.metadata.instance.cycles_world_to_object;
    const auto normal_to_world = transpose(world_to_object);
    const auto object_origin =
        (world_to_object * make_float4(world_ray->origin(), 1.0f)).xyz();
    const auto object_direction =
        (world_to_object * make_float4(world_ray->direction(), 0.0f)).xyz();
    const auto object_ray = make_ray(object_origin, object_direction,
                                     world_ray->t_min(), world_ray->t_max());
    auto intersection = _ribbon->intersect(
        object_ray, primitive.curve.control_points,
        primitive.curve.metadata.geometry.curve_subdivision_level);

    // curve_shader_setup transforms D * t to object space and then
    // safe_normalize_len() updates the object-space distance.
    const auto object_segment =
        (world_to_object *
         make_float4(world_ray->direction() * committed_distance, 0.0f))
            .xyz();
    const auto object_distance = length(object_segment);
    const auto unit_object_direction =
        select(object_segment, object_segment / object_distance,
               object_distance != 0.0f);
    const auto object_position =
        object_origin + unit_object_direction * object_distance;
    const auto object_dpdu4 =
        _ribbon->derivative(primitive.curve.control_points, intersection.u);
    const auto object_dpdu = object_dpdu4.xyz();
    const auto tangent = normalize(object_dpdu);
    const auto bitangent = normalize(cross(tangent, -unit_object_direction));
    const auto sine = intersection.v;
    const auto cosine = sqrt(max(1.0f - sine * sine, 0.0f));
    const auto object_shading_normal = normalize(
        sine * bitangent - cosine * normalize(cross(tangent, bitangent)));
    const auto position =
        (object_to_world * make_float4(object_position, 1.0f)).xyz();
    const auto shading_normal = normalize(
        (normal_to_world * make_float4(object_shading_normal, 0.0f)).xyz());
    const auto dpdu = (object_to_world * make_float4(object_dpdu, 0.0f)).xyz();
    const auto incoming = -world_ray->direction();
    const auto geometric_normal = incoming;
    const auto dpdv = cross(dpdu, geometric_normal);

    Float2 uv = make_float2(0.0f);
    $if ((primitive.curve.metadata.geometry.attribute_domains &
          geometry_curve_default_uv) != 0u) {
      uv = scene->heap
               ->buffer<luisa::float2>(
                   primitive.curve.metadata.geometry.bindless_base + 2u)
               .read(primitive.curve.metadata.segment.curve_index);
    };

    const auto intercepts = scene->heap->buffer<float>(
        primitive.curve.metadata.geometry.bindless_base + 3u);
    const auto intercept_begin =
        intercepts.read(primitive.curve.metadata.segment.key_begin);
    const auto intercept_end =
        intercepts.read(primitive.curve.metadata.segment.key_end);
    const auto intercept = lerp(intercept_begin, intercept_end, intersection.u);
    const auto curve_length =
        scene->heap
            ->buffer<float>(primitive.curve.metadata.geometry.bindless_base +
                            5u)
            .read(primitive.curve.metadata.segment.curve_index);
    const auto curve_random =
        scene->heap
            ->buffer<float>(primitive.curve.metadata.geometry.bindless_base +
                            6u)
            .read(primitive.curve.metadata.segment.curve_index);

    // curve_thickness uses linear radius interpolation between the two
    // source keys, independently of the Catmull-Rom intersection radius.
    const auto object_diameter =
        2.0f * lerp(primitive.curve.control_points.begin.w,
                    primitive.curve.control_points.end.w, intersection.u);
    constexpr auto inverse_sqrt_three = 0.57735026918962576451f;
    const auto object_radius_diagonal =
        make_float3(object_diameter * inverse_sqrt_three);
    const auto thickness = length(
        (object_to_world * make_float4(object_radius_diagonal, 0.0f)).xyz());
    const auto tangent_normal =
        normalize(incoming - dpdu * (dot(dpdu, incoming) / dot(dpdu, dpdu)));

    return {.primitive = std::move(primitive),
            .intersection = std::move(intersection),
            .object_position = std::move(object_position),
            .position = std::move(position),
            .object_shading_normal = std::move(object_shading_normal),
            .shading_normal = std::move(shading_normal),
            .geometric_normal = std::move(geometric_normal),
            .object_dpdu = std::move(object_dpdu),
            .dpdu = std::move(dpdu),
            .dpdv = std::move(dpdv),
            .uv = std::move(uv),
            .intercept = std::move(intercept),
            .length = std::move(curve_length),
            .thickness = std::move(thickness),
            .tangent_normal = std::move(tangent_normal),
            .random = std::move(curve_random)};
  }
};

} // namespace

std::shared_ptr<const CurveGeometryComponent> make_curve_geometry_component() {
  return std::make_shared<CyclesCurveGeometryComponent>();
}

} // namespace psycles::luisa_backend::detail
