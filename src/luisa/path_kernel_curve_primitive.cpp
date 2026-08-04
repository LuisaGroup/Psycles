#include "path_kernel_curve_primitive.h"

#include "path_kernel_primitive_material.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class BindlessCurvePrimitiveComponent final : public CurvePrimitiveComponent {

private:
  std::shared_ptr<const PrimitiveMaterialComponent> _material;

public:
  BindlessCurvePrimitiveComponent()
      : _material{make_primitive_material_component()} {}

  CurveSegmentContext
  emit_segment(const std::shared_ptr<LuisaSceneData> &scene,
               Expr<std::uint32_t> instance_id,
               Expr<std::uint32_t> segment_id) const noexcept override {
    UInt resolved_instance_id = instance_id;
    UInt resolved_segment_id = segment_id;
    Var<InstanceGpu> instance =
        scene->instance_buffer->read(resolved_instance_id);
    Var<GeometryGpu> geometry =
        scene->geometry_buffer->read(instance.geometry_index);
    Var<CurveSegmentGpu> segment =
        scene->heap->buffer<CurveSegmentGpu>(geometry.bindless_base)
            .read(resolved_segment_id);
    const auto keys =
        scene->heap->buffer<luisa::float4>(geometry.bindless_base + 1u);
    CurveControlPoints control_points{.before = keys.read(segment.key_before),
                                      .begin = keys.read(segment.key_begin),
                                      .end = keys.read(segment.key_end),
                                      .after = keys.read(segment.key_after)};
    return {.instance_id = std::move(resolved_instance_id),
            .segment_id = std::move(resolved_segment_id),
            .instance = std::move(instance),
            .geometry = std::move(geometry),
            .segment = std::move(segment),
            .control_points = std::move(control_points)};
  }

  CurvePrimitiveContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       Expr<std::uint32_t> instance_id,
       Expr<std::uint32_t> segment_id) const noexcept override {
    auto curve = emit_segment(scene, instance_id, segment_id);
    UInt material_slot =
        scene->heap->buffer<luisa::uint>(curve.geometry.bindless_base + 4u)
            .read(curve.segment.curve_index);
    // Cycles curve shader IDs do not carry SHADER_SMOOTH_NORMAL; ribbon
    // smoothing is reconstructed geometrically from u/v at the hit.
    auto material = _material->emit(scene, curve.instance_id, curve.instance,
                                    curve.geometry, material_slot, false);
    UInt cycles_primitive_index = curve.segment.cycles_curve_index;
    return {.curve = std::move(curve),
            .material_slot = std::move(material_slot),
            .material_binding = std::move(material.binding),
            .cycles_surface_shader = std::move(material.cycles_surface_shader),
            .cycles_object_index = std::move(material.cycles_object_index),
            .cycles_primitive_index = std::move(cycles_primitive_index)};
  }
};

} // namespace

std::shared_ptr<const CurvePrimitiveComponent>
make_curve_primitive_component() {
  return std::make_shared<BindlessCurvePrimitiveComponent>();
}

} // namespace psycles::luisa_backend::detail
