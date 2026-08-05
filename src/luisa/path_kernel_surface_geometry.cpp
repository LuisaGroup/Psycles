#include "path_kernel_builder.h"
#include "path_kernel_surface_primitive.h"

#include <psycles/luisa/cycles_transform.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceGeometryStageImpl final : public SurfaceGeometryStage {

private:
  std::shared_ptr<const SurfacePrimitiveGeometryComponent> _geometry;

public:
  SurfaceGeometryStageImpl()
      : _geometry{make_surface_primitive_geometry_component()} {}

  SurfaceGeometryContext
  emit(PathBounceContext &bounce) const noexcept override {
    auto &sample = bounce.sample;
    auto &invocation = sample.invocation;
    const auto &config = invocation.config;
    const auto &scene = config.scene;
    auto &hit = bounce.hit;
    auto &ray = sample.ray;
    auto &ray_dP = sample.ray_dP;
    auto &ray_dD = sample.ray_dD;
    auto &ray_visibility = sample.ray_visibility;
    auto &ray_events = sample.ray_events;
    auto &path_depth = sample.path_depth;
    auto &diffuse_depth = sample.diffuse_depth;
    auto &glossy_depth = sample.glossy_depth;
    auto &transparent_depth = sample.transparent_depth;
    auto &transmission_depth = sample.transmission_depth;
    auto &terminate_after_transparent = sample.terminate_after_transparent;
    auto &minimum_bsdf_pdf = sample.minimum_bsdf_pdf;
    const auto &surface_query = invocation.surface_query;
    const auto &kernel_parameters = invocation.parameters;
    const auto &safe_normalize = config.light_transport.safe_normalize;
    const auto reflective_caustics = config.reflective_caustics;
    const auto refractive_caustics = config.refractive_caustics;

    auto primitive =
        _geometry->emit(scene, hit, ray, ray_dP, ray_dD, safe_normalize);
    auto point = std::move(primitive.point);
    point.ray_visibility = ray_visibility;
    point.ray_events = ray_events;
    point.ray_depth = path_depth;
    point.diffuse_depth = diffuse_depth;
    point.glossy_depth = glossy_depth;
    point.transparent_depth = transparent_depth;
    point.transmission_depth = transmission_depth;

    Float3 shadow_shading_normal = point.shading_normal;
    $if((!primitive.is_curve) & (!bounce.subsurface_exit) &
        primitive.triangle_smooth &
        (primitive.instance.shadow_terminator_geometry_offset > 0.0f)) {
      shadow_shading_normal =
          invocation.surface_shading_normal(primitive.surface_tag, point);
    };

    UInt path_lobe_mask = surface_query.lobe_mask;
    const Bool previous_ray_was_diffuse =
        (ray_events & static_cast<std::uint32_t>(contract::event_diffuse)) !=
        0u;
    const auto path_reflective_caustics =
        Bool{reflective_caustics} | !previous_ray_was_diffuse;
    const auto path_refractive_caustics =
        Bool{refractive_caustics} | !previous_ray_was_diffuse;
    if (!reflective_caustics) {
      path_lobe_mask = select(
          path_lobe_mask,
          path_lobe_mask &
              ~static_cast<std::uint32_t>(contract::event_glossy),
          previous_ray_was_diffuse);
    }
    if (!refractive_caustics) {
      path_lobe_mask = select(
          path_lobe_mask,
          path_lobe_mask &
              ~static_cast<std::uint32_t>(contract::event_transmission),
          previous_ray_was_diffuse);
    }
    path_lobe_mask = select(
        path_lobe_mask,
        static_cast<std::uint32_t>(contract::event_transparent),
        terminate_after_transparent);
    SurfaceQuery path_surface_query{
        .lobe_mask = path_lobe_mask,
        .transport_mode = surface_query.transport_mode,
        .glossy_filter_roughness = 0.0f,
        .reflective_caustics = path_reflective_caustics,
        .refractive_caustics = path_refractive_caustics,
        .subsurface_exit = bounce.subsurface_exit};
    const auto blur_pdf = kernel_parameters.filter_glossy * minimum_bsdf_pdf;
    const auto filter_glossy_enabled =
        kernel_parameters.filter_glossy < std::numeric_limits<float>::max();
    path_surface_query.glossy_filter_roughness = select(
        0.0f, sqrt(max(1.0f - blur_pdf, 0.0f)) * 0.5f,
        filter_glossy_enabled & (blur_pdf < 1.0f));

    return {.bounce = bounce,
            .instance = std::move(primitive.instance),
            .p0 = std::move(primitive.p0),
            .p1 = std::move(primitive.p1),
            .p2 = std::move(primitive.p2),
            .n0 = std::move(primitive.n0),
            .n1 = std::move(primitive.n1),
            .n2 = std::move(primitive.n2),
            .object_to_world = primitive.object_to_world,
            .world_to_object = primitive.world_to_object,
            .wp0 = std::move(primitive.wp0),
            .wp1 = std::move(primitive.wp1),
            .wp2 = std::move(primitive.wp2),
            .geometric_normal = point.geometric_normal,
            .hit_position = std::move(primitive.hit_position),
            .object_hit_position = std::move(primitive.object_hit_position),
            .differential_radius = primitive.differential_radius,
            .is_curve = primitive.is_curve,
            .cycles_transform_applied =
                primitive.cycles_transform_applied,
            .triangle_smooth = std::move(primitive.triangle_smooth),
            .shadow_shading_normal = std::move(shadow_shading_normal),
            .surface_tag = std::move(primitive.surface_tag),
            .cycles_surface_shader =
                std::move(primitive.cycles_surface_shader),
            .cycles_object_index = std::move(primitive.cycles_object_index),
            .cycles_primitive_index =
                std::move(primitive.cycles_primitive_index),
            .volume_stack_entry = std::move(primitive.volume_stack_entry),
            .surface_has_volume = std::move(primitive.surface_has_volume),
            .point = std::move(point),
            .path_surface_query = std::move(path_surface_query)};
  }
};

} // namespace

Float3 SurfaceGeometryContext::make_ray_origin(Float3 direction) const noexcept {
  Float3 origin = hit_position;
  $if(!is_curve) {
    const Float3 object_direction =
        cycles_transform::direction(world_to_object, direction);
    const Float3 cycles_origin = select(
        object_hit_position, hit_position, cycles_transform_applied);
    const Float3 cycles_direction = select(
        object_direction, direction, cycles_transform_applied);
    origin = surface_ray::origin_with_explicit_self_exclusion(
        hit_position, geometric_normal, cycles_origin, cycles_direction,
        p0, p1, p2);
  };
  return origin;
}

surface_ray::ShadowOrigin
SurfaceGeometryContext::make_shadow_origin(Float3 direction) const noexcept {
  Float3 position = hit_position;
  Bool skip_self = true;
  $if(!is_curve) {
    const auto triangle = surface_ray::surface_shadow_origin(
        hit_position, shadow_shading_normal, geometric_normal, direction,
        instance.shadow_terminator_geometry_offset, triangle_smooth,
        object_to_world, world_to_object, cycles_transform_applied,
        bounce.hit->bary, p0, p1, p2, n0, n1, n2);
    position = triangle.position;
    skip_self = triangle.skip_self;
  };
  return {.position = std::move(position), .skip_self = std::move(skip_self)};
}

std::unique_ptr<SurfaceGeometryStage> make_surface_geometry_stage() {
  return std::make_unique<SurfaceGeometryStageImpl>();
}

} // namespace psycles::luisa_backend::detail
