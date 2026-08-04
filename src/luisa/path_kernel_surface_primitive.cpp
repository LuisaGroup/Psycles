#include "path_kernel_surface_primitive.h"

#include "path_kernel_curve_geometry.h"
#include "path_kernel_triangle_geometry.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class CyclesSurfacePrimitiveGeometryComponent final
    : public SurfacePrimitiveGeometryComponent {

private:
  std::shared_ptr<const TriangleGeometryComponent> _triangles;
  std::shared_ptr<const CurveGeometryComponent> _curves;

public:
  CyclesSurfacePrimitiveGeometryComponent()
      : _triangles{make_triangle_geometry_component()},
        _curves{make_curve_geometry_component()} {}

  SurfacePrimitiveGeometryContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       const Var<luisa::compute::CommittedHit> &hit,
       const Var<luisa::compute::Ray> &ray, Expr<float> ray_dP,
       Expr<float> ray_dD,
       const SafeNormalizeCallable &safe_normalize) const noexcept override {
    const Bool is_curve = hit->is_procedural();
    const auto object_to_world = scene->accel->instance_transform(hit->inst);
    const auto world_to_object = inverse(object_to_world);
    const auto normal_to_world = transpose(world_to_object);
    const Float differential_radius =
        ray_dP + hit->committed_ray_t * ray_dD;

    Var<InstanceGpu> instance;
    Var<MaterialBindingGpu> material_binding;
    Float3 p0 = make_float3(0.0f);
    Float3 p1 = make_float3(0.0f);
    Float3 p2 = make_float3(0.0f);
    Float3 n0 = make_float3(0.0f);
    Float3 n1 = make_float3(0.0f);
    Float3 n2 = make_float3(0.0f);
    Float3 wp0 = make_float3(0.0f);
    Float3 wp1 = make_float3(0.0f);
    Float3 wp2 = make_float3(0.0f);
    Float3 object_position = make_float3(0.0f);
    Float3 hit_position = make_float3(0.0f);
    Float3 object_shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    Float3 shading_normal = make_float3(0.0f, 0.0f, 1.0f);
    Float3 geometric_normal = make_float3(0.0f, 0.0f, 1.0f);
    Float3 object_tangent = make_float3(1.0f, 0.0f, 0.0f);
    Float tangent_sign = 1.0f;
    Float3 tangent = make_float3(1.0f, 0.0f, 0.0f);
    Float3 generated = make_float3(0.0f);
    Float3 generated0 = make_float3(0.0f);
    Float3 generated1 = make_float3(0.0f);
    Float3 generated2 = make_float3(0.0f);
    Float3 generated_dx = make_float3(0.0f);
    Float3 generated_dy = make_float3(0.0f);
    Float2 uv = make_float2(0.0f);
    Float2 uv0 = make_float2(0.0f);
    Float2 uv1 = make_float2(0.0f);
    Float2 uv2 = make_float2(0.0f);
    Float2 uv_dx = make_float2(0.0f);
    Float2 uv_dy = make_float2(0.0f);
    Float2 barycentric = make_float2(0.0f);
    Float2 barycentric_dx = make_float2(0.0f);
    Float2 barycentric_dy = make_float2(0.0f);
    Float random_per_island = 0.0f;
    Bool back_facing = false;
    Bool triangle_smooth = false;
    Float curve_intercept = 0.0f;
    Float curve_length = 0.0f;
    Float curve_thickness = 0.0f;
    Float3 curve_tangent_normal = make_float3(0.0f);
    Float curve_random = 0.0f;
    UInt surface_tag = 0u;
    UInt cycles_surface_shader = 0u;
    UInt cycles_object_index = 0u;
    UInt cycles_primitive_index = 0u;
    Bool surface_has_volume = false;
    UInt volume_object = 0u;
    UInt volume_shader = 0u;
    UInt volume_surface_tag = 0u;
    UInt volume_parameter_block = 0u;
    UInt volume_instance_id = 0u;
    UInt volume_sample_method = 0u;
    Bool volume_valid = false;

    $if(is_curve) {
      const auto curve = _curves->emit(scene, hit->inst, hit->prim, ray,
                                       hit->committed_ray_t);
      const auto volume = curve.primitive.volume_stack_entry();
      instance = curve.primitive.curve.instance;
      material_binding = curve.primitive.material_binding;
      object_position = curve.object_position;
      hit_position = curve.position;
      object_shading_normal = curve.object_shading_normal;
      shading_normal = curve.shading_normal;
      geometric_normal = curve.geometric_normal;
      object_tangent = normalize(curve.object_dpdu);
      tangent = normalize(curve.dpdu);
      barycentric = make_float2(curve.intersection.u, curve.intersection.v);
      curve_intercept = curve.intercept;
      curve_length = curve.length;
      curve_thickness = curve.thickness;
      curve_tangent_normal = curve.tangent_normal;
      curve_random = curve.random;
      surface_tag = curve.primitive.material_binding.surface_tag;
      cycles_surface_shader = curve.primitive.cycles_surface_shader;
      cycles_object_index = curve.primitive.cycles_object_index;
      cycles_primitive_index = curve.primitive.cycles_primitive_index;
      surface_has_volume = curve.primitive.has_volume;
      volume_object = volume.object;
      volume_shader = volume.shader;
      volume_surface_tag = volume.surface_tag;
      volume_parameter_block = volume.parameter_block;
      volume_instance_id = volume.instance_id;
      volume_sample_method = volume.sample_method;
      volume_valid = volume.valid;
    }
    $else {
      const auto triangle = _triangles->emit(scene, hit->inst, hit->prim);
      const auto &primitive = triangle.primitive;
      const auto volume = primitive.volume_stack_entry();
      instance = primitive.instance;
      material_binding = primitive.material_binding;
      p0 = triangle.p0;
      p1 = triangle.p1;
      p2 = triangle.p2;
      n0 = triangle.n0;
      n1 = triangle.n1;
      n2 = triangle.n2;
      wp0 = (object_to_world * make_float4(p0, 1.0f)).xyz();
      wp1 = (object_to_world * make_float4(p1, 1.0f)).xyz();
      wp2 = (object_to_world * make_float4(p2, 1.0f)).xyz();
      const auto object_geometric_normal = safe_normalize(
          cross(p1 - p0, p2 - p0), make_float3(0.0f, 0.0f, 1.0f));
      geometric_normal = safe_normalize(
          (normal_to_world * make_float4(object_geometric_normal, 0.0f)).xyz(),
          -ray->direction());
      object_shading_normal =
          select(object_geometric_normal,
                 triangle_interpolate(hit->bary, n0, n1, n2),
                 primitive.smooth);
      const auto packed_object_tangent = triangle_interpolate(
          hit->bary, triangle.tangent0, triangle.tangent1, triangle.tangent2);
      object_tangent = packed_object_tangent.xyz();
      tangent_sign = packed_object_tangent.w;
      shading_normal = safe_normalize(
          (normal_to_world * make_float4(object_shading_normal, 0.0f)).xyz(),
          geometric_normal);
      back_facing = dot(geometric_normal, -ray->direction()) < 0.0f;
      geometric_normal = select(geometric_normal, -geometric_normal, back_facing);
      shading_normal = select(shading_normal, -shading_normal, back_facing);
      shading_normal = select(shading_normal, -shading_normal,
                              dot(shading_normal, geometric_normal) < 0.0f);
      object_position = p0 + hit->bary.x * (p1 - p0) +
                        hit->bary.y * (p2 - p0);
      hit_position =
          (object_to_world * make_float4(object_position, 1.0f)).xyz();
      tangent = safe_normalize(
          (object_to_world * make_float4(object_tangent, 0.0f)).xyz(),
          safe_normalize((wp1 - wp0) -
                             geometric_normal * dot(wp1 - wp0, geometric_normal),
                         make_float3(1.0f, 0.0f, 0.0f)));
      generated = triangle_interpolate(
          hit->bary, triangle.generated0, triangle.generated1,
          triangle.generated2);
      generated0 = triangle.generated0;
      generated1 = triangle.generated1;
      generated2 = triangle.generated2;
      uv = triangle_interpolate(hit->bary, triangle.uv0, triangle.uv1,
                                triangle.uv2);
      uv0 = triangle.uv0;
      uv1 = triangle.uv1;
      uv2 = triangle.uv2;
      barycentric = hit->bary;
      random_per_island = triangle.random_per_island;
      triangle_smooth = primitive.smooth;
      surface_tag = primitive.material_binding.surface_tag;
      cycles_surface_shader = primitive.cycles_surface_shader;
      cycles_object_index = primitive.cycles_object_index;
      cycles_primitive_index = primitive.cycles_primitive_index;
      surface_has_volume = primitive.has_volume;
      volume_object = volume.object;
      volume_shader = volume.shader;
      volume_surface_tag = volume.surface_tag;
      volume_parameter_block = volume.parameter_block;
      volume_instance_id = volume.instance_id;
      volume_sample_method = volume.sample_method;
      volume_valid = volume.valid;
    };

    const auto normal_components_differ =
        (geometric_normal.x != geometric_normal.y) |
        (geometric_normal.x != geometric_normal.z);
    Float3 compact_dx = select(
        make_float3(geometric_normal.z - geometric_normal.y,
                    geometric_normal.x + geometric_normal.z,
                    -geometric_normal.y - geometric_normal.x),
        make_float3(geometric_normal.z - geometric_normal.y,
                    geometric_normal.x - geometric_normal.z,
                    geometric_normal.y - geometric_normal.x),
        normal_components_differ);
    compact_dx = safe_normalize(compact_dx, tangent);
    const Float3 compact_dy = cross(geometric_normal, compact_dx);
    const Float3 dPdx = compact_dx * differential_radius;
    const Float3 dPdy = compact_dy * differential_radius;
    const Float3 object_dPdx =
        (world_to_object * make_float4(dPdx, 0.0f)).xyz();
    const Float3 object_dPdy =
        (world_to_object * make_float4(dPdy, 0.0f)).xyz();

    $if(!is_curve) {
      const Float3 edge1 = wp1 - wp0;
      const Float3 edge2 = wp2 - wp0;
      const Float gram00 = dot(edge1, edge1);
      const Float gram01 = dot(edge1, edge2);
      const Float gram11 = dot(edge2, edge2);
      const Float gram_determinant = gram00 * gram11 - gram01 * gram01;
      const Bool valid_gram = abs(gram_determinant) > 1.0e-20f;
      const Float safe_gram_determinant =
          select(1.0f, gram_determinant, valid_gram);
      const auto barycentric_differential = [&](Float3 differential) noexcept {
        const Float projected1 = dot(differential, edge1);
        const Float projected2 = dot(differential, edge2);
        auto delta = make_float2(
            (projected1 * gram11 - projected2 * gram01) /
                safe_gram_determinant,
            (projected2 * gram00 - projected1 * gram01) /
                safe_gram_determinant);
        delta = select(make_float2(0.0f), delta, valid_gram);
        return select(delta, -delta, back_facing);
      };
      barycentric_dx = barycentric_differential(dPdx);
      barycentric_dy = barycentric_differential(dPdy);
      generated_dx = (generated1 - generated0) * barycentric_dx.x +
                     (generated2 - generated0) * barycentric_dx.y;
      generated_dy = (generated1 - generated0) * barycentric_dy.x +
                     (generated2 - generated0) * barycentric_dy.y;
      uv_dx = (uv1 - uv0) * barycentric_dx.x +
              (uv2 - uv0) * barycentric_dx.y;
      uv_dy = (uv1 - uv0) * barycentric_dy.x +
              (uv2 - uv0) * barycentric_dy.y;
    };

    const auto object_hit_position =
        (world_to_object * make_float4(hit_position, 1.0f)).xyz();
    SurfacePoint point{
        .position = hit_position,
        .object_position = object_position,
        .object_location =
            (object_to_world * make_float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz(),
        .generated = generated,
        .geometric_normal = geometric_normal,
        .shading_normal = shading_normal,
        .object_shading_normal = object_shading_normal,
        .object_tangent = object_tangent,
        .tangent_sign = tangent_sign,
        .normal_to_world_x =
            (normal_to_world * make_float4(1.0f, 0.0f, 0.0f, 0.0f)).xyz(),
        .normal_to_world_y =
            (normal_to_world * make_float4(0.0f, 1.0f, 0.0f, 0.0f)).xyz(),
        .normal_to_world_z =
            (normal_to_world * make_float4(0.0f, 0.0f, 1.0f, 0.0f)).xyz(),
        .dpdu = tangent,
        .dpdv = cross(shading_normal, tangent),
        .dPdx = dPdx,
        .dPdy = dPdy,
        .object_dPdx = object_dPdx,
        .object_dPdy = object_dPdy,
        .generated_dx = generated_dx,
        .generated_dy = generated_dy,
        .incoming = -ray->direction(),
        .uv = uv,
        .uv_dx = uv_dx,
        .uv_dy = uv_dy,
        .geometry_index = instance.geometry_index,
        .barycentric = barycentric,
        .barycentric_dx = barycentric_dx,
        .barycentric_dy = barycentric_dy,
        .instance_id = hit->inst,
        .primitive_id = hit->prim,
        .parameter_block = material_binding.parameter_block,
        .object_random = instance.object_random,
        .particle_index = instance.particle_index,
        .random_per_island = random_per_island,
        .is_curve = is_curve,
        .curve_intercept = curve_intercept,
        .curve_length = curve_length,
        .curve_thickness = curve_thickness,
        .curve_tangent_normal = curve_tangent_normal,
        .curve_random = curve_random,
        .ray_visibility = 0u,
        .ray_events = 0u,
        .ray_depth = 0u,
        .diffuse_depth = 0u,
        .glossy_depth = 0u,
        .transparent_depth = 0u,
        .transmission_depth = 0u,
        .ray_length = hit->committed_ray_t,
        .time = 0.0f,
        .use_bump_map_correction =
            (material_binding.flags & material_flag_use_bump_map_correction) !=
            0u,
        .back_facing = back_facing};
    return {.instance = std::move(instance),
            .p0 = std::move(p0),
            .p1 = std::move(p1),
            .p2 = std::move(p2),
            .n0 = std::move(n0),
            .n1 = std::move(n1),
            .n2 = std::move(n2),
            .object_to_world = object_to_world,
            .world_to_object = world_to_object,
            .wp0 = std::move(wp0),
            .wp1 = std::move(wp1),
            .wp2 = std::move(wp2),
            .hit_position = std::move(hit_position),
            .object_hit_position = std::move(object_hit_position),
            .differential_radius = differential_radius,
            .is_curve = is_curve,
            .triangle_smooth = std::move(triangle_smooth),
            .surface_tag = std::move(surface_tag),
            .cycles_surface_shader = std::move(cycles_surface_shader),
            .cycles_object_index = std::move(cycles_object_index),
            .cycles_primitive_index = std::move(cycles_primitive_index),
            .volume_stack_entry =
                {.object = std::move(volume_object),
                 .shader = std::move(volume_shader),
                 .surface_tag = std::move(volume_surface_tag),
                 .parameter_block = std::move(volume_parameter_block),
                 .instance_id = std::move(volume_instance_id),
                 .sample_method = std::move(volume_sample_method),
                 .valid = std::move(volume_valid)},
            .surface_has_volume = std::move(surface_has_volume),
            .point = std::move(point)};
  }
};

} // namespace

std::shared_ptr<const SurfacePrimitiveGeometryComponent>
make_surface_primitive_geometry_component() {
  return std::make_shared<CyclesSurfacePrimitiveGeometryComponent>();
}

} // namespace psycles::luisa_backend::detail
