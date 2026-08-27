#include "path_kernel_surface_primitive.h"

#include "cycles_triangle_surface_component.h"
#include "path_kernel_curve_geometry.h"
#include "path_kernel_triangle_geometry.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class CyclesSurfacePrimitiveGeometryComponent final
    : public SurfacePrimitiveGeometryComponent {

private:
  ScenePrimitiveStagePlan _plan;
  std::shared_ptr<const TriangleGeometryComponent> _triangles;
  std::shared_ptr<const CurveGeometryComponent> _curves;
  std::shared_ptr<const CyclesTriangleSurfaceComponent> _triangle_surface;

public:
  explicit CyclesSurfacePrimitiveGeometryComponent(ScenePrimitiveStagePlan plan)
      : _plan{plan},
        _triangles{plan.triangles ? make_triangle_geometry_component()
                                  : nullptr},
        _curves{plan.curves ? make_curve_geometry_component() : nullptr},
        _triangle_surface{plan.triangles
                              ? make_cycles_triangle_surface_component()
                              : nullptr} {}

  SurfacePrimitiveGeometryContext
  emit(const std::shared_ptr<LuisaSceneData> &scene,
       const Var<luisa::compute::CommittedHit> &hit,
       const Var<luisa::compute::Ray> &ray, Expr<float> ray_dP,
       Expr<float> ray_dD,
       const SafeNormalizeCallable &safe_normalize) const noexcept override {
    Bool is_curve = false;
    if (_plan.curves) {
      is_curve = _plan.triangles ? hit->is_procedural() : Bool{true};
    }
    const auto object_to_world = scene->accel->instance_transform(hit->inst);
    const auto world_to_object = inverse(object_to_world);
    const auto normal_to_world = transpose(world_to_object);
    const Float differential_radius = ray_dP + hit->committed_ray_t * ray_dD;

    Var<InstanceGpu> instance;
    Var<MaterialBindingGpu> material_binding;
    Float3 p0 = make_float3(0.0f);
    Float3 p1 = make_float3(0.0f);
    Float3 p2 = make_float3(0.0f);
    Float3 cycles_p0 = make_float3(0.0f);
    Float3 cycles_p1 = make_float3(0.0f);
    Float3 cycles_p2 = make_float3(0.0f);
    Float3 n0 = make_float3(0.0f);
    Float3 n1 = make_float3(0.0f);
    Float3 n2 = make_float3(0.0f);
    Float3 cycles_n0 = make_float3(0.0f);
    Float3 cycles_n1 = make_float3(0.0f);
    Float3 cycles_n2 = make_float3(0.0f);
    Float3 undisplaced_p0 = make_float3(0.0f);
    Float3 undisplaced_p1 = make_float3(0.0f);
    Float3 undisplaced_p2 = make_float3(0.0f);
    Float3 undisplaced_n0 = make_float3(0.0f);
    Float3 undisplaced_n1 = make_float3(0.0f);
    Float3 undisplaced_n2 = make_float3(0.0f);
    Float4 undisplaced_tangent0 = make_float4(0.0f);
    Float4 undisplaced_tangent1 = make_float4(0.0f);
    Float4 undisplaced_tangent2 = make_float4(0.0f);
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
    Float3 surface_dpdu = make_float3(1.0f, 0.0f, 0.0f);
    Float3 surface_dpdv = make_float3(0.0f, 1.0f, 0.0f);
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
    Bool cycles_transform_applied = false;
    Bool triangle_smooth = false;
    UInt emission_sampling =
        static_cast<std::uint32_t>(contract::EmissionSampling::none);
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

    const auto emit_curve = [&] {
      const auto curve =
          _curves->emit(scene, hit->inst, hit->prim, ray, hit->committed_ray_t);
      const auto volume = curve.primitive.volume_stack_entry();
      instance = curve.primitive.curve.metadata.instance;
      material_binding = curve.primitive.material_binding;
      object_position = curve.object_position;
      hit_position = curve.position;
      object_shading_normal = curve.object_shading_normal;
      shading_normal = curve.shading_normal;
      geometric_normal = curve.geometric_normal;
      object_tangent = normalize(curve.object_dpdu);
      tangent = normalize(curve.dpdu);
      surface_dpdu = curve.dpdu;
      surface_dpdv = curve.dpdv;
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
      emission_sampling = curve.primitive.triangle_emission_sampling;
      surface_has_volume = curve.primitive.has_volume;
      volume_object = volume.object;
      volume_shader = volume.shader;
      volume_surface_tag = volume.surface_tag;
      volume_parameter_block = volume.parameter_block;
      volume_instance_id = volume.instance_id;
      volume_sample_method = volume.sample_method;
      volume_valid = volume.valid;
    };
    const auto emit_triangle = [&] {
      const auto triangle = _triangles->emit(scene, hit->inst, hit->prim);
      const auto &primitive = triangle.primitive;
      const auto volume = primitive.volume_stack_entry();
      instance = primitive.instance;
      material_binding = primitive.material_binding;
      p0 = triangle.p0;
      p1 = triangle.p1;
      p2 = triangle.p2;
      cycles_transform_applied = instance.cycles_transform_applied != 0u;
      n0 = triangle.n0;
      n1 = triangle.n1;
      n2 = triangle.n2;
      undisplaced_p0 = triangle.undisplaced_p0;
      undisplaced_p1 = triangle.undisplaced_p1;
      undisplaced_p2 = triangle.undisplaced_p2;
      undisplaced_n0 = triangle.undisplaced_n0;
      undisplaced_n1 = triangle.undisplaced_n1;
      undisplaced_n2 = triangle.undisplaced_n2;
      undisplaced_tangent0 = triangle.undisplaced_tangent0;
      undisplaced_tangent1 = triangle.undisplaced_tangent1;
      undisplaced_tangent2 = triangle.undisplaced_tangent2;
      const auto surface = _triangle_surface->resolve(
          {.object_to_world = object_to_world,
           .normal_to_world = normal_to_world,
           .transform_applied = cycles_transform_applied,
           .barycentric = hit->bary,
           .ray_direction = ray->direction(),
           .smooth = primitive.smooth,
           .p0 = p0,
           .p1 = p1,
           .p2 = p2,
           .final_p0 = triangle.cycles_intersection_p0,
           .final_p1 = triangle.cycles_intersection_p1,
           .final_p2 = triangle.cycles_intersection_p2,
           .n0 = n0,
           .n1 = n1,
           .n2 = n2},
          safe_normalize);
      cycles_p0 = surface.p0;
      cycles_p1 = surface.p1;
      cycles_p2 = surface.p2;
      cycles_n0 = surface.n0;
      cycles_n1 = surface.n1;
      cycles_n2 = surface.n2;
      wp0 = surface.world_p0;
      wp1 = surface.world_p1;
      wp2 = surface.world_p2;
      const auto derivatives =
          cycles_triangle_surface_derivatives(surface);
      surface_dpdu = derivatives.dpdu;
      surface_dpdv = derivatives.dpdv;
      object_position = surface.object_position;
      hit_position = surface.position;
      object_shading_normal = surface.object_shading_normal;
      shading_normal = surface.shading_normal;
      geometric_normal = surface.geometric_normal;
      back_facing = surface.back_facing;
      const auto packed_object_tangent = triangle_interpolate(
          hit->bary, triangle.tangent0, triangle.tangent1, triangle.tangent2);
      object_tangent = packed_object_tangent.xyz();
      tangent_sign = packed_object_tangent.w;
      tangent = safe_normalize(
          (object_to_world * make_float4(object_tangent, 0.0f)).xyz(),
          safe_normalize((wp1 - wp0) - geometric_normal *
                                           dot(wp1 - wp0, geometric_normal),
                         make_float3(1.0f, 0.0f, 0.0f)));
      generated =
          triangle_interpolate(hit->bary, triangle.generated0,
                               triangle.generated1, triangle.generated2);
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
      emission_sampling = primitive.triangle_emission_sampling;
      surface_has_volume = primitive.has_volume;
      volume_object = volume.object;
      volume_shader = volume.shader;
      volume_surface_tag = volume.surface_tag;
      volume_parameter_block = volume.parameter_block;
      volume_instance_id = volume.instance_id;
      volume_sample_method = volume.sample_method;
      volume_valid = volume.valid;
    };

    if (_plan.mixed()) {
      $if(is_curve) { emit_curve(); }
      $else { emit_triangle(); };
    } else if (_plan.curves) {
      emit_curve();
    } else {
      emit_triangle();
    }

    const auto normal_components_differ =
        (geometric_normal.x != geometric_normal.y) |
        (geometric_normal.x != geometric_normal.z);
    Float3 compact_dx =
        select(make_float3(geometric_normal.z - geometric_normal.y,
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

    Float3 undisplaced_position = hit_position;
    Float3 undisplaced_object_position = object_position;
    Float3 undisplaced_shading_normal = shading_normal;
    Float3 undisplaced_object_shading_normal = object_shading_normal;
    Float3 undisplaced_object_tangent = object_tangent;
    Float undisplaced_tangent_sign = tangent_sign;
    Float3 undisplaced_dPdx = dPdx;
    Float3 undisplaced_dPdy = dPdy;
    Float3 undisplaced_object_dPdx = object_dPdx;
    Float3 undisplaced_object_dPdy = object_dPdy;

    const auto emit_triangle_differentials = [&] {
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
        auto delta = make_float2((projected1 * gram11 - projected2 * gram01) /
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
      uv_dx = (uv1 - uv0) * barycentric_dx.x + (uv2 - uv0) * barycentric_dx.y;
      uv_dy = (uv1 - uv0) * barycentric_dy.x + (uv2 - uv0) * barycentric_dy.y;

      // Cycles NODE_ENTER_BUMP_EVAL interpolates ATTR_STD_POSITION_
      // UNDISPLACED with the hit's current barycentric differentials. It
      // restores P afterwards but deliberately keeps the bump result in N.
      const auto undisplaced_object_geometric_normal =
          safe_normalize(cross(undisplaced_p1 - undisplaced_p0,
                               undisplaced_p2 - undisplaced_p0),
                         make_float3(0.0f, 0.0f, 1.0f));
      undisplaced_object_shading_normal =
          select(undisplaced_object_geometric_normal,
                 triangle_interpolate(barycentric, undisplaced_n0,
                                      undisplaced_n1, undisplaced_n2),
                 triangle_smooth);
      undisplaced_shading_normal =
          safe_normalize((normal_to_world *
                          make_float4(undisplaced_object_shading_normal, 0.0f))
                             .xyz(),
                         geometric_normal);
      undisplaced_shading_normal = select(
          undisplaced_shading_normal, -undisplaced_shading_normal, back_facing);
      undisplaced_object_position = triangle_interpolate(
          barycentric, undisplaced_p0, undisplaced_p1, undisplaced_p2);
      undisplaced_position =
          (object_to_world * make_float4(undisplaced_object_position, 1.0f))
              .xyz();
      undisplaced_object_dPdx =
          (undisplaced_p1 - undisplaced_p0) * barycentric_dx.x +
          (undisplaced_p2 - undisplaced_p0) * barycentric_dx.y;
      undisplaced_object_dPdy =
          (undisplaced_p1 - undisplaced_p0) * barycentric_dy.x +
          (undisplaced_p2 - undisplaced_p0) * barycentric_dy.y;
      undisplaced_dPdx =
          (object_to_world * make_float4(undisplaced_object_dPdx, 0.0f)).xyz();
      undisplaced_dPdy =
          (object_to_world * make_float4(undisplaced_object_dPdy, 0.0f)).xyz();
      const auto packed_undisplaced_tangent =
          triangle_interpolate(barycentric, undisplaced_tangent0,
                               undisplaced_tangent1, undisplaced_tangent2);
      undisplaced_object_tangent = packed_undisplaced_tangent.xyz();
      undisplaced_tangent_sign = packed_undisplaced_tangent.w;
    };

    if (_plan.triangles) {
      if (_plan.curves) {
        $if(!is_curve) { emit_triangle_differentials(); };
      } else {
        emit_triangle_differentials();
      }
    }

    const auto transformed_object_hit_position =
        (world_to_object * make_float4(hit_position, 1.0f)).xyz();
    const auto object_hit_position =
        select(transformed_object_hit_position, hit_position,
               cycles_transform_applied);
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
        .undisplaced_position = undisplaced_position,
        .undisplaced_object_position = undisplaced_object_position,
        .undisplaced_shading_normal = undisplaced_shading_normal,
        .undisplaced_object_shading_normal = undisplaced_object_shading_normal,
        .undisplaced_object_tangent = undisplaced_object_tangent,
        .undisplaced_tangent_sign = undisplaced_tangent_sign,
        .normal_to_world_x =
            (normal_to_world * make_float4(1.0f, 0.0f, 0.0f, 0.0f)).xyz(),
        .normal_to_world_y =
            (normal_to_world * make_float4(0.0f, 1.0f, 0.0f, 0.0f)).xyz(),
        .normal_to_world_z =
            (normal_to_world * make_float4(0.0f, 0.0f, 1.0f, 0.0f)).xyz(),
        .dpdu = surface_dpdu,
        .dpdv = surface_dpdv,
        .dPdx = dPdx,
        .dPdy = dPdy,
        .object_dPdx = object_dPdx,
        .object_dPdy = object_dPdy,
        .undisplaced_dPdx = undisplaced_dPdx,
        .undisplaced_dPdy = undisplaced_dPdy,
        .undisplaced_object_dPdx = undisplaced_object_dPdx,
        .undisplaced_object_dPdy = undisplaced_object_dPdy,
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
        .triangle_smooth = triangle_smooth,
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
    return {
        .instance = std::move(instance),
        .p0 = std::move(cycles_p0),
        .p1 = std::move(cycles_p1),
        .p2 = std::move(cycles_p2),
        .n0 = std::move(cycles_n0),
        .n1 = std::move(cycles_n1),
        .n2 = std::move(cycles_n2),
        .object_to_world = object_to_world,
        .world_to_object = world_to_object,
        .wp0 = std::move(wp0),
        .wp1 = std::move(wp1),
        .wp2 = std::move(wp2),
        .hit_position = std::move(hit_position),
        .object_hit_position = std::move(object_hit_position),
        .differential_radius = differential_radius,
        .is_curve = is_curve,
        .cycles_transform_applied = std::move(cycles_transform_applied),
        .triangle_smooth = std::move(triangle_smooth),
        .emission_sampling = std::move(emission_sampling),
        .surface_tag = std::move(surface_tag),
        .cycles_surface_shader = std::move(cycles_surface_shader),
        .cycles_object_index = std::move(cycles_object_index),
        .cycles_primitive_index = std::move(cycles_primitive_index),
        .volume_stack_entry = {.object = std::move(volume_object),
                               .shader = std::move(volume_shader),
                               .surface_tag = std::move(volume_surface_tag),
                               .parameter_block =
                                   std::move(volume_parameter_block),
                               .instance_id = std::move(volume_instance_id),
                               .sample_method = std::move(volume_sample_method),
                               .valid = std::move(volume_valid)},
        .surface_has_volume = std::move(surface_has_volume),
        .surface_has_bssrdf_bump =
            (material_binding.flags & material_flag_has_bssrdf_bump) != 0u,
        .point = std::move(point)};
  }
};

} // namespace

std::shared_ptr<const SurfacePrimitiveGeometryComponent>
make_surface_primitive_geometry_component(ScenePrimitiveStagePlan plan) {
  return std::make_shared<CyclesSurfacePrimitiveGeometryComponent>(plan);
}

} // namespace psycles::luisa_backend::detail
