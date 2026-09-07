#include "path_kernel_surface_primitive.h"

#include "cycles_svm_internal.h"
#include "path_kernel_triangle_primitive.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_shadow.h"

#include <psycles/luisa/cycles_transform.h>

namespace psycles::luisa_backend::detail {
namespace {
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

class CyclesSvmSurfaceGeometry final : public SurfacePrimitiveGeometryComponent {
  std::shared_ptr<const TrianglePrimitiveComponent> _primitive{
      make_triangle_primitive_component()};

public:
  SurfacePrimitiveGeometryContext emit(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::CommittedHit> &hit,
      const Var<luisa::compute::Ray> &ray, Expr<float> ray_dP,
      Expr<float> ray_dD, Expr<float> ray_time,
      const Var<RenderKernelParameters> &parameters,
      const SafeNormalizeCallable &) const noexcept override {
    LUISA_ASSERT(scene->cycles_svm && scene->curve_geometries.empty(),
                 "Native surface geometry requires the admitted triangle image.");
    const PathCyclesSvmKernelGlobals kg{
        scene, parameters, scene->camera.projection, true, true};
    auto primitive = _primitive->emit(scene, hit->inst, hit->prim);
    // Surface SVM initializes its own per-path LCG state immediately before
    // material evaluation. Ray time belongs to the common geometric state.
    auto setup = setup_cycles_svm_ray_shader_data(
        scene, kg, ray, hit->inst, hit->prim, hit->bary,
        hit->committed_ray_t, ray_dP, ray_dD, ray_time, 0u, parameters);
    const auto &sd = setup.shader_data;
    const auto &transforms = setup.transforms;
    const auto vertices = kg.triangle_vertices(sd.object, sd.prim);
    const auto normals = svm::detail::triangle_normals(kg, sd);
    const auto applied =
        (sd.object_flag & unsigned(abi::SD_OBJECT_TRANSFORM_APPLIED)) != 0u;
    const auto smooth = (sd.shader & svm::shader_smooth_normal) != 0u;
    Float3 wp0 = vertices.v0, wp1 = vertices.v1, wp2 = vertices.v2;
    $if(!applied) {
      wp0 = cycles_transform::point(transforms.object_to_world, vertices.v0);
      wp1 = cycles_transform::point(transforms.object_to_world, vertices.v1);
      wp2 = cycles_transform::point(transforms.object_to_world, vertices.v2);
    };
    const auto object_P = cycles_transform::point(transforms.world_to_object, sd.P);
    const auto dP = svm::detail::differential_from_compact(sd.Ng, sd.dP);
    SurfacePoint point;
    // Only the common integrator and native ShaderData adapter consume this
    // point. SVM attributes (UV, generated, normal-map tangent, etc.) are read
    // from the typed Cycles image at their node, never eagerly evaluated by
    // the legacy geometric/material path.
    point.position = sd.P;
    point.object_position = object_P;
    point.object_location = cycles_transform::point(
        transforms.object_to_world, make_float3(0.0f));
    point.geometric_normal = sd.Ng;
    point.shading_normal = sd.N;
    point.incoming = sd.wi;
    point.dpdu = sd.dPdu;
    point.dpdv = sd.dPdv;
    point.dPdx = dP.dx;
    point.dPdy = dP.dy;
    point.object_dPdx = cycles_transform::direction(transforms.world_to_object, dP.dx);
    point.object_dPdy = cycles_transform::direction(transforms.world_to_object, dP.dy);
    point.barycentric = make_float2(sd.u, sd.v);
    point.barycentric_dx = make_float2(sd.du.dx, sd.dv.dx);
    point.barycentric_dy = make_float2(sd.du.dy, sd.dv.dy);
    point.instance_id = hit->inst;
    point.primitive_id = hit->prim;
    point.geometry_index = primitive.instance.geometry_index;
    point.parameter_block = primitive.material_binding.parameter_block;
    point.object_random = primitive.instance.object_random;
    point.particle_index = primitive.instance.particle_index;
    point.triangle_smooth = smooth;
    point.ray_length = sd.ray_length;
    point.time = sd.time;
    point.back_facing = (sd.flag & unsigned(abi::SD_BACKFACING)) != 0u;
    point.use_bump_map_correction =
        (sd.flag & unsigned(abi::SD_USE_BUMP_MAP_CORRECTION)) != 0u;

    return {
        .instance = std::move(primitive.instance),
        .p0 = vertices.v0, .p1 = vertices.v1, .p2 = vertices.v2,
        .n0 = normals.n0, .n1 = normals.n1, .n2 = normals.n2,
        .object_to_world = transforms.object_to_world,
        .world_to_object = transforms.world_to_object,
        .wp0 = wp0, .wp1 = wp1, .wp2 = wp2,
        .hit_position = sd.P,
        .object_hit_position = select(object_P, sd.P, applied),
        .differential_radius = sd.dP,
        .is_curve = false,
        .cycles_transform_applied = applied,
        .triangle_smooth = smooth,
        .emission_sampling = primitive.triangle_emission_sampling,
        .surface_tag = primitive.material_binding.surface_tag,
        .cycles_surface_shader = sd.shader,
        .cycles_object_index = sd.object,
        .cycles_primitive_index = sd.prim,
        .volume_stack_entry = primitive.volume_stack_entry(),
        .surface_has_volume = primitive.has_volume,
        .surface_has_bssrdf_bump =
            (sd.flag & unsigned(abi::SD_HAS_BSSRDF_BUMP)) != 0u,
        .point = std::move(point)};
  }
};
} // namespace

std::shared_ptr<const SurfacePrimitiveGeometryComponent>
make_cycles_svm_surface_geometry_component() {
  return std::make_shared<CyclesSvmSurfaceGeometry>();
}
} // namespace psycles::luisa_backend::detail
