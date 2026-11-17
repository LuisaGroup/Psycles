#include "path_kernel_surface_primitive.h"

#include "cycles_svm_internal.h"
#include "path_kernel_primitive_material.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_shadow.h"

#include <psycles/luisa/cycles_transform.h>

namespace psycles::luisa_backend::detail {
namespace {
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

class CyclesSvmSurfaceGeometry final : public SurfacePrimitiveGeometryComponent {
  std::shared_ptr<const PrimitiveMaterialComponent> _material{
      make_primitive_material_component()};

public:
  SurfacePrimitiveGeometryContext emit(
      const std::shared_ptr<LuisaSceneData> &scene,
      const Var<luisa::compute::CommittedHit> &hit,
      const Var<luisa::compute::Ray> &ray, Expr<float> ray_dP,
      Expr<float> ray_dD, Expr<float> ray_time,
      const Var<RenderKernelParameters> &parameters,
      const SafeNormalizeCallable &) const noexcept override {
    LUISA_ASSERT(scene->cycles_svm,
                 "Surface geometry requires the native Cycles image.");
    const PathCyclesSvmKernelGlobals kg{
        scene, parameters, scene->camera.projection, true, true};
    // Surface SVM initializes its own per-path LCG state immediately before
    // material evaluation. Ray time belongs to the common geometric state.
    const bool lazy_cycles_geometry = scene->native_cycles_svm_surface;
    Var<InstanceGpu> instance;
    svm::TriangleVertices vertices{};
    svm::TriangleNormals normals{};
    auto setup = setup_cycles_svm_ray_shader_data(
        scene, kg, ray, hit->inst, hit->prim, hit->bary,
        hit->committed_ray_t, ray_dP, ray_dD, ray_time, 0u, parameters,
        &instance,
        lazy_cycles_geometry ? nullptr : &vertices,
        lazy_cycles_geometry ? nullptr : &normals);
    const auto &sd = setup.shader_data;
    const auto &transforms = setup.transforms;
    const auto is_curve = (sd.type & unsigned(abi::PRIMITIVE_CURVE)) != 0u;
    const auto applied =
        (sd.object_flag & unsigned(abi::SD_OBJECT_TRANSFORM_APPLIED)) != 0u;
    const auto smooth = (sd.shader & svm::shader_smooth_normal) != 0u;
    Float shadow_terminator_geometry_offset =
        instance.shadow_terminator_geometry_offset;

    const bool fast_native_surface =
        scene->native_cycles_svm_surface &&
        scene->volume_metadata.stack_size == 0u;

    UInt surface_tag = 0u;
    UInt parameter_block = 0u;
    UInt triangle_emission_sampling =
        static_cast<std::uint32_t>(contract::EmissionSampling::none);
    UInt volume_sampling = 0u;
    Bool has_volume = false;
    if (fast_native_surface) {
      const Bool may_emit =
          (sd.flag & static_cast<unsigned>(abi::SD_HAS_EMISSION)) != 0u;
      const Bool front =
          (sd.flag & static_cast<unsigned>(abi::SD_MIS_FRONT)) != 0u;
      const Bool back =
          (sd.flag & static_cast<unsigned>(abi::SD_MIS_BACK)) != 0u;
      const UInt emission_sampling = select(
          select(UInt{static_cast<std::uint32_t>(
                     contract::EmissionSampling::none)},
                 UInt{static_cast<std::uint32_t>(
                     contract::EmissionSampling::back)},
                 back),
          select(UInt{static_cast<std::uint32_t>(
                     contract::EmissionSampling::front)},
                 UInt{static_cast<std::uint32_t>(
                     contract::EmissionSampling::front_back)},
                 back),
          front);
      const Bool light_visible =
          (instance.visibility_mask & mesh_light_sampling_visibility) != 0u;
      triangle_emission_sampling = select(
          UInt{static_cast<std::uint32_t>(contract::EmissionSampling::none)},
          emission_sampling,
          may_emit & light_visible & (!is_curve) & (front | back));
    } else {
      const auto geometry = scene->geometry_buffer->read(instance.geometry_index);
      UInt material_slot;
      const auto triangle_geometry = [&] {
        material_slot =
            _material->triangle_material_slot(scene, geometry, hit->prim);
      };
      const auto curve_geometry = [&] {
        const auto segment =
            scene->heap->buffer<CurveSegmentGpu>(geometry.bindless_base)
                .read(hit->prim);
        material_slot =
            _material->curve_material_slot(scene, geometry, segment);
      };
      if (scene->curve_geometries.empty()) {
        triangle_geometry();
      } else if (scene->geometries.empty()) {
        curve_geometry();
      } else {
        $if(is_curve) { curve_geometry(); }
        $else { triangle_geometry(); };
      }
      auto material = _material->emit(scene, hit->inst, instance, geometry,
                                      material_slot, smooth);
      surface_tag = material.binding.surface_tag;
      parameter_block = material.binding.parameter_block;
      triangle_emission_sampling = material.triangle_emission_sampling;
      volume_sampling = material.binding.volume_sampling;
      has_volume = material.has_volume;
    }

    SurfacePoint point;
    point.position = sd.P;
    if (!fast_native_surface) {
      const auto w2o = transforms.fetch_world_to_object();
      const auto o2w = transforms.fetch_object_to_world();
      const auto object_P =
          cycles_transform::point(w2o, sd.P);
      const auto dP = svm::detail::differential_from_compact(sd.Ng, sd.dP);
      point.object_position = object_P;
      point.object_location = cycles_transform::point(
          o2w, make_float3(0.0f));
      point.dPdx = dP.dx;
      point.dPdy = dP.dy;
      point.object_dPdx =
          cycles_transform::direction(w2o, dP.dx);
      point.object_dPdy =
          cycles_transform::direction(w2o, dP.dy);
    }
    point.geometric_normal = sd.Ng;
    point.shading_normal = sd.N;
    point.incoming = sd.wi;
    point.dpdu = sd.dPdu;
    point.dpdv = sd.dPdv;
    point.barycentric = make_float2(sd.u, sd.v);
    point.barycentric_dx = make_float2(sd.du.dx, sd.dv.dx);
    point.barycentric_dy = make_float2(sd.du.dy, sd.dv.dy);
    point.instance_id = hit->inst;
    point.primitive_id = hit->prim;
    point.geometry_index = instance.geometry_index;
    point.parameter_block = parameter_block;
    point.object_random = instance.object_random;
    point.particle_index = instance.particle_index;
    point.is_curve = is_curve;
    point.triangle_smooth = smooth;
    point.ray_length = sd.ray_length;
    point.time = sd.time;
    point.back_facing = (sd.flag & unsigned(abi::SD_BACKFACING)) != 0u;
    point.use_bump_map_correction =
        (sd.flag & unsigned(abi::SD_USE_BUMP_MAP_CORRECTION)) != 0u;

    if (lazy_cycles_geometry) {
      SurfacePrimitiveGeometryContext result{
          .instance = Var<InstanceGpu>{},
          .p0 = Float3{}, .p1 = Float3{}, .p2 = Float3{},
          .n0 = Float3{}, .n1 = Float3{}, .n2 = Float3{},
          .object_to_world = luisa::compute::Float4x4{},
          .world_to_object = luisa::compute::Float4x4{},
          .wp0 = Float3{}, .wp1 = Float3{}, .wp2 = Float3{},
          .hit_position = sd.P,
          .object_hit_position = Float3{},
          .differential_radius = sd.dP,
          .is_curve = is_curve,
          .cycles_primitive_type = sd.type,
          .cycles_transform_applied = applied,
          .triangle_smooth = smooth,
          .emission_sampling = triangle_emission_sampling,
          .surface_tag = surface_tag,
          .cycles_surface_shader = sd.shader,
          .cycles_object_index = sd.object,
          .cycles_primitive_index = sd.prim,
          .cycles_shader_flags = sd.flag,
          .cycles_object_flags = sd.object_flag,
          .volume_stack_entry = {
              .object = sd.object, .shader = sd.shader,
              .surface_tag = surface_tag,
              .parameter_block = parameter_block,
              .instance_id = hit->inst,
              .sample_method = volume_sampling,
              .valid = has_volume},
          .surface_has_volume = has_volume,
          .surface_has_bssrdf_bump =
              (sd.flag & unsigned(abi::SD_HAS_BSSRDF_BUMP)) != 0u,
          .point = std::move(point),
          .shadow_terminator_geometry_offset =
              std::move(shadow_terminator_geometry_offset),
          .lazy_cycles_geometry = true};
      return result;
    }

    Float3 p0 = vertices.v0, p1 = vertices.v1, p2 = vertices.v2;
    Float3 n0 = normals.n0, n1 = normals.n1, n2 = normals.n2;
    Float3 wp0 = p0, wp1 = p1, wp2 = p2;
    const auto o2w = transforms.fetch_object_to_world();
    const auto w2o = transforms.fetch_world_to_object();
    Float3 object_hit_position = sd.P;
    $if(!applied) {
      object_hit_position = cycles_transform::point(w2o, sd.P);
    };

    return {
        .instance = instance,
        .p0 = p0, .p1 = p1, .p2 = p2,
        .n0 = n0, .n1 = n1, .n2 = n2,
        .object_to_world = o2w,
        .world_to_object = w2o,
        .wp0 = wp0, .wp1 = wp1, .wp2 = wp2,
        .hit_position = sd.P,
        .object_hit_position = object_hit_position,
        .differential_radius = sd.dP,
        .is_curve = is_curve,
        .cycles_primitive_type = sd.type,
        .cycles_transform_applied = applied,
        .triangle_smooth = smooth,
        .emission_sampling = triangle_emission_sampling,
        .surface_tag = surface_tag,
        .cycles_surface_shader = sd.shader,
        .cycles_object_index = sd.object,
        .cycles_primitive_index = sd.prim,
        .cycles_shader_flags = sd.flag,
        .cycles_object_flags = sd.object_flag,
        .volume_stack_entry = {
            .object = sd.object, .shader = sd.shader,
            .surface_tag = surface_tag,
            .parameter_block = parameter_block,
            .instance_id = hit->inst,
            .sample_method = volume_sampling,
            .valid = has_volume},
        .surface_has_volume = has_volume,
        .surface_has_bssrdf_bump =
            (sd.flag & unsigned(abi::SD_HAS_BSSRDF_BUMP)) != 0u,
        .point = std::move(point),
        .shadow_terminator_geometry_offset =
            std::move(shadow_terminator_geometry_offset),
        .lazy_cycles_geometry = false};
  }
};
} // namespace

std::shared_ptr<const SurfacePrimitiveGeometryComponent>
make_surface_primitive_geometry_component(ScenePrimitiveStagePlan) {
  return std::make_shared<CyclesSvmSurfaceGeometry>();
}
} // namespace psycles::luisa_backend::detail
