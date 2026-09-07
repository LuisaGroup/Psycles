#include "path_tracer_cycles_svm_shadow.h"

#include "cycles_svm_internal.h"
#include "path_tracer_cycles_svm_curve.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_shader_data.h"

#include <psycles/luisa/cycles_noise.h>

#include <algorithm>
#include <utility>

namespace psycles::luisa_backend::detail {
using namespace luisa::compute;
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

CyclesSvmShadowShaderData setup_cycles_svm_ray_shader_data(
    const std::shared_ptr<LuisaSceneData> &scene, const svm::KernelGlobals &kg,
    const Var<luisa::compute::Ray> &ray,
    Expr<unsigned> instance_id, Expr<unsigned> primitive_id,
    Expr<luisa::float2> barycentric, Expr<float> distance,
    Expr<float> ray_dP, Expr<float> ray_dD, Expr<float> ray_time,
    Expr<unsigned> lcg_state,
    const Var<RenderKernelParameters> &parameters) noexcept {
  const auto instance = scene->instance_buffer->read(instance_id);
  const auto object_index = instance.cycles_object_index;
  const auto primitive_index =
      instance.cycles_primitive_offset + primitive_id;
  const auto object =
      scene->cycles_svm->objects->object_buffer->read(object_index);
  const auto unpack = [](Var<abi::PackedTransform> t) noexcept {
    return svm::detail::transform_from_rows(
        make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
        make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
        make_float4(t.z.x, t.z.y, t.z.z, t.z.w));
  };
  svm::TransformState transforms{parameters.camera_transform,
                                 parameters.camera_inverse_transform,
                                 unpack(object.tfm), unpack(object.itfm)};
  const auto identity = make_float4x4(1.0f);
  svm::ShaderData sd{make_float3(0.0f),
                     make_float3(0.0f),
                     make_float3(0.0f),
                     -ray->direction(),
                     0u,
                     0u,
                     0u,
                     0u,
                     primitive_index,
                     barycentric.x,
                     barycentric.y,
                     object_index,
                     ray_time,
                     distance,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     make_float3(0.0f),
                     make_float3(0.0f),
                     identity,
                     identity,
                     lcg_state,
                     nullptr};
  sd.ray_P = ray->origin();
  sd.object_flag =
      scene->cycles_svm->objects->object_flag_buffer->read(object_index);
  sd.type = object.primitive_type.cast<unsigned>();
  const auto triangle_setup = [&] {
    assume(sd.type == static_cast<unsigned>(abi::PRIMITIVE_TRIANGLE));
    sd.shader = scene->cycles_svm->geometry->triangle_shader_buffer->read(sd.prim);
    cycles_svm_triangle_shader_setup(
        kg, transforms, kg.triangle_vertices(sd.object, sd.prim), sd);
  };
  const auto curve_setup = [&] {
    cycles_svm_curve_shader_setup(scene, kg, transforms, instance, primitive_id, ray, sd);
  };
  if (scene->curve_geometries.empty()) {
    triangle_setup();
  } else if (scene->geometries.empty()) {
    curve_setup();
  } else {
    $if((sd.type & unsigned(abi::PRIMITIVE_CURVE)) != 0u) { curve_setup(); }
    $else { triangle_setup(); };
  }
  sd.flag = (*scene->cycles_svm->kernel_shader_buffer)
                ->read(sd.shader & svm::shader_mask)
                .flags.cast<unsigned>();
  cycles_svm_shader_setup_backfacing(sd);
  sd.dP = ray_dP + distance * ray_dD;
  sd.dI = ray_dD;
  cycles_svm_shader_setup_dudv(sd);
  return {std::move(sd), std::move(transforms)};
}

CyclesSvmShadowShaderData setup_cycles_svm_shadow_shader_data(
    const std::shared_ptr<LuisaSceneData> &scene, const svm::KernelGlobals &kg,
    const Var<luisa::compute::Ray> &ray,
    const Var<ShadowIntersectionCall> &intersection, Expr<float> ray_dP,
    Expr<float> ray_dD, const Var<ShadowShaderContextCall> &context,
    const Var<RenderKernelParameters> &parameters) noexcept {
  return setup_cycles_svm_ray_shader_data(
      scene, kg, ray, intersection.instance, intersection.primitive,
      intersection.barycentric, intersection.distance, ray_dP, ray_dD,
      context.ray_time,
      cycles_noise::hash_uint3(context.rng_hash ^ 0xb4bc3953u,
                               context.rng_offset, context.sample_index),
      parameters);
}

EvaluateShadowSurfaceCallable make_cycles_svm_shadow_surface_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
  LUISA_ASSERT(
      scene->cycles_svm && scene->cycles_svm->geometry &&
          scene->cycles_svm->objects,
      "Native Cycles shadow setup requires the native geometry image.");
  return [scene](Var<luisa::compute::Ray> ray,
                 Var<ShadowIntersectionCall> intersection, Float ray_dP,
                 Float ray_dD, Var<ShadowShaderContextCall> context,
                 Var<RenderKernelParameters> parameters) noexcept {
    // Caustic suppression is conditioned on diffuse visibility in Cycles.
    // SURFACE_SHADOW has no diffuse visibility bit, so both configurations
    // have the same effective behavior here.
    const PathCyclesSvmKernelGlobals kg{scene, parameters,
                                        scene->camera.projection, true, true};
    auto setup = setup_cycles_svm_shadow_shader_data(
        scene, kg, ray, intersection, ray_dP, ray_dD, context, parameters);
    auto &sd = setup.shader_data;
    Var<ShadowSurfaceEvaluationCall> surface;
    surface.object = sd.object;
    surface.primitive = sd.prim;
    surface.kind = select(geometry_kind_triangle, geometry_kind_curve,
                          (sd.type & unsigned(abi::PRIMITIVE_CURVE)) != 0u);
    const auto volume_only =
        (sd.flag & static_cast<unsigned>(abi::SD_HAS_ONLY_VOLUME)) != 0u;
    surface.volume_boundary = volume_only.cast<unsigned>();
    $if(!volume_only) {
      const svm::PathState state{
          svm::path_ray_visibility_shadow, 0u,
          context.path.ray_depth,          context.path.transparent_depth,
          context.path.diffuse_depth,      context.path.glossy_depth,
          context.path.transmission_depth, 0u};
      svm::EvaluationResult result;
      svm::eval_nodes(kg, *scene->cycles_svm->word_buffer,
                      abi::SHADER_TYPE_SURFACE,
                      scene->cycles_svm->kernel_features,
                      svm::kernel_feature_node_mask_surface_shadow,
                      scene->cycles_svm->compilation.table.node_types_used,
                      setup.transforms, sd, state, result,
                      std::max<std::size_t>(
                          1u, scene->cycles_svm->compilation.table.peak_stack_usage));
      $if(result.status !=
          static_cast<unsigned>(svm::EvaluationStatus::ended)) {
        dsl::unreachable("native Cycles shadow SVM did not reach NODE_END");
      };
    };
    // surface_shader_transparency and the explicit ray-portal shadow veto.
    // Transparency is a spectrum; Cycles does not clamp it to [0, 1].
    $if((sd.flag & static_cast<unsigned>(abi::SD_RAY_PORTAL)) == 0u) {
      $if(volume_only) { surface.transmittance = make_float3(1.0f); }
      $elif((sd.flag & static_cast<unsigned>(abi::SD_TRANSPARENT)) != 0u) {
        surface.transmittance = sd.closure_transparent_extinction;
      };
    };
    return surface;
  };
}

} // namespace psycles::luisa_backend::detail
