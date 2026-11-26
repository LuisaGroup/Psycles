#include "path_tracer_cycles_svm_shadow.h"

#include "cycles_svm_internal.h"
#include "path_tracer_cycles_svm_curve.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "virtual_texture_cache.h"
#include "path_tracer_cycles_svm_shader_data.h"

#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_svm_translator.h>

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
    const Var<RenderKernelParameters> &parameters,
    Var<InstanceGpu> *out_instance,
    svm::TriangleVertices *out_vertices,
    svm::TriangleNormals *out_normals) noexcept {
  const auto instance = scene->instance_buffer->read(instance_id);
  if (out_instance != nullptr) {
    *out_instance = instance;
  }
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
  svm::TransformState transforms{
      parameters.camera_transform,
      parameters.camera_inverse_transform,
      [scene, object_index, unpack]() noexcept {
        return unpack(
            scene->cycles_svm->objects->object_buffer->read(object_index).tfm);
      },
      [scene, object_index, unpack]() noexcept {
        return unpack(
            scene->cycles_svm->objects->object_buffer->read(object_index).itfm);
      }};
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
  const auto pos_offset = object.position_offset.cast<unsigned>();
  const auto triangle_setup = [&] {
    assume(sd.type == static_cast<unsigned>(abi::PRIMITIVE_TRIANGLE));
    sd.shader = scene->cycles_svm->geometry->triangle_shader_buffer->read(sd.prim);
    const auto indices = kg.triangle_vertex_indices(sd.prim);
    const auto &vertex_buf = scene->cycles_svm->geometry->triangle_vertex_buffer;
    const auto unpack_v = [](Var<abi::packed_float3> v) noexcept {
      return make_float3(v.x, v.y, v.z);
    };
    const svm::TriangleVertices vertices{
        .v0 = unpack_v(vertex_buf->read(pos_offset + indices.x)),
        .v1 = unpack_v(vertex_buf->read(pos_offset + indices.y)),
        .v2 = unpack_v(vertex_buf->read(pos_offset + indices.z))};
    if (out_vertices != nullptr) {
      *out_vertices = vertices;
    }
    cycles_svm_triangle_shader_setup(kg, transforms, vertices, sd, out_normals);
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
    const std::shared_ptr<LuisaSceneData> &scene,
    bool enable_vt) noexcept {
  LUISA_ASSERT(
      scene->cycles_svm && scene->cycles_svm->geometry &&
          scene->cycles_svm->objects,
      "Native Cycles shadow setup requires the native geometry image.");
  return [scene, enable_vt](Var<luisa::compute::Ray> ray,
                            Var<ShadowIntersectionCall> intersection,
                            Float ray_dP, Float ray_dD,
                            Var<ShadowShaderContextCall> context,
                            Var<RenderKernelParameters> parameters) noexcept {
    const auto backend_name = scene->device.backend_name();
    const bool is_simd = backend_name == "simd";
    const bool is_simd_wide = is_simd && scene->device.compute_warp_size() > 1u;
    const bool cpu_backend = backend_name == "fallback" || is_simd;
    Var<ShadowSurfaceEvaluationCall> surface;
    surface.vt_miss_page = vt_invalid_page;
    $outline_when_with_name(is_simd_wide, "simd_eval_shadow_surface") {
      // Caustic suppression is conditioned on diffuse visibility in Cycles.
      // SURFACE_SHADOW has no diffuse visibility bit, so both configurations
      // have the same effective behavior here.
      const PathCyclesSvmKernelGlobals kg{
          scene, parameters, scene->camera.projection, true, true,
          PathCyclesSvmShaderDomain::shadow_surface, {}, false,
          enable_vt ? &surface.vt_miss_page : nullptr};
      auto setup = setup_cycles_svm_shadow_shader_data(
          scene, kg, ray, intersection, ray_dP, ray_dD, context, parameters);
      auto &sd = setup.shader_data;
      surface.object = sd.object;
      surface.primitive = sd.prim;
      surface.kind = select(geometry_kind_triangle, geometry_kind_curve,
                            (sd.type & unsigned(abi::PRIMITIVE_CURVE)) != 0u);
      const auto volume_only =
          (sd.flag & static_cast<unsigned>(abi::SD_HAS_ONLY_VOLUME)) != 0u;
      surface.volume_boundary = volume_only.cast<unsigned>();
      const Bool eval_shadow_svm =
          cycles_svm_has_domain_metadata(*scene)
              ? (!volume_only &
                 ((sd.flag &
                   static_cast<unsigned>(abi::SD_HAS_TRANSPARENT_SHADOW |
                                         abi::SD_RAY_PORTAL)) != 0u))
              : !volume_only;
      $if(eval_shadow_svm) {
        const svm::PathState state{
            svm::path_ray_visibility_shadow, 0u,
            context.path.ray_depth,          context.path.transparent_depth,
            context.path.diffuse_depth,      context.path.glossy_depth,
            context.path.transmission_depth,
            (scene->cycles_svm->kernel_features & svm::kernel_feature_node_portal)
                ? UInt{context.path.portal_depth} : UInt{0u}};
        const auto usage = cycles_svm_domain_entry_usage(
            *scene, PathCyclesSvmShaderDomain::shadow_surface);
        const auto active_mask = cycles_svm_domain_shader_mask(
            *scene, PathCyclesSvmShaderDomain::shadow_surface);
        svm::translate_svm_nodes_assume_valid(
            svm::SvmTranslationRequest{
                .kernel_globals = kg,
                .table = scene->cycles_svm->compilation.table,
                .words = *scene->cycles_svm->word_buffer,
                .shader_type = abi::SHADER_TYPE_SURFACE,
                .kernel_features = scene->cycles_svm->kernel_features,
                .node_feature_mask = svm::kernel_feature_node_mask_surface_shadow,
                .node_types_used = usage.node_types_used,
                .transform_state = setup.transforms,
                .shader_data = sd,
                .path_state = state,
                .stack_size = std::max<std::size_t>(1u, usage.peak_stack_usage),
                .noise_usage = usage.noise_usage,
                .active_shader_mask = active_mask,
                .max_groups_per_outlined_chunk =
                    is_simd_wide
                        ? std::size_t{2u}
                        : (backend_name == "fallback" ? std::size_t{16u}
                                                      : std::size_t{0u}),
                .max_chunk_jit_weight =
                    is_simd_wide ? std::size_t{180u} : std::size_t{900u},
                .outline_chunk_dispatcher = !is_simd_wide,
                .shadow_eval_only = true},
            cpu_backend ? svm::SvmBackendStrategy::direct_jit
                        : svm::SvmBackendStrategy::auto_select);
      };
      // surface_shader_transparency and the explicit ray-portal shadow veto.
      // Transparency is a spectrum; Cycles does not clamp it to [0, 1].
      $if((sd.flag & static_cast<unsigned>(abi::SD_RAY_PORTAL)) == 0u) {
        $if(volume_only) { surface.transmittance = make_float3(1.0f); }
        $elif((sd.flag & static_cast<unsigned>(abi::SD_TRANSPARENT)) != 0u) {
          surface.transmittance = sd.closure_transparent_extinction;
        };
      };
    };
    return surface;
  };
}

} // namespace psycles::luisa_backend::detail
