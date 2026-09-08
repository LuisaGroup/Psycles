/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#include "path_tracer_cycles_svm_background.h"

#include "cycles_svm_internal.h"
#include "path_tracer_cycles_svm_emission.h"
#include "path_tracer_cycles_svm_kernel_globals.h"

#include <psycles/luisa/background_sampling.h>

namespace psycles::luisa_backend::detail {
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

std::uint32_t cycles_svm_background_feature_mask(
    CyclesSvmBackgroundEvaluation evaluation) noexcept {
  switch (evaluation) {
    case CyclesSvmBackgroundEvaluation::forward:
      return svm::kernel_feature_node_mask_surface_background;
    case CyclesSvmBackgroundEvaluation::light:
      return svm::kernel_feature_node_mask_surface_light;
    case CyclesSvmBackgroundEvaluation::importance_bake:
      return svm::kernel_feature_node_mask_surface_light &
             ~(svm::kernel_feature_node_raytrace | svm::kernel_feature_node_light_path);
  }
  LUISA_ERROR_WITH_LOCATION("Invalid background evaluation domain.");
}

svm::ShaderData setup_cycles_svm_background_shader_data(
    const LuisaSceneData &scene, Float3 origin, Float3 direction,
    Float differential, Float time, UInt visibility,
    CyclesSvmBackgroundEvaluation evaluation) noexcept {
  if (evaluation != CyclesSvmBackgroundEvaluation::importance_bake) {
    const auto map_dD = scene.background_map_weight > 0.0f
                           ? std::min(pi / scene.background_map_height,
                                      2.0f * pi / scene.background_map_width)
                           : ray_maximum;
    if (evaluation == CyclesSvmBackgroundEvaluation::forward) {
      differential = select(min(differential, map_dD), differential,
                            (visibility & svm::path_ray_visibility_camera) != 0u);
    } else {
      differential = min(differential, map_dD);
    }
  }
  const auto normal = -direction;
  const auto basis = svm::detail::differential_from_compact(normal, 1.0f);
  const auto identity = make_float4x4(1.0f);
  const auto shader = scene.cycles_background_shader_id;
  svm::ShaderData sd{
      direction, normal, normal, normal, 0u, shader,
      (*scene.cycles_svm->kernel_shader_buffer)
          ->read(shader & svm::shader_mask).flags.cast<unsigned>(),
      0u, svm::primitive_none, 0.0f, 0.0f, svm::object_none, time,
      ray_maximum, differential, differential,
      differential, 0.0f, 0.0f, differential,
      basis.dx, basis.dy, identity, identity};
  sd.ray_P = origin;
  return sd;
}

CyclesSvmBackgroundBakeRay cycles_svm_background_bake_ray(
    Float u, Float v, std::uint32_t width, std::uint32_t height) noexcept {
  const auto direction = background_sampling::equirectangular_to_direction(u, v);
  const auto du = background_sampling::equirectangular_to_direction(
      u + 1.0f / width, v);
  const auto dv = background_sampling::equirectangular_to_direction(
      u, v + 1.0f / height);
  return {direction, 0.5f * (length(du - direction) + length(dv - direction))};
}

Float3 evaluate_cycles_svm_background_emission(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    Float3 origin, Float3 direction, Float differential, Float time,
    const svm::PathState &state, UInt lcg_state,
    CyclesSvmBackgroundEvaluation evaluation) noexcept {
  Float3 emission = make_float3(0.0f);
  Bool constant = false;
  // kernel_background_evaluate deliberately runs the full shader even if its
  // output is constant. Forward and sampled emission use KernelShader's fast path.
  if (evaluation != CyclesSvmBackgroundEvaluation::importance_bake) {
    constant = cycles_svm_constant_emission(
        *scene, scene->cycles_background_shader_id, emission);
  }
  $if(!constant) {
    const PathCyclesSvmKernelGlobals kg{
        scene, parameters, scene->camera.projection, true, true};
    auto sd = setup_cycles_svm_background_shader_data(
        *scene, origin, direction, differential, time, state.visibility, evaluation);
    sd.lcg_state = lcg_state;
    const auto identity = make_float4x4(1.0f);
    const svm::TransformState transforms{parameters.camera_transform,
                                         parameters.camera_inverse_transform,
                                         identity, identity};
    const auto usage = scene->cycles_svm->compilation.table.usage_for(
        abi::SHADER_TYPE_SURFACE);
    svm::eval_nodes_assume_valid(
        kg, *scene->cycles_svm->word_buffer, abi::SHADER_TYPE_SURFACE,
        scene->cycles_svm->kernel_features, cycles_svm_background_feature_mask(evaluation),
        usage.node_types_used,
        transforms, sd, state,
        std::max<std::size_t>(1u, usage.peak_stack_usage));
    $if((sd.flag & svm::shader_data_emission) != 0u) {
      emission = sd.closure_emission_background;
    };
  };
  return emission;
}

} // namespace psycles::luisa_backend::detail
