/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#include "path_tracer_cycles_svm_emission.h"

#include "cycles_svm_internal.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_tracer_cycles_svm_shader_data.h"

#include <algorithm>

namespace psycles::luisa_backend::detail {
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

Bool cycles_svm_constant_emission(const LuisaSceneData &scene, UInt shader,
                                  Float3 &emission) noexcept {
  const auto record = (*scene.cycles_svm->kernel_shader_buffer)->read(
      shader & svm::shader_mask);
  const auto constant =
      (record.flags & static_cast<int>(abi::SD_HAS_CONSTANT_EMISSION)) != 0;
  $if(constant) {
    const auto value = record.constant_emission;
    emission = make_float3(value.x, value.y, value.z);
  };
  return constant;
}

Float3 evaluate_cycles_svm_lamp_emission(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    UInt shader, UInt object, UInt primitive,
    Float3 position, Float3 normal, Float3 incoming, Float2 uv,
    Float distance, Float time, const svm::PathState &state) noexcept {
  Float3 emission = make_float3(0.0f);
  const auto constant = cycles_svm_constant_emission(*scene, shader, emission);
  $if(!constant) {
    // light_sample_shader_eval_forward uses visibility NONE and EMISSION.
    // Caustic suppression therefore cannot be active in this domain.
    const PathCyclesSvmKernelGlobals kg{
        scene, parameters, scene->camera.projection, true, true};
    const auto identity = make_float4x4(1.0f);
    svm::TransformState transforms{parameters.camera_transform,
                                   parameters.camera_inverse_transform,
                                   identity, identity};
    svm::ShaderData sd{
        position, normal, normal, incoming,
        static_cast<unsigned>(abi::PRIMITIVE_LAMP), shader,
        (*scene->cycles_svm->kernel_shader_buffer)
            ->read(shader & svm::shader_mask).flags.cast<unsigned>(),
        0u, primitive, uv.x, uv.y, object, time, distance,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        make_float3(0.0f), make_float3(0.0f), identity, identity};
    $if(object != svm::object_none) {
      const auto record = scene->cycles_svm->objects->object_buffer->read(object);
      const auto unpack = [](Var<abi::PackedTransform> value) noexcept {
        return svm::detail::transform_from_rows(
            make_float4(value.x.x, value.x.y, value.x.z, value.x.w),
            make_float4(value.y.x, value.y.y, value.y.z, value.y.w),
            make_float4(value.z.x, value.z.y, value.z.z, value.z.w));
      };
      sd.object_flag = scene->cycles_svm->objects->object_flag_buffer->read(object);
      transforms.object_to_world = unpack(record.tfm);
      transforms.world_to_object = unpack(record.itfm);
    };
    $if(primitive != svm::primitive_none) {
      cycles_svm_shader_setup_backfacing(sd);
    };
    const auto usage = scene->cycles_svm->compilation.table.usage_for(
        abi::SHADER_TYPE_SURFACE);
    svm::eval_nodes_assume_valid(
        kg, *scene->cycles_svm->word_buffer, abi::SHADER_TYPE_SURFACE,
        scene->cycles_svm->kernel_features,
        svm::kernel_feature_node_mask_surface_light,
        usage.node_types_used,
        transforms, sd, state,
        std::max<std::size_t>(1u, usage.peak_stack_usage), usage.noise_usage);
    $if(((sd.flag & svm::shader_data_emission) != 0u) &
        (abs(dot(sd.Ng, sd.wi)) > 0.0f)) {
      emission = sd.closure_emission_background;
    };
  };
  return emission;
}

} // namespace psycles::luisa_backend::detail
