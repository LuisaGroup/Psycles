/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#include "path_tracer_cycles_svm_volume.h"
#include "path_tracer_cycles_svm_kernel_globals.h"
#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::detail {
namespace svm = cycles_svm;
namespace abi = compiler::cycles_svm;

svm::ShaderData setup_cycles_svm_volume_shader_data(
    Float3 origin, Float3 direction, Float minimum, Float time, UInt object,
    svm::ClosurePool *closures) noexcept {
  const auto identity = make_float4x4(1.0f);
  const auto zero = make_float3(0.0f);
  svm::ShaderData sd{
      origin + direction * minimum, -direction, -direction, -direction,
      static_cast<unsigned>(abi::PRIMITIVE_VOLUME), ~0u, 0u, 0u,
      svm::primitive_none, 0.0f, 0.0f, object, time, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f, zero, zero, identity, identity, 0u, closures};
  sd.ray_P = origin;
  return sd;
}

Bool evaluate_cycles_svm_volume_entry(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    const VolumeStackEntry &entry, svm::ShaderData &sd,
    const svm::PathState &state, bool shadow, std::uint32_t node_feature_mask) noexcept {
  const auto valid = entry.shader != ~0u;
  $if(valid) {
    sd.object = entry.object;
    sd.shader = entry.shader;
    sd.flag &= ~static_cast<unsigned>(abi::SD_SHADER_FLAGS);
    sd.flag |= (*scene->cycles_svm->kernel_shader_buffer)
                   ->read(sd.shader & svm::shader_mask).flags.cast<unsigned>();
    sd.object_flag &= ~static_cast<unsigned>(abi::SD_OBJECT_FLAGS);
    Bool visible = true;
    $if(sd.object != svm::object_none) {
      sd.object_flag |= scene->cycles_svm->objects->object_flag_buffer->read(sd.object);
      if (shadow) {
        const auto object = scene->cycles_svm->objects->object_buffer->read(sd.object);
        visible = (object.visibility & state.visibility) != 0u;
      }
    };
    $if(visible) {
      const auto identity = make_float4x4(1.0f);
      luisa::compute::Float4x4 object_to_world = identity, world_to_object = identity;
      $if(sd.object != svm::object_none) {
        const auto object = scene->cycles_svm->objects->object_buffer->read(sd.object);
        const auto unpack = [](Var<abi::PackedTransform> t) {
          return svm::detail::transform_from_rows(
              make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
              make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
              make_float4(t.z.x, t.z.y, t.z.z, t.z.w));
        };
        // Static admitted object image; invisible shadow entries return
        // before transform setup, exactly as volume_shader_eval_entry.
        object_to_world = unpack(object.tfm);
        world_to_object = unpack(object.itfm);
      };
      const PathCyclesSvmKernelGlobals kg{
          scene, parameters, scene->camera.projection, true, true};
      const svm::TransformState transforms{
          parameters.camera_transform, parameters.camera_inverse_transform,
          object_to_world, world_to_object};
      svm::eval_nodes_assume_valid(
          kg, *scene->cycles_svm->word_buffer, abi::SHADER_TYPE_VOLUME,
          scene->cycles_svm->kernel_features, node_feature_mask,
          scene->cycles_svm->compilation.table.node_types_used, transforms,
          sd, state, std::max<std::size_t>(
                         1u, scene->cycles_svm->compilation.table.peak_stack_usage));
    };
  };
  return valid;
}

void evaluate_cycles_svm_volume_stack(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters, const VolumeStack &stack,
    svm::ShaderData &sd, const svm::PathState &state, bool shadow) noexcept {
  if (sd.closure != nullptr) {
    sd.closure->reset();
    $if(((state.visibility & svm::path_ray_visibility_shadow) != 0u) |
        ((state.flag & (svm::path_ray_terminate | svm::path_ray_emission)) != 0u)) {
      sd.closure->set_left(0u);
    };
  }
  sd.flag = svm::shader_data_is_volume_shader_eval |
            (sd.flag & static_cast<unsigned>(abi::SD_CACHE_MISS));
  sd.object_flag = 0u;
  UInt i = 0u;
  $loop {
    const auto entry = stack.entry(i);
    $if(!evaluate_cycles_svm_volume_entry(scene, parameters, entry, sd, state, shadow)) {
      $break;
    };
    if (!shadow && sd.closure != nullptr) {
      $if(i > 0u) { sd.closure->merge_volume_closures(); };
    }
    i += 1u;
  };
}

VolumeCoefficients cycles_svm_volume_coefficients(const svm::ShaderData &sd) noexcept {
  auto result = VolumeCoefficients::zero();
  result.has_extinction = (sd.flag & svm::shader_data_extinction) != 0u;
  result.has_emission = (sd.flag & svm::shader_data_emission) != 0u;
  result.has_scatter = (sd.flag & svm::shader_data_scatter) != 0u;
  $if(result.has_extinction) { result.sigma_t = sd.closure_transparent_extinction; };
  $if(result.has_emission) { result.emission = sd.closure_emission_background; };
  if (sd.closure != nullptr) {
    $if(result.has_scatter) {
      UInt i = 0u;
      $while(i < sd.closure->count()) {
        const auto phase = sd.closure->volume_common(i);
        $if((phase.type >= static_cast<unsigned>(abi::CLOSURE_VOLUME_ID)) &
            (phase.type <= static_cast<unsigned>(abi::CLOSURE_VOLUME_DRAINE_ID))) {
          result.sigma_s += phase.weight;
        };
        i += 1u;
      };
    };
  }
  return result;
}
Bool cycles_svm_volume_is_homogeneous(
    const std::shared_ptr<LuisaSceneData> &scene, const VolumeStackEntry &entry) noexcept {
  const auto flags = (*scene->cycles_svm->kernel_shader_buffer)
      ->read(entry.shader & svm::shader_mask).flags.cast<unsigned>();
  Bool homogeneous = (flags & static_cast<unsigned>(abi::SD_HETEROGENEOUS_VOLUME)) == 0u;
  $if(homogeneous & ((flags & static_cast<unsigned>(abi::SD_NEED_VOLUME_ATTRIBUTES)) != 0u) &
      (entry.object != svm::object_none)) {
    homogeneous = (scene->cycles_svm->objects->object_flag_buffer->read(entry.object) &
                   static_cast<unsigned>(abi::SD_OBJECT_HAS_VOLUME_ATTRIBUTES)) == 0u;
  };
  return homogeneous;
}

PathCyclesSvmVolumeShader::PathCyclesSvmVolumeShader(
    std::shared_ptr<LuisaSceneData> scene,
    const Var<RenderKernelParameters> &parameters,
    Float3 origin, Float3 direction, Float minimum, Float time, UInt object,
    const svm::PathState &state, UInt lcg_state, bool shadow,
    bool coefficients_only) noexcept
    : _scene{std::move(scene)}, _parameters{parameters},
      _closures{shadow || coefficients_only ? 0u : _scene->cycles_svm->compilation.max_closures},
      _sd{setup_cycles_svm_volume_shader_data(
          origin, direction, minimum, time, object, &_closures)},
      _state{state}, _shadow{shadow} {
  _sd.flag = svm::shader_data_is_volume_shader_eval;
  _sd.lcg_state = lcg_state;
}

VolumeCoefficients PathCyclesSvmVolumeShader::evaluate(
    const VolumeStack &stack, Float3 position, VolumePhaseSet *phases) const noexcept {
  _sd.P = position;
  evaluate_cycles_svm_volume_stack(_scene, _parameters, stack, _sd, _state, _shadow);
  if (phases != nullptr) { phases->copy_from(_closures); }
  return cycles_svm_volume_coefficients(_sd);
}

VolumeCoefficients PathCyclesSvmVolumeShader::evaluate_entry(
    const VolumeStackEntry &entry, Float3 position, std::uint32_t mask) const noexcept {
  _sd.P = position;
  _sd.closure->set_left(0u);
  _sd.closure_transparent_extinction = make_float3(0.0f);
  _sd.closure_emission_background = make_float3(0.0f);
  static_cast<void>(evaluate_cycles_svm_volume_entry(
      _scene, _parameters, entry, _sd, _state, _shadow, mask));
  auto result = VolumeCoefficients::zero();
  result.sigma_t = _sd.closure_transparent_extinction;
  result.emission = _sd.closure_emission_background;
  return result;
}

UInt PathCyclesSvmVolumeShader::shader_flags(const VolumeStackEntry &entry) const noexcept {
  return (*_scene->cycles_svm->kernel_shader_buffer)->read(entry.shader & svm::shader_mask)
      .flags.cast<unsigned>();
}

Float PathCyclesSvmVolumeShader::object_density(const VolumeStackEntry &entry) const noexcept {
  Float density = 1.0f;
  $if(entry.object != svm::object_none) {
    density = _scene->cycles_svm->objects->object_buffer->read(entry.object).volume_density;
  };
  return density;
}

luisa::compute::Float4x4 PathCyclesSvmVolumeShader::object_transform(
    const VolumeStackEntry &entry, bool inverse) const noexcept {
  luisa::compute::Float4x4 result = make_float4x4(1.0f);
  $if(entry.object != svm::object_none) {
    const auto object = _scene->cycles_svm->objects->object_buffer->read(entry.object);
    const auto t = inverse ? object.itfm : object.tfm;
    result = svm::detail::transform_from_rows(
        make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
        make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
        make_float4(t.z.x, t.z.y, t.z.z, t.z.w));
  };
  return result;
}

Bool PathCyclesSvmVolumeShader::camera_ray() const noexcept {
  return (_state.visibility & static_cast<unsigned>(abi::PATH_RAY_VISIBILITY_CAMERA)) != 0u;
}

Bool PathCyclesSvmVolumeShader::homogeneous(const VolumeStackEntry &entry) const noexcept {
  return cycles_svm_volume_is_homogeneous(_scene, entry);
}
} // namespace psycles::luisa_backend::detail
