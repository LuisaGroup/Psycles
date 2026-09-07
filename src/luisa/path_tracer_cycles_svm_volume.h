#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/volume_phase_set.h>
#include <psycles/luisa/volume_stack.h>
#include <psycles/luisa/volume_shader.h>
#include <psycles/luisa/volume_majorant_prepass.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] cycles_svm::ShaderData setup_cycles_svm_volume_shader_data(
    Float3 origin, Float3 direction, Float minimum, Float time,
    UInt object = cycles_svm::object_none,
    cycles_svm::ClosurePool *closures = nullptr) noexcept;

// Returns false only at SHADER_NONE. An invisible shadow entry remains in
// the stack and returns true without executing the shader.
[[nodiscard]] Bool evaluate_cycles_svm_volume_entry(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    const VolumeStackEntry &entry, cycles_svm::ShaderData &sd,
    const cycles_svm::PathState &state, bool shadow,
    std::uint32_t node_feature_mask = cycles_svm::kernel_feature_node_mask_volume) noexcept;

void evaluate_cycles_svm_volume_stack(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters, const VolumeStack &stack,
    cycles_svm::ShaderData &sd, const cycles_svm::PathState &state,
    bool shadow) noexcept;

[[nodiscard]] VolumeCoefficients cycles_svm_volume_coefficients(
    const cycles_svm::ShaderData &sd) noexcept;

[[nodiscard]] Bool cycles_svm_volume_is_homogeneous(
    const std::shared_ptr<LuisaSceneData> &scene,
    const VolumeStackEntry &entry) noexcept;

// One ShaderData lifetime for a complete volume ray/interval. In particular,
// stochastic texture LCG state survives successive entries and candidates.
class PathCyclesSvmVolumeShader final : public VolumeShaderEvaluator {
  std::shared_ptr<LuisaSceneData> _scene;
  const Var<RenderKernelParameters> &_parameters;
  cycles_svm::ClosurePool _closures;
  mutable cycles_svm::ShaderData _sd;
  cycles_svm::PathState _state;
  bool _shadow;
public:
  PathCyclesSvmVolumeShader(
      std::shared_ptr<LuisaSceneData> scene,
      const Var<RenderKernelParameters> &parameters,
      Float3 origin, Float3 direction, Float minimum, Float time, UInt object,
      const cycles_svm::PathState &state, UInt lcg_state, bool shadow,
      bool coefficients_only = false) noexcept;
  [[nodiscard]] VolumeCoefficients evaluate(
      const VolumeStack &stack, Float3 position,
      VolumePhaseSet *phases = nullptr) const noexcept override;
  [[nodiscard]] VolumeCoefficients evaluate_entry(
      const VolumeStackEntry &entry, Float3 position,
      std::uint32_t mask = cycles_svm::kernel_feature_node_mask_volume) const noexcept;
  [[nodiscard]] UInt shader_flags(const VolumeStackEntry &entry) const noexcept;
  [[nodiscard]] Float object_density(const VolumeStackEntry &entry) const noexcept;
  [[nodiscard]] luisa::compute::Float4x4 object_transform(
      const VolumeStackEntry &entry, bool inverse) const noexcept;
  [[nodiscard]] Bool camera_ray() const noexcept;
  [[nodiscard]] Bool homogeneous(const VolumeStackEntry &entry) const noexcept;
};

[[nodiscard]] VolumeMajorantCellExtrema evaluate_cycles_svm_volume_density_cell(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Var<RenderKernelParameters> &parameters,
    const VolumeStackEntry &entry, const VolumeMajorantGrid &grid,
    UInt cell_index) noexcept;

} // namespace psycles::luisa_backend::detail
