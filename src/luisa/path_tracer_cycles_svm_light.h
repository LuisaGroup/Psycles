#pragma once

#include "path_kernel_direct_light_task.h"

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::detail {

// ShaderData lives only inside SHADE_LIGHT_NEE. Exposing this host/JIT result
// lets the Cycles oracle regression inspect the complete setup, independently
// of which fields a particular material happens to read.
struct CyclesSvmLightShaderData {
  cycles_svm::ShaderData shader_data;
  cycles_svm::TransformState transforms;
  Bool background;
};

[[nodiscard]] CyclesSvmLightShaderData setup_cycles_svm_light_shader_data(
    const std::shared_ptr<LuisaSceneData> &scene,
    const cycles_svm::KernelGlobals &kernel_globals,
    const Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) noexcept;

// Host/JIT component. The shadow task is the complete device-side input;
// no surface closure, sampled UV, or emitter ShaderData crosses the cut.
class DirectLightEmissionComponent {
public:
  virtual ~DirectLightEmissionComponent() noexcept = default;
  [[nodiscard]] virtual Float3
  evaluate(const Var<DirectLightTaskCall> &task,
           const Var<RenderKernelParameters> &parameters) const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const DirectLightEmissionComponent>
make_cycles_svm_light_emission_component(
    const std::shared_ptr<LuisaSceneData> &scene,
    CameraProjection camera_projection, bool reflective_caustics,
    bool refractive_caustics) noexcept;

} // namespace psycles::luisa_backend::detail
