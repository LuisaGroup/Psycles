#pragma once

#include "path_tracer_surfaces.h"

#include <psycles/compiler/cycles_svm_compiler.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace psycles::luisa_backend::detail {

struct CyclesSvmSurfaceBucketSpecialization {
  std::vector<std::uint8_t> active_shader_mask;
  std::optional<compiler::cycles_svm::ShaderEntryUsage> entry_usage;
  std::size_t max_surface_closures{0u};
};

// Constructs the production surface component backed by the uploaded Cycles
// 5.2.1 DeviceScene image. The component evaluates one native SVM word stream
// into one retained ShaderData closure pool per path hit.
[[nodiscard]] std::shared_ptr<const SurfacePopulationComponent>
make_cycles_svm_surface_population_component(
    const std::shared_ptr<LuisaSceneData> &scene,
    CyclesSvmSurfaceBucketSpecialization bucket = {}) noexcept;

} // namespace psycles::luisa_backend::detail
