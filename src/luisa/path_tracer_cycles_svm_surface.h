#pragma once

#include "path_tracer_surfaces.h"

#include <memory>

namespace psycles::luisa_backend::detail {

// Constructs the production surface component backed by the uploaded Cycles
// 5.2.1 DeviceScene image. The component evaluates one native SVM word stream
// into one retained ShaderData closure pool per path hit.
[[nodiscard]] std::shared_ptr<const SurfacePopulationComponent>
make_cycles_svm_surface_population_component(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

} // namespace psycles::luisa_backend::detail
