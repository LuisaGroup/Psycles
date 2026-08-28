#pragma once

#include "path_tracer_internal.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace psycles::luisa_backend::detail {

// Build is transactional: either every topology, automatic-normal domain,
// and Bump height subprogram fits the typed runtime contract, or no runtime is
// returned. This prevents a frame from mixing two value semantics.
[[nodiscard]] std::unique_ptr<SurfaceValueRuntime>
build_surface_value_runtime(
    luisa::compute::Device &device,
    std::span<const std::shared_ptr<const compiler::SurfaceProgram>> programs,
    std::span<const compiler::SurfaceClosurePlan> closure_plans,
    std::span<const std::uint32_t> bssrdf_bump_tags,
    std::string &diagnostic,
    std::uint32_t region_handler_site_budget = 0u);

void upload_surface_value_runtime(
    Stream &stream,
    SurfaceValueRuntime &runtime) noexcept;

} // namespace psycles::luisa_backend::detail
