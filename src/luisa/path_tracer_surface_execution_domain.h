#pragma once

#include "path_tracer_internal.h"

#include <cstdint>
#include <span>

namespace psycles::luisa_backend::detail {

enum class SurfaceValueProgramDomain {
  preparation,
  emission,
  bssrdf,
};

struct SurfaceValueProgramDomainView {
  std::span<const std::uint32_t> value_variants;
  std::span<const std::uint32_t> normal_variants;
  std::uint32_t program_offset{};
  bool automatic_normal_is_conditional{};
};

[[nodiscard]] SurfaceValueProgramDomainView
surface_value_program_domain(const SurfaceValueRuntime &runtime,
                             SurfaceValueProgramDomain domain) noexcept;

enum class SurfaceClosureProgramDomain {
  population,
  bssrdf,
};

struct SurfaceClosureProgramDomainView {
  std::span<const std::uint32_t> static_variants;
  compiler::PrincipledClosureFeatureMask principled_features{};
};

[[nodiscard]] SurfaceClosureProgramDomainView
surface_closure_program_domain(const SurfaceValueRuntime &runtime,
                               SurfaceClosureProgramDomain domain) noexcept;

} // namespace psycles::luisa_backend::detail
