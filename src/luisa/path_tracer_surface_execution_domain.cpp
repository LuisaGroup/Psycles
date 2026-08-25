#include "path_tracer_surface_execution_domain.h"

#include <cstdlib>

namespace psycles::luisa_backend::detail {

SurfaceValueProgramDomainView
surface_value_program_domain(const SurfaceValueRuntime &runtime,
                             SurfaceValueProgramDomain domain) noexcept {
  switch (domain) {
  case SurfaceValueProgramDomain::preparation:
    return {.value_variants = runtime.preparation_value_static_variants,
            .normal_variants = runtime.normal_value_static_variants,
            .height_variants = runtime.height_value_static_variants,
            .program_offset = SurfaceValueRuntime::preparation_program_offset};
  case SurfaceValueProgramDomain::emission:
    return {.value_variants = runtime.emission_value_static_variants,
            .normal_variants = runtime.emission_normal_value_static_variants,
            .height_variants = runtime.emission_height_value_static_variants,
            .program_offset = SurfaceValueRuntime::emission_program_offset,
            .automatic_normal_is_conditional = true};
  case SurfaceValueProgramDomain::bssrdf:
    return {.value_variants = runtime.bssrdf_value_static_variants,
            .normal_variants = runtime.bssrdf_normal_value_static_variants,
            .height_variants = runtime.bssrdf_height_value_static_variants,
            .program_offset = SurfaceValueRuntime::preparation_program_offset};
  }
  std::abort();
}

SurfaceClosureProgramDomainView
surface_closure_program_domain(const SurfaceValueRuntime &runtime,
                               SurfaceClosureProgramDomain domain) noexcept {
  switch (domain) {
  case SurfaceClosureProgramDomain::population:
    return {.static_variants = runtime.closure_static_variants,
            .principled_features = runtime.used_principled_closure_features};
  case SurfaceClosureProgramDomain::bssrdf:
    return {.static_variants = runtime.bssrdf_closure_static_variants,
            .principled_features = runtime.bssrdf_principled_closure_features};
  }
  std::abort();
}

} // namespace psycles::luisa_backend::detail
