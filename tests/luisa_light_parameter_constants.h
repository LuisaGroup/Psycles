#pragma once
#include <psycles/luisa/analytic_light_parameters.h>

namespace psycles::test_support {
// Embed scene-owned fixture inputs as DSL constants. These functions only
// expose the production scene preparation result; they do not evaluate light.
inline auto light_parameter_constant(luisa_backend::AreaLightParameters p) {
  return luisa::compute::Var<luisa_backend::AreaLightParameters>{p.tan_half_spread, p.normalize_spread};
}
inline auto light_parameter_constant(luisa_backend::SpotLightParameters p) {
  return luisa::compute::Var<luisa_backend::SpotLightParameters>{p.cos_half_spot_angle,
      p.half_cot_half_spot_angle, p.spot_smooth, p.cos_half_larger_spread, p.ray_segment_dp};
}
} // namespace psycles::test_support
