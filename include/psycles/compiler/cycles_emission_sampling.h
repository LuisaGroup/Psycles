#pragma once

#include <psycles/contract/scene.h>
#include <psycles/core/math.h>

namespace psycles::compiler {

// Resolve Blender's authored sampling policy to the effective policy stored
// by Cycles after Shader::estimate_emission(). The estimate is metadata only:
// it controls emitter discovery and never replaces device-side graph
// evaluation. A value-to-closure conversion receives Cycles' 0.1 importance
// scale before the strict AUTO threshold is tested.
[[nodiscard]] contract::EmissionSampling
resolve_cycles_emission_sampling(
    contract::EmissionSampling authored,
    Vec3f emission_estimate,
    bool from_auto_conversion = false) noexcept;

}// namespace psycles::compiler
