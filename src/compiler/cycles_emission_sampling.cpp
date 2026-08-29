#include <psycles/compiler/cycles_emission_sampling.h>

#include <algorithm>
#include <cmath>

namespace psycles::compiler {

contract::EmissionSampling resolve_cycles_emission_sampling(
    contract::EmissionSampling authored,
    Vec3f emission_estimate,
    bool from_auto_conversion) noexcept {
    using contract::EmissionSampling;
    if (emission_estimate == Vec3f{}) {
        return EmissionSampling::none;
    }
    if (authored != EmissionSampling::automatic) {
        return authored;
    }
    const auto scale = from_auto_conversion ? 0.1f : 1.0f;
    const auto importance = std::max({
        std::abs(emission_estimate.x * scale),
        std::abs(emission_estimate.y * scale),
        std::abs(emission_estimate.z * scale)});
    return importance > 0.5f
               ? EmissionSampling::front_back
               : EmissionSampling::none;
}

}// namespace psycles::compiler
