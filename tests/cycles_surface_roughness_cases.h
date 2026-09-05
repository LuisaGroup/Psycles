#pragma once

#include <array>
#include <cstdint>

namespace psycles::test_support {

// Input data shared with the external Cycles HIP probe, not a shading oracle.
// IDs are Cycles 5.2.1 ClosureType values. The roughness field is the common
// post-setup AOV value; anisotropic microfacets carry alpha_x=.04, alpha_y=.64.
struct RoughnessClosureInput {
    std::uint32_t type{};
    std::array<float, 3u> weight{0.25f, 0.64f, 1.0f};
    float roughness{0.4f};
    float alpha_x{0.04f};
    float alpha_y{0.64f};
};

struct RoughnessCase {
    std::uint32_t count{};
    std::array<RoughnessClosureInput, 2u> closures{};
};

inline constexpr std::array roughness_single_types{
    7u, 8u, 9u, 12u, 13u, 14u, 15u, 16u, 18u, 19u, 20u, 21u,
    22u, 23u, 24u, 25u, 26u, 27u, 28u, 29u, 31u, 33u, 34u, 35u};
inline constexpr auto roughness_cases = [] {
    std::array<RoughnessCase, 10u + roughness_single_types.size()> cases{};
    const auto single = [&](std::size_t index, std::uint32_t type) {
        cases[index].count = 1u;
        cases[index].closures[0].type = type;
    };
    single(1u, 32u); // True BSSRDF: no contributing BSDF.
    single(2u, 3u);  // Oren-Nayar.
    single(3u, 4u);  // Rough translucent.
    single(4u, 3u);  // Thin-wall subsurface's physical pair.
    cases[4].count = 2u;
    cases[4].closures[1].type = 4u;
    single(5u, 2u);  // Lambert is skipped by the roughness pass.
    cases[6] = cases[4];
    cases[6].closures[0].roughness = 0.2f;
    cases[6].closures[1].roughness = 0.7f;
    for (auto channel = 0u; channel < 3u; ++channel) {
        cases[4].closures[0].weight[channel] *= 0.4f;
        cases[4].closures[1].weight[channel] *= 0.6f;
        cases[6].closures[0].weight[channel] *= 0.25f;
        cases[6].closures[1].weight[channel] *= 0.75f;
    }
    single(7u, 30u); // Transparent is singular, not absent.
    cases[8] = cases[6];
    cases[8].closures[0].weight = {-1.0f, -2.0f, -3.0f};
    cases[8].closures[1].weight = {1.0f, 2.0f, 3.0f};
    cases[9] = cases[6];
    cases[9].closures[0].weight = {-1.0f, 0.0f, 1.0f};
    cases[9].closures[1].weight = {0.0f, 0.0f, 0.0f};
    for (auto i = 0u; i < roughness_single_types.size(); ++i) {
        single(10u + i, roughness_single_types[i]);
    }
    return cases;
}();

} // namespace psycles::test_support
