#pragma once
#include <array>

inline constexpr unsigned probe_count = 32u;
struct ProbeInput { std::array<float, 4> coordinate, parameters; };
inline auto probe_inputs() {
    std::array<ProbeInput, probe_count> result{};
    for (unsigned i = 0; i < probe_count; ++i) {
        const auto f = static_cast<float>(i);
        result[i] = {{0.37f + 0.713f * f, -1.25f - 0.17f * f, 2.5f + 1.125f * f, 0.0f},
                     {static_cast<float>(i % 8u) + (i % 3u == 0u ? 0.25f : 0.0f),
                      0.35f + static_cast<float>(i % 4u) * 0.15f, 2.0f,
                      i % 2u == 0u ? 0.0f : 1.0f}};
    }
    result[28].coordinate = {99999.5f, 100000.0f, -100000.5f, 0.0f};
    result[29].coordinate = {999999.5f, 1000000.0f, -1000000.5f, 0.0f};
    result[30].coordinate = {1.0e12f, -1.0e17f, 1.0e30f, 0.0f};
    result[31].coordinate = {0.0f, -0.0f, 0.125f, 0.0f};
    return result;
}
