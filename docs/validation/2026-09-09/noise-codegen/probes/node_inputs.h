#pragma once
#include "inputs.h"
#include <bit>
#include <cstdint>

template<typename Node>
auto node_inputs() {
    const auto inputs = probe_inputs();
    std::array<Node, probe_count> result{};
    for (unsigned i = 0; i < probe_count; ++i) {
        auto &n = result[i];
        const auto &q = inputs[i].parameters;
        n.dimensions = 3u;
        n.noise_type = static_cast<decltype(n.noise_type)>(1u);
        n.normalize = q[3] != 0.0f;
        n.w.bits = std::bit_cast<std::uint32_t>(0.0f);
        n.scale.bits = std::bit_cast<std::uint32_t>(1.0f);
        n.detail.bits = std::bit_cast<std::uint32_t>(q[0]);
        n.roughness.bits = i % 2u == 0u ? (0x7fc00000u | 3u) : std::bit_cast<std::uint32_t>(q[1]);
        n.lacunarity.bits = std::bit_cast<std::uint32_t>(q[2]);
        n.offset.bits = std::bit_cast<std::uint32_t>(0.37f);
        n.gain.bits = std::bit_cast<std::uint32_t>(1.11f);
        n.distortion.bits = std::bit_cast<std::uint32_t>(i % 3u == 0u ? 0.3f : 0.0f);
        n.vector = 0u;
        n.value_offset = 24u;
        n.color_offset = i < probe_count / 2u ? 25u : 255u;
    }
    return result;
}
