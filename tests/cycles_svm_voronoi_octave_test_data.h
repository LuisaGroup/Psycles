#pragma once

#include <array>
#include <bit>
#include <cstdint>

// Inputs only, shared by the original HIP observer and Luisa regression.
// No expected shader values or host Voronoi implementation lives here.
namespace psycles::test_support::voronoi_octave {
inline constexpr unsigned case_count = 96u;
inline constexpr unsigned output_begin = 16u;
inline constexpr unsigned output_count = 9u;
inline constexpr unsigned stack_size = output_begin + output_count;

template<typename Node>
struct Case {
  Node node;
  std::array<float, 8u> inputs;
};

template<typename Node>
auto cases() {
  static_assert(sizeof(Node) == 13u * sizeof(std::uint32_t));
  std::array<Case<Node>, case_count> result{};
  for (unsigned i = 0; i < case_count; ++i) {
    auto &test = result[i];
    auto &node = test.node;
    const auto fractional = i % 2u != 0u;
    const auto smooth = (i / 4u) % 2u != 0u;
    test.inputs = {0.173f, -0.25f, 1.375f, 0.0f, -0.437f,
                   fractional ? 1.25f : 0.0f, 0.63f,
                   smooth ? 0.42f : (fractional ? -0.2f : 0.0f)};
    const auto bits = [](float value) { return std::bit_cast<std::uint32_t>(value); };
    node.dimensions = i / 24u + 1u;
    node.feature = static_cast<decltype(node.feature)>((i / 8u) % 3u);
    node.metric = static_cast<decltype(node.metric)>((i / 2u) % 4u);
    node.normalize = (i / 2u) % 2u;
    node.w.bits = 0x7fc00000u | 4u;
    node.scale.bits = bits(2.35f);
    node.detail.bits = 0x7fc00000u | 5u;
    node.roughness.bits = bits(test.inputs[6u]);
    node.lacunarity.bits = bits(2.17f);
    node.smoothness.bits = 0x7fc00000u | 7u;
    node.exponent.bits = bits(1.7f);
    node.randomness.bits = bits(0.81f);
    node.coord = 0u;
    node.distance_offset = output_begin;
    node.color_offset = output_begin + 1u;
    node.position_offset = output_begin + 4u;
    node.w_out_offset = output_begin + 7u;
    node.radius_offset = output_begin + 8u;
  }
  return result;
}
} // namespace psycles::test_support::voronoi_octave
