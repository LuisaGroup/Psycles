#pragma once

#include <psycles/contract/shader_graph.h>
#include <array>
#include <cstdint>
#include <string_view>

namespace psycles::test_support {
// Input graphs for the surface preparation regression, not a CPU evaluator.
[[nodiscard]] contract::ShaderGraph make_layered_principled_graph();
[[nodiscard]] contract::ShaderGraph make_automatic_normal_graph();
[[nodiscard]] contract::ShaderGraph make_bssrdf_bump_graph(bool zero_weight);
[[nodiscard]] contract::ShaderGraph make_nested_bump_graph();
[[nodiscard]] contract::ShaderGraph make_capacity_transparency_graph(std::uint32_t prefix_count = 11u);
[[nodiscard]] contract::ShaderGraph make_transformed_emission_graph(
    const std::array<float, 16u> &world_to_object, std::string_view label);
[[nodiscard]] contract::ShaderGraph make_light_path_depth_emission_graph(bool portal);
} // namespace psycles::test_support
