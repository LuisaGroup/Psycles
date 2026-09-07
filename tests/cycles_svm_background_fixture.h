#pragma once

#include <psycles/compiler/cycles_svm_node_types.h>

#include <array>
#include <cstring>
#include <vector>

namespace psycles::test_support {

struct BackgroundInput {
  std::array<float, 3> origin;
  std::array<float, 3> direction;
  float differential;
  float time;
  float u;
  float v;
  unsigned visibility;
  unsigned flag;
};

inline constexpr std::array background_inputs{
    BackgroundInput{{1, 2, 3}, {0, 0, 1}, 0.3f, 0.25f, 0.25f, 0.5f, 1u, 0u},
    BackgroundInput{{-3, 2, 1}, {0.6f, 0, 0.8f}, 0.3f, 0.75f, 0.75f, 0.5f, 4u, 1u},
    BackgroundInput{{2, -1, 4}, {0, -1, 0}, 0.001f, 0.5f, 0.0f, 0.0f, 8u, 2u},
    BackgroundInput{{1, 3, -2}, {0, 0, -1}, 0.0f, 0.9f, 1.0f, 1.0f, 2u, 0u},
    BackgroundInput{{-4, -2, 3}, {0.577350269f, 0.577350269f, 0.577350269f},
                    0.07f, 0.1f, 0.98f, 0.98f, 16u, 0u},
    BackgroundInput{{2, 4, 6}, {-0.577350269f, -0.577350269f, -0.577350269f},
                    0.03f, 0.6f, 0.02f, 0.02f, 1u, 1u}};
inline constexpr unsigned background_width = 32u;
inline constexpr unsigned background_height = 16u;
inline constexpr unsigned background_shader = 1u << 30u;
inline constexpr unsigned background_shader_flags =
    compiler::cycles_svm::SD_HAS_EMISSION | compiler::cycles_svm::SD_USE_BUMP_MAP_CORRECTION;

// A shared typed input stream, not expected output or a reference evaluator.
// Geometry Position drives color; Light Path Ray Depth drives strength.
// Both implementations execute exactly these bytes in their own PC loop.
inline std::vector<unsigned> background_words(bool window = false) {
  namespace abi = compiler::cycles_svm;
  std::vector<unsigned> words{abi::NODE_SHADER_JUMP, 4u, 0u, 0u};
  const auto append = [&](const auto &payload) {
    const auto offset = words.size();
    words.resize(offset + sizeof(payload) / sizeof(unsigned));
    std::memcpy(words.data() + offset, &payload, sizeof(payload));
  };
  if (window) {
    words.push_back(abi::NODE_TEX_COORD);
    append(abi::SVMNodeTexCoord{abi::NODE_TEXCO_WINDOW, abi::NODE_BUMP_OFFSET_CENTER, 0u, 0u, 0.0f});
  } else {
    words.push_back(abi::NODE_GEOMETRY);
    append(abi::SVMNodeGeometry{abi::NODE_GEOM_P, abi::NODE_BUMP_OFFSET_CENTER, 0u, 0u, 0.0f});
  }
  words.push_back(abi::NODE_LIGHT_PATH);
  append(abi::SVMNodeLightPath{abi::NODE_LP_ray_depth, 3u, {}});
  abi::SVMNodeEmissionWeight emission{};
  emission.color.x.bits = 0x7fc00000u;
  emission.strength.bits = 0x7fc00003u;
  words.push_back(abi::NODE_EMISSION_WEIGHT);
  append(emission);
  words.push_back(abi::NODE_CLOSURE_BACKGROUND);
  append(abi::SVMNodeClosureBackground{255u, {}});
  words[2] = words[3] = unsigned(words.size());
  words.push_back(abi::NODE_END);
  return words;
}

} // namespace psycles::test_support
