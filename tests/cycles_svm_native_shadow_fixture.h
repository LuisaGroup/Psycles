#pragma once

#include "cycles_svm_nee_setup_fixture.h"

#include <psycles/compiler/cycles_svm_node_types.h>

#include <bit>
#include <cstring>
#include <vector>

namespace psycles::test_support {
namespace shadow_abi = compiler::cycles_svm;

// Inputs only: exact typed node words, geometry and integrator state. Expected
// values are produced by the external Cycles HIP setup/interpreter/integrator.
struct NativeShadowInput {
  NeeSetupInput geometry{};
  float u{0.2f}, v{0.3f};
  shadow_abi::packed_float3 weight{0.2f, 0.5f, 0.8f};
  shadow_abi::ClosureType closure{shadow_abi::CLOSURE_BSDF_TRANSPARENT_ID};
  int light_path{-1};
  unsigned shader_flags{};
};

inline constexpr auto native_shadow_inputs = [] {
  namespace a = shadow_abi;
  std::array<NativeShadowInput, 25u> c{};
  for (auto i = 0u; i < 9u; ++i) {
    c[i].geometry = nee_setup_cases[i];
  }
  c[9].closure = a::CLOSURE_BSDF_DIFFUSE_ID;
  c[10].closure = a::CLOSURE_BSDF_RAY_PORTAL_ID;
  c[11].shader_flags = a::SD_HAS_ONLY_VOLUME;
  c[11].closure = a::CLOSURE_BSDF_DIFFUSE_ID;
  c[12].weight = {-0.5f, -0.25f, -1.5f};
  c[13].weight = {1.0f, 0.0f, 0.0f};
  c[14].weight = {0.0f, 1.0f, 0.0f};
  constexpr a::NodeLightPath queries[]{
      a::NODE_LP_ray_transparent, a::NODE_LP_shadow,      a::NODE_LP_camera,
      a::NODE_LP_diffuse,         a::NODE_LP_glossy,      a::NODE_LP_singular,
      a::NODE_LP_ray_depth,       a::NODE_LP_ray_diffuse, a::NODE_LP_ray_glossy,
      a::NODE_LP_ray_transmission};
  for (auto i = 15u; i < c.size(); ++i) {
    c[i].light_path = queries[i - 15u];
  }
  for (auto i = 0u; i < c.size(); ++i) {
    c[i].geometry.shader |= i;
  }
  return c;
}();

struct NativeShadowBatchInput {
  // Unsorted intersection input, deliberately testing the production sorting
  // network. Exactly four hits must re-enter INTERSECT_SHADOW in Cycles even
  // when the next traversal will be empty.
  std::array<unsigned, 4u> objects{0u, 1u, 2u, 3u};
  std::array<float, 4u> distances{4.0f, 1.0f, 3.0f, 2.0f};
  unsigned count{1u};
  shadow_abi::packed_float3 throughput{1.0f, 1.0f, 1.0f};
  unsigned transparent_bounce{3u};
  unsigned rng_offset{72u};
  unsigned volume_bounds_bounce{};
};

inline constexpr auto native_shadow_batches = [] {
  std::array<NativeShadowBatchInput, 14u> c{};
  c[0].count = 0u;
  c[2].count = 2u;
  c[3].objects[0] = 12u;
  c[4].objects = {13u, 14u, 0u, 0u};
  c[4].distances = {1.0f, 2.0f, 3.0f, 4.0f};
  c[4].count = 2u;
  c[5].objects = {0u, 9u, 0u, 0u};
  c[5].distances = c[4].distances;
  c[5].count = 2u;
  c[6].objects = {11u, 15u, 0u, 0u};
  c[6].distances = c[4].distances;
  c[6].count = 2u;
  c[7] = c[6];
  c[7].objects[1] = 11u;
  c[7].volume_bounds_bounce = 1023u;
  c[8].count = 0u;
  c[8].volume_bounds_bounce = 1025u;
  c[9].count = 4u;
  c[10].rng_offset = 65528u;
  c[11].transparent_bounce = 65535u;
  c[12].objects[0] = 14u;
  c[12].throughput = {1.0f, 0.0f, 0.0f};
  c[13].count = 4u;
  c[13].objects = {15u, 15u, 15u, 15u};
  return c;
}();

inline auto make_native_shadow_image() {
  namespace a = shadow_abi;
  std::vector<unsigned> words(native_shadow_inputs.size() * 4u);
  const auto append = [&words](const auto &payload) {
    static_assert(sizeof(payload) % sizeof(unsigned) == 0u);
    const auto offset = words.size();
    words.resize(offset + sizeof(payload) / sizeof(unsigned));
    std::memcpy(words.data() + offset, &payload, sizeof(payload));
  };
  const auto scalar = [](float x) {
    return a::SVMInputFloat{std::bit_cast<unsigned>(x)};
  };
  const auto color = [&scalar](a::packed_float3 x) {
    return a::SVMInputFloat3{scalar(x.x), scalar(x.y), scalar(x.z)};
  };
  for (auto i = 0u; i < native_shadow_inputs.size(); ++i) {
    const auto &c = native_shadow_inputs[i];
    words[i * 4u] = a::NODE_SHADER_JUMP;
    words[i * 4u + 1u] = static_cast<unsigned>(words.size());
    if (c.light_path < 0) {
      words.push_back(a::NODE_VALUE_F);
      append(a::SVMNodeValueF{1.0f, 0u, {}});
    } else {
      words.push_back(a::NODE_LIGHT_PATH);
      append(a::SVMNodeLightPath{
          static_cast<a::NodeLightPath>(c.light_path), 0u, {}});
    }
    words.push_back(a::NODE_CLOSURE_SET_WEIGHT);
    append(c.weight);
    words.push_back(a::NODE_CLOSURE_BSDF);
    append(a::SVMNodeClosureBsdf{c.closure, 0u, {}});
    if (c.closure == a::CLOSURE_BSDF_DIFFUSE_ID) {
      append(
          a::SVMNodeDiffuseBsdfData{color(c.weight), scalar(0.3f), 255u, {}});
    } else if (c.closure == a::CLOSURE_BSDF_RAY_PORTAL_ID) {
      append(a::SVMNodeRayPortalBsdfData{color({0.0f, 0.0f, 1.0f}), 255u, {}});
    } else {
      append(a::SVMNodeSimpleBsdfData{scalar(0.0f), 255u, {}});
    }
    words[i * 4u + 2u] = words[i * 4u + 3u] =
        static_cast<unsigned>(words.size());
    words.push_back(a::NODE_END);
  }
  return words;
}

} // namespace psycles::test_support
