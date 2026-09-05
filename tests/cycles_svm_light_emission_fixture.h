#pragma once

#include <psycles/compiler/cycles_svm_node_types.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace psycles::test_support {

// Shared *inputs*, not a reference evaluator. The HIP oracle passes these
// exact bytes to Cycles' own typed payload reader and closure implementation.
struct LightEmissionCase {
  float alpha{1.0f};
  float mix_weight{1.0f};
  float sheen_weight{};
  float coat_weight{};
  float emission_strength{2.0f};
  std::array<float, 3u> emission_color{1.0f, 0.5f, 0.25f};
  std::array<float, 3u> sheen_tint{0.3f, 0.5f, 0.9f};
  std::array<float, 3u> coat_tint{0.7f, 0.8f, 0.9f};
  float sheen_roughness{0.4f};
  float coat_roughness{0.2f};
  float coat_ior{1.45f};
  float cosine{0.8f};
  unsigned repetitions{1u};
};

inline constexpr auto light_emission_cases = [] {
  std::array<LightEmissionCase, 16u> cases{};
  cases[1].alpha = 0.25f;
  cases[2].sheen_weight = 0.8f;
  cases[3].coat_weight = 0.7f;
  cases[4].alpha = 0.65f;
  cases[4].mix_weight = 0.45f;
  cases[4].sheen_weight = 0.8f;
  cases[4].coat_weight = 0.7f;
  cases[5] = cases[4];
  cases[5].repetitions = 2u;
  cases[6] = cases[4];
  cases[6].emission_strength = 0.0f;
  cases[7] = cases[4];
  cases[7].mix_weight = 0.0f;
  cases[8] = cases[4];
  cases[8].mix_weight = 1.0e-7f;
  cases[9] = cases[3];
  cases[9].coat_tint = {0.0f, 0.0f, 0.0f};
  cases[10] = cases[4];
  cases[10].alpha = 0.0f;
  cases[11] = cases[4];
  cases[11].cosine = 0.01f;
  cases[12] = cases[2];
  cases[12].sheen_tint = {-1.0f, -0.5f, -0.25f};
  cases[13].emission_color = {-1.0f, 0.5f, -0.25f};
  cases[14] = cases[4];
  cases[14].coat_roughness = 0.0f;
  cases[15] = cases[4];
  cases[15].sheen_roughness = 0.0f;
  return cases;
}();

inline auto make_light_emission_image() {
  namespace abi = compiler::cycles_svm;
  std::vector<std::uint32_t> words(4u * light_emission_cases.size());
  const auto append = [&words](const auto &payload) {
    static_assert(sizeof(payload) % sizeof(std::uint32_t) == 0u);
    const auto offset = words.size();
    words.resize(offset + sizeof(payload) / sizeof(std::uint32_t));
    std::memcpy(words.data() + offset, &payload, sizeof(payload));
  };
  const auto scalar = [](float value) {
    abi::SVMInputFloat result{};
    std::memcpy(&result.bits, &value, sizeof(value));
    return result;
  };
  const auto color = [&scalar](std::array<float, 3u> value) {
    return abi::SVMInputFloat3{scalar(value[0]), scalar(value[1]),
                               scalar(value[2])};
  };
  for (auto i = 0u; i < light_emission_cases.size(); ++i) {
    const auto &input = light_emission_cases[i];
    words[4u * i] = abi::NODE_SHADER_JUMP;
    words[4u * i + 1u] = static_cast<std::uint32_t>(words.size());
    words.push_back(abi::NODE_VALUE_F);
    append(abi::SVMNodeValueF{input.mix_weight, 0u, {0u, 0u, 0u}});
    for (auto repetition = 0u; repetition < input.repetitions; ++repetition) {
      words.push_back(abi::NODE_CLOSURE_BSDF);
      append(abi::SVMNodeClosureBsdf{
          abi::CLOSURE_BSDF_PRINCIPLED_ID, 0u, {0u, 0u, 0u}});
      abi::SVMNodePrincipledBsdfData data{};
      data.distribution = abi::CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
      data.ior = scalar(1.5f);
      data.roughness = scalar(0.5f);
      data.alpha = scalar(input.alpha);
      data.sheen_weight = scalar(input.sheen_weight);
      data.coat_weight = scalar(input.coat_weight);
      data.normal_offset = 255u;
      data.tangent_offset = 255u;
      data.coat_normal_offset = 255u;
      data.emission_color = color(input.emission_color);
      data.emission_strength = scalar(input.emission_strength);
      data.sheen_tint = color(input.sheen_tint);
      data.sheen_roughness = scalar(input.sheen_roughness);
      data.coat_tint = color(input.coat_tint);
      data.coat_roughness = scalar(input.coat_roughness);
      data.coat_ior = scalar(input.coat_ior);
      data.thin_wall.offset = 255u;
      append(data);
    }
    // An observable successor catches incorrect typed-payload consumption.
    words.push_back(abi::NODE_CLOSURE_SET_WEIGHT);
    append(abi::packed_float3{0.125f, 0.25f, 0.5f});
    words[4u * i + 2u] = static_cast<std::uint32_t>(words.size());
    words[4u * i + 3u] = static_cast<std::uint32_t>(words.size());
    words.push_back(abi::NODE_END);
  }
  return words;
}

} // namespace psycles::test_support
