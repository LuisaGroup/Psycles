#pragma once

#include "cycles_svm_light_emission_fixture.h"

namespace psycles::test_support {

// Serialized inputs shared with the external Cycles HIP interpreter. No
// expected shading values or alternate evaluator are defined here.
struct ShadowClosureCase {
  compiler::cycles_svm::ClosureType type;
  std::array<float, 3u> weight{0.2f, 0.5f, 0.8f};
  float mix_weight{1.0f};
  unsigned repetitions{1u};
  bool transparent_successor{};
  bool thin_wall{};
  float roughness{0.3f};
};

inline constexpr auto shadow_closure_cases = [] {
  using namespace compiler::cycles_svm;
  return std::to_array<ShadowClosureCase>({
      {CLOSURE_BSDF_DIFFUSE_ID},
      {CLOSURE_BSDF_TRANSLUCENT_ID},
      {CLOSURE_BSDF_TRANSPARENT_ID},
      {CLOSURE_BSDF_SHEEN_ID},
      {CLOSURE_BSDF_ASHIKHMIN_VELVET_ID},
      {CLOSURE_BSDF_DIFFUSE_TOON_ID},
      {CLOSURE_BSDF_GLOSSY_TOON_ID},
      {CLOSURE_BSDF_RAY_PORTAL_ID},
      {CLOSURE_BSDF_MICROFACET_GGX_ID},
      {CLOSURE_BSDF_MICROFACET_BECKMANN_ID},
      {CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID},
      {CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID},
      {CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID},
      {CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID},
      {CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID},
      {CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID},
      {CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID},
      {CLOSURE_BSDF_PHYSICAL_CONDUCTOR},
      {CLOSURE_BSDF_F82_CONDUCTOR},
      {CLOSURE_BSDF_HAIR_REFLECTION_ID},
      {CLOSURE_BSDF_HAIR_TRANSMISSION_ID},
      {CLOSURE_BSDF_HAIR_CHIANG_ID},
      {CLOSURE_BSDF_HAIR_HUANG_ID},
      {CLOSURE_BSSRDF_BURLEY_ID},
      {CLOSURE_BSSRDF_RANDOM_WALK_ID},
      {CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID},
      {CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID},
      {CLOSURE_BSDF_TRANSPARENT_ID, {0.2f, 0.5f, 0.8f}, 0.0f},
      {CLOSURE_BSDF_TRANSPARENT_ID, {0.2f, 0.5f, 0.8f}, 1.0e-7f},
      {CLOSURE_BSDF_TRANSPARENT_ID, {-0.5f, 1.0f, 2.0f}, 0.75f, 2u},
      {CLOSURE_BSDF_RAY_PORTAL_ID, {0.2f, 0.5f, 0.8f}, 1.0f, 1u, true},
      {CLOSURE_BSDF_RAY_PORTAL_ID, {-0.5f, 1.0f, 2.0f}, 0.75f, 2u},
      {CLOSURE_BSDF_PRINCIPLED_ID,
       {1.0f, 1.0f, 1.0f},
       1.0f,
       1u,
       false,
       true,
       0.0f},
      {CLOSURE_BSDF_PRINCIPLED_ID,
       {1.0f, 1.0f, 1.0f},
       1.0f,
       1u,
       false,
       true,
       0.3f},
      {CLOSURE_BSDF_PRINCIPLED_ID,
       {1.0f, 1.0f, 1.0f},
       1.0f,
       1u,
       false,
       false,
       0.0f},
  });
}();

inline constexpr auto shadow_surface_case_count =
    light_emission_cases.size() + shadow_closure_cases.size();

inline float shadow_surface_cosine(unsigned i) {
  return i < light_emission_cases.size() ? light_emission_cases[i].cosine
                                         : 0.8f;
}

inline auto make_shadow_surface_image() {
  namespace abi = compiler::cycles_svm;
  const auto source = make_light_emission_image();
  constexpr auto source_header_words = 4u * light_emission_cases.size();
  constexpr auto header_words = 4u * shadow_surface_case_count;
  constexpr auto relocation = header_words - source_header_words;
  std::vector<std::uint32_t> words(header_words);
  words.insert(words.end(), source.begin() + source_header_words, source.end());
  for (auto i = 0u; i < light_emission_cases.size(); ++i) {
    words[i * 4u] = abi::NODE_SHADER_JUMP;
    for (auto field = 1u; field < 4u; ++field) {
      words[i * 4u + field] = source[i * 4u + field] + relocation;
    }
  }
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
  for (auto i = 0u; i < shadow_closure_cases.size(); ++i) {
    const auto shader = i + light_emission_cases.size();
    const auto &input = shadow_closure_cases[i];
    words[shader * 4u] = abi::NODE_SHADER_JUMP;
    words[shader * 4u + 1u] = static_cast<std::uint32_t>(words.size());
    words.push_back(abi::NODE_VALUE_F);
    append(abi::SVMNodeValueF{input.mix_weight, 0u, {0u, 0u, 0u}});
    words.push_back(abi::NODE_CLOSURE_SET_WEIGHT);
    append(
        abi::packed_float3{input.weight[0], input.weight[1], input.weight[2]});
    for (auto repeat = 0u; repeat < input.repetitions; ++repeat) {
      words.push_back(abi::NODE_CLOSURE_BSDF);
      append(abi::SVMNodeClosureBsdf{input.type, 0u, {0u, 0u, 0u}});
      using namespace abi;
      switch (input.type) {
      case CLOSURE_BSDF_DIFFUSE_ID:
        append(SVMNodeDiffuseBsdfData{
            color(input.weight), scalar(input.roughness), 255u, {}});
        break;
      case CLOSURE_BSDF_DIFFUSE_TOON_ID:
      case CLOSURE_BSDF_GLOSSY_TOON_ID:
        append(SVMNodeToonBsdfData{scalar(0.3f), scalar(0.2f), 255u, {}});
        break;
      case CLOSURE_BSDF_RAY_PORTAL_ID:
        append(SVMNodeRayPortalBsdfData{color({0.0f, 0.0f, 1.0f}), 255u, {}});
        break;
      case CLOSURE_BSDF_MICROFACET_GGX_ID:
      case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
      case CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID:
      case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
        append(SVMNodeGlossyBsdfData{color(input.weight),
                                     scalar(input.roughness),
                                     scalar(0.2f),
                                     scalar(0.1f),
                                     255u,
                                     255u,
                                     {}});
        break;
      case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
        append(SVMNodeRefractionBsdfData{
            scalar(input.roughness), scalar(1.5f), 255u, {}});
        break;
      case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
      case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      case CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID:
        append(SVMNodeGlassBsdfData{color(input.weight),
                                    scalar(input.roughness),
                                    scalar(1.5f),
                                    scalar(400.0f),
                                    scalar(1.3f),
                                    255u,
                                    {}});
        break;
      case CLOSURE_BSDF_PHYSICAL_CONDUCTOR:
      case CLOSURE_BSDF_F82_CONDUCTOR:
        append(SVMNodeMetallicBsdfData{CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID,
                                       color(input.weight),
                                       color({0.8f, 0.7f, 0.6f}),
                                       scalar(input.roughness),
                                       scalar(0.2f),
                                       scalar(0.1f),
                                       scalar(400.0f),
                                       scalar(1.3f),
                                       255u,
                                       255u,
                                       {}});
        break;
      case CLOSURE_BSDF_HAIR_REFLECTION_ID:
      case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
        append(SVMNodeHairBsdfData{
            scalar(input.roughness), scalar(0.2f), scalar(0.1f), 255u, {}});
        break;
      case CLOSURE_BSDF_HAIR_CHIANG_ID:
      case CLOSURE_BSDF_HAIR_HUANG_ID: {
        SVMNodePrincipledHairBsdfData data{};
        data.color = color(input.weight);
        data.roughness = scalar(input.roughness);
        data.radial_roughness = scalar(0.2f);
        data.ior = scalar(1.55f);
        data.random = scalar(0.4f);
        data.aspect_ratio = scalar(1.0f);
        data.R = data.TT = data.TRT = scalar(1.0f);
        data.attr_random = data.attr_normal = -1;
        append(data);
        break;
      }
      case CLOSURE_BSSRDF_BURLEY_ID:
      case CLOSURE_BSSRDF_RANDOM_WALK_ID:
      case CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID:
      case CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID:
        append(SVMNodeBssrdfData{color({1.0f, 0.2f, 0.1f}),
                                 scalar(0.1f),
                                 scalar(1.4f),
                                 scalar(0.2f),
                                 scalar(input.roughness),
                                 255u,
                                 {}});
        break;
      case CLOSURE_BSDF_PRINCIPLED_ID: {
        SVMNodePrincipledBsdfData data{};
        data.distribution = CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
        data.ior = scalar(1.5f);
        data.roughness = scalar(input.roughness);
        data.alpha = scalar(0.8f);
        data.transmission_weight = scalar(0.7f);
        data.base_color = color({0.7f, 0.5f, 0.3f});
        data.specular_tint = color({1.0f, 1.0f, 1.0f});
        data.specular_ior_level = scalar(0.5f);
        data.normal_offset = data.tangent_offset = data.coat_normal_offset =
            255u;
        data.thin_wall.value = input.thin_wall;
        data.thin_wall.offset = 255u;
        append(data);
        break;
      }
      default:
        append(SVMNodeSimpleBsdfData{scalar(input.roughness), 255u, {}});
        break;
      }
    }
    if (input.transparent_successor) {
      words.push_back(abi::NODE_CLOSURE_BSDF);
      append(abi::SVMNodeClosureBsdf{abi::CLOSURE_BSDF_TRANSPARENT_ID, 0u, {}});
      append(abi::SVMNodeSimpleBsdfData{scalar(0.0f), 255u, {}});
    }
    words.push_back(abi::NODE_CLOSURE_SET_WEIGHT);
    append(abi::packed_float3{0.125f, 0.25f, 0.5f});
    words[shader * 4u + 2u] = words[shader * 4u + 3u] =
        static_cast<std::uint32_t>(words.size());
    words.push_back(abi::NODE_END);
  }
  return words;
}

} // namespace psycles::test_support
