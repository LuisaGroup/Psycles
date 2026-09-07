#pragma once

#include <psycles/compiler/cycles_svm_node_types.h>

#include <array>
#include <bit>
#include <cstring>
#include <vector>

namespace psycles::test_support::volume_stack_fixture {
namespace abi = compiler::cycles_svm;
inline constexpr unsigned shader_count = 9u, state_count = 5u;
inline constexpr std::array capacities{0u, 1u, 3u, 12u, 64u};
inline constexpr std::array<std::array<int, 3>, 12> stacks{{
    {0, -1, -1}, {0, 1, -1}, {1, 0, 2}, {3, 3, -1},
    {4, -1, -1}, {4, 0, -1}, {5, 5, -1}, {6, -1, -1},
    {7, 8, -1}, {8, 0, -1}, {-1, -1, -1}, {0, 8, 0}}};
inline constexpr unsigned count = stacks.size() * state_count;
inline constexpr unsigned object(unsigned row, unsigned entry) {
  return (row % 2u == 0u && entry == 0u) ? ~0u : (row + entry) % 4u;
}
inline constexpr std::array densities{1.0f, 2.0f, 0.25f, 0.0f};
inline constexpr unsigned visibility(unsigned state) {
  return state == 2u ? abi::PATH_RAY_VISIBILITY_SHADOW :
         state == 1u ? abi::PATH_RAY_VISIBILITY_DIFFUSE : abi::PATH_RAY_VISIBILITY_CAMERA;
}
inline constexpr unsigned flag(unsigned state) {
  return state == 3u ? 1u << 6u : state == 4u ? 1u << 10u : 0u;
}
inline constexpr unsigned shader_flags(unsigned shader) {
  return abi::SD_HAS_VOLUME | (shader % 2u ? abi::SD_VOLUME_CUBIC : 0u);
}
// Inputs shared with original GPU execution. No expected values live here.
inline std::vector<unsigned> words() {
  std::vector<unsigned> w(shader_count * 4u);
  const auto emit = [&](auto tag, const auto &payload) {
    w.push_back(static_cast<unsigned>(tag));
    const auto offset = w.size();
    w.resize(offset + sizeof(payload) / 4u);
    std::memcpy(w.data() + offset, &payload, sizeof(payload));
  };
  const auto scalar = [](float x) { return abi::SVMInputFloat{std::bit_cast<unsigned>(x)}; };
  const auto vec = [&](float x, float y, float z) {
    return abi::SVMInputFloat3{scalar(x), scalar(y), scalar(z)};
  };
  const auto scatter = [&](unsigned type, float p, float extra) {
    emit(abi::NODE_CLOSURE_SET_WEIGHT, abi::SVMNodeClosureSetWeight{{0.2f, 0.4f, 0.8f}});
    emit(abi::NODE_CLOSURE_VOLUME, abi::SVMNodeClosureVolume{
        static_cast<abi::ClosureType>(type), scalar(1.5f), scalar(p), scalar(extra), abi::SVM_STACK_INVALID});
  };
  for (unsigned s = 0u; s < shader_count; ++s) {
    w[4u * s] = abi::NODE_SHADER_JUMP;
    w[4u * s + 1u] = w[4u * s + 3u] = unsigned(w.size());
    w.push_back(abi::NODE_END);
    w[4u * s + 2u] = unsigned(w.size());
    if (s == 0u) {
      scatter(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, 0.2f, 0.0f);
      scatter(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, 0.2f, 0.0f);
    } else if (s == 1u) {
      scatter(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, 0.2f, 0.0f);
      scatter(abi::CLOSURE_VOLUME_DRAINE_ID, 0.3f, 0.5f);
      scatter(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, 0.2f, 0.0f);
    } else if (s == 2u) {
      scatter(abi::CLOSURE_VOLUME_DRAINE_ID, 0.3f, 0.5f);
      scatter(abi::CLOSURE_VOLUME_RAYLEIGH_ID, 0.0f, 0.0f);
      scatter(abi::CLOSURE_VOLUME_RAYLEIGH_ID, 0.0f, 0.0f);
    } else if (s == 3u) {
      scatter(abi::CLOSURE_VOLUME_FOURNIER_FORAND_ID, 1.33f, 0.1f);
      scatter(abi::CLOSURE_VOLUME_FOURNIER_FORAND_ID, 1.33f, 0.1f);
      scatter(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, -0.4f, 0.0f);
    } else if (s == 4u) {
      for (unsigned i = 0u; i < 12u; ++i) {
        scatter(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID, float(i) * 0.05f, 0.0f);
      }
    } else if (s == 5u) {
      emit(abi::NODE_CLOSURE_SET_WEIGHT, abi::SVMNodeClosureSetWeight{{-1.0f, 0.4f, 0.8f}});
      emit(abi::NODE_VOLUME_COEFFICIENTS, abi::SVMNodeVolumeCoefficients{
          abi::CLOSURE_VOLUME_DRAINE_ID, vec(0.1f, -0.2f, 0.3f), vec(-0.1f, 0.2f, 0.5f),
          scalar(0.3f), scalar(0.5f), abi::SVM_STACK_INVALID});
    } else if (s == 6u || s == 7u) {
      if (s == 6u) {
        emit(abi::NODE_TEX_COORD, abi::SVMNodeTexCoord{
            abi::NODE_TEXCO_OBJECT, abi::NODE_BUMP_OFFSET_CENTER, 0u, 0u, 0.0f});
      } else {
        emit(abi::NODE_GEOMETRY, abi::SVMNodeGeometry{
            abi::NODE_GEOM_P, abi::NODE_BUMP_OFFSET_CENTER, 0u, 0u, 0.0f});
      }
      emit(abi::NODE_LIGHT_PATH, abi::SVMNodeLightPath{abi::NODE_LP_ray_depth, 3u, {}});
      abi::SVMNodeEmissionWeight emission{};
      emission.color.x.bits = 0x7fc00000u;
      emission.strength.bits = 0x7fc00003u;
      emit(abi::NODE_EMISSION_WEIGHT, emission);
      emit(abi::NODE_CLOSURE_EMISSION, abi::SVMNodeClosureEmission{abi::SVM_STACK_INVALID, {}});
    }
    w.push_back(abi::NODE_END);
  }
  return w;
}

struct Output {
  unsigned meta[8]{};
  float setup[32]{};
  float coefficients[9]{};
  float closures[16][8]{};
  float phases[8][8]{};
};
} // namespace psycles::test_support::volume_stack_fixture
