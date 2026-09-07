#pragma once
#include <psycles/compiler/cycles_svm_node_types.h>
#include <array>
#include <bit>
#include <cstring>
#include <vector>

namespace psycles::test_support::volume_density_fixture {
namespace abi = compiler::cycles_svm;
inline constexpr unsigned shader_count = 5u;
struct Input { unsigned shader, cell, flags; };
// Shader 1 deliberately supplies the homogeneous flag with a spatial input:
// this probes which sample count the kernel metadata selects, independently
// of the compiler's spatial-varying analysis. It is not an authored scene.
inline constexpr std::array inputs{
    Input{0, 0, 0}, Input{0, 12345, 0}, Input{0, 2097151, 0},
    Input{1, 0, 0}, Input{1, 12345, 0}, Input{1, 2097151, 0},
    Input{2, 12345, 0}, Input{3, 12345, 0}, Input{4, 12345, 0},
    Input{0, 12345, abi::SD_OBJECT_TRANSFORM_APPLIED},
    Input{2, 12345, abi::SD_OBJECT_TRANSFORM_APPLIED},
    Input{4, 12345, abi::SD_OBJECT_TRANSFORM_APPLIED}};
inline constexpr unsigned shader_flags(unsigned shader) {
    return abi::SD_HAS_VOLUME | (shader == 1 ? 0u : abi::SD_HETEROGENEOUS_VOLUME) |
        (shader == 3 ? abi::SD_HAS_LIGHT_PATH_NODE : 0u);
}
inline std::vector<unsigned> words() {
    std::vector<unsigned> w(shader_count * 4u);
    const auto emit = [&](auto tag, const auto &payload) {
        w.push_back(static_cast<unsigned>(tag));
        auto offset = w.size(); w.resize(offset + sizeof(payload) / 4u);
        std::memcpy(w.data() + offset, &payload, sizeof(payload));
    };
    for (unsigned s = 0; s < shader_count; ++s) {
        w[4u * s] = abi::NODE_SHADER_JUMP;
        w[4u * s + 1] = w[4u * s + 3] = unsigned(w.size());
        w.push_back(abi::NODE_END);
        w[4u * s + 2] = unsigned(w.size());
        if (s == 2u || s == 4u) {
            emit(abi::NODE_TEX_COORD, abi::SVMNodeTexCoord{
                s == 2u ? abi::NODE_TEXCO_OBJECT : abi::NODE_TEXCO_VOLUME_GENERATED,
                abi::NODE_BUMP_OFFSET_CENTER, 0u, 0u, 0.0f});
        } else {
            emit(abi::NODE_GEOMETRY, abi::SVMNodeGeometry{
                abi::NODE_GEOM_P, abi::NODE_BUMP_OFFSET_CENTER, 0u, 0u, 0.0f});
        }
        abi::SVMNodeEmissionWeight emission{};
        emission.color.x.bits = 0x7fc00000u;
        emission.strength.bits = std::bit_cast<unsigned>(1.0f);
        if (s == 3u) {
            emit(abi::NODE_LIGHT_PATH, abi::SVMNodeLightPath{abi::NODE_LP_ray_depth, 3u, {}});
            emission.strength.bits = 0x7fc00003u;
        }
        emit(abi::NODE_EMISSION_WEIGHT, emission);
        emit(abi::NODE_CLOSURE_EMISSION, abi::SVMNodeClosureEmission{abi::SVM_STACK_INVALID, {}});
        w.push_back(abi::NODE_END);
    }
    return w;
}
} // namespace psycles::test_support::volume_density_fixture
