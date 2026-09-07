#pragma once
#include <psycles/compiler/cycles_svm_types.h>
#include <array>

namespace psycles::test_support {
using compiler::cycles_svm::packed_float3;
using compiler::cycles_svm::PackedTransform;
inline constexpr std::array static_normal_inputs{
    packed_float3{0.6f, 0.8f, 0.0f}, packed_float3{0.0f, 0.0f, 1.0f},
    packed_float3{0.0f, 0.0f, -1.0f}, packed_float3{1.0f, 0.0f, 0.0f},
    packed_float3{-0.4f, -0.67f, 0.61f}, packed_float3{0.1f, -0.9f, -0.4f},
    packed_float3{0.02f, 0.5f, 0.5f}, packed_float3{-0.01f, 0.01f, 1.0f}};
// Normal transforms (inverse transpose), not position transforms.
inline constexpr std::array static_normal_transforms{
    PackedTransform{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}},
    PackedTransform{{0.5f, 0, 0, 0}, {0, 0.140625f, 0, 0}, {0, 0, 1, 0}},
    PackedTransform{{0.6f, -0.8f, 0, 0}, {0.8f, 0.6f, 0, 0}, {0, 0, 1, 0}},
    PackedTransform{{-1, 0.125f, 0, 0}, {0, 0.75f, -0.5f, 0}, {0, 0, 2, 0}}};
} // namespace psycles::test_support
