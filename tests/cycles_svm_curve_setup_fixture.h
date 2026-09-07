#pragma once

#include <psycles/compiler/cycles_svm_types.h>

#include <array>

namespace psycles::test_support {
namespace curve_setup_abi = compiler::cycles_svm;

// Inputs only. Expected intersection coordinates and all ShaderData fields
// come from the original Cycles HIP ribbon_intersect/shader_setup_from_ray.
struct CurveSetupInput {
  curve_setup_abi::PackedTransform tfm, itfm;
  std::array<curve_setup_abi::packed_float4, 6u> keys{};
  curve_setup_abi::packed_float3 ray_P, ray_D{0.0f, 0.0f, 1.0f};
  float dP{0.03f}, dD{0.01f}, time{0.37f};
  unsigned segment{}, object_flags{}, shader_flags{}, subdivision_level{3u};
};
inline constexpr unsigned curve_setup_primitive_base = 11u;
inline constexpr auto curve_setup_inputs = [] {
  namespace a = curve_setup_abi;
  std::array<CurveSetupInput, 24u> result{};
  for (auto i = 0u; i < result.size(); ++i) {
    auto &c = result[i];
    const auto mode = i / 6u;
    c.segment = (i / 2u) % 3u;
    c.keys = {{{}, {}, {-0.04f, -1.5f, 0.05f, 0.22f},
               {-0.015f, -0.5f, -0.03f, 0.20f},
               {0.02f, 0.5f, 0.04f, 0.18f}, {0.01f, 1.5f, 0.01f, 0.16f}}};
    const auto sx = mode == 1u ? 2.0f : (mode == 2u ? -2.0f : (mode == 3u ? -1.0f : 1.0f));
    const auto sy = (mode == 1u || mode == 2u) ? 0.5f : 1.0f;
    const auto sz = (mode == 1u || mode == 2u) ? 1.5f : 1.0f;
    const auto tx = mode == 0u ? 0.0f : 2.0f;
    const auto ty = mode == 0u ? 0.0f : -1.0f;
    const auto tz = mode == 0u ? 0.0f : 3.0f;
    c.tfm = {{sx, 0.0f, 0.0f, tx}, {0.0f, sy, 0.0f, ty}, {0.0f, 0.0f, sz, tz}};
    c.itfm = {{1.0f / sx, 0.0f, 0.0f, -tx / sx},
               {0.0f, 1.0f / sy, 0.0f, -ty / sy},
               {0.0f, 0.0f, 1.0f / sz, -tz / sz}};
    const auto x = (i & 1u) ? 0.065f : -0.06f;
    c.ray_P = {sx * x + tx, sy * (float(c.segment) - 1.0f) + ty, -2.0f * sz + tz};
    if (mode >= 2u) { c.object_flags |= a::SD_OBJECT_NEGATIVE_SCALE; }
    if (mode == 3u) {
      c.object_flags |= a::SD_OBJECT_TRANSFORM_APPLIED;
      for (auto &key : c.keys) {
        key.x = sx * key.x + tx;
        key.y = sy * key.y + ty;
        key.z = sz * key.z + tz;
      }
    }
    c.shader_flags = (i & 1u) ? a::SD_USE_BUMP_MAP_CORRECTION : 0u;
    c.time = 0.125f + float(i) * 0.02f;
    if ((i % 3u) == 0u) { c.dP = c.dD = 0.0f; }
  }
  return result;
}();
} // namespace psycles::test_support
