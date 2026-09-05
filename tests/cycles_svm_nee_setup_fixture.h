#pragma once

#include <psycles/compiler/cycles_svm_types.h>

#include <array>
#include <cstdint>
#include <limits>

namespace psycles::test_support {

namespace nee_abi = compiler::cycles_svm;
inline constexpr std::uint32_t nee_shader_smooth_normal = 1u << 31u;
enum class NeeSetupKind : std::uint32_t {
  triangle,
  point,
  spot,
  area,
  sun,
  background
};

// Shared inputs only. Both implementations receive the same ray, matrices,
// geometric vertices/normals, and authored lamp parameters.
struct NeeSetupInput {
  NeeSetupKind kind{NeeSetupKind::triangle};
  nee_abi::PackedTransform tfm{{1.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 1.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 1.0f, 0.0f}};
  nee_abi::PackedTransform itfm{tfm};
  nee_abi::packed_float3 vertices[3]{
      {-2.0f, -2.0f, 0.0f}, {2.0f, -2.0f, 0.0f}, {0.0f, 2.0f, 0.0f}};
  nee_abi::packed_float3 normals[3]{
      {0.0f, 0.0f, 1.0f}, {0.6f, 0.0f, 0.8f}, {0.0f, 0.6f, 0.8f}};
  nee_abi::packed_float3 ray_P{0.2f, -0.3f, 2.0f};
  nee_abi::packed_float3 ray_D{0.0f, 0.0f, -1.0f};
  float distance{2.0f};
  float dP{0.03f};
  float dD{0.01f};
  float time{0.37f};
  std::uint32_t object_flags{};
  std::uint32_t shader{0u};
  float radius{0.5f};
  float size_u{2.0f};
  float size_v{3.0f};
  float angle{1.1f};
  bool sphere{true};
  std::uint32_t sample{17u};
  std::uint32_t rng_hash{0x384fb52au};
  std::uint32_t rng_offset{72u};
};

inline constexpr auto nee_setup_cases = [] {
  std::array<NeeSetupInput, 19u> c{};
  c[1].shader = nee_shader_smooth_normal;
  c[2] = c[1];
  c[2].object_flags = nee_abi::SD_OBJECT_HAS_CORNER_NORMALS;
  c[3] = c[1];
  c[3].ray_P.z = -2.0f;
  c[3].ray_D.z = 1.0f;
  c[4].object_flags =
      nee_abi::SD_OBJECT_NEGATIVE_SCALE | nee_abi::SD_OBJECT_TRANSFORM_APPLIED;
  for (auto &v : c[4].vertices) {
    v.x = -v.x;
  }
  c[4].tfm.x.x = c[4].itfm.x.x = -1.0f;
  c[5] = c[1];
  c[5].object_flags = nee_abi::SD_OBJECT_NEGATIVE_SCALE;
  c[5].tfm = {{-2.0f, 0.0f, 0.0f, 1.0f},
              {0.0f, 3.0f, 0.0f, -0.5f},
              {0.0f, 0.0f, 0.5f, 0.25f}};
  c[5].itfm = {{-0.5f, 0.0f, 0.0f, 0.5f},
               {0.0f, 1.0f / 3.0f, 0.0f, 1.0f / 6.0f},
               {0.0f, 0.0f, 2.0f, -0.5f}};
  c[5].ray_P = {1.0f, -0.5f, 2.25f};
  c[6].tfm = {{0.0f, 0.0f, 1.0f, 0.0f},
              {1.0f, 0.0f, 0.0f, 0.0f},
              {0.0f, 1.0f, 0.0f, 0.0f}};
  c[6].itfm = {{0.0f, 1.0f, 0.0f, 0.0f},
               {0.0f, 0.0f, 1.0f, 0.0f},
               {1.0f, 0.0f, 0.0f, 0.0f}};
  c[6].ray_P = {2.0f, 0.1f, -0.2f};
  c[6].ray_D = {-1.0f, 0.0f, 0.0f};
  c[7].tfm = {{1.0f, 0.0f, 0.0f, 0.0f},
              {0.0f, 0.0f, -1.0f, 0.0f},
              {0.0f, 1.0f, 0.0f, 0.0f}};
  c[7].itfm = {{1.0f, 0.0f, 0.0f, 0.0f},
               {0.0f, 0.0f, 1.0f, 0.0f},
               {0.0f, -1.0f, 0.0f, 0.0f}};
  c[7].ray_P = {0.1f, -2.0f, 0.2f};
  c[7].ray_D = {0.0f, 1.0f, 0.0f};
  for (auto &v : c[8].vertices) {
    v.y *= 1.0e-9f;
  }
  c[8].ray_P = {0.0f, 0.0f, 2.0f};
  for (auto i = 9u; i < 17u; ++i) {
    c[i].kind = NeeSetupKind::point;
    c[i].ray_P = {0.3f, 0.0f, 2.4f};
  }
  c[10].sphere = false;
  c[11].radius = 0.0f;
  c[11].ray_P = {0.0f, 0.0f, 2.0f};
  c[12].kind = NeeSetupKind::spot;
  c[12].ray_D = {-0.6f, 0.0f, -0.8f};
  c[12].ray_P = {1.5f, 0.0f, 2.0f};
  c[13] = c[12];
  c[13].sphere = false;
  c[13].tfm.x.x = 2.0f;
  c[13].tfm.y.y = 3.0f;
  c[13].tfm.z.z = 4.0f;
  c[13].itfm.x.x = 0.5f;
  c[13].itfm.y.y = 1.0f / 3.0f;
  c[13].itfm.z.z = 0.25f;
  c[14].kind = NeeSetupKind::area;
  c[14].ray_P = {0.3f, 0.4f, -2.0f};
  c[14].ray_D = {0.0f, 0.0f, 1.0f};
  c[15].kind = NeeSetupKind::sun;
  c[15].angle = 1.5f;
  c[15].distance = std::numeric_limits<float>::max();
  c[15].ray_D = {0.6f, 0.0f, 0.8f};
  c[16] = c[15];
  c[16].angle = 0.0f;
  c[16].ray_D = {0.0f, 0.0f, 1.0f};
  c[17].kind = NeeSetupKind::background;
  c[17].ray_D = {0.6f, 0.0f, 0.8f};
  c[17].distance = std::numeric_limits<float>::max();
  c[17].dD = 0.2f;
  c[18] = c[17];
  c[18].dD = 1.0e-4f;
  for (auto i = 0u; i < c.size(); ++i) {
    c[i].sample += i;
    c[i].rng_offset += 8u * i;
    c[i].rng_hash ^= i * 4711u;
    c[i].time += static_cast<float>(i) * 0.01f;
  }
  return c;
}();

} // namespace psycles::test_support
