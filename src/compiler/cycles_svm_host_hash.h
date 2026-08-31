/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/core/math.h>

#include <bit>
#include <cstdint>

namespace psycles::compiler::cycles_svm::host_hash {

// Host spelling of Cycles 5.2.1 util/hash.h. These functions are used only
// when Blender's shader-tree inliner would have evaluated a function node at
// compile time; device evaluation remains in the Luisa DSL implementation.

inline void final(std::uint32_t &a, std::uint32_t &b,
                  std::uint32_t &c) noexcept {
  c ^= b;
  c -= std::rotl(b, 14);
  a ^= c;
  a -= std::rotl(c, 11);
  b ^= a;
  b -= std::rotl(a, 25);
  c ^= b;
  c -= std::rotl(b, 16);
  a ^= c;
  a -= std::rotl(c, 4);
  b ^= a;
  b -= std::rotl(a, 14);
  c ^= b;
  c -= std::rotl(b, 24);
}

inline void mix(std::uint32_t &a, std::uint32_t &b,
                std::uint32_t &c) noexcept {
  a -= c;
  a ^= std::rotl(c, 4);
  c += b;
  b -= a;
  b ^= std::rotl(a, 6);
  a += c;
  c -= b;
  c ^= std::rotl(b, 8);
  b += a;
  a -= c;
  a ^= std::rotl(c, 16);
  c += b;
  b -= a;
  b ^= std::rotl(a, 19);
  a += c;
  c -= b;
  c ^= std::rotl(b, 4);
  b += a;
}

[[nodiscard]] inline std::uint32_t hash_uint(std::uint32_t x) noexcept {
  auto a = std::uint32_t{0xdeadbeefu + (1u << 2u) + 13u};
  auto b = a;
  auto c = a;
  a += x;
  final(a, b, c);
  return c;
}

[[nodiscard]] inline std::uint32_t hash_uint2(std::uint32_t x,
                                              std::uint32_t y) noexcept {
  auto a = std::uint32_t{0xdeadbeefu + (2u << 2u) + 13u};
  auto b = a;
  auto c = a;
  a += x;
  b += y;
  final(a, b, c);
  return c;
}

[[nodiscard]] inline std::uint32_t hash_uint3(std::uint32_t x,
                                              std::uint32_t y,
                                              std::uint32_t z) noexcept {
  auto a = std::uint32_t{0xdeadbeefu + (3u << 2u) + 13u};
  auto b = a;
  auto c = a;
  a += x;
  b += y;
  c += z;
  final(a, b, c);
  return c;
}

[[nodiscard]] inline std::uint32_t hash_uint4(std::uint32_t x,
                                              std::uint32_t y,
                                              std::uint32_t z,
                                              std::uint32_t w) noexcept {
  auto a = std::uint32_t{0xdeadbeefu + (4u << 2u) + 13u};
  auto b = a;
  auto c = a;
  a += x;
  b += y;
  c += z;
  mix(a, b, c);
  a += w;
  final(a, b, c);
  return c;
}

[[nodiscard]] inline float uint_to_float_inclusive(
    std::uint32_t value) noexcept {
  return static_cast<float>(value) *
         (1.0f / static_cast<float>(0xffffffffu));
}

[[nodiscard]] inline float hash_float(float value) noexcept {
  return uint_to_float_inclusive(
      hash_uint(std::bit_cast<std::uint32_t>(value)));
}

[[nodiscard]] inline float hash_float2(float x, float y) noexcept {
  return uint_to_float_inclusive(
      hash_uint2(std::bit_cast<std::uint32_t>(x),
                 std::bit_cast<std::uint32_t>(y)));
}

[[nodiscard]] inline float hash_float3(Vec3f value) noexcept {
  return uint_to_float_inclusive(
      hash_uint3(std::bit_cast<std::uint32_t>(value.x),
                 std::bit_cast<std::uint32_t>(value.y),
                 std::bit_cast<std::uint32_t>(value.z)));
}

[[nodiscard]] inline float hash_float4(Vec3f value, float w) noexcept {
  return uint_to_float_inclusive(
      hash_uint4(std::bit_cast<std::uint32_t>(value.x),
                 std::bit_cast<std::uint32_t>(value.y),
                 std::bit_cast<std::uint32_t>(value.z),
                 std::bit_cast<std::uint32_t>(w)));
}

[[nodiscard]] inline float evaluate_value(Vec3f vector, float w,
                                          std::uint32_t dimensions) noexcept {
  switch (dimensions) {
    case 1u:
      return hash_float(w);
    case 2u:
      return hash_float2(vector.x, vector.y);
    case 3u:
      return hash_float3(vector);
    case 4u:
      return hash_float4(vector, w);
    default:
      return 0.0f;
  }
}

[[nodiscard]] inline Vec3f evaluate_color(Vec3f vector, float w,
                                          std::uint32_t dimensions) noexcept {
  switch (dimensions) {
    case 1u:
      return {hash_float(w), hash_float2(w, 1.0f),
              hash_float2(w, 2.0f)};
    case 2u:
      return {hash_float2(vector.x, vector.y),
              uint_to_float_inclusive(hash_uint3(
                  std::bit_cast<std::uint32_t>(vector.x),
                  std::bit_cast<std::uint32_t>(vector.y),
                  std::bit_cast<std::uint32_t>(1.0f))),
              uint_to_float_inclusive(hash_uint3(
                  std::bit_cast<std::uint32_t>(vector.x),
                  std::bit_cast<std::uint32_t>(vector.y),
                  std::bit_cast<std::uint32_t>(2.0f)))};
    case 3u:
      return {hash_float3(vector), hash_float4(vector, 1.0f),
              hash_float4(vector, 2.0f)};
    case 4u: {
      const auto x = std::bit_cast<std::uint32_t>(vector.x);
      const auto y = std::bit_cast<std::uint32_t>(vector.y);
      const auto z = std::bit_cast<std::uint32_t>(vector.z);
      const auto wb = std::bit_cast<std::uint32_t>(w);
      return {hash_float4(vector, w),
              uint_to_float_inclusive(hash_uint4(z, x, wb, y)),
              uint_to_float_inclusive(hash_uint4(wb, z, y, x))};
    }
    default:
      return {1.0f, 0.0f, 1.0f};
  }
}

} // namespace psycles::compiler::cycles_svm::host_hash
