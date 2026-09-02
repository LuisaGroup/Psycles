#include <psycles/compiler/cycles_hash.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace psycles::compiler {

std::uint32_t cycles_murmur_hash3(std::string_view value) noexcept {
  constexpr auto c1 = std::uint32_t{0xcc9e2d51u};
  constexpr auto c2 = std::uint32_t{0x1b873593u};
  auto h1 = std::uint32_t{};
  const auto *bytes =
      reinterpret_cast<const std::uint8_t *>(value.data());
  const auto blocks = value.size() / 4u;
  for (auto i = std::size_t{}; i < blocks; ++i) {
    const auto base = i * 4u;
    auto k1 = static_cast<std::uint32_t>(bytes[base]) |
              (static_cast<std::uint32_t>(bytes[base + 1u]) << 8u) |
              (static_cast<std::uint32_t>(bytes[base + 2u]) << 16u) |
              (static_cast<std::uint32_t>(bytes[base + 3u]) << 24u);
    k1 *= c1;
    k1 = std::rotl(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = std::rotl(h1, 13);
    h1 = h1 * 5u + 0xe6546b64u;
  }
  const auto *tail = bytes + blocks * 4u;
  auto k1 = std::uint32_t{};
  switch (value.size() & 3u) {
  case 3u:
    k1 ^= static_cast<std::uint32_t>(tail[2u]) << 16u;
    [[fallthrough]];
  case 2u:
    k1 ^= static_cast<std::uint32_t>(tail[1u]) << 8u;
    [[fallthrough]];
  case 1u:
    k1 ^= static_cast<std::uint32_t>(tail[0u]);
    k1 *= c1;
    k1 = std::rotl(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    break;
  default:
    break;
  }
  h1 ^= static_cast<std::uint32_t>(value.size());
  h1 ^= h1 >> 16u;
  h1 *= 0x85ebca6bu;
  h1 ^= h1 >> 13u;
  h1 *= 0xc2b2ae35u;
  h1 ^= h1 >> 16u;
  return h1;
}

float cycles_hash_to_float(std::uint32_t hash) noexcept {
  const auto mantissa = hash & ((1u << 23u) - 1u);
  auto exponent = (hash >> 23u) & ((1u << 8u) - 1u);
  exponent = std::clamp(exponent, 1u, 254u) << 23u;
  const auto sign = (hash >> 31u) << 31u;
  return std::bit_cast<float>(sign | exponent | mantissa);
}

float cycles_cryptomatte_hash(std::string_view value) noexcept {
  return cycles_hash_to_float(cycles_murmur_hash3(value));
}

} // namespace psycles::compiler
