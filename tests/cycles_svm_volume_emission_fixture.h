#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace psycles::test_support {

inline constexpr unsigned volume_emission_case_count = 16u;
inline constexpr std::array volume_emission_densities{0.0f, 0.25f, 2.0f};

// Shared inputs only. Four object choices (world/zero/quarter/double density),
// camera/non-camera visibility, and absent/pre-existing emission state.
inline constexpr unsigned volume_emission_object(unsigned i) noexcept {
  return i % 4u == 0u ? ~0u : i % 4u - 1u;
}
inline constexpr bool volume_emission_camera(unsigned i) noexcept {
  return (i & 4u) == 0u;
}
inline constexpr bool volume_emission_accumulate(unsigned i) noexcept {
  return (i & 8u) != 0u;
}

inline std::vector<std::uint32_t> read_volume_emission_words(const char *path) {
  std::ifstream file{path};
  unsigned count{};
  file >> count >> std::hex;
  if (!file || count != 23u) { throw std::runtime_error{"invalid word oracle"}; }
  std::vector<std::uint32_t> words(count);
  for (auto &word : words) { file >> word; }
  if (!file) { throw std::runtime_error{"truncated word oracle"}; }
  return words;
}

} // namespace psycles::test_support
