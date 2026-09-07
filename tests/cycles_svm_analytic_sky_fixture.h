#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace psycles::test_support {
struct SkyDirection {
  float x, y, z;
};
inline constexpr std::array<SkyDirection, 16> analytic_sky_directions{
    {{0, 0, 1},
     {0, 0, -1},
     {1, 0, 0},
     {-1, 0, 0},
     {0, 1, 0},
     {0, -1, 0},
     {0.6f, 0, 0.8f},
     {-0.6f, 0, 0.8f},
     {0, 0.8f, 0.6f},
     {0, -0.8f, 0.6f},
     {0.6f, 0, -0.8f},
     {0, -0.8f, -0.6f},
     {1, 0, 0.0001f},
     {1, 0, -0.0001f},
     {-0.9461538195610046f, 0.0615384615957737f, 0.31781429052352905f},
     {0.9461538195610046f, -0.0615384615957737f, -0.31781429052352905f}}};
inline constexpr std::array<SkyDirection, 3> analytic_sky_xyz_to_rgb{
    {{3.2404542f, -1.5371385f, -0.4985314f},
     {-0.9692660f, 1.8760108f, 0.0415560f},
     {0.0556434f, -0.2040259f, 1.0572252f}}};

inline std::vector<std::uint32_t> read_analytic_sky_words(const char *path) {
  std::ifstream file{path};
  unsigned count{};
  file >> count >> std::hex;
  if (!file || count != 52u) {
    throw std::runtime_error{"invalid analytic sky oracle"};
  }
  std::vector<std::uint32_t> result(count);
  for (auto &word : result) {
    file >> word;
  }
  if (!file) {
    throw std::runtime_error{"truncated analytic sky oracle"};
  }
  return result;
}
} // namespace psycles::test_support
