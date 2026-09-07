#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace psycles::test_support {
struct MapRangeImage {
  std::string name;
  std::vector<std::uint32_t> words;
};

inline std::vector<MapRangeImage> read_map_range_images(const char *path,
                                                       unsigned expected_count = 24u) {
  std::ifstream file{path};
  unsigned count{};
  file >> count;
  if (!file || count != expected_count) {
    throw std::runtime_error{"invalid Map Range image count"};
  }
  std::vector<MapRangeImage> images(count);
  for (auto &image : images) {
    unsigned size{};
    file >> std::quoted(image.name) >> std::dec >> size;
    if (!file || size < 7u || size > 256u) {
      throw std::runtime_error{"invalid Map Range image header"};
    }
    image.words.resize(size);
    file >> std::hex;
    for (auto &word : image.words) {
      file >> word;
    }
    if (!file) {
      throw std::runtime_error{"truncated Map Range image"};
    }
  }
  std::string trailing;
  if (file >> trailing) {
    throw std::runtime_error{"trailing Map Range image data"};
  }
  return images;
}

struct MapRangePoint {
  float x, y, z;
};
inline constexpr std::array<MapRangePoint, 16> map_range_points{
    {{-5, -1, -1},
     {-2, -2, 1},
     {-1, 2, -1},
     {-0.75f, 0.5f, 0.5f},
     {0, 2, 1},
     {0.1f, -0.5f, 0.5f},
     {0.5f, 1.5f, 0},
     {0.75f, 0, 0},
     {1, 0.25f, 2},
     {1.5f, 0.5f, 0.5f},
     {2, 2, -1},
     {2.5f, -2, 1},
     {3, 0, 0},
     {4, 2, 1},
     {9, -1, 1},
     {-9, 1, 2}}};
} // namespace psycles::test_support
