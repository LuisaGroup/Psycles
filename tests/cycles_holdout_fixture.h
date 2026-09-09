#pragma once

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace psycles::test_support::holdout {
inline constexpr std::array<std::string_view, 12u> cases{
    "ordinary-emission", "object-emission", "object-transparent",
    "object-mixed-emission", "object-mixed-diffuse", "object-colored-transparent",
    "object-mixed-opaque", "object-secondary-emission", "node-holdout",
    "node-mixed-emission", "node-mixed-transparent", "node-dynamic-holdout"};

inline void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

inline std::filesystem::path fixture(std::string_view name, std::string_view suffix) {
  return std::filesystem::path{PSYCLES_HOLDOUT_FIXTURES} / (std::string{name} + std::string{suffix});
}

// Decode original Blender bytes into a unique test-owned bundle. This is input
// packing, never a geometry, shader or transport reference implementation.
struct Bundle {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("psycles-holdout-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  explicit Bundle(std::string_view name) {
    require(std::filesystem::create_directory(path), "cannot create holdout test bundle");
    std::filesystem::copy_file(fixture(name, "-scene.json"), path / "scene.json");
    std::ifstream encoded{fixture(name, "-geometry.txt")};
    std::size_t expected{}, count{};
    encoded >> expected;
    require(bool(encoded) && expected > 0u, "missing original geometry length");
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    unsigned byte{};
    while (encoded >> std::hex >> byte) {
      require(byte <= 255u, "invalid original geometry byte");
      geometry.put(static_cast<char>(byte));
      ++count;
    }
    require(encoded.eof() && count == expected && geometry.good(), "truncated original geometry");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  Bundle(const Bundle &) = delete;
  Bundle &operator=(const Bundle &) = delete;
};
} // namespace psycles::test_support::holdout
