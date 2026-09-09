#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace psycles::test_support::light_parameters {
// Original table observations. Input [0..8] is the object linear transform,
// [9..11] position, [12..15] raw angle/spread, smooth, size/radius, area size V.
// Derived [16..20] is Area(tan,normalization,unused...) or Spot's five fields.
// [21..24] = invarea/eval_fac, radius, len_u, len_v; [25..33] = dir, axis_u/v.
struct Record {
  unsigned spot{};
  std::string name_hex;
  unsigned normalize{}, ellipse{}, sphere{};
  std::array<float, 34> v{};
};

inline std::vector<Record> read(const char *path) {
  std::ifstream file{path};
  std::string magic;
  unsigned count{};
  if (!(file >> magic >> count) || magic != "PSYLIGHT1" || count == 0 || count > 1024) {
    throw std::runtime_error{"invalid native light parameter fixture"};
  }
  std::vector<Record> result(count);
  for (auto &r : result) {
    if (!(file >> r.spot >> r.name_hex >> r.normalize >> r.ellipse >> r.sphere) ||
        r.spot > 1 || r.normalize > 1 || r.ellipse > 1 || r.sphere > 1) {
      throw std::runtime_error{"invalid native light record"};
    }
    for (auto &v : r.v) {
      std::uint32_t bits{};
      if (!(file >> bits)) { throw std::runtime_error{"truncated native light record"}; }
      v = std::bit_cast<float>(bits);
    }
  }
  if (file >> magic) { throw std::runtime_error{"trailing native light data"}; }
  return result;
}

inline constexpr unsigned probes = 8, outputs = 40;
// Authored geometry and random inputs only, not host light sampling.
inline constexpr std::array<std::array<float, 3>, probes> origins{{
    {0, 0, -3}, {0.02f, 0.01f, -0.2f}, {0.4f, 0, -3}, {3, 0, -3},
    {0, 0, 3}, {-0.3f, 0.2f, -0.7f}, {0.1f, 0.1f, -2}, {2, 2, -4}}};
inline constexpr std::array<std::array<float, 2>, probes> randoms{{
    {0.2f, 0.4f}, {0.3f, 0.7f}, {0.01f, 0.99f}, {0.7f, 0.9f},
    {0.5f, 0.5f}, {0.81f, 0.32f}, {0.1f, 0.1f}, {0.6f, 0.6f}}};
inline constexpr std::array<std::array<float, 3>, probes> local_rays{{
    {0, 0, 1}, {0.3f, 0, 0.9539392f}, {0.6f, 0, 0.8f}, {1, 0, 0},
    {0, 0, -1}, {0.1f, 0.2f, 0.97f}, {0.1f, 0.2f, 0}, {0.1f, 0.2f, 0}}};

struct ProbeInput {
  std::array<float,3> offset{}, normal{}, interval_direction{}, local_ray{};
  std::array<float,2> random{};
};

inline ProbeInput input(const Record &record, unsigned probe) {
  ProbeInput result;
  result.random = randoms.at(probe);
  result.local_ray = local_rays.at(probe);
  // Transform authored geometry inputs, not sampling results. A fixed world
  // ray accidentally lies almost in the plane of several Barbershop lamps;
  // those singular PDFs cannot be a no-last-bit-parity regression gate.
  std::array<std::array<float,3>,3> basis;
  for (unsigned column = 0; column < 3; ++column) {
    const auto p = record.v.data()+3*column;
    const auto length = std::sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);
    for (unsigned row = 0; row < 3; ++row) { basis[column][row] = p[row]/length; }
  }
  for (unsigned row = 0; row < 3; ++row) {
    for (unsigned column = 0; column < 3; ++column) {
      result.offset[row] += basis[column][row]*origins[probe][column];
    }
    result.normal[row] = basis[2][row];
    // Avoid a cone-axis double root for spots. For areas, use nearly normal
    // rays into the emitting halfspace, checking plane/segment clipping and
    // outside-footprint rejection. A transverse ray through a 0.001-radian
    // area's cone subtracts O(1e6) terms to obtain O(1) coefficients: original
    // GPU FMA contraction alone changes its endpoint by about 0.1 units.
    // Those ill-conditioned cone-edge probes remain diagnostic-only; neither
    // a cross-backend exact-root requirement nor slower runtime math is valid.
    result.interval_direction[row] = record.spot ?
        0.6f*basis[0][row] + 0.8f*basis[2][row] :
        0.01f*basis[0][row] - 0.999949999f*basis[2][row];
  }
  if (probe == 6 || probe == 7) {
    if (!record.spot) { result.local_ray[2] = probe == 6 ? 1.0f : 0.8f; }
    else {
      const auto cosine = record.v[16];
      // With zero smooth the native table contains +inf. Exactly at the
      // boundary, (z-cosine)*inf is NaN and outside finite-only fast math.
      // Test an interior value there, and the exact boundary when finite.
      // The oracle harness itself is built with -ffast-math. Validate the
      // serialized exponent as integer data, not an optimized isfinite().
      const bool finite_blend = (std::bit_cast<std::uint32_t>(record.v[18]) & 0x7f800000u) != 0x7f800000u;
      result.local_ray[2] = probe == 6 && finite_blend ?
          cosine : 0.5f*(1.0f+cosine);
    }
  }
  return result;
}
} // namespace psycles::test_support::light_parameters
