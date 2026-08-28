#pragma once

#include <cstdint>

#include <psycles/compiler/surface_program.h>

namespace psycles::compiler {

// The SVM instruction reserves fourteen bits for an opcode-owned immediate.
// The finite-domain codecs below are the formal boundary between authored
// static node properties and device data: a property may leave typed-handler
// identity only when its codec is total and injective on the valid domain.
inline constexpr std::uint32_t surface_value_svm_immediate_value_mask =
    (1u << 14u) - 1u;

inline constexpr std::uint32_t surface_value_voronoi_dimensions_mask = 0x7u;
inline constexpr std::uint32_t surface_value_voronoi_feature_shift = 3u;
inline constexpr std::uint32_t surface_value_voronoi_feature_mask =
    0x7u << surface_value_voronoi_feature_shift;
inline constexpr std::uint32_t surface_value_voronoi_metric_shift = 6u;
inline constexpr std::uint32_t surface_value_voronoi_metric_mask =
    0x3u << surface_value_voronoi_metric_shift;
inline constexpr std::uint32_t surface_value_voronoi_normalize_bit = 1u << 8u;
inline constexpr std::uint32_t surface_value_voronoi_configuration_mask =
    surface_value_voronoi_dimensions_mask | surface_value_voronoi_feature_mask |
    surface_value_voronoi_metric_mask | surface_value_voronoi_normalize_bit;

inline constexpr std::uint32_t surface_value_clamp_mode_mask = 0x1u;
inline constexpr std::uint32_t surface_value_map_range_interpolation_mask =
    0x3u;
inline constexpr std::uint32_t surface_value_map_range_clamp_bit = 1u << 2u;
inline constexpr std::uint32_t surface_value_map_range_configuration_mask =
    surface_value_map_range_interpolation_mask |
    surface_value_map_range_clamp_bit;

static_assert(static_cast<std::uint32_t>(ClampMode::minmax) == 0u);
static_assert(static_cast<std::uint32_t>(ClampMode::range) == 1u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::linear) == 0u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::stepped) == 1u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::smoothstep) ==
              2u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::smootherstep) ==
              3u);
static_assert((surface_value_map_range_interpolation_mask &
               surface_value_map_range_clamp_bit) == 0u);
static_assert((surface_value_map_range_configuration_mask &
               ~surface_value_svm_immediate_value_mask) == 0u);
static_assert((surface_value_voronoi_configuration_mask &
               ~surface_value_svm_immediate_value_mask) == 0u);

[[nodiscard]] constexpr std::uint32_t encode_surface_value_voronoi_immediate(
    std::uint32_t dimensions, VoronoiFeature feature,
    VoronoiDistanceMetric metric, bool normalize) noexcept {
  return dimensions |
         (static_cast<std::uint32_t>(feature)
          << surface_value_voronoi_feature_shift) |
         (static_cast<std::uint32_t>(metric)
          << surface_value_voronoi_metric_shift) |
         (normalize ? surface_value_voronoi_normalize_bit : 0u);
}

[[nodiscard]] constexpr std::uint32_t
decode_surface_value_voronoi_dimensions(std::uint32_t immediate) noexcept {
  return immediate & surface_value_voronoi_dimensions_mask;
}

[[nodiscard]] constexpr VoronoiFeature
decode_surface_value_voronoi_feature(std::uint32_t immediate) noexcept {
  return static_cast<VoronoiFeature>(
      (immediate & surface_value_voronoi_feature_mask) >>
      surface_value_voronoi_feature_shift);
}

[[nodiscard]] constexpr VoronoiDistanceMetric
decode_surface_value_voronoi_metric(std::uint32_t immediate) noexcept {
  return static_cast<VoronoiDistanceMetric>(
      (immediate & surface_value_voronoi_metric_mask) >>
      surface_value_voronoi_metric_shift);
}

[[nodiscard]] constexpr bool
decode_surface_value_voronoi_normalize(std::uint32_t immediate) noexcept {
  return (immediate & surface_value_voronoi_normalize_bit) != 0u;
}

[[nodiscard]] constexpr bool
surface_value_voronoi_immediate_contract_holds() noexcept {
  constexpr auto feature_count =
      static_cast<std::uint32_t>(VoronoiFeature::n_sphere_radius) + 1u;
  constexpr auto metric_count =
      static_cast<std::uint32_t>(VoronoiDistanceMetric::minkowski) + 1u;
  for (auto dimensions = 1u; dimensions <= 4u; ++dimensions) {
    for (auto feature = 0u; feature < feature_count; ++feature) {
      for (auto metric = 0u; metric < metric_count; ++metric) {
        for (auto normalize = 0u; normalize < 2u; ++normalize) {
          const auto encoded = encode_surface_value_voronoi_immediate(
              dimensions, static_cast<VoronoiFeature>(feature),
              static_cast<VoronoiDistanceMetric>(metric), normalize != 0u);
          if ((encoded & ~surface_value_voronoi_configuration_mask) != 0u ||
              decode_surface_value_voronoi_dimensions(encoded) != dimensions ||
              decode_surface_value_voronoi_feature(encoded) !=
                  static_cast<VoronoiFeature>(feature) ||
              decode_surface_value_voronoi_metric(encoded) !=
                  static_cast<VoronoiDistanceMetric>(metric) ||
              decode_surface_value_voronoi_normalize(encoded) !=
                  (normalize != 0u)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr std::uint32_t encode_surface_value_clamp_immediate(
    ClampMode mode) noexcept {
  return static_cast<std::uint32_t>(mode);
}

[[nodiscard]] constexpr ClampMode decode_surface_value_clamp_immediate(
    std::uint32_t immediate) noexcept {
  return static_cast<ClampMode>(immediate & surface_value_clamp_mode_mask);
}

[[nodiscard]] constexpr std::uint32_t encode_surface_value_map_range_immediate(
    MapRangeInterpolation interpolation, bool clamp_result) noexcept {
  return static_cast<std::uint32_t>(interpolation) |
         (clamp_result ? surface_value_map_range_clamp_bit : 0u);
}

[[nodiscard]] constexpr MapRangeInterpolation
decode_surface_value_map_range_interpolation(
    std::uint32_t immediate) noexcept {
  return static_cast<MapRangeInterpolation>(
      immediate & surface_value_map_range_interpolation_mask);
}

[[nodiscard]] constexpr bool decode_surface_value_map_range_clamp(
    std::uint32_t immediate) noexcept {
  return (immediate & surface_value_map_range_clamp_bit) != 0u;
}

[[nodiscard]] constexpr bool surface_value_clamp_immediate_contract_holds()
    noexcept {
  constexpr auto mode_count = static_cast<std::uint32_t>(ClampMode::range) + 1u;
  for (auto mode = 0u; mode < mode_count; ++mode) {
    const auto semantic_mode = static_cast<ClampMode>(mode);
    const auto encoded = encode_surface_value_clamp_immediate(semantic_mode);
    if ((encoded & ~surface_value_clamp_mode_mask) != 0u ||
        decode_surface_value_clamp_immediate(encoded) != semantic_mode) {
      return false;
    }
    for (auto other = 0u; other < mode_count; ++other) {
      if (mode != other &&
          encoded == encode_surface_value_clamp_immediate(
                         static_cast<ClampMode>(other))) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr bool surface_value_map_range_immediate_contract_holds()
    noexcept {
  for (auto interpolation = 0u;
       interpolation < map_range_interpolation_count;
       ++interpolation) {
    for (auto clamp = 0u; clamp < 2u; ++clamp) {
      const auto semantic_interpolation =
          static_cast<MapRangeInterpolation>(interpolation);
      const auto encoded = encode_surface_value_map_range_immediate(
          semantic_interpolation, clamp != 0u);
      if ((encoded & ~surface_value_map_range_configuration_mask) != 0u ||
          decode_surface_value_map_range_interpolation(encoded) !=
              semantic_interpolation ||
          decode_surface_value_map_range_clamp(encoded) != (clamp != 0u)) {
        return false;
      }
      for (auto other_interpolation = 0u;
           other_interpolation < map_range_interpolation_count;
           ++other_interpolation) {
        for (auto other_clamp = 0u; other_clamp < 2u; ++other_clamp) {
          if ((interpolation != other_interpolation || clamp != other_clamp) &&
              encoded == encode_surface_value_map_range_immediate(
                             static_cast<MapRangeInterpolation>(
                                 other_interpolation),
                             other_clamp != 0u)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

static_assert(surface_value_voronoi_immediate_contract_holds());
static_assert(surface_value_clamp_immediate_contract_holds());
static_assert(surface_value_map_range_immediate_contract_holds());

} // namespace psycles::compiler
