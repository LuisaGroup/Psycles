#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace psycles::sampling::tabulated_sobol {

// Cycles sampling constants, verified through Blender main@4fe17ef6.
// Keeping these values in one host/device contract prevents control-flow
// changes from shifting later random samples.
inline constexpr std::uint32_t min_sequence_size = 256u;
inline constexpr std::uint32_t max_sequence_size = 8192u;
inline constexpr std::uint32_t pattern_count = 256u;
inline constexpr std::uint32_t component_count = 4u;

inline constexpr std::uint32_t camera_filter_dimension = 0u;
inline constexpr std::uint32_t camera_lens_time_dimension = 1u;

inline constexpr std::uint32_t terminate_dimension = 0u;
inline constexpr std::uint32_t light_dimension = 1u;
inline constexpr std::uint32_t light_terminate_dimension = 2u;
inline constexpr std::uint32_t surface_bsdf_dimension = 3u;
inline constexpr std::uint32_t surface_ao_dimension = 4u;
inline constexpr std::uint32_t surface_bevel_dimension = 5u;
inline constexpr std::uint32_t surface_bsdf_guiding_dimension = 6u;
inline constexpr std::uint32_t volume_phase_dimension = 3u;
inline constexpr std::uint32_t volume_reservoir_dimension = 4u;
inline constexpr std::uint32_t volume_scatter_distance_dimension = 5u;
inline constexpr std::uint32_t volume_expansion_order_dimension = 6u;
inline constexpr std::uint32_t volume_shade_offset_dimension = 7u;
inline constexpr std::uint32_t
    volume_phase_guiding_distance_dimension = 8u;
inline constexpr std::uint32_t
    volume_phase_guiding_equiangular_dimension = 9u;
inline constexpr std::uint32_t bounce_dimension_count = 16u;
inline constexpr std::uint32_t first_bounce_offset =
    bounce_dimension_count;

struct Sample4 {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float w{0.0f};
};

static_assert(sizeof(Sample4) == sizeof(float) * component_count);

[[nodiscard]] bool is_valid_sequence_size(
    std::uint32_t sequence_size) noexcept;

// Matches Cycles' next_power_of_two(clamped_aa_samples - 1), bounded by
// MIN_TAB_SOBOL_SAMPLES and MAX_TAB_SOBOL_SAMPLES.
[[nodiscard]] std::uint32_t sequence_size_for_samples(
    std::uint32_t sample_count) noexcept;

[[nodiscard]] std::vector<Sample4> generate_table(
    std::uint32_t sequence_size);

}// namespace psycles::sampling::tabulated_sobol
