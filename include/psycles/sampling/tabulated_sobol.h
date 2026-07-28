#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace psycles::sampling::tabulated_sobol {

// Blender 4.5.10 Cycles sampling constants from
// intern/cycles/kernel/types.h. Keeping these values in one host/device
// contract prevents control-flow changes from shifting later random samples.
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

[[nodiscard]] std::uint32_t pixel_hash(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t seed) noexcept;

[[nodiscard]] std::uint32_t path_dimension(
    std::uint32_t bounce,
    std::uint32_t bounce_dimension) noexcept;

[[nodiscard]] std::vector<Sample4> generate_table(
    std::uint32_t sequence_size);

[[nodiscard]] std::uint32_t shuffled_sample_index(
    std::uint32_t sample,
    std::uint32_t dimension,
    std::uint32_t seed,
    std::uint32_t sequence_size);

// Lookup for Cycles' default scrambling_distance == 1.0 path. The returned
// four components are the same table row used by its 1D/2D/3D/4D helpers.
[[nodiscard]] Sample4 sample_4d(
    std::span<const Sample4> table,
    std::uint32_t sequence_size,
    std::uint32_t sample,
    std::uint32_t rng_hash,
    std::uint32_t dimension);

}// namespace psycles::sampling::tabulated_sobol
