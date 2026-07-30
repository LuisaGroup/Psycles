#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_sampler.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/sampling/tabulated_sobol.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_sampler {

static_assert(
    sampling::tabulated_sobol::pattern_count == 256u,
    "The device Sobol shuffle specializes Cycles' 256-pattern table.");

using luisa::compute::BufferFloat4;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4;
using luisa::compute::UInt;
using luisa::compute::make_float2;
using luisa::compute::make_float3;

// Luisa DSL lowering of Cycles' TABULATED_SOBOL path for
// scrambling_distance == 1.0, verified through Blender main@4fe17ef6.
// The host side only builds the immutable lookup table; hashing, shuffling,
// dimension selection, and lookup remain device-side Luisa operations.

[[nodiscard]] inline UInt reverse_bits(UInt value) noexcept {
    value =
        ((value >> 1u) & 0x55555555u) |
        ((value & 0x55555555u) << 1u);
    value =
        ((value >> 2u) & 0x33333333u) |
        ((value & 0x33333333u) << 2u);
    value =
        ((value >> 4u) & 0x0f0f0f0fu) |
        ((value & 0x0f0f0f0fu) << 4u);
    value =
        ((value >> 8u) & 0x00ff00ffu) |
        ((value & 0x00ff00ffu) << 8u);
    return (value >> 16u) | (value << 16u);
}

[[nodiscard]] inline UInt reversed_bit_owen(
    UInt value,
    UInt seed) noexcept {
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16u) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return value;
}

[[nodiscard]] inline UInt nested_uniform_scramble(
    UInt value,
    UInt seed) noexcept {
    return reverse_bits(
        reversed_bit_owen(reverse_bits(value), seed));
}

[[nodiscard]] inline UInt hash_wang_seeded_uint(
    UInt value,
    UInt seed) noexcept {
    value = (value ^ 61u) ^ seed;
    value += value << 3u;
    value ^= value >> 4u;
    value *= 0x27d4eb2du;
    return value;
}

[[nodiscard]] inline UInt shuffled_pattern_index(
    UInt dimension,
    UInt seed) noexcept {
    // NUM_TAB_SOBOL_PATTERNS is exactly 256, so the rejection loop in
    // Cycles' generic hash_shuffle_uint executes exactly once.
    UInt value = dimension & 0xffu;
    constexpr auto mask = std::uint32_t{0xffu};
    value ^= seed;
    value *= 0xe170893du;
    value ^= seed >> 16u;
    value ^= (value & mask) >> 4u;
    value ^= seed >> 8u;
    value *= 0x0929eb3fu;
    value ^= seed >> 23u;
    value ^= (value & mask) >> 1u;
    value *= 1u | (seed >> 27u);
    value *= 0x6935fa69u;
    value ^= (value & mask) >> 11u;
    value *= 0x74dcb303u;
    value ^= (value & mask) >> 2u;
    value *= 0x9e501cc3u;
    value ^= (value & mask) >> 2u;
    value *= 0xc860a3dfu;
    value &= mask;
    value ^= value >> 5u;
    return value;
}

[[nodiscard]] inline UInt pixel_hash(
    UInt x,
    UInt y,
    UInt seed) noexcept {
    UInt qx = 1103515245u * ((x >> 1u) ^ y);
    UInt qy = 1103515245u * ((y >> 1u) ^ x);
    UInt hash = 1103515245u * (qx ^ (qy >> 3u));
    return hash ^ seed;
}

[[nodiscard]] inline UInt path_dimension(
    UInt bounce,
    std::uint32_t bounce_dimension) noexcept {
    return sampling::tabulated_sobol::first_bounce_offset +
           bounce *
               sampling::tabulated_sobol::bounce_dimension_count +
           bounce_dimension;
}

// Production path sampling is addressed from Cycles' explicit RNG state, not
// from an enclosing loop counter. Surface transparency, portals, volume
// bounds, subsurface walks, and future split paths can all advance rng_offset
// without sharing the same notion of a renderer loop iteration.
[[nodiscard]] inline UInt path_state_dimension(
    UInt rng_offset,
    std::uint32_t bounce_dimension) noexcept {
    return rng_offset + bounce_dimension;
}

[[nodiscard]] inline UInt shuffled_sample_index(
    UInt sample,
    UInt dimension,
    UInt seed,
    UInt sequence_size) noexcept {
    const auto pattern =
        shuffled_pattern_index(dimension, seed);
    const auto sample_mask = sequence_size - 1u;
    const auto shuffled = nested_uniform_scramble(
        sample,
        hash_wang_seeded_uint(dimension, seed));
    sample =
        (sample & ~sample_mask) |
        (shuffled & sample_mask);
    return ((pattern * sequence_size) + sample) %
           (sequence_size *
            sampling::tabulated_sobol::pattern_count);
}

[[nodiscard]] inline Float4 sample_4d(
    const BufferFloat4 &table,
    UInt sequence_size,
    UInt sample,
    UInt rng_hash,
    UInt dimension) noexcept {
    return table.read(shuffled_sample_index(
        sample,
        dimension,
        rng_hash,
        sequence_size));
}

[[nodiscard]] inline Float sample_1d(
    const BufferFloat4 &table,
    UInt sequence_size,
    UInt sample,
    UInt rng_hash,
    UInt dimension) noexcept {
    return sample_4d(
               table,
               sequence_size,
               sample,
               rng_hash,
               dimension)
        .x;
}

[[nodiscard]] inline Float2 sample_2d(
    const BufferFloat4 &table,
    UInt sequence_size,
    UInt sample,
    UInt rng_hash,
    UInt dimension) noexcept {
    return sample_4d(
               table,
               sequence_size,
               sample,
               rng_hash,
               dimension)
        .xy();
}

[[nodiscard]] inline Float3 sample_3d(
    const BufferFloat4 &table,
    UInt sequence_size,
    UInt sample,
    UInt rng_hash,
    UInt dimension) noexcept {
    return sample_4d(
               table,
               sequence_size,
               sample,
               rng_hash,
               dimension)
        .xyz();
}

}// namespace psycles::luisa_backend::cycles_sampler
