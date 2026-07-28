/* SPDX-FileCopyrightText: 2019-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Derived from Blender 4.5.10 Cycles:
 *   intern/cycles/scene/tabulated_sobol.cpp
 *   intern/cycles/kernel/sample/tabulated_sobol.h
 *   intern/cycles/kernel/sample/util.h
 *   intern/cycles/util/hash.h
 */

#include <psycles/sampling/tabulated_sobol.h>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

namespace psycles::sampling::tabulated_sobol {
namespace {

using XorTable = std::array<
    std::array<std::uint32_t, 32u>,
    component_count>;

inline constexpr XorTable xors{{
    {{
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    }},
    {{
        0x00000000u, 0x00000001u, 0x00000001u, 0x00000007u,
        0x00000001u, 0x00000013u, 0x00000015u, 0x0000007fu,
        0x00000001u, 0x00000103u, 0x00000105u, 0x0000070fu,
        0x00000111u, 0x00001333u, 0x00001555u, 0x00007fffu,
        0x00000001u, 0x00010003u, 0x00010005u, 0x0007000fu,
        0x00010011u, 0x00130033u, 0x00150055u, 0x007f00ffu,
        0x00010101u, 0x01030303u, 0x01050505u, 0x070f0f0fu,
        0x01111111u, 0x13333333u, 0x15555555u, 0x7fffffffu,
    }},
    {{
        0x00000000u, 0x00000001u, 0x00000003u, 0x00000001u,
        0x00000005u, 0x0000001fu, 0x0000002bu, 0x0000003du,
        0x00000011u, 0x00000133u, 0x00000377u, 0x00000199u,
        0x00000445u, 0x00001ccfu, 0x00002ddbu, 0x0000366du,
        0x00000101u, 0x00010303u, 0x00030707u, 0x00010909u,
        0x00051515u, 0x001f3f3fu, 0x002b6b6bu, 0x003dbdbdu,
        0x00101011u, 0x01303033u, 0x03707077u, 0x01909099u,
        0x04515145u, 0x1cf3f3cfu, 0x2db6b6dbu, 0x36dbdb6du,
    }},
    {{
        0x00000000u, 0x00000001u, 0x00000000u, 0x00000003u,
        0x0000000du, 0x0000000cu, 0x00000005u, 0x0000004fu,
        0x00000014u, 0x000000e7u, 0x00000329u, 0x0000039cu,
        0x00000011u, 0x00001033u, 0x00000044u, 0x000030bbu,
        0x0000d1cdu, 0x0000c2ecu, 0x00005415u, 0x0004fc3fu,
        0x00015054u, 0x000e5c97u, 0x0032e5b9u, 0x0039725cu,
        0x00000101u, 0x01000303u, 0x00000404u, 0x03000b0bu,
        0x0d001d1du, 0x0c002c2cu, 0x05004545u, 0x4f00cfcfu,
    }},
}};

[[nodiscard]] std::uint32_t reverse_bits(
    std::uint32_t value) noexcept {
    value = ((value >> 1u) & 0x55555555u) |
            ((value & 0x55555555u) << 1u);
    value = ((value >> 2u) & 0x33333333u) |
            ((value & 0x33333333u) << 2u);
    value = ((value >> 4u) & 0x0f0f0f0fu) |
            ((value & 0x0f0f0f0fu) << 4u);
    value = ((value >> 8u) & 0x00ff00ffu) |
            ((value & 0x00ff00ffu) << 8u);
    return (value >> 16u) | (value << 16u);
}

[[nodiscard]] std::uint32_t hash_hp_uint(
    std::uint32_t value) noexcept {
    value ^= value >> 16u;
    value *= 0x21f0aaadu;
    value ^= value >> 15u;
    value *= 0xd35a2d97u;
    value ^= value >> 15u;
    return value ^ 0xe6fe3bebu;
}

[[nodiscard]] std::uint32_t hash_hp_seeded_uint(
    std::uint32_t value,
    std::uint32_t seed) noexcept {
    seed ^= seed << 19u;
    return hash_hp_uint(value ^ seed);
}

[[nodiscard]] float uint_to_float_exclusive(
    std::uint32_t value) noexcept {
    return static_cast<float>(value) *
           (1.0f / 4294967808.0f);
}

[[nodiscard]] float hash_hp_float(
    std::uint32_t value) noexcept {
    return uint_to_float_exclusive(hash_hp_uint(value));
}

[[nodiscard]] std::uint32_t hash_wang_seeded_uint(
    std::uint32_t value,
    std::uint32_t seed) noexcept {
    value = (value ^ 61u) ^ seed;
    value += value << 3u;
    value ^= value >> 4u;
    value *= 0x27d4eb2du;
    return value;
}

[[nodiscard]] std::uint32_t reversed_bit_owen(
    std::uint32_t value,
    std::uint32_t seed) noexcept {
    value ^= value * 0x3d20adeau;
    value += seed;
    value *= (seed >> 16u) | 1u;
    value ^= value * 0x05526c56u;
    value ^= value * 0x53a22864u;
    return value;
}

[[nodiscard]] std::uint32_t nested_uniform_scramble(
    std::uint32_t value,
    std::uint32_t seed) noexcept {
    return reverse_bits(reversed_bit_owen(reverse_bits(value), seed));
}

[[nodiscard]] std::uint32_t hash_shuffle_uint(
    std::uint32_t value,
    std::uint32_t length,
    std::uint32_t seed) noexcept {
    value %= length;
    const auto mask =
        (std::uint32_t{1u}
         << (32u - static_cast<std::uint32_t>(
                        std::countl_zero(length - 1u)))) -
        1u;

    do {
        value ^= seed;
        value *= 0xe170893du;
        value ^= seed >> 16u;
        value ^= (value & mask) >> 4u;
        value ^= seed >> 8u;
        value *= 0x0929eb3fu;
        value ^= seed >> 23u;
        value ^= (value & mask) >> 1u;
        value *= 1u | seed >> 27u;
        value *= 0x6935fa69u;
        value ^= (value & mask) >> 11u;
        value *= 0x74dcb303u;
        value ^= (value & mask) >> 2u;
        value *= 0x9e501cc3u;
        value ^= (value & mask) >> 2u;
        value *= 0xc860a3dfu;
        value &= mask;
        value ^= value >> 5u;
    } while (value >= length);

    return value;
}

void generate_sequence(
    std::span<Sample4> points,
    std::uint32_t rng_seed) {
    auto rng_index =
        hash_hp_seeded_uint(rng_seed, 0x44605a73u);

    points[0].x = hash_hp_float(rng_index++);
    points[0].y = hash_hp_float(rng_index++);
    points[0].z = hash_hp_float(rng_index++);
    points[0].w = hash_hp_float(rng_index++);

    std::uint32_t log_n = 0u;
    for (std::uint32_t n = 1u;
         n < points.size();
         ++log_n, n *= 2u) {
        const auto strata_count =
            static_cast<float>(n * 2u);
        for (std::uint32_t i = 0u;
             i < n && n + i < points.size();
             ++i) {
            const auto occupied_x = static_cast<std::uint32_t>(
                points[i ^ xors[0][log_n]].x *
                strata_count);
            const auto occupied_y = static_cast<std::uint32_t>(
                points[i ^ xors[1][log_n]].y *
                strata_count);
            const auto occupied_z = static_cast<std::uint32_t>(
                points[i ^ xors[2][log_n]].z *
                strata_count);
            const auto occupied_w = static_cast<std::uint32_t>(
                points[i ^ xors[3][log_n]].w *
                strata_count);

            auto &point = points[n + i];
            point.x =
                (static_cast<float>(occupied_x ^ 1u) +
                 hash_hp_float(rng_index++)) /
                strata_count;
            point.y =
                (static_cast<float>(occupied_y ^ 1u) +
                 hash_hp_float(rng_index++)) /
                strata_count;
            point.z =
                (static_cast<float>(occupied_z ^ 1u) +
                 hash_hp_float(rng_index++)) /
                strata_count;
            point.w =
                (static_cast<float>(occupied_w ^ 1u) +
                 hash_hp_float(rng_index++)) /
                strata_count;
        }
    }
}

}// namespace

bool is_valid_sequence_size(
    std::uint32_t sequence_size) noexcept {
    return sequence_size >= min_sequence_size &&
           sequence_size <= max_sequence_size &&
           std::has_single_bit(sequence_size);
}

std::uint32_t sequence_size_for_samples(
    std::uint32_t sample_count) noexcept {
    const auto clamped_count = std::clamp(
        sample_count,
        1u,
        max_sequence_size);
    const auto unbounded =
        clamped_count == 1u
            ? 1u
            : std::uint32_t{1u}
                  << (32u - static_cast<std::uint32_t>(
                                 std::countl_zero(
                                     clamped_count - 1u)));
    return std::clamp(
        unbounded,
        min_sequence_size,
        max_sequence_size);
}

std::uint32_t pixel_hash(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t seed) noexcept {
    const auto qx = 1103515245u * ((x >> 1u) ^ y);
    const auto qy = 1103515245u * ((y >> 1u) ^ x);
    const auto hash = 1103515245u * (qx ^ (qy >> 3u));
    return hash ^ seed;
}

std::uint32_t path_dimension(
    std::uint32_t bounce,
    std::uint32_t bounce_dimension) noexcept {
    return first_bounce_offset +
           bounce * bounce_dimension_count +
           bounce_dimension;
}

std::vector<Sample4> generate_table(
    std::uint32_t sequence_size) {
    if (!is_valid_sequence_size(sequence_size)) {
        throw std::invalid_argument{
            "tabulated Sobol sequence size must be a power of two "
            "between 256 and 8192"};
    }

    std::vector<Sample4> table{
        static_cast<std::size_t>(sequence_size) *
        pattern_count};
    for (std::uint32_t pattern = 0u;
         pattern < pattern_count;
         ++pattern) {
        const auto offset =
            static_cast<std::size_t>(pattern) *
            sequence_size;
        generate_sequence(
            std::span<Sample4>{
                table.data() + offset,
                sequence_size},
            pattern);
    }
    return table;
}

std::uint32_t shuffled_sample_index(
    std::uint32_t sample,
    std::uint32_t dimension,
    std::uint32_t seed,
    std::uint32_t sequence_size) {
    if (!is_valid_sequence_size(sequence_size)) {
        throw std::invalid_argument{
            "invalid tabulated Sobol sequence size"};
    }

    const auto pattern =
        hash_shuffle_uint(dimension, pattern_count, seed);
    const auto sample_mask = sequence_size - 1u;
    const auto shuffled = nested_uniform_scramble(
        sample,
        hash_wang_seeded_uint(dimension, seed));
    sample =
        (sample & ~sample_mask) |
        (shuffled & sample_mask);
    return ((pattern * sequence_size) + sample) %
           (sequence_size * pattern_count);
}

Sample4 sample_4d(
    std::span<const Sample4> table,
    std::uint32_t sequence_size,
    std::uint32_t sample,
    std::uint32_t rng_hash,
    std::uint32_t dimension) {
    const auto expected_size =
        static_cast<std::size_t>(sequence_size) *
        pattern_count;
    if (table.size() != expected_size) {
        throw std::invalid_argument{
            "tabulated Sobol table has an unexpected size"};
    }
    const auto index = shuffled_sample_index(
        sample,
        dimension,
        rng_hash,
        sequence_size);
    return table[index];
}

}// namespace psycles::sampling::tabulated_sobol
