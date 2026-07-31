#include <psycles/sampling/tabulated_sobol.h>

#include <bit>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace psycles::sampling::tabulated_sobol;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] std::uint64_t table_fingerprint(
    const std::vector<Sample4> &table) noexcept {
    auto hash = std::uint64_t{14695981039346656037ull};
    for (const auto &sample : table) {
        for (const auto value : {
                 sample.x,
                 sample.y,
                 sample.z,
                 sample.w}) {
            auto bits = std::bit_cast<std::uint32_t>(value);
            for (auto byte = 0u; byte < 4u; ++byte) {
                hash ^= (bits >> (byte * 8u)) & 0xffu;
                hash *= 1099511628211ull;
            }
        }
    }
    return hash;
}

void expect_bits(
    Sample4 sample,
    std::array<std::uint32_t, 4u> expected,
    const std::string &message) {
    const std::array actual{
        std::bit_cast<std::uint32_t>(sample.x),
        std::bit_cast<std::uint32_t>(sample.y),
        std::bit_cast<std::uint32_t>(sample.z),
        std::bit_cast<std::uint32_t>(sample.w)};
    expect(actual == expected, message);
}

void test_sequence_sizing_and_dimensions() {
    expect(sequence_size_for_samples(0u) == 256u,
           "zero samples did not clamp to the minimum sequence");
    expect(sequence_size_for_samples(1u) == 256u,
           "one sample did not clamp to the minimum sequence");
    expect(sequence_size_for_samples(256u) == 256u,
           "256 samples selected the wrong sequence");
    expect(sequence_size_for_samples(257u) == 512u,
           "257 samples did not advance to the next power of two");
    expect(sequence_size_for_samples(9000u) == 8192u,
           "large sample count did not clamp to the maximum sequence");

    expect(first_bounce_offset + terminate_dimension == 16u,
           "first-bounce termination dimension changed");
    expect(first_bounce_offset + light_dimension == 17u,
           "first-bounce light dimension changed");
    expect(first_bounce_offset + surface_bsdf_dimension == 19u,
           "first-bounce BSDF dimension changed");
    expect(
        volume_phase_dimension == surface_bsdf_dimension,
        "Cycles volume-phase/BSDF dimension alias changed");
    expect(
        volume_reservoir_dimension == surface_ao_dimension,
        "Cycles volume-reservoir/AO dimension alias changed");
    expect(
        first_bounce_offset +
                volume_scatter_distance_dimension ==
            21u,
        "first-bounce volume scatter-distance dimension changed");
    expect(
        volume_phase_guiding_equiangular_dimension == 9u,
        "volume equiangular-guiding dimension changed");
    expect(first_bounce_offset + bounce_dimension_count +
                   light_dimension ==
               33u,
           "next-bounce light dimension changed");
}

void test_current_cycles_table_fixture() {
    const auto table = generate_table(256u);
    expect(table.size() == 256u * 256u,
           "256-sample table has the wrong row count");

    // Verified unchanged against Blender main@4fe17ef6's authoritative LUT
    // construction. The FNV-1a pass covers every IEEE-754 component in all
    // 256 patterns rather than checking only representative rows.
    constexpr auto expected_fingerprint =
        std::uint64_t{10168221949122797448ull};
    const auto actual_fingerprint = table_fingerprint(table);
    expect(
        actual_fingerprint == expected_fingerprint,
        "Cycles table fingerprint changed: " +
            std::to_string(actual_fingerprint));

    expect_bits(
        table[0],
        {0x3f2ba2d4u, 0x3f4e3b6bu, 0x3ea1b218u, 0x3ef603ccu},
        "first tabulated-Sobol row changed");
    expect_bits(
        table[1],
        {0x3dc33810u, 0x3ce32146u, 0x3f4ad558u, 0x3f3a60a0u},
        "second tabulated-Sobol row changed");

}

}// namespace

int main() {
    try {
        test_sequence_sizing_and_dimensions();
        test_current_cycles_table_fixture();
        std::cout
            << "All current-Cycles tabulated-Sobol LUT tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Tabulated-Sobol test failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
