#include <psycles/contract/cycles_pointiness.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using psycles::Vec3f;

void expect(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

void expect_near(
    const float actual,
    const float expected,
    const std::string &message) {
    expect(
        std::abs(actual - expected) <= 2.0e-6f,
        message + ": got " + std::to_string(actual) +
            ", expected " + std::to_string(expected));
}

void test_split_cube_welding_and_edge_deduplication() {
    // A cube represented as six independent faces exercises both pieces of
    // Cycles' formal topology quotient: three equal-position vertices weld at
    // every corner, and the two copies of every geometric edge deduplicate.
    const std::array face_positions{
        std::array<Vec3f, 4u>{
            Vec3f{1.0f, -1.0f, -1.0f},
            Vec3f{1.0f, 1.0f, -1.0f},
            Vec3f{1.0f, 1.0f, 1.0f},
            Vec3f{1.0f, -1.0f, 1.0f}},
        std::array<Vec3f, 4u>{
            Vec3f{-1.0f, -1.0f, 1.0f},
            Vec3f{-1.0f, 1.0f, 1.0f},
            Vec3f{-1.0f, 1.0f, -1.0f},
            Vec3f{-1.0f, -1.0f, -1.0f}},
        std::array<Vec3f, 4u>{
            Vec3f{-1.0f, 1.0f, -1.0f},
            Vec3f{-1.0f, 1.0f, 1.0f},
            Vec3f{1.0f, 1.0f, 1.0f},
            Vec3f{1.0f, 1.0f, -1.0f}},
        std::array<Vec3f, 4u>{
            Vec3f{-1.0f, -1.0f, 1.0f},
            Vec3f{-1.0f, -1.0f, -1.0f},
            Vec3f{1.0f, -1.0f, -1.0f},
            Vec3f{1.0f, -1.0f, 1.0f}},
        std::array<Vec3f, 4u>{
            Vec3f{-1.0f, -1.0f, 1.0f},
            Vec3f{1.0f, -1.0f, 1.0f},
            Vec3f{1.0f, 1.0f, 1.0f},
            Vec3f{-1.0f, 1.0f, 1.0f}},
        std::array<Vec3f, 4u>{
            Vec3f{-1.0f, 1.0f, -1.0f},
            Vec3f{1.0f, 1.0f, -1.0f},
            Vec3f{1.0f, -1.0f, -1.0f},
            Vec3f{-1.0f, -1.0f, -1.0f}}};
    const std::array face_normals{
        Vec3f{1.0f, 0.0f, 0.0f},
        Vec3f{-1.0f, 0.0f, 0.0f},
        Vec3f{0.0f, 1.0f, 0.0f},
        Vec3f{0.0f, -1.0f, 0.0f},
        Vec3f{0.0f, 0.0f, 1.0f},
        Vec3f{0.0f, 0.0f, -1.0f}};

    std::vector<Vec3f> positions;
    std::vector<Vec3f> normals;
    std::vector<std::array<std::uint32_t, 2u>> edges;
    for (std::size_t face = 0u; face < face_positions.size(); ++face) {
        const auto base = static_cast<std::uint32_t>(positions.size());
        positions.insert(
            positions.end(),
            face_positions[face].begin(),
            face_positions[face].end());
        normals.insert(normals.end(), 4u, face_normals[face]);
        for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
            edges.emplace_back(std::array<std::uint32_t, 2u>{
                base + corner,
                base + (corner + 1u) % 4u});
        }
    }

    const auto values =
        psycles::contract::make_cycles_pointiness_attribute(
            positions, normals, edges);
    const auto expected =
        std::acos(-1.0f / std::sqrt(3.0f)) /
        std::numbers::pi_v<float>;
    expect(values.size() == positions.size(), "Pointiness extent changed");
    for (std::size_t vertex = 0u; vertex < values.size(); ++vertex) {
        expect_near(
            values[vertex],
            expected,
            "split-cube Pointiness differs at vertex " +
                std::to_string(vertex));
    }
}

void test_isolated_vertex_and_validation() {
    const std::array positions{Vec3f{0.0f, 0.0f, 0.0f}};
    const std::array normals{Vec3f{0.0f, 0.0f, 1.0f}};
    const auto values =
        psycles::contract::make_cycles_pointiness_attribute(
            positions,
            normals,
            std::span<const std::array<std::uint32_t, 2u>>{});
    expect(values == std::vector<float>{0.0f},
           "isolated Cycles Pointiness is not zero");

    auto rejected_normals = false;
    try {
        static_cast<void>(
            psycles::contract::make_cycles_pointiness_attribute(
                positions,
                std::span<const Vec3f>{},
                std::span<
                    const std::array<std::uint32_t, 2u>>{}));
    } catch (const std::invalid_argument &) {
        rejected_normals = true;
    }
    expect(rejected_normals, "mismatched point normals were accepted");

    auto rejected_edge = false;
    try {
        const std::array<std::array<std::uint32_t, 2u>, 1u>
            invalid_edges{{{0u, 1u}}};
        static_cast<void>(
            psycles::contract::make_cycles_pointiness_attribute(
                positions, normals, invalid_edges));
    } catch (const std::invalid_argument &) {
        rejected_edge = true;
    }
    expect(rejected_edge, "out-of-range Pointiness edge was accepted");
}

}// namespace

int main() {
    try {
        test_split_cube_welding_and_edge_deduplication();
        test_isolated_vertex_and_validation();
    } catch (const std::exception &error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
