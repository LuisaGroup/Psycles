#include "path_tracer_generated_coordinates.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

using namespace psycles;
using namespace psycles::contract;
using namespace psycles::luisa_backend::detail;

[[nodiscard]] bool near(
    float actual,
    float expected) noexcept {
    return std::abs(actual - expected) <=
           1.0e-6f;
}

[[nodiscard]] bool near(
    Vec3f actual,
    Vec3f expected) noexcept {
    return
        near(actual.x, expected.x) &&
        near(actual.y, expected.y) &&
        near(actual.z, expected.z);
}

[[nodiscard]] bool check(
    bool condition,
    const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

}// namespace

int main() {
    TriangleMeshDesc derived;
    derived.positions = {
        Vec3f{-2.0f, 4.0f, 7.0f},
        Vec3f{2.0f, 10.0f, 7.0f}};
    const auto bounds =
        make_generated_coordinate_mapping(
            derived);
    if (!check(
            near(
                bounds.apply(
                    derived.positions[0u]),
                Vec3f{0.0f, 0.0f, 0.5f}) &&
                near(
                    bounds.apply(
                        derived.positions[1u]),
                    Vec3f{1.0f, 1.0f, 0.5f}) &&
                near(
                    bounds.apply(
                        Vec3f{0.0f, 7.0f, 100.0f}),
                    Vec3f{0.5f, 0.5f, 0.5f}),
            "derived Generated mapping changed")) {
        return EXIT_FAILURE;
    }

    TriangleMeshDesc explicit_geometry;
    Mat4f explicit_transform;
    explicit_transform.elements = {
        2.0f, 3.0f, 5.0f, 0.0f,
        7.0f, 11.0f, 13.0f, 0.0f,
        17.0f, 19.0f, 23.0f, 0.0f,
        29.0f, 31.0f, 37.0f, 1.0f};
    explicit_geometry.generated_transform =
        explicit_transform;
    const auto authored =
        make_generated_coordinate_mapping(
            explicit_geometry);
    if (!check(
            authored.object_to_generated ==
                    explicit_transform &&
                near(
                    authored.apply(
                        Vec3f{1.0f, 2.0f, 3.0f}),
                    Vec3f{
                        96.0f, 113.0f, 137.0f}),
            "explicit Generated affine transform changed")) {
        return EXIT_FAILURE;
    }

    std::cout
        << "Psycles Generated-coordinate mapping regression passed\n";
    return EXIT_SUCCESS;
}
