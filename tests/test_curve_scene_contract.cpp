#include <psycles/contract/scene.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace psycles::contract;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] CurveGeometryDesc valid_curve() {
    return {
        .name = "Hair",
        .shape = CurveShape::ribbon,
        .subdivisions = 2u,
        .keys = {
            {-0.25f, 0.0f, 0.0f, 0.1f},
            {0.0f, 0.0f, 1.0f, 0.05f},
            {0.25f, 0.0f, 2.0f, 0.0f}},
        .curve_first_key = {0u},
        .material_slots = {},
        .curve_material_slots = {},
        .default_uv_layer = "RootUV",
        .uv_layers = {{"RootUV", {{0.25f, 0.75f}}}},
        .intercept = {0.0f, 0.5f, 1.0f},
        .length = {2.0615528f},
        .random = {0.86031276f}};
}

void test_curve_transaction_validation_is_atomic() {
    SceneDatabase scene;
    SceneDelta initial{.base_revision = 0u, .commands = {}};
    initial.commands.emplace_back(UpsertCurveGeometry{
        .id = GeometryId{1u},
        .value = valid_curve()});

    const auto first = scene.apply(initial);
    expect(first.committed, "valid curve geometry did not commit");
    expect(scene.snapshot().revision == 1u, "scene revision did not advance");
    expect(
        scene.snapshot().curve_geometries.contains(GeometryId{1u}),
        "committed curve geometry is missing");

    auto malformed = valid_curve();
    malformed.name = "Invalid Hair";
    malformed.curve_first_key.front() = 1u;
    SceneDelta invalid{.base_revision = 1u, .commands = {}};
    invalid.commands.emplace_back(UpsertCurveGeometry{
        .id = GeometryId{2u},
        .value = std::move(malformed)});

    const auto rejected = scene.apply(invalid);
    expect(!rejected.committed, "nonzero curve first key was accepted");
    expect(scene.snapshot().revision == 1u, "rejected delta changed revision");
    expect(
        !scene.snapshot().curve_geometries.contains(GeometryId{2u}),
        "rejected curve partially mutated the scene");

    auto mismatched_uv = valid_curve();
    mismatched_uv.name = "Invalid UV Hair";
    mismatched_uv.uv_layers.at("RootUV").clear();
    SceneDelta invalid_uv{.base_revision = 1u, .commands = {}};
    invalid_uv.commands.emplace_back(UpsertCurveGeometry{
        .id = GeometryId{3u},
        .value = std::move(mismatched_uv)});
    expect(!scene.apply(invalid_uv).committed,
           "mismatched curve-domain UV layer was accepted");

    auto missing_default = valid_curve();
    missing_default.name = "Missing Default UV Hair";
    missing_default.default_uv_layer = "AbsentUV";
    SceneDelta invalid_default{.base_revision = 1u, .commands = {}};
    invalid_default.commands.emplace_back(UpsertCurveGeometry{
        .id = GeometryId{4u},
        .value = std::move(missing_default)});
    expect(!scene.apply(invalid_default).committed,
           "missing default curve UV layer was accepted");
}

}// namespace

int main() {
    try {
        test_curve_transaction_validation_is_atomic();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "curve scene contract test failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
