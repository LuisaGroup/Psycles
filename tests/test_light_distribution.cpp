#include <psycles/compiler/cycles_emission_sampling.h>
#include <psycles/sampling/light_distribution.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void require_near(
    float actual,
    float expected,
    std::string_view message) {
    require(
        std::abs(actual - expected) <= 1.0e-6f,
        message);
}

}// namespace

int main() {
    using psycles::Vec3f;
    using psycles::compiler::resolve_cycles_emission_sampling;
    using psycles::contract::EmissionSampling;
    using namespace psycles::sampling;

    require(
        resolve_cycles_emission_sampling(
            EmissionSampling::automatic,
            Vec3f{0.5f, 0.0f, 0.0f}) ==
            EmissionSampling::none,
        "AUTO accepted Cycles' strict threshold endpoint");
    require(
        resolve_cycles_emission_sampling(
            EmissionSampling::automatic,
            Vec3f{0.0f, -0.5001f, 0.0f}) ==
            EmissionSampling::front_back,
        "AUTO did not use the absolute maximum emission estimate");
    require(
        resolve_cycles_emission_sampling(
            EmissionSampling::front,
            Vec3f{}) == EmissionSampling::none,
        "zero emission did not disable an explicit sampling policy");
    require(
        resolve_cycles_emission_sampling(
            EmissionSampling::back,
            Vec3f{0.01f, 0.0f, 0.0f}) ==
            EmissionSampling::back,
        "non-zero emission changed an explicit sampling policy");
    require(
        resolve_cycles_emission_sampling(
            EmissionSampling::automatic,
            Vec3f{5.0f, 0.0f, 0.0f},
            true) == EmissionSampling::none &&
            resolve_cycles_emission_sampling(
                EmissionSampling::automatic,
                Vec3f{5.001f, 0.0f, 0.0f},
                true) == EmissionSampling::front_back,
        "automatic value-to-closure importance scale diverged from Cycles");

    // Flat Archiviz exposed this exact population bug: a large display with
    // AUTO emission 0.3 must be absent, while the following unit-estimate
    // lamp remains emitter zero. Keeping the policy projection separate from
    // CDF construction makes the ordering proof explicit.
    const auto display_policy = resolve_cycles_emission_sampling(
        EmissionSampling::automatic,
        Vec3f{0.3f, 0.3f, 0.3f});
    const auto lamp_policy = resolve_cycles_emission_sampling(
        EmissionSampling::automatic,
        Vec3f{1.0f, 1.0f, 1.0f});
    require(
        display_policy == EmissionSampling::none &&
            lamp_policy == EmissionSampling::front_back,
        "effective sampling policies do not reproduce Cycles emitter discovery");

    const float triangle_areas[]{1.0f, 3.0f};
    const auto mixed = build_cycles_light_distribution(
        std::span{triangle_areas}, 2u, true);
    require(mixed.usable(), "mixed distribution is unusable");
    require(
        mixed.emitter_count == 5u,
        "mixed emitter count mismatch");
    require(
        mixed.entries.size() == 6u,
        "mixed sentinel is missing");
    require_near(
        mixed.triangle_area_pdf,
        0.125f,
        "triangle area PDF mismatch");
    require_near(
        mixed.light_selection_pdf,
        1.0f / 6.0f,
        "lamp selection PDF mismatch");

    const float expected_cdf[]{
        0.0f,
        0.125f,
        0.5f,
        2.0f / 3.0f,
        5.0f / 6.0f,
        1.0f};
    const float expected_pdf[]{
        0.125f,
        0.375f,
        1.0f / 6.0f,
        1.0f / 6.0f,
        1.0f / 6.0f};
    for (std::size_t index = 0u;
         index < mixed.entries.size();
         ++index) {
        require_near(
            mixed.entries[index].cumulative,
            expected_cdf[index],
            "mixed cumulative distribution mismatch");
        if (index < mixed.emitter_count) {
            require_near(
                mixed.entries[index].selection_pdf,
                expected_pdf[index],
                "mixed selection PDF mismatch");
        }
    }
    require(
        mixed.entries[0u].kind ==
            LightDistributionEmitterKind::emissive_triangle &&
            mixed.entries[0u].index == 0u,
        "first triangle identity mismatch");
    require(
        mixed.entries[1u].kind ==
            LightDistributionEmitterKind::emissive_triangle &&
            mixed.entries[1u].index == 1u,
        "second triangle identity mismatch");
    require(
        mixed.entries[2u].kind ==
            LightDistributionEmitterKind::analytic_light &&
            mixed.entries[2u].index == 0u,
        "first analytic light identity mismatch");
    require(
        mixed.entries[4u].kind ==
            LightDistributionEmitterKind::environment,
        "environment identity mismatch");

    const float degenerate_areas[]{0.0f, 2.0f, -1.0f};
    const auto triangles_only =
        build_cycles_light_distribution(
            std::span{degenerate_areas}, 0u, false);
    require(
        triangles_only.usable(),
        "non-degenerate triangle distribution is unusable");
    require_near(
        triangles_only.entries[0u].selection_pdf,
        0.0f,
        "zero-area triangle received probability");
    require_near(
        triangles_only.entries[1u].selection_pdf,
        1.0f,
        "positive triangle did not receive all probability");
    require_near(
        triangles_only.entries[2u].selection_pdf,
        0.0f,
        "invalid triangle area received probability");
    const auto lights_only =
        build_cycles_light_distribution({}, 2u, false);
    require(lights_only.usable(), "lamp-only distribution is unusable");
    require_near(
        lights_only.entries[0u].selection_pdf,
        0.5f,
        "first lamp probability mismatch");
    require_near(
        lights_only.entries[1u].selection_pdf,
        0.5f,
        "second lamp probability mismatch");

    const auto empty =
        build_cycles_light_distribution({}, 0u, false);
    require(
        !empty.usable(),
        "empty distribution unexpectedly became usable");
    std::cout << "Cycles flat light distribution tests passed.\n";
    return EXIT_SUCCESS;
}
