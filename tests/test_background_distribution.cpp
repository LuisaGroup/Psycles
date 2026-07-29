#include <psycles/sampling/background_distribution.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void require_near(float actual,
                  float expected,
                  std::string_view message,
                  float tolerance = 1.0e-6f) {
    require(std::abs(actual - expected) <= tolerance, message);
}

} // namespace

int main() {
    using psycles::Vec3f;
    using psycles::sampling::build_cycles_background_map_distribution;

    // Two equal-latitude rows make the sin(theta) factor identical. The
    // values are chosen so every conditional and marginal term has a simple
    // analytic result.
    constexpr std::array radiance{Vec3f{1.0f, 1.0f, 1.0f},
                                  Vec3f{3.0f, 3.0f, 3.0f},
                                  Vec3f{2.0f, 2.0f, 2.0f},
                                  Vec3f{6.0f, 6.0f, 6.0f}};
    const auto distribution =
        build_cycles_background_map_distribution(radiance, 2u, 2u);
    require(distribution.valid(), "2x2 distribution is invalid");
    require(distribution.conditional.size() == 6u,
            "conditional sentinel layout mismatch");
    require(distribution.marginal.size() == 3u,
            "marginal sentinel layout mismatch");

    const auto sine = std::sin(0.25f * 3.14159265358979323846f);
    require_near(distribution.conditional[0u].function,
                 sine,
                 "row 0 function[0] mismatch");
    require_near(distribution.conditional[1u].function,
                 3.0f * sine,
                 "row 0 function[1] mismatch");
    require_near(distribution.conditional[1u].cumulative,
                 0.25f,
                 "row 0 normalized CDF mismatch");
    require_near(distribution.conditional[2u].function,
                 2.0f * sine,
                 "row 0 integral mismatch");
    require_near(distribution.conditional[5u].function,
                 4.0f * sine,
                 "row 1 integral mismatch");
    require_near(distribution.marginal[1u].cumulative,
                 1.0f / 3.0f,
                 "marginal normalized CDF mismatch");
    require_near(distribution.marginal[2u].function,
                 3.0f * sine,
                 "marginal integral mismatch");
    require_near(distribution.marginal[2u].cumulative,
                 1.0f,
                 "marginal sentinel mismatch");

    constexpr std::array black{Vec3f{}, Vec3f{}, Vec3f{}, Vec3f{}};
    const auto degenerate =
        build_cycles_background_map_distribution(black, 2u, 2u);
    require(degenerate.valid(), "black distribution is invalid");
    require_near(degenerate.marginal.back().function,
                 0.0f,
                 "black map acquired nonzero mass");
    require_near(degenerate.marginal.back().cumulative,
                 1.0f,
                 "black map sentinel mismatch");

    auto rejected = false;
    try {
        static_cast<void>(build_cycles_background_map_distribution(
            std::span<const Vec3f>{radiance}.first(3u), 2u, 2u));
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "radiance cardinality mismatch was accepted");
    return EXIT_SUCCESS;
}
