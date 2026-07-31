#include <psycles/luisa/heterogeneous_volume_transmittance.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

void expect(
    bool condition,
    const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] bool close(
    float actual,
    float expected,
    float tolerance = 5.0e-5f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

[[nodiscard]] bool close(
    luisa::float4 actual,
    luisa::float4 expected) noexcept {
    return close(actual.x, expected.x) &&
           close(actual.y, expected.y) &&
           close(actual.z, expected.z) &&
           close(actual.w, expected.w);
}

class FixtureRandomSource final
    : public HeterogeneousVolumeTrackingRandomSource {

  public:
    Float scatter_distance(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.5f;
    }

    Float shade_offset(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.5f;
    }

    Float expansion_order(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        // floor(log(0.005) / log(0.1)) == 2, hence N == 4 and the
        // truncated geometric PMF is 0.9 * 0.1^2 == 0.009.
        return 0.005f;
    }

    Float transmittance_shade_offset(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.25f;
    }
};

class LinearCollisionProvider final
    : public HeterogeneousVolumeCollisionProvider {

  public:
    VolumeCoefficients evaluate(
        Float distance,
        Bool evaluate_emission,
        VolumePhaseSet *phases)
        const noexcept override {
        static_cast<void>(
            evaluate_emission);
        static_cast<void>(phases);
        return {
            .sigma_t =
                make_float3(
                    0.2f + 0.1f * distance,
                    0.3f + 0.05f * distance,
                    0.1f + 0.2f * distance),
            .sigma_s = make_float3(0.0f),
            .emission = make_float3(0.0f),
            .has_extinction = true,
            .has_scatter = false,
            .has_emission = false};
    }
};

void run_backend(
    std::string_view backend,
    const char *program) {
    Context context{program};
    auto device =
        context.create_device(
            std::string{backend});
    auto stream = device.create_stream();
    auto output =
        device.create_buffer<luisa::float4>(4u);

    Kernel1D evaluate =
        [](BufferFloat4 records) noexcept {
            FixtureRandomSource random;
            LinearCollisionProvider
                collisions;
            const HeterogeneousVolumeTransmittance
                transmittance;

            const auto debiased =
                transmittance.evaluate(
                    {.minimum = 1.0f,
                     .maximum = 3.0f,
                     .sigma_minimum = 0.25f,
                     .sigma_maximum = 1.0f,
                     .object = 0u,
                     .shader = 0u,
                     .node = 0u,
                     .valid = true,
                     .no_overlap = true,
                     .lookup_complete = true},
                    1.0f,
                    3.0f,
                    random,
                    collisions,
                    64u);
            records.write(
                0u,
                make_float4(
                    debiased.transmittance,
                    debiased.probability_mass));
            records.write(
                1u,
                make_float4(
                    cast<float>(
                        debiased.base_samples),
                    cast<float>(
                        debiased
                            .independent_estimators),
                    cast<float>(
                        debiased.evaluations),
                    0.0f));

            const auto zero_range =
                transmittance.evaluate(
                    {.minimum = 1.0f,
                     .maximum = 3.0f,
                     .sigma_minimum = 0.5f,
                     .sigma_maximum = 0.5f,
                     .object = 0u,
                     .shader = 0u,
                     .node = 0u,
                     .valid = true,
                     .no_overlap = true,
                     .lookup_complete = true},
                    1.0f,
                    3.0f,
                    random,
                    collisions,
                    64u);
            records.write(
                2u,
                make_float4(
                    zero_range.transmittance,
                    zero_range
                        .probability_mass));
            records.write(
                3u,
                make_float4(
                    cast<float>(
                        zero_range.base_samples),
                    cast<float>(
                        zero_range
                            .independent_estimators),
                    cast<float>(
                        zero_range.evaluations),
                    0.0f));
        };
    auto shader =
        device.compile(
            evaluate,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = false});

    std::array<luisa::float4, 4u>
        actual{};
    stream
        << shader(output).dispatch(1u)
        << output.copy_to(
               luisa::span{actual})
        << synchronize();

    // These values are the direct evaluation of current Cycles'
    // volume_transmittance(): k=2, N=4, h=0.25, xi=0.25 and PMF=0.009.
    // The second estimator pins Cycles' sigma_c==0 branch, which forces
    // k=N=1 and evaluates the linear extinction at t=1.5.
    constexpr std::array<luisa::float4, 4u>
        expected{{
            {0.439893270f,
             0.448397773f,
             0.327155094f,
             0.009f},
            {2.0f, 4.0f, 8.0f, 0.0f},
            {0.496585304f,
             0.472366553f,
             0.449328964f,
             1.0f},
            {1.0f, 1.0f, 1.0f, 0.0f},
        }};
    for (std::size_t index = 0u;
         index < expected.size();
         ++index) {
        expect(
            close(
                actual[index],
                expected[index]),
            "Cycles residual-ratio transmittance state " +
                std::to_string(index) +
                " changed on " +
                std::string{backend});
    }
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto backend =
            std::string_view{
                argc > 1
                    ? argv[1]
                    : "fallback"};
        run_backend(
            backend,
            argv[0]);
        std::cout
            << "All current-Cycles residual-ratio "
               "transmittance fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Heterogeneous volume transmittance fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
