#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/cycles_volume_phase.h>
#include <psycles/luisa/heterogeneous_volume_segment.h>

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

inline constexpr std::uint32_t
    tracking_seed = 0xe35fad82u;
inline constexpr std::uint32_t
    path_rng_offset = 16u;

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

[[nodiscard]] bool equal(
    luisa::uint4 actual,
    luisa::uint4 expected) noexcept {
    return actual.x == expected.x &&
           actual.y == expected.y &&
           actual.z == expected.z &&
           actual.w == expected.w;
}

class SingleSegmentSequence final
    : public VolumeMajorantSegmentSequence {

  public:
    VolumeMajorantSegment
    current() const noexcept override {
        return {
            .minimum = 0.0f,
            .maximum = 5.0f,
            .sigma_minimum = 0.0f,
            .sigma_maximum = 2.0f,
            .object = 10u,
            .shader = 20u,
            .node = 0u,
            .valid = true,
            .no_overlap = true,
            .lookup_complete = true};
    }

    Bool advance(
        Float shade_offset) noexcept override {
        static_cast<void>(shade_offset);
        return false;
    }
};

class FixtureRandomSource final
    : public HeterogeneousVolumeTrackingRandomSource {

  private:
    UInt _initial_offset;

  public:
    explicit FixtureRandomSource(
        UInt initial_offset) noexcept
        : _initial_offset{
              initial_offset} {}

    Float scatter_distance(
        UInt rng_offset)
        const noexcept override {
        auto result = def(0.9999f);
        result = select(
            result,
            0.6321205588f,
            rng_offset ==
                _initial_offset);
        result = select(
            result,
            0.7768698399f,
            rng_offset ==
                _initial_offset +
                    heterogeneous_volume_tracking_rng_stride);
        return result;
    }

    Float shade_offset(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.5f;
    }
};

class FixtureCollisionProvider final
    : public HeterogeneousVolumeCollisionProvider {

  private:
    UInt *_calls;

  public:
    explicit FixtureCollisionProvider(
        UInt *calls) noexcept
        : _calls{calls} {}

    VolumeCoefficients evaluate(
        Float distance,
        Bool evaluate_emission,
        VolumePhaseSet *phases)
        const noexcept override {
        *_calls += 1u;
        const auto scattering =
            distance >= 1.0f;
        const auto sigma_t =
            select(
                make_float3(
                    0.5f, 1.0f, 1.5f),
                make_float3(
                    1.0f, 1.5f, 2.0f),
                scattering);
        const auto sigma_s =
            select(
                make_float3(0.0f),
                make_float3(
                    0.4f, 0.8f, 1.2f),
                scattering);
        const auto emission =
            select(
                make_float3(
                    0.1f, 0.2f, 0.3f),
                make_float3(
                    0.2f, 0.1f, 0.4f),
                scattering);
        if (phases != nullptr) {
            phases->add(
                cycles_volume_phase::
                    henyey_greenstein(
                        0.42f),
                sigma_s);
        }
        return {
            .sigma_t = sigma_t,
            .sigma_s = sigma_s,
            .emission =
                select(
                    make_float3(0.0f),
                    emission,
                    evaluate_emission),
            .has_extinction = true,
            .has_scatter = scattering,
            .has_emission =
                evaluate_emission};
    }
};

[[nodiscard]] UInt flags(
    const HeterogeneousVolumeSegmentResult
        &result) noexcept {
    return select(
               0u,
               1u,
               result.transport
                   .selected_scatter) |
           select(
               0u,
               2u,
               result.transport
                   .traversal_exhausted) |
           select(
               0u,
               4u,
               result.transport
                   .step_limit_exceeded) |
           select(
               0u,
               8u,
               result.transport
                   .majorant_exceeded) |
           select(
               0u,
               16u,
               result.transport.active) |
           select(
               0u,
               32u,
               result.scattered) |
           select(
               0u,
               64u,
               result.phase_failed);
}

void run_backend(
    std::string_view backend,
    const char *program) {
    Context context{program};
    auto device =
        context.create_device(backend);
    auto stream =
        device.create_stream();
    auto float_output =
        device.create_buffer<luisa::float4>(6u);
    auto uint_output =
        device.create_buffer<luisa::uint4>(2u);

    const auto segment =
        make_heterogeneous_volume_segment_component(
            4u);
    Kernel1D evaluate =
        [segment = segment.get()](
            BufferFloat4 float_records,
            BufferUInt4 uint_records) noexcept {
            const auto initial_offset =
                cycles_sampler::
                    scramble_path_offset(
                        path_rng_offset,
                        tracking_seed);
            SingleSegmentSequence sequence;
            FixtureRandomSource random{
                initial_offset};
            UInt collision_calls = 0u;
            FixtureCollisionProvider collisions{
                &collision_calls};
            const auto phase_axis =
                normalize(
                    make_float3(
                        0.2f,
                        -0.3f,
                        0.9327379f));
            const auto result =
                segment->emit(
                    sequence,
                    random,
                    collisions,
                    1.0f,
                    0.0f,
                    5.0f,
                    phase_axis,
                    make_float3(1.0f),
                    0.1f,
                    make_float2(
                        0.17f, 0.83f),
                    initial_offset,
                    false);
            float_records.write(
                0u,
                make_float4(
                    result.transport
                        .throughput,
                    result.transport
                        .distance));
            float_records.write(
                1u,
                make_float4(
                    result.transport.emission,
                    result.transport
                        .null_transmittance));
            float_records.write(
                2u,
                make_float4(
                    result.transport
                        .reservoir_random,
                    result.transport
                        .optical_depth,
                    result.coefficients
                        .sigma_t.x,
                    result.coefficients
                        .sigma_s.x));
            float_records.write(
                3u,
                make_float4(
                    result.phase.direction,
                    result.phase.pdf));
            float_records.write(
                4u,
                make_float4(
                    result.coefficients
                        .sigma_t,
                    result.coefficients
                        .emission.x));
            float_records.write(
                5u,
                make_float4(
                    result.phase
                        .sampled_roughness,
                    result.phase
                        .selection_rescaled,
                    cast<float>(
                        result.phase
                            .closure_index),
                    cast<float>(
                        result.phase
                            .closure_type)));
            uint_records.write(
                0u,
                make_uint4(
                    result.transport.steps,
                    result.transport
                        .next_tracking_rng_offset,
                    collision_calls,
                    flags(result)));
            uint_records.write(
                1u,
                make_uint4(
                    initial_offset,
                    path_rng_offset,
                    tracking_seed,
                    0u));
        };
    auto shader =
        device.compile(
            evaluate,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = false});

    std::array<luisa::float4, 6u>
        actual_float{};
    std::array<luisa::uint4, 2u>
        actual_uint{};
    stream
        << shader(
               float_output,
               uint_output)
               .dispatch(1u)
        << float_output.copy_to(
               luisa::span{actual_float})
        << uint_output.copy_to(
               luisa::span{actual_uint})
        << synchronize();

    // Transport values are pinned to volume_integrate_step_scattering() and
    // volume_distance_sampling_finalize() in official Cycles main
    // 0ae970969e3f. The phase value reuses the official Cycles CPU-kernel
    // oracle already covered exhaustively by test_luisa_volume_phase.
    constexpr std::array<luisa::float4, 6u>
        expected_float{{
            {0.268873110f,
             0.358497481f,
             0.268873110f,
             1.25f},
            {0.125f,
             0.125f,
             0.2f,
             0.442116024f},
            {0.179248740f,
             10.0f,
             1.0f,
             0.4f},
            {-0.117953472f,
             -0.899912298f,
             -0.419815481f,
             0.044300843f},
            {1.0f, 1.5f, 2.0f, 0.2f},
            {0.58f, 0.17f, 0.0f, 0.0f},
        }};
    for (std::size_t index = 0u;
         index < expected_float.size();
         ++index) {
        expect(
            close(
                actual_float[index],
                expected_float[index]),
            "Cycles heterogeneous segment float state " +
                std::to_string(index) +
                " changed on " +
                std::string{backend});
    }

    constexpr std::uint32_t
        initial_offset = 0x15982c28u;
    const std::array<luisa::uint4, 2u>
        expected_uint{{
            {3u,
             initial_offset + 32u,
             3u,
             51u},
            {initial_offset,
             path_rng_offset,
             tracking_seed,
             0u},
        }};
    for (std::size_t index = 0u;
         index < expected_uint.size();
         ++index) {
        expect(
            equal(
                actual_uint[index],
                expected_uint[index]),
            "Cycles heterogeneous segment integer state " +
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
            << "All current-Cycles heterogeneous segment "
               "fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Heterogeneous volume segment fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
