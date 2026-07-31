#include <psycles/luisa/heterogeneous_volume_shadow.h>

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

[[nodiscard]] bool equal(
    luisa::uint4 actual,
    luisa::uint4 expected) noexcept {
    return actual.x == expected.x &&
           actual.y == expected.y &&
           actual.z == expected.z &&
           actual.w == expected.w;
}

class GroupingSegmentSequence final
    : public VolumeMajorantSegmentSequence {

  private:
    UInt _index{0u};

  public:
    VolumeMajorantSegment current()
        const noexcept override {
        Float minimum = 0.0f;
        Float maximum = 1.0f;
        Float sigma_minimum = 0.0f;
        Float sigma_maximum = 0.2f;
        minimum =
            select(
                minimum,
                1.0f,
                _index == 1u);
        maximum =
            select(
                maximum,
                2.0f,
                _index == 1u);
        sigma_maximum =
            select(
                sigma_maximum,
                1.2f,
                _index == 1u);
        minimum =
            select(
                minimum,
                2.0f,
                _index == 2u);
        maximum =
            select(
                maximum,
                3.0f,
                _index == 2u);
        sigma_maximum =
            select(
                sigma_maximum,
                1.5f,
                _index == 2u);
        minimum =
            select(
                minimum,
                3.0f,
                _index == 3u);
        maximum =
            select(
                maximum,
                4.0f,
                _index == 3u);
        sigma_minimum =
            select(
                sigma_minimum,
                0.1f,
                _index == 3u);
        sigma_maximum =
            select(
                sigma_maximum,
                0.2f,
                _index == 3u);
        return {
            .minimum = minimum,
            .maximum = maximum,
            .sigma_minimum =
                sigma_minimum,
            .sigma_maximum =
                sigma_maximum,
            .object = 0u,
            .shader = 0u,
            .node = _index,
            .valid = _index < 4u,
            .no_overlap = true,
            .lookup_complete = true};
    }

    Bool advance(
        Float shade_offset)
        noexcept override {
        static_cast<void>(shade_offset);
        const auto advanced =
            _index + 1u < 4u;
        _index =
            select(
                _index,
                _index + 1u,
                advanced);
        return advanced;
    }
};

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
        return 0.5f;
    }

    Float transmittance_shade_offset(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.5f;
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
                    0.1f + 0.2f * distance,
                    0.3f + 0.05f * distance),
            .sigma_s = make_float3(0.0f),
            .emission = make_float3(0.0f),
            .has_extinction = true,
            .has_scatter = false,
            .has_emission = false};
    }
};

class ZeroCollisionProvider final
    : public HeterogeneousVolumeCollisionProvider {

  public:
    VolumeCoefficients evaluate(
        Float distance,
        Bool evaluate_emission,
        VolumePhaseSet *phases)
        const noexcept override {
        static_cast<void>(distance);
        static_cast<void>(evaluate_emission);
        static_cast<void>(phases);
        return VolumeCoefficients::zero();
    }
};

[[nodiscard]] UInt flags(
    const HeterogeneousVolumeShadowResult
        &result) noexcept {
    return select(
               0u,
               1u,
               result.traversal_exhausted) |
           select(
               0u,
               2u,
               result.lookup_complete) |
           select(
               0u,
               4u,
               result.throughput_terminated);
}

void run_backend(
    std::string_view backend,
    const char *program) {
    Context context{program};
    auto device =
        context.create_device(
            std::string{backend});
    auto stream = device.create_stream();
    auto float_output =
        device.create_buffer<luisa::float4>(4u);
    auto uint_output =
        device.create_buffer<luisa::uint4>(3u);
    const auto shadow =
        make_heterogeneous_volume_shadow_component();

    Kernel1D evaluate =
        [shadow = shadow.get()](
            BufferFloat4 float_records,
            BufferUInt4 uint_records) noexcept {
            FixtureRandomSource random;
            LinearCollisionProvider
                collisions;
            ZeroCollisionProvider
                zero_collisions;
            GroupingSegmentSequence
                complete_sequence;
            const auto complete =
                shadow->emit(
                    complete_sequence,
                    random,
                    collisions,
                    make_float3(
                        1.0f,
                        2.0f,
                        3.0f),
                    100u);
            float_records.write(
                0u,
                make_float4(
                    complete.throughput,
                    cast<float>(
                        complete
                            .next_tracking_rng_offset)));
            float_records.write(
                1u,
                make_float4(
                    cast<float>(
                        complete.groups),
                    cast<float>(
                        complete.source_segments),
                    cast<float>(
                        complete
                            .closure_evaluations),
                    cast<float>(
                        flags(complete))));
            uint_records.write(
                0u,
                make_uint4(
                    complete.groups,
                    complete.source_segments,
                    complete
                        .closure_evaluations,
                    complete
                        .next_tracking_rng_offset));

            GroupingSegmentSequence
                terminated_sequence;
            const auto terminated =
                shadow->emit(
                    terminated_sequence,
                    random,
                    collisions,
                    make_float3(1.0e-7f),
                    100u);
            float_records.write(
                2u,
                make_float4(
                    terminated.throughput,
                    cast<float>(
                        terminated
                            .next_tracking_rng_offset)));
            uint_records.write(
                1u,
                make_uint4(
                    terminated.groups,
                    terminated.source_segments,
                    terminated
                        .closure_evaluations,
                    flags(terminated)));

            // Shadow-invisible media remain in Cycles' copied shadow stack
            // and majorant hierarchy. Their closure evaluations produce zero
            // coefficients, but grouping and RNG advancement are unchanged.
            GroupingSegmentSequence
                invisible_sequence;
            const auto invisible =
                shadow->emit(
                    invisible_sequence,
                    random,
                    zero_collisions,
                    make_float3(1.0f),
                    100u);
            float_records.write(
                3u,
                make_float4(
                    invisible.throughput,
                    cast<float>(
                        invisible
                            .next_tracking_rng_offset)));
            uint_records.write(
                2u,
                make_uint4(
                    invisible.groups,
                    invisible.source_segments,
                    invisible
                        .closure_evaluations,
                    flags(invisible)));
        };
    auto shader =
        device.compile(
            evaluate,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = false});

    std::array<luisa::float4, 4u>
        actual_float{};
    std::array<luisa::uint4, 3u>
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

    // Cycles groups [0,1] and [1,2], then evaluates [2,3] and [3,4]
    // independently. Their biased k values are 2, 2, and 1. The resulting
    // optical depths are (1.6, 2.0, 1.6).
    constexpr std::array<luisa::float4, 4u>
        expected_float{{
            {0.201896518f,
             0.270670566f,
             0.605689555f,
             164.0f},
            {3.0f, 4.0f, 5.0f, 3.0f},
            {5.48811636e-8f,
             5.48811636e-8f,
             4.96585333e-8f,
             116.0f},
            {1.0f, 1.0f, 1.0f, 164.0f},
        }};
    for (std::size_t index = 0u;
         index < expected_float.size();
         ++index) {
        expect(
            close(
                actual_float[index],
                expected_float[index]),
            "Cycles heterogeneous shadow float state " +
                std::to_string(index) +
                " changed on " +
                std::string{backend});
    }

    constexpr std::array<luisa::uint4, 3u>
        expected_uint{{
            {3u, 4u, 5u, 164u},
            {1u, 2u, 2u, 6u},
            {3u, 4u, 5u, 3u},
        }};
    for (std::size_t index = 0u;
         index < expected_uint.size();
         ++index) {
        expect(
            equal(
                actual_uint[index],
                expected_uint[index]),
            "Cycles heterogeneous shadow integer state " +
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
            << "All current-Cycles heterogeneous shadow "
               "fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Heterogeneous volume shadow fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
