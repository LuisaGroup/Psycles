#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/heterogeneous_volume_candidate.h>

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

class FixtureSegmentSequence final
    : public VolumeMajorantSegmentSequence {

  private:
    const BufferFloat4 &_segments;
    const BufferFloat &_shade_offsets;
    std::uint32_t _count;
    UInt _index;
    UInt _shade_count;

  public:
    FixtureSegmentSequence(
        const BufferFloat4 &segments,
        const BufferFloat &shade_offsets,
        std::uint32_t count,
        UInt first = 0u) noexcept
        : _segments{segments},
          _shade_offsets{shade_offsets},
          _count{count},
          _index{first},
          _shade_count{0u} {}

    VolumeMajorantSegment
    current() const noexcept override {
        const auto safe_index =
            min(
                _index,
                static_cast<std::uint32_t>(
                    _count - 1u));
        const auto values =
            _segments.read(safe_index);
        const auto valid =
            _index < _count;
        return {
            .minimum = values.x,
            .maximum = values.y,
            .sigma_minimum = values.z,
            .sigma_maximum = values.w,
            .object = 10u + _index,
            .shader = 20u + _index,
            .node = _index,
            .valid = valid,
            .no_overlap = true,
            .lookup_complete = true};
    }

    Bool advance(
        Float shade_offset) noexcept override {
        const auto advanced =
            _index + 1u < _count;
        $if(advanced) {
            _shade_offsets.write(
                _shade_count,
                shade_offset);
            _shade_count += 1u;
            _index += 1u;
        };
        return advanced;
    }

    [[nodiscard]] UInt shade_count()
        const noexcept {
        return _shade_count;
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
        auto result = def(0.99f);
        result = select(
            result,
            0.8f,
            rng_offset ==
                _initial_offset);
        result = select(
            result,
            0.98f,
            rng_offset ==
                _initial_offset +
                    heterogeneous_volume_tracking_rng_stride);
        return result;
    }

    Float shade_offset(
        UInt rng_offset)
        const noexcept override {
        return select(
            0.75f,
            0.25f,
            rng_offset ==
                _initial_offset);
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

class ZeroRandomSource final
    : public HeterogeneousVolumeTrackingRandomSource {

  public:
    Float scatter_distance(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.0f;
    }

    Float shade_offset(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.0f;
    }

    Float expansion_order(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.0f;
    }

    Float transmittance_shade_offset(
        UInt rng_offset)
        const noexcept override {
        static_cast<void>(rng_offset);
        return 0.0f;
    }
};

[[nodiscard]] UInt flags(
    const HeterogeneousVolumeCandidate
        &candidate) noexcept {
    return select(
               0u, 1u, candidate.candidate) |
           select(
               0u,
               2u,
               candidate.traversal_exhausted) |
           select(
               0u,
               4u,
               candidate.step_limit_exceeded) |
           select(
               0u,
               8u,
               candidate.segment.valid) |
           select(
               0u,
               16u,
               candidate.segment
                   .lookup_complete) |
           select(
               0u,
               32u,
               candidate.segment.no_overlap);
}

void run_backend(
    std::string_view backend,
    const char *program) {
    constexpr std::array<luisa::float4, 4u>
        host_segments{{
            {0.0f, 1.0f, 0.0f, 0.0f},
            {1.0f, 2.0f, 0.25f, 1.0f},
            {2.0f, 4.0f, 0.5f, 2.0f},
            {4.0f, 8.0f, 0.125f, 0.5f},
        }};

    Context context{program};
    auto device =
        context.create_device(backend);
    auto stream =
        device.create_stream();
    auto segments =
        device.create_buffer<luisa::float4>(
            host_segments.size());
    auto shade_offsets =
        device.create_buffer<float>(3u);
    auto float_output =
        device.create_buffer<luisa::float4>(6u);
    auto uint_output =
        device.create_buffer<luisa::uint4>(5u);

    Kernel1D evaluate =
        [](BufferFloat4 segment_buffer,
           BufferFloat shade_output,
           BufferFloat4 float_records,
           BufferUInt4 uint_records) noexcept {
            const auto initial_offset =
                cycles_sampler::
                    scramble_path_offset(
                        path_rng_offset,
                        tracking_seed);
            FixtureSegmentSequence sequence{
                segment_buffer,
                shade_output,
                4u};
            FixtureRandomSource random{
                initial_offset};
            HeterogeneousVolumeCandidateWalk
                walk{
                    sequence,
                    random,
                    1.0f,
                    0.0f,
                    initial_offset};

            const auto first =
                walk.advance();
            const auto second =
                walk.advance();
            const auto exhausted =
                walk.advance();
            float_records.write(
                0u,
                make_float4(
                    first.distance,
                    first.step_distance,
                    first.sampled_majorant,
                    first.scatter_random));
            float_records.write(
                1u,
                make_float4(
                    first.optical_depth,
                    first.segment.minimum,
                    first.segment.maximum,
                    first.segment
                        .sigma_maximum));
            float_records.write(
                2u,
                make_float4(
                    second.distance,
                    second.step_distance,
                    second.sampled_majorant,
                    second.scatter_random));
            float_records.write(
                3u,
                make_float4(
                    second.optical_depth,
                    second.segment.minimum,
                    second.segment.maximum,
                    second.segment
                        .sigma_maximum));
            float_records.write(
                4u,
                make_float4(
                    exhausted.distance,
                    exhausted.step_distance,
                    exhausted.sampled_majorant,
                    exhausted.scatter_random));
            float_records.write(
                5u,
                make_float4(
                    exhausted.optical_depth,
                    cast<float>(
                        sequence
                            .shade_count()),
                    0.0f,
                    0.0f));
            uint_records.write(
                0u,
                make_uint4(
                    first.sample_rng_offset,
                    first.next_rng_offset,
                    first.step,
                    flags(first)));
            uint_records.write(
                1u,
                make_uint4(
                    second.sample_rng_offset,
                    second.next_rng_offset,
                    second.step,
                    flags(second)));
            uint_records.write(
                2u,
                make_uint4(
                    exhausted
                        .sample_rng_offset,
                    exhausted
                        .next_rng_offset,
                    exhausted.step,
                    flags(exhausted)));

            FixtureSegmentSequence
                maximum_sequence{
                    segment_buffer,
                    shade_output,
                    4u,
                    2u};
            ZeroRandomSource zero_random;
            HeterogeneousVolumeCandidateWalk
                maximum_walk{
                    maximum_sequence,
                    zero_random,
                    1.0f,
                    2.0f,
                    0u};
            UInt candidate_count = 0u;
            UInt final_step = 0u;
            UInt final_offset = 0u;
            Bool limit_exceeded = false;
            Bool continue_walk = true;
            $while(continue_walk) {
                const auto next =
                    maximum_walk.advance();
                candidate_count +=
                    select(
                        0u,
                        1u,
                        next.candidate);
                final_step = next.step;
                final_offset =
                    next.next_rng_offset;
                limit_exceeded =
                    next.step_limit_exceeded;
                continue_walk =
                    next.candidate;
            };
            uint_records.write(
                3u,
                make_uint4(
                    candidate_count,
                    final_step,
                    final_offset,
                    select(
                        0u,
                        1u,
                        limit_exceeded)));
            uint_records.write(
                4u,
                make_uint4(
                    initial_offset,
                    heterogeneous_volume_tracking_rng_stride,
                    heterogeneous_volume_maximum_steps,
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
    std::array<luisa::uint4, 5u>
        actual_uint{};
    std::array<float, 3u>
        actual_shade_offsets{};
    stream
        << segments.copy_from(
               luisa::span{host_segments})
        << shader(
               segments,
               shade_offsets,
               float_output,
               uint_output)
               .dispatch(1u)
        << float_output.copy_to(
               luisa::span{actual_float})
        << uint_output.copy_to(
               luisa::span{actual_uint})
        << shade_offsets.copy_to(
               luisa::span{
                   actual_shade_offsets})
        << synchronize();

    // Pinned to volume_integrate_advance() in Blender/Cycles main
    // 0ae970969e3f. This is a device state-transition oracle, not a Psycles
    // CPU renderer or alternate transport implementation.
    constexpr std::array<luisa::float4, 6u>
        expected_float{{
            {2.30471896f,
             0.304718956f,
             2.0f,
             0.456343634f},
            {5.0f, 2.0f, 4.0f, 2.0f},
            {5.04292184f,
             1.04292184f,
             0.5f,
             0.406347364f},
            {7.0f, 4.0f, 8.0f, 0.5f},
            {14.2532622f,
             9.21034037f,
             0.5f,
             0.99f},
            {7.0f, 3.0f, 0.0f, 0.0f},
        }};
    for (std::size_t index = 0u;
         index < expected_float.size();
         ++index) {
        expect(
            close(
                actual_float[index],
                expected_float[index]),
            "Cycles candidate float state " +
                std::to_string(index) +
                " changed on " +
                std::string{backend});
    }

    constexpr auto initial_offset =
        std::uint32_t{0x15982c28u};
    const std::array<luisa::uint4, 5u>
        expected_uint{{
            {initial_offset,
             initial_offset + 16u,
             1u,
             57u},
            {initial_offset + 16u,
             initial_offset + 32u,
             2u,
             57u},
            {initial_offset + 32u,
             initial_offset + 32u,
             3u,
             2u},
            {1025u,
             1026u,
             1025u * 16u,
             1u},
            {initial_offset,
             16u,
             1024u,
             0u},
        }};
    for (std::size_t index = 0u;
         index < expected_uint.size();
         ++index) {
        expect(
            equal(
                actual_uint[index],
                expected_uint[index]),
            "Cycles candidate integer state " +
                std::to_string(index) +
                " changed on " +
                std::string{backend});
    }
    constexpr std::array<float, 3u>
        expected_shade_offsets{{
            0.25f, 0.25f, 0.75f}};
    for (std::size_t index = 0u;
         index <
         expected_shade_offsets.size();
         ++index) {
        expect(
            close(
                actual_shade_offsets[index],
                expected_shade_offsets[index]),
            "Cycles candidate shade-offset handoff " +
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
            << "All current-Cycles heterogeneous candidate "
               "fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Heterogeneous volume candidate fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
