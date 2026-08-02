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

class FixtureLightProvider final
    : public VolumeDirectLightProvider {

  private:
    UInt *_phase_trace;
    Bool *_deferred_receiving;

  public:
    FixtureLightProvider(
        UInt *phase_trace,
        Bool *deferred_receiving) noexcept
        : _phase_trace{phase_trace},
          _deferred_receiving{
              deferred_receiving} {}

    VolumeDirectDirectionSample sample_direction(
        Float distance)
        const noexcept override {
        *_phase_trace =
            *_phase_trace * 10u + 1u;
        return {
            .direction =
                normalize(
                    make_float3(
                        0.3f,
                        0.4f,
                        0.8660254f)),
            .valid = distance > 0.0f};
    }

    void evaluate_constant_emission()
        const noexcept override {
        *_phase_trace =
            *_phase_trace * 10u + 2u;
    }

    void evaluate_deferred_emission(
        Bool receiving_nonzero)
        const noexcept override {
        *_phase_trace =
            *_phase_trace * 10u + 3u;
        *_deferred_receiving =
            receiving_nonzero;
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
        device.create_buffer<luisa::float4>(26u);
    auto uint_output =
        device.create_buffer<luisa::uint4>(3u);

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

            const HeterogeneousVolumeScatterProbability
                guiding_probability;
            const auto guiding =
                guiding_probability.evaluate(
                    {.scattered_radiance =
                         make_float3(0.0f),
                     .transmitted_radiance =
                         make_float3(0.0f),
                     .majorant_optical_depth =
                         1.0f,
                     .enabled = true});
            UInt direct_phase_trace = 0u;
            Bool deferred_receiving = false;
            FixtureLightProvider
                direction{
                    &direct_phase_trace,
                    &deferred_receiving};
            SingleSegmentSequence
                guided_sequence;
            FixtureRandomSource
                guided_random{
                    initial_offset};
            UInt guided_calls = 0u;
            FixtureCollisionProvider
                guided_collisions{
                    &guided_calls};
            const auto guided =
                segment->emit(
                    {.segments =
                         guided_sequence,
                     .random = guided_random,
                     .collisions =
                         guided_collisions,
                     .guiding = guiding,
                     .direct =
                         {.requested_method =
                              volume_sample_distance,
                          .light_position =
                              make_float3(
                                  1.0f,
                                  0.0f,
                                  2.0f),
                          .interval =
                              {.minimum = 0.0f,
                               .maximum = 5.0f},
                          .enabled = true},
                     .direct_light =
                         &direction,
                     .ray_minimum = 0.0f,
                     .ray_maximum = 5.0f,
                     .segment_origin =
                         make_float3(0.0f),
                     .phase_axis =
                         phase_axis,
                     .throughput =
                         make_float3(1.0f),
                     .direct_random = 0.25f,
                     .reservoir_random =
                         0.1f,
                     .phase_random =
                         make_float2(
                             0.17f,
                             0.83f),
                     .tracking_rng_offset =
                         initial_offset,
                     .terminate = false});
            float_records.write(
                6u,
                make_float4(
                    guiding
                        .scatter_probability,
                    guiding.majorant_scale,
                    cast<float>(
                        guided_calls),
                    select(
                        0.0f,
                        1.0f,
                        guiding.enabled)));
            float_records.write(
                7u,
                make_float4(
                    guided.transport
                        .throughput,
                    guided.transport
                        .distance));
            float_records.write(
                8u,
                make_float4(
                    guided.transport.emission,
                    guided.transport
                        .null_transmittance));
            float_records.write(
                9u,
                make_float4(
                    guided.transport
                        .unguided_scatter_probability,
                    guided.transport
                        .guided_scatter_probability,
                    guided.transport
                        .reservoir_random,
                    cast<float>(
                        flags(guided))));
            float_records.write(
                10u,
                make_float4(
                    guided.direct_transport
                        .throughput,
                    guided.direct_transport
                        .distance));
            float_records.write(
                11u,
                make_float4(
                    guided.direct_transport
                        .distance_pdf,
                    guided.direct_transport
                        .equiangular_pdf,
                    guided.direct_transport
                        .mis_weight,
                    cast<float>(
                        guided.direct_transport
                            .sample_method)));
            float_records.write(
                12u,
                make_float4(
                    guided.direct_phase.value,
                    guided.direct_phase.pdf,
                    guided.direct_phase
                        .sample_weight,
                    select(
                        0.0f,
                        1.0f,
                        guided.direct_phase
                            .valid)));
            uint_records.write(
                2u,
                make_uint4(
                    direct_phase_trace,
                    select(
                        0u,
                        1u,
                        deferred_receiving),
                    0u,
                    0u));

            SingleSegmentSequence
                mis_sequence;
            FixtureRandomSource
                mis_random{
                    initial_offset};
            UInt mis_calls = 0u;
            FixtureCollisionProvider
                mis_collisions{
                    &mis_calls};
            const auto mis =
                segment->emit(
                    {.segments = mis_sequence,
                     .random = mis_random,
                     .collisions =
                         mis_collisions,
                     .guiding =
                         {.scatter_probability =
                              1.0f,
                          .majorant_scale = 1.0f,
                          .enabled = false},
                     .direct =
                         {.requested_method =
                              volume_sample_mis,
                          .light_position =
                              make_float3(
                                  1.0f,
                                  0.0f,
                                  2.0f),
                          .interval =
                              {.minimum = 0.0f,
                               .maximum = 5.0f},
                          .enabled = true},
                     .direct_light =
                         &direction,
                     .ray_minimum = 0.0f,
                     .ray_maximum = 5.0f,
                     .segment_origin =
                         make_float3(0.0f),
                     .phase_axis =
                         phase_axis,
                     .throughput =
                         make_float3(1.0f),
                     .direct_random = 0.625f,
                     .reservoir_random =
                         0.1f,
                     .phase_random =
                         make_float2(
                             0.17f,
                             0.83f),
                     .tracking_rng_offset =
                         initial_offset,
                     .terminate = false});
            float_records.write(
                13u,
                make_float4(
                    mis.transport.throughput,
                    mis.transport.distance));
            float_records.write(
                14u,
                make_float4(
                    mis.direct_transport
                        .throughput,
                    mis.direct_transport
                        .distance));
            float_records.write(
                15u,
                make_float4(
                    mis.direct_transport
                        .distance_pdf,
                    mis.direct_transport
                        .equiangular_pdf,
                    mis.direct_transport
                        .mis_weight,
                    cast<float>(
                        mis.direct_transport
                            .sample_method)));
            float_records.write(
                16u,
                make_float4(
                    mis.direct_phase.value,
                    mis.direct_phase.pdf,
                    mis.direct_phase
                        .sample_weight,
                    select(
                        0.0f,
                        1.0f,
                        mis.direct_phase.valid)));
            float_records.write(
                17u,
                make_float4(
                    mis.transport
                        .unguided_scatter_probability,
                    mis.transport
                        .guided_scatter_probability,
                    mis.transport
                        .reservoir_random,
                    cast<float>(
                        flags(mis))));

            SingleSegmentSequence
                distance_mis_sequence;
            FixtureRandomSource
                distance_mis_random{
                    initial_offset};
            UInt distance_mis_calls = 0u;
            FixtureCollisionProvider
                distance_mis_collisions{
                    &distance_mis_calls};
            const auto distance_mis =
                segment->emit(
                    {.segments =
                         distance_mis_sequence,
                     .random =
                         distance_mis_random,
                     .collisions =
                         distance_mis_collisions,
                     .guiding =
                         {.scatter_probability =
                              1.0f,
                          .majorant_scale = 1.0f,
                          .enabled = false},
                     .direct =
                         {.requested_method =
                              volume_sample_mis,
                          .light_position =
                              make_float3(
                                  1.0f,
                                  0.0f,
                                  2.0f),
                          .interval =
                              {.minimum = 0.0f,
                               .maximum = 5.0f},
                          .enabled = true},
                     .direct_light =
                         &direction,
                     .ray_minimum = 0.0f,
                     .ray_maximum = 5.0f,
                     .segment_origin =
                         make_float3(0.0f),
                     .phase_axis =
                         phase_axis,
                     .throughput =
                         make_float3(1.0f),
                     .direct_random = 0.25f,
                     .reservoir_random =
                         0.1f,
                     .phase_random =
                         make_float2(
                             0.17f,
                             0.83f),
                     .tracking_rng_offset =
                         initial_offset,
                     .terminate = false});
            float_records.write(
                18u,
                make_float4(
                    distance_mis.transport
                        .throughput,
                    distance_mis.transport
                        .distance));
            float_records.write(
                19u,
                make_float4(
                    distance_mis
                        .direct_transport
                        .throughput,
                    distance_mis
                        .direct_transport
                        .distance));
            float_records.write(
                20u,
                make_float4(
                    distance_mis
                        .direct_transport
                        .distance_pdf,
                    distance_mis
                        .direct_transport
                        .equiangular_pdf,
                    distance_mis
                        .direct_transport
                        .mis_weight,
                    cast<float>(
                        distance_mis
                            .direct_transport
                            .sample_method)));
            float_records.write(
                21u,
                make_float4(
                    distance_mis.direct_phase
                        .value,
                    distance_mis.direct_phase
                        .pdf,
                    cast<float>(
                        distance_mis_calls),
                    cast<float>(
                        flags(
                            distance_mis))));

            SingleSegmentSequence
                guided_scatter_sequence;
            FixtureRandomSource
                guided_scatter_random{
                    initial_offset};
            UInt guided_scatter_calls = 0u;
            FixtureCollisionProvider
                guided_scatter_collisions{
                    &guided_scatter_calls};
            const auto guided_scatter =
                segment->emit(
                    {.segments =
                         guided_scatter_sequence,
                     .random =
                         guided_scatter_random,
                     .collisions =
                         guided_scatter_collisions,
                     .guiding = guiding,
                     .direct =
                         {.requested_method =
                              volume_sample_none,
                          .light_position =
                              make_float3(0.0f),
                          .interval =
                              {.minimum = 0.0f,
                               .maximum = 5.0f},
                          .enabled = false},
                     .direct_light =
                         nullptr,
                     .ray_minimum = 0.0f,
                     .ray_maximum = 5.0f,
                     .segment_origin =
                         make_float3(0.0f),
                     .phase_axis =
                         phase_axis,
                     .throughput =
                         make_float3(1.0f),
                     .direct_random = 0.0f,
                     .reservoir_random =
                         0.9f,
                     .phase_random =
                         make_float2(
                             0.17f,
                             0.83f),
                     .tracking_rng_offset =
                         initial_offset,
                     .terminate = false});
            float_records.write(
                22u,
                make_float4(
                    guided_scatter.transport
                        .throughput,
                    guided_scatter.transport
                        .distance));
            float_records.write(
                23u,
                make_float4(
                    guided_scatter.transport
                        .emission,
                    guided_scatter.transport
                        .null_transmittance));
            float_records.write(
                24u,
                make_float4(
                    guided_scatter.transport
                        .unguided_scatter_probability,
                    guided_scatter.transport
                        .guided_scatter_probability,
                    guided_scatter.transport
                        .reservoir_random,
                    cast<float>(
                        guided_scatter_calls)));
            float_records.write(
                25u,
                make_float4(
                    guided_scatter.phase
                        .direction,
                    cast<float>(
                        flags(
                            guided_scatter))));
        };
    auto shader =
        device.compile(
            evaluate,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = false});

    std::array<luisa::float4, 26u>
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

    // Records 0..5 pin weighted-delta tracking. Records 6..12 pin empty-
    // history VSPG, its defensive transmit selection, and non-MIS distance
    // NEE. Records 13..17 and 18..21 pin both equiangular and distance arms
    // of Cycles' heterogeneous two-technique MIS. Records 22..25 reuse the
    // same VSPG masses but force the defensive scatter selection. All
    // equations and random remappings correspond to official Cycles main
    // 6f7add4a791e; phase values reuse the separately exhaustive Cycles phase
    // fixtures.
    constexpr std::array<luisa::float4, 26u>
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
            {0.790988386f,
             1.0f,
             3.0f,
             1.0f},
            {1.40298247f,
             0.467660815f,
             0.0f,
             5.0f},
            {0.125f,
             0.125f,
             0.200000003f,
             0.442116022f},
            {0.557883978f,
             0.732712269f,
             0.374128670f,
             18.0f},
            {0.150000006f,
             0.199999988f,
             0.150000006f,
             1.25f},
            {1.11576796f,
             0.248494565f,
             1.0f,
             1.0f},
            {0.161441714f,
             0.161441714f,
             0.800000012f,
             1.0f},
            {0.268873125f,
             0.358497471f,
             0.268873125f,
             1.25f},
            {0.438155144f,
             0.292103410f,
             0.0f,
             1.52656865f},
            {0.342331618f,
             0.339373589f,
             0.991321862f,
             2.0f},
            {0.161441714f,
             0.161441714f,
             0.800000012f,
             1.0f},
            {0.557883978f,
             0.557883978f,
             0.179248735f,
             51.0f},
            {0.268873125f,
             0.358497471f,
             0.268873125f,
             1.25f},
            {0.512334168f,
             0.683112204f,
             0.512334168f,
             1.25f},
            {1.11576796f,
             0.248494565f,
             1.90548682f,
             1.0f},
            {0.161441714f,
             0.161441714f,
             4.0f,
             49.0f},
            {0.204718843f,
             0.272958428f,
             0.204718843f,
             1.25f},
            {0.125f,
             0.125f,
             0.200000003f,
             0.442116022f},
            {0.557883978f,
             0.732712269f,
             0.863520741f,
             3.0f},
            {-0.117953509f,
             -0.899912238f,
             -0.419815451f,
             51.0f},
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
    const std::array<luisa::uint4, 3u>
        expected_uint{{
            {3u,
             initial_offset + 32u,
             3u,
             51u},
            {initial_offset,
             path_rng_offset,
             tracking_seed,
             0u},
            {123u,
             1u,
             0u,
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
