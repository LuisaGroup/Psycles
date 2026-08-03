#include <psycles/luisa/cycles_sample_mapping.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace sample_mapping =
    psycles::luisa_backend::cycles_sample_mapping;

struct SampleCase {
    luisa::float3 normal;
    luisa::float2 random;
    luisa::float3 expected_direction;
    float expected_pdf;
};

constexpr auto inverse_sqrt_two = 0.70710678118654752440f;
constexpr auto inverse_sqrt_three = 0.57735026918962576451f;
constexpr auto sqrt_two_thirds = 0.81649658092772603273f;
constexpr auto inverse_sqrt_six = 0.40824829046386301637f;

constexpr std::array sample_cases{// Exact first-bounce diffuse sample
                                  // recorded by the Cycles path oracle.
    SampleCase{{0.0f, 0.0f, 1.0f},
        {0.820808350f, 0.676392674f},
        {0.601930976f, -0.222150967f, 0.767025411f},
        0.244151756f},
    SampleCase{{0.0f, 0.0f, 1.0f},
        {0.5f, 0.5f},
        {0.0f, 0.0f, 1.0f},
        sample_mapping::inverse_pi},
    SampleCase{{0.0f, 1.0f, 0.0f},
        {0.75f, 0.5f},
        {-0.353553391f, 0.866025404f, 0.353553391f},
        0.275664448f},
    SampleCase{
        {inverse_sqrt_three, inverse_sqrt_three, inverse_sqrt_three},
        {0.5f, 0.5f},
        {inverse_sqrt_three, inverse_sqrt_three, inverse_sqrt_three},
        sample_mapping::inverse_pi}};

[[nodiscard]] bool approximately_equal(
    float actual, float expected, float tolerance = 2.5e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(1.0f,
                   std::max(std::abs(actual), std::abs(expected)));
}

[[nodiscard]] bool approximately_equal(luisa::float3 actual,
    luisa::float3 expected,
    float tolerance = 2.5e-6f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();

    std::array<luisa::float4, sample_cases.size()> inputs{};
    for (std::size_t i = 0u; i < sample_cases.size(); ++i) {
        inputs[i] = {sample_cases[i].normal.x,
            sample_cases[i].normal.y,
            sample_cases[i].normal.z,
            0.0f};
    }
    constexpr std::array randoms{sample_cases[0u].random,
        sample_cases[1u].random,
        sample_cases[2u].random,
        sample_cases[3u].random};

    auto input_buffer =
        device.create_buffer<luisa::float4>(inputs.size());
    auto random_buffer =
        device.create_buffer<luisa::float2>(randoms.size());
    auto sample_buffer =
        device.create_buffer<luisa::float4>(inputs.size());
    auto tangent_buffer =
        device.create_buffer<luisa::float4>(inputs.size());
    auto bitangent_buffer =
        device.create_buffer<luisa::float4>(inputs.size());
    auto ggx_buffer = device.create_buffer<luisa::float4>(1u);
    auto beckmann_buffer = device.create_buffer<luisa::float4>(2u);

    Kernel1D evaluate = [](BufferFloat4 input,
                            BufferFloat2 random,
                            BufferFloat4 samples,
                            BufferFloat4 tangents,
                            BufferFloat4 bitangents,
                            BufferFloat4 ggx,
                            BufferFloat4 beckmann) noexcept {
        const auto index = dispatch_x();
        const auto normal = input.read(index).xyz();
        const auto basis = sample_mapping::make_orthonormals(normal);
        const auto sample = sample_mapping::sample_cosine_hemisphere(
            normal, random.read(index));
        samples.write(index, make_float4(sample.direction, sample.pdf));
        tangents.write(index, make_float4(basis.tangent, 0.0f));
        bitangents.write(index, make_float4(basis.bitangent, 0.0f));
        $if(index == 0u) {
            // Exact first glossy bounce recorded by the current Cycles
            // path oracle in Lone Monk. This locks the whole isotropic
            // VNDF mapping, not only its probability density.
            const auto ggx_direction =
                sample_mapping::sample_ggx_visible_normal_reflection(
                    make_float3(-0.9796296954154968f,
                        -0.1097114309668541f,
                        0.16819368302822113f),
                    make_float3(-0.18318170309066772f,
                        -0.982463538646698f,
                        0.03478366881608963f),
                    0.24967573583126068f,
                    make_float2(
                        0.04747750982642174f, 0.6334579586982727f));
            ggx.write(0u, make_float4(ggx_direction, 0.0f));

            const auto normal = make_float3(0.0f, 0.0f, 1.0f);
            const auto oblique_incoming =
                normalize(make_float3(0.9539392f, 0.0f, 0.3f));
            const auto sample_random = make_float2(0.37f, 0.73f);
            const auto oblique_half =
                sample_mapping::sample_beckmann_visible_normal(
                    normal, oblique_incoming, 0.64f, sample_random);
            const auto oblique_direction =
                2.0f * dot(oblique_incoming, oblique_half) * oblique_half -
                oblique_incoming;
            beckmann.write(0u, make_float4(oblique_direction, 0.0f));

            const auto normal_half =
                sample_mapping::sample_beckmann_visible_normal(
                    normal, normal, 0.64f, sample_random);
            const auto normal_direction =
                2.0f * dot(normal, normal_half) * normal_half - normal;
            beckmann.write(1u, make_float4(normal_direction, 0.0f));
        };
    };
    auto shader = device.compile(evaluate);

    std::array<luisa::float4, sample_cases.size()> samples{};
    std::array<luisa::float4, sample_cases.size()> tangents{};
    std::array<luisa::float4, sample_cases.size()> bitangents{};
    std::array<luisa::float4, 1u> ggx{};
    std::array<luisa::float4, 2u> beckmann{};
    stream << input_buffer.copy_from(luisa::span{inputs})
           << random_buffer.copy_from(luisa::span{randoms})
           << shader(input_buffer,
                  random_buffer,
                  sample_buffer,
                  tangent_buffer,
                  bitangent_buffer,
                  ggx_buffer,
                  beckmann_buffer)
                  .dispatch(
                      static_cast<std::uint32_t>(sample_cases.size()))
           << sample_buffer.copy_to(luisa::span{samples})
           << tangent_buffer.copy_to(luisa::span{tangents})
           << bitangent_buffer.copy_to(luisa::span{bitangents})
           << ggx_buffer.copy_to(luisa::span{ggx})
           << beckmann_buffer.copy_to(luisa::span{beckmann}) << synchronize();

    for (std::size_t i = 0u; i < sample_cases.size(); ++i) {
        const auto actual_direction = samples[i].xyz();
        if (!approximately_equal(
                actual_direction, sample_cases[i].expected_direction) ||
            !approximately_equal(
                samples[i].w, sample_cases[i].expected_pdf)) {
            std::cerr << "Cycles cosine-hemisphere mapping failed on "
                      << backend << " for case " << i << ": direction {"
                      << actual_direction.x << ", "
                      << actual_direction.y << ", "
                      << actual_direction.z << "}, pdf " << samples[i].w
                      << '\n';
            return EXIT_FAILURE;
        }
        const auto tangent = tangents[i].xyz();
        const auto bitangent = bitangents[i].xyz();
        const auto normal = sample_cases[i].normal;
        if (!approximately_equal(dot(tangent, normal), 0.0f) ||
            !approximately_equal(dot(bitangent, normal), 0.0f) ||
            !approximately_equal(dot(tangent, bitangent), 0.0f) ||
            !approximately_equal(dot(tangent, tangent), 1.0f) ||
            !approximately_equal(dot(bitangent, bitangent), 1.0f)) {
            std::cerr << "orthonormal-basis invariant failed on "
                      << backend << " for case " << i << '\n';
            return EXIT_FAILURE;
        }
    }

    // The equal-component branch is easy to lose in a generic basis
    // helper. Lock its Cycles-defined orientation, not merely its
    // orthogonality.
    constexpr auto expected_equal_tangent =
        luisa::float3{0.0f, inverse_sqrt_two, -inverse_sqrt_two};
    constexpr auto expected_equal_bitangent = luisa::float3{
        -sqrt_two_thirds, inverse_sqrt_six, inverse_sqrt_six};
    if (!approximately_equal(
            tangents[3u].xyz(), expected_equal_tangent) ||
        !approximately_equal(
            bitangents[3u].xyz(), expected_equal_bitangent)) {
        std::cerr
            << "Cycles equal-component basis orientation failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }

    constexpr auto expected_ggx_direction =
        luisa::float3{-0.3878505229949951f,
            0.9008376598358154f,
            -0.1950989067554474f};
    if (!approximately_equal(ggx[0u].xyz(), expected_ggx_direction)) {
        const auto actual = ggx[0u].xyz();
        std::cerr << "Cycles GGX VNDF mapping failed on " << backend
                  << ": direction {" << actual.x << ", " << actual.y
                  << ", " << actual.z << "}\n";
        return EXIT_FAILURE;
    }

    // Fixed current-Cycles VNDF samples cover both the oblique root solve and
    // its normal-incidence branch. The former NDF sampler maps these same
    // random numbers to different rays despite having the right marginal D.
    constexpr std::array expected_beckmann_directions{
        luisa::float3{-0.368741920f, -0.324739778f, 0.870961235f},
        luisa::float3{0.716639308f, 0.555882428f, 0.421215894f}};
    for (std::size_t index = 0u;
         index < expected_beckmann_directions.size(); ++index) {
        if (!approximately_equal(beckmann[index].xyz(),
                                 expected_beckmann_directions[index],
                                 1.0e-5f)) {
            const auto actual = beckmann[index].xyz();
            std::cerr << "Cycles Beckmann VNDF mapping failed on " << backend
                      << " for case " << index << ": direction {" << actual.x
                      << ", " << actual.y << ", " << actual.z << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
