#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;

struct FmodCase {
    float dividend;
    float multiplier;
    float divisor;
    const char *name;
};

constexpr std::array cases{
    FmodCase{5.2f, 1.0f, 2.0f, "ordinary-positive"},
    FmodCase{-5.2f, 1.0f, 2.0f, "ordinary-negative"},
    // Cycles' Magic Texture reduces scaled coordinates modulo two pi. The
    // quotient here is far wider than float32's significand, so expanding
    // fmod as x - y * trunc(x / y) cannot recover the remainder.
    FmodCase{
        1.0e20f,
        1.0e-3f,
        6.28318530717958647692f,
        "cycles-magic-large-coordinate"},
    FmodCase{
        -1.0e20f,
        1.0e-3f,
        6.28318530717958647692f,
        "cycles-magic-large-negative-coordinate"},
    FmodCase{
        std::numeric_limits<float>::max(),
        1.0f,
        3.14159265358979323846f,
        "maximum-finite"},
    FmodCase{
        std::numeric_limits<float>::denorm_min() * 17.0f,
        1.0f,
        std::numeric_limits<float>::denorm_min() * 3.0f,
        "subnormal-significands"},
    FmodCase{0.75f, 1.0f, 2.0f, "magnitude-less-than-divisor"},
    FmodCase{6.0f, 1.0f, 3.0f, "positive-signed-zero"},
    FmodCase{-6.0f, 1.0f, 3.0f, "negative-signed-zero"},
    FmodCase{-0.0f, 1.0f, 3.0f, "negative-zero-input"},
    FmodCase{1.0f, 1.0f, std::numeric_limits<float>::infinity(),
             "infinite-divisor"},
    FmodCase{1.0f, 1.0f, 0.0f, "zero-divisor"},
    FmodCase{std::numeric_limits<float>::infinity(), 1.0f, 1.0f,
             "infinite-dividend"}};

constexpr std::uint32_t subnormal_case_index = 5u;
constexpr std::size_t vector_result_width = 12u;
constexpr std::size_t result_count = cases.size() + vector_result_width;

[[nodiscard]] auto bits(float value) noexcept {
    return std::bit_cast<std::uint32_t>(value);
}

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    std::array<float, cases.size() * 3u> inputs{};
    for (std::size_t index = 0u; index < cases.size(); ++index) {
        inputs[index * 3u] = cases[index].dividend;
        inputs[index * 3u + 1u] = cases[index].multiplier;
        inputs[index * 3u + 2u] = cases[index].divisor;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto input = device.create_buffer<float>(inputs.size());
    auto output = device.create_buffer<float>(result_count);
    Kernel1D evaluate = [](
                            BufferFloat input_values,
                            BufferFloat output_values) noexcept {
        const auto index = dispatch_x();
        const auto dividend = input_values.read(index * 3u);
        const auto multiplier = input_values.read(index * 3u + 1u);
        const auto divisor = input_values.read(index * 3u + 2u);
        Float evaluated_dividend = dividend;
        // Keep this case free of a preceding floating-point operation: GPU
        // execution modes may flush subnormal arithmetic independently of
        // fmod, while the primitive itself must preserve the input bits.
        $if(index != subnormal_case_index) {
            evaluated_dividend = dividend * multiplier;
        };
        output_values.write(
            index,
            fmod(evaluated_dividend, divisor));
        $if(index == 0u) {
            auto scaled = [&](std::uint32_t case_index) noexcept {
                return input_values.read(case_index * 3u) *
                       input_values.read(case_index * 3u + 1u);
            };
            const auto vector_dividends = make_float4(
                scaled(0u), scaled(1u), scaled(2u), scaled(3u));
            const auto vector_divisors = make_float4(
                input_values.read(2u), input_values.read(5u),
                input_values.read(8u), input_values.read(11u));
            constexpr auto base =
                static_cast<std::uint32_t>(cases.size());
            const auto vector_vector =
                fmod(vector_dividends, vector_divisors);
            output_values.write(base, vector_vector.x);
            output_values.write(base + 1u, vector_vector.y);
            output_values.write(base + 2u, vector_vector.z);
            output_values.write(base + 3u, vector_vector.w);
            const auto vector_scalar =
                fmod(vector_dividends, input_values.read(2u));
            output_values.write(base + 4u, vector_scalar.x);
            output_values.write(base + 5u, vector_scalar.y);
            output_values.write(base + 6u, vector_scalar.z);
            output_values.write(base + 7u, vector_scalar.w);
            const auto scalar_vector =
                fmod(scaled(2u), vector_divisors);
            output_values.write(base + 8u, scalar_vector.x);
            output_values.write(base + 9u, scalar_vector.y);
            output_values.write(base + 10u, scalar_vector.z);
            output_values.write(base + 11u, scalar_vector.w);
        };
    };
    auto shader = device.compile(
        evaluate,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = false});

    std::array<float, result_count> actual{};
    stream << input.copy_from(luisa::span{inputs})
           << shader(input, output).dispatch(cases.size())
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (std::size_t index = 0u; index < cases.size(); ++index) {
        const auto evaluated_dividend =
            index == subnormal_case_index ?
                cases[index].dividend :
                cases[index].dividend * cases[index].multiplier;
        const auto expected = std::fmod(
            evaluated_dividend, cases[index].divisor);
        const auto matches =
            std::isnan(expected) ? std::isnan(actual[index]) :
                                   bits(actual[index]) == bits(expected);
        if (!matches) {
            std::cerr << "strict fmod failed on " << backend << " for "
                      << cases[index].name << ": got "
                      << std::setprecision(9) << actual[index] << " (0x"
                      << std::hex << bits(actual[index]) << "), expected "
                      << std::dec << expected << " (0x" << std::hex
                      << bits(expected) << ")\n";
            return EXIT_FAILURE;
        }
    }
    auto check_vector_lane = [&](std::size_t lane,
                                 float expected,
                                 std::string_view route) noexcept {
        const auto index = cases.size() + lane;
        const auto matches =
            std::isnan(expected) ? std::isnan(actual[index]) :
                                   bits(actual[index]) == bits(expected);
        if (!matches) {
            std::cerr << "strict fmod failed on " << backend << " for "
                      << route << " lane " << lane % 4u << ": got "
                      << std::setprecision(9) << actual[index] << " (0x"
                      << std::hex << bits(actual[index]) << "), expected "
                      << std::dec << expected << " (0x" << std::hex
                      << bits(expected) << ")\n";
        }
        return matches;
    };
    for (std::size_t lane = 0u; lane < 4u; ++lane) {
        const auto dividend =
            cases[lane].dividend * cases[lane].multiplier;
        if (!check_vector_lane(
                lane,
                std::fmod(dividend, cases[lane].divisor),
                "vector-vector")) {
            return EXIT_FAILURE;
        }
        if (!check_vector_lane(
                4u + lane,
                std::fmod(dividend, cases[0u].divisor),
                "vector-scalar")) {
            return EXIT_FAILURE;
        }
        const auto scalar_dividend =
            cases[2u].dividend * cases[2u].multiplier;
        if (!check_vector_lane(
                8u + lane,
                std::fmod(scalar_dividend, cases[lane].divisor),
                "scalar-vector")) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
