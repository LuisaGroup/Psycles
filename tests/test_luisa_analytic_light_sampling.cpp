#include <psycles/luisa/analytic_light_sampling.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace analytic_light =
    psycles::luisa_backend::analytic_light_sampling;

struct PointCase {
    float distance_squared;
    float light_cosine;
    float radius;
    bool normalize_power;
    bool use_mis;
};

constexpr std::array point_cases{
    // The first record is the Cycles point-path oracle probe.
    PointCase{2.09093976020813f, 1.0f, 0.0f, true, true},
    PointCase{4.0f, 1.0f, 0.0f, false, true},
    PointCase{9.0f, 0.8f, 0.25f, true, true},
    PointCase{1.5f, 0.6f, 0.5f, false, false},
};

[[nodiscard]] bool approximately_equal(
    float actual,
    double expected,
    double relative_tolerance = 2.0e-6) noexcept {
    const auto error =
        std::abs(static_cast<double>(actual) - expected);
    return error <=
           1.0e-7 +
               relative_tolerance *
                   std::max(std::abs(expected),
                            std::abs(static_cast<double>(actual)));
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto input =
        device.create_buffer<luisa::float4>(
            point_cases.size());
    auto flags =
        device.create_buffer<luisa::uint2>(
            point_cases.size());
    auto output =
        device.create_buffer<luisa::float4>(
            point_cases.size());

    std::array<luisa::float4, point_cases.size()> input_data{};
    std::array<luisa::uint2, point_cases.size()> flag_data{};
    for (std::size_t index = 0u;
         index < point_cases.size();
         ++index) {
        const auto &value = point_cases[index];
        input_data[index] = {
            value.distance_squared,
            value.light_cosine,
            value.radius,
            0.0f};
        flag_data[index] = {
            value.normalize_power ? 1u : 0u,
            value.use_mis ? 1u : 0u};
    }

    Kernel1D evaluate = [](
                            BufferFloat4 cases,
                            BufferUInt2 case_flags,
                            BufferFloat4 results) noexcept {
        const auto index = dispatch_x();
        const auto value = cases.read(index);
        const auto flags_value = case_flags.read(index);
        const auto normalize_power =
            flags_value.x != 0u;
        const auto use_mis =
            flags_value.y != 0u;
        const auto pdf =
            analytic_light::point_disk_pdf(
                value.x,
                value.y,
                value.z);
        const auto eval_factor =
            analytic_light::point_eval_factor(
                value.z,
                normalize_power);
        const auto has_competing_technique =
            analytic_light::
                point_has_competing_bsdf_technique(
                    value.z,
                    use_mis);
        results.write(
            index,
            make_float4(
                pdf,
                eval_factor,
                eval_factor / pdf,
                select(
                    0.0f,
                    1.0f,
                    has_competing_technique)));
    };

    auto shader = device.compile(evaluate);
    std::array<luisa::float4, point_cases.size()> results{};
    stream << input.copy_from(luisa::span{input_data})
           << flags.copy_from(luisa::span{flag_data})
           << shader(input, flags, output)
                  .dispatch(
                      static_cast<std::uint32_t>(
                          point_cases.size()))
           << output.copy_to(luisa::span{results})
           << synchronize();

    for (std::size_t index = 0u;
         index < point_cases.size();
         ++index) {
        const auto &value = point_cases[index];
        const auto radius_squared =
            static_cast<double>(value.radius) *
            static_cast<double>(value.radius);
        const auto inverse_area =
            radius_squared > 0.0
                ? 1.0 /
                      (radius_squared *
                       static_cast<double>(
                           analytic_light::pi))
                : 1.0;
        const auto expected_pdf =
            inverse_area *
            static_cast<double>(
                value.distance_squared) /
            static_cast<double>(value.light_cosine);
        const auto area =
            radius_squared > 0.0
                ? 4.0 *
                      static_cast<double>(
                          analytic_light::pi) *
                      radius_squared
                : 4.0;
        const auto expected_eval =
            (value.normalize_power
                 ? 1.0 / area
                 : 1.0) *
            static_cast<double>(
                analytic_light::inverse_pi);
        const auto expected_competing =
            value.use_mis && value.radius > 0.0f;
        if (!approximately_equal(
                results[index].x,
                expected_pdf) ||
            !approximately_equal(
                results[index].y,
                expected_eval) ||
            !approximately_equal(
                results[index].z,
                expected_eval / expected_pdf) ||
            (results[index].w > 0.5f) !=
                expected_competing) {
            std::cerr
                << "point-light sampling invariant failed on "
                << backend << " for case " << index
                << ": got {" << results[index].x
                << ", " << results[index].y
                << ", " << results[index].z
                << ", " << results[index].w
                << "}, expected {" << expected_pdf
                << ", " << expected_eval
                << ", "
                << expected_eval / expected_pdf
                << ", " << expected_competing
                << "}\n";
            return EXIT_FAILURE;
        }
    }

    // Lock the exact semantic values exposed by the Cycles oracle: distance
    // squared is the conditional PDF and a normalized delta point uses
    // 1 / (4 pi). The use-MIS property does not create a competing BSDF
    // technique for a measure-zero emitter.
    if (!approximately_equal(
            results[0u].x,
            2.09093976020813) ||
        !approximately_equal(
            results[0u].y,
            0.07957746833562851) ||
        results[0u].w != 0.0f) {
        std::cerr
            << "Cycles delta-point oracle regression failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
