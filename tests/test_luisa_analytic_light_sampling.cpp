#include <psycles/luisa/analytic_light_intersection.h>
#include <psycles/luisa/analytic_light_sampling.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace analytic_light =
    psycles::luisa_backend::analytic_light_sampling;
namespace analytic_intersection =
    psycles::luisa_backend::analytic_light_intersection;

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
    auto rectangle_output =
        device.create_buffer<luisa::float4>(2u);
    auto intersection_output =
        device.create_buffer<luisa::float4>(4u);
    constexpr std::size_t finite_case_count = 3u;
    auto finite_direction_output =
        device.create_buffer<luisa::float4>(
            finite_case_count);
    auto finite_position_output =
        device.create_buffer<luisa::float4>(
            finite_case_count);
    auto finite_normal_output =
        device.create_buffer<luisa::float4>(
            finite_case_count);
    auto finite_uv_output =
        device.create_buffer<luisa::float4>(
            finite_case_count);
    auto forward_position_output =
        device.create_buffer<luisa::float4>(
            finite_case_count);
    auto forward_normal_output =
        device.create_buffer<luisa::float4>(
            finite_case_count);

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
                            BufferFloat4 results,
                            BufferFloat4 rectangle_result,
                            BufferFloat4 intersection_result) noexcept {
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
        $if (index == 0u) {
            const auto rectangle =
                analytic_light::
                    sample_rectangle_solid_angle(
                        make_float3(
                            0.10316085815429688f,
                            0.034410953521728516f,
                            0.0f),
                        make_float3(
                            0.3700000047683716f,
                            -0.20999999344348907f,
                            1.399999976158142f),
                        make_float3(1.0f, 0.0f, 0.0f),
                        0.800000011920929f,
                        make_float3(0.0f, 1.0f, 0.0f),
                        0.5f,
                        make_float2(
                            0.5302490592002869f,
                            0.8410298824310303f));
            rectangle_result.write(
                0u,
                make_float4(
                    rectangle.position,
                    rectangle.pdf));
        };
        $if (index == 1u) {
            const auto rectangle =
                analytic_light::
                    sample_rectangle_solid_angle(
                        make_float3(
                            -0.6530890464782715f,
                            0.996910572052002f,
                            0.0f),
                        make_float3(
                            0.3700000047683716f,
                            -0.20999999344348907f,
                            1.399999976158142f),
                        make_float3(1.0f, 0.0f, 0.0f),
                        0.800000011920929f,
                        make_float3(0.0f, 1.0f, 0.0f),
                        0.5f,
                        make_float2(
                            0.1745041161775589f,
                            0.9358701109886169f));
            rectangle_result.write(
                1u,
                make_float4(
                    rectangle.position,
                    rectangle.pdf));
        };
        $if (index == 0u) {
            const auto intersection =
                analytic_intersection::
                    intersect_area(
                        make_float3(
                            0.30941081047058105f,
                            -0.1030890941619873f,
                            0.0f),
                        make_float3(
                            -0.006292062345892191f,
                            0.013297063298523426f,
                            0.9998918175697327f),
                        0.0f,
                        1.0e10f,
                        make_float3(
                            0.3700000047683716f,
                            -0.20999999344348907f,
                            1.399999976158142f),
                        make_float3(1.0f, 0.0f, 0.0f),
                        0.800000011920929f,
                        make_float3(0.0f, 1.0f, 0.0f),
                        0.5f,
                        make_float3(0.0f, 0.0f, 1.0f),
                        false,
                        true,
                        analytic_light::pi,
                        true);
            intersection_result.write(
                0u,
                make_float4(
                    intersection.position,
                    select(
                        -1.0f,
                        intersection.distance,
                        intersection.valid)));
            intersection_result.write(
                1u,
                make_float4(
                    intersection.conditional_pdf,
                    intersection.evaluation_factor,
                    intersection.uv));
        };
        $if (index == 2u) {
            // Cycles 5.3 collision-position oracle for the narrow-spread
            // elliptical area light used by the volume direct-light fixture.
            // The hit geometry is the authored ellipse, while its PDF is the
            // spread-clamped collision measure.
            const auto intersection =
                analytic_intersection::
                    intersect_area(
                        make_float3(
                            0.1251223087310791f,
                            -0.3748776912689209f,
                            -0.7516117095947266f),
                        make_float3(
                            0.23187683522701263f,
                            0.16386552155017853f,
                            0.9588436484336853f),
                        0.0f,
                        1.0e10f,
                        make_float3(
                            0.2f,
                            -0.1f,
                            -0.4f),
                        make_float3(1.0f, 0.0f, 0.0f),
                        1.0f,
                        make_float3(0.0f, 1.0f, 0.0f),
                        0.6f,
                        make_float3(0.0f, 0.0f, 1.0f),
                        true,
                        false,
                        1.2f,
                        true);
            intersection_result.write(
                2u,
                make_float4(
                    intersection.position,
                    select(
                        -1.0f,
                        intersection.distance,
                        intersection.valid)));
            intersection_result.write(
                3u,
                make_float4(
                    intersection.conditional_pdf,
                    intersection.evaluation_factor,
                    intersection.uv));
        };
    };

    Kernel1D evaluate_finite =
        [](BufferFloat4 direction_result,
           BufferFloat4 position_result,
           BufferFloat4 normal_result,
           BufferFloat4 uv_result,
           BufferFloat4 forward_position_result,
           BufferFloat4 forward_normal_result) noexcept {
            const auto index = dispatch_x();
            const auto reference = make_float3(
                0.10316085815429688f,
                0.034410953521728516f,
                0.0f);
            const auto center = make_float3(
                0.3700000047683716f,
                -0.20999999344348907f,
                1.399999976158142f);
            const auto shading_normal =
                make_float3(0.0f, 0.0f, 1.0f);
            const auto axis_x =
                make_float3(1.0f, 0.0f, 0.0f);
            const auto axis_y =
                make_float3(0.0f, 1.0f, 0.0f);
            const auto axis_z =
                make_float3(0.0f, 0.0f, 1.0f);
            const auto axis_scale =
                make_float3(1.0f);
            const auto random = make_float2(
                0.5302490592002869f,
                0.8410298824310303f);
            const auto spot = index == 2u;
            const auto sphere = index != 1u;
            const auto radius = select(
                0.19f,
                0.13f,
                spot);
            analytic_light::FiniteLightSample sample{
                .valid = false,
                .direction = make_float3(0.0f),
                .position = center,
                .normal = make_float3(0.0f),
                .uv = make_float2(0.0f),
                .distance = 0.0f,
                .conditional_pdf = 0.0f,
                .evaluation_factor = 0.0f};
            $if (spot) {
                sample =
                    analytic_light::sample_spot_light(
                        reference,
                        shading_normal,
                        false,
                        center,
                        radius,
                        sphere,
                        axis_x,
                        axis_y,
                        axis_z,
                        axis_scale,
                        0.9200000166893005f,
                        0.3700000047683716f,
                        random,
                        true);
            }
            $else {
                sample =
                    analytic_light::sample_point_light(
                        reference,
                        shading_normal,
                        false,
                        center,
                        radius,
                        sphere,
                        axis_x,
                        axis_y,
                        axis_z,
                        axis_scale,
                        random,
                        true);
            };
            analytic_intersection::PointIntersection
                forward{
                    .valid = false,
                    .distance = 0.0f,
                    .position = reference,
                    .normal = make_float3(0.0f),
                    .uv = make_float2(0.0f),
                    .conditional_pdf = 0.0f,
                    .evaluation_factor = 0.0f};
            $if (spot) {
                forward =
                    analytic_intersection::intersect_spot(
                        reference,
                        sample.direction,
                        0.0f,
                        1.0e10f,
                        center,
                        radius,
                        sphere,
                        axis_x,
                        axis_y,
                        axis_z,
                        axis_scale,
                        0.9200000166893005f,
                        0.3700000047683716f,
                        true,
                        shading_normal,
                        false);
            }
            $else {
                forward =
                    analytic_intersection::intersect_point(
                        reference,
                        sample.direction,
                        0.0f,
                        1.0e10f,
                        center,
                        radius,
                        sphere,
                        axis_x,
                        axis_y,
                        axis_z,
                        axis_scale,
                        true,
                        shading_normal,
                        false);
            };
            direction_result.write(
                index,
                make_float4(
                    sample.direction,
                    sample.conditional_pdf));
            position_result.write(
                index,
                make_float4(
                    sample.position,
                    sample.distance));
            normal_result.write(
                index,
                make_float4(
                    sample.normal,
                    sample.evaluation_factor));
            uv_result.write(
                index,
                make_float4(
                    sample.uv,
                    select(
                        0.0f, 1.0f, sample.valid),
                    select(
                        0.0f, 1.0f, forward.valid)));
            forward_position_result.write(
                index,
                make_float4(
                    forward.position,
                    forward.distance));
            forward_normal_result.write(
                index,
                make_float4(
                    forward.normal,
                    forward.conditional_pdf));
        };

    auto shader = device.compile(
        evaluate,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = false});
    auto finite_shader = device.compile(
        evaluate_finite,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = false});
    std::array<luisa::float4, point_cases.size()> results{};
    std::array<luisa::float4, 2u> rectangle_result{};
    std::array<luisa::float4, 4u> intersection_result{};
    std::array<luisa::float4, finite_case_count>
        finite_direction_result{};
    std::array<luisa::float4, finite_case_count>
        finite_position_result{};
    std::array<luisa::float4, finite_case_count>
        finite_normal_result{};
    std::array<luisa::float4, finite_case_count>
        finite_uv_result{};
    std::array<luisa::float4, finite_case_count>
        forward_position_result{};
    std::array<luisa::float4, finite_case_count>
        forward_normal_result{};
    stream << input.copy_from(luisa::span{input_data})
           << flags.copy_from(luisa::span{flag_data})
           << shader(
                  input,
                  flags,
                  output,
                  rectangle_output,
                  intersection_output)
                  .dispatch(
                      static_cast<std::uint32_t>(
                          point_cases.size()))
           << output.copy_to(luisa::span{results})
           << rectangle_output.copy_to(
                  luisa::span{rectangle_result})
           << intersection_output.copy_to(
                  luisa::span{intersection_result})
           << finite_shader(
                  finite_direction_output,
                  finite_position_output,
                  finite_normal_output,
                  finite_uv_output,
                  forward_position_output,
                  forward_normal_output)
                  .dispatch(
                      static_cast<std::uint32_t>(
                          finite_case_count))
           << finite_direction_output.copy_to(
                  luisa::span{
                      finite_direction_result})
           << finite_position_output.copy_to(
                  luisa::span{
                      finite_position_result})
           << finite_normal_output.copy_to(
                  luisa::span{
                      finite_normal_result})
           << finite_uv_output.copy_to(
                  luisa::span{finite_uv_result})
           << forward_position_output.copy_to(
                  luisa::span{
                      forward_position_result})
           << forward_normal_output.copy_to(
                  luisa::span{
                      forward_normal_result})
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
    const auto rectangle = rectangle_result[0u];
    if (!approximately_equal(
            rectangle.x,
            0.36552444100379944) ||
        !approximately_equal(
            rectangle.y,
            -0.03522232174873352) ||
        !approximately_equal(
            rectangle.z,
            1.399999976158142) ||
        !approximately_equal(
            rectangle.w,
            5.639498710632324)) {
        std::cerr
            << "Cycles rectangle solid-angle oracle regression failed on "
            << backend << ": got {" << rectangle.x << ", "
            << rectangle.y << ", " << rectangle.z << ", "
            << rectangle.w << "}\n";
        return EXIT_FAILURE;
    }
    const auto grazing_rectangle =
        rectangle_result[1u];
    if (!approximately_equal(
            grazing_rectangle.x,
            0.0831405520439148) ||
        !approximately_equal(
            grazing_rectangle.y,
            0.013823390007019043) ||
        !approximately_equal(
            grazing_rectangle.z,
            1.399999976158142) ||
        !approximately_equal(
            grazing_rectangle.w,
            16.717134475708008)) {
        std::cerr
            << "Cycles grazing rectangle oracle regression failed on "
            << backend << ": got {"
            << grazing_rectangle.x << ", "
            << grazing_rectangle.y << ", "
            << grazing_rectangle.z << ", "
            << grazing_rectangle.w << "}\n";
        return EXIT_FAILURE;
    }
    const auto intersection_position =
        intersection_result[0u];
    const auto intersection_measure =
        intersection_result[1u];
    if (!approximately_equal(
            intersection_position.x,
            0.30060097575187683) ||
        !approximately_equal(
            intersection_position.y,
            -0.0844711884856224) ||
        !approximately_equal(
            intersection_position.z,
            1.399999976158142) ||
        !approximately_equal(
            intersection_position.w,
            1.4001514911651611) ||
        !approximately_equal(
            intersection_measure.x,
            5.228616237640381) ||
        !approximately_equal(
            intersection_measure.y,
            0.7957746982574463) ||
        !approximately_equal(
            intersection_measure.z,
            0.7510576248168945) ||
        !approximately_equal(
            intersection_measure.w,
            -0.1643088459968567)) {
        std::cerr
            << "Cycles forward rectangle intersection regression failed on "
            << backend << ": position/distance={"
            << intersection_position.x << ", "
            << intersection_position.y << ", "
            << intersection_position.z << ", "
            << intersection_position.w << "}, measure/uv={"
            << intersection_measure.x << ", "
            << intersection_measure.y << ", "
            << intersection_measure.z << ", "
            << intersection_measure.w << "}\n";
        return EXIT_FAILURE;
    }
    const auto narrow_intersection_position =
        intersection_result[2u];
    const auto narrow_intersection_measure =
        intersection_result[3u];
    if (!approximately_equal(
            narrow_intersection_position.x,
            0.21015244722366333) ||
        !approximately_equal(
            narrow_intersection_position.y,
            -0.3147875666618347) ||
        !approximately_equal(
            narrow_intersection_position.z,
            -0.4000000059604645) ||
        !approximately_equal(
            narrow_intersection_position.w,
            0.36670389771461487) ||
        !approximately_equal(
            narrow_intersection_measure.x,
            1.3708510398864746) ||
        !approximately_equal(
            narrow_intersection_measure.y,
            3.115095615386963) ||
        !approximately_equal(
            narrow_intersection_measure.z,
            0.14202071726322174) ||
        !approximately_equal(
            narrow_intersection_measure.w,
            0.34782683849334717)) {
        std::cerr
            << std::setprecision(10)
            << "Cycles narrow-spread ellipse intersection regression failed on "
            << backend << ": position/distance={"
            << narrow_intersection_position.x << ", "
            << narrow_intersection_position.y << ", "
            << narrow_intersection_position.z << ", "
            << narrow_intersection_position.w
            << "}, measure/uv={"
            << narrow_intersection_measure.x << ", "
            << narrow_intersection_measure.y << ", "
            << narrow_intersection_measure.z << ", "
            << narrow_intersection_measure.w << "}\n";
        return EXIT_FAILURE;
    }

    constexpr std::array finite_direction_oracles{
        luisa::float4{
            0.24016082286834717f,
            -0.09855160862207413f,
            0.9657175540924072f,
            18.35677719116211f},
        luisa::float4{
            0.22996534407138824f,
            -0.09208722412586212f,
            0.96883225440979f,
            18.65931510925293f},
        luisa::float4{
            0.2227325737476349f,
            -0.12098376452922821f,
            0.9673433899879456f,
            39.30256271362305f}};
    constexpr std::array finite_position_oracles{
        luisa::float4{
            0.4156992435455322f,
            -0.0938410609960556f,
            1.2567565441131592f,
            1.3013709783554077f},
        luisa::float4{
            0.43702536821365356f,
            -0.09928160160779953f,
            1.406554102897644f,
            1.4518035650253296f},
        luisa::float4{
            0.4034624397754669f,
            -0.12870670855045319f,
            1.3042311668395996f,
            1.348260760307312f}};
    constexpr std::array finite_normal_oracles{
        luisa::float4{
            0.24052239954471588f,
            0.6113628149032593f,
            -0.7539128065109253f,
            0.7016701698303223f},
        luisa::float4{
            -0.22996534407138824f,
            0.09208722412586212f,
            -0.96883225440979f,
            0.7016701698303223f},
        luisa::float4{
            0.25740334391593933f,
            0.6253330111503601f,
            -0.736683189868927f,
            1.4988338947296143f}};
    const auto float4_equal =
        [](luisa::float4 actual,
           luisa::float4 expected) noexcept {
            return approximately_equal(
                       actual.x, expected.x, 1.5e-5) &&
                   approximately_equal(
                       actual.y, expected.y, 1.5e-5) &&
                   approximately_equal(
                       actual.z, expected.z, 1.5e-5) &&
                   approximately_equal(
                       actual.w, expected.w, 1.5e-5);
        };
    for (std::size_t index = 0u;
         index < finite_case_count;
         ++index) {
        if (!float4_equal(
                finite_direction_result[index],
                finite_direction_oracles[index]) ||
            !float4_equal(
                finite_position_result[index],
                finite_position_oracles[index]) ||
            !float4_equal(
                finite_normal_result[index],
                finite_normal_oracles[index]) ||
            finite_uv_result[index].z != 1.0f ||
            finite_uv_result[index].w != 1.0f) {
            const auto direction =
                finite_direction_result[index];
            const auto position =
                finite_position_result[index];
            const auto normal =
                finite_normal_result[index];
            std::cerr
                << std::setprecision(10)
                << "Cycles finite-light NEE oracle failed on "
                << backend << " for case " << index
                << ": direction/pdf={" << direction.x
                << ", " << direction.y << ", "
                << direction.z << ", " << direction.w
                << "}, position/distance={"
                << position.x << ", " << position.y
                << ", " << position.z << ", "
                << position.w << "}, normal/eval={"
                << normal.x << ", " << normal.y << ", "
                << normal.z << ", " << normal.w
                << "}\n";
            return EXIT_FAILURE;
        }
        const auto forward_position =
            forward_position_result[index];
        const auto forward_normal =
            forward_normal_result[index];
        // NEE explicitly remaps the hit to the authored sphere radius after
        // the law-of-cosines solve; a fresh forward solve can therefore differ
        // by a few single-precision ulps in P/Ng while retaining the same
        // directional measure.
        if (!approximately_equal(
                forward_position.x,
                finite_position_result[index].x,
                1.0e-5) ||
            !approximately_equal(
                forward_position.y,
                finite_position_result[index].y,
                1.0e-5) ||
            !approximately_equal(
                forward_position.z,
                finite_position_result[index].z,
                1.0e-5) ||
            !approximately_equal(
                forward_position.w,
                finite_position_result[index].w,
                1.0e-5) ||
            !approximately_equal(
                forward_normal.x,
                finite_normal_result[index].x,
                1.0e-5) ||
            !approximately_equal(
                forward_normal.y,
                finite_normal_result[index].y,
                1.0e-5) ||
            !approximately_equal(
                forward_normal.z,
                finite_normal_result[index].z,
                1.0e-5) ||
            !approximately_equal(
                forward_normal.w,
                finite_direction_result[index].w)) {
            std::cerr
                << "finite-light NEE/forward reciprocity failed on "
                << backend << " for case " << index
                << ": forward position/distance={"
                << forward_position.x << ", "
                << forward_position.y << ", "
                << forward_position.z << ", "
                << forward_position.w
                << "}, forward normal/pdf={"
                << forward_normal.x << ", "
                << forward_normal.y << ", "
                << forward_normal.z << ", "
                << forward_normal.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
