#include <psycles/luisa/cycles_volume_phase.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace phase =
    psycles::luisa_backend::cycles_volume_phase;

constexpr std::array cosines{
    -0.8f,
    -0.1f,
    0.0f,
    0.65f,
    0.999f};
constexpr std::array randoms{
    luisa::float2{0.17f, 0.83f},
    luisa::float2{0.61f, 0.07f},
    luisa::float2{0.93f, 0.41f}};
constexpr std::array diameters{
    0.05f,
    0.5f,
    2.0f,
    10.0f};
inline constexpr std::size_t record_count = 32u;

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 2.0e-5f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

[[nodiscard]] bool approximately_equal(
    luisa::float4 actual,
    luisa::float4 expected,
    float tolerance = 2.0e-5f) noexcept {
    return approximately_equal(
               actual.x, expected.x, tolerance) &&
           approximately_equal(
               actual.y, expected.y, tolerance) &&
           approximately_equal(
               actual.z, expected.z, tolerance) &&
           approximately_equal(
               actual.w, expected.w, tolerance);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{
            argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output =
        device.create_buffer<luisa::float4>(
            record_count);

    Kernel1D evaluate =
        [](BufferFloat4 records) noexcept {
            const auto hg =
                phase::henyey_greenstein(0.42f);
            const auto rayleigh = phase::rayleigh();
            const auto draine =
                phase::draine(-0.31f, 2.4f);
            const auto fournier_forand =
                phase::fournier_forand(
                    0.12f, 1.33f);

            for (auto index = 0u;
                 index < cosines.size();
                 ++index) {
                const Float cosine{
                    cosines[index]};
                records.write(
                    index,
                    make_float4(
                        phase::evaluate(hg, cosine),
                        phase::evaluate(
                            rayleigh, cosine),
                        phase::evaluate(
                            draine, cosine),
                        cosine));
            }
            records.write(
                5u,
                make_float4(
                    fournier_forand.parameters,
                    0.0f));
            for (auto index = 0u;
                 index < cosines.size();
                 ++index) {
                records.write(
                    6u + index,
                    make_float4(
                        phase::evaluate(
                            fournier_forand,
                            Float{cosines[index]}),
                        0.0f,
                        0.0f,
                        0.0f));
            }

            const auto axis = normalize(
                make_float3(
                    0.2f,
                    -0.3f,
                    0.9327379f));
            for (auto random_index = 0u;
                 random_index < randoms.size();
                 ++random_index) {
                const Float2 random{
                    randoms[random_index]};
                const std::array closures{
                    hg,
                    rayleigh,
                    draine,
                    fournier_forand};
                for (auto closure_index = 0u;
                     closure_index <
                     closures.size();
                     ++closure_index) {
                    const auto sampled =
                        phase::sample(
                            closures[closure_index],
                            axis,
                            random);
                    records.write(
                        11u +
                            random_index * 4u +
                            closure_index,
                        make_float4(
                            sampled.direction,
                            sampled.pdf));
                }
            }
            for (auto index = 0u;
                 index < diameters.size();
                 ++index) {
                const auto mie =
                    phase::mie_parameters(
                        Float{diameters[index]});
                records.write(
                    23u + index,
                    make_float4(
                        mie.henyey_greenstein_g,
                        mie.draine_g,
                        mie.draine_alpha,
                        mie.draine_weight));
            }
            records.write(
                27u,
                make_float4(
                    phase::henyey_greenstein(
                        2.0f)
                        .parameters.x,
                    phase::henyey_greenstein(
                        -2.0f)
                        .parameters.x,
                    phase::fournier_forand(
                        -2.0f, 0.5f)
                        .parameters.x,
                    0.0f));

            const auto special_random =
                make_float2(0.37f, 0.72f);
            const auto draine_rayleigh =
                phase::sample(
                    phase::draine(0.0f, 1.0f),
                    axis,
                    special_random);
            const auto direct_rayleigh =
                phase::sample(
                    rayleigh,
                    axis,
                    special_random);
            const auto draine_hg =
                phase::sample(
                    phase::draine(0.42f, 0.0f),
                    axis,
                    special_random);
            const auto direct_hg =
                phase::sample(
                    hg,
                    axis,
                    special_random);
            records.write(
                28u,
                make_float4(
                    draine_rayleigh.direction,
                    draine_rayleigh.pdf));
            records.write(
                29u,
                make_float4(
                    direct_rayleigh.direction,
                    direct_rayleigh.pdf));
            records.write(
                30u,
                make_float4(
                    draine_hg.direction,
                    draine_hg.pdf));
            records.write(
                31u,
                make_float4(
                    direct_hg.direction,
                    direct_hg.pdf));
        };

    auto shader = device.compile(evaluate);
    std::array<luisa::float4, record_count>
        actual{};
    stream
        << shader(output).dispatch(1u)
        << output.copy_to(luisa::span{actual})
        << synchronize();

    // Generated by compiling the unmodified volume_util.h from official
    // Cycles main b82c3f0 as a CPU oracle. The table covers every current
    // phase family, deterministic direction mapping, Fournier-Forand Newton
    // inversion, Mie's fitted split, and Cycles setup clamps.
    constexpr std::array expected{
        luisa::float4{
            0.0260802992f,
            0.097880289f,
            0.200841382f,
            -0.8f},
        luisa::float4{
            0.046317365f,
            0.0602799281f,
            0.0358505212f,
            -0.1f},
        luisa::float4{
            0.0513657853f,
            0.0596830994f,
            0.0320821926f,
            0.0f},
        luisa::float4{
            0.130942956f,
            0.084899202f,
            0.0403973497f,
            0.65f},
        luisa::float4{
            0.334655464f,
            0.1192469f,
            0.0556322373f,
            0.999f},
        luisa::float4{
            1.33000004f,
            -0.442451179f,
            0.180487335f,
            0.0f},
        luisa::float4{
            0.0190321561f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0208995752f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0230108351f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0722317621f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            6.62556696f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            -0.117953472f,
            -0.899912298f,
            -0.419815481f,
            0.044300843f},
        luisa::float4{
            -0.208764628f,
            -0.415082395f,
            -0.885507822f,
            0.0926465988f},
        luisa::float4{
            -0.219230354f,
            -0.100169748f,
            -0.970517457f,
            0.291887999f},
        luisa::float4{
            0.191538081f,
            -0.380830377f,
            0.904589117f,
            2.88411236f},
        luisa::float4{
            0.812768638f,
            -0.276587814f,
            0.51274389f,
            0.152933449f},
        luisa::float4{
            0.98489809f,
            -0.168242604f,
            0.0408668816f,
            0.064550288f},
        luisa::float4{
            0.748615801f,
            0.0706236959f,
            -0.659231961f,
            0.0814913437f},
        luisa::float4{
            0.712173164f,
            -0.2961815f,
            0.636463642f,
            0.140289679f},
        luisa::float4{
            0.0898205638f,
            -0.0658485889f,
            0.993778825f,
            0.295885146f},
        luisa::float4{
            0.0114951432f,
            0.0955966115f,
            0.995353818f,
            0.108244121f},
        luisa::float4{
            -0.089179188f,
            0.297496527f,
            0.950548768f,
            0.0455983877f},
        luisa::float4{
            -0.438733011f,
            0.897183418f,
            -0.0507471859f,
            0.0177368727f},
        luisa::float4{
            0.0345000029f,
            0.0256611835f,
            250.0f,
            0.252180696f},
        luisa::float4{
            0.793295205f,
            0.718332887f,
            250.0f,
            0.404244363f},
        luisa::float4{
            0.918084502f,
            0.387489855f,
            11.379179f,
            0.411993027f},
        luisa::float4{
            0.988176703f,
            0.555677176f,
            21.9954796f,
            0.481955439f},
        luisa::float4{
            0.999f,
            -0.999f,
            1.001f,
            0.0f}};

    for (auto index = std::size_t{0u};
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "Cycles volume phase oracle failed on "
                << backend << " at record " << index
                << ": got {" << actual[index].x
                << ", " << actual[index].y
                << ", " << actual[index].z
                << ", " << actual[index].w
                << "}, expected {"
                << expected[index].x << ", "
                << expected[index].y << ", "
                << expected[index].z << ", "
                << expected[index].w << "}\n";
            return EXIT_FAILURE;
        }
    }

    for (auto index = std::size_t{11u};
         index < 23u;
         ++index) {
        const auto direction = actual[index].xyz();
        if (!approximately_equal(
                dot(direction, direction),
                1.0f,
                4.0e-5f) ||
            !(actual[index].w > 0.0f) ||
            !std::isfinite(actual[index].w)) {
            std::cerr
                << "volume phase sample invariant failed on "
                << backend << " at record " << index
                << '\n';
            return EXIT_FAILURE;
        }
    }
    if (!approximately_equal(
            actual[28u], actual[29u]) ||
        !approximately_equal(
            actual[30u], actual[31u])) {
        std::cerr
            << "Draine special-case reduction failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
