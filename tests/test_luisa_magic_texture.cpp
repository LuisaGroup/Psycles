#include <psycles/luisa/cycles_magic.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

// Blender/Cycles 5.3 CPU and HIP agree on this raw Magic Texture closure
// evaluation. The probe keeps Vector dynamic so Blender cannot replace the
// node with its distinct host-side constant evaluator: this value therefore
// comes from the actual Cycles SVM kernel, not from a second renderer.
constexpr luisa::float3 cycles_large_coordinate_oracle{
    0.555838704f,
    0.657886386f,
    0.937048197f};

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    cycles_magic::prepare(2u);
    Kernel1D evaluate = [](
                            BufferFloat input,
                            BufferFloat4 output) noexcept {
        const auto vector = make_float3(
            input.read(0u),
            input.read(1u),
            input.read(2u));
        const auto color = cycles_magic::evaluate(
            2u,
            vector,
            input.read(3u),
            input.read(4u));
        output.write(0u, make_float4(color, 1.0f));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto input = device.create_buffer<float>(5u);
    auto output = device.create_buffer<luisa::float4>(1u);
    auto shader = device.compile(
        evaluate,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = false});
    constexpr std::array authored{
        1.0e20f,
        -0.375f,
        0.8125f,
        0.001f,
        1.0f};
    std::array<luisa::float4, 1u> actual{};
    stream << input.copy_from(luisa::span{authored})
           << shader(input, output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    constexpr auto tolerance = 3.0e-6f;
    const auto mismatch =
        std::abs(actual[0].x - cycles_large_coordinate_oracle.x) >
            tolerance ||
        std::abs(actual[0].y - cycles_large_coordinate_oracle.y) >
            tolerance ||
        std::abs(actual[0].z - cycles_large_coordinate_oracle.z) >
            tolerance;
    if (mismatch) {
        std::cerr << "Cycles Magic Texture oracle failed on " << backend
                  << ": got {" << actual[0].x << ", " << actual[0].y
                  << ", " << actual[0].z << "}, expected {"
                  << cycles_large_coordinate_oracle.x << ", "
                  << cycles_large_coordinate_oracle.y << ", "
                  << cycles_large_coordinate_oracle.z << "}\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
