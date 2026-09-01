#include <psycles/luisa/cycles_magic.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/translators/ast2xir.h>

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

    cycles_magic::prepare();
    Kernel1D evaluate = [](
                            BufferFloat input,
                            BufferUInt depth,
                            BufferFloat4 output) noexcept {
        const auto vector = make_float3(
            input.read(0u),
            input.read(1u),
            input.read(2u));
        const auto color = cycles_magic::evaluate(
            depth.read(0u),
            vector,
            input.read(3u),
            input.read(4u));
        output.write(0u, make_float4(color, 1.0f));
    };

    auto module = luisa::compute::xir::ast_to_xir_translate(
        evaluate.function()->function(), {});
    std::size_t loop_count = 0u;
    std::size_t switch_count = 0u;
    for (auto *function : module->function_list()) {
        if (auto *definition = function->definition()) {
            definition->traverse_instructions(
                [&](const luisa::compute::xir::Instruction
                        *instruction) noexcept {
                    loop_count += instruction->isa<
                                      luisa::compute::xir::LoopInst>()
                                      ? 1u
                                      : 0u;
                    switch_count += instruction->isa<
                                        luisa::compute::xir::SwitchInst>()
                                        ? 1u
                                        : 0u;
                });
        }
    }
    if (loop_count != 1u || switch_count != 1u) {
        std::cerr
            << "Magic depth must lower to one runtime loop/switch, got "
            << loop_count << " loop(s) and " << switch_count
            << " switch(es)\n";
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto input = device.create_buffer<float>(5u);
    auto depth = device.create_buffer<std::uint32_t>(1u);
    auto output = device.create_buffer<luisa::float4>(1u);
    constexpr std::array authored{
        1.0e20f,
        -0.375f,
        0.8125f,
        0.001f,
        1.0f};
    constexpr std::array authored_depth{2u};
    stream << input.copy_from(luisa::span{authored})
           << depth.copy_from(luisa::span{authored_depth});

    constexpr auto tolerance = 3.0e-6f;
    for (auto fast_math : {false, true}) {
        auto shader = device.compile(
            evaluate,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = fast_math});
        std::array<luisa::float4, 1u> actual{};
        stream << shader(input, depth, output).dispatch(1u)
               << output.copy_to(luisa::span{actual})
               << synchronize();
        const auto mismatch =
            std::abs(actual[0].x - cycles_large_coordinate_oracle.x) >
                tolerance ||
            std::abs(actual[0].y - cycles_large_coordinate_oracle.y) >
                tolerance ||
            std::abs(actual[0].z - cycles_large_coordinate_oracle.z) >
                tolerance;
        if (mismatch) {
            std::cerr << "Cycles Magic Texture oracle failed on "
                      << backend << " in "
                      << (fast_math ? "fast" : "precise")
                      << " mode: got {" << actual[0].x << ", "
                      << actual[0].y << ", " << actual[0].z
                      << "}, expected {"
                      << cycles_large_coordinate_oracle.x << ", "
                      << cycles_large_coordinate_oracle.y << ", "
                      << cycles_large_coordinate_oracle.z << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
