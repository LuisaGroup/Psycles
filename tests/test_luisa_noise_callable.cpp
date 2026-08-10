#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <psycles/luisa/cycles_noise.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::xir_instruction_count;

constexpr auto case_count = std::uint32_t{8u};
constexpr auto result_stride = std::uint32_t{2u};

struct ModuleShape {
    std::size_t instructions{};
    std::size_t callable_definitions{};
    std::size_t loops{};
};

template<typename... Args>
[[nodiscard]] ModuleShape module_shape(
    const Kernel1D<Args...> &kernel) {
    auto module = luisa::compute::xir::ast_to_xir_translate(
        kernel.function()->function(), {});
    ModuleShape result;
    for (auto *function : module->function_list()) {
        result.callable_definitions +=
            function->derived_function_tag() ==
                    luisa::compute::xir::DerivedFunctionTag::CALLABLE
                ? 1u
                : 0u;
        if (auto *definition = function->definition()) {
            definition->traverse_instructions(
                [&](const luisa::compute::xir::Instruction
                        *instruction) noexcept {
                    ++result.instructions;
                    result.loops +=
                        instruction->isa<
                            luisa::compute::xir::LoopInst>()
                            ? 1u
                            : 0u;
                });
        }
    }
    return result;
}

[[nodiscard]] Float test_coordinate(UInt index) noexcept {
    const auto value = cast<float>(index);
    const auto ordinary = -2.125f + 0.713f * value;
    const auto large = 1000000.25f + 17.0f * value;
    return select(ordinary, large, index >= 4u);
}

[[nodiscard]] Kernel1D<Buffer<float>>
make_repeated_3d_kernel(bool shared) {
    constexpr auto repetition_count = std::uint32_t{8u};
    return [shared](BufferFloat output) noexcept {
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const Float3 p = make_float3(
                0.17f + 0.31f * static_cast<float>(index),
                -1.25f + 0.19f * static_cast<float>(index),
                2.75f - 0.23f * static_cast<float>(index));
            output.write(
                index,
                shared
                    ? cycles_noise::signed_noise(p)
                    : cycles_noise::signed_noise_inline(p));
        }
    };
}

[[nodiscard]] Kernel1D<Buffer<luisa::float4>>
make_lone_monk_noise_family_kernel() {
    return [](BufferFloat4 output) noexcept {
        const auto vector = make_float3(0.37f, -1.25f, 2.5f);
        constexpr auto type = cycles_noise::Type::fbm;
        output.write(
            0u,
            cycles_noise::evaluate_texture_shared(
                3u,
                type,
                true,
                false,
                vector,
                0.0f,
                4.25f,
                0.55f,
                2.0f,
                0.0f,
                1.0f,
                0.3f));
        output.write(
            1u,
            cycles_noise::evaluate_texture_shared(
                3u,
                type,
                true,
                true,
                vector,
                0.0f,
                4.25f,
                0.55f,
                2.0f,
                0.0f,
                1.0f,
                0.3f));
        output.write(
            2u,
            cycles_noise::evaluate_texture_shared(
                2u,
                type,
                false,
                false,
                vector,
                0.0f,
                4.25f,
                0.55f,
                2.0f,
                0.0f,
                1.0f,
                0.3f));
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_3d_kernel(false);
    const auto shared_shape =
        make_repeated_3d_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "signed 3D Perlin 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared signed Perlin did not reduce repeated XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    Kernel1D used_only = [](BufferFloat output) noexcept {
        const auto value = cast<float>(dispatch_x());
        output.write(
            0u,
            cycles_noise::signed_noise(
                make_float3(value, value + 1.0f, value - 2.0f)));
    };
    const auto used_only_shape = module_shape(used_only);
    if (used_only_shape.callable_definitions != 1u) {
        std::cerr
            << "A 3D-only signed-noise module reached "
            << used_only_shape.callable_definitions
            << " definitions instead of one\n";
        return EXIT_FAILURE;
    }

    Kernel1D all_dimensions = [](BufferFloat4 output) noexcept {
        const auto value = cast<float>(dispatch_x());
        output.write(
            0u,
            make_float4(
                cycles_noise::signed_noise(value),
                cycles_noise::signed_noise(
                    make_float2(value, value + 1.0f)),
                cycles_noise::signed_noise(
                    make_float3(value, value + 1.0f, value - 2.0f)),
                cycles_noise::signed_noise(
                    make_float4(
                        value,
                        value + 1.0f,
                        value - 2.0f,
                        value + 3.0f))));
    };
    const auto all_dimensions_shape = module_shape(all_dimensions);
    if (all_dimensions_shape.callable_definitions != 4u) {
        std::cerr
            << "Signed-noise dimension family reached "
            << all_dimensions_shape.callable_definitions
            << " definitions instead of four\n";
        return EXIT_FAILURE;
    }

    const auto scene_family =
        make_lone_monk_noise_family_kernel();
    const auto scene_shape = module_shape(scene_family);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "Lone Monk Noise family XIR: "
            << scene_shape.instructions
            << ", definitions="
            << scene_shape.callable_definitions
            << ", runtime loops=" << scene_shape.loops << '\n';
    }
    if (scene_shape.instructions > 15000u ||
        scene_shape.callable_definitions != 5u ||
        scene_shape.loops != 5u) {
        std::cerr
            << "Lone Monk Noise family shape regression: instructions="
            << scene_shape.instructions
            << ", definitions="
            << scene_shape.callable_definitions
            << ", runtime loops=" << scene_shape.loops << '\n';
        return EXIT_FAILURE;
    }

    Kernel1D compare = [](BufferFloat4 output) noexcept {
        const auto index = dispatch_x();
        const auto p1 = test_coordinate(index);
        const auto p2 = make_float2(
            p1,
            -0.375f * p1 + 0.25f);
        const auto p3 = make_float3(
            p2,
            0.125f * p1 - 3.0f);
        const auto p4 = make_float4(
            p3,
            -0.0625f * p1 + 7.0f);
        const auto base = index * result_stride;
        output.write(
            base,
            make_float4(
                cycles_noise::signed_noise_inline(p1),
                cycles_noise::signed_noise_inline(p2),
                cycles_noise::signed_noise_inline(p3),
                cycles_noise::signed_noise_inline(p4)));
        output.write(
            base + 1u,
            make_float4(
                cycles_noise::signed_noise(p1),
                cycles_noise::signed_noise(p2),
                cycles_noise::signed_noise(p3),
                cycles_noise::signed_noise(p4)));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(
        case_count * result_stride);
    auto shader = device.compile(compare);
    std::vector<luisa::float4> actual(
        case_count * result_stride);
    stream << shader(output).dispatch(case_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto index = std::uint32_t{0u};
         index < case_count;
         ++index) {
        const auto direct = actual[index * result_stride];
        const auto shared = actual[index * result_stride + 1u];
        if (!approximately_equal(direct, shared)) {
            std::cerr
                << "Signed-noise callable mismatch on " << backend
                << ", case " << index << ": direct={"
                << direct.x << ", " << direct.y << ", "
                << direct.z << ", " << direct.w << "}, shared={"
                << shared.x << ", " << shared.y << ", "
                << shared.z << ", " << shared.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
