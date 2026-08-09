#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>
#include <luisa/xir/translators/ast2xir.h>

namespace psycles::test_support {

template<typename... Args>
[[nodiscard]] std::size_t xir_instruction_count(
    const luisa::compute::Kernel1D<Args...> &kernel) {
    auto module = luisa::compute::xir::ast_to_xir_translate(
        kernel.function()->function(), {});
    auto instructions = std::size_t{0u};
    for (auto *function : module->function_list()) {
        if (auto *definition = function->definition()) {
            definition->traverse_instructions(
                [&](const luisa::compute::xir::Instruction *) noexcept {
                    ++instructions;
                });
        }
    }
    return instructions;
}

template<typename... Args>
[[nodiscard]] bool require_bounded_xir(
    std::string_view name,
    const luisa::compute::Kernel1D<Args...> &kernel,
    std::size_t maximum_instructions) {
    const auto instructions = xir_instruction_count(kernel);
    const auto report =
        std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr;
    if (report || instructions > maximum_instructions) {
        std::cerr << name << ": XIR instructions=" << instructions
                  << ", ceiling=" << maximum_instructions << '\n';
    }
    return instructions <= maximum_instructions;
}

}// namespace psycles::test_support
