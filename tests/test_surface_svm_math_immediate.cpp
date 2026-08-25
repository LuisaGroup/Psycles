#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] SurfaceProgram make_math_program(
    std::uint32_t tag,
    std::uint64_t operation,
    std::uint64_t reserved = 0u) {
    std::vector<ParameterDesc> parameters;
    std::vector<ValueInstruction> values;
    parameters.reserve(value_operand::ternary::count);
    values.reserve(value_operand::ternary::count + 1u);
    for (auto index = std::uint32_t{0u};
         index < value_operand::ternary::count;
         ++index) {
        parameters.emplace_back(ParameterDesc{
            .id = ParameterId{index},
            .node = NodeId{index + 1u},
            .socket = "Value" + std::to_string(index),
            .type = SocketType::floating,
            .default_value = SocketValue::floating(0.25f * (index + 1u)),
            .source = ParameterSource::input});
        values.emplace_back(ValueInstruction{
            .operation = ValueOperation::parameter,
            .result_type = SocketType::floating,
            .parameter = ParameterId{index}});
    }
    values.emplace_back(ValueInstruction{
        .operation = ValueOperation::math,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::ternary>({
            {value_operand::ternary::a, ValueExpressionId{0u}},
            {value_operand::ternary::b, ValueExpressionId{1u}},
            {value_operand::ternary::c, ValueExpressionId{2u}}}),
        .static_u0 = operation,
        .static_u1 = reserved});
    return SurfaceProgram{
        tag, std::move(parameters), std::move(values), {}, {}};
}

[[nodiscard]] SurfaceValueStoragePlan make_plan(
    const SurfaceProgram &program) {
    const auto count = program.value_instructions().size();
    auto outputs = std::vector<bool>(count, false);
    outputs.back() = true;
    return plan_surface_value_storage(
        program, std::vector<bool>(count, true), outputs);
}

void test_math_immediate_quotient() {
    const std::vector operations{
        MathOperation::add,
        MathOperation::power,
        MathOperation::smooth_maximum,
        MathOperation::sine,
        MathOperation::degrees};
    std::vector<SurfaceProgram> programs;
    std::vector<SurfaceValueStoragePlan> plans;
    programs.reserve(operations.size());
    plans.reserve(operations.size());
    for (auto index = std::size_t{0u}; index < operations.size(); ++index) {
        programs.emplace_back(make_math_program(
            static_cast<std::uint32_t>(index),
            static_cast<std::uint64_t>(operations[index])));
        plans.emplace_back(make_plan(programs.back()));
        const auto image = lower_surface_value_program(
            programs.back(), plans.back());
        require(image.valid && image.instructions.size() == 1u,
                "valid Math mode failed bytecode lowering");
        require(surface_value_svm_immediate(image.instructions.front()) ==
                    static_cast<std::uint32_t>(operations[index]),
                "Math mode was not preserved by its opcode immediate");
    }

    std::vector<SurfaceValueExecutionInput> inputs;
    inputs.reserve(programs.size());
    for (auto index = std::size_t{0u}; index < programs.size(); ++index) {
        inputs.emplace_back(SurfaceValueExecutionInput{
            .program = &programs[index], .storage = &plans[index]});
    }
    const auto scene = build_surface_value_executable_scene(inputs);
    std::vector<std::uint16_t> expected;
    expected.reserve(operations.size());
    for (const auto operation : operations) {
        expected.emplace_back(static_cast<std::uint16_t>(operation));
    }
    require(scene.valid && scene.variants.size() == 1u &&
                scene.instruction_variants ==
                    std::vector<std::uint32_t>(operations.size(), 0u) &&
                scene.variants.front().instruction.static_u0 == 0u &&
                scene.variants.front().instruction.static_u1 == 0u &&
                scene.variants.front().svm_immediates == expected,
            "Math variants did not form the exact immediate quotient");

    auto mismatched =
        lower_surface_value_program(programs.front(), plans.front());
    mismatched.instructions.front().control |=
        1u << surface_value_svm_immediate_shift;
    const auto mismatched_scene =
        build_surface_value_scene_image(std::vector{mismatched});
    require(!mismatched_scene.valid &&
                mismatched_scene.diagnostic.find(
                    "immediate disagrees with immutable metadata") !=
                    std::string::npos,
            "serialized Math accepted an immediate/metadata disagreement");

    const auto invalid_mode =
        make_math_program(100u, math_operation_count);
    const auto invalid_reserved = make_math_program(
        101u, static_cast<std::uint64_t>(MathOperation::add), 1u);
    require(!lower_surface_value_program(invalid_mode, make_plan(invalid_mode))
                 .valid &&
                !lower_surface_value_program(
                     invalid_reserved, make_plan(invalid_reserved))
                     .valid,
            "Math lowering accepted fields outside its exact immediate ABI");
}

} // namespace

int main() {
    try {
        test_math_immediate_quotient();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Surface Math immediate test failure: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
