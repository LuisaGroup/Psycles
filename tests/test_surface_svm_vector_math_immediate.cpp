#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>

#include <array>
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

[[nodiscard]] SurfaceProgram make_vector_math_program(
    std::uint32_t tag,
    ValueOperation output,
    std::uint64_t operation,
    std::uint64_t reserved = 0u) {
    if (output != ValueOperation::vector_math_value &&
        output != ValueOperation::vector_math_vector) {
        std::abort();
    }
    const std::array types{
        SocketType::vector,
        SocketType::vector,
        SocketType::vector,
        SocketType::floating};
    const std::array defaults{
        SocketValue::vector({0.75f, -0.4f, 1.2f}),
        SocketValue::vector({0.5f, 1.25f, -0.8f}),
        SocketValue::vector({-0.2f, 0.7f, 0.3f}),
        SocketValue::floating(0.65f)};
    std::vector<ParameterDesc> parameters;
    std::vector<ValueInstruction> values;
    parameters.reserve(value_operand::vector_math::count);
    values.reserve(value_operand::vector_math::count + 1u);
    for (auto index = std::uint32_t{0u};
         index < value_operand::vector_math::count;
         ++index) {
        parameters.emplace_back(ParameterDesc{
            .id = ParameterId{index},
            .node = NodeId{index + 1u},
            .socket = "Input" + std::to_string(index),
            .type = types[index],
            .default_value = defaults[index],
            .source = ParameterSource::input});
        values.emplace_back(ValueInstruction{
            .operation = ValueOperation::parameter,
            .result_type = types[index],
            .parameter = ParameterId{index}});
    }
    values.emplace_back(ValueInstruction{
        .operation = output,
        .result_type = output == ValueOperation::vector_math_value
                           ? SocketType::floating
                           : SocketType::vector,
        .operands = make_value_operands<value_operand::vector_math>({
            {value_operand::vector_math::a, ValueExpressionId{0u}},
            {value_operand::vector_math::b, ValueExpressionId{1u}},
            {value_operand::vector_math::c, ValueExpressionId{2u}},
            {value_operand::vector_math::scale, ValueExpressionId{3u}}}),
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

void require_exact_quotient(
    ValueOperation output,
    const std::vector<VectorMathOperation> &operations) {
    std::vector<SurfaceProgram> programs;
    std::vector<SurfaceValueStoragePlan> plans;
    programs.reserve(operations.size());
    plans.reserve(operations.size());
    for (auto index = std::size_t{0u}; index < operations.size(); ++index) {
        programs.emplace_back(make_vector_math_program(
            static_cast<std::uint32_t>(index),
            output,
            static_cast<std::uint64_t>(operations[index])));
        plans.emplace_back(make_plan(programs.back()));
        const auto image = lower_surface_value_program(
            programs.back(), plans.back());
        require(image.valid && image.instructions.size() == 1u,
                "valid Vector Math mode failed bytecode lowering");
        require(surface_value_svm_immediate(image.instructions.front()) ==
                    static_cast<std::uint32_t>(operations[index]),
                "Vector Math mode was not preserved by its immediate");
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
            "Vector Math modes did not form the exact immediate quotient");
}

void test_vector_math_immediate_quotient() {
    require_exact_quotient(
        ValueOperation::vector_math_vector,
        {VectorMathOperation::add,
         VectorMathOperation::refract,
         VectorMathOperation::power,
         VectorMathOperation::tangent,
         VectorMathOperation::round});
    require_exact_quotient(
        ValueOperation::vector_math_value,
        {VectorMathOperation::dot_product,
         VectorMathOperation::distance,
         VectorMathOperation::length});

    auto vector_program = make_vector_math_program(
        100u,
        ValueOperation::vector_math_vector,
        static_cast<std::uint64_t>(VectorMathOperation::add));
    auto value_program = make_vector_math_program(
        101u,
        ValueOperation::vector_math_value,
        static_cast<std::uint64_t>(VectorMathOperation::add));
    auto vector_plan = make_plan(vector_program);
    auto value_plan = make_plan(value_program);
    const std::array separate_outputs{
        SurfaceValueExecutionInput{
            .program = &vector_program, .storage = &vector_plan},
        SurfaceValueExecutionInput{
            .program = &value_program, .storage = &value_plan}};
    require(build_surface_value_executable_scene(separate_outputs)
                    .variants.size() == 2u,
            "typed Vector Math output families were incorrectly merged");

    auto mismatched =
        lower_surface_value_program(vector_program, vector_plan);
    mismatched.instructions.front().control |=
        1u << surface_value_svm_immediate_shift;
    const auto mismatched_scene =
        build_surface_value_scene_image(std::vector{mismatched});
    require(!mismatched_scene.valid &&
                mismatched_scene.diagnostic.find(
                    "immediate disagrees with immutable metadata") !=
                    std::string::npos,
            "serialized Vector Math accepted metadata disagreement");

    const auto invalid_mode = make_vector_math_program(
        102u,
        ValueOperation::vector_math_vector,
        vector_math_operation_count);
    const auto invalid_reserved = make_vector_math_program(
        103u,
        ValueOperation::vector_math_vector,
        static_cast<std::uint64_t>(VectorMathOperation::add),
        1u);
    require(!lower_surface_value_program(invalid_mode, make_plan(invalid_mode))
                 .valid &&
                !lower_surface_value_program(
                     invalid_reserved, make_plan(invalid_reserved))
                     .valid,
            "Vector Math lowering accepted fields outside its immediate ABI");
}

void test_vector_math_round_graph_lowering() {
    ShaderGraph graph;
    const auto vector_math = graph.add_node(
        node_type::vector_math, "Cycles Vector Math Round");
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf, "Vector Math Round consumer");
    require(
        graph.set_property(
            vector_math,
            "Operation",
            SocketValue::string("ROUND")) &&
            graph.set_input(
                vector_math,
                "A",
                SocketValue::vector({-1.5f, 1.5f, 2.49f})),
        "failed to construct Vector Math Round graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    graph.set_root(
        ShaderDomain::surface_normal,
        OutputRef{.node = vector_math, .socket = "Vector"});

    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(graph);
    require(shader.ok(), "Vector Math Round graph failed to compile");
    const auto lowered = compile_surface_program(*shader.program);
    require(lowered.ok(), "Vector Math Round graph failed to lower");

    const auto expected = static_cast<std::uint64_t>(
        VectorMathOperation::round);
    auto found = false;
    for (const auto &instruction :
         lowered.program->value_instructions()) {
        if (instruction.operation !=
            ValueOperation::vector_math_vector) {
            continue;
        }
        found = true;
        require(
            instruction.static_u0 == expected,
            "Vector Math Round silently lowered as another operation");
    }
    require(found, "Vector Math Round producer was not retained");
}

} // namespace

int main() {
    try {
        test_vector_math_immediate_quotient();
        test_vector_math_round_graph_lowering();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Surface Vector Math immediate test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
