#include "graph_surface_internal.h"

#include <array>
#include <cstdlib>

namespace psycles::luisa_backend::detail {

std::unique_ptr<ValueNode> make_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    using Factory = std::unique_ptr<ValueNode> (*)(
        const compiler::ValueInstruction &) noexcept;
    constexpr std::array<Factory, 6u> factories{
        try_make_math_value_node,
        try_make_context_value_node,
        try_make_image_value_node,
        try_make_normal_value_node,
        try_make_wave_value_node,
        try_make_procedural_value_node};
    for (const auto factory : factories) {
        if (auto node = factory(instruction)) {
            return node;
        }
    }
    // ValueOperation is a closed compiler IR enum. Reaching this point means
    // a new operation was added without an AST node implementation, which is
    // a construction-time programming error rather than a shader fallback.
    std::abort();
}

std::vector<bool> GraphSurfaceImplementation::value_dependency_mask(
    compiler::ValueExpressionId root) const {
    const auto instruction_count =
        _program->value_instructions().size();
    std::vector<bool> active(instruction_count, false);
    std::vector<compiler::ValueExpressionId> pending;
    pending.emplace_back(root);
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (!id.valid() ||
            id.value >= instruction_count ||
            active[id.value]) {
            continue;
        }
        active[id.value] = true;
        const auto &instruction =
            _program->value_instructions()[id.value];
        const std::array dependencies{
            instruction.a,
            instruction.b,
            instruction.c,
            instruction.d,
            instruction.e,
            instruction.f,
            instruction.g,
            instruction.h,
            instruction.i,
            instruction.j};
        for (const auto dependency : dependencies) {
            if (dependency.valid()) {
                pending.emplace_back(dependency);
            }
        }
    }
    return active;
}

TracedValues GraphSurfaceImplementation::trace_values(
    const ShaderServices &services,
    const SurfacePoint &point,
    const std::vector<bool> *active_mask) const noexcept {
    TracedValues result;
    result.shading_normal = point.shading_normal;
    const auto &instructions =
        _program->value_instructions();
    result.values.reserve(instructions.size());
    ValueEvaluationContext context{
        .services = services,
        .point = point,
        .result = result,
        .surface = *this};
    for (std::size_t instruction_index = 0u;
         instruction_index < instructions.size();
         ++instruction_index) {
        if (active_mask != nullptr &&
            !(*active_mask)[instruction_index]) {
            result.values.emplace_back(
                SurfaceValueExpression::zero(
                    instructions[instruction_index].result_type));
            continue;
        }
        result.values.emplace_back(
            _value_nodes[instruction_index]->evaluate(context));
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
