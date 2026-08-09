#include "graph_surface_internal.h"

#include <array>
#include <cstdlib>

namespace psycles::luisa_backend::detail {

std::unique_ptr<ValueNode> make_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    using Factory = std::unique_ptr<ValueNode> (*)(
        const compiler::ValueInstruction &) noexcept;
    constexpr std::array<Factory, 8u> factories{
        try_make_math_value_node,
        try_make_context_value_node,
        try_make_image_value_node,
        try_make_normal_value_node,
        try_make_magic_value_node,
        try_make_wave_value_node,
        try_make_voronoi_value_node,
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
        for (const auto dependency : instruction.operands) {
            if (dependency.valid()) {
                pending.emplace_back(dependency);
            }
        }
    }
    return active;
}

TracedValues GraphSurfaceImplementation::trace_value_stage(
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

SurfacePoint GraphSurfaceImplementation::automatic_bump_point(
    const SurfacePoint &point) const noexcept {
    if (!_automatic_bump_uses_undisplaced_geometry) {
        return point;
    }
    auto result = point;
    result.position = point.undisplaced_position;
    result.object_position = point.undisplaced_object_position;
    result.shading_normal = point.undisplaced_shading_normal;
    result.object_shading_normal =
        point.undisplaced_object_shading_normal;
    result.dPdx = point.undisplaced_dPdx;
    result.dPdy = point.undisplaced_dPdy;
    result.object_dPdx = point.undisplaced_object_dPdx;
    result.object_dPdy = point.undisplaced_object_dPdy;
    return result;
}

TracedValues GraphSurfaceImplementation::trace_values(
    const ShaderServices &services,
    const SurfacePoint &point,
    const std::vector<bool> *active_mask) const noexcept {
    // Explicit masks denote isolated compiler domains (displacement or a
    // Bump offset sample) and must not recursively enter SetNormal.
    if (active_mask != nullptr ||
        !_program->surface_normal_root().valid()) {
        return trace_value_stage(services, point, active_mask);
    }

    // Cycles executes the automatic bump region, stores its root in sd->N,
    // and only then enters the surface region. Re-recording the typed value
    // program with an updated point is the Luisa multistage equivalent: all
    // later context nodes observe the new normal, while ordinary Bump nodes
    // remain pure values.
    const auto bump_values = trace_value_stage(
        services,
        automatic_bump_point(point),
        &_surface_normal_dependency_mask);
    auto surface_point = point;
    surface_point.shading_normal =
        bump_values.values[
            _program->surface_normal_root().value].vector();
    return trace_value_stage(services, surface_point, nullptr);
}

}// namespace psycles::luisa_backend::detail
