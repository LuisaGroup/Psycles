// Graph value dependency analysis and topological evaluator.
// Included by <psycles/luisa/graph_surface.h>; not a standalone header.

    [[nodiscard]] std::vector<bool> value_dependency_mask(
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
    [[nodiscard]] TracedValues trace_values(
        const ShaderServices &services,
        const SurfacePoint &point,
        const std::vector<bool> *active_mask = nullptr) const noexcept {
        TracedValues result;
        result.shading_normal = point.shading_normal;
        const auto &instructions =
            _program->value_instructions();
        result.values.reserve(instructions.size());
        for (std::size_t instruction_index = 0u;
             instruction_index < instructions.size();
             ++instruction_index) {
            if (active_mask != nullptr &&
                !(*active_mask)[instruction_index]) {
                result.values.emplace_back(
                    make_float4(0.0f));
                continue;
            }
            const auto &instruction =
                instructions[instruction_index];
            Float4 value = make_float4(0.0f);
            switch (instruction.operation) {
#include <psycles/luisa/detail/graph_surface_value_math_cases.inl>
#include <psycles/luisa/detail/graph_surface_value_context_cases.inl>
#include <psycles/luisa/detail/graph_surface_value_image_cases.inl>
#include <psycles/luisa/detail/graph_surface_value_procedural_cases.inl>
            }
            result.values.emplace_back(value);
        }
        return result;
    }
