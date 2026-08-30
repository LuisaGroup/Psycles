#include "path_tracer_attribute_residency.h"
#include "path_tracer_scene_geometry.h"

#include <psycles/compiler/surface_execution_plan.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

enum class ResolutionState : std::uint8_t {
    unvisited,
    visiting,
    resolved,
    unknown
};

[[nodiscard]] std::optional<std::uint64_t> resolve_unsigned_expression(
    const compiler::SurfaceProgram &program,
    const compiler::SurfaceParameterBlock &parameters,
    compiler::ValueExpressionId expression,
    std::vector<ResolutionState> &states,
    std::vector<std::optional<std::uint64_t>> &values) {
    if (!expression.valid() ||
        expression.value >= program.value_instructions().size()) {
        return std::nullopt;
    }
    const auto index = static_cast<std::size_t>(expression.value);
    switch (states[index]) {
        case ResolutionState::resolved:
            return values[index];
        case ResolutionState::visiting:
        case ResolutionState::unknown:
            return std::nullopt;
        case ResolutionState::unvisited:
            break;
    }
    states[index] = ResolutionState::visiting;
    const auto &instruction = program.value_instructions()[index];
    std::optional<std::uint64_t> value;
    if (instruction.operation == compiler::ValueOperation::parameter &&
        instruction.parameter.valid()) {
        const auto *parameter = parameters.find(instruction.parameter);
        if (parameter != nullptr &&
            parameter->type == contract::SocketType::unsigned_integer) {
            value = std::get<std::uint64_t>(parameter->value);
        }
    } else if (instruction.operation ==
                   compiler::ValueOperation::passthrough &&
               !instruction.operands.empty()) {
        value = resolve_unsigned_expression(
            program,
            parameters,
            instruction.operands.front(),
            states,
            values);
    }
    states[index] = value ? ResolutionState::resolved
                          : ResolutionState::unknown;
    values[index] = value;
    return value;
}

void require_expression(
    SurfaceAttributeDemand &result,
    const compiler::SurfaceProgram &program,
    const compiler::SurfaceParameterBlock &parameters,
    compiler::ValueExpressionId expression,
    std::vector<ResolutionState> &states,
    std::vector<std::optional<std::uint64_t>> &values) {
    if (result.all) {
        return;
    }
    const auto value = resolve_unsigned_expression(
        program, parameters, expression, states, values);
    if (value) {
        result.ids.emplace(*value);
    } else {
        result.all = true;
        result.ids.clear();
    }
}

[[nodiscard]] std::uint64_t float4_bytes(std::size_t count) noexcept {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    return count > maximum / 16u
               ? maximum
               : static_cast<std::uint64_t>(count) * 16u;
}

[[nodiscard]] std::uint64_t float2_bytes(std::size_t count) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return count > maximum / 8u
               ? maximum
               : static_cast<std::uint64_t>(count) * 8u;
}

void add_bytes(std::uint64_t &total, std::uint64_t bytes) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    total = bytes > maximum - total ? maximum : total + bytes;
}

void merge_material(
    SurfaceAttributeDemand &demand,
    contract::MaterialId id,
    const std::map<contract::MaterialId, SurfaceAttributeDemand> &materials) {
    const auto iter = materials.find(id);
    if (iter == materials.end()) {
        demand.all = true;
        demand.ids.clear();
        return;
    }
    demand.merge(iter->second);
}

void close_tangent_dependencies(
    SurfaceAttributeDemand &demand,
    const contract::TriangleMeshDesc &geometry) {
    if (demand.all) {
        return;
    }
    for (const auto &[name, tangent] : geometry.uv_tangent_layers) {
        static_cast<void>(tangent);
        if (demand.contains(contract::uv_tangent_attribute_id(name)) ||
            demand.contains(
                contract::uv_undisplaced_tangent_attribute_id(name))) {
            // MikkTSpace construction consumes the corresponding UV layer at
            // scene-build time. Retaining it is the dependency closure which
            // makes tangent pruning independent of material graph shape.
            demand.ids.emplace(contract::uv_attribute_id(name));
        }
    }
}

void measure_geometry(
    SceneAttributeResidencyPlan &plan,
    const contract::TriangleMeshDesc &geometry,
    const GeometryAttributeResidency &residency) {
    const auto measure = [&](std::uint64_t id, std::size_t count) {
        const auto bytes = float4_bytes(count);
        ++plan.source_binding_count;
        add_bytes(plan.source_device_bytes, bytes);
        if (residency.contains(id)) {
            ++plan.resident_binding_count;
            add_bytes(plan.resident_device_bytes, bytes);
        }
    };
    for (const auto &[name, attribute] : geometry.color_attributes) {
        measure(contract::attribute_id(name), attribute.values.size());
    }
    for (const auto &[name, attribute] : geometry.uv_layers) {
        measure(contract::uv_attribute_id(name), attribute.values.size());
    }
    const auto tangent_count = geometry.triangles.size() * 3u;
    for (const auto &[name, attribute] : geometry.uv_tangent_layers) {
        static_cast<void>(attribute);
        measure(contract::uv_tangent_attribute_id(name), tangent_count);
        measure(
            contract::uv_undisplaced_tangent_attribute_id(name),
            tangent_count);
    }
    if (geometry.pointiness_source) {
        measure(
            contract::cycles_pointiness_attribute_id,
            geometry.positions.size());
    }
}

void measure_geometry(
    SceneAttributeResidencyPlan &plan,
    const contract::CurveGeometryDesc &geometry,
    const GeometryAttributeResidency &residency) {
    for (const auto &[name, values] : geometry.uv_layers) {
        const auto id = contract::uv_attribute_id(name);
        const auto bytes = float2_bytes(values.size());
        ++plan.source_binding_count;
        add_bytes(plan.source_device_bytes, bytes);
        if (residency.contains(id)) {
            ++plan.resident_binding_count;
            add_bytes(plan.resident_device_bytes, bytes);
        }
    }
}

}// namespace

bool SurfaceAttributeDemand::contains(std::uint64_t id) const noexcept {
    return all || ids.contains(id);
}

void SurfaceAttributeDemand::merge(const SurfaceAttributeDemand &other) {
    if (all || other.all) {
        all = true;
        ids.clear();
        return;
    }
    ids.insert(other.ids.begin(), other.ids.end());
}

SurfaceAttributeDemand collect_surface_attribute_demand(
    const compiler::SurfaceProgram &program,
    const compiler::SurfaceParameterBlock &parameters) {
    SurfaceAttributeDemand result;
    std::vector<ResolutionState> states(
        program.value_instructions().size(), ResolutionState::unvisited);
    std::vector<std::optional<std::uint64_t>> values(
        program.value_instructions().size());
    const auto require = [&](compiler::ValueExpressionId expression) {
        require_expression(
            result, program, parameters, expression, states, values);
    };
    namespace operand = compiler::value_operand;
    for (const auto &instruction : program.value_instructions()) {
        switch (instruction.operation) {
            case compiler::ValueOperation::pointiness:
            case compiler::ValueOperation::sampled_pointiness:
                result.ids.emplace(
                    contract::cycles_pointiness_attribute_id);
                break;
            case compiler::ValueOperation::attribute_color:
            case compiler::ValueOperation::attribute_factor:
            case compiler::ValueOperation::attribute_alpha:
                require(instruction.operand(operand::attribute::id));
                break;
            case compiler::ValueOperation::sampled_attribute_color:
            case compiler::ValueOperation::sampled_attribute_factor:
            case compiler::ValueOperation::sampled_attribute_alpha:
                require(instruction.operand(operand::sampled_attribute::id));
                break;
            case compiler::ValueOperation::uv:
                if (instruction.static_u0 != 0u) {
                    require(instruction.operand(operand::uv::map));
                }
                break;
            case compiler::ValueOperation::sampled_uv:
                if (instruction.static_u0 != 0u) {
                    require(instruction.operand(operand::sampled_uv::map));
                }
                break;
            case compiler::ValueOperation::normal_map:
                if (compiler::normal_map_has_named_tangent(
                        instruction.static_u0)) {
                    require(instruction.operand(operand::normal_map::uv_map));
                }
                break;
            case compiler::ValueOperation::sampled_normal_map:
                if (compiler::normal_map_has_named_tangent(
                        instruction.static_u0)) {
                    require(instruction.operand(
                        operand::sampled_normal_map::uv_map));
                }
                break;
            default:
                break;
        }
    }
    return result;
}

const GeometryAttributeResidency &SceneAttributeResidencyPlan::geometry(
    contract::GeometryId id) const noexcept {
    static const GeometryAttributeResidency conservative{
        .demand = SurfaceAttributeDemand{.all = true}};
    const auto iter = geometries.find(id);
    return iter == geometries.end() ? conservative : iter->second;
}

SceneAttributeResidencyPlan build_scene_attribute_residency_plan(
    const contract::SceneSnapshot &snapshot,
    const compiler::MaterialLibrary &materials) {
    std::map<contract::MaterialId, SurfaceAttributeDemand> material_demands;
    for (const auto &[id, material] : materials.materials()) {
        material_demands.emplace(
            id,
            collect_surface_attribute_demand(
                *material.surface_program(), material.parameters()));
    }

    SceneAttributeResidencyPlan result;
    const auto reachability =
        build_scene_material_reachability(snapshot);
    for (const auto &[geometry_id, geometry] : snapshot.geometries) {
        auto &residency = result.geometries[geometry_id];
        const auto reachable =
            reachability.surface_by_geometry.find(geometry_id);
        if (reachable !=
            reachability.surface_by_geometry.end()) {
            for (const auto material : reachable->second) {
                merge_material(residency.demand, material, material_demands);
            }
        }
        close_tangent_dependencies(residency.demand, geometry);
        measure_geometry(result, geometry, residency);
    }
    for (const auto &[geometry_id, geometry] :
         snapshot.curve_geometries) {
        auto &residency = result.geometries[geometry_id];
        const auto reachable =
            reachability.surface_by_geometry.find(geometry_id);
        if (reachable != reachability.surface_by_geometry.end()) {
            for (const auto material : reachable->second) {
                merge_material(
                    residency.demand, material, material_demands);
            }
        }
        measure_geometry(result, geometry, residency);
    }
    return result;
}

}// namespace psycles::luisa_backend::detail
