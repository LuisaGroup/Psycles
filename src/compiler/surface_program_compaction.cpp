#include "surface_program_compaction.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace psycles::compiler::detail {
namespace {

constexpr auto closure_dependencies = std::array{
    &ClosureInstruction::a,
    &ClosureInstruction::b};

constexpr auto volume_value_dependencies = std::array{
    &VolumeInstruction::color,
    &VolumeInstruction::density,
    &VolumeInstruction::anisotropy,
    &VolumeInstruction::ior,
    &VolumeInstruction::backscatter,
    &VolumeInstruction::alpha,
    &VolumeInstruction::diameter,
    &VolumeInstruction::scatter_coefficients,
    &VolumeInstruction::absorption_coefficients,
    &VolumeInstruction::absorption_color,
    &VolumeInstruction::emission_coefficients,
    &VolumeInstruction::emission_strength,
    &VolumeInstruction::emission_color,
    &VolumeInstruction::blackbody_intensity,
    &VolumeInstruction::blackbody_tint,
    &VolumeInstruction::temperature,
    &VolumeInstruction::factor};

constexpr auto volume_dependencies = std::array{
    &VolumeInstruction::a,
    &VolumeInstruction::b};

template<typename Id>
void enqueue(
    Id id,
    std::vector<bool> &reachable,
    std::vector<Id> &worklist) {
    if (!id.valid() || id.value >= reachable.size() ||
        reachable[id.value]) {
        return;
    }
    reachable[id.value] = true;
    worklist.emplace_back(id);
}

template<typename Id>
[[nodiscard]] Id remap(
    Id id,
    const std::vector<Id> &mapping) noexcept {
    return id.valid() && id.value < mapping.size()
               ? mapping[id.value]
               : Id{};
}

template<typename Item, typename Id>
[[nodiscard]] std::vector<Item> compact(
    std::vector<Item> source,
    const std::vector<bool> &reachable,
    std::vector<Id> &mapping) {
    std::vector<Item> result;
    result.reserve(source.size());
    for (std::size_t index = 0u; index < source.size(); ++index) {
        if (!reachable[index]) {
            continue;
        }
        mapping[index] = Id{
            static_cast<std::uint32_t>(result.size())};
        result.emplace_back(std::move(source[index]));
    }
    result.shrink_to_fit();
    return result;
}

}// namespace

SurfaceProgramStorage compact_surface_program(
    SurfaceProgramStorage storage) {
    std::vector<bool> reachable_values(storage.values.size(), false);
    std::vector<bool> reachable_closures(storage.closures.size(), false);
    std::vector<bool> reachable_volumes(storage.volumes.size(), false);
    std::vector<bool> reachable_parameters(
        storage.parameters.size(), false);

    std::vector<ValueExpressionId> value_worklist;
    std::vector<ClosureExpressionId> closure_worklist;
    std::vector<VolumeExpressionId> volume_worklist;
    enqueue(storage.root, reachable_closures, closure_worklist);
    enqueue(storage.volume_root, reachable_volumes, volume_worklist);
    enqueue(
        storage.surface_normal_root,
        reachable_values,
        value_worklist);
    enqueue(
        storage.displacement_root,
        reachable_values,
        value_worklist);

    while (!closure_worklist.empty()) {
        const auto id = closure_worklist.back();
        closure_worklist.pop_back();
        const auto &instruction = storage.closures[id.value];
        for (const auto dependency : closure_value_dependency_members) {
            enqueue(
                instruction.*dependency,
                reachable_values,
                value_worklist);
        }
        for (const auto dependency : closure_dependencies) {
            enqueue(
                instruction.*dependency,
                reachable_closures,
                closure_worklist);
        }
    }

    while (!volume_worklist.empty()) {
        const auto id = volume_worklist.back();
        volume_worklist.pop_back();
        const auto &instruction = storage.volumes[id.value];
        for (const auto dependency : volume_value_dependencies) {
            enqueue(
                instruction.*dependency,
                reachable_values,
                value_worklist);
        }
        for (const auto dependency : volume_dependencies) {
            enqueue(
                instruction.*dependency,
                reachable_volumes,
                volume_worklist);
        }
    }

    while (!value_worklist.empty()) {
        const auto id = value_worklist.back();
        value_worklist.pop_back();
        const auto &instruction = storage.values[id.value];
        if (instruction.parameter.valid() &&
            instruction.parameter.value < reachable_parameters.size()) {
            reachable_parameters[instruction.parameter.value] = true;
        }
        for (const auto dependency : instruction.operands) {
            enqueue(
                dependency,
                reachable_values,
                value_worklist);
        }
    }

    std::vector<ParameterId> parameter_mapping(
        storage.parameters.size());
    std::vector<ValueExpressionId> value_mapping(
        storage.values.size());
    std::vector<ClosureExpressionId> closure_mapping(
        storage.closures.size());
    std::vector<VolumeExpressionId> volume_mapping(
        storage.volumes.size());

    storage.parameters = compact(
        std::move(storage.parameters),
        reachable_parameters,
        parameter_mapping);
    storage.values = compact(
        std::move(storage.values),
        reachable_values,
        value_mapping);
    storage.closures = compact(
        std::move(storage.closures),
        reachable_closures,
        closure_mapping);
    storage.volumes = compact(
        std::move(storage.volumes),
        reachable_volumes,
        volume_mapping);

    for (std::size_t index = 0u;
         index < storage.parameters.size();
         ++index) {
        storage.parameters[index].id = ParameterId{
            static_cast<std::uint32_t>(index)};
    }
    for (auto &instruction : storage.values) {
        instruction.parameter = remap(
            instruction.parameter, parameter_mapping);
        for (auto &dependency : instruction.operands) {
            dependency = remap(dependency, value_mapping);
        }
    }
    for (auto &instruction : storage.closures) {
        for (const auto dependency : closure_value_dependency_members) {
            instruction.*dependency = remap(
                instruction.*dependency, value_mapping);
        }
        for (const auto dependency : closure_dependencies) {
            instruction.*dependency = remap(
                instruction.*dependency, closure_mapping);
        }
    }
    for (auto &instruction : storage.volumes) {
        for (const auto dependency : volume_value_dependencies) {
            instruction.*dependency = remap(
                instruction.*dependency, value_mapping);
        }
        for (const auto dependency : volume_dependencies) {
            instruction.*dependency = remap(
                instruction.*dependency, volume_mapping);
        }
    }
    storage.root = remap(storage.root, closure_mapping);
    storage.volume_root = remap(
        storage.volume_root, volume_mapping);
    storage.surface_normal_root = remap(
        storage.surface_normal_root, value_mapping);
    storage.displacement_root = remap(
        storage.displacement_root, value_mapping);
    return storage;
}

}// namespace psycles::compiler::detail
