#include <psycles/compiler/surface_svm_schedule.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

using ValueSet = std::vector<std::uint32_t>;
inline constexpr auto invalid_index =
    std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] constexpr std::size_t bank_index(SurfaceValueBank bank) noexcept {
  return static_cast<std::size_t>(bank);
}

[[nodiscard]] bool contains(const ValueSet &set, std::uint32_t value) noexcept {
  return std::binary_search(set.begin(), set.end(), value);
}

void insert(ValueSet &set, std::uint32_t value) {
  const auto position = std::lower_bound(set.begin(), set.end(), value);
  if (position == set.end() || *position != value) {
    set.insert(position, value);
  }
}

void erase(ValueSet &set, std::uint32_t value) {
  const auto position = std::lower_bound(set.begin(), set.end(), value);
  if (position != set.end() && *position == value) {
    set.erase(position);
  }
}

[[nodiscard]] ValueSet set_union(const ValueSet &a, const ValueSet &b) {
  ValueSet result;
  result.reserve(a.size() + b.size());
  std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                 std::back_inserter(result));
  return result;
}

[[nodiscard]] ValueSet set_intersection(const ValueSet &a, const ValueSet &b) {
  ValueSet result;
  result.reserve(std::min(a.size(), b.size()));
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(result));
  return result;
}

struct ProgramPoint {
  std::uint32_t definition{invalid_index};
  ValueSet uses;
  std::vector<std::uint32_t> successors;
};

struct Builder {
  const SurfaceProgram &program;
  const SurfaceSvmSchedulePlan &schedule;
  SurfaceValueStorageCapacity capacity;
  SurfaceSvmStoragePlan result;
  std::vector<ProgramPoint> points;
  std::vector<ValueSet> adjacency;
  std::array<std::vector<std::uint32_t>, 3u> vertices;

  Builder(const SurfaceProgram &program_value,
          const SurfaceSvmSchedulePlan &schedule_value,
          SurfaceValueStorageCapacity capacity_value) noexcept
      : program{program_value}, schedule{schedule_value},
        capacity{capacity_value} {}

  [[nodiscard]] SurfaceSvmStoragePlan reject(std::string diagnostic) {
    SurfaceSvmStoragePlan rejected;
    rejected.diagnostic = std::move(diagnostic);
    return rejected;
  }

  [[nodiscard]] bool active(std::uint32_t value) const noexcept {
    return value < schedule.value_regions.size() &&
           schedule.value_regions[value] != surface_svm_invalid_region;
  }

  [[nodiscard]] bool initialize_quotient() {
    const auto &values = program.value_instructions();
    result.locations.resize(values.size());
    result.representatives.assign(values.size(), invalid_index);
    adjacency.resize(values.size());

    for (auto index = std::size_t{}; index < values.size(); ++index) {
      if (!active(static_cast<std::uint32_t>(index))) {
        continue;
      }
      if (result.active_values == std::numeric_limits<std::uint32_t>::max()) {
        result.diagnostic = "the structured SVM active-value count overflows";
        return false;
      }
      ++result.active_values;
      const auto &instruction = values[index];
      auto bank = SurfaceValueBank::scalar;
      if (!classify_surface_value_type(instruction.result_type, bank)) {
        result.diagnostic =
            "an active structured-SVM value has no physical bank";
        return false;
      }
      for (const auto operand : instruction.operands) {
        if (!operand.valid() || operand.value >= index) {
          result.diagnostic =
              "the structured-SVM value graph is not a strict DAG";
          return false;
        }
        if (!active(operand.value)) {
          result.diagnostic =
              "the structured-SVM active values are not transitively closed";
          return false;
        }
      }

      if (instruction.operation == ValueOperation::parameter) {
        if (!instruction.parameter.valid() ||
            instruction.parameter.value >= program.parameters().size()) {
          result.diagnostic =
              "an active structured-SVM parameter has no binding";
          return false;
        }
        const auto &parameter =
            program.parameters()[instruction.parameter.value];
        if (parameter.id != instruction.parameter ||
            parameter.type != instruction.result_type) {
          result.diagnostic =
              "an active structured-SVM parameter has an inconsistent type";
          return false;
        }
        result.representatives[index] = static_cast<std::uint32_t>(index);
        result.locations[index] = {.storage =
                                       SurfaceValueStorageClass::parameter,
                                   .bank = bank,
                                   .index = instruction.parameter.value};
        ++result.parameter_values;
        continue;
      }

      if (instruction.operation == ValueOperation::passthrough) {
        if (instruction.operands.size() != 1u) {
          result.diagnostic =
              "a structured-SVM Passthrough does not have one operand";
          return false;
        }
        const auto source = instruction.operands.front().value;
        const auto representative = result.representatives[source];
        if (representative == invalid_index) {
          result.diagnostic =
              "a structured-SVM Passthrough source has no representative";
          return false;
        }
        auto source_bank = SurfaceValueBank::scalar;
        if (!classify_surface_value_type(values[representative].result_type,
                                         source_bank) ||
            source_bank != bank) {
          result.diagnostic =
              "a structured-SVM Passthrough crosses physical banks";
          return false;
        }
        result.representatives[index] = representative;
        ++result.alias_values;
        continue;
      }

      result.representatives[index] = static_cast<std::uint32_t>(index);
      vertices[bank_index(bank)].emplace_back(
          static_cast<std::uint32_t>(index));
      ++result.local_values;
    }
    return result.active_values ==
           result.parameter_values + result.local_values + result.alias_values;
  }

  [[nodiscard]] bool add_use(ProgramPoint &point, ValueExpressionId value) {
    if (!value.valid() || value.value >= result.representatives.size() ||
        !active(value.value)) {
      result.diagnostic =
          "a structured-SVM instruction references an inactive value";
      return false;
    }
    const auto representative = result.representatives[value.value];
    if (representative == invalid_index) {
      result.diagnostic = "a structured-SVM use has no quotient representative";
      return false;
    }
    if (result.locations[representative].storage !=
        SurfaceValueStorageClass::parameter) {
      insert(point.uses, representative);
    }
    return true;
  }

  [[nodiscard]] bool add_closure_factor_use(ProgramPoint &point,
                                            std::uint32_t closure_index) {
    const auto &closures = program.closure_instructions();
    if (closure_index >= closures.size() ||
        closures[closure_index].operation != ClosureOperation::mix) {
      result.diagnostic = "a structured-SVM control record does not name a Mix";
      return false;
    }
    return add_use(point, closures[closure_index].factor);
  }

  [[nodiscard]] bool make_program_points() {
    const auto &values = program.value_instructions();
    const auto &closures = program.closure_instructions();
    points.resize(schedule.instructions.size());
    std::vector<std::uint32_t> definition_counts(values.size(), 0u);
    auto end_count = std::uint32_t{};

    for (auto index = std::size_t{}; index < schedule.instructions.size();
         ++index) {
      const auto &instruction = schedule.instructions[index];
      auto &point = points[index];
      switch (instruction.kind) {
      case SurfaceSvmScheduleInstructionKind::value: {
        if (instruction.source >= values.size() ||
            !active(instruction.source) ||
            values[instruction.source].operation == ValueOperation::parameter) {
          result.diagnostic =
              "a structured-SVM value record has an invalid source";
          return false;
        }
        const auto representative = result.representatives[instruction.source];
        if (representative == instruction.source) {
          point.definition = representative;
          if (++definition_counts[representative] != 1u) {
            result.diagnostic =
                "a structured-SVM local value has multiple definitions";
            return false;
          }
          for (const auto operand : values[instruction.source].operands) {
            if (!add_use(point, operand)) {
              return false;
            }
          }
        } else if (values[instruction.source].operation !=
                   ValueOperation::passthrough) {
          result.diagnostic =
              "a non-Passthrough structured-SVM value was aliased";
          return false;
        }
        break;
      }
      case SurfaceSvmScheduleInstructionKind::mix_closure:
      case SurfaceSvmScheduleInstructionKind::jump_if_one:
      case SurfaceSvmScheduleInstructionKind::jump_if_zero:
        if (!add_closure_factor_use(point, instruction.source)) {
          return false;
        }
        break;
      case SurfaceSvmScheduleInstructionKind::closure_leaf: {
        if (instruction.source >= closures.size()) {
          result.diagnostic =
              "a structured-SVM leaf is outside the closure program";
          return false;
        }
        const auto &closure = closures[instruction.source];
        if (closure.operation == ClosureOperation::add ||
            closure.operation == ClosureOperation::mix ||
            closure.operation == ClosureOperation::null_closure) {
          result.diagnostic = "a structured-SVM leaf names a control closure";
          return false;
        }
        const auto operands = surface_closure_operands(closure);
        if (operands.size() !=
            surface_closure_operand_count(closure.operation)) {
          result.diagnostic =
              "a structured-SVM leaf has an inconsistent operand ABI";
          return false;
        }
        for (const auto operand : operands) {
          if (operand.valid() && active(operand.value) &&
              !add_use(point, operand)) {
            return false;
          }
        }
        break;
      }
      case SurfaceSvmScheduleInstructionKind::end:
        ++end_count;
        if (index + 1u != schedule.instructions.size()) {
          result.diagnostic =
              "structured-SVM End is not the final program point";
          return false;
        }
        break;
      }

      if (instruction.kind == SurfaceSvmScheduleInstructionKind::end) {
        continue;
      }
      if (index + 1u >= schedule.instructions.size()) {
        result.diagnostic = "the structured SVM has no terminal record";
        return false;
      }
      if (instruction.kind == SurfaceSvmScheduleInstructionKind::jump_if_one ||
          instruction.kind == SurfaceSvmScheduleInstructionKind::jump_if_zero) {
        if (instruction.target <= index ||
            instruction.target >= schedule.instructions.size()) {
          result.diagnostic =
              "a structured-SVM guard is not a forward CFG edge";
          return false;
        }
        point.successors.emplace_back(static_cast<std::uint32_t>(index + 1u));
        if (instruction.target != index + 1u) {
          point.successors.emplace_back(instruction.target);
        }
      } else {
        point.successors.emplace_back(static_cast<std::uint32_t>(index + 1u));
      }
    }
    if (end_count != 1u) {
      result.diagnostic =
          "the structured SVM does not have one final End record";
      return false;
    }
    for (const auto vertex_bank :
         {SurfaceValueBank::scalar, SurfaceValueBank::vector,
          SurfaceValueBank::unsigned_integer}) {
      for (const auto value : vertices[bank_index(vertex_bank)]) {
        if (definition_counts[value] != 1u) {
          result.diagnostic =
              "an active structured-SVM local has no unique definition";
          return false;
        }
      }
    }
    return true;
  }

  [[nodiscard]] bool prove_definite_assignment() {
    std::vector<std::vector<std::uint32_t>> predecessors(points.size());
    for (auto index = std::size_t{}; index < points.size(); ++index) {
      for (const auto successor : points[index].successors) {
        predecessors[successor].emplace_back(static_cast<std::uint32_t>(index));
      }
    }
    std::vector<ValueSet> defined_out(points.size());
    for (auto index = std::size_t{}; index < points.size(); ++index) {
      ValueSet defined;
      if (index != 0u) {
        if (predecessors[index].empty()) {
          result.diagnostic =
              "the structured SVM contains an unreachable program point";
          return false;
        }
        defined = defined_out[predecessors[index].front()];
        for (auto predecessor_index = std::size_t{1u};
             predecessor_index < predecessors[index].size();
             ++predecessor_index) {
          defined = set_intersection(
              defined, defined_out[predecessors[index][predecessor_index]]);
        }
      }
      for (const auto use : points[index].uses) {
        if (!contains(defined, use)) {
          result.diagnostic =
              "a structured-SVM local definition does not dominate its use";
          return false;
        }
      }
      const auto definition = points[index].definition;
      if (definition != invalid_index) {
        if (contains(defined, definition)) {
          result.diagnostic =
              "a structured-SVM local is redefined on one CFG path";
          return false;
        }
        insert(defined, definition);
      }
      defined_out[index] = std::move(defined);
    }
    return true;
  }

  void add_interference(std::uint32_t a, std::uint32_t b) {
    if (a == b) {
      return;
    }
    auto bank_a = SurfaceValueBank::scalar;
    auto bank_b = SurfaceValueBank::scalar;
    const auto &values = program.value_instructions();
    if (!classify_surface_value_type(values[a].result_type, bank_a) ||
        !classify_surface_value_type(values[b].result_type, bank_b) ||
        bank_a != bank_b) {
      return;
    }
    adjacency[a].emplace_back(b);
    adjacency[b].emplace_back(a);
  }

  [[nodiscard]] bool solve_liveness() {
    std::vector<std::uint32_t> use_counts(program.value_instructions().size(),
                                          0u);
    for (const auto &point : points) {
      for (const auto use : point.uses) {
        if (use_counts[use] == std::numeric_limits<std::uint32_t>::max()) {
          result.diagnostic = "a structured-SVM local use count overflows";
          return false;
        }
        ++use_counts[use];
      }
    }
    for (const auto &bank_vertices : vertices) {
      for (const auto value : bank_vertices) {
        if (use_counts[value] == 0u) {
          result.diagnostic = "an active structured-SVM local has no consumer";
          return false;
        }
      }
    }

    std::vector<ValueSet> live_in(points.size());
    for (auto index = points.size(); index-- > 0u;) {
      ValueSet live_out;
      for (const auto successor : points[index].successors) {
        live_out = set_union(live_out, live_in[successor]);
      }
      const auto definition = points[index].definition;
      if (definition != invalid_index) {
        if (!contains(live_out, definition)) {
          result.diagnostic =
              "a structured-SVM definition is not live after its write";
          return false;
        }
        for (const auto live : live_out) {
          add_interference(definition, live);
        }
        erase(live_out, definition);
      }
      live_in[index] = set_union(live_out, points[index].uses);
    }

    auto adjacency_extent = std::uint64_t{};
    for (auto &neighbors : adjacency) {
      std::ranges::sort(neighbors);
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                      neighbors.end());
      adjacency_extent += neighbors.size();
    }
    if ((adjacency_extent & 1u) != 0u) {
      result.diagnostic =
          "the structured-SVM interference relation is not symmetric";
      return false;
    }
    result.interference_edges = adjacency_extent / 2u;
    return true;
  }

  [[nodiscard]] bool adjacent(std::uint32_t a, std::uint32_t b) const noexcept {
    return std::binary_search(adjacency[a].begin(), adjacency[a].end(), b);
  }

  [[nodiscard]] bool color_bank(SurfaceValueBank bank,
                                std::uint32_t slot_capacity,
                                std::uint32_t &slot_count) {
    const auto typed_index = bank_index(bank);
    const auto &bank_vertices = vertices[typed_index];
    if (bank_vertices.empty()) {
      slot_count = 0u;
      result.maximum_interference_clique[typed_index] = 0u;
      return true;
    }

    const auto value_count = result.locations.size();
    std::vector<std::uint32_t> labels(value_count, 0u);
    std::vector<bool> numbered(value_count, false);
    std::vector<std::uint32_t> peo(bank_vertices.size());
    for (auto number = bank_vertices.size(); number-- > 0u;) {
      auto selected = invalid_index;
      auto selected_label = std::uint32_t{};
      for (const auto candidate : bank_vertices) {
        if (numbered[candidate] ||
            (selected != invalid_index &&
             (labels[candidate] < selected_label ||
              (labels[candidate] == selected_label && candidate > selected)))) {
          continue;
        }
        selected = candidate;
        selected_label = labels[candidate];
      }
      if (selected == invalid_index) {
        result.diagnostic =
            "maximum-cardinality search lost an interference vertex";
        return false;
      }
      peo[number] = selected;
      numbered[selected] = true;
      for (const auto neighbor : adjacency[selected]) {
        if (!numbered[neighbor]) {
          ++labels[neighbor];
        }
      }
    }

    std::vector<std::uint32_t> position(value_count, invalid_index);
    for (auto index = std::size_t{}; index < peo.size(); ++index) {
      position[peo[index]] = static_cast<std::uint32_t>(index);
    }

    auto maximum_clique = std::uint32_t{1u};
    for (auto index = std::size_t{}; index < peo.size(); ++index) {
      const auto vertex = peo[index];
      auto parent = invalid_index;
      auto later_count = std::uint32_t{};
      for (const auto neighbor : adjacency[vertex]) {
        if (position[neighbor] <= index) {
          continue;
        }
        ++later_count;
        if (parent == invalid_index ||
            position[neighbor] < position[parent]) {
          parent = neighbor;
        }
      }
      maximum_clique = std::max(maximum_clique, later_count + 1u);
      if (parent == invalid_index) {
        continue;
      }
      for (const auto neighbor : adjacency[vertex]) {
        if (neighbor != parent && position[neighbor] > index &&
            !adjacent(parent, neighbor)) {
          result.diagnostic =
              "the structured-SVM SSA interference graph is not chordal";
          return false;
        }
      }
    }

    std::vector<std::uint32_t> colors(value_count, invalid_index);
    std::vector<bool> forbidden(bank_vertices.size(), false);
    auto used_colors = std::uint32_t{};
    for (auto order = peo.size(); order-- > 0u;) {
      std::fill(forbidden.begin(), forbidden.end(), false);
      const auto vertex = peo[order];
      for (const auto neighbor : adjacency[vertex]) {
        const auto color = colors[neighbor];
        if (color != invalid_index) {
          forbidden[color] = true;
        }
      }
      auto color = std::uint32_t{};
      while (color < forbidden.size() && forbidden[color]) {
        ++color;
      }
      if (color == forbidden.size()) {
        result.diagnostic =
            "the structured-SVM interference coloring exhausted its domain";
        return false;
      }
      colors[vertex] = color;
      used_colors = std::max(used_colors, color + 1u);
    }
    if (used_colors != maximum_clique) {
      result.diagnostic = "the verified chordal coloring is not clique-optimal";
      return false;
    }
    if (used_colors > slot_capacity) {
      result.diagnostic =
          "the structured-SVM typed storage capacity is insufficient";
      return false;
    }
    for (const auto vertex : bank_vertices) {
      result.locations[vertex] = {.storage =
                                      SurfaceValueStorageClass::local_slot,
                                  .bank = bank,
                                  .index = colors[vertex]};
    }
    slot_count = used_colors;
    result.maximum_interference_clique[typed_index] = maximum_clique;
    return true;
  }

  [[nodiscard]] bool publish_aliases() {
    const auto &values = program.value_instructions();
    for (auto index = std::size_t{}; index < values.size(); ++index) {
      if (!active(static_cast<std::uint32_t>(index))) {
        continue;
      }
      const auto representative = result.representatives[index];
      if (representative == invalid_index) {
        result.diagnostic =
            "an active structured-SVM value has no representative";
        return false;
      }
      if (representative != index) {
        const auto location = result.locations[representative];
        if (location.storage == SurfaceValueStorageClass::inactive) {
          result.diagnostic =
              "a structured-SVM alias representative has no location";
          return false;
        }
        result.locations[index] = location;
      }
    }
    return true;
  }

  [[nodiscard]] SurfaceSvmStoragePlan build() {
    if (!schedule.valid ||
        schedule.value_regions.size() != program.value_instructions().size()) {
      return reject("cannot color an incompatible structured-SVM schedule");
    }
    if (!initialize_quotient() || !make_program_points() ||
        !prove_definite_assignment() || !solve_liveness()) {
      return reject(std::move(result.diagnostic));
    }
    if (!color_bank(SurfaceValueBank::scalar, capacity.scalar_slots,
                    result.scalar_slots) ||
        !color_bank(SurfaceValueBank::vector, capacity.vector_slots,
                    result.vector_slots) ||
        !color_bank(SurfaceValueBank::unsigned_integer,
                    capacity.unsigned_integer_slots,
                    result.unsigned_integer_slots) ||
        !publish_aliases()) {
      return reject(std::move(result.diagnostic));
    }
    result.valid = true;
    return std::move(result);
  }
};

} // namespace

bool SurfaceSvmStoragePlan::compatible(
    const SurfaceProgram &program,
    const SurfaceSvmSchedulePlan &schedule) const noexcept {
  return valid && schedule.valid &&
         locations.size() == program.value_instructions().size() &&
         representatives.size() == locations.size() &&
         schedule.value_regions.size() == locations.size() &&
         active_values == parameter_values + local_values + alias_values;
}

std::size_t SurfaceSvmStoragePlan::payload_bytes() const noexcept {
  return static_cast<std::size_t>(scalar_slots) * sizeof(float) +
         static_cast<std::size_t>(vector_slots) * sizeof(float) * 3u +
         static_cast<std::size_t>(unsigned_integer_slots) *
             sizeof(std::uint64_t);
}

SurfaceSvmStoragePlan
plan_surface_svm_storage(const SurfaceProgram &program,
                         const SurfaceSvmSchedulePlan &schedule,
                         SurfaceValueStorageCapacity capacity) {
  return Builder{program, schedule, capacity}.build();
}

} // namespace psycles::compiler
