#include <psycles/compiler/surface_svm_schedule.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace psycles::compiler {
namespace {

struct Region {
  ClosureExpressionId closure{};
  std::uint32_t parent{surface_svm_invalid_region};
  std::uint32_t depth{};
  std::uint32_t left{surface_svm_invalid_region};
  std::uint32_t right{surface_svm_invalid_region};
  bool dynamic_mix{};
  bool left_active{};
  bool right_active{};
  bool conditional{};
  std::vector<ValueExpressionId> values;
};

struct Builder {
  const SurfaceProgram &program;
  const SurfaceClosurePlan &closure_plan;
  const SurfaceValueDependencyPlan &dependencies;
  SurfaceClosureEndpointMask endpoints{};
  SurfaceSvmSchedulePlan result;
  std::vector<Region> regions;

  Builder(const SurfaceProgram &program_value,
          const SurfaceClosurePlan &closure_plan_value,
          const SurfaceValueDependencyPlan &dependencies_value,
          SurfaceClosureEndpointMask endpoints_value) noexcept
      : program{program_value}, closure_plan{closure_plan_value},
        dependencies{dependencies_value}, endpoints{endpoints_value} {}

  [[nodiscard]] SurfaceSvmSchedulePlan reject(std::string diagnostic) {
    result = {};
    result.diagnostic = std::move(diagnostic);
    return std::move(result);
  }

  [[nodiscard]] bool closure_active(ClosureExpressionId id) const noexcept {
    if (!id.valid() || id.value >= program.closure_instructions().size() ||
        id.value >= dependencies.physical_closures.size() ||
        id.value >= dependencies.emission_closures.size() ||
        !closure_plan.entry(id).reachable) {
      return false;
    }
    const auto physical =
        (endpoints & surface_closure_endpoint_bit(
                         SurfaceClosureEndpoint::physical)) != 0u &&
        dependencies.physical_closures[id.value];
    const auto emission =
        (endpoints & surface_closure_endpoint_bit(
                         SurfaceClosureEndpoint::emission)) != 0u &&
        dependencies.emission_closures[id.value];
    return physical || emission;
  }

  [[nodiscard]] bool value_active(ValueExpressionId id) const noexcept {
    if (!id.valid() || id.value >= dependencies.physical.size() ||
        id.value >= dependencies.emission.size()) {
      return false;
    }
    const auto physical =
        (endpoints & surface_closure_endpoint_bit(
                         SurfaceClosureEndpoint::physical)) != 0u &&
        dependencies.physical[id.value];
    const auto emission =
        (endpoints & surface_closure_endpoint_bit(
                         SurfaceClosureEndpoint::emission)) != 0u &&
        dependencies.emission[id.value];
    return physical || emission;
  }

  [[nodiscard]] bool valid_child(ClosureExpressionId parent,
                                 ClosureExpressionId child) const noexcept {
    return child.valid() &&
           child.value < program.closure_instructions().size() &&
           child.value < parent.value;
  }

  [[nodiscard]] std::uint32_t build_region(ClosureExpressionId closure,
                                           std::uint32_t parent,
                                           std::uint32_t depth,
                                           bool conditional) {
    if (!closure_active(closure)) {
      return surface_svm_invalid_region;
    }
    if (regions.size() >= std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic = "the structured surface SVM has too many regions";
      return surface_svm_invalid_region;
    }
    const auto index = static_cast<std::uint32_t>(regions.size());
    regions.emplace_back(Region{.closure = closure,
                                .parent = parent,
                                .depth = depth,
                                .conditional = conditional,
                                .values = {}});

    const auto &node = program.closure_instructions()[closure.value];
    if (node.operation != ClosureOperation::add &&
        node.operation != ClosureOperation::mix) {
      return index;
    }
    if (!valid_child(closure, node.a) || !valid_child(closure, node.b)) {
      result.diagnostic =
          "a closure-tree edge is not a strict topological predecessor";
      return surface_svm_invalid_region;
    }
    const auto left_active = closure_active(node.a);
    const auto right_active = closure_active(node.b);
    const auto both_reachable = closure_plan.entry(node.a).reachable &&
                                closure_plan.entry(node.b).reachable;
    const auto dynamic_mix = node.operation == ClosureOperation::mix &&
                             both_reachable && (left_active || right_active);
    regions[index].dynamic_mix = dynamic_mix;
    regions[index].left_active = left_active;
    regions[index].right_active = right_active;
    if (left_active) {
      const auto left =
          build_region(node.a, index, depth + 1u, conditional || dynamic_mix);
      if (left == surface_svm_invalid_region) {
        if (result.diagnostic.empty()) {
          result.diagnostic = "an active left closure branch has no region";
        }
        return surface_svm_invalid_region;
      }
      regions[index].left = left;
    }
    if (right_active) {
      const auto right =
          build_region(node.b, index, depth + 1u, conditional || dynamic_mix);
      if (right == surface_svm_invalid_region) {
        if (result.diagnostic.empty()) {
          result.diagnostic = "an active right closure branch has no region";
        }
        return surface_svm_invalid_region;
      }
      regions[index].right = right;
    }
    return index;
  }

  [[nodiscard]] bool is_ancestor(std::uint32_t ancestor,
                                 std::uint32_t region) const noexcept {
    if (ancestor >= regions.size() || region >= regions.size()) {
      return false;
    }
    while (regions[region].depth > regions[ancestor].depth) {
      region = regions[region].parent;
    }
    return ancestor == region;
  }

  [[nodiscard]] std::uint32_t lca(std::uint32_t a,
                                  std::uint32_t b) const noexcept {
    if (a == surface_svm_invalid_region) {
      return b;
    }
    if (b == surface_svm_invalid_region) {
      return a;
    }
    while (regions[a].depth > regions[b].depth) {
      a = regions[a].parent;
    }
    while (regions[b].depth > regions[a].depth) {
      b = regions[b].parent;
    }
    while (a != b) {
      a = regions[a].parent;
      b = regions[b].parent;
    }
    return a;
  }

  void add_value_use(ValueExpressionId value, std::uint32_t region) {
    if (!value.valid() || !value_active(value)) {
      return;
    }
    if (value.value >= result.value_regions.size()) {
      result.diagnostic = "a closure references a value outside the program";
      return;
    }
    result.value_regions[value.value] =
        lca(result.value_regions[value.value], region);
  }

  void collect_direct_uses(std::uint32_t region_index) {
    if (!result.diagnostic.empty()) {
      return;
    }
    const auto &region = regions[region_index];
    const auto &closure = program.closure_instructions()[region.closure.value];
    if (region.dynamic_mix) {
      add_value_use(closure.factor, region_index);
    }
    if (closure.operation == ClosureOperation::add ||
        closure.operation == ClosureOperation::mix) {
      if (region.left != surface_svm_invalid_region) {
        collect_direct_uses(region.left);
      }
      if (region.right != surface_svm_invalid_region) {
        collect_direct_uses(region.right);
      }
      return;
    }

    const auto operands = surface_closure_operands(closure);
    if (operands.size() != surface_closure_operand_count(closure.operation)) {
      result.diagnostic =
          "an active closure leaf has an inconsistent operand ABI";
      return;
    }
    for (const auto operand : operands) {
      // Dependency analysis is the authoritative capability projection. It
      // removes feature-disabled Principled, anisotropy, Thin Film and Hair
      // operands while preserving linked runtime values conservatively.
      add_value_use(operand, region_index);
    }
  }

  void emit(SurfaceSvmScheduleInstructionKind kind, std::uint32_t source,
            std::uint32_t target = 0u) {
    result.instructions.emplace_back(
        SurfaceSvmScheduleInstruction{kind, source, target});
  }

  void emit_region(std::uint32_t region_index) {
    auto &region = regions[region_index];
    for (const auto value : region.values) {
      emit(SurfaceSvmScheduleInstructionKind::value, value.value);
      ++result.value_instruction_count;
      if (region.conditional) {
        ++result.conditional_value_instruction_count;
      }
    }
    const auto &closure = program.closure_instructions()[region.closure.value];
    if (region.dynamic_mix) {
      emit(SurfaceSvmScheduleInstructionKind::mix_closure,
           region.closure.value);
      ++result.mix_instruction_count;
      if (region.left != surface_svm_invalid_region) {
        const auto guard = result.instructions.size();
        emit(SurfaceSvmScheduleInstructionKind::jump_if_one,
             region.closure.value);
        emit_region(region.left);
        result.instructions[guard].target =
            static_cast<std::uint32_t>(result.instructions.size());
        ++result.conditional_branch_count;
      }
      if (region.right != surface_svm_invalid_region) {
        const auto guard = result.instructions.size();
        emit(SurfaceSvmScheduleInstructionKind::jump_if_zero,
             region.closure.value);
        emit_region(region.right);
        result.instructions[guard].target =
            static_cast<std::uint32_t>(result.instructions.size());
        ++result.conditional_branch_count;
      }
      return;
    }
    if (closure.operation == ClosureOperation::add ||
        closure.operation == ClosureOperation::mix) {
      if (region.left != surface_svm_invalid_region) {
        emit_region(region.left);
      }
      if (region.right != surface_svm_invalid_region) {
        emit_region(region.right);
      }
      return;
    }
    if (closure.operation != ClosureOperation::null_closure) {
      emit(SurfaceSvmScheduleInstructionKind::closure_leaf,
           region.closure.value);
      ++result.closure_leaf_count;
    }
  }

  [[nodiscard]] SurfaceSvmSchedulePlan build() {
    if (!closure_plan.compatible(program) ||
        !dependencies.compatible(program)) {
      return reject(
          "surface SVM schedule inputs are not derived from one program");
    }
    constexpr auto valid_endpoints =
        surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical) |
        surface_closure_endpoint_bit(SurfaceClosureEndpoint::emission);
    if (endpoints == 0u || (endpoints & ~valid_endpoints) != 0u) {
      return reject("surface SVM endpoint projection is empty or invalid");
    }
    result.value_regions.assign(program.value_instructions().size(),
                                surface_svm_invalid_region);
    const auto root = program.root();
    if (!root.valid()) {
      emit(SurfaceSvmScheduleInstructionKind::end, ~std::uint32_t{0u});
      result.valid = true;
      return std::move(result);
    }
    if (root.value >= program.closure_instructions().size()) {
      return reject("the surface closure root is outside the program");
    }
    const auto root_region =
        build_region(root, surface_svm_invalid_region, 0u, false);
    if (!result.diagnostic.empty()) {
      return reject(std::move(result.diagnostic));
    }
    if (root_region == surface_svm_invalid_region) {
      emit(SurfaceSvmScheduleInstructionKind::end, ~std::uint32_t{0u});
      result.valid = true;
      return std::move(result);
    }

    collect_direct_uses(root_region);
    if (!result.diagnostic.empty()) {
      return reject(std::move(result.diagnostic));
    }

    const auto &values = program.value_instructions();
    // Reverse topological propagation computes the meet-over-all-uses
    // placement. If v is placed in R, every operand of v must dominate R;
    // joining R into each operand and iterating in reverse order is exact
    // because every edge points to a strictly earlier ValueExpressionId.
    for (auto index = values.size(); index-- > 0u;) {
      const auto region = result.value_regions[index];
      if (region == surface_svm_invalid_region) {
        continue;
      }
      const auto &value = values[index];
      for (const auto operand : value.operands) {
        if (!operand.valid() || operand.value >= index) {
          return reject("the value graph is not a strict topological DAG");
        }
        result.value_regions[operand.value] =
            lca(result.value_regions[operand.value], region);
      }
    }

    for (auto index = std::size_t{}; index < values.size(); ++index) {
      const auto region = result.value_regions[index];
      if (region == surface_svm_invalid_region) {
        continue;
      }
      const auto &value = values[index];
      for (const auto operand : value.operands) {
        const auto operand_region = result.value_regions[operand.value];
        if (operand_region == surface_svm_invalid_region ||
            !is_ancestor(operand_region, region)) {
          return reject("a scheduled value operand does not dominate its user");
        }
      }
      if (value.operation != ValueOperation::parameter) {
        regions[region].values.emplace_back(
            ValueExpressionId{static_cast<std::uint32_t>(index)});
      }
    }

    result.region_count = static_cast<std::uint32_t>(regions.size());
    emit_region(root_region);
    emit(SurfaceSvmScheduleInstructionKind::end, ~std::uint32_t{0u});
    result.valid = true;
    return std::move(result);
  }
};

} // namespace

SurfaceSvmSchedulePlan
plan_surface_svm_schedule(const SurfaceProgram &program,
                          const SurfaceClosurePlan &closure_plan,
                          const SurfaceValueDependencyPlan &dependencies,
                          SurfaceClosureEndpointMask endpoints) {
  return Builder{program, closure_plan, dependencies, endpoints}.build();
}

} // namespace psycles::compiler
