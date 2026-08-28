#include <psycles/compiler/surface_svm_schedule.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

struct WeightOperation {
  SurfaceSvmWeightId primary;
  SurfaceSvmWeightId secondary;
};

struct Region {
  ClosureExpressionId closure{};
  SurfaceSvmWeightId incoming_weight;
  std::uint32_t parent{surface_svm_invalid_region};
  std::uint32_t depth{};
  std::uint32_t left{surface_svm_invalid_region};
  std::uint32_t right{surface_svm_invalid_region};
  bool dynamic_mix{};
  bool left_effect{};
  bool right_effect{};
  bool subtree_effect{};
  std::vector<ValueExpressionId> values;
  std::vector<WeightOperation> weight_operations;
  std::vector<ClosureExpressionId> leaves;
};

struct LeafAggregate {
  ClosureExpressionId closure{};
  std::uint32_t region{surface_svm_invalid_region};
  std::vector<SurfaceSvmWeightId> incoming_weights;
  SurfaceSvmWeightId final_weight;
};

struct Builder {
  const SurfaceProgram &program;
  const SurfaceClosurePlan &closure_plan;
  const SurfaceValueDependencyPlan &dependencies;
  SurfaceClosureEndpointMask endpoints{};
  SurfaceSvmSchedulePlan result;
  std::vector<Region> regions;
  std::vector<LeafAggregate> leaf_aggregates;
  std::vector<bool> leaf_seen;
  std::vector<ClosureExpressionId> leaf_order;

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
    return ((endpoints & surface_closure_endpoint_bit(
                             SurfaceClosureEndpoint::physical)) != 0u &&
            dependencies.physical_closures[id.value]) ||
           ((endpoints & surface_closure_endpoint_bit(
                             SurfaceClosureEndpoint::emission)) != 0u &&
            dependencies.emission_closures[id.value]);
  }

  [[nodiscard]] bool value_active(ValueExpressionId id) const noexcept {
    if (!id.valid() || id.value >= dependencies.physical.size() ||
        id.value >= dependencies.emission.size()) {
      return false;
    }
    return ((endpoints & surface_closure_endpoint_bit(
                             SurfaceClosureEndpoint::physical)) != 0u &&
            dependencies.physical[id.value]) ||
           ((endpoints & surface_closure_endpoint_bit(
                             SurfaceClosureEndpoint::emission)) != 0u &&
            dependencies.emission[id.value]);
  }

  [[nodiscard]] bool valid_child(ClosureExpressionId parent,
                                 ClosureExpressionId child) const noexcept {
    return child.valid() &&
           child.value < program.closure_instructions().size() &&
           child.value < parent.value;
  }

  [[nodiscard]] SurfaceSvmWeightId append_weight(
      SurfaceSvmWeightExpression expression) {
    if (result.weight_expressions.size() >=
        std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic =
          "the structured surface SVM has too many weight expressions";
      return {};
    }
    const auto id = SurfaceSvmWeightId{static_cast<std::uint32_t>(
        result.weight_expressions.size())};
    result.weight_expressions.emplace_back(std::move(expression));
    return id;
  }

  [[nodiscard]] std::pair<SurfaceSvmWeightId, SurfaceSvmWeightId>
  append_mix_weights(ClosureExpressionId source,
                     SurfaceSvmWeightId parent,
                     ValueExpressionId factor,
                     bool left_active,
                     bool right_active) {
    auto left = SurfaceSvmWeightId{};
    auto right = SurfaceSvmWeightId{};
    if (left_active) {
      left = append_weight(SurfaceSvmWeightExpression{
          .operation = SurfaceSvmWeightOperation::mix_left,
          .a = parent,
          .b = {},
          .factor = factor,
          .source_mix = source,
          .pair = {}});
      if (!left.valid()) {
        return {};
      }
    }
    if (right_active) {
      right = append_weight(SurfaceSvmWeightExpression{
          .operation = SurfaceSvmWeightOperation::mix_right,
          .a = parent,
          .b = {},
          .factor = factor,
          .source_mix = source,
          .pair = {}});
      if (!right.valid()) {
        return {};
      }
    }
    if (left.valid() && right.valid()) {
      result.weight_expressions[left.value].pair = right;
      result.weight_expressions[right.value].pair = left;
    }
    return {left, right};
  }

  [[nodiscard]] std::uint32_t build_region(
      ClosureExpressionId closure,
      SurfaceSvmWeightId incoming_weight,
      std::uint32_t parent,
      std::uint32_t depth) {
    if (!closure_active(closure)) {
      return surface_svm_invalid_region;
    }
    if (regions.size() >= std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic = "the structured surface SVM has too many regions";
      return surface_svm_invalid_region;
    }
    const auto index = static_cast<std::uint32_t>(regions.size());
    regions.emplace_back(Region{.closure = closure,
                                .incoming_weight = incoming_weight,
                                .parent = parent,
                                .depth = depth,
                                .values = {},
                                .weight_operations = {},
                                .leaves = {}});

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

    auto left_weight = incoming_weight;
    auto right_weight = incoming_weight;
    if (dynamic_mix) {
      const auto weights = append_mix_weights(
          closure, incoming_weight, node.factor, left_active, right_active);
      if ((left_active && !weights.first.valid()) ||
          (right_active && !weights.second.valid())) {
        if (result.diagnostic.empty()) {
          result.diagnostic = "an active Mix has no weight expression";
        }
        return surface_svm_invalid_region;
      }
      left_weight = weights.first;
      right_weight = weights.second;
    }

    if (left_active) {
      const auto left =
          build_region(node.a, left_weight, index, depth + 1u);
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
          build_region(node.b, right_weight, index, depth + 1u);
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

  void add_weight_use(SurfaceSvmWeightId weight, std::uint32_t region) {
    if (!weight.valid()) {
      return;
    }
    if (weight.value >= result.weight_regions.size()) {
      result.diagnostic =
          "a closure references a weight outside the structured algebra";
      return;
    }
    result.weight_regions[weight.value] =
        lca(result.weight_regions[weight.value], region);
  }

  [[nodiscard]] SurfaceSvmWeightId add_weights(SurfaceSvmWeightId a,
                                                SurfaceSvmWeightId b) {
    // Do not rewrite complementary Mix weights to their parent. The real
    // source domain is IEEE float, not the real numbers: a linked factor may
    // be NaN, in which case both Cycles child weights and their sum are NaN.
    // Contracting (1 - saturate(f)) + saturate(f) to one would therefore
    // change whether the closure is rejected by the downstream weight test.
    // A finite-range analysis may prove that identity later; without such a
    // proof the explicit Add is the only semantics-preserving SSA form.
    return append_weight(SurfaceSvmWeightExpression{
        .operation = SurfaceSvmWeightOperation::add,
        .a = a,
        .b = b,
        .factor = {},
        .source_mix = {},
        .pair = {}});
  }

  [[nodiscard]] bool aggregate_leaves() {
    const auto closure_count = program.closure_instructions().size();
    leaf_aggregates.resize(closure_count);
    leaf_seen.assign(closure_count, false);
    for (auto region_index = std::size_t{};
         region_index < regions.size(); ++region_index) {
      const auto &region = regions[region_index];
      const auto &closure =
          program.closure_instructions()[region.closure.value];
      if (closure.operation == ClosureOperation::add ||
          closure.operation == ClosureOperation::mix ||
          closure.operation == ClosureOperation::null_closure) {
        continue;
      }
      auto &aggregate = leaf_aggregates[region.closure.value];
      if (!leaf_seen[region.closure.value]) {
        leaf_seen[region.closure.value] = true;
        aggregate.closure = region.closure;
        leaf_order.emplace_back(region.closure);
      }
      aggregate.region =
          lca(aggregate.region, static_cast<std::uint32_t>(region_index));
      aggregate.incoming_weights.emplace_back(region.incoming_weight);
    }

    for (const auto closure : leaf_order) {
      auto &aggregate = leaf_aggregates[closure.value];
      if (aggregate.region == surface_svm_invalid_region ||
          aggregate.incoming_weights.empty()) {
        result.diagnostic = "an active closure leaf has no occurrence";
        return false;
      }
      auto weight = aggregate.incoming_weights.front();
      for (auto index = std::size_t{1u};
           index < aggregate.incoming_weights.size(); ++index) {
        weight = add_weights(weight, aggregate.incoming_weights[index]);
        if (!weight.valid() && !result.diagnostic.empty()) {
          return false;
        }
      }
      aggregate.final_weight = weight;
      regions[aggregate.region].leaves.emplace_back(closure);

      const auto &instruction =
          program.closure_instructions()[closure.value];
      const auto operands = surface_closure_operands(instruction);
      if (operands.size() !=
          surface_closure_operand_count(instruction.operation)) {
        result.diagnostic =
            "an active closure leaf has an inconsistent operand ABI";
        return false;
      }
      for (const auto operand : operands) {
        add_value_use(operand, aggregate.region);
      }
      if (!result.diagnostic.empty()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool place_weight_algebra() {
    result.weight_regions.assign(result.weight_expressions.size(),
                                 surface_svm_invalid_region);
    for (const auto closure : leaf_order) {
      const auto &aggregate = leaf_aggregates[closure.value];
      add_weight_use(aggregate.final_weight, aggregate.region);
    }
    if (!result.diagnostic.empty()) {
      return false;
    }

    for (auto index = result.weight_expressions.size(); index-- > 0u;) {
      auto region = result.weight_regions[index];
      if (region == surface_svm_invalid_region) {
        continue;
      }
      const auto &expression = result.weight_expressions[index];
      if (expression.pair.valid()) {
        if (expression.pair.value >= result.weight_expressions.size()) {
          result.diagnostic = "a Mix weight has an invalid sibling";
          return false;
        }
        const auto pair_region = result.weight_regions[expression.pair.value];
        if (pair_region != surface_svm_invalid_region) {
          region = lca(region, pair_region);
          result.weight_regions[index] = region;
          result.weight_regions[expression.pair.value] = region;
        }
      }
      add_weight_use(expression.a, region);
      if (expression.operation == SurfaceSvmWeightOperation::add) {
        add_weight_use(expression.b, region);
      } else {
        add_value_use(expression.factor, region);
      }
      if (!result.diagnostic.empty()) {
        return false;
      }
    }

    for (auto index = std::size_t{};
         index < result.weight_expressions.size(); ++index) {
      const auto region = result.weight_regions[index];
      if (region == surface_svm_invalid_region) {
        continue;
      }
      const auto id =
          SurfaceSvmWeightId{static_cast<std::uint32_t>(index)};
      const auto &expression = result.weight_expressions[index];
      if (expression.operation == SurfaceSvmWeightOperation::add) {
        regions[region].weight_operations.emplace_back(
            WeightOperation{.primary = id, .secondary = {}});
        continue;
      }
      const auto pair_live =
          expression.pair.valid() &&
          result.weight_regions[expression.pair.value] !=
              surface_svm_invalid_region;
      if (pair_live) {
        if (expression.pair.value < index) {
          continue;
        }
        const auto &pair = result.weight_expressions[expression.pair.value];
        if (result.weight_regions[expression.pair.value] != region ||
            pair.pair != id || pair.source_mix != expression.source_mix) {
          result.diagnostic =
              "paired Mix weights do not share one transaction region";
          return false;
        }
        const auto left =
            expression.operation == SurfaceSvmWeightOperation::mix_left
                ? id
                : expression.pair;
        const auto right =
            expression.operation == SurfaceSvmWeightOperation::mix_right
                ? id
                : expression.pair;
        regions[region].weight_operations.emplace_back(
            WeightOperation{.primary = left, .secondary = right});
      } else {
        regions[region].weight_operations.emplace_back(
            WeightOperation{.primary = id, .secondary = {}});
      }
    }
    return true;
  }

  [[nodiscard]] bool propagate_value_regions() {
    const auto &values = program.value_instructions();
    for (auto index = values.size(); index-- > 0u;) {
      const auto region = result.value_regions[index];
      if (region == surface_svm_invalid_region) {
        continue;
      }
      for (const auto operand : values[index].operands) {
        if (!operand.valid() || operand.value >= index) {
          result.diagnostic =
              "the value graph is not a strict topological DAG";
          return false;
        }
        result.value_regions[operand.value] =
            lca(result.value_regions[operand.value], region);
      }
    }
    return true;
  }

  [[nodiscard]] bool region_has_direct_payload(
      std::uint32_t region) const noexcept {
    if (!regions[region].weight_operations.empty() ||
        !regions[region].leaves.empty()) {
      return true;
    }
    const auto &values = program.value_instructions();
    for (auto index = std::size_t{}; index < values.size(); ++index) {
      if (result.value_regions[index] == region &&
          values[index].operation != ValueOperation::parameter) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool compute_region_effect(std::uint32_t region_index) {
    auto &region = regions[region_index];
    region.left_effect = region.left != surface_svm_invalid_region &&
                         compute_region_effect(region.left);
    region.right_effect = region.right != surface_svm_invalid_region &&
                          compute_region_effect(region.right);
    region.subtree_effect = region_has_direct_payload(region_index) ||
                            region.left_effect || region.right_effect;
    return region.subtree_effect;
  }

  [[nodiscard]] bool add_control_uses() {
    for (auto region_index = std::size_t{};
         region_index < regions.size(); ++region_index) {
      const auto &region = regions[region_index];
      if (!region.dynamic_mix ||
          (!region.left_effect && !region.right_effect)) {
        continue;
      }
      const auto &closure =
          program.closure_instructions()[region.closure.value];
      add_value_use(closure.factor,
                    static_cast<std::uint32_t>(region_index));
    }
    return result.diagnostic.empty();
  }

  [[nodiscard]] bool publish_values() {
    const auto &values = program.value_instructions();
    for (auto index = std::size_t{}; index < values.size(); ++index) {
      const auto region = result.value_regions[index];
      if (region == surface_svm_invalid_region) {
        continue;
      }
      for (const auto operand : values[index].operands) {
        const auto operand_region = result.value_regions[operand.value];
        if (operand_region == surface_svm_invalid_region ||
            !is_ancestor(operand_region, region)) {
          result.diagnostic =
              "a scheduled value operand does not dominate its user";
          return false;
        }
      }
      if (values[index].operation != ValueOperation::parameter) {
        regions[region].values.emplace_back(ValueExpressionId{
            static_cast<std::uint32_t>(index)});
      }
    }
    return true;
  }

  [[nodiscard]] bool validate_weight_dominance() {
    for (auto index = std::size_t{};
         index < result.weight_expressions.size(); ++index) {
      const auto region = result.weight_regions[index];
      if (region == surface_svm_invalid_region) {
        continue;
      }
      const auto &expression = result.weight_expressions[index];
      const auto validate_operand = [&](SurfaceSvmWeightId operand) {
        return !operand.valid() ||
               (operand.value < result.weight_regions.size() &&
                result.weight_regions[operand.value] !=
                    surface_svm_invalid_region &&
                is_ancestor(result.weight_regions[operand.value], region));
      };
      if (!validate_operand(expression.a) ||
          (expression.operation == SurfaceSvmWeightOperation::add &&
           !validate_operand(expression.b))) {
        result.diagnostic =
            "a scheduled closure weight does not dominate its use";
        return false;
      }
      if (expression.operation != SurfaceSvmWeightOperation::add) {
        if (!expression.factor.valid() ||
            expression.factor.value >= result.value_regions.size() ||
            result.value_regions[expression.factor.value] ==
                surface_svm_invalid_region ||
            !is_ancestor(result.value_regions[expression.factor.value],
                         region)) {
          result.diagnostic =
              "a scheduled Mix factor does not dominate its weight operation";
          return false;
        }
      }
    }
    for (const auto closure : leaf_order) {
      const auto &aggregate = leaf_aggregates[closure.value];
      if (aggregate.final_weight.valid() &&
          (aggregate.final_weight.value >= result.weight_regions.size() ||
           result.weight_regions[aggregate.final_weight.value] ==
               surface_svm_invalid_region ||
           !is_ancestor(
               result.weight_regions[aggregate.final_weight.value],
               aggregate.region))) {
        result.diagnostic =
            "an accumulated closure weight does not dominate its leaf";
        return false;
      }
    }
    return true;
  }

  void emit(SurfaceSvmScheduleInstructionKind kind,
            std::uint32_t source,
            std::uint32_t target = 0u,
            SurfaceSvmWeightId weight = {},
            SurfaceSvmWeightId secondary_weight = {}) {
    result.instructions.emplace_back(SurfaceSvmScheduleInstruction{
        .kind = kind,
        .source = source,
        .target = target,
        .weight = weight,
        .secondary_weight = secondary_weight});
  }

  void emit_weight_operation(const WeightOperation &operation) {
    const auto &expression =
        result.weight_expressions[operation.primary.value];
    if (expression.operation == SurfaceSvmWeightOperation::add) {
      emit(SurfaceSvmScheduleInstructionKind::add_closure_weight,
           operation.primary.value, 0u, operation.primary);
      ++result.weight_add_instruction_count;
      return;
    }
    emit(SurfaceSvmScheduleInstructionKind::mix_closure,
         expression.source_mix.value, 0u, operation.primary,
         operation.secondary);
    ++result.mix_instruction_count;
  }

  void emit_region(std::uint32_t region_index, bool conditional) {
    const auto &region = regions[region_index];
    for (const auto value : region.values) {
      emit(SurfaceSvmScheduleInstructionKind::value, value.value);
      ++result.value_instruction_count;
      if (conditional) {
        ++result.conditional_value_instruction_count;
      }
    }
    for (const auto &weight : region.weight_operations) {
      emit_weight_operation(weight);
    }
    for (const auto closure : region.leaves) {
      const auto &aggregate = leaf_aggregates[closure.value];
      emit(SurfaceSvmScheduleInstructionKind::closure_leaf,
           closure.value, 0u, aggregate.final_weight);
      ++result.closure_leaf_count;
    }

    const auto &closure = program.closure_instructions()[region.closure.value];
    if (region.dynamic_mix &&
        (region.left_effect || region.right_effect)) {
      if (region.left_effect) {
        const auto guard = result.instructions.size();
        emit(SurfaceSvmScheduleInstructionKind::jump_if_one,
             region.closure.value);
        emit_region(region.left, true);
        result.instructions[guard].target =
            static_cast<std::uint32_t>(result.instructions.size());
        ++result.conditional_branch_count;
      }
      if (region.right_effect) {
        const auto guard = result.instructions.size();
        emit(SurfaceSvmScheduleInstructionKind::jump_if_zero,
             region.closure.value);
        emit_region(region.right, true);
        result.instructions[guard].target =
            static_cast<std::uint32_t>(result.instructions.size());
        ++result.conditional_branch_count;
      }
      return;
    }
    if (closure.operation == ClosureOperation::add ||
        closure.operation == ClosureOperation::mix) {
      if (region.left_effect) {
        emit_region(region.left, conditional);
      }
      if (region.right_effect) {
        emit_region(region.right, conditional);
      }
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
    result.endpoints = endpoints;
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
    const auto root_region = build_region(
        root, SurfaceSvmWeightId{}, surface_svm_invalid_region, 0u);
    if (!result.diagnostic.empty()) {
      return reject(std::move(result.diagnostic));
    }
    if (root_region == surface_svm_invalid_region) {
      emit(SurfaceSvmScheduleInstructionKind::end, ~std::uint32_t{0u});
      result.valid = true;
      return std::move(result);
    }

    if (!aggregate_leaves() || !place_weight_algebra() ||
        !propagate_value_regions()) {
      return reject(std::move(result.diagnostic));
    }
    static_cast<void>(compute_region_effect(root_region));
    if (!add_control_uses() || !propagate_value_regions() ||
        !publish_values() || !validate_weight_dominance()) {
      return reject(std::move(result.diagnostic));
    }
    // Factor dependencies added for live guards can only add prefix payload;
    // they cannot make an otherwise empty child subtree observable.
    static_cast<void>(compute_region_effect(root_region));

    result.region_count = static_cast<std::uint32_t>(regions.size());
    emit_region(root_region, false);
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
