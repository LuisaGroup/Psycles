#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/compiler/surface_svm_program.h>
#include <psycles/compiler/surface_svm_schedule.h>
#include <psycles/contract/scene.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error{message};
  }
}

struct CompiledGraph {
  std::shared_ptr<const SurfaceProgram> program;
  SurfaceParameterBlock parameters;
};

[[nodiscard]] CompiledGraph compile_graph(ShaderGraph graph) {
  ShaderCompiler compiler{make_core_node_registry()};
  const auto shader = compiler.compile(graph);
  require(shader.ok(), "structured-SVM fixture failed to compile");
  const auto lowered = compile_surface_program(*shader.program);
  require(lowered.ok(), "structured-SVM fixture failed to lower");
  auto binding = bind_surface_parameters(*lowered.program, *shader.program);
  require(binding.ok(), "structured-SVM fixture failed to bind");
  return {.program = lowered.program,
          .parameters = std::move(*binding.parameters)};
}

struct PlannedGraph {
  CompiledGraph compiled;
  SurfaceClosurePlan closures;
  SurfaceValueDependencyPlan dependencies;
};

[[nodiscard]] PlannedGraph analyze(ShaderGraph graph) {
  auto compiled = compile_graph(std::move(graph));
  auto closures =
      analyze_surface_closure_plan(*compiled.program, compiled.parameters);
  auto dependencies =
      analyze_surface_value_dependencies(*compiled.program, closures);
  return {.compiled = std::move(compiled),
          .closures = std::move(closures),
          .dependencies = std::move(dependencies)};
}

[[nodiscard]] SurfaceSvmSchedulePlan
schedule(const PlannedGraph &graph,
         SurfaceClosureEndpointMask endpoints = all_surface_closure_endpoints) {
  auto plan = plan_surface_svm_schedule(*graph.compiled.program, graph.closures,
                                        graph.dependencies, endpoints);
  require(plan.valid, "structured-SVM schedule failed: " + plan.diagnostic);
  return plan;
}

[[nodiscard]] SurfaceSvmStoragePlan
storage(const PlannedGraph &graph, const SurfaceSvmSchedulePlan &plan,
        SurfaceValueStorageCapacity capacity = {}) {
  auto allocation =
      plan_surface_svm_storage(*graph.compiled.program, plan, capacity);
  require(allocation.valid,
          "structured-SVM storage failed: " + allocation.diagnostic);
  require(allocation.compatible(*graph.compiled.program, plan),
          "structured-SVM storage is not compatible with its schedule");
  return allocation;
}

[[nodiscard]] SurfaceSvmProgramImage
lower(const PlannedGraph &graph, const SurfaceSvmSchedulePlan &plan,
      const SurfaceSvmStoragePlan &allocation) {
  auto image = lower_surface_svm_program(
      *graph.compiled.program, graph.closures, graph.dependencies, plan,
      allocation);
  require(image.valid, "unified surface SVM failed to lower: " +
                           image.diagnostic);
  const auto diagnostic = validate_surface_svm_program_image(image);
  require(diagnostic.empty(),
          "unified surface SVM failed to validate: " + diagnostic);
  return image;
}

[[nodiscard]] SurfaceSvmProgramImage
lower(const PlannedGraph &graph, const SurfaceSvmSchedulePlan &plan) {
  return lower(graph, plan, storage(graph, plan));
}

[[nodiscard]] ValueExpressionId find_value(const SurfaceProgram &program,
                                           NodeId source,
                                           ValueOperation operation) {
  const auto &values = program.value_instructions();
  for (auto index = std::size_t{}; index < values.size(); ++index) {
    if (values[index].source_node == source &&
        values[index].operation == operation) {
      return ValueExpressionId{static_cast<std::uint32_t>(index)};
    }
  }
  throw std::runtime_error{"fixture value instruction is absent"};
}

[[nodiscard]] ClosureExpressionId find_closure(const SurfaceProgram &program,
                                               NodeId source) {
  const auto &closures = program.closure_instructions();
  for (auto index = std::size_t{}; index < closures.size(); ++index) {
    if (closures[index].source_node == source) {
      return ClosureExpressionId{static_cast<std::uint32_t>(index)};
    }
  }
  throw std::runtime_error{"fixture closure instruction is absent"};
}

[[nodiscard]] std::size_t
find_schedule_instruction(const SurfaceSvmSchedulePlan &plan,
                          SurfaceSvmScheduleInstructionKind kind,
                          std::uint32_t source) {
  for (auto index = std::size_t{}; index < plan.instructions.size(); ++index) {
    const auto &instruction = plan.instructions[index];
    if (instruction.kind == kind && instruction.source == source) {
      return index;
    }
  }
  throw std::runtime_error{"fixture schedule instruction is absent"};
}

[[nodiscard]] std::size_t
find_schedule_instruction(const SurfaceSvmSchedulePlan &plan,
                          SurfaceSvmScheduleInstructionKind kind) {
  for (auto index = std::size_t{}; index < plan.instructions.size(); ++index) {
    if (plan.instructions[index].kind == kind) {
      return index;
    }
  }
  throw std::runtime_error{"fixture schedule instruction kind is absent"};
}

[[nodiscard]] std::size_t
find_bytecode_instruction(const SurfaceSvmProgramImage &image,
                          SurfaceSvmBytecodeKind kind) {
  for (auto index = std::size_t{}; index < image.instructions.size(); ++index) {
    if (surface_svm_bytecode_kind(image.instructions[index]) == kind) {
      return index;
    }
  }
  throw std::runtime_error{"fixture bytecode instruction kind is absent"};
}

void verify_total_schedule_contract(const SurfaceProgram &program,
                                    const SurfaceSvmSchedulePlan &plan) {
  require(plan.valid && !plan.instructions.empty(),
          "schedule contract received an invalid image");
  const auto &values = program.value_instructions();
  const auto &closures = program.closure_instructions();
  require(plan.value_regions.size() == values.size(),
          "schedule region map is not parallel to the value IR");
  require(plan.weight_regions.size() == plan.weight_expressions.size(),
          "schedule region map is not parallel to the weight IR");

  std::vector<std::uint32_t> value_emissions(values.size(), 0u);
  std::vector<std::uint32_t> weight_emissions(
      plan.weight_expressions.size(), 0u);
  auto end_count = std::uint32_t{};
  for (auto index = std::size_t{}; index < plan.instructions.size(); ++index) {
    const auto &instruction = plan.instructions[index];
    switch (instruction.kind) {
    case SurfaceSvmScheduleInstructionKind::value:
      require(instruction.source < values.size() &&
                  values[instruction.source].operation !=
                      ValueOperation::parameter,
              "schedule emitted an invalid or parameter value");
      ++value_emissions[instruction.source];
      break;
    case SurfaceSvmScheduleInstructionKind::mix_closure:
      require(instruction.source < closures.size() &&
                  closures[instruction.source].operation ==
                      ClosureOperation::mix && instruction.weight.valid() &&
                  instruction.weight.value <
                      plan.weight_expressions.size(),
              "schedule emitted a non-Mix control instruction");
      ++weight_emissions[instruction.weight.value];
      if (instruction.secondary_weight.valid()) {
        require(instruction.secondary_weight.value <
                    plan.weight_expressions.size(),
                "schedule emitted an invalid secondary Mix weight");
        ++weight_emissions[instruction.secondary_weight.value];
      }
      break;
    case SurfaceSvmScheduleInstructionKind::add_closure_weight:
      require(instruction.source < plan.weight_expressions.size() &&
                  instruction.weight.valid() &&
                  instruction.weight.value == instruction.source &&
                  plan.weight_expressions[instruction.source].operation ==
                      SurfaceSvmWeightOperation::add,
              "schedule emitted a malformed closure-weight Add");
      ++weight_emissions[instruction.weight.value];
      break;
    case SurfaceSvmScheduleInstructionKind::jump_if_one:
    case SurfaceSvmScheduleInstructionKind::jump_if_zero:
      require(instruction.source < closures.size() &&
                  closures[instruction.source].operation ==
                      ClosureOperation::mix &&
                  instruction.target > index &&
                  instruction.target < plan.instructions.size(),
              "schedule contains a non-forward or malformed Mix guard");
      break;
    case SurfaceSvmScheduleInstructionKind::closure_leaf:
      require(
          instruction.source < closures.size() &&
              closures[instruction.source].operation != ClosureOperation::add &&
              closures[instruction.source].operation != ClosureOperation::mix &&
              closures[instruction.source].operation !=
                  ClosureOperation::null_closure &&
              (!instruction.weight.valid() ||
               instruction.weight.value < plan.weight_expressions.size()),
          "schedule emitted a control closure as a leaf");
      break;
    case SurfaceSvmScheduleInstructionKind::end:
      ++end_count;
      require(index + 1u == plan.instructions.size(),
              "schedule End is not the unique final instruction");
      break;
    }
  }
  require(end_count == 1u,
          "schedule does not have exactly one terminal instruction");

  auto expected_value_count = std::uint32_t{};
  for (auto index = std::size_t{}; index < values.size(); ++index) {
    const auto scheduled =
        plan.value_regions[index] != surface_svm_invalid_region &&
        values[index].operation != ValueOperation::parameter;
    require(value_emissions[index] == (scheduled ? 1u : 0u),
            "a scheduled computed value is missing or duplicated");
    expected_value_count += scheduled ? 1u : 0u;
  }
  require(plan.value_instruction_count == expected_value_count,
          "schedule value count disagrees with its region proof");

  auto expected_weight_count = std::uint32_t{};
  for (auto index = std::size_t{};
       index < plan.weight_expressions.size(); ++index) {
    const auto &expression = plan.weight_expressions[index];
    const auto valid_operand = [&](SurfaceSvmWeightId operand) {
      return !operand.valid() || operand.value < index;
    };
    require(valid_operand(expression.a) &&
                (expression.operation != SurfaceSvmWeightOperation::add ||
                 valid_operand(expression.b)),
            "weight algebra is not a strict SSA DAG");
    const auto scheduled =
        plan.weight_regions[index] != surface_svm_invalid_region;
    require(weight_emissions[index] == (scheduled ? 1u : 0u),
            "a scheduled weight is missing, duplicated, or spuriously live");
    expected_weight_count += scheduled ? 1u : 0u;
  }
  require(expected_weight_count ==
              plan.mix_instruction_count * 2u -
                  std::ranges::count_if(
                      plan.instructions,
                      [](const SurfaceSvmScheduleInstruction &instruction) {
                        return instruction.kind ==
                                   SurfaceSvmScheduleInstructionKind::
                                       mix_closure &&
                               !instruction.secondary_weight.valid();
                      }) +
                  plan.weight_add_instruction_count,
          "weight instruction counts do not cover the active SSA results");
}

struct ExecutedSchedule {
  std::vector<std::uint32_t> values;
  std::vector<std::uint32_t> leaves;
};

[[nodiscard]] ExecutedSchedule
execute_control(const SurfaceSvmSchedulePlan &plan,
                const std::map<std::uint32_t, float> &mix_factors) {
  ExecutedSchedule result;
  for (auto pc = std::size_t{}; pc < plan.instructions.size();) {
    const auto &instruction = plan.instructions[pc];
    switch (instruction.kind) {
    case SurfaceSvmScheduleInstructionKind::value:
      result.values.emplace_back(instruction.source);
      ++pc;
      break;
    case SurfaceSvmScheduleInstructionKind::mix_closure:
    case SurfaceSvmScheduleInstructionKind::add_closure_weight:
      ++pc;
      break;
    case SurfaceSvmScheduleInstructionKind::jump_if_one: {
      const auto factor = mix_factors.find(instruction.source);
      require(factor != mix_factors.end(), "missing fixture Mix factor");
      pc = factor->second >= 1.0f ? instruction.target : pc + 1u;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::jump_if_zero: {
      const auto factor = mix_factors.find(instruction.source);
      require(factor != mix_factors.end(), "missing fixture Mix factor");
      pc = factor->second <= 0.0f ? instruction.target : pc + 1u;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::closure_leaf:
      result.leaves.emplace_back(instruction.source);
      ++pc;
      break;
    case SurfaceSvmScheduleInstructionKind::end:
      return result;
    }
  }
  throw std::runtime_error{"schedule execution fell off the image"};
}

struct WeightedLeaf {
  std::uint32_t closure{};
  float weight{};
};

[[nodiscard]] std::vector<WeightedLeaf> execute_weighted_control(
    const SurfaceSvmSchedulePlan &plan,
    const std::map<std::uint32_t, float> &mix_factors) {
  std::vector<float> weights(plan.weight_expressions.size(), 0.0f);
  std::vector<WeightedLeaf> leaves;
  const auto read_weight = [&](SurfaceSvmWeightId weight) {
    return weight.valid() ? weights[weight.value] : 1.0f;
  };
  for (auto pc = std::size_t{}; pc < plan.instructions.size();) {
    const auto &instruction = plan.instructions[pc];
    switch (instruction.kind) {
    case SurfaceSvmScheduleInstructionKind::value:
      ++pc;
      break;
    case SurfaceSvmScheduleInstructionKind::mix_closure: {
      const auto factor = mix_factors.find(instruction.source);
      require(factor != mix_factors.end(), "missing weighted Mix factor");
      const auto saturated = std::clamp(factor->second, 0.0f, 1.0f);
      const auto evaluate = [&](SurfaceSvmWeightId output) {
        if (!output.valid()) {
          return;
        }
        const auto &expression = plan.weight_expressions[output.value];
        const auto parent = read_weight(expression.a);
        weights[output.value] =
            expression.operation == SurfaceSvmWeightOperation::mix_left
                ? parent * (1.0f - saturated)
                : parent * saturated;
      };
      evaluate(instruction.weight);
      evaluate(instruction.secondary_weight);
      ++pc;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::add_closure_weight: {
      const auto &expression =
          plan.weight_expressions[instruction.weight.value];
      weights[instruction.weight.value] =
          read_weight(expression.a) + read_weight(expression.b);
      ++pc;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::jump_if_one: {
      const auto factor = mix_factors.find(instruction.source);
      require(factor != mix_factors.end(), "missing weighted Mix guard");
      pc = factor->second >= 1.0f ? instruction.target : pc + 1u;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::jump_if_zero: {
      const auto factor = mix_factors.find(instruction.source);
      require(factor != mix_factors.end(), "missing weighted Mix guard");
      pc = factor->second <= 0.0f ? instruction.target : pc + 1u;
      break;
    }
    case SurfaceSvmScheduleInstructionKind::closure_leaf:
      leaves.emplace_back(WeightedLeaf{
          .closure = instruction.source,
          .weight = read_weight(instruction.weight)});
      ++pc;
      break;
    case SurfaceSvmScheduleInstructionKind::end:
      return leaves;
    }
  }
  throw std::runtime_error{"weighted schedule execution fell off the image"};
}

void require_weighted_leaf(const std::vector<WeightedLeaf> &leaves,
                           std::uint32_t closure, float weight,
                           const std::string &message) {
  const auto found = std::ranges::find_if(
      leaves, [=](const WeightedLeaf &leaf) {
        return leaf.closure == closure && leaf.weight == weight;
      });
  require(found != leaves.end(), message);
}

struct WeightedBytecodeLeaf {
  ClosureOperation operation{};
  float weight{};
};

[[nodiscard]] std::vector<WeightedBytecodeLeaf> execute_weighted_bytecode(
    const SurfaceSvmProgramImage &image,
    const std::map<std::uint32_t, float> &factors) {
  std::vector<float> scalars(image.scalar_slots, 0.0f);
  std::vector<WeightedBytecodeLeaf> leaves;
  const auto read_factor = [&](std::uint32_t address) {
    const auto found = factors.find(address);
    require(found != factors.end(), "missing unified-bytecode Mix factor");
    return found->second;
  };
  const auto read_weight = [&](std::uint32_t slot) {
    if (slot == surface_svm_root_weight_slot) {
      return 1.0f;
    }
    require(slot < scalars.size(), "unified-bytecode weight exceeds bank");
    return scalars[slot];
  };
  for (auto pc = std::size_t{}; pc < image.instructions.size();) {
    const auto &instruction = image.instructions[pc];
    switch (surface_svm_bytecode_kind(instruction)) {
    case SurfaceSvmBytecodeKind::value:
      ++pc;
      break;
    case SurfaceSvmBytecodeKind::mix_closure: {
      const auto saturated =
          std::clamp(read_factor(instruction.payload0), 0.0f, 1.0f);
      const auto parent = read_weight(instruction.payload1);
      if ((instruction.control & surface_svm_mix_left_result_bit) != 0u) {
        const auto slot = surface_svm_mix_left_weight_slot(instruction);
        scalars[slot] = parent * (1.0f - saturated);
      }
      if ((instruction.control & surface_svm_mix_right_result_bit) != 0u) {
        const auto slot = surface_svm_mix_right_weight_slot(instruction);
        scalars[slot] = parent * saturated;
      }
      ++pc;
      break;
    }
    case SurfaceSvmBytecodeKind::add_closure_weight:
      scalars[instruction.payload2] =
          read_weight(instruction.payload0) +
          read_weight(instruction.payload1);
      ++pc;
      break;
    case SurfaceSvmBytecodeKind::jump_if_one:
      pc = read_factor(instruction.payload0) >= 1.0f
               ? instruction.payload1
               : pc + 1u;
      break;
    case SurfaceSvmBytecodeKind::jump_if_zero:
      pc = read_factor(instruction.payload0) <= 0.0f
               ? instruction.payload1
               : pc + 1u;
      break;
    case SurfaceSvmBytecodeKind::closure_leaf: {
      const auto legacy = SurfaceClosureBytecodeInstruction{
          .control = surface_svm_closure_control(instruction)};
      leaves.emplace_back(WeightedBytecodeLeaf{
          .operation = surface_closure_operation(legacy),
          .weight = read_weight(instruction.payload1)});
      ++pc;
      break;
    }
    case SurfaceSvmBytecodeKind::set_normal:
      throw std::runtime_error{
          "weight-only bytecode fixture unexpectedly contains SetNormal"};
    case SurfaceSvmBytecodeKind::end:
      return leaves;
    case SurfaceSvmBytecodeKind::invalid:
      throw std::runtime_error{"unified-bytecode execution saw invalid opcode"};
    }
  }
  throw std::runtime_error{"unified-bytecode execution fell off the image"};
}

void require_weighted_bytecode_leaf(
    const std::vector<WeightedBytecodeLeaf> &leaves,
    ClosureOperation operation, float weight, const std::string &message) {
  const auto found = std::ranges::find_if(
      leaves, [=](const WeightedBytecodeLeaf &leaf) {
        return leaf.operation == operation && leaf.weight == weight;
      });
  require(found != leaves.end(), message);
}

struct ConditionalGraph {
  ShaderGraph graph;
  NodeId shared_value;
  NodeId left_value;
  NodeId right_value;
  NodeId diffuse;
  NodeId emission;
  NodeId mix;
};

[[nodiscard]] ConditionalGraph make_conditional_graph() {
  ConditionalGraph fixture;
  auto &graph = fixture.graph;
  const auto base_a =
      graph.add_node(node_type::constant_color, "Structured SVM shared A");
  const auto base_b =
      graph.add_node(node_type::constant_color, "Structured SVM shared B");
  const auto left_b =
      graph.add_node(node_type::constant_color, "Structured SVM left B");
  const auto right_b =
      graph.add_node(node_type::constant_color, "Structured SVM right B");
  const auto factor =
      graph.add_node(node_type::constant_float, "Structured SVM linked factor");
  fixture.shared_value =
      graph.add_node(node_type::mix_color, "Structured SVM shared value");
  fixture.left_value =
      graph.add_node(node_type::mix_color, "Structured SVM left value");
  fixture.right_value =
      graph.add_node(node_type::mix_color, "Structured SVM right value");
  fixture.diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Structured SVM left closure");
  fixture.emission =
      graph.add_node(node_type::emission, "Structured SVM right closure");
  fixture.mix =
      graph.add_node(node_type::mix_closure, "Structured SVM closure Mix");

  require(graph.set_input(base_a, "Color",
                          SocketValue::color({0.1f, 0.2f, 0.3f})) &&
              graph.set_input(base_b, "Color",
                              SocketValue::color({0.7f, 0.6f, 0.5f})) &&
              graph.set_input(left_b, "Color",
                              SocketValue::color({0.9f, 0.2f, 0.1f})) &&
              graph.set_input(right_b, "Color",
                              SocketValue::color({0.1f, 0.8f, 0.4f})) &&
              graph.set_input(factor, "Value", SocketValue::floating(0.37f)) &&
              graph.connect({.node = base_a, .socket = "Color"},
                            fixture.shared_value, "A") &&
              graph.connect({.node = base_b, .socket = "Color"},
                            fixture.shared_value, "B") &&
              graph.connect({.node = fixture.shared_value, .socket = "Color"},
                            fixture.left_value, "A") &&
              graph.connect({.node = left_b, .socket = "Color"},
                            fixture.left_value, "B") &&
              graph.connect({.node = fixture.shared_value, .socket = "Color"},
                            fixture.right_value, "A") &&
              graph.connect({.node = right_b, .socket = "Color"},
                            fixture.right_value, "B") &&
              graph.connect({.node = fixture.left_value, .socket = "Color"},
                            fixture.diffuse, "Color") &&
              graph.connect({.node = fixture.right_value, .socket = "Color"},
                            fixture.emission, "Color") &&
              graph.connect({.node = factor, .socket = "Value"}, fixture.mix,
                            "Factor") &&
              graph.connect({.node = fixture.diffuse, .socket = "Closure"},
                            fixture.mix, "A") &&
              graph.connect({.node = fixture.emission, .socket = "Closure"},
                            fixture.mix, "B"),
          "failed to configure structured conditional graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = fixture.mix, .socket = "Closure"});
  return fixture;
}

void test_shared_values_are_hoisted_and_private_values_are_guarded() {
  auto fixture = make_conditional_graph();
  const auto shared_node = fixture.shared_value;
  const auto left_node = fixture.left_value;
  const auto right_node = fixture.right_value;
  const auto diffuse_node = fixture.diffuse;
  const auto emission_node = fixture.emission;
  const auto mix_node = fixture.mix;
  auto analyzed = analyze(std::move(fixture.graph));
  const auto plan = schedule(analyzed);
  verify_total_schedule_contract(*analyzed.compiled.program, plan);

  const auto shared = find_value(*analyzed.compiled.program, shared_node,
                                 ValueOperation::mix);
  const auto left = find_value(*analyzed.compiled.program, left_node,
                               ValueOperation::mix);
  const auto right = find_value(*analyzed.compiled.program, right_node,
                                ValueOperation::mix);
  const auto mix = find_closure(*analyzed.compiled.program, mix_node);
  const auto diffuse = find_closure(*analyzed.compiled.program, diffuse_node);
  const auto emission = find_closure(*analyzed.compiled.program, emission_node);

  const auto shared_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::value, shared.value);
  const auto mix_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::mix_closure, mix.value);
  const auto left_guard = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_one, mix.value);
  const auto left_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::value, left.value);
  const auto right_guard = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_zero, mix.value);
  const auto right_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::value, right.value);
  require(shared_pc < mix_pc && mix_pc < left_guard && left_guard < left_pc &&
              left_pc < right_guard && right_guard < right_pc &&
              plan.conditional_value_instruction_count == 2u,
          "LCA placement did not separate shared and branch-private values");

  const auto left_only = execute_control(plan, {{mix.value, 0.0f}});
  require(left_only.leaves == std::vector<std::uint32_t>{diffuse.value} &&
              std::ranges::find(left_only.values, left.value) !=
                  left_only.values.end() &&
              std::ranges::find(left_only.values, right.value) ==
                  left_only.values.end(),
          "factor zero did not execute only the left private subgraph");
  const auto right_only = execute_control(plan, {{mix.value, 1.0f}});
  require(right_only.leaves == std::vector<std::uint32_t>{emission.value} &&
              std::ranges::find(right_only.values, right.value) !=
                  right_only.values.end() &&
              std::ranges::find(right_only.values, left.value) ==
                  right_only.values.end(),
          "factor one did not execute only the right private subgraph");
  const auto below_zero = execute_control(plan, {{mix.value, -0.25f}});
  const auto above_one = execute_control(plan, {{mix.value, 1.25f}});
  require(below_zero.leaves == left_only.leaves &&
              above_one.leaves == right_only.leaves,
          "Mix guards do not use Cycles' <= 0 and >= 1 thresholds");
  const auto both = execute_control(plan, {{mix.value, 0.37f}});
  require(both.leaves ==
              std::vector<std::uint32_t>{diffuse.value, emission.value},
          "interior Mix factor did not execute both closure branches");
}

void test_endpoint_projection_moves_shared_work_inside_the_live_guard() {
  auto fixture = make_conditional_graph();
  const auto shared_node = fixture.shared_value;
  const auto left_node = fixture.left_value;
  const auto right_node = fixture.right_value;
  const auto mix_node = fixture.mix;
  auto analyzed = analyze(std::move(fixture.graph));
  const auto physical = schedule(
      analyzed, surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical));
  verify_total_schedule_contract(*analyzed.compiled.program, physical);

  const auto shared = find_value(*analyzed.compiled.program, shared_node,
                                 ValueOperation::mix);
  const auto left = find_value(*analyzed.compiled.program, left_node,
                               ValueOperation::mix);
  const auto right = find_value(*analyzed.compiled.program, right_node,
                                ValueOperation::mix);
  const auto mix = find_closure(*analyzed.compiled.program, mix_node);
  const auto guard = find_schedule_instruction(
      physical, SurfaceSvmScheduleInstructionKind::jump_if_one, mix.value);
  require(
      find_schedule_instruction(physical,
                                SurfaceSvmScheduleInstructionKind::value,
                                shared.value) > guard &&
          find_schedule_instruction(physical,
                                    SurfaceSvmScheduleInstructionKind::value,
                                    left.value) > guard &&
          physical.value_regions[right.value] == surface_svm_invalid_region &&
          physical.conditional_branch_count == 1u &&
          physical.conditional_value_instruction_count == 2u,
      "endpoint projection retained cross-domain work above the guard");

  const auto skipped = execute_control(physical, {{mix.value, 1.0f}});
  require(skipped.values.empty() && skipped.leaves.empty(),
          "physical endpoint did not skip all work at factor one");
}

[[nodiscard]] ShaderGraph
make_nested_mix_graph(NodeId &inner_out, NodeId &outer_out, NodeId &diffuse_out,
                      NodeId &glossy_out, NodeId &emission_out) {
  ShaderGraph graph;
  diffuse_out = graph.add_node(node_type::diffuse_bsdf, "Nested diffuse");
  glossy_out = graph.add_node(node_type::glossy_bsdf, "Nested glossy");
  emission_out = graph.add_node(node_type::emission, "Nested emission");
  inner_out = graph.add_node(node_type::mix_closure, "Nested inner Mix");
  outer_out = graph.add_node(node_type::mix_closure, "Nested outer Mix");
  require(
      graph.set_input(inner_out, "Factor", SocketValue::floating(0.2f)) &&
          graph.set_input(outer_out, "Factor", SocketValue::floating(0.7f)) &&
          graph.connect({.node = diffuse_out, .socket = "Closure"}, inner_out,
                        "A") &&
          graph.connect({.node = glossy_out, .socket = "Closure"}, inner_out,
                        "B") &&
          graph.connect({.node = inner_out, .socket = "Closure"}, outer_out,
                        "A") &&
          graph.connect({.node = emission_out, .socket = "Closure"}, outer_out,
                        "B"),
      "failed to configure nested structured Mix graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = outer_out, .socket = "Closure"});
  return graph;
}

void test_nested_guards_have_exact_forward_targets() {
  NodeId inner_node, outer_node, diffuse_node, glossy_node, emission_node;
  auto analyzed = analyze(make_nested_mix_graph(
      inner_node, outer_node, diffuse_node, glossy_node, emission_node));
  const auto plan = schedule(analyzed);
  verify_total_schedule_contract(*analyzed.compiled.program, plan);
  const auto inner = find_closure(*analyzed.compiled.program, inner_node);
  const auto outer = find_closure(*analyzed.compiled.program, outer_node);
  const auto diffuse = find_closure(*analyzed.compiled.program, diffuse_node);
  const auto glossy = find_closure(*analyzed.compiled.program, glossy_node);
  const auto emission = find_closure(*analyzed.compiled.program, emission_node);

  const auto outer_left = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_one, outer.value);
  const auto outer_right = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_zero, outer.value);
  const auto inner_left = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_one, inner.value);
  const auto inner_right = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_zero, inner.value);
  require(plan.instructions[outer_left].target == outer_right &&
              plan.instructions[inner_left].target == inner_right &&
              plan.instructions[inner_right].target == outer_right &&
              plan.instructions[outer_right].target ==
                  plan.instructions.size() - 1u,
          "nested Mix forward targets do not delimit exact subtrees");

  const auto path =
      execute_control(plan, {{outer.value, 0.0f}, {inner.value, 1.0f}});
  require(path.leaves == std::vector<std::uint32_t>{glossy.value},
          "nested guards did not select the unique requested leaf");
  const auto outer_right_path =
      execute_control(plan, {{outer.value, 1.0f}, {inner.value, 0.0f}});
  require(outer_right_path.leaves ==
                  std::vector<std::uint32_t>{emission.value} &&
              std::ranges::find(outer_right_path.leaves, diffuse.value) ==
                  outer_right_path.leaves.end(),
          "outer guard entered a skipped nested subtree");
}

void test_add_sibling_is_outside_mix_control() {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Add diffuse");
  const auto glossy = graph.add_node(node_type::glossy_bsdf, "Add glossy");
  const auto emission = graph.add_node(node_type::emission, "Add sibling");
  const auto mix = graph.add_node(node_type::mix_closure, "Add child Mix");
  const auto add = graph.add_node(node_type::add_closure, "Add root");
  require(graph.set_input(mix, "Factor", SocketValue::floating(0.4f)) &&
              graph.connect({.node = diffuse, .socket = "Closure"}, mix, "A") &&
              graph.connect({.node = glossy, .socket = "Closure"}, mix, "B") &&
              graph.connect({.node = mix, .socket = "Closure"}, add, "A") &&
              graph.connect({.node = emission, .socket = "Closure"}, add, "B"),
          "failed to configure Add sibling graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = add, .socket = "Closure"});
  const auto analyzed = analyze(std::move(graph));
  const auto plan = schedule(analyzed);
  verify_total_schedule_contract(*analyzed.compiled.program, plan);
  const auto mix_id = find_closure(*analyzed.compiled.program, mix);
  const auto diffuse_id = find_closure(*analyzed.compiled.program, diffuse);
  const auto glossy_id = find_closure(*analyzed.compiled.program, glossy);
  const auto emission_id = find_closure(*analyzed.compiled.program, emission);
  require(execute_control(plan, {{mix_id.value, 0.0f}}).leaves ==
                  std::vector<std::uint32_t>{diffuse_id.value,
                                             emission_id.value} &&
              execute_control(plan, {{mix_id.value, 1.0f}}).leaves ==
                  std::vector<std::uint32_t>{glossy_id.value,
                                             emission_id.value} &&
              execute_control(plan, {{mix_id.value, 0.4f}}).leaves ==
                  std::vector<std::uint32_t>{diffuse_id.value,
                                             glossy_id.value,
                                             emission_id.value},
          "Mix guard skipped or captured its unconditional Add sibling");
}

void test_static_mix_pruning_removes_control_records() {
  for (const auto factor : {0.0f, 1.0f}) {
    ShaderGraph graph;
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Static Mix diffuse");
    const auto emission =
        graph.add_node(node_type::emission, "Static Mix emission");
    const auto mix = graph.add_node(node_type::mix_closure, "Static Mix root");
    require(
        graph.set_input(mix, "Factor", SocketValue::floating(factor)) &&
            graph.connect({.node = diffuse, .socket = "Closure"}, mix, "A") &&
            graph.connect({.node = emission, .socket = "Closure"}, mix, "B"),
        "failed to configure statically pruned Mix");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = mix, .socket = "Closure"});
    const auto analyzed = analyze(std::move(graph));
    const auto plan = schedule(analyzed);
    verify_total_schedule_contract(*analyzed.compiled.program, plan);
    require(plan.mix_instruction_count == 0u &&
                plan.conditional_branch_count == 0u &&
                plan.closure_leaf_count == 1u,
            "statically selected Mix retained dynamic control flow");
  }
}

void test_shared_closure_dag_is_accepted_without_value_recomputation() {
  ShaderGraph graph;
  const auto a = graph.add_node(node_type::constant_color, "DAG color A");
  const auto b = graph.add_node(node_type::constant_color, "DAG color B");
  const auto value = graph.add_node(node_type::mix_color, "DAG shared value");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "DAG shared closure");
  const auto add = graph.add_node(node_type::add_closure, "DAG Add");
  require(
      graph.connect({.node = a, .socket = "Color"}, value, "A") &&
          graph.connect({.node = b, .socket = "Color"}, value, "B") &&
          graph.connect({.node = value, .socket = "Color"}, diffuse, "Color") &&
          graph.connect({.node = diffuse, .socket = "Closure"}, add, "A") &&
          graph.connect({.node = diffuse, .socket = "Closure"}, add, "B"),
      "failed to configure shared closure DAG");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = add, .socket = "Closure"});
  const auto analyzed = analyze(std::move(graph));
  const auto plan = schedule(analyzed);
  verify_total_schedule_contract(*analyzed.compiled.program, plan);
  const auto value_id =
      find_value(*analyzed.compiled.program, value, ValueOperation::mix);
  const auto closure_id = find_closure(*analyzed.compiled.program, diffuse);
  const auto path = execute_control(plan, {});
  require(std::ranges::count(path.values, value_id.value) == 1u &&
              std::ranges::count(path.leaves, closure_id.value) == 1u &&
              plan.closure_leaf_count == 1u &&
              plan.weight_add_instruction_count == 1u,
          "closure DAG was rejected, duplicated, or lost weight accumulation");
  const auto weighted = execute_weighted_control(plan, {});
  require(weighted.size() == 1u,
          "shared Add closure emitted more than one weighted leaf");
  require_weighted_leaf(weighted, closure_id.value, 2.0f,
                        "shared Add closure did not accumulate weight two");
  const auto allocation = storage(analyzed, plan);
  require(allocation.weight_values == 1u &&
              allocation.weight_locations.size() ==
                  plan.weight_expressions.size(),
          "shared Add weight was not included in scalar SSA coloring");
  const auto image = lower(analyzed, plan, allocation);
  const auto bytecode = execute_weighted_bytecode(image, {});
  require(bytecode.size() == 1u,
          "unified bytecode duplicated the shared Add leaf");
  require_weighted_bytecode_leaf(
      bytecode, ClosureOperation::diffuse, 2.0f,
      "unified bytecode did not add two root-weight occurrences");
}

void test_complementary_shared_mix_leaf_preserves_ieee_weight_algebra() {
  ShaderGraph graph;
  const auto factor = graph.add_node(node_type::constant_float,
                                     "Shared Mix linked factor");
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf,
                                      "Shared Mix closure");
  const auto mix = graph.add_node(node_type::mix_closure,
                                  "Shared Mix root");
  require(graph.set_input(factor, "Value", SocketValue::floating(0.37f)) &&
              graph.connect({.node = factor, .socket = "Value"}, mix,
                            "Factor") &&
              graph.connect({.node = diffuse, .socket = "Closure"}, mix,
                            "A") &&
              graph.connect({.node = diffuse, .socket = "Closure"}, mix,
                            "B"),
          "failed to configure complementary shared Mix leaf");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});
  const auto analyzed = analyze(std::move(graph));
  const auto plan = schedule(analyzed);
  verify_total_schedule_contract(*analyzed.compiled.program, plan);
  const auto closure = find_closure(*analyzed.compiled.program, diffuse);
  const auto mix_id = find_closure(*analyzed.compiled.program, mix);
  require(plan.closure_leaf_count == 1u &&
              plan.mix_instruction_count == 1u &&
              plan.weight_add_instruction_count == 1u &&
              plan.conditional_branch_count == 0u &&
              plan.weight_expressions.size() == 3u &&
              std::ranges::all_of(
                  plan.weight_regions,
                  [](std::uint32_t region) {
                    return region != surface_svm_invalid_region;
                  }),
          "complementary shared Mix did not retain its exact weight algebra");
  const auto leaves = execute_weighted_control(plan, {{mix_id.value, 0.37f}});
  require(leaves.size() == 1u,
          "complementary shared Mix emitted duplicate leaves");
  require_weighted_leaf(leaves, closure.value, 1.0f,
                        "complementary shared Mix did not retain root weight");
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto nan_leaves =
      execute_weighted_control(plan, {{mix_id.value, nan}});
  require(nan_leaves.size() == 1u &&
              nan_leaves.front().closure == closure.value &&
              std::isnan(nan_leaves.front().weight),
          "complementary shared Mix incorrectly folded NaN weight to one");
  const auto allocation = storage(analyzed, plan);
  require(allocation.weight_values == 3u && allocation.scalar_slots == 2u,
          "complementary Mix weights were not clique-optimally colored");
  const auto image = lower(analyzed, plan, allocation);
  require(image.mix_instruction_count == 1u &&
              image.weight_add_instruction_count == 1u &&
              image.conditional_branch_count == 0u &&
              image.closure_leaf_count == 1u,
          "unified bytecode changed complementary-Mix cardinality");
  const auto factor_id =
      analyzed.compiled.program->closure_instructions()[mix_id.value].factor;
  const auto factor_address = image.value_addresses[factor_id.value];
  const auto bytecode_leaves =
      execute_weighted_bytecode(image, {{factor_address, 0.37f}});
  require(bytecode_leaves.size() == 1u,
          "unified bytecode duplicated the shared Mix leaf");
  require_weighted_bytecode_leaf(
      bytecode_leaves, ClosureOperation::diffuse, 1.0f,
      "unified bytecode changed complementary finite weight");
  const auto bytecode_nan =
      execute_weighted_bytecode(image, {{factor_address, nan}});
  require(bytecode_nan.size() == 1u &&
              bytecode_nan.front().operation == ClosureOperation::diffuse &&
              std::isnan(bytecode_nan.front().weight),
          "unified bytecode lost complementary Mix NaN semantics");

  auto aliased_mix = image;
  const auto mix_pc = find_bytecode_instruction(
      aliased_mix, SurfaceSvmBytecodeKind::mix_closure);
  const auto left =
      surface_svm_mix_left_weight_slot(aliased_mix.instructions[mix_pc]);
  aliased_mix.instructions[mix_pc].payload2 = left | (left << 16u);
  require(validate_surface_svm_program_image(aliased_mix).find("aliased") !=
              std::string::npos,
          "unified bytecode accepted aliased binary Mix outputs");

  auto wrong_operand_bank = image;
  require(!wrong_operand_bank.closure_operands.empty(),
          "unified diffuse fixture has no closure operands");
  wrong_operand_bank.closure_operands.front() = 0u;
  require(validate_surface_svm_program_image(wrong_operand_bank)
                  .find("closure projection") != std::string::npos,
          "unified bytecode accepted a closure operand in the wrong bank");

  auto incompatible = allocation;
  --incompatible.weight_values;
  require(!incompatible.compatible(*analyzed.compiled.program, plan),
          "storage compatibility ignored an active closure weight");
}

void test_shared_leaf_weight_is_hoisted_and_accumulated_before_control() {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf,
                                      "Hoisted shared diffuse");
  const auto glossy = graph.add_node(node_type::glossy_bsdf,
                                     "Hoisted private glossy");
  const auto mix = graph.add_node(node_type::mix_closure,
                                  "Hoisted child Mix");
  const auto add = graph.add_node(node_type::add_closure,
                                  "Hoisted Add root");
  require(graph.set_input(mix, "Factor", SocketValue::floating(0.25f)) &&
              graph.connect({.node = diffuse, .socket = "Closure"}, mix,
                            "A") &&
              graph.connect({.node = glossy, .socket = "Closure"}, mix,
                            "B") &&
              graph.connect({.node = mix, .socket = "Closure"}, add, "A") &&
              graph.connect({.node = diffuse, .socket = "Closure"}, add,
                            "B"),
          "failed to configure hoisted shared-leaf graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = add, .socket = "Closure"});
  const auto analyzed = analyze(std::move(graph));
  const auto plan = schedule(analyzed);
  verify_total_schedule_contract(*analyzed.compiled.program, plan);
  const auto diffuse_id = find_closure(*analyzed.compiled.program, diffuse);
  const auto glossy_id = find_closure(*analyzed.compiled.program, glossy);
  const auto mix_id = find_closure(*analyzed.compiled.program, mix);
  const auto mix_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::mix_closure, mix_id.value);
  const auto add_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::add_closure_weight);
  const auto diffuse_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::closure_leaf,
      diffuse_id.value);
  const auto guard_pc = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_zero, mix_id.value);
  require(plan.mix_instruction_count == 1u &&
              plan.weight_add_instruction_count == 1u &&
              plan.conditional_branch_count == 1u &&
              mix_pc < add_pc && add_pc < diffuse_pc &&
              diffuse_pc < guard_pc,
          "shared leaf weight/setup was not hoisted before private control");

  const auto at_zero =
      execute_weighted_control(plan, {{mix_id.value, 0.0f}});
  require(at_zero.size() == 1u,
          "zero factor retained the private right closure");
  require_weighted_leaf(at_zero, diffuse_id.value, 2.0f,
                        "zero factor produced the wrong shared-leaf weight");
  const auto interior =
      execute_weighted_control(plan, {{mix_id.value, 0.25f}});
  require(interior.size() == 2u,
          "interior factor did not retain both unique leaves");
  require_weighted_leaf(interior, diffuse_id.value, 1.75f,
                        "interior factor produced the wrong accumulated weight");
  require_weighted_leaf(interior, glossy_id.value, 0.25f,
                        "interior factor produced the wrong private weight");
  const auto at_one =
      execute_weighted_control(plan, {{mix_id.value, 1.0f}});
  require_weighted_leaf(at_one, diffuse_id.value, 1.0f,
                        "factor one lost the unconditional shared contribution");
  require_weighted_leaf(at_one, glossy_id.value, 1.0f,
                        "factor one lost the private right contribution");

  const auto allocation = storage(analyzed, plan);
  require(allocation.weight_values == 3u && allocation.scalar_slots == 2u &&
              allocation.maximum_interference_clique[
                  static_cast<std::size_t>(SurfaceValueBank::scalar)] == 2u,
          "hoisted closure-weight SSA was not clique-optimally colored");
  const auto image = lower(analyzed, plan, allocation);
  const auto bytecode_mix = find_bytecode_instruction(
      image, SurfaceSvmBytecodeKind::mix_closure);
  const auto bytecode_add = find_bytecode_instruction(
      image, SurfaceSvmBytecodeKind::add_closure_weight);
  const auto bytecode_leaf = find_bytecode_instruction(
      image, SurfaceSvmBytecodeKind::closure_leaf);
  const auto bytecode_guard = find_bytecode_instruction(
      image, SurfaceSvmBytecodeKind::jump_if_zero);
  require(bytecode_mix < bytecode_add && bytecode_add < bytecode_leaf &&
              bytecode_leaf < bytecode_guard,
          "unified bytecode lost the proven hoist/control order");
  const auto factor_id =
      analyzed.compiled.program->closure_instructions()[mix_id.value].factor;
  const auto factor_address = image.value_addresses[factor_id.value];
  const auto bytecode_zero =
      execute_weighted_bytecode(image, {{factor_address, 0.0f}});
  require(bytecode_zero.size() == 1u,
          "unified bytecode retained the zero-weight private closure");
  require_weighted_bytecode_leaf(
      bytecode_zero, ClosureOperation::diffuse, 2.0f,
      "unified bytecode changed the zero-factor shared weight");
  const auto bytecode_interior =
      execute_weighted_bytecode(image, {{factor_address, 0.25f}});
  require(bytecode_interior.size() == 2u,
          "unified bytecode lost an interior Mix closure");
  require_weighted_bytecode_leaf(
      bytecode_interior, ClosureOperation::diffuse, 1.75f,
      "unified bytecode changed the interior shared weight");
  require_weighted_bytecode_leaf(
      bytecode_interior, ClosureOperation::glossy, 0.25f,
      "unified bytecode changed the interior private weight");

  auto backward_guard = image;
  backward_guard.instructions[bytecode_guard].payload1 =
      static_cast<std::uint32_t>(bytecode_guard);
  require(validate_surface_svm_program_image(backward_guard)
                  .find("forward") != std::string::npos,
          "unified bytecode accepted a non-forward closure guard");
}

void test_cfg_storage_is_clique_optimal_and_read_before_write() {
  auto fixture = make_conditional_graph();
  const auto shared_node = fixture.shared_value;
  const auto left_node = fixture.left_value;
  const auto right_node = fixture.right_value;
  auto analyzed = analyze(std::move(fixture.graph));
  const auto plan = schedule(analyzed);
  const auto allocation = storage(analyzed, plan);
  const auto &program = *analyzed.compiled.program;
  const auto shared = find_value(program, shared_node, ValueOperation::mix);
  const auto left = find_value(program, left_node, ValueOperation::mix);
  const auto right = find_value(program, right_node, ValueOperation::mix);
  const auto &shared_location = allocation.locations[shared.value];
  const auto &left_location = allocation.locations[left.value];
  const auto &right_location = allocation.locations[right.value];

  require(allocation.vector_slots == 2u &&
              allocation.maximum_interference_clique[static_cast<std::size_t>(
                  SurfaceValueBank::vector)] == 2u &&
              shared_location.storage == SurfaceValueStorageClass::local_slot &&
              left_location.storage == SurfaceValueStorageClass::local_slot &&
              right_location.storage == SurfaceValueStorageClass::local_slot &&
              shared_location.index != left_location.index &&
              shared_location.index == right_location.index,
          "CFG coloring missed exact interference or last-use slot donation");

  const auto physical_plan = schedule(
      analyzed, surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical));
  const auto physical = storage(analyzed, physical_plan);
  require(
      physical.vector_slots == 1u &&
          physical.locations[shared.value].index ==
              physical.locations[left.value].index &&
          physical.locations[right.value].storage ==
              SurfaceValueStorageClass::inactive,
      "one-sided Mix projection did not collapse its value chain to one slot");
}

void test_interleaved_closure_use_reuses_slots_across_branches() {
  ShaderGraph graph;
  const auto left_a =
      graph.add_node(node_type::constant_color, "Storage left A");
  const auto left_b =
      graph.add_node(node_type::constant_color, "Storage left B");
  const auto right_a =
      graph.add_node(node_type::constant_color, "Storage right A");
  const auto right_b =
      graph.add_node(node_type::constant_color, "Storage right B");
  const auto left_value =
      graph.add_node(node_type::mix_color, "Storage left value");
  const auto right_value =
      graph.add_node(node_type::mix_color, "Storage right value");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Storage diffuse");
  const auto emission = graph.add_node(node_type::emission, "Storage emission");
  const auto mix =
      graph.add_node(node_type::mix_closure, "Storage closure Mix");
  require(
      graph.connect({.node = left_a, .socket = "Color"}, left_value, "A") &&
          graph.connect({.node = left_b, .socket = "Color"}, left_value, "B") &&
          graph.connect({.node = right_a, .socket = "Color"}, right_value,
                        "A") &&
          graph.connect({.node = right_b, .socket = "Color"}, right_value,
                        "B") &&
          graph.connect({.node = left_value, .socket = "Color"}, diffuse,
                        "Color") &&
          graph.connect({.node = right_value, .socket = "Color"}, emission,
                        "Color") &&
          graph.set_input(mix, "Factor", SocketValue::floating(0.5f)) &&
          graph.connect({.node = diffuse, .socket = "Closure"}, mix, "A") &&
          graph.connect({.node = emission, .socket = "Closure"}, mix, "B"),
      "failed to configure independent branch storage graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});
  const auto analyzed = analyze(std::move(graph));
  const auto plan = schedule(analyzed);
  const auto allocation = storage(analyzed, plan);
  const auto left =
      find_value(*analyzed.compiled.program, left_value, ValueOperation::mix);
  const auto right =
      find_value(*analyzed.compiled.program, right_value, ValueOperation::mix);
  require(allocation.vector_slots == 1u &&
              allocation.locations[left.value].index ==
                  allocation.locations[right.value].index,
          "closure-interleaved liveness retained both branch outputs");

  const auto legacy = plan_surface_value_storage(
      *analyzed.compiled.program, analyzed.dependencies.preparation,
      analyzed.dependencies.preparation_outputs);
  require(legacy.valid && legacy.vector_slots >= 2u &&
              allocation.vector_slots < legacy.vector_slots,
          "fixture did not prove a storage reduction over the split stream");
}

void test_passthrough_quotient_has_no_device_definition() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Storage geometry");
  const auto point_to_vector =
      graph.add_node(node_type::point_to_vector, "Storage point alias");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Storage color alias");
  const auto emission =
      graph.add_node(node_type::emission, "Storage alias emission");
  require(graph.connect({.node = geometry, .socket = "Position"},
                        point_to_vector, "Point") &&
              graph.connect({.node = point_to_vector, .socket = "Vector"},
                            vector_to_color, "Vector") &&
              graph.connect({.node = vector_to_color, .socket = "Color"},
                            emission, "Color"),
          "failed to configure Passthrough quotient graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const auto analyzed = analyze(std::move(graph));
  const auto plan = schedule(analyzed);
  const auto allocation = storage(analyzed, plan);
  const auto point = find_value(*analyzed.compiled.program, point_to_vector,
                                ValueOperation::passthrough);
  const auto color = find_value(*analyzed.compiled.program, vector_to_color,
                                ValueOperation::passthrough);
  const auto &point_location = allocation.locations[point.value];
  const auto &color_location = allocation.locations[color.value];
  require(allocation.alias_values >= 2u &&
              allocation.representatives[point.value] ==
                  allocation.representatives[color.value] &&
              point_location.storage == color_location.storage &&
              point_location.bank == color_location.bank &&
              point_location.index == color_location.index &&
              allocation.vector_slots == 1u,
          "same-bank Passthroughs were not contracted before CFG liveness");
  const auto image = lower(analyzed, plan, allocation);
  require(image.value_instruction_count + allocation.alias_values ==
              plan.value_instruction_count,
          "unified bytecode did not erase exactly the Passthrough quotient");
}

void test_unified_jump_targets_cross_erased_aliases_exactly() {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Guarded alias geometry");
  const auto point_to_vector =
      graph.add_node(node_type::point_to_vector, "Guarded point alias");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Guarded color alias");
  const auto roughness =
      graph.add_node(node_type::math, "Guarded scalar roughness");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Guarded alias diffuse");
  const auto emission =
      graph.add_node(node_type::emission, "Guarded alias emission");
  const auto mix =
      graph.add_node(node_type::mix_closure, "Guarded alias Mix");
  require(
      graph.connect({.node = geometry, .socket = "Position"},
                    point_to_vector, "Point") &&
          graph.connect({.node = point_to_vector, .socket = "Vector"},
                        vector_to_color, "Vector") &&
          graph.connect({.node = vector_to_color, .socket = "Color"},
                        diffuse, "Color") &&
          graph.set_property(roughness, "Operation",
                             SocketValue::string("ADD")) &&
          graph.set_input(roughness, "B", SocketValue::floating(0.2f)) &&
          graph.connect({.node = geometry, .socket = "Backfacing"},
                        roughness, "A") &&
          graph.connect({.node = roughness, .socket = "Value"}, diffuse,
                        "Roughness") &&
          graph.connect({.node = roughness, .socket = "Value"}, mix,
                        "Factor") &&
          graph.connect({.node = diffuse, .socket = "Closure"}, mix, "A") &&
          graph.connect({.node = emission, .socket = "Closure"}, mix, "B"),
      "failed to configure guarded Passthrough graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});
  const auto analyzed = analyze(std::move(graph));
  const auto plan = schedule(analyzed);
  const auto allocation = storage(analyzed, plan);
  require(allocation.alias_values >= 2u,
          "guarded alias fixture did not form a quotient");
  const auto image = lower(analyzed, plan, allocation);
  const auto left_guard = find_bytecode_instruction(
      image, SurfaceSvmBytecodeKind::jump_if_one);
  const auto right_guard = find_bytecode_instruction(
      image, SurfaceSvmBytecodeKind::jump_if_zero);
  require(left_guard < right_guard &&
              image.instructions[left_guard].payload1 == right_guard,
          "unified guard target was not remapped across erased aliases");

  const auto mix_id = find_closure(*analyzed.compiled.program, mix);
  const auto factor_id =
      analyzed.compiled.program->closure_instructions()[mix_id.value].factor;
  const auto factor_address = image.value_addresses[factor_id.value];
  const auto at_zero =
      execute_weighted_bytecode(image, {{factor_address, 0.0f}});
  require(at_zero.size() == 1u,
          "unified zero-factor alias path executed two leaves");
  require_weighted_bytecode_leaf(
      at_zero, ClosureOperation::diffuse, 1.0f,
      "unified zero-factor alias path selected the wrong leaf");
  const auto at_one =
      execute_weighted_bytecode(image, {{factor_address, 1.0f}});
  require(at_one.size() == 1u,
          "unified one-factor alias path executed two leaves");
  require_weighted_bytecode_leaf(
      at_one, ClosureOperation::emission, 1.0f,
      "unified one-factor alias path selected the wrong leaf");

  const auto roughness_id = find_value(
      *analyzed.compiled.program, roughness, ValueOperation::math);
  const auto roughness_address =
      SurfaceValueAddress{image.value_addresses[roughness_id.value]};
  require(roughness_address.valid() && !roughness_address.parameter() &&
              roughness_address.bank() == SurfaceValueBank::scalar,
          "guarded scalar fixture has no local roughness address");
  auto value_as_weight = image;
  const auto first_leaf = find_bytecode_instruction(
      value_as_weight, SurfaceSvmBytecodeKind::closure_leaf);
  value_as_weight.instructions[first_leaf].payload1 =
      roughness_address.index();
  require(validate_surface_svm_program_image(value_as_weight)
                  .find("undefined weight") != std::string::npos,
          "unified bytecode confused a scalar value with a weight slot");

  auto weight_as_value = image;
  const auto mix_pc = find_bytecode_instruction(
      weight_as_value, SurfaceSvmBytecodeKind::mix_closure);
  const auto weight_slot = surface_svm_mix_left_weight_slot(
      weight_as_value.instructions[mix_pc]);
  weight_as_value.instructions[right_guard].payload0 =
      static_cast<std::uint32_t>(SurfaceValueBank::scalar)
          << SurfaceValueAddress::bank_shift |
      weight_slot;
  require(validate_surface_svm_program_image(weight_as_value)
                  .find("undefined value") != std::string::npos,
          "unified bytecode confused a weight slot with a scalar value");
}

void test_storage_rejects_skipped_definition_and_capacity_overflow() {
  auto fixture = make_conditional_graph();
  const auto mix_node = fixture.mix;
  auto analyzed = analyze(std::move(fixture.graph));
  auto plan = schedule(
      analyzed, surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical));
  const auto mix = find_closure(*analyzed.compiled.program, mix_node);
  const auto guard = find_schedule_instruction(
      plan, SurfaceSvmScheduleInstructionKind::jump_if_one, mix.value);

  auto skipped_definition = plan;
  auto leaf_pc = std::size_t{};
  while (leaf_pc < skipped_definition.instructions.size() &&
         skipped_definition.instructions[leaf_pc].kind !=
             SurfaceSvmScheduleInstructionKind::closure_leaf) {
    ++leaf_pc;
  }
  require(leaf_pc < skipped_definition.instructions.size(),
          "malformed-storage fixture has no closure leaf");
  skipped_definition.instructions[guard].target =
      static_cast<std::uint32_t>(leaf_pc);
  const auto invalid =
      plan_surface_svm_storage(*analyzed.compiled.program, skipped_definition);
  require(!invalid.valid &&
              invalid.diagnostic.find("does not dominate") != std::string::npos,
          "CFG storage accepted a path that skips a required definition");

  auto mismatched_factor = plan;
  const auto mix_pc = find_schedule_instruction(
      mismatched_factor, SurfaceSvmScheduleInstructionKind::mix_closure);
  const auto mix_weight = mismatched_factor.instructions[mix_pc].weight;
  require(mix_weight.valid() &&
              mix_weight.value < mismatched_factor.weight_expressions.size(),
          "malformed-factor fixture has no Mix weight");
  const auto actual_factor =
      analyzed.compiled.program->closure_instructions()[mix.value].factor;
  const auto alternative = ValueExpressionId{
      actual_factor.value == 0u ? 1u : 0u};
  require(alternative.value <
              analyzed.compiled.program->value_instructions().size(),
          "malformed-factor fixture has no alternative value");
  mismatched_factor.weight_expressions[mix_weight.value].factor = alternative;
  const auto inconsistent = plan_surface_svm_storage(
      *analyzed.compiled.program, mismatched_factor);
  require(!inconsistent.valid &&
              inconsistent.diagnostic.find("inconsistent primary") !=
                  std::string::npos,
          "CFG storage accepted a Mix weight with the wrong source factor");

  const auto insufficient = plan_surface_svm_storage(
      *analyzed.compiled.program, plan,
      SurfaceValueStorageCapacity{
          .scalar_slots = std::numeric_limits<std::uint32_t>::max(),
          .vector_slots = 0u,
          .unsigned_integer_slots = std::numeric_limits<std::uint32_t>::max()});
  require(!insufficient.valid &&
              insufficient.diagnostic.find("capacity") != std::string::npos,
          "CFG storage ignored a component-wise typed capacity limit");
}

void test_nine_scalar_clique_uses_cycles_lane_stack_not_legacy_bank_limit() {
  constexpr auto value_count = 9u;
  std::vector<ValueInstruction> values;
  values.reserve(value_count);
  for (auto index = std::uint32_t{}; index < value_count; ++index) {
    values.emplace_back(ValueInstruction{
        .operation = ValueOperation::path_ray_length,
        .source_node = NodeId{index + 1u},
        .result_type = SocketType::floating});
  }
  ClosureInstruction closure{.operation = ClosureOperation::principled};
  closure.roughness = ValueExpressionId{0u};
  closure.diffuse_roughness = ValueExpressionId{1u};
  closure.subsurface_weight = ValueExpressionId{2u};
  closure.subsurface_scale = ValueExpressionId{3u};
  closure.subsurface_ior = ValueExpressionId{4u};
  closure.subsurface_anisotropy = ValueExpressionId{5u};
  closure.transmission_weight = ValueExpressionId{6u};
  closure.metallic = ValueExpressionId{7u};
  closure.ior = ValueExpressionId{8u};
  const SurfaceProgram program{
      0x9a11u, {}, std::move(values), {closure}, ClosureExpressionId{0u}};

  SurfaceSvmSchedulePlan schedule;
  schedule.valid = true;
  schedule.endpoints =
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical);
  schedule.value_regions.assign(value_count, 0u);
  schedule.region_count = 1u;
  schedule.value_instruction_count = value_count;
  schedule.closure_leaf_count = 1u;
  for (auto index = std::uint32_t{}; index < value_count; ++index) {
    schedule.instructions.emplace_back(SurfaceSvmScheduleInstruction{
        .kind = SurfaceSvmScheduleInstructionKind::value,
        .source = index});
  }
  schedule.instructions.emplace_back(SurfaceSvmScheduleInstruction{
      .kind = SurfaceSvmScheduleInstructionKind::closure_leaf,
      .source = 0u});
  schedule.instructions.emplace_back(SurfaceSvmScheduleInstruction{
      .kind = SurfaceSvmScheduleInstructionKind::end});

  const auto cycles_capacity = SurfaceValueStorageCapacity{
      .scalar_slots = surface_svm_stack_lane_capacity,
      .vector_slots = surface_svm_stack_lane_capacity / 3u,
      .unsigned_integer_slots = surface_svm_stack_lane_capacity / 2u,
      .stack_lanes = surface_svm_stack_lane_capacity};
  const auto allocation =
      plan_surface_svm_storage(program, schedule, cycles_capacity);
  require(allocation.valid && allocation.scalar_slots == value_count &&
              allocation.maximum_interference_clique[0u] == value_count &&
              allocation.stack_lanes == value_count &&
              allocation.lane_bases == std::array<std::uint32_t, 3u>{0u, 9u,
                                                                     9u} &&
              allocation.payload_bytes() == value_count * sizeof(float),
          "a nine-scalar live clique was not represented by nine physical "
          "Cycles SVM lanes");

  auto legacy_limit = cycles_capacity;
  legacy_limit.scalar_slots = 8u;
  const auto rejected_bank =
      plan_surface_svm_storage(program, schedule, legacy_limit);
  require(!rejected_bank.valid &&
              rejected_bank.diagnostic.find("scalar bank requires 9") !=
                  std::string::npos,
          "the regression fixture no longer proves the removed eight-scalar "
          "bank failure");

  auto short_stack = cycles_capacity;
  short_stack.stack_lanes = 8u;
  const auto rejected_stack =
      plan_surface_svm_storage(program, schedule, short_stack);
  require(!rejected_stack.valid &&
              rejected_stack.diagnostic.find("9 lanes are required") !=
                  std::string::npos,
          "the physical lane capacity did not reject an exact one-lane "
          "overflow");
}

} // namespace

int main() {
  try {
    test_shared_values_are_hoisted_and_private_values_are_guarded();
    test_endpoint_projection_moves_shared_work_inside_the_live_guard();
    test_nested_guards_have_exact_forward_targets();
    test_add_sibling_is_outside_mix_control();
    test_static_mix_pruning_removes_control_records();
    test_shared_closure_dag_is_accepted_without_value_recomputation();
    test_complementary_shared_mix_leaf_preserves_ieee_weight_algebra();
    test_shared_leaf_weight_is_hoisted_and_accumulated_before_control();
    test_cfg_storage_is_clique_optimal_and_read_before_write();
    test_interleaved_closure_use_reuses_slots_across_branches();
    test_passthrough_quotient_has_no_device_definition();
    test_unified_jump_targets_cross_erased_aliases_exactly();
    test_storage_rejects_skipped_definition_and_capacity_overflow();
    test_nine_scalar_clique_uses_cycles_lane_stack_not_legacy_bank_limit();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "surface SVM schedule tests passed\n";
  return 0;
}
