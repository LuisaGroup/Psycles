#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
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
  require(shader.ok(), "closure-bytecode fixture failed to compile");
  const auto lowered = compile_surface_program(*shader.program);
  require(lowered.ok(), "closure-bytecode fixture failed to lower");
  auto binding = bind_surface_parameters(*lowered.program, *shader.program);
  require(binding.ok(), "closure-bytecode fixture failed to bind");
  return {
      .program = lowered.program,
      .parameters = std::move(*binding.parameters)};
}

struct PlannedProgram {
  SurfaceClosurePlan closure_plan;
  SurfaceValueDependencyPlan dependencies;
  SurfaceValueProgramImage values;
  SurfaceClosureProgramImage closures;
};

[[nodiscard]] PlannedProgram plan_program(const CompiledGraph &compiled) {
  auto closure_plan = analyze_surface_closure_plan(
      *compiled.program, compiled.parameters);
  auto dependencies = analyze_surface_value_dependencies(
      *compiled.program, closure_plan);
  auto storage = plan_surface_value_storage(
      *compiled.program,
      dependencies.preparation,
      dependencies.preparation_outputs);
  require(storage.valid, "closure-bytecode value storage plan failed");
  auto values = lower_surface_value_program(*compiled.program, storage);
  require(values.valid, "closure-bytecode value lowering failed");
  auto closures = lower_surface_closure_program(
      *compiled.program,
      closure_plan,
      dependencies,
      values.value_addresses);
  require(closures.valid,
          "closure-bytecode lowering failed: " + closures.diagnostic);
  return {
      .closure_plan = std::move(closure_plan),
      .dependencies = std::move(dependencies),
      .values = std::move(values),
      .closures = std::move(closures)};
}

[[nodiscard]] ShaderGraph make_domain_mix_graph(float factor) {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(
      node_type::diffuse_bsdf, "Physical closure leaf");
  const auto emission = graph.add_node(
      node_type::emission, "Emission closure leaf");
  const auto mix = graph.add_node(
      node_type::mix_closure, "Physical/emission Mix");
  require(
      graph.set_input(
          diffuse,
          "Color",
          SocketValue::color({0.2f, 0.4f, 0.7f})) &&
          graph.set_input(
              emission,
              "Color",
              SocketValue::color({0.8f, 0.3f, 0.1f})) &&
          graph.set_input(
              emission,
              "Strength",
              SocketValue::floating(2.0f)) &&
          graph.set_input(
              mix,
              "Factor",
              SocketValue::floating(factor)) &&
          graph.connect(
              {.node = diffuse, .socket = "Closure"}, mix, "A") &&
          graph.connect(
              {.node = emission, .socket = "Closure"}, mix, "B"),
      "failed to configure closure-bytecode Mix graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = mix, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_nested_mix_graph() {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(
      node_type::diffuse_bsdf, "Nested Mix diffuse");
  const auto glossy = graph.add_node(
      node_type::glossy_bsdf, "Nested Mix glossy");
  const auto emission = graph.add_node(
      node_type::emission, "Nested Mix emission");
  const auto inner = graph.add_node(
      node_type::mix_closure, "Nested physical Mix");
  const auto outer = graph.add_node(
      node_type::mix_closure, "Nested domain Mix");
  require(
      graph.set_input(
          inner, "Factor", SocketValue::floating(0.2f)) &&
          graph.set_input(
              outer, "Factor", SocketValue::floating(0.7f)) &&
          graph.connect(
              {.node = diffuse, .socket = "Closure"}, inner, "A") &&
          graph.connect(
              {.node = glossy, .socket = "Closure"}, inner, "B") &&
          graph.connect(
              {.node = inner, .socket = "Closure"}, outer, "A") &&
          graph.connect(
              {.node = emission, .socket = "Closure"}, outer, "B"),
      "failed to configure nested closure-bytecode graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = outer, .socket = "Closure"});
  return graph;
}

[[nodiscard]] const ClosureInstruction &find_closure(
    const SurfaceProgram &program, ClosureOperation operation) {
  for (const auto &closure : program.closure_instructions()) {
    if (closure.operation == operation) {
      return closure;
    }
  }
  throw std::runtime_error{"fixture closure operation is absent"};
}

void test_mixed_endpoint_flattening() {
  const auto compiled = compile_graph(make_domain_mix_graph(0.25f));
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 2u &&
              image.principled_features.size() == 2u,
          "Mix did not flatten to exactly two parallel leaves");
  require(surface_closure_operation(image.instructions[0u]) ==
                  ClosureOperation::diffuse &&
              surface_closure_operation(image.instructions[1u]) ==
                  ClosureOperation::emission,
          "flattened closure order is not left-to-right DFS");
  require(surface_closure_endpoints(image.instructions[0u]) ==
                  surface_closure_endpoint_bit(
                      SurfaceClosureEndpoint::physical) &&
              surface_closure_endpoints(image.instructions[1u]) ==
                  surface_closure_endpoint_bit(
                      SurfaceClosureEndpoint::emission),
          "closure consumer domains were not preserved");
  require(image.instructions[0u].mix_term_count == 1u &&
              image.instructions[1u].mix_term_count == 1u &&
              image.mix_terms.size() == 2u &&
              image.maximum_mix_depth == 1u,
          "live Mix branches did not retain one factor term");

  const auto &mix = find_closure(
      *compiled.program, ClosureOperation::mix);
  const auto factor_address =
      planned.values.value_addresses[mix.factor.value];
  const auto &left_term =
      image.mix_terms[image.instructions[0u].mix_term_begin];
  const auto &right_term =
      image.mix_terms[image.instructions[1u].mix_term_begin];
  require(left_term.address == factor_address &&
              left_term.flags == surface_closure_mix_complement &&
              right_term.address == factor_address &&
              right_term.flags == 0u,
          "Mix polarity/address encoding changed its weight relation");
  require(image.operands.size() ==
              surface_closure_operand::diffuse::count +
                  surface_closure_operand::emission::count &&
              image.instructions[0u].operand_begin == 0u &&
              image.instructions[1u].operand_begin ==
                  surface_closure_operand::diffuse::count,
          "closure operand stream is not densely semantic");
}

void test_nested_mix_path_recurrence() {
  const auto compiled = compile_graph(make_nested_mix_graph());
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 3u &&
              surface_closure_operation(image.instructions[0u]) ==
                  ClosureOperation::diffuse &&
              surface_closure_operation(image.instructions[1u]) ==
                  ClosureOperation::glossy &&
              surface_closure_operation(image.instructions[2u]) ==
                  ClosureOperation::emission,
          "nested Mix did not preserve exact left-to-right leaf order");

  const auto outer_id = compiled.program->root();
  require(outer_id.valid() &&
              compiled.program->closure_instructions()[outer_id.value]
                      .operation == ClosureOperation::mix,
          "nested Mix fixture has no Mix root");
  auto inner_id = ClosureExpressionId{};
  for (auto index = std::size_t{0u};
       index < compiled.program->closure_instructions().size(); ++index) {
    if (index != outer_id.value &&
        compiled.program->closure_instructions()[index].operation ==
            ClosureOperation::mix) {
      inner_id = ClosureExpressionId{static_cast<std::uint32_t>(index)};
      break;
    }
  }
  require(inner_id.valid(), "nested Mix fixture has no inner Mix");
  const auto outer_factor = planned.values.value_addresses[
      compiled.program->closure_instructions()[outer_id.value]
          .factor.value];
  const auto inner_factor = planned.values.value_addresses[
      compiled.program->closure_instructions()[inner_id.value]
          .factor.value];

  require(image.instructions[0u].mix_term_count == 2u &&
              image.instructions[1u].mix_term_count == 2u &&
              image.instructions[2u].mix_term_count == 1u &&
              image.maximum_mix_depth == 2u,
          "nested Mix path depth is inconsistent");
  const auto term = [&](std::size_t leaf, std::size_t depth)
      -> const SurfaceClosureMixTerm & {
    return image.mix_terms[
        image.instructions[leaf].mix_term_begin + depth];
  };
  require(term(0u, 0u).address == outer_factor &&
              term(0u, 0u).flags == surface_closure_mix_complement &&
              term(0u, 1u).address == inner_factor &&
              term(0u, 1u).flags == surface_closure_mix_complement &&
              term(1u, 0u).address == outer_factor &&
              term(1u, 0u).flags == surface_closure_mix_complement &&
              term(1u, 1u).address == inner_factor &&
              term(1u, 1u).flags == 0u &&
              term(2u, 0u).address == outer_factor &&
              term(2u, 0u).flags == 0u,
          "nested Mix path no longer encodes the weight recurrence");
}

void test_statically_pruned_mix() {
  const auto compiled = compile_graph(make_domain_mix_graph(0.0f));
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 1u &&
              surface_closure_operation(image.instructions.front()) ==
                  ClosureOperation::diffuse &&
              image.instructions.front().mix_term_count == 0u &&
              image.mix_terms.empty(),
          "a statically unreachable Mix branch or factor survived");
  require(image.used_operations ==
              (1u << static_cast<std::uint32_t>(
                   ClosureOperation::diffuse)),
          "scene operation mask includes a pruned closure family");
}

[[nodiscard]] ShaderGraph make_principled_graph() {
  ShaderGraph graph;
  const auto principled = graph.add_node(
      node_type::principled_bsdf, "All-feature Principled closure");
  require(
      graph.set_input(
          principled,
          "BaseColor",
          SocketValue::color({0.35f, 0.22f, 0.11f})) &&
          graph.set_input(
              principled, "Alpha", SocketValue::floating(0.8f)) &&
          graph.set_input(
              principled,
              "SheenWeight",
              SocketValue::floating(0.2f)) &&
          graph.set_input(
              principled,
              "CoatWeight",
              SocketValue::floating(0.3f)) &&
          graph.set_input(
              principled,
              "Metallic",
              SocketValue::floating(0.4f)) &&
          graph.set_input(
              principled,
              "TransmissionWeight",
              SocketValue::floating(0.5f)) &&
          graph.set_input(
              principled,
              "SubsurfaceWeight",
              SocketValue::floating(0.6f)) &&
          graph.set_input(
              principled,
              "EmissionColor",
              SocketValue::color({0.1f, 0.2f, 0.3f})) &&
          graph.set_input(
              principled,
              "EmissionStrength",
              SocketValue::floating(1.7f)),
      "failed to configure closure-bytecode Principled graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = principled, .socket = "Closure"});
  return graph;
}

void test_principled_static_contract() {
  const auto compiled = compile_graph(make_principled_graph());
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 1u &&
              image.principled_features.size() == 1u,
          "Principled graph did not lower to one data instruction");
  const auto &instruction = image.instructions.front();
  require(surface_closure_operation(instruction) ==
                  ClosureOperation::principled &&
              surface_closure_endpoints(instruction) ==
                  (surface_closure_endpoint_bit(
                       SurfaceClosureEndpoint::physical) |
                   surface_closure_endpoint_bit(
                       SurfaceClosureEndpoint::emission)) &&
              instruction.mix_term_count == 0u &&
              image.operands.size() ==
                  surface_closure_operand::principled::count,
          "Principled opcode/endpoints/operand ABI is inconsistent");

  const auto root = compiled.program->root();
  const auto expected_features =
      planned.closure_plan.entry(root).principled_features;
  require(image.principled_features.front() == expected_features &&
              image.used_principled_features == expected_features,
          "Principled feature mask was not preserved as scene data");
  require(surface_closure_bssrdf_method(instruction) ==
              compiled.program->closure_instructions()[root.value]
                  .subsurface_method,
          "BSSRDF method was omitted from closure control");
  require((instruction.control & ~surface_closure_control_mask) == 0u,
          "closure control contains undefined bits");
}

void test_missing_live_address_rejected() {
  const auto compiled = compile_graph(make_domain_mix_graph(0.25f));
  auto planned = plan_program(compiled);
  const auto &mix = find_closure(
      *compiled.program, ClosureOperation::mix);
  planned.values.value_addresses[mix.factor.value] =
      SurfaceValueAddress::invalid_value;
  const auto invalid = lower_surface_closure_program(
      *compiled.program,
      planned.closure_plan,
      planned.dependencies,
      planned.values.value_addresses);
  require(!invalid.valid &&
              invalid.diagnostic.find("factor address") !=
                  std::string::npos,
          "missing live Mix address was not rejected transactionally");
}

void test_executable_scene_relocation() {
  const auto compiled = compile_graph(make_domain_mix_graph(0.25f));
  const auto planned = plan_program(compiled);
  const auto storage = plan_surface_value_storage(
      *compiled.program,
      planned.dependencies.preparation,
      planned.dependencies.preparation_outputs);
  require(storage.valid, "scene-relocation value storage plan failed");

  const auto inputs = std::vector{
      SurfaceValueExecutionInput{
          .program = compiled.program.get(),
          .storage = &storage,
          .closure_plan = &planned.closure_plan},
      SurfaceValueExecutionInput{
          .program = compiled.program.get(),
          .storage = &storage,
          .closure_plan = &planned.closure_plan}};
  const auto executable = build_surface_value_executable_scene(inputs);
  require(executable.valid,
          "scene closure aggregation failed: " + executable.diagnostic);
  const auto &scene = executable.values;
  require(scene.programs.size() == 2u &&
              scene.programs[0u].closure_begin == 0u &&
              scene.programs[0u].closure_count == 2u &&
              scene.programs[1u].closure_begin == 2u &&
              scene.programs[1u].closure_count == 2u,
          "closure ranges are not published in program descriptors");
  require(scene.closure_instructions.size() == 4u &&
              scene.closure_principled_features.size() == 4u &&
              scene.closure_operands.size() == 10u &&
              scene.closure_mix_terms.size() == 4u,
          "parallel closure streams changed size during aggregation");
  require(scene.closure_instructions[2u].operand_begin == 5u &&
              scene.closure_instructions[2u].mix_term_begin == 2u,
          "the second closure program was not rebased exactly once");
  require(scene.maximum_closure_mix_depth == 1u &&
              scene.used_closure_operations ==
                  ((1u << static_cast<std::uint32_t>(
                        ClosureOperation::diffuse)) |
                   (1u << static_cast<std::uint32_t>(
                        ClosureOperation::emission))),
          "scene-wide closure specialization masks are inconsistent");

  const auto inactive = std::vector<bool>(
      compiled.program->value_instructions().size(), false);
  const auto wrong_storage = plan_surface_value_storage(
      *compiled.program, inactive, inactive);
  require(wrong_storage.valid,
          "empty diagnostic storage plan unexpectedly failed");
  const auto mismatched = build_surface_value_executable_scene(
      std::vector{SurfaceValueExecutionInput{
          .program = compiled.program.get(),
          .storage = &wrong_storage,
          .closure_plan = &planned.closure_plan}});
  require(!mismatched.valid &&
              mismatched.diagnostic.find("closure lowering") !=
                  std::string::npos,
          "a closure plan paired with the wrong typed storage was accepted");
}

} // namespace

int main() {
  try {
    test_mixed_endpoint_flattening();
    test_nested_mix_path_recurrence();
    test_statically_pruned_mix();
    test_principled_static_contract();
    test_missing_live_address_rejected();
    test_executable_scene_relocation();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "surface closure execution-plan tests passed\n";
  return 0;
}
