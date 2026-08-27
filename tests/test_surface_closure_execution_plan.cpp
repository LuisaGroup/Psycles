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
#include <string_view>
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

[[nodiscard]] ShaderGraph make_add_after_mix_graph() {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(
      node_type::diffuse_bsdf, "Add-after-Mix diffuse");
  const auto glossy = graph.add_node(
      node_type::glossy_bsdf, "Add-after-Mix glossy");
  const auto emission = graph.add_node(
      node_type::emission, "Add-after-Mix sibling");
  const auto mix = graph.add_node(
      node_type::mix_closure, "Add-after-Mix branch");
  const auto add = graph.add_node(
      node_type::add_closure, "Add-after-Mix root");
  require(
      graph.set_input(mix, "Factor", SocketValue::floating(0.25f)) &&
          graph.connect(
              {.node = diffuse, .socket = "Closure"}, mix, "A") &&
          graph.connect(
              {.node = glossy, .socket = "Closure"}, mix, "B") &&
          graph.connect(
              {.node = mix, .socket = "Closure"}, add, "A") &&
          graph.connect(
              {.node = emission, .socket = "Closure"}, add, "B"),
      "failed to configure Add-after-Mix closure graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = add, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_normal_dependent_emission_graph() {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry, "Emission Geometry normal");
    const auto normal_to_vector = graph.add_node(
        node_type::normal_to_vector,
        "Emission normal to vector");
    const auto vector_to_color = graph.add_node(
        node_type::vector_to_color,
        "Emission normal to color");
    const auto emission = graph.add_node(
        node_type::emission, "Normal-dependent emission");
    require(
        graph.connect(
            {.node = geometry, .socket = "Normal"},
            normal_to_vector,
            "Normal") &&
            graph.connect(
                {.node = normal_to_vector, .socket = "Vector"},
                vector_to_color,
                "Vector") &&
            graph.connect(
                {.node = vector_to_color, .socket = "Color"},
                emission,
                "Color") &&
            graph.set_input(
                emission,
                "Strength",
                SocketValue::floating(1.0f)),
        "failed to configure normal-dependent emission graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
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

void test_mixed_endpoint_linear_weight_program() {
  const auto compiled = compile_graph(make_domain_mix_graph(0.25f));
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 3u &&
              image.principled_features.size() == 3u,
          "root Mix did not lower to one weight instruction and two leaves");
  require(surface_closure_instruction_kind(image.instructions[0u]) ==
                  SurfaceClosureInstructionKind::mix_both &&
              surface_closure_instruction_kind(image.instructions[1u]) ==
                  SurfaceClosureInstructionKind::leaf &&
              surface_closure_instruction_kind(image.instructions[2u]) ==
                  SurfaceClosureInstructionKind::leaf,
          "Mix weight definitions are not topologically ordered");
  require(surface_closure_operation(image.instructions[1u]) ==
                  ClosureOperation::diffuse &&
              surface_closure_operation(image.instructions[2u]) ==
                  ClosureOperation::emission,
          "linear closure order is not left-to-right DFS");
  require(surface_closure_endpoints(image.instructions[1u]) ==
                  surface_closure_endpoint_bit(
                      SurfaceClosureEndpoint::physical) &&
              surface_closure_endpoints(image.instructions[2u]) ==
                  surface_closure_endpoint_bit(
                      SurfaceClosureEndpoint::emission),
          "closure consumer domains were not preserved");

  const auto &mix = find_closure(
      *compiled.program, ClosureOperation::mix);
  const auto factor_address =
      planned.values.value_addresses[mix.factor.value];
  require(surface_closure_mix_factor_address(image.instructions[0u]) ==
                  factor_address &&
              surface_closure_mix_parent_weight_slot(
                  image.instructions[0u]) ==
                  surface_closure_root_weight_slot &&
              surface_closure_mix_left_weight_slot(
                  image.instructions[0u]) == 0u &&
              surface_closure_mix_right_weight_slot(
                  image.instructions[0u]) == 1u &&
              surface_closure_leaf_weight_slot(image.instructions[1u]) == 0u &&
              surface_closure_leaf_weight_slot(image.instructions[2u]) == 1u &&
              image.maximum_mix_depth == 1u && image.mix_slots == 2u,
          "binary Mix recurrence or exact weight-slot coloring changed");
  require(image.operands.size() ==
              surface_closure_operand::diffuse::count +
                  surface_closure_operand::emission::count &&
              surface_closure_leaf_operand_begin(image.instructions[1u]) == 0u &&
              surface_closure_leaf_operand_begin(image.instructions[2u]) ==
                  surface_closure_operand::diffuse::count,
          "closure operand stream is not densely semantic");
}

void test_endpoint_projection() {
  const auto compiled = compile_graph(make_domain_mix_graph(0.25f));
  const auto closure_plan =
      analyze_surface_closure_plan(*compiled.program, compiled.parameters);
  const auto dependencies =
      analyze_surface_value_dependencies(*compiled.program, closure_plan);
  require(!dependencies.emission_values_observe_shading_normal &&
              !dependencies.emission_closures_observe_shading_normal &&
              !dependencies.emission_observes_shading_normal(),
          "plain Emission acquired a false SetNormal dependence");
  const auto emission_storage = plan_surface_value_storage(
      *compiled.program, dependencies.emission, dependencies.emission_outputs);
  require(emission_storage.valid, "emission-only value storage plan failed");
  const auto emission_values =
      lower_surface_value_program(*compiled.program, emission_storage);
  require(emission_values.valid, "emission-only value lowering failed");
  const auto emission_endpoint =
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::emission);
  const auto emission = lower_surface_closure_program(
      *compiled.program, closure_plan, dependencies,
      emission_values.value_addresses, emission_endpoint);
  require(emission.valid,
          "emission closure projection failed: " + emission.diagnostic);
  require(emission.instructions.size() == 2u &&
              surface_closure_instruction_kind(emission.instructions[0u]) ==
                  SurfaceClosureInstructionKind::mix_right &&
              surface_closure_operation(emission.instructions[1u]) ==
                  ClosureOperation::emission &&
              surface_closure_endpoints(emission.instructions[1u]) ==
                  emission_endpoint &&
              emission.used_operations == (1u << static_cast<std::uint32_t>(
                                               ClosureOperation::emission)),
          "emission projection retained a physical closure family");
  const auto &mix = find_closure(*compiled.program, ClosureOperation::mix);
  require(surface_closure_mix_factor_address(emission.instructions[0u]) ==
                  emission_values.value_addresses[mix.factor.value] &&
              surface_closure_mix_parent_weight_slot(
                  emission.instructions[0u]) ==
                  surface_closure_root_weight_slot &&
              surface_closure_mix_unary_weight_slot(
                  emission.instructions[0u]) == 0u &&
              surface_closure_leaf_weight_slot(
                  emission.instructions[1u]) == 0u &&
              emission.mix_slots == 1u,
          "one-sided emission projection changed the linear Mix recurrence");

  const auto executable = build_surface_value_executable_scene(std::vector{
      SurfaceValueExecutionInput{.program = compiled.program.get(),
                                 .storage = &emission_storage,
                                 .closure_plan = &closure_plan,
                                 .closure_endpoints = emission_endpoint}});
  require(executable.valid && executable.values.programs.size() == 1u &&
              executable.values.programs.front().closure_count == 2u,
          "executable scene rejected a valid linear emission projection");

  const auto invalid = lower_surface_closure_program(
      *compiled.program, closure_plan, dependencies,
      emission_values.value_addresses, 0u);
  require(!invalid.valid && invalid.diagnostic.find("endpoint projection") !=
                                std::string::npos,
          "an empty closure endpoint projection was accepted");
}

void test_emission_shading_normal_observation() {
  const auto observes = [](ValueOperation operation,
                           std::uint64_t static_u0 = 0u,
                           std::uint64_t static_u1 = 0u) noexcept {
    return value_instruction_observes_shading_normal(
        ValueInstruction{.operation = operation,
                         .static_u0 = static_u0,
                         .static_u1 = static_u1});
  };
  require(!observes(ValueOperation::parameter) &&
              observes(ValueOperation::shading_normal) &&
              observes(ValueOperation::normal_map),
          "direct shading-normal observation classification changed");
  require(observes(ValueOperation::fresnel) &&
              !observes(ValueOperation::fresnel, 1u) &&
              observes(ValueOperation::layer_weight_fresnel) &&
              !observes(ValueOperation::layer_weight_facing, 1u),
          "linked-normal context-node classification changed");
  require(!observes(ValueOperation::image_color, 0u, 0u) &&
              observes(ValueOperation::image_color, 0u, 1u << 12u) &&
              observes(ValueOperation::image_alpha, 0u, 1u << 12u) &&
              !observes(ValueOperation::environment_color, 0u, 1u << 12u),
          "Box texture normal observation classification changed");
  require(observes(ValueOperation::bump, 0u) &&
              !observes(ValueOperation::bump, 2u) &&
              observes(ValueOperation::bump, 4u) &&
              observes(ValueOperation::bump, 6u),
          "Bump normal/fallback observation classification changed");

  const auto compiled = compile_graph(make_normal_dependent_emission_graph());
  const auto closure_plan =
      analyze_surface_closure_plan(*compiled.program, compiled.parameters);
  const auto dependencies =
      analyze_surface_value_dependencies(*compiled.program, closure_plan);
  require(dependencies.emission_values_observe_shading_normal &&
              !dependencies.emission_closures_observe_shading_normal &&
              dependencies.emission_observes_shading_normal(),
          "emission DAG lost its transitive shading-normal observation");
}

void test_nested_mix_linear_recurrence() {
  const auto compiled = compile_graph(make_nested_mix_graph());
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 5u &&
              surface_closure_operation(image.instructions[2u]) ==
                  ClosureOperation::diffuse &&
              surface_closure_operation(image.instructions[3u]) ==
                  ClosureOperation::glossy &&
              surface_closure_operation(image.instructions[4u]) ==
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

  require(surface_closure_instruction_kind(image.instructions[0u]) ==
                  SurfaceClosureInstructionKind::mix_both &&
              surface_closure_instruction_kind(image.instructions[1u]) ==
                  SurfaceClosureInstructionKind::mix_both &&
              surface_closure_mix_factor_address(image.instructions[0u]) ==
                  outer_factor &&
              surface_closure_mix_factor_address(image.instructions[1u]) ==
                  inner_factor &&
              surface_closure_mix_parent_weight_slot(
                  image.instructions[0u]) ==
                  surface_closure_root_weight_slot &&
              surface_closure_mix_left_weight_slot(
                  image.instructions[0u]) == 0u &&
              surface_closure_mix_right_weight_slot(
                  image.instructions[0u]) == 1u &&
              surface_closure_mix_parent_weight_slot(
                  image.instructions[1u]) == 0u &&
              surface_closure_mix_left_weight_slot(
                  image.instructions[1u]) == 0u &&
              surface_closure_mix_right_weight_slot(
                  image.instructions[1u]) == 2u &&
              surface_closure_leaf_weight_slot(image.instructions[2u]) == 0u &&
              surface_closure_leaf_weight_slot(image.instructions[3u]) == 2u &&
              surface_closure_leaf_weight_slot(image.instructions[4u]) == 1u &&
              image.maximum_mix_depth == 2u && image.mix_slots == 3u,
          "nested Mix recurrence or exact live-interval coloring changed");
}

void test_add_sibling_retains_root_weight() {
  const auto compiled = compile_graph(make_add_after_mix_graph());
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 4u &&
              surface_closure_instruction_kind(image.instructions[0u]) ==
                  SurfaceClosureInstructionKind::mix_both &&
              surface_closure_operation(image.instructions[1u]) ==
                  ClosureOperation::diffuse &&
              surface_closure_operation(image.instructions[2u]) ==
                  ClosureOperation::glossy &&
              surface_closure_operation(image.instructions[3u]) ==
                  ClosureOperation::emission,
          "Add(Mix(A, B), C) did not retain linear left-to-right order");
  require(surface_closure_mix_parent_weight_slot(
                  image.instructions[0u]) ==
                  surface_closure_root_weight_slot &&
              surface_closure_mix_left_weight_slot(
                  image.instructions[0u]) == 0u &&
              surface_closure_mix_right_weight_slot(
                  image.instructions[0u]) == 1u &&
              surface_closure_leaf_weight_slot(image.instructions[1u]) == 0u &&
              surface_closure_leaf_weight_slot(image.instructions[2u]) == 1u &&
              surface_closure_leaf_weight_slot(image.instructions[3u]) ==
                  surface_closure_root_weight_slot &&
              image.mix_slots == 2u,
          "Add-after-Mix did not preserve the sibling's root weight");
}

void test_statically_pruned_mix() {
  const auto compiled = compile_graph(make_domain_mix_graph(0.0f));
  const auto planned = plan_program(compiled);
  const auto &image = planned.closures;
  require(image.instructions.size() == 1u &&
              surface_closure_operation(image.instructions.front()) ==
                  ClosureOperation::diffuse &&
              image.mix_slots == 0u &&
              image.maximum_mix_depth == 0u,
          "a statically unreachable Mix branch or factor survived");
  require(image.used_operations ==
              (1u << static_cast<std::uint32_t>(
                   ClosureOperation::diffuse)),
          "scene operation mask includes a pruned closure family");
}

[[nodiscard]] ShaderGraph make_principled_graph(
    float thin_film_thickness = 0.0f) {
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
              SocketValue::floating(1.7f)) &&
          graph.set_input(
              principled,
              "ThinFilmThickness",
              SocketValue::floating(thin_film_thickness)) &&
          graph.set_input(
              principled,
              "ThinFilmIOR",
              SocketValue::floating(1.33f)),
      "failed to configure closure-bytecode Principled graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = principled, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_dielectric_graph(
    bool refraction, float thin_film_thickness = 0.0f) {
  ShaderGraph graph;
  const auto closure = graph.add_node(
      refraction ? node_type::refraction_bsdf : node_type::glass_bsdf,
      refraction ? "Refraction operand contract" : "Glass operand contract");
  auto configured =
      graph.set_input(closure, "Color", SocketValue::color({0.3f, 0.5f, 0.8f})) &&
      graph.set_input(closure, "Roughness", SocketValue::floating(0.27f)) &&
      graph.set_input(closure, "IOR", SocketValue::floating(1.47f));
  if (!refraction) {
    configured = configured &&
                 graph.set_input(
                     closure,
                     "ThinFilmThickness",
                     SocketValue::floating(thin_film_thickness)) &&
                 graph.set_input(
                     closure,
                     "ThinFilmIOR",
                     SocketValue::floating(1.31f));
  }
  require(configured, "failed to configure dielectric operand contract");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = closure, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_glossy_anisotropy_graph(
    float anisotropy,
    bool linked_anisotropy = false) {
  ShaderGraph graph;
  const auto glossy = graph.add_node(
      node_type::glossy_bsdf, "Anisotropic Glossy closure");
  const auto geometry = graph.add_node(
      node_type::geometry, "Glossy tangent source");
  require(
      graph.set_input(
          glossy,
          "Anisotropy",
          SocketValue::floating(anisotropy)) &&
          graph.set_input(
              glossy,
              "Rotation",
              SocketValue::floating(0.125f)) &&
          graph.connect(
              {.node = geometry, .socket = "Tangent"},
              glossy,
              "Tangent"),
      "failed to configure Glossy anisotropy inputs");
  if (linked_anisotropy) {
    const auto value = graph.add_node(
        node_type::constant_float, "Linked Glossy anisotropy");
    require(
        graph.set_input(
            value,
            "Value",
            SocketValue::floating(anisotropy)) &&
            graph.connect(
                {.node = value, .socket = "Value"},
                glossy,
                "Anisotropy"),
        "failed to link Glossy anisotropy");
  }
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = glossy, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_metallic_graph(
    std::string fresnel_type,
    std::string distribution,
    float anisotropy,
    float thin_film_thickness) {
  ShaderGraph graph;
  const auto metallic = graph.add_node(
      node_type::metallic_bsdf, "Standalone Metallic contract");
  const auto geometry = graph.add_node(
      node_type::geometry, "Standalone Metallic frame");
  require(
      graph.set_property(
          metallic,
          "FresnelType",
          SocketValue::string(std::move(fresnel_type))) &&
          graph.set_property(
              metallic,
              "Distribution",
              SocketValue::string(std::move(distribution))) &&
          graph.set_input(
              metallic,
              "BaseColor",
              SocketValue::color({0.17f, 0.31f, 0.73f})) &&
          graph.set_input(
              metallic,
              "EdgeTint",
              SocketValue::color({0.91f, 0.67f, 0.43f})) &&
          graph.set_input(
              metallic,
              "IOR",
              SocketValue::vector({1.9f, 2.7f, 3.4f})) &&
          graph.set_input(
              metallic,
              "Extinction",
              SocketValue::vector({4.6f, 3.2f, 2.1f})) &&
          graph.set_input(
              metallic,
              "Roughness",
              SocketValue::floating(0.38f)) &&
          graph.set_input(
              metallic,
              "Anisotropy",
              SocketValue::floating(anisotropy)) &&
          graph.set_input(
              metallic,
              "Rotation",
              SocketValue::floating(0.2f)) &&
          graph.set_input(
              metallic,
              "ThinFilmThickness",
              SocketValue::floating(thin_film_thickness)) &&
          graph.set_input(
              metallic,
              "ThinFilmIOR",
              SocketValue::floating(1.37f)) &&
          graph.connect(
              {.node = geometry, .socket = "Tangent"},
              metallic,
              "Tangent") &&
          graph.connect(
              {.node = geometry, .socket = "Normal"},
              metallic,
              "Normal"),
      "failed to configure standalone Metallic graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = metallic, .socket = "Closure"});
  return graph;
}

void test_standalone_metallic_static_contract() {
  const auto f82 = compile_graph(make_metallic_graph(
      "F82", "multi_ggx", 1.0e-4f, 0.1f));
  const auto f82_plan = plan_program(f82);
  const auto f82_root = f82.program->root();
  const auto &f82_closure =
      f82.program->closure_instructions()[f82_root.value];
  const auto &f82_instruction = f82_plan.closures.instructions.front();
  const auto f82_begin =
      surface_closure_leaf_operand_begin(f82_instruction);
  require(
      surface_closure_operation(f82_instruction) ==
              ClosureOperation::metallic_f82 &&
          f82_closure.metallic_base_ior.valid() &&
          f82_closure.metallic_edge_tint_k.valid() &&
          !f82_closure.color.valid() &&
          f82_closure.preserve_ggx_energy &&
          !f82_closure.beckmann &&
          !f82_plan.closure_plan.entry(f82_root).microfacet_anisotropy &&
          !f82_plan.closure_plan.entry(f82_root).thin_film &&
          f82_plan.closures.operands.size() ==
              surface_closure_operand::metallic::count &&
          (f82_instruction.control &
           (surface_closure_microfacet_anisotropy |
            surface_closure_thin_film)) == 0u &&
          f82_plan.closures.operands[
              f82_begin + surface_closure_operand::metallic::anisotropy] ==
              SurfaceValueAddress::invalid_value &&
          f82_plan.closures.operands[
              f82_begin + surface_closure_operand::metallic::rotation] ==
              SurfaceValueAddress::invalid_value &&
          f82_plan.closures.operands[
              f82_begin + surface_closure_operand::metallic::tangent] ==
              SurfaceValueAddress::invalid_value &&
          f82_plan.closures.operands[
              f82_begin +
              surface_closure_operand::metallic::thin_film_thickness] ==
              SurfaceValueAddress::invalid_value &&
          f82_plan.closures.operands[
              f82_begin + surface_closure_operand::metallic::thin_film_ior] ==
              SurfaceValueAddress::invalid_value,
      "F82 Metallic did not erase inactive topology operands exactly");

  const auto conductor = compile_graph(make_metallic_graph(
      "physical_conductor", "BECKMANN", 0.63f, 420.0f));
  const auto conductor_plan = plan_program(conductor);
  const auto conductor_root = conductor.program->root();
  const auto &conductor_closure =
      conductor.program->closure_instructions()[conductor_root.value];
  const auto &conductor_instruction =
      conductor_plan.closures.instructions.front();
  const auto conductor_begin =
      surface_closure_leaf_operand_begin(conductor_instruction);
  require(
      surface_closure_operation(conductor_instruction) ==
              ClosureOperation::metallic_conductor &&
          conductor_closure.metallic_base_ior.valid() &&
          conductor_closure.metallic_edge_tint_k.valid() &&
          !conductor_closure.color.valid() &&
          !conductor_closure.preserve_ggx_energy &&
          conductor_closure.beckmann &&
          conductor_plan.closure_plan.entry(conductor_root)
              .microfacet_anisotropy &&
          conductor_plan.closure_plan.entry(conductor_root).thin_film &&
          conductor_plan.closures.operands.size() ==
              surface_closure_operand::metallic::count &&
          (conductor_instruction.control &
           surface_closure_microfacet_anisotropy) != 0u &&
          (conductor_instruction.control & surface_closure_thin_film) != 0u &&
          conductor_plan.closures.operands[
              conductor_begin +
              surface_closure_operand::metallic::base_ior] ==
              conductor_plan.values.value_addresses[
                  conductor_closure.metallic_base_ior.value] &&
          conductor_plan.closures.operands[
              conductor_begin +
              surface_closure_operand::metallic::edge_tint_k] ==
              conductor_plan.values.value_addresses[
                  conductor_closure.metallic_edge_tint_k.value] &&
          conductor_plan.closures.operands[
              conductor_begin + surface_closure_operand::metallic::tangent] !=
              SurfaceValueAddress::invalid_value &&
          conductor_plan.closures.operands[
              conductor_begin +
              surface_closure_operand::metallic::thin_film_thickness] !=
              SurfaceValueAddress::invalid_value,
      "physical-conductor Metallic lost its live typed operand slice");

  const auto has_parameter = [](const SurfaceProgram &program,
                                std::string_view socket) noexcept {
    for (const auto &parameter : program.parameters()) {
      if (parameter.socket == socket) {
        return true;
      }
    }
    return false;
  };
  require(
      has_parameter(*f82.program, "BaseColor") &&
          has_parameter(*f82.program, "EdgeTint") &&
          !has_parameter(*f82.program, "IOR") &&
          !has_parameter(*f82.program, "Extinction") &&
          !has_parameter(*conductor.program, "BaseColor") &&
          !has_parameter(*conductor.program, "EdgeTint") &&
          has_parameter(*conductor.program, "IOR") &&
          has_parameter(*conductor.program, "Extinction"),
      "static Metallic Fresnel selection retained the inactive value graph");
}

[[nodiscard]] ShaderGraph make_sheen_graph(std::string distribution) {
  ShaderGraph graph;
  const auto sheen = graph.add_node(
      node_type::sheen_bsdf, "Standalone Sheen contract");
  const auto geometry = graph.add_node(
      node_type::geometry, "Standalone Sheen normal");
  require(
      graph.set_property(
          sheen,
          "Distribution",
          SocketValue::string(std::move(distribution))) &&
          graph.set_input(
              sheen,
              "Color",
              SocketValue::color({0.6f, 0.3f, 0.15f})) &&
          graph.set_input(
              sheen,
              "Roughness",
              SocketValue::floating(0.37f)) &&
          graph.connect(
              {.node = geometry, .socket = "Normal"},
              sheen,
              "Normal"),
      "failed to configure standalone Sheen graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = sheen, .socket = "Closure"});
  return graph;
}

void test_standalone_sheen_static_contract() {
  const auto verify = [](std::string distribution,
                          ClosureOperation expected_operation) {
    const auto compiled = compile_graph(
        make_sheen_graph(std::move(distribution)));
    const auto planned = plan_program(compiled);
    const auto root = compiled.program->root();
    const auto &closure =
        compiled.program->closure_instructions()[root.value];
    const auto &instruction = planned.closures.instructions.front();
    const auto begin = surface_closure_leaf_operand_begin(instruction);
    require(
        closure.operation == expected_operation &&
            surface_closure_operation(instruction) == expected_operation &&
            closure.color.valid() && closure.normal.valid() &&
            closure.roughness.valid() &&
            planned.closures.operands.size() ==
                surface_closure_operand::sheen::count &&
            planned.closures.operands[
                begin + surface_closure_operand::sheen::color] ==
                planned.values.value_addresses[closure.color.value] &&
            planned.closures.operands[
                begin + surface_closure_operand::sheen::normal] ==
                planned.values.value_addresses[closure.normal.value] &&
            planned.closures.operands[
                begin + surface_closure_operand::sheen::roughness] ==
                planned.values.value_addresses[closure.roughness.value] &&
            surface_closure_endpoints(instruction) ==
                surface_closure_endpoint_bit(
                    SurfaceClosureEndpoint::physical) &&
            (instruction.control &
             (surface_closure_microfacet_anisotropy |
              surface_closure_thin_film |
              surface_closure_preserve_ggx_energy |
              surface_closure_beckmann)) == 0u &&
            planned.closures.principled_features.front() == 0u,
        "standalone Sheen did not retain exactly its typed operand slice");
    for (const auto &parameter : compiled.program->parameters()) {
      require(parameter.socket != "Distribution" &&
                  parameter.socket != "Weight",
              "static Sheen metadata leaked into the runtime parameter ABI");
    }
  };
  verify("MICROFIBER", ClosureOperation::sheen_microfiber);
  verify("ASHIKHMIN", ClosureOperation::sheen_ashikhmin);
}

void test_microfacet_anisotropy_specialization() {
  const auto isotropic = compile_graph(
      make_glossy_anisotropy_graph(1.0e-4f));
  const auto isotropic_plan = plan_program(isotropic);
  const auto isotropic_root = isotropic.program->root();
  const auto &isotropic_closure =
      isotropic.program->closure_instructions()[isotropic_root.value];
  const auto isotropic_begin = surface_closure_leaf_operand_begin(
      isotropic_plan.closures.instructions.front());
  require(
      !isotropic_plan.closure_plan.entry(isotropic_root)
           .microfacet_anisotropy &&
          !isotropic_plan.dependencies.physical[
              isotropic_closure.microfacet_anisotropy.value] &&
          !isotropic_plan.dependencies.physical[
              isotropic_closure.microfacet_rotation.value] &&
          !isotropic_plan.dependencies.physical[
              isotropic_closure.tangent.value] &&
          (isotropic_plan.closures.instructions.front().control &
           surface_closure_microfacet_anisotropy) == 0u &&
          isotropic_plan.closures.operands[
              isotropic_begin +
              surface_closure_operand::glossy::anisotropy] ==
              SurfaceValueAddress::invalid_value &&
          isotropic_plan.closures.operands[
              isotropic_begin +
              surface_closure_operand::glossy::rotation] ==
              SurfaceValueAddress::invalid_value &&
          isotropic_plan.closures.operands[
              isotropic_begin +
              surface_closure_operand::glossy::tangent] ==
              SurfaceValueAddress::invalid_value,
      "static Glossy isotropy did not erase its complete operand slice");

  const auto anisotropic = compile_graph(
      make_glossy_anisotropy_graph(-0.71f));
  const auto anisotropic_plan = plan_program(anisotropic);
  const auto anisotropic_root = anisotropic.program->root();
  const auto &anisotropic_closure =
      anisotropic.program->closure_instructions()[anisotropic_root.value];
  const auto anisotropic_begin = surface_closure_leaf_operand_begin(
      anisotropic_plan.closures.instructions.front());
  require(
      anisotropic_plan.closure_plan.entry(anisotropic_root)
          .microfacet_anisotropy &&
          anisotropic_plan.dependencies.physical[
              anisotropic_closure.microfacet_anisotropy.value] &&
          anisotropic_plan.dependencies.physical[
              anisotropic_closure.microfacet_rotation.value] &&
          anisotropic_plan.dependencies.physical[
              anisotropic_closure.tangent.value] &&
          (anisotropic_plan.closures.instructions.front().control &
           surface_closure_microfacet_anisotropy) != 0u &&
          anisotropic_plan.closures.operands[
              anisotropic_begin +
              surface_closure_operand::glossy::anisotropy] !=
              SurfaceValueAddress::invalid_value &&
          anisotropic_plan.closures.operands[
              anisotropic_begin +
              surface_closure_operand::glossy::rotation] !=
              SurfaceValueAddress::invalid_value &&
          anisotropic_plan.closures.operands[
              anisotropic_begin +
              surface_closure_operand::glossy::tangent] !=
              SurfaceValueAddress::invalid_value,
      "live Glossy anisotropy was not retained as one typed operand slice");

  const auto linked_zero = compile_graph(
      make_glossy_anisotropy_graph(0.0f, true));
  const auto linked_plan = analyze_surface_closure_plan(
      *linked_zero.program, linked_zero.parameters);
  require(
      linked_plan.entry(linked_zero.program->root())
          .microfacet_anisotropy,
      "linked Glossy anisotropy was incorrectly host-folded by value");

  auto merged = isotropic_plan.closure_plan;
  merged.merge(anisotropic_plan.closure_plan);
  require(
      merged.entry(isotropic_root).microfacet_anisotropy,
      "material-plan union lost a topology peer's anisotropy capability");
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
              surface_closure_leaf_weight_slot(instruction) ==
                  surface_closure_root_weight_slot &&
              instruction.payload2 == 0u &&
              image.operands.size() ==
                  surface_closure_operand::principled::count,
          "Principled opcode/endpoints/operand ABI is inconsistent");

  const auto root = compiled.program->root();
  const auto expected_features =
      planned.closure_plan.entry(root).principled_features;
  require(image.principled_features.front() == expected_features &&
              image.used_principled_features == expected_features,
          "Principled feature mask was not preserved as scene data");
  require(!planned.dependencies.emission_values_observe_shading_normal &&
              planned.dependencies.emission_closures_observe_shading_normal &&
              planned.dependencies.emission_observes_shading_normal(),
          "Principled Sheen/Coat emission lost its SetNormal dependence");
  require(surface_closure_bssrdf_method(instruction) ==
              compiled.program->closure_instructions()[root.value]
                  .subsurface_method,
          "BSSRDF method was omitted from closure control");
  require((instruction.control & ~surface_closure_control_mask) == 0u,
          "closure control contains undefined bits");
  const auto operand_begin = surface_closure_leaf_operand_begin(instruction);
  require(!planned.closure_plan.entry(root).thin_film &&
              (instruction.control & surface_closure_thin_film) == 0u &&
              image.operands[
                  operand_begin +
                  surface_closure_operand::principled::thin_film_thickness] ==
                  SurfaceValueAddress::invalid_value &&
              image.operands[
                  operand_begin +
                  surface_closure_operand::principled::thin_film_ior] ==
                  SurfaceValueAddress::invalid_value,
          "inactive Principled Thin Film survived its static operand slice");

  const auto emission_storage = plan_surface_value_storage(
      *compiled.program, planned.dependencies.emission,
      planned.dependencies.emission_outputs);
  require(emission_storage.valid,
          "Principled emission value projection failed");
  const auto emission_values =
      lower_surface_value_program(*compiled.program, emission_storage);
  require(emission_values.valid, "Principled emission value lowering failed");
  const auto emission = lower_surface_closure_program(
      *compiled.program, planned.closure_plan, planned.dependencies,
      emission_values.value_addresses,
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::emission));
  const auto emission_features =
      principled_closure_feature_bit(PrincipledClosureFeature::alpha) |
      principled_closure_feature_bit(PrincipledClosureFeature::sheen) |
      principled_closure_feature_bit(PrincipledClosureFeature::coat) |
      principled_closure_feature_bit(PrincipledClosureFeature::emission);
  require(emission.valid && emission.instructions.size() == 1u &&
              emission.principled_features.front() ==
                  (expected_features & emission_features) &&
              emission.used_principled_features ==
                  (expected_features & emission_features),
          "Principled emission retained an unobserved physical feature");
}

void test_thin_film_static_contract() {
  const auto principled = compile_graph(make_principled_graph(325.0f));
  const auto principled_plan = plan_program(principled);
  const auto principled_root = principled.program->root();
  const auto &principled_closure =
      principled.program->closure_instructions()[principled_root.value];
  const auto &principled_instruction =
      principled_plan.closures.instructions.front();
  const auto principled_begin =
      surface_closure_leaf_operand_begin(principled_instruction);
  require(
      principled_plan.closure_plan.entry(principled_root).thin_film &&
          (principled_instruction.control & surface_closure_thin_film) != 0u &&
          principled_plan.closures.operands.size() ==
              surface_closure_operand::principled::count &&
          principled_plan.closures.operands[
              principled_begin +
              surface_closure_operand::principled::thin_film_thickness] ==
              principled_plan.values.value_addresses[
                  principled_closure.thin_film_thickness.value] &&
          principled_plan.closures.operands[
              principled_begin +
              surface_closure_operand::principled::thin_film_ior] ==
              principled_plan.values.value_addresses[
                  principled_closure.thin_film_ior.value],
      "active Principled Thin Film did not preserve its typed operand slice");

  const auto glass = compile_graph(make_dielectric_graph(false, 410.0f));
  const auto glass_plan = plan_program(glass);
  const auto glass_root = glass.program->root();
  const auto &glass_closure =
      glass.program->closure_instructions()[glass_root.value];
  const auto &glass_instruction = glass_plan.closures.instructions.front();
  const auto glass_begin = surface_closure_leaf_operand_begin(glass_instruction);
  require(
      surface_closure_operation(glass_instruction) == ClosureOperation::glass &&
          glass_plan.closure_plan.entry(glass_root).thin_film &&
          (glass_instruction.control & surface_closure_thin_film) != 0u &&
          glass_plan.closures.operands.size() ==
              surface_closure_operand::glass::count &&
          glass_plan.closures.operands[
              glass_begin +
              surface_closure_operand::glass::thin_film_thickness] ==
              glass_plan.values.value_addresses[
                  glass_closure.thin_film_thickness.value] &&
          glass_plan.closures.operands[
              glass_begin + surface_closure_operand::glass::thin_film_ior] ==
              glass_plan.values.value_addresses[
                  glass_closure.thin_film_ior.value],
      "active Glass Thin Film did not preserve its typed operand slice");

  const auto refraction = compile_graph(make_dielectric_graph(true));
  const auto refraction_plan = plan_program(refraction);
  const auto &refraction_instruction =
      refraction_plan.closures.instructions.front();
  require(
      surface_closure_operation(refraction_instruction) ==
              ClosureOperation::refraction &&
          (refraction_instruction.control & surface_closure_thin_film) == 0u &&
          refraction_plan.closures.operands.size() ==
              surface_closure_operand::refraction::count &&
          surface_closure_operand::refraction::count == 4u &&
          surface_closure_operand::glass::count == 6u,
      "Refraction inherited Glass Thin Film operands or control metadata");
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

void test_malformed_linear_weight_program_rejected() {
  const auto compiled = compile_graph(make_nested_mix_graph());
  const auto planned = plan_program(compiled);
  const auto reject_mutation = [&](SurfaceClosureProgramImage malformed,
                                   std::string_view expected) {
    const auto scene = build_surface_execution_scene_image(
        std::span{&planned.values, 1u}, std::span{&malformed, 1u});
    require(!scene.valid &&
                scene.diagnostic.find(expected) != std::string::npos,
            "malformed linear closure-weight program was accepted: " +
                std::string{expected});
  };

  auto undefined_parent = planned.closures;
  undefined_parent.instructions[0u].payload1 = 0u;
  reject_mutation(std::move(undefined_parent), "undefined parent weight");

  auto undefined_leaf = planned.closures;
  undefined_leaf.instructions[2u].payload1 =
      planned.closures.mix_slots;
  reject_mutation(std::move(undefined_leaf), "undefined incoming weight");

  auto aliased_outputs = planned.closures;
  aliased_outputs.instructions[0u].payload2 = 0u;
  reject_mutation(std::move(aliased_outputs), "aliased result weights");

  auto out_of_range_output = planned.closures;
  out_of_range_output.instructions[0u].payload2 =
      planned.closures.mix_slots |
      (surface_closure_mix_right_weight_slot(
           planned.closures.instructions[0u])
       << 16u);
  reject_mutation(std::move(out_of_range_output), "invalid or aliased");

  auto malformed_unary = planned.closures;
  malformed_unary.instructions[0u].control =
      make_surface_closure_instruction_kind(
          SurfaceClosureInstructionKind::mix_left);
  reject_mutation(std::move(malformed_unary), "unary Mix");

  auto reserved_leaf_payload = planned.closures;
  reserved_leaf_payload.instructions[2u].payload2 = 1u;
  reject_mutation(
      std::move(reserved_leaf_payload), "nonzero reserved payload");

  auto inexact_coloring = planned.closures;
  inexact_coloring.mix_slots += 1u;
  reject_mutation(std::move(inexact_coloring), "not dense");

  auto semantic_control = planned.closures;
  semantic_control.instructions[0u].control |=
      surface_closure_endpoint_mask;
  reject_mutation(std::move(semantic_control), "leaf semantic bits");

  auto illegal_thin_film = planned.closures;
  illegal_thin_film.instructions[2u].control |= surface_closure_thin_film;
  reject_mutation(std::move(illegal_thin_film),
                  "thin film on an incompatible projection");
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
              scene.programs[0u].closure_count == 3u &&
              scene.programs[1u].closure_begin == 3u &&
              scene.programs[1u].closure_count == 3u,
          "closure ranges are not published in program descriptors");
  require(scene.closure_instructions.size() == 6u &&
              scene.closure_principled_features.size() == 6u &&
              scene.closure_operands.size() == 10u,
          "parallel closure streams changed size during aggregation");
  require(surface_closure_leaf_operand_begin(
                  scene.closure_instructions[4u]) == 5u &&
              surface_closure_instruction_kind(
                  scene.closure_instructions[3u]) ==
                  SurfaceClosureInstructionKind::mix_both &&
              surface_closure_mix_factor_address(
                  scene.closure_instructions[3u]) ==
                  surface_closure_mix_factor_address(
                      planned.closures.instructions[0u]) &&
              surface_closure_instruction_kind(
                  scene.closure_instructions[5u]) ==
                  SurfaceClosureInstructionKind::leaf,
          "the second closure program was not relocated exactly once");
  require(scene.maximum_closure_mix_depth == 1u &&
              scene.maximum_closure_mix_slots == 2u &&
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
    test_mixed_endpoint_linear_weight_program();
    test_endpoint_projection();
    test_emission_shading_normal_observation();
    test_nested_mix_linear_recurrence();
    test_add_sibling_retains_root_weight();
    test_statically_pruned_mix();
    test_principled_static_contract();
    test_standalone_metallic_static_contract();
    test_standalone_sheen_static_contract();
    test_thin_film_static_contract();
    test_microfacet_anisotropy_specialization();
    test_missing_live_address_rejected();
    test_malformed_linear_weight_program_rejected();
    test_executable_scene_relocation();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "surface closure execution-plan tests passed\n";
  return 0;
}
