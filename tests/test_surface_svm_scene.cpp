#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/compiler/surface_svm_program.h>
#include <psycles/compiler/surface_svm_schedule.h>
#include <psycles/contract/scene.h>

#include <algorithm>
#include <array>
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
  require(shader.ok(), "surface-scene fixture failed to compile");
  const auto lowered = compile_surface_program(*shader.program);
  require(lowered.ok(), "surface-scene fixture failed to lower");
  auto binding = bind_surface_parameters(*lowered.program, *shader.program);
  require(binding.ok(), "surface-scene fixture failed to bind");
  return {.program = lowered.program,
          .parameters = std::move(*binding.parameters)};
}

[[nodiscard]] SurfaceSvmProgramImage lower_graph(
    ShaderGraph graph,
    SurfaceClosureEndpointMask endpoints = all_surface_closure_endpoints) {
  const auto compiled = compile_graph(std::move(graph));
  const auto closures =
      analyze_surface_closure_plan(*compiled.program, compiled.parameters);
  const auto dependencies =
      analyze_surface_value_dependencies(*compiled.program, closures);
  const auto schedule = plan_surface_svm_schedule(*compiled.program, closures,
                                                  dependencies, endpoints);
  require(schedule.valid,
          "surface-scene schedule failed: " + schedule.diagnostic);
  const auto storage = plan_surface_svm_storage(*compiled.program, schedule);
  require(storage.valid, "surface-scene storage failed: " + storage.diagnostic);
  const auto image = lower_surface_svm_program(*compiled.program, closures,
                                               dependencies, schedule, storage);
  require(image.valid, "surface-scene bytecode failed: " + image.diagnostic);
  return image;
}

[[nodiscard]] ShaderGraph make_conditional_graph() {
  ShaderGraph graph;
  const auto color_a =
      graph.add_node(node_type::constant_color, "Scene color A");
  const auto color_b =
      graph.add_node(node_type::constant_color, "Scene color B");
  const auto color_factor =
      graph.add_node(node_type::constant_float, "Scene color factor");
  const auto mixed_color =
      graph.add_node(node_type::mix_color, "Scene mixed color");
  const auto emission_color =
      graph.add_node(node_type::constant_color, "Scene emission color");
  const auto closure_factor =
      graph.add_node(node_type::constant_float, "Scene closure factor");
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Scene diffuse");
  const auto emission = graph.add_node(node_type::emission, "Scene emission");
  const auto mix = graph.add_node(node_type::mix_closure, "Scene closure Mix");
  require(graph.set_input(color_a, "Color",
                          SocketValue::color({0.1f, 0.2f, 0.3f})) &&
              graph.set_input(color_b, "Color",
                              SocketValue::color({0.7f, 0.6f, 0.5f})) &&
              graph.set_input(color_factor, "Value",
                              SocketValue::floating(0.37f)) &&
              graph.set_input(emission_color, "Color",
                              SocketValue::color({0.8f, 0.3f, 0.1f})) &&
              graph.set_input(closure_factor, "Value",
                              SocketValue::floating(0.41f)) &&
              graph.connect({.node = color_factor, .socket = "Value"},
                            mixed_color, "Factor") &&
              graph.connect({.node = color_a, .socket = "Color"}, mixed_color,
                            "A") &&
              graph.connect({.node = color_b, .socket = "Color"}, mixed_color,
                            "B") &&
              graph.connect({.node = mixed_color, .socket = "Color"}, diffuse,
                            "Color") &&
              graph.connect({.node = emission_color, .socket = "Color"},
                            emission, "Color") &&
              graph.connect({.node = closure_factor, .socket = "Value"}, mix,
                            "Factor") &&
              graph.connect({.node = diffuse, .socket = "Closure"}, mix, "A") &&
              graph.connect({.node = emission, .socket = "Closure"}, mix, "B"),
          "failed to configure conditional surface-scene graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});
  return graph;
}

[[nodiscard]] std::uint32_t local_address(SurfaceValueBank bank,
                                          std::uint32_t index) noexcept {
  return (static_cast<std::uint32_t>(bank) << SurfaceValueAddress::bank_shift) |
         index;
}

[[nodiscard]] std::uint32_t one_operand(std::uint32_t encoded) {
  SurfaceValueOperandAddress operand;
  require(encode_surface_value_operand_address(SurfaceValueAddress{encoded},
                                               operand),
          "fixture operand cannot be compactly encoded");
  return static_cast<std::uint32_t>(operand.encoded()) |
         (static_cast<std::uint32_t>(SurfaceValueOperandAddress::invalid_value)
          << surface_value_operand_lane_bits);
}

[[nodiscard]] SurfaceValueProgramImage make_normal_prefix() {
  SurfaceValueProgramImage normal;
  normal.valid = true;
  normal.vector_slots = 1u;
  normal.instructions.emplace_back(SurfaceValueBytecodeInstruction{
      .control = make_surface_value_control(ValueOperation::surface_position,
                                            SurfaceValueBank::vector, 0u),
      .result = local_address(SurfaceValueBank::vector, 0u),
      .operand_payload = surface_value_invalid_operand_word,
      .metadata_index = SurfaceValueAddress::invalid_value});
  require(validate_surface_value_program_image(normal).empty(),
          "manual automatic-normal prefix is invalid");
  return normal;
}

[[nodiscard]] SurfaceSvmProgramImage make_epoch_root() {
  SurfaceSvmProgramImage root;
  root.valid = true;
  root.endpoints =
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical);
  root.scalar_slots = 1u;
  root.vector_slots = 2u;
  root.value_instruction_count = 2u;
  root.instructions.emplace_back(
      make_surface_svm_value_instruction(SurfaceValueBytecodeInstruction{
          .control = make_surface_value_control(
              ValueOperation::surface_position, SurfaceValueBank::vector, 0u),
          .result = local_address(SurfaceValueBank::vector, 1u),
          .operand_payload = surface_value_invalid_operand_word,
          .metadata_index = SurfaceValueAddress::invalid_value}));
  root.instructions.emplace_back(
      make_surface_svm_value_instruction(SurfaceValueBytecodeInstruction{
          .control = make_surface_value_control(
              ValueOperation::vector_to_scalar, SurfaceValueBank::scalar, 0u),
          .result = local_address(SurfaceValueBank::scalar, 0u),
          .operand_payload =
              one_operand(local_address(SurfaceValueBank::vector, 1u)),
          .metadata_index = SurfaceValueAddress::invalid_value}));
  root.instructions.emplace_back(
      SurfaceSvmBytecodeInstruction{.control = surface_svm_end_opcode,
                                    .payload0 = surface_svm_invalid_payload,
                                    .payload1 = surface_svm_invalid_payload,
                                    .payload2 = surface_svm_invalid_payload});
  require(validate_surface_svm_program_image(root).empty(),
          "manual structured root is invalid");
  return root;
}

[[nodiscard]] SurfaceSvmProgramImage make_static_table_program() {
  std::vector<float> transform(16u, 0.0f);
  for (auto diagonal = std::size_t{}; diagonal < 4u; ++diagonal) {
    transform[diagonal * 5u] = 1.0f;
  }
  const SurfaceProgram source{
      17u,
      {},
      {ValueInstruction{.operation =
                            ValueOperation::object_position_with_transform,
                        .result_type = SocketType::point,
                        .static_u0 = 7u,
                        .static_f0 = 0.25f,
                        .static_f1 = -0.0f,
                        .static_table = transform}},
      {},
      {}};
  const auto storage = plan_surface_value_storage(
      source, std::vector<bool>{true}, std::vector<bool>{true});
  require(storage.valid, "static-table fixture storage failed");
  const auto values = lower_surface_value_program(source, storage);
  require(values.valid && values.instructions.size() == 1u &&
              values.metadata.size() == 1u && values.static_data == transform,
          "static-table fixture value lowering failed");

  SurfaceSvmProgramImage program;
  program.valid = true;
  program.endpoints =
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical);
  program.instructions.emplace_back(
      make_surface_svm_value_instruction(values.instructions.front()));
  program.instructions.emplace_back(
      SurfaceSvmBytecodeInstruction{.control = surface_svm_end_opcode,
                                    .payload0 = surface_svm_invalid_payload,
                                    .payload1 = surface_svm_invalid_payload,
                                    .payload2 = surface_svm_invalid_payload});
  program.value_operands = values.operands;
  program.value_metadata = values.metadata;
  program.static_data = values.static_data;
  program.value_addresses = values.value_addresses;
  program.scalar_slots = values.scalar_slots;
  program.vector_slots = values.vector_slots;
  program.unsigned_integer_slots = values.unsigned_integer_slots;
  program.value_instruction_count = 1u;
  require(validate_surface_svm_program_image(program).empty(),
          "static-table unified program is invalid");
  return program;
}

void test_set_normal_starts_a_new_local_lifetime_epoch() {
  const auto normal = make_normal_prefix();
  const auto root = make_epoch_root();
  const auto composed = compose_surface_svm_normal_transaction(
      normal, local_address(SurfaceValueBank::vector, 0u), root, true);
  require(composed.valid,
          "SetNormal composition failed: " + composed.diagnostic);
  require(
      composed.surface_normal_transition_count == 1u &&
          composed.instructions.size() == 5u &&
          surface_svm_bytecode_kind(composed.instructions[1u]) ==
              SurfaceSvmBytecodeKind::set_normal &&
          composed.flags ==
              surface_value_program_automatic_normal_uses_undisplaced_geometry,
      "SetNormal composition lost its canonical transaction boundary");

  // Prefix vector slot zero is physically present but semantically dead after
  // SetNormal. Repointing a root read to it must fail even though a naive
  // linear scan would still see an earlier definition.
  auto stale_prefix_read = composed;
  stale_prefix_read.instructions[3u].payload1 =
      one_operand(local_address(SurfaceValueBank::vector, 0u));
  require(validate_surface_svm_program_image(stale_prefix_read)
                  .find("undefined value") != std::string::npos,
          "SetNormal allowed the root to observe a stale prefix local");

  const std::array programs{composed, composed};
  const auto scene = build_surface_svm_scene_image(programs);
  require(scene.valid && validate_surface_svm_scene_image(scene).empty() &&
              scene.programs[1u].flags == composed.flags,
          "scene aggregation lost the SetNormal transaction");
}

void test_scene_rebases_control_and_typed_side_streams() {
  const auto program = lower_graph(make_conditional_graph());
  require(program.conditional_branch_count != 0u &&
              !program.value_operands.empty() &&
              !program.closure_operands.empty(),
          "conditional scene fixture lacks relocation classes");
  const std::array programs{program, program};
  const auto scene = build_surface_svm_scene_image(programs);
  require(scene.valid, "surface scene aggregation failed: " + scene.diagnostic);
  require(validate_surface_svm_scene_image(scene).empty(),
          "surface scene aggregate failed its public verifier");
  const auto &descriptor = scene.programs[1u];
  const auto &side = scene.side_ranges[1u];
  require(descriptor.instruction_begin == program.instructions.size() &&
              side.value_operand_begin == program.value_operands.size() &&
              side.closure_operand_begin == program.closure_operands.size(),
          "second program does not begin at the first program's exact extent");

  auto observed_guard = false;
  auto observed_overflow_operands = false;
  auto observed_closure_operands = false;
  for (auto local_pc = std::size_t{}; local_pc < program.instructions.size();
       ++local_pc) {
    const auto &local = program.instructions[local_pc];
    const auto &global =
        scene.instructions[descriptor.instruction_begin + local_pc];
    switch (surface_svm_bytecode_kind(local)) {
    case SurfaceSvmBytecodeKind::value: {
      const auto local_value = surface_svm_value_instruction(local);
      const auto global_value = surface_svm_value_instruction(global);
      if (surface_value_operand_count(local_value) >
          surface_value_inline_operand_capacity) {
        require(global_value.operand_payload ==
                    local_value.operand_payload + side.value_operand_begin,
                "overflow value operands were not rebased exactly");
        observed_overflow_operands = true;
      }
      break;
    }
    case SurfaceSvmBytecodeKind::jump_if_one:
    case SurfaceSvmBytecodeKind::jump_if_zero:
      require(global.payload1 == local.payload1 + descriptor.instruction_begin,
              "closure guard target was not rebased exactly");
      observed_guard = true;
      break;
    case SurfaceSvmBytecodeKind::closure_leaf:
      require(global.payload0 == local.payload0 + side.closure_operand_begin,
              "closure operand begin was not rebased exactly");
      observed_closure_operands = true;
      break;
    case SurfaceSvmBytecodeKind::mix_closure:
    case SurfaceSvmBytecodeKind::add_closure_weight:
    case SurfaceSvmBytecodeKind::set_normal:
    case SurfaceSvmBytecodeKind::end:
    case SurfaceSvmBytecodeKind::invalid:
      break;
    }
  }
  require(observed_guard && observed_overflow_operands &&
              observed_closure_operands,
          "scene relocation fixture did not exercise every required class");

  auto local_guard_target = scene;
  for (auto pc = descriptor.instruction_begin;
       pc < descriptor.instruction_begin + descriptor.instruction_count; ++pc) {
    const auto kind =
        surface_svm_bytecode_kind(local_guard_target.instructions[pc]);
    if (kind == SurfaceSvmBytecodeKind::jump_if_one ||
        kind == SurfaceSvmBytecodeKind::jump_if_zero) {
      local_guard_target.instructions[pc].payload1 -=
          descriptor.instruction_begin;
      break;
    }
  }
  require(validate_surface_svm_scene_image(local_guard_target)
                  .find("leaves its program slice") != std::string::npos,
          "scene verifier accepted a non-rebased guard target");

  auto cross_program_closure_operands = scene;
  for (auto pc = descriptor.instruction_begin;
       pc < descriptor.instruction_begin + descriptor.instruction_count; ++pc) {
    if (surface_svm_bytecode_kind(
            cross_program_closure_operands.instructions[pc]) ==
        SurfaceSvmBytecodeKind::closure_leaf) {
      cross_program_closure_operands.instructions[pc].payload0 = 0u;
      break;
    }
  }
  require(validate_surface_svm_scene_image(cross_program_closure_operands)
                  .find("before its program slice") != std::string::npos,
          "scene verifier accepted a cross-program closure operand range");

  auto noncanonical_descriptor = scene;
  noncanonical_descriptor.programs[1u].reserved = 1u;
  require(validate_surface_svm_scene_image(noncanonical_descriptor)
                  .find("not dense and canonical") != std::string::npos,
          "scene verifier accepted a noncanonical descriptor");
}

void test_scene_rebases_metadata_and_static_tables() {
  const auto program = make_static_table_program();
  require(!program.value_metadata.empty() && !program.static_data.empty(),
          "static-table fixture lacks metadata relocation classes");
  const std::array programs{program, program};
  const auto scene = build_surface_svm_scene_image(programs);
  require(scene.valid,
          "static-table scene aggregation failed: " + scene.diagnostic);
  const auto &descriptor = scene.programs[1u];
  const auto &side = scene.side_ranges[1u];
  auto observed_metadata = false;
  for (auto local_pc = std::size_t{}; local_pc < program.instructions.size();
       ++local_pc) {
    const auto local =
        surface_svm_value_instruction(program.instructions[local_pc]);
    if (surface_svm_bytecode_kind(program.instructions[local_pc]) !=
            SurfaceSvmBytecodeKind::value ||
        local.metadata_index == SurfaceValueAddress::invalid_value) {
      continue;
    }
    const auto global = surface_svm_value_instruction(
        scene.instructions[descriptor.instruction_begin + local_pc]);
    require(global.metadata_index == local.metadata_index + side.metadata_begin,
            "value metadata index was not rebased exactly");
    observed_metadata = true;
  }
  require(observed_metadata,
          "static-table fixture did not expose a metadata reference");
  for (auto local_index = std::size_t{};
       local_index < program.value_metadata.size(); ++local_index) {
    require(scene.value_metadata[side.metadata_begin + local_index]
                    .static_table_begin ==
                program.value_metadata[local_index].static_table_begin +
                    side.static_data_begin,
            "metadata static-table begin was not rebased exactly");
  }
  require(validate_surface_svm_scene_image(scene).empty(),
          "static-table aggregate failed its public verifier");
}

} // namespace

int main() {
  try {
    test_set_normal_starts_a_new_local_lifetime_epoch();
    test_scene_rebases_control_and_typed_side_streams();
    test_scene_rebases_metadata_and_static_tables();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "surface SVM scene tests passed\n";
  return 0;
}
