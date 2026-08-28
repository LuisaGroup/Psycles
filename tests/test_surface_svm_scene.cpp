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

struct ProvenanceFixture {
  SurfaceProgram program;
  SurfaceSvmProgramImage image;
  std::vector<std::uint32_t> instruction_sources;
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

[[nodiscard]] constexpr std::uint32_t lane_extent(
    std::uint32_t scalar_slots, std::uint32_t vector_slots,
    std::uint32_t unsigned_integer_slots) noexcept {
  return scalar_slots + 3u * vector_slots + 2u * unsigned_integer_slots;
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
  root.stack_lanes = lane_extent(root.scalar_slots, root.vector_slots, 0u);
  root.value_instruction_count = 2u;
  root.instructions.emplace_back(
      make_surface_svm_value_instruction(SurfaceValueBytecodeInstruction{
          .control = make_surface_value_control(
              ValueOperation::surface_position, SurfaceValueBank::vector, 0u),
          // Scalar lane zero precedes two three-lane vector colors. The
          // second vector therefore begins at physical lane four.
          .result = local_address(SurfaceValueBank::vector, 4u),
          .operand_payload = surface_value_invalid_operand_word,
          .metadata_index = SurfaceValueAddress::invalid_value}));
  root.instructions.emplace_back(
      make_surface_svm_value_instruction(SurfaceValueBytecodeInstruction{
          .control = make_surface_value_control(
              ValueOperation::vector_to_scalar, SurfaceValueBank::scalar, 0u),
          .result = local_address(SurfaceValueBank::scalar, 0u),
          .operand_payload =
              one_operand(local_address(SurfaceValueBank::vector, 4u)),
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
  program.stack_lanes = lane_extent(
      program.scalar_slots, program.vector_slots,
      program.unsigned_integer_slots);
  program.value_instruction_count = 1u;
  require(validate_surface_svm_program_image(program).empty(),
          "static-table unified program is invalid");
  return program;
}

[[nodiscard]] ProvenanceFixture
make_provenance_fixture(SurfaceProgram program, std::vector<bool> outputs) {
  require(outputs.size() == program.value_instructions().size(),
          "provenance fixture output mask has the wrong extent");
  auto active = std::vector<bool>(outputs.size(), true);
  const auto storage =
      plan_surface_value_storage(program, active, std::move(outputs));
  require(storage.valid,
          "provenance fixture storage failed: " + storage.diagnostic);
  const auto values = lower_surface_value_program(program, storage);
  require(values.valid,
          "provenance fixture lowering failed: " + values.diagnostic);

  SurfaceSvmProgramImage image;
  image.valid = true;
  image.endpoints =
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical);
  image.value_operands = values.operands;
  image.value_metadata = values.metadata;
  image.static_data = values.static_data;
  image.value_addresses = values.value_addresses;
  image.scalar_slots = values.scalar_slots;
  image.vector_slots = values.vector_slots;
  image.unsigned_integer_slots = values.unsigned_integer_slots;
  image.stack_lanes = lane_extent(
      image.scalar_slots, image.vector_slots,
      image.unsigned_integer_slots);
  image.value_instruction_count =
      static_cast<std::uint32_t>(values.instructions.size());
  image.instructions.reserve(values.instructions.size() + 1u);
  std::vector<std::uint32_t> sources;
  sources.reserve(values.instructions.size() + 1u);
  for (auto index = std::size_t{}; index < values.instructions.size();
       ++index) {
    image.instructions.emplace_back(
        make_surface_svm_value_instruction(values.instructions[index]));
    sources.emplace_back(storage.instructions[index].value);
  }
  image.instructions.emplace_back(
      SurfaceSvmBytecodeInstruction{.control = surface_svm_end_opcode,
                                    .payload0 = surface_svm_invalid_payload,
                                    .payload1 = surface_svm_invalid_payload,
                                    .payload2 = surface_svm_invalid_payload});
  sources.emplace_back(SurfaceValueAddress::invalid_value);
  require(validate_surface_svm_program_image(image).empty(),
          "provenance fixture unified image is invalid");
  return ProvenanceFixture{.program = std::move(program),
                           .image = std::move(image),
                           .instruction_sources = std::move(sources)};
}

[[nodiscard]] ParameterDesc make_float_parameter(std::uint32_t index) {
  return ParameterDesc{.id = ParameterId{index},
                       .node = NodeId{index + 1u},
                       .socket = "Value",
                       .type = SocketType::floating,
                       .default_value = SocketValue::floating(0.0f),
                       .source = ParameterSource::input};
}

[[nodiscard]] ValueInstruction make_float_parameter_value(std::uint32_t index) {
  return ValueInstruction{.operation = ValueOperation::parameter,
                          .source_node = NodeId{index + 1u},
                          .result_type = SocketType::floating,
                          .parameter = ParameterId{index}};
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

void test_lane_stack_rejects_cross_bank_overlap_and_has_cycles_bound() {
  const auto make_value = [](ValueOperation operation, SurfaceValueBank bank,
                             std::uint32_t lane) {
    return make_surface_svm_value_instruction(
        SurfaceValueBytecodeInstruction{
            .control = make_surface_value_control(operation, bank, 0u),
            .result = local_address(bank, lane),
            .operand_payload = surface_value_invalid_operand_word,
            .metadata_index = SurfaceValueAddress::invalid_value});
  };

  ClosureInstruction diffuse{.operation = ClosureOperation::diffuse};
  SurfaceSvmProgramImage safe;
  safe.valid = true;
  safe.endpoints =
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical);
  safe.stack_lanes = 4u;
  safe.scalar_slots = 1u;
  safe.vector_slots = 1u;
  safe.value_instruction_count = 2u;
  safe.closure_leaf_count = 1u;
  safe.used_closure_operations =
      1u << static_cast<std::uint32_t>(ClosureOperation::diffuse);
  safe.instructions = {
      make_value(ValueOperation::path_ray_length,
                 SurfaceValueBank::scalar, 0u),
      make_value(ValueOperation::surface_position,
                 SurfaceValueBank::vector, 1u),
      SurfaceSvmBytecodeInstruction{
          .control = surface_svm_closure_leaf_opcode |
                     (make_surface_closure_control(diffuse, safe.endpoints)
                      << surface_svm_closure_control_shift),
          .payload0 = 0u,
          .payload1 = surface_svm_root_weight_slot,
          .payload2 = 0u},
      SurfaceSvmBytecodeInstruction{
          .control = surface_svm_end_opcode,
          .payload0 = surface_svm_invalid_payload,
          .payload1 = surface_svm_invalid_payload,
          .payload2 = surface_svm_invalid_payload}};
  safe.closure_operands = {
      local_address(SurfaceValueBank::vector, 1u),
      SurfaceValueAddress::invalid_value,
      local_address(SurfaceValueBank::scalar, 0u)};
  require(validate_surface_svm_program_image(safe).empty(),
          "non-overlapping mixed-bank lane fixture is invalid");

  auto overlap = safe;
  overlap.instructions[1u] =
      make_value(ValueOperation::surface_position,
                 SurfaceValueBank::vector, 0u);
  overlap.closure_operands[0u] =
      local_address(SurfaceValueBank::vector, 0u);
  require(validate_surface_svm_program_image(overlap)
                  .find("undefined value") != std::string::npos,
          "lane verifier accepted a vector write that clobbers a live "
          "scalar from another semantic bank");

  auto maximum = safe;
  maximum.stack_lanes = surface_svm_stack_lane_capacity;
  require(validate_surface_svm_program_image(maximum).empty(),
          "the exact Cycles SVM stack bound was rejected");
  maximum.stack_lanes = surface_svm_stack_lane_capacity + 1u;
  require(validate_surface_svm_program_image(maximum)
                  .find("255-lane") != std::string::npos,
          "the bytecode verifier accepted a stack beyond Cycles' bound");
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
  noncanonical_descriptor.programs[1u].instruction_begin += 1u;
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

void test_unified_evaluator_provenance_and_exact_interning() {
  const SurfaceProgram parameter_normal_program{
      100u,
      {ParameterDesc{.id = ParameterId{0u},
                     .node = NodeId{1u},
                     .socket = "Normal",
                     .type = SocketType::normal,
                     .default_value = SocketValue::normal({0.0f, 0.0f, 1.0f}),
                     .source = ParameterSource::input}},
      {ValueInstruction{.operation = ValueOperation::parameter,
                        .result_type = SocketType::normal,
                        .parameter = ParameterId{0u}}},
      {},
      {}};
  const auto parameter_normal_storage =
      plan_surface_value_storage(parameter_normal_program, {true}, {true});
  const auto parameter_normal_prefix = lower_surface_value_program(
      parameter_normal_program, parameter_normal_storage);
  SurfaceSvmProgramImage parameter_normal_root;
  parameter_normal_root.valid = true;
  parameter_normal_root.endpoints =
      surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical);
  parameter_normal_root.instructions.emplace_back(
      SurfaceSvmBytecodeInstruction{.control = surface_svm_end_opcode,
                                    .payload0 = surface_svm_invalid_payload,
                                    .payload1 = surface_svm_invalid_payload,
                                    .payload2 = surface_svm_invalid_payload});
  const auto parameter_normal_image = compose_surface_svm_normal_transaction(
      parameter_normal_prefix, parameter_normal_prefix.value_addresses.front(),
      parameter_normal_root, false);
  const std::array parameter_normal_sources{SurfaceValueAddress::invalid_value,
                                            SurfaceValueAddress::invalid_value};
  const std::array parameter_normal_input{SurfaceSvmEvaluatorProgramInput{
      .program = &parameter_normal_program,
      .image = &parameter_normal_image,
      .instruction_sources = parameter_normal_sources,
      .surface_normal_output = ValueExpressionId{0u}}};
  const auto parameter_normal_scene =
      build_surface_svm_executable_scene(parameter_normal_input);
  require(parameter_normal_scene.valid &&
              parameter_normal_scene.value_variants.empty(),
          "unified provenance rejected a parameter-backed SetNormal "
          "transaction: " +
              parameter_normal_scene.diagnostic);

  const auto make_absolute_program = [](std::uint32_t signature,
                                        bool parameter_source,
                                        float static_f0 = 0.0f) {
    std::vector<ParameterDesc> parameters;
    std::vector<ValueInstruction> values;
    if (parameter_source) {
      parameters.emplace_back(make_float_parameter(0u));
      values.emplace_back(make_float_parameter_value(0u));
    } else {
      values.emplace_back(
          ValueInstruction{.operation = ValueOperation::path_ray_length,
                           .result_type = SocketType::floating});
    }
    values.emplace_back(ValueInstruction{
        .operation = ValueOperation::absolute,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::unary>(
            {{value_operand::unary::input, ValueExpressionId{0u}}}),
        .static_f0 = static_f0});
    return SurfaceProgram{
        signature, std::move(parameters), std::move(values), {}, {}};
  };

  auto parameter_absolute =
      make_provenance_fixture(make_absolute_program(101u, true), {false, true});
  auto local_absolute = make_provenance_fixture(
      make_absolute_program(102u, false), {false, true});
  const std::array route_inputs{
      SurfaceSvmEvaluatorProgramInput{
          .program = &parameter_absolute.program,
          .image = &parameter_absolute.image,
          .instruction_sources = parameter_absolute.instruction_sources},
      SurfaceSvmEvaluatorProgramInput{.program = &local_absolute.program,
                                      .image = &local_absolute.image,
                                      .instruction_sources =
                                          local_absolute.instruction_sources}};
  const auto route_scene = build_surface_svm_executable_scene(route_inputs);
  require(route_scene.valid,
          "unified evaluator scene failed: " + route_scene.diagnostic);
  require(route_scene.image.programs.size() == 2u &&
              route_scene.value_variants.size() == 2u &&
              route_scene.instruction_variants.size() ==
                  route_scene.image.instructions.size(),
          "unified evaluator scene lost its exact parallel domains");
  const auto parameter_absolute_pc =
      route_scene.image.programs[0u].instruction_begin;
  const auto local_absolute_pc =
      route_scene.image.programs[1u].instruction_begin + 1u;
  const auto absolute_variant =
      route_scene.instruction_variants[parameter_absolute_pc];
  require(
      absolute_variant == route_scene.instruction_variants[local_absolute_pc] &&
          route_scene.value_variants[absolute_variant].instruction.operation ==
              ValueOperation::absolute &&
          route_scene.value_variants[absolute_variant].operand_routes ==
              std::vector{SurfaceValueOperandRoute::dynamic},
      "independently constructed exact evaluators did not merge or lost "
      "the local/parameter route join");
  for (auto pc = std::size_t{}; pc < route_scene.image.instructions.size();
       ++pc) {
    const auto is_value =
        surface_svm_bytecode_kind(route_scene.image.instructions[pc]) ==
        SurfaceSvmBytecodeKind::value;
    require(is_value == (route_scene.instruction_variants[pc] !=
                         SurfaceValueAddress::invalid_value),
            "unified evaluator relation is not total by bytecode kind");
  }

  const auto make_math_program = [](std::uint32_t signature,
                                    MathOperation operation) {
    std::vector<ParameterDesc> parameters;
    std::vector<ValueInstruction> values;
    for (auto index = std::uint32_t{}; index < 3u; ++index) {
      parameters.emplace_back(make_float_parameter(index));
      values.emplace_back(make_float_parameter_value(index));
    }
    values.emplace_back(ValueInstruction{
        .operation = ValueOperation::math,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::ternary>(
            {{value_operand::ternary::a, ValueExpressionId{0u}},
             {value_operand::ternary::b, ValueExpressionId{1u}},
             {value_operand::ternary::c, ValueExpressionId{2u}}}),
        .static_u0 = static_cast<std::uint64_t>(operation)});
    return SurfaceProgram{
        signature, std::move(parameters), std::move(values), {}, {}};
  };
  auto math_add = make_provenance_fixture(
      make_math_program(103u, MathOperation::add), {false, false, false, true});
  auto math_subtract =
      make_provenance_fixture(make_math_program(104u, MathOperation::subtract),
                              {false, false, false, true});
  const std::array immediate_inputs{
      SurfaceSvmEvaluatorProgramInput{.program = &math_add.program,
                                      .image = &math_add.image,
                                      .instruction_sources =
                                          math_add.instruction_sources},
      SurfaceSvmEvaluatorProgramInput{.program = &math_subtract.program,
                                      .image = &math_subtract.image,
                                      .instruction_sources =
                                          math_subtract.instruction_sources}};
  const auto immediate_scene =
      build_surface_svm_executable_scene(immediate_inputs);
  require(immediate_scene.valid &&
              immediate_scene.value_variants.size() == 1u &&
              immediate_scene.value_variants.front().svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(MathOperation::add),
                      static_cast<std::uint16_t>(MathOperation::subtract)} &&
              immediate_scene.value_variants.front().operand_routes ==
                  std::vector(3u, SurfaceValueOperandRoute::parameter),
          "opcode-owned immediates multiplied evaluator bodies or lost their "
          "exact finite domain");

  auto positive_zero = make_provenance_fixture(
      make_absolute_program(105u, true, 0.0f), {false, true});
  auto negative_zero = make_provenance_fixture(
      make_absolute_program(106u, true, -0.0f), {false, true});
  const std::array exact_bit_inputs{
      SurfaceSvmEvaluatorProgramInput{.program = &positive_zero.program,
                                      .image = &positive_zero.image,
                                      .instruction_sources =
                                          positive_zero.instruction_sources},
      SurfaceSvmEvaluatorProgramInput{.program = &negative_zero.program,
                                      .image = &negative_zero.image,
                                      .instruction_sources =
                                          negative_zero.instruction_sources}};
  const auto exact_bit_scene =
      build_surface_svm_executable_scene(exact_bit_inputs);
  require(exact_bit_scene.valid &&
              exact_bit_scene.value_variants.size() == 2u &&
              std::bit_cast<std::uint32_t>(
                  exact_bit_scene.value_variants[0u].instruction.static_f0) !=
                  std::bit_cast<std::uint32_t>(
                      exact_bit_scene.value_variants[1u].instruction.static_f0),
          "exact evaluator interning erased signed-zero semantic bits");

  auto mismatched_metadata = negative_zero.image;
  require(mismatched_metadata.value_metadata.size() == 1u,
          "signed-zero fixture lacks exact metadata");
  mismatched_metadata.value_metadata.front().static_f0 = 0.0f;
  require(validate_surface_svm_program_image(mismatched_metadata).empty(),
          "metadata mutation unexpectedly broke structural bytecode");
  const std::array mismatched_metadata_input{SurfaceSvmEvaluatorProgramInput{
      .program = &negative_zero.program,
      .image = &mismatched_metadata,
      .instruction_sources = negative_zero.instruction_sources}};
  const auto mismatched_metadata_scene =
      build_surface_svm_executable_scene(mismatched_metadata_input);
  require(!mismatched_metadata_scene.valid &&
              mismatched_metadata_scene.diagnostic.find("metadata disagrees") !=
                  std::string::npos,
          "unified provenance accepted bytecode data from another exact "
          "source value");

  const SurfaceProgram ordered_sources{
      107u,
      {},
      {ValueInstruction{.operation = ValueOperation::path_ray_length,
                        .result_type = SocketType::floating},
       ValueInstruction{.operation = ValueOperation::curve_length,
                        .result_type = SocketType::floating},
       ValueInstruction{
           .operation = ValueOperation::subtract,
           .result_type = SocketType::floating,
           .operands = make_value_operands<value_operand::binary>(
               {{value_operand::binary::a, ValueExpressionId{0u}},
                {value_operand::binary::b, ValueExpressionId{1u}}})}},
      {},
      {}};
  auto ordered = make_provenance_fixture(ordered_sources, {false, false, true});
  auto swapped = ordered.image;
  auto value = surface_svm_value_instruction(swapped.instructions[2u]);
  const auto first = surface_value_operand_from_word(value.operand_payload,
                                                     value_operand::binary::a);
  const auto second = surface_value_operand_from_word(value.operand_payload,
                                                      value_operand::binary::b);
  value.operand_payload = static_cast<std::uint32_t>(second.encoded()) |
                          (static_cast<std::uint32_t>(first.encoded())
                           << surface_value_operand_lane_bits);
  swapped.instructions[2u] = make_surface_svm_value_instruction(value);
  require(validate_surface_svm_program_image(swapped).empty(),
          "operand permutation unexpectedly broke structural bytecode");
  const std::array swapped_input{SurfaceSvmEvaluatorProgramInput{
      .program = &ordered.program,
      .image = &swapped,
      .instruction_sources = ordered.instruction_sources}};
  const auto swapped_scene = build_surface_svm_executable_scene(swapped_input);
  require(!swapped_scene.valid &&
              swapped_scene.diagnostic.find("does not contain its source") !=
                  std::string::npos,
          "unified provenance accepted a type-correct but semantically "
          "permuted local operand");
}

} // namespace

int main() {
  try {
    test_set_normal_starts_a_new_local_lifetime_epoch();
    test_lane_stack_rejects_cross_bank_overlap_and_has_cycles_bound();
    test_scene_rebases_control_and_typed_side_streams();
    test_scene_rebases_metadata_and_static_tables();
    test_unified_evaluator_provenance_and_exact_interning();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "surface SVM scene tests passed\n";
  return 0;
}
