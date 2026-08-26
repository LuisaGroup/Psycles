#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_bump_expansion.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

#include "surface_program_metadata_closure_tests.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
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

void test_surface_value_storage_plan() {
  const auto make_parameter = [](std::uint32_t index) {
    return ParameterDesc{.id = ParameterId{index},
                         .node = NodeId{index + 1u},
                         .socket = "Value",
                         .type = SocketType::floating,
                         .default_value = SocketValue::floating(0.0f),
                         .source = ParameterSource::input};
  };
  const auto make_parameter_value = [](std::uint32_t index) {
    return ValueInstruction{.operation = ValueOperation::parameter,
                            .source_node = NodeId{index + 1u},
                            .result_type = SocketType::floating,
                            .parameter = ParameterId{index}};
  };

  std::vector<ValueInstruction> values;
  values.emplace_back(make_parameter_value(0u));
  values.emplace_back(make_parameter_value(1u));
  values.emplace_back(ValueInstruction{
      .operation = ValueOperation::add,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::binary>(
          {{value_operand::binary::a, ValueExpressionId{0u}},
           {value_operand::binary::b, ValueExpressionId{1u}}})});
  values.emplace_back(ValueInstruction{
      .operation = ValueOperation::absolute,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::unary>(
          {{value_operand::unary::input, ValueExpressionId{2u}}})});
  values.emplace_back(ValueInstruction{
      .operation = ValueOperation::add,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::binary>(
          {{value_operand::binary::a, ValueExpressionId{3u}},
           {value_operand::binary::b, ValueExpressionId{1u}}})});
  const SurfaceProgram program{
      1u, {make_parameter(0u), make_parameter(1u)}, std::move(values), {}, {}};
  const std::vector<bool> active(5u, true);
  auto outputs = std::vector<bool>(5u, false);
  outputs[4u] = true;
  const auto plan = plan_surface_value_storage(program, active, outputs);
  require(plan.compatible(program),
          "typed value storage plan failed: " + plan.diagnostic);
  require(plan.active_values == 5u && plan.parameter_values == 2u &&
              plan.instructions.size() == 3u,
          "typed value storage plan did not elide parameter instructions");
  require(plan.scalar_slots == 1u && plan.vector_slots == 0u &&
              plan.unsigned_integer_slots == 0u &&
              plan.payload_bytes() == sizeof(float),
          "read-before-write liveness did not reuse a dying scalar slot");
  require(plan.locations[0u].storage == SurfaceValueStorageClass::parameter &&
              plan.locations[1u].storage == SurfaceValueStorageClass::parameter,
          "parameter expressions were copied into local slots");
  for (auto index = std::size_t{2u}; index < 5u; ++index) {
    require(plan.locations[index].storage ==
                    SurfaceValueStorageClass::local_slot &&
                plan.locations[index].bank == SurfaceValueBank::scalar &&
                plan.locations[index].index == 0u,
            "a linear scalar chain did not share its single live slot");
  }
  const auto image = lower_surface_value_program(program, plan);
  require(image.valid,
          "typed value program lowering failed: " + image.diagnostic);
  require(image.instructions.size() == 3u && image.operands.size() == 5u &&
              image.metadata.empty() && image.static_data.empty() &&
              image.value_addresses.size() == 5u,
          "compact value program has an unexpected stream extent");
  for (auto index = std::size_t{0u}; index < 2u; ++index) {
    const auto address = SurfaceValueAddress{image.value_addresses[index]};
    require(address.valid() && address.parameter() &&
                address.bank() == SurfaceValueBank::scalar &&
                address.index() == index,
            "compact value program lost a parameter address");
  }
  for (auto index = std::size_t{2u}; index < 5u; ++index) {
    const auto address = SurfaceValueAddress{image.value_addresses[index]};
    require(address.valid() && !address.parameter() &&
                address.bank() == SurfaceValueBank::scalar &&
                address.index() == 0u,
            "compact value program lost a reused local address");
  }
  require(image.instructions[0u].operand_begin == 0u &&
              image.instructions[1u].operand_begin == 2u &&
              image.instructions[2u].operand_begin == 3u &&
              surface_value_operation(image.instructions[0u]) ==
                  ValueOperation::add &&
              surface_value_operand_count(image.instructions[0u]) == 2u &&
              surface_value_result_bank(image.instructions[0u]) ==
                  SurfaceValueBank::scalar &&
              surface_value_operation(image.instructions[1u]) ==
                  ValueOperation::absolute &&
              surface_value_operand_count(image.instructions[1u]) == 1u,
          "compact value program changed topological instruction order");

  // Deliberately place a shallow independent branch before a two-value
  // branch. Source order needs three scalar slots: the shallow result stays
  // live while both deep operands are formed. Cycles-style Sethi-Ullman
  // scheduling evaluates the high-pressure branch first and needs two.
  std::vector<ValueInstruction> pressure_values;
  pressure_values.emplace_back(make_parameter_value(0u));
  pressure_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::absolute,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::unary>(
          {{value_operand::unary::input, ValueExpressionId{0u}}})});
  pressure_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::clamp01,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::unary>(
          {{value_operand::unary::input, ValueExpressionId{0u}}})});
  pressure_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::add,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::binary>(
          {{value_operand::binary::a, ValueExpressionId{0u}},
           {value_operand::binary::b, ValueExpressionId{0u}}})});
  pressure_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::add,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::binary>(
          {{value_operand::binary::a, ValueExpressionId{2u}},
           {value_operand::binary::b, ValueExpressionId{3u}}})});
  pressure_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::add,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::binary>(
          {{value_operand::binary::a, ValueExpressionId{4u}},
           {value_operand::binary::b, ValueExpressionId{1u}}})});
  const SurfaceProgram pressure_program{
      2u, {make_parameter(0u)}, std::move(pressure_values), {}, {}};
  const auto pressure_active =
      std::vector<bool>(pressure_program.value_instructions().size(), true);
  auto pressure_outputs =
      std::vector<bool>(pressure_program.value_instructions().size(), false);
  pressure_outputs[5u] = true;
  const auto pressure_plan = plan_surface_value_storage(
      pressure_program, pressure_active, pressure_outputs);
  require(pressure_plan.compatible(pressure_program),
          "Sethi-Ullman value plan failed: " + pressure_plan.diagnostic);
  require(pressure_plan.instructions ==
              std::vector<ValueExpressionId>{
                  ValueExpressionId{2u}, ValueExpressionId{3u},
                  ValueExpressionId{4u}, ValueExpressionId{1u},
                  ValueExpressionId{5u}},
          "value scheduler did not prioritize the high-pressure branch");
  require(pressure_plan.scalar_slots == 2u,
          "Sethi-Ullman scheduling did not remove the avoidable live slot");
  const auto pressure_image =
      lower_surface_value_program(pressure_program, pressure_plan);
  require(pressure_image.valid && pressure_image.instructions.size() == 5u &&
              surface_value_operation(pressure_image.instructions[0u]) ==
                  ValueOperation::clamp01 &&
              surface_value_operation(pressure_image.instructions[3u]) ==
                  ValueOperation::absolute,
          "compact bytecode did not preserve the scheduled dependency order");

  // Cycles has one scalar stack, whereas the Psycles interpreter has typed
  // banks. This DAG is the minimal counterexample where scalar-stack
  // Sethi-Ullman order would keep two vectors live (28 B total) while the
  // authored topological order needs one vector and one scalar (16 B).
  // The planner must retain the lower-pressure legal order.
  const SurfaceProgram typed_pressure_program{
      3u,
      {},
      {ValueInstruction{.operation = ValueOperation::surface_position,
                        .result_type = SocketType::point},
       ValueInstruction{
           .operation = ValueOperation::vector_to_scalar,
           .result_type = SocketType::floating,
           .operands = make_value_operands<value_operand::unary>(
               {{value_operand::unary::input, ValueExpressionId{0u}}})},
       ValueInstruction{
           .operation = ValueOperation::passthrough,
           .result_type = SocketType::vector,
           .operands = make_value_operands<value_operand::unary>(
               {{value_operand::unary::input, ValueExpressionId{0u}}})},
       ValueInstruction{
           .operation = ValueOperation::absolute,
           .result_type = SocketType::floating,
           .operands = make_value_operands<value_operand::unary>(
               {{value_operand::unary::input, ValueExpressionId{1u}}})}},
      {},
      {}};
  const auto typed_pressure_active = std::vector<bool>(4u, true);
  const auto typed_pressure_outputs =
      std::vector<bool>{false, false, true, true};
  const auto typed_pressure_plan = plan_surface_value_storage(
      typed_pressure_program, typed_pressure_active, typed_pressure_outputs);
  require(typed_pressure_plan.compatible(typed_pressure_program),
          "typed-bank pressure plan failed: " + typed_pressure_plan.diagnostic);
  require(typed_pressure_plan.instructions ==
                  std::vector<ValueExpressionId>{
                      ValueExpressionId{0u}, ValueExpressionId{1u},
                      ValueExpressionId{2u}, ValueExpressionId{3u}} &&
              typed_pressure_plan.scalar_slots == 1u &&
              typed_pressure_plan.vector_slots == 1u &&
              typed_pressure_plan.payload_bytes() == 16u,
          "typed-bank no-regression selection retained a worse schedule");

  std::vector<float> transform(16u, 0.0f);
  for (auto diagonal = std::size_t{0u}; diagonal < 4u; ++diagonal) {
    transform[diagonal * 5u] = 1.0f;
  }
  const SurfaceProgram metadata_program{
      3u,
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
  const auto metadata_plan = plan_surface_value_storage(
      metadata_program, std::vector<bool>{true}, std::vector<bool>{true});
  const auto metadata_image =
      lower_surface_value_program(metadata_program, metadata_plan);
  require(metadata_image.valid && metadata_image.instructions.size() == 1u &&
              metadata_image.metadata.size() == 1u &&
              metadata_image.static_data == transform &&
              metadata_image.metadata[0u].static_u0 == 7u &&
              metadata_image.metadata[0u].static_f0 == 0.25f &&
              std::bit_cast<std::uint32_t>(
                  metadata_image.metadata[0u].static_f1) == 0x80000000u &&
              metadata_image.metadata[0u].static_table_begin == 0u &&
              metadata_image.metadata[0u].static_table_count == 16u,
          "compact value program lost immutable instruction metadata");

  const std::vector scene_programs{image, metadata_image, image,
                                   metadata_image};
  const auto scene_image = build_surface_value_scene_image(scene_programs);
  require(scene_image.valid && scene_image.programs.size() == 4u &&
              scene_image.instructions.size() == 8u &&
              scene_image.operands.size() == 10u &&
              scene_image.metadata.size() == 2u &&
              scene_image.static_data.size() == 32u,
          "scene value-program aggregation changed a stream extent");
  require(scene_image.programs[0u].instruction_begin == 0u &&
              scene_image.programs[0u].instruction_count == 3u &&
              scene_image.programs[1u].instruction_begin == 3u &&
              scene_image.programs[2u].instruction_begin == 4u &&
              scene_image.programs[3u].instruction_begin == 7u,
          "scene value-program descriptors do not preserve tag order");
  require(scene_image.instructions[4u].operand_begin == 5u &&
              scene_image.instructions[7u].metadata_index == 1u &&
              scene_image.metadata[1u].static_table_begin == 16u,
          "scene value-program aggregation did not rebase a global stream");
  require(scene_image.programs[0u].scalar_slots == 1u &&
              scene_image.programs[1u].vector_slots == 1u,
          "scene value-program descriptors lost typed slot bounds");

  const std::vector execution_inputs{
      SurfaceValueExecutionInput{.program = &program, .storage = &plan},
      SurfaceValueExecutionInput{.program = &metadata_program,
                                 .storage = &metadata_plan},
      SurfaceValueExecutionInput{.program = &program, .storage = &plan},
      SurfaceValueExecutionInput{.program = &metadata_program,
                                 .storage = &metadata_plan}};
  const auto executable_scene =
      build_surface_value_executable_scene(execution_inputs);
  require(executable_scene.valid && executable_scene.variants.size() == 3u &&
              executable_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 1u, 0u, 2u, 0u, 1u, 0u, 2u},
          "exact immutable variants were not interned in first-use order");
  require(executable_scene.variants[0u].operand_types ==
                  std::vector<SocketType>{SocketType::floating,
                                          SocketType::floating} &&
              executable_scene.variants[0u].instruction.operands ==
                  std::vector<ValueExpressionId>{ValueExpressionId{0u},
                                                 ValueExpressionId{1u}},
          "an immutable variant lost its typed normalized operands");

  const auto make_passthrough_program = [](std::uint32_t tag, SocketType type,
                                           SocketValue value) {
    return SurfaceProgram{
        tag,
        {ParameterDesc{.id = ParameterId{0u},
                       .node = NodeId{tag + 1u},
                       .socket = "Value",
                       .type = type,
                       .default_value = std::move(value),
                       .source = ParameterSource::input}},
        {ValueInstruction{.operation = ValueOperation::parameter,
                          .source_node = NodeId{tag + 1u},
                          .result_type = type,
                          .parameter = ParameterId{0u}},
         ValueInstruction{
             .operation = ValueOperation::passthrough,
             .source_node = NodeId{tag + 1u},
             .result_type = type,
             .operands = make_value_operands<value_operand::unary>(
                 {{value_operand::unary::input, ValueExpressionId{0u}}})}},
        {},
        {}};
  };
  const auto float_passthrough = make_passthrough_program(
      30u, SocketType::floating, SocketValue::floating(0.25f));
  const auto boolean_passthrough = make_passthrough_program(
      31u, SocketType::boolean, SocketValue::boolean(true));
  const auto color_passthrough = make_passthrough_program(
      32u, SocketType::color, SocketValue::color({0.1f, 0.2f, 0.3f}));
  const auto normal_passthrough = make_passthrough_program(
      33u, SocketType::normal, SocketValue::normal({0.0f, 0.0f, 1.0f}));
  const auto uint_passthrough = make_passthrough_program(
      34u, SocketType::unsigned_integer, SocketValue::unsigned_integer(7u));
  const auto make_passthrough_plan = [](const SurfaceProgram &source) {
    return plan_surface_value_storage(source, std::vector<bool>{true, true},
                                      std::vector<bool>{false, true});
  };
  const auto float_passthrough_plan = make_passthrough_plan(float_passthrough);
  const auto boolean_passthrough_plan =
      make_passthrough_plan(boolean_passthrough);
  const auto color_passthrough_plan = make_passthrough_plan(color_passthrough);
  const auto normal_passthrough_plan =
      make_passthrough_plan(normal_passthrough);
  const auto uint_passthrough_plan = make_passthrough_plan(uint_passthrough);
  const std::vector passthrough_inputs{
      SurfaceValueExecutionInput{.program = &float_passthrough,
                                 .storage = &float_passthrough_plan},
      SurfaceValueExecutionInput{.program = &boolean_passthrough,
                                 .storage = &boolean_passthrough_plan},
      SurfaceValueExecutionInput{.program = &color_passthrough,
                                 .storage = &color_passthrough_plan},
      SurfaceValueExecutionInput{.program = &normal_passthrough,
                                 .storage = &normal_passthrough_plan},
      SurfaceValueExecutionInput{.program = &uint_passthrough,
                                 .storage = &uint_passthrough_plan}};
  const auto passthrough_scene =
      build_surface_value_executable_scene(passthrough_inputs);
  require(passthrough_scene.valid && passthrough_scene.variants.size() == 3u &&
              passthrough_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u, 1u, 1u, 2u} &&
              passthrough_scene.variants[0u].instruction.result_type ==
                  SocketType::floating &&
              passthrough_scene.variants[0u].operand_types ==
                  std::vector<SocketType>{SocketType::floating} &&
              passthrough_scene.variants[1u].instruction.result_type ==
                  SocketType::vector &&
              passthrough_scene.variants[1u].operand_types ==
                  std::vector<SocketType>{SocketType::vector} &&
              passthrough_scene.variants[2u].instruction.result_type ==
                  SocketType::unsigned_integer &&
              passthrough_scene.variants[2u].operand_types ==
                  std::vector<SocketType>{SocketType::unsigned_integer},
          "nominal socket spellings did not quotient to the three exact SVM "
          "execution banks");

  const SurfaceProgram transaction_program{
      35u,
      {ParameterDesc{.id = ParameterId{0u},
                     .node = NodeId{40u},
                     .socket = "Normal",
                     .type = SocketType::normal,
                     .default_value = SocketValue::normal({0.0f, 0.0f, 1.0f}),
                     .source = ParameterSource::input},
       ParameterDesc{.id = ParameterId{1u},
                     .node = NodeId{41u},
                     .socket = "Color",
                     .type = SocketType::color,
                     .default_value = SocketValue::color({0.2f, 0.4f, 0.6f}),
                     .source = ParameterSource::input}},
      {ValueInstruction{.operation = ValueOperation::parameter,
                        .source_node = NodeId{40u},
                        .result_type = SocketType::normal,
                        .parameter = ParameterId{0u}},
       ValueInstruction{
           .operation = ValueOperation::passthrough,
           .source_node = NodeId{40u},
           .result_type = SocketType::normal,
           .operands = make_value_operands<value_operand::unary>(
               {{value_operand::unary::input, ValueExpressionId{0u}}})},
       ValueInstruction{.operation = ValueOperation::parameter,
                        .source_node = NodeId{41u},
                        .result_type = SocketType::color,
                        .parameter = ParameterId{1u}},
       ValueInstruction{
           .operation = ValueOperation::passthrough,
           .source_node = NodeId{41u},
           .result_type = SocketType::color,
           .operands = make_value_operands<value_operand::unary>(
               {{value_operand::unary::input, ValueExpressionId{2u}}})}},
      {},
      {},
      {},
      {},
      ValueExpressionId{1u}};
  const auto transaction_normal_plan = plan_surface_value_storage(
      transaction_program, std::vector<bool>{true, true, false, false},
      std::vector<bool>{false, true, false, false});
  const auto transaction_root_plan = plan_surface_value_storage(
      transaction_program, std::vector<bool>{false, false, true, true},
      std::vector<bool>{false, false, false, true});
  const auto transaction_scene = build_surface_value_executable_scene(
      std::vector{SurfaceValueExecutionInput{
          .program = &transaction_program,
          .storage = &transaction_root_plan,
          .surface_normal_storage = &transaction_normal_plan,
          .surface_normal_output = ValueExpressionId{1u},
          .surface_normal_uses_undisplaced_geometry = true}});
  const auto transaction_invalid = SurfaceValueAddress::invalid_value;
  require(
      transaction_scene.valid && transaction_scene.variants.size() == 1u &&
          transaction_scene.instruction_variants ==
              std::vector<std::uint32_t>{0u, transaction_invalid, 0u} &&
          transaction_scene.values.programs.size() == 1u &&
          transaction_scene.values.programs[0u].instruction_count == 3u &&
          transaction_scene.values.programs[0u].flags ==
              surface_value_program_automatic_normal_uses_undisplaced_geometry &&
          transaction_scene.values.instructions.size() == 3u &&
          is_surface_value_surface_normal_transition(
              transaction_scene.values.instructions[1u]) &&
          transaction_scene.values.instructions[1u].operand_begin == 1u &&
          transaction_scene.values.instructions[1u].result ==
              transaction_scene.values.instructions[0u].result &&
          transaction_scene.values.instructions[2u].result ==
              transaction_scene.values.instructions[0u].result &&
          transaction_scene.values.operands.size() == 2u,
      "automatic normal and endpoint root did not compose into one exact "
      "transaction stream with a consuming slot-overlap boundary");

  const auto parameter_normal_plan = plan_surface_value_storage(
      transaction_program, std::vector<bool>{true, false, false, false},
      std::vector<bool>{true, false, false, false});
  const auto parameter_normal_scene = build_surface_value_executable_scene(
      std::vector{SurfaceValueExecutionInput{
          .program = &transaction_program,
          .storage = &transaction_root_plan,
          .surface_normal_storage = &parameter_normal_plan,
          .surface_normal_output = ValueExpressionId{0u}}});
  require(parameter_normal_scene.valid &&
              parameter_normal_scene.instruction_variants ==
                  std::vector<std::uint32_t>{transaction_invalid, 0u} &&
              parameter_normal_scene.values.instructions.size() == 2u &&
              is_surface_value_surface_normal_transition(
                  parameter_normal_scene.values.instructions.front()) &&
              SurfaceValueAddress{
                  parameter_normal_scene.values.instructions.front().result}
                  .parameter(),
          "a parameter-only automatic normal lost its zero-instruction "
          "transaction boundary");

  const auto incomplete_transaction = build_surface_value_executable_scene(
      std::vector{SurfaceValueExecutionInput{.program = &transaction_program,
                                             .storage = &transaction_root_plan,
                                             .surface_normal_storage =
                                                 &transaction_normal_plan}});
  require(!incomplete_transaction.valid &&
              incomplete_transaction.diagnostic.find(
                  "automatic-normal transaction") != std::string::npos,
          "an incomplete automatic-normal transaction was accepted");

  const auto transaction_normal_image =
      lower_surface_value_program(transaction_program, transaction_normal_plan);
  require(transaction_normal_image.valid &&
              transaction_normal_image.instructions.size() == 1u,
          "failed to construct the transaction verifier fixture");
  auto raw_transaction = transaction_normal_image;
  raw_transaction.instructions.emplace_back(SurfaceValueBytecodeInstruction{
      .control = surface_value_surface_normal_transition_control,
      .result = transaction_normal_image.instructions.front().result,
      .operand_begin =
          static_cast<std::uint32_t>(raw_transaction.operands.size()),
      .metadata_index = SurfaceValueAddress::invalid_value});
  require(build_surface_value_scene_image(std::vector{raw_transaction}).valid,
          "the bytecode verifier rejected a well-formed normal commit");

  auto uninitialized_instruction = transaction_normal_image;
  require(!uninitialized_instruction.operands.empty(),
          "the definite-initialization fixture has no operand");
  uninitialized_instruction.operands.front() =
      uninitialized_instruction.instructions.front().result;
  const auto uninitialized_instruction_scene =
      build_surface_value_scene_image(std::vector{uninitialized_instruction});
  require(!uninitialized_instruction_scene.valid &&
              uninitialized_instruction_scene.diagnostic.find(
                  "uninitialized") != std::string::npos,
          "the bytecode verifier accepted an ordinary read-before-write");

  auto repeated_transition = raw_transaction;
  repeated_transition.instructions.emplace_back(
      repeated_transition.instructions.back());
  const auto repeated_transition_scene =
      build_surface_value_scene_image(std::vector{repeated_transition});
  require(!repeated_transition_scene.valid &&
              repeated_transition_scene.diagnostic.find("multiple") !=
                  std::string::npos,
          "the bytecode verifier accepted two normal commits");

  auto unknown_transaction_flags = raw_transaction;
  unknown_transaction_flags.flags = surface_value_program_flag_mask << 1u;
  const auto unknown_flags_scene =
      build_surface_value_scene_image(std::vector{unknown_transaction_flags});
  require(!unknown_flags_scene.valid &&
              unknown_flags_scene.diagnostic.find("unknown flags") !=
                  std::string::npos,
          "the bytecode verifier accepted unknown transaction flags");

  auto uninitialized_transition = raw_transaction;
  uninitialized_transition.instructions.erase(
      uninitialized_transition.instructions.begin());
  uninitialized_transition.operands.clear();
  uninitialized_transition.instructions.front().operand_begin = 0u;
  const auto uninitialized_transition_scene =
      build_surface_value_scene_image(std::vector{uninitialized_transition});
  require(!uninitialized_transition_scene.valid &&
              uninitialized_transition_scene.diagnostic.find("uninitialized") !=
                  std::string::npos,
          "the bytecode verifier accepted an uninitialized normal commit");

  auto positive_zero_values = metadata_program.value_instructions();
  positive_zero_values.front().static_f1 = 0.0f;
  const SurfaceProgram positive_zero_program{
      4u, {}, std::move(positive_zero_values), {}, {}};
  const auto positive_zero_plan = plan_surface_value_storage(
      positive_zero_program, std::vector<bool>{true}, std::vector<bool>{true});
  const std::vector zero_inputs{
      SurfaceValueExecutionInput{.program = &metadata_program,
                                 .storage = &metadata_plan},
      SurfaceValueExecutionInput{.program = &positive_zero_program,
                                 .storage = &positive_zero_plan}};
  const auto signed_zero_scene =
      build_surface_value_executable_scene(zero_inputs);
  require(signed_zero_scene.valid && signed_zero_scene.variants.size() == 2u,
          "exact immutable-variant interning merged signed zero");

  auto translated_transform = transform;
  translated_transform[12u] = 3.5f;
  translated_transform[13u] = -2.25f;
  auto translated_values = metadata_program.value_instructions();
  translated_values.front().static_table = translated_transform;
  const SurfaceProgram translated_program{
      5u, {}, std::move(translated_values), {}, {}};
  const auto translated_plan = plan_surface_value_storage(
      translated_program, std::vector<bool>{true}, std::vector<bool>{true});
  const std::vector transform_inputs{
      SurfaceValueExecutionInput{.program = &metadata_program,
                                 .storage = &metadata_plan},
      SurfaceValueExecutionInput{.program = &translated_program,
                                 .storage = &translated_plan}};
  const auto transform_scene =
      build_surface_value_executable_scene(transform_inputs);
  require(transform_scene.valid && transform_scene.variants.size() == 1u &&
              transform_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u},
          "equal-shape static tables did not share one semantic evaluator");
  require(
      transform_scene.values.metadata.size() == 2u &&
          transform_scene.values.metadata[0u].static_table_begin == 0u &&
          transform_scene.values.metadata[0u].static_table_count == 16u &&
          transform_scene.values.metadata[1u].static_table_begin == 16u &&
          transform_scene.values.metadata[1u].static_table_count == 16u &&
          transform_scene.values.static_data ==
              [&] {
                auto data = transform;
                data.insert(data.end(), translated_transform.begin(),
                            translated_transform.end());
                return data;
              }(),
      "semantic interning lost distinct static-table bytecode payloads");
  require(transform_scene.variants[0u].instruction.static_table.size() == 16u &&
              std::all_of(
                  transform_scene.variants[0u].instruction.static_table.begin(),
                  transform_scene.variants[0u].instruction.static_table.end(),
                  [](float value) noexcept { return value == 0.0f; }),
          "a semantic evaluator retained an authored static-table payload");

  auto foreign_normalize_image = metadata_image;
  foreign_normalize_image.instructions.front().control |=
      surface_value_noise_normalize_immediate_bit
      << surface_value_svm_immediate_shift;
  const std::vector foreign_normalize_images{foreign_normalize_image};
  const auto foreign_normalize_scene =
      build_surface_value_scene_image(foreign_normalize_images);
  require(!foreign_normalize_scene.valid &&
              foreign_normalize_scene.diagnostic.find(
                  "without an immediate contract") != std::string::npos,
          "serialized non-Noise opcode accepted a Noise-owned control bit");

  const auto make_mix_program = [](std::uint32_t tag, BlendOperation operation,
                                   bool clamp_factor, bool clamp_result) {
    std::vector<ParameterDesc> parameters{
        ParameterDesc{.id = ParameterId{0u},
                      .node = NodeId{1u},
                      .socket = "A",
                      .type = SocketType::color,
                      .default_value = SocketValue::color({0.0f, 0.0f, 0.0f}),
                      .source = ParameterSource::input},
        ParameterDesc{.id = ParameterId{1u},
                      .node = NodeId{2u},
                      .socket = "B",
                      .type = SocketType::color,
                      .default_value = SocketValue::color({1.0f, 1.0f, 1.0f}),
                      .source = ParameterSource::input},
        ParameterDesc{.id = ParameterId{2u},
                      .node = NodeId{3u},
                      .socket = "Factor",
                      .type = SocketType::floating,
                      .default_value = SocketValue::floating(0.5f),
                      .source = ParameterSource::input}};
    std::vector<ValueInstruction> mix_values{
        ValueInstruction{.operation = ValueOperation::parameter,
                         .result_type = SocketType::color,
                         .parameter = ParameterId{0u}},
        ValueInstruction{.operation = ValueOperation::parameter,
                         .result_type = SocketType::color,
                         .parameter = ParameterId{1u}},
        ValueInstruction{.operation = ValueOperation::parameter,
                         .result_type = SocketType::floating,
                         .parameter = ParameterId{2u}},
        ValueInstruction{
            .operation = ValueOperation::mix,
            .result_type = SocketType::color,
            .operands = make_value_operands<value_operand::mix>(
                {{value_operand::mix::a, ValueExpressionId{0u}},
                 {value_operand::mix::b, ValueExpressionId{1u}},
                 {value_operand::mix::factor, ValueExpressionId{2u}}}),
            .static_u0 = static_cast<std::uint64_t>(operation),
            .static_u1 = (clamp_factor ? 1u : 0u) | (clamp_result ? 2u : 0u)}};
    return SurfaceProgram{
        tag, std::move(parameters), std::move(mix_values), {}, {}};
  };
  const auto make_mix_plan = [](const SurfaceProgram &mix_program) {
    auto outputs = std::vector<bool>(4u, false);
    outputs.back() = true;
    return plan_surface_value_storage(mix_program, std::vector<bool>(4u, true),
                                      outputs);
  };
  const auto plain_mix =
      make_mix_program(30u, BlendOperation::mix, false, false);
  const auto clamped_multiply =
      make_mix_program(31u, BlendOperation::multiply, true, false);
  const auto clamped_overlay =
      make_mix_program(32u, BlendOperation::overlay, false, true);
  const auto clamped_value =
      make_mix_program(33u, BlendOperation::value, true, true);
  const auto plain_mix_plan = make_mix_plan(plain_mix);
  const auto clamped_multiply_plan = make_mix_plan(clamped_multiply);
  const auto clamped_overlay_plan = make_mix_plan(clamped_overlay);
  const auto clamped_value_plan = make_mix_plan(clamped_value);
  const auto plain_mix_image =
      lower_surface_value_program(plain_mix, plain_mix_plan);
  const auto clamped_multiply_image =
      lower_surface_value_program(clamped_multiply, clamped_multiply_plan);
  const auto clamped_overlay_image =
      lower_surface_value_program(clamped_overlay, clamped_overlay_plan);
  const auto clamped_value_image =
      lower_surface_value_program(clamped_value, clamped_value_plan);
  require(plain_mix_image.valid && clamped_multiply_image.valid &&
              clamped_overlay_image.valid && clamped_value_image.valid &&
              surface_value_svm_immediate(
                  plain_mix_image.instructions.front()) == 0u &&
              surface_value_svm_immediate(
                  clamped_multiply_image.instructions.front()) ==
                  (static_cast<std::uint32_t>(BlendOperation::multiply) |
                   surface_value_mix_factor_clamp_bit) &&
              surface_value_svm_immediate(
                  clamped_overlay_image.instructions.front()) ==
                  (static_cast<std::uint32_t>(BlendOperation::overlay) |
                   surface_value_mix_result_clamp_bit) &&
              surface_value_svm_immediate(
                  clamped_value_image.instructions.front()) ==
                  (static_cast<std::uint32_t>(BlendOperation::value) |
                   surface_value_mix_factor_clamp_bit |
                   surface_value_mix_result_clamp_bit),
          "Mix properties were not preserved by the opcode-owned immediate");

  const std::vector mix_inputs{
      SurfaceValueExecutionInput{.program = &plain_mix,
                                 .storage = &plain_mix_plan},
      SurfaceValueExecutionInput{.program = &clamped_multiply,
                                 .storage = &clamped_multiply_plan},
      SurfaceValueExecutionInput{.program = &clamped_overlay,
                                 .storage = &clamped_overlay_plan},
      SurfaceValueExecutionInput{.program = &clamped_value,
                                 .storage = &clamped_value_plan}};
  const auto mix_scene = build_surface_value_executable_scene(mix_inputs);
  const std::vector<std::uint16_t> expected_mix_immediates{
      0u,
      static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(BlendOperation::multiply) |
          surface_value_mix_factor_clamp_bit),
      static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(BlendOperation::overlay) |
          surface_value_mix_result_clamp_bit),
      static_cast<std::uint16_t>(
          static_cast<std::uint32_t>(BlendOperation::value) |
          surface_value_mix_factor_clamp_bit |
          surface_value_mix_result_clamp_bit)};
  require(mix_scene.valid && mix_scene.variants.size() == 1u &&
              mix_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u, 0u, 0u} &&
              mix_scene.variants.front().instruction.static_u0 == 0u &&
              mix_scene.variants.front().instruction.static_u1 == 0u &&
              mix_scene.variants.front().svm_immediates ==
                  expected_mix_immediates,
          "Mix SVM interning did not form the exact immediate quotient");

  auto mismatched_mix_image = plain_mix_image;
  mismatched_mix_image.instructions.front().control |=
      surface_value_mix_factor_clamp_bit << surface_value_svm_immediate_shift;
  const auto mismatched_mix_scene =
      build_surface_value_scene_image(std::vector{mismatched_mix_image});
  require(!mismatched_mix_scene.valid &&
              mismatched_mix_scene.diagnostic.find(
                  "immediate disagrees with immutable metadata") !=
                  std::string::npos,
          "serialized Mix accepted an immediate/metadata disagreement");

  const auto invalid_mix = make_mix_program(
      34u,
      static_cast<BlendOperation>(
          static_cast<std::uint32_t>(BlendOperation::value) + 1u),
      false, false);
  const auto invalid_mix_plan = make_mix_plan(invalid_mix);
  const auto invalid_mix_image =
      lower_surface_value_program(invalid_mix, invalid_mix_plan);
  require(!invalid_mix_image.valid &&
              invalid_mix_image.diagnostic.find("immediate contract") !=
                  std::string::npos,
          "Mix lowering truncated an unrepresentable operation");

  const auto make_mapping_program =
      [](std::uint32_t tag, MappingVectorType type, std::uint64_t axes) {
        std::vector<ParameterDesc> parameters;
        std::vector<ValueInstruction> mapping_values;
        parameters.reserve(value_operand::mapping::count);
        mapping_values.reserve(value_operand::mapping::count + 1u);
        for (auto index = std::uint32_t{0u};
             index < value_operand::mapping::count; ++index) {
          parameters.emplace_back(ParameterDesc{
              .id = ParameterId{index},
              .node = NodeId{index + 1u},
              .socket = "Vector" + std::to_string(index),
              .type = SocketType::vector,
              .default_value =
                  SocketValue::vector(index == value_operand::mapping::scale
                                          ? psycles::Vec3f{1.0f, 1.0f, 1.0f}
                                          : psycles::Vec3f{}),
              .source = ParameterSource::input});
          mapping_values.emplace_back(
              ValueInstruction{.operation = ValueOperation::parameter,
                               .result_type = SocketType::vector,
                               .parameter = ParameterId{index}});
        }
        mapping_values.emplace_back(ValueInstruction{
            .operation = ValueOperation::mapping,
            .result_type = SocketType::vector,
            .operands = make_value_operands<value_operand::mapping>(
                {{value_operand::mapping::vector, ValueExpressionId{0u}},
                 {value_operand::mapping::location, ValueExpressionId{1u}},
                 {value_operand::mapping::rotation, ValueExpressionId{2u}},
                 {value_operand::mapping::scale, ValueExpressionId{3u}}}),
            .static_u0 = static_cast<std::uint64_t>(type),
            .static_u1 = axes});
        return SurfaceProgram{
            tag, std::move(parameters), std::move(mapping_values), {}, {}};
      };
  const auto make_mapping_plan = [](const SurfaceProgram &mapping_program) {
    const auto count = mapping_program.value_instructions().size();
    auto outputs = std::vector<bool>(count, false);
    outputs.back() = true;
    return plan_surface_value_storage(mapping_program,
                                      std::vector<bool>(count, true), outputs);
  };
  constexpr auto remap_yzx = std::uint64_t{0x39u};
  constexpr auto remap_zxy = std::uint64_t{0x1eu};
  const auto point_mapping =
      make_mapping_program(40u, MappingVectorType::point, 0u);
  const auto texture_mapping =
      make_mapping_program(41u, MappingVectorType::texture, remap_yzx);
  const auto normal_mapping =
      make_mapping_program(42u, MappingVectorType::normal, remap_zxy);
  const auto point_mapping_plan = make_mapping_plan(point_mapping);
  const auto texture_mapping_plan = make_mapping_plan(texture_mapping);
  const auto normal_mapping_plan = make_mapping_plan(normal_mapping);
  const auto point_mapping_image =
      lower_surface_value_program(point_mapping, point_mapping_plan);
  const auto texture_mapping_image =
      lower_surface_value_program(texture_mapping, texture_mapping_plan);
  const auto normal_mapping_image =
      lower_surface_value_program(normal_mapping, normal_mapping_plan);
  const auto texture_mapping_immediate = static_cast<std::uint16_t>(
      static_cast<std::uint32_t>(MappingVectorType::texture) |
      (remap_yzx << surface_value_mapping_axes_shift));
  const auto normal_mapping_immediate = static_cast<std::uint16_t>(
      static_cast<std::uint32_t>(MappingVectorType::normal) |
      (remap_zxy << surface_value_mapping_axes_shift));
  require(point_mapping_image.valid && texture_mapping_image.valid &&
              normal_mapping_image.valid &&
              surface_value_svm_immediate(
                  point_mapping_image.instructions.front()) == 0u &&
              surface_value_svm_immediate(
                  texture_mapping_image.instructions.front()) ==
                  texture_mapping_immediate &&
              surface_value_svm_immediate(
                  normal_mapping_image.instructions.front()) ==
                  normal_mapping_immediate,
          "Mapping type/axis semantics were not preserved by bytecode");
  const std::vector mapping_inputs{
      SurfaceValueExecutionInput{.program = &point_mapping,
                                 .storage = &point_mapping_plan},
      SurfaceValueExecutionInput{.program = &texture_mapping,
                                 .storage = &texture_mapping_plan},
      SurfaceValueExecutionInput{.program = &normal_mapping,
                                 .storage = &normal_mapping_plan}};
  const auto mapping_scene =
      build_surface_value_executable_scene(mapping_inputs);
  require(mapping_scene.valid && mapping_scene.variants.size() == 1u &&
              mapping_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u, 0u} &&
              mapping_scene.variants.front().instruction.static_u0 == 0u &&
              mapping_scene.variants.front().instruction.static_u1 == 0u &&
              mapping_scene.variants.front().svm_immediates ==
                  std::vector<std::uint16_t>{0u, normal_mapping_immediate,
                                             texture_mapping_immediate},
          "Mapping SVM interning did not form the exact immediate quotient");

  auto mismatched_mapping_image = point_mapping_image;
  mismatched_mapping_image.instructions.front().control |=
      1u << surface_value_svm_immediate_shift;
  const auto mismatched_mapping_scene =
      build_surface_value_scene_image(std::vector{mismatched_mapping_image});
  require(!mismatched_mapping_scene.valid &&
              mismatched_mapping_scene.diagnostic.find(
                  "immediate disagrees with immutable metadata") !=
                  std::string::npos,
          "serialized Mapping accepted an immediate/metadata disagreement");
  const auto invalid_mapping_type =
      make_mapping_program(43u, static_cast<MappingVectorType>(4u), 0u);
  const auto invalid_mapping_axes =
      make_mapping_program(44u, MappingVectorType::point, 0x40u);
  require(
      !lower_surface_value_program(invalid_mapping_type,
                                   make_mapping_plan(invalid_mapping_type))
              .valid &&
          !lower_surface_value_program(invalid_mapping_axes,
                                       make_mapping_plan(invalid_mapping_axes))
               .valid,
      "Mapping lowering truncated an unrepresentable immediate");

  const auto make_image_program = [](std::uint32_t tag,
                                     ValueOperation operation,
                                     std::uint64_t configuration) {
    const auto environment = operation == ValueOperation::environment_color ||
                             operation == ValueOperation::environment_alpha;
    const auto color = operation == ValueOperation::image_color ||
                       operation == ValueOperation::environment_color;
    std::vector<ParameterDesc> parameters{
        ParameterDesc{.id = ParameterId{0u},
                      .node = NodeId{1u},
                      .socket = "Vector",
                      .type = SocketType::vector,
                      .default_value = SocketValue::vector({0.0f, 0.0f, 0.0f}),
                      .source = ParameterSource::input},
        ParameterDesc{.id = ParameterId{1u},
                      .node = NodeId{2u},
                      .socket = "Image",
                      .type = SocketType::unsigned_integer,
                      .default_value = SocketValue::unsigned_integer(0u),
                      .source = ParameterSource::input}};
    std::vector<ValueInstruction> image_values{
        ValueInstruction{.operation = ValueOperation::parameter,
                         .result_type = SocketType::vector,
                         .parameter = ParameterId{0u}},
        ValueInstruction{.operation = ValueOperation::parameter,
                         .result_type = SocketType::unsigned_integer,
                         .parameter = ParameterId{1u}}};
    std::vector<ValueExpressionId> operands;
    if (environment) {
      operands = make_value_operands<value_operand::environment_texture>(
          {{value_operand::environment_texture::vector, ValueExpressionId{0u}},
           {value_operand::environment_texture::image, ValueExpressionId{1u}}});
    } else {
      parameters.emplace_back(
          ParameterDesc{.id = ParameterId{2u},
                        .node = NodeId{3u},
                        .socket = "ProjectionBlend",
                        .type = SocketType::floating,
                        .default_value = SocketValue::floating(0.0f),
                        .source = ParameterSource::input});
      image_values.emplace_back(
          ValueInstruction{.operation = ValueOperation::parameter,
                           .result_type = SocketType::floating,
                           .parameter = ParameterId{2u}});
      operands = make_value_operands<value_operand::image_texture>(
          {{value_operand::image_texture::vector, ValueExpressionId{0u}},
           {value_operand::image_texture::image, ValueExpressionId{1u}},
           {value_operand::image_texture::projection_blend,
            ValueExpressionId{2u}}});
    }
    image_values.emplace_back(ValueInstruction{
        .operation = operation,
        .result_type = color ? SocketType::color : SocketType::floating,
        .operands = std::move(operands),
        .static_u1 = configuration});
    return SurfaceProgram{
        tag, std::move(parameters), std::move(image_values), {}, {}};
  };
  const auto make_image_plan = [](const SurfaceProgram &image_program) {
    const auto count = image_program.value_instructions().size();
    auto outputs = std::vector<bool>(count, false);
    outputs.back() = true;
    return plan_surface_value_storage(image_program,
                                      std::vector<bool>(count, true), outputs);
  };
  constexpr auto linear_srgb = surface_value_image_srgb_bit |
                               (1u << surface_value_image_interpolation_shift);
  constexpr auto cubic_box_extend =
      2u | surface_value_image_unassociate_alpha_bit |
      (2u << surface_value_image_interpolation_shift) |
      (1u << surface_value_image_projection_shift);
  constexpr auto nearest_sphere_clip =
      1u | (2u << surface_value_image_projection_shift);
  constexpr auto linear_mirrorball =
      (1u << surface_value_image_interpolation_shift) |
      (1u << surface_value_image_projection_shift);
  const auto image_flat =
      make_image_program(50u, ValueOperation::image_color, linear_srgb);
  const auto image_box =
      make_image_program(51u, ValueOperation::image_color, cubic_box_extend);
  const auto image_sphere =
      make_image_program(52u, ValueOperation::image_color, nearest_sphere_clip);
  const auto alpha_flat =
      make_image_program(53u, ValueOperation::image_alpha, linear_srgb);
  const auto alpha_box =
      make_image_program(54u, ValueOperation::image_alpha, cubic_box_extend);
  const auto environment_mirrorball = make_image_program(
      55u, ValueOperation::environment_color, linear_mirrorball);
  const auto image_flat_plan = make_image_plan(image_flat);
  const auto image_box_plan = make_image_plan(image_box);
  const auto image_sphere_plan = make_image_plan(image_sphere);
  const auto alpha_flat_plan = make_image_plan(alpha_flat);
  const auto alpha_box_plan = make_image_plan(alpha_box);
  const auto environment_mirrorball_plan =
      make_image_plan(environment_mirrorball);
  const std::vector image_inputs{
      SurfaceValueExecutionInput{.program = &image_flat,
                                 .storage = &image_flat_plan},
      SurfaceValueExecutionInput{.program = &image_box,
                                 .storage = &image_box_plan},
      SurfaceValueExecutionInput{.program = &image_sphere,
                                 .storage = &image_sphere_plan},
      SurfaceValueExecutionInput{.program = &alpha_flat,
                                 .storage = &alpha_flat_plan},
      SurfaceValueExecutionInput{.program = &alpha_box,
                                 .storage = &alpha_box_plan},
      SurfaceValueExecutionInput{.program = &environment_mirrorball,
                                 .storage = &environment_mirrorball_plan}};
  const auto image_scene = build_surface_value_executable_scene(image_inputs);
  require(image_scene.valid && image_scene.variants.size() == 5u &&
              image_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 1u, 0u, 2u, 3u, 4u} &&
              image_scene.variants[0u].instruction.static_u1 == 0u &&
              image_scene.variants[0u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(linear_srgb),
                      static_cast<std::uint16_t>(nearest_sphere_clip)} &&
              image_scene.variants[1u].instruction.static_u1 ==
                  (1u << surface_value_image_projection_shift) &&
              image_scene.variants[1u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(cubic_box_extend)} &&
              image_scene.variants[2u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(linear_srgb)} &&
              image_scene.variants[3u].instruction.static_u1 ==
                  (1u << surface_value_image_projection_shift) &&
              image_scene.variants[3u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(cubic_box_extend)} &&
              image_scene.variants[4u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(linear_mirrorball)},
          "Image SVM interning merged outside the exact evaluator quotient");

  auto mismatched_image =
      lower_surface_value_program(image_flat, image_flat_plan);
  require(mismatched_image.valid,
          "valid Image configuration failed bytecode lowering");
  mismatched_image.instructions.front().control ^=
      1u << surface_value_svm_immediate_shift;
  const auto mismatched_image_scene =
      build_surface_value_scene_image(std::vector{mismatched_image});
  require(!mismatched_image_scene.valid &&
              mismatched_image_scene.diagnostic.find(
                  "immediate disagrees with immutable metadata") !=
                  std::string::npos,
          "serialized Image accepted an immediate/metadata disagreement");
  const auto invalid_image_extension =
      make_image_program(56u, ValueOperation::image_color, 4u);
  const auto invalid_image_reserved =
      make_image_program(57u, ValueOperation::image_color, 1u << 14u);
  const auto invalid_environment_projection =
      make_image_program(58u, ValueOperation::environment_color,
                         2u << surface_value_image_projection_shift);
  require(
      !lower_surface_value_program(invalid_image_extension,
                                   make_image_plan(invalid_image_extension))
              .valid &&
          !lower_surface_value_program(invalid_image_reserved,
                                       make_image_plan(invalid_image_reserved))
               .valid &&
          !lower_surface_value_program(
               invalid_environment_projection,
               make_image_plan(invalid_environment_projection))
               .valid,
      "Image lowering accepted a configuration outside its exact ABI");

  const auto make_table_program = [&](std::uint32_t table_parameter,
                                      std::uint64_t interpolation = 0u) {
    std::vector<ValueInstruction> table_values;
    table_values.emplace_back(make_parameter_value(0u));
    table_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::color_ramp,
        .result_type = SocketType::color,
        .parameter = ParameterId{table_parameter},
        .operands = make_value_operands<value_operand::color_ramp>(
            {{value_operand::color_ramp::factor, ValueExpressionId{0u}}}),
        .static_u0 = interpolation});
    return SurfaceProgram{10u + table_parameter,
                          {make_parameter(0u)},
                          std::move(table_values),
                          {},
                          {}};
  };
  const auto table_program_a = make_table_program(3u);
  const auto table_program_b = make_table_program(7u, 3u);
  const auto table_plan_a =
      plan_surface_value_storage(table_program_a, std::vector<bool>(2u, true),
                                 std::vector<bool>{false, true});
  const auto table_plan_b =
      plan_surface_value_storage(table_program_b, std::vector<bool>(2u, true),
                                 std::vector<bool>{false, true});
  const std::vector table_inputs{
      SurfaceValueExecutionInput{.program = &table_program_a,
                                 .storage = &table_plan_a},
      SurfaceValueExecutionInput{.program = &table_program_b,
                                 .storage = &table_plan_b}};
  const auto table_scene = build_surface_value_executable_scene(table_inputs);
  require(table_scene.valid && table_scene.variants.size() == 1u &&
              table_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u} &&
              table_scene.variants.front().instruction.static_u0 == 0u &&
              table_scene.variants.front().svm_immediates ==
                  std::vector<std::uint16_t>{0u, 3u} &&
              table_scene.values.metadata.size() == 2u &&
              table_scene.values.metadata[0u].parameter == 3u &&
              table_scene.values.metadata[1u].parameter == 7u,
          "Color Ramp record data changed the evaluator body or lost its "
          "late-bound shader-table address");

  // The compact path refines Bump into one pure topological graph. This is
  // deliberately a structural test, not a host renderer: Cycles remains the
  // only numerical oracle, while the compiler invariants are checked here
  // without duplicating shader semantics on the CPU.
  std::vector<ParameterDesc> expanded_parameters;
  for (auto index = 0u; index < 3u; ++index) {
    expanded_parameters.emplace_back(make_parameter(index));
  }
  expanded_parameters.emplace_back(
      ParameterDesc{.id = ParameterId{3u},
                    .node = NodeId{103u},
                    .socket = "Normal",
                    .type = SocketType::normal,
                    .default_value = SocketValue::normal({0.0f, 0.0f, 1.0f}),
                    .source = ParameterSource::input});
  std::vector<ValueInstruction> expanded_values;
  for (auto index = 0u; index < 3u; ++index) {
    expanded_values.emplace_back(make_parameter_value(index));
  }
  expanded_values.emplace_back(
      ValueInstruction{.operation = ValueOperation::parameter,
                       .source_node = NodeId{103u},
                       .result_type = SocketType::normal,
                       .parameter = ParameterId{3u}});
  expanded_values.emplace_back(
      ValueInstruction{.operation = ValueOperation::surface_position,
                       .source_node = NodeId{104u},
                       .result_type = SocketType::point});
  expanded_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::vector_to_scalar,
      .source_node = NodeId{105u},
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::unary>(
          {{value_operand::unary::input, ValueExpressionId{4u}}})});
  expanded_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::bump,
      .source_node = NodeId{106u},
      .result_type = SocketType::normal,
      .operands = make_value_operands<value_operand::bump>(
          {{value_operand::bump::height, ValueExpressionId{5u}},
           {value_operand::bump::strength, ValueExpressionId{0u}},
           {value_operand::bump::distance, ValueExpressionId{1u}},
           {value_operand::bump::filter_width, ValueExpressionId{2u}},
           {value_operand::bump::normal, ValueExpressionId{3u}}}),
      .static_u0 = 2u});
  const SurfaceProgram expanded_source{
      22u,
      expanded_parameters,
      expanded_values,
      {ClosureInstruction{.operation = ClosureOperation::diffuse,
                          .source_node = NodeId{107u},
                          .color = ValueExpressionId{4u},
                          .normal = ValueExpressionId{6u},
                          .roughness = ValueExpressionId{0u}}},
      ClosureExpressionId{0u},
      {},
      {},
      ValueExpressionId{6u},
      ValueExpressionId{5u}};
  const auto expanded = expand_surface_bump_program(expanded_source);
  require(expanded.valid && expanded.program &&
              expanded.root_values.size() == expanded_values.size() &&
              expanded.bump_count == 1u &&
              expanded.sampled_instruction_count == 2u,
          "single-stream Bump refinement did not report its exact graph "
          "domain");
  const auto &refined_values = expanded.program->value_instructions();
  auto parameter_count = std::size_t{0u};
  auto zero_count = std::size_t{0u};
  auto sampled_position_count = std::size_t{0u};
  auto bump_sample_count = std::size_t{0u};
  for (auto index = std::size_t{0u}; index < refined_values.size(); ++index) {
    const auto &instruction = refined_values[index];
    parameter_count +=
        instruction.operation == ValueOperation::parameter ? 1u : 0u;
    zero_count +=
        instruction.operation == ValueOperation::bump_offset_zero ? 1u : 0u;
    sampled_position_count +=
        instruction.operation == ValueOperation::sampled_surface_position ? 1u
                                                                          : 0u;
    bump_sample_count +=
        instruction.operation == ValueOperation::bump_samples ? 1u : 0u;
    require(instruction.operation != ValueOperation::bump,
            "single-stream Bump refinement retained a recursive opcode");
    for (const auto operand : instruction.operands) {
      require(operand.valid() && operand.value < index,
              "single-stream Bump refinement violated strict "
              "topological order");
    }
  }
  require(parameter_count == expanded_parameters.size() && zero_count == 1u &&
              sampled_position_count == 2u && bump_sample_count == 1u,
          "Bump refinement duplicated invariant parameters or lost an exact "
          "differential sample");
  const auto &expanded_closure =
      expanded.program->closure_instructions().front();
  require(expanded_closure.color == expanded.root_values[4u] &&
              expanded_closure.normal == expanded.root_values[6u] &&
              expanded_closure.roughness == expanded.root_values[0u] &&
              expanded.program->surface_normal_root() ==
                  expanded.root_values[6u] &&
              expanded.program->displacement_root() == expanded.root_values[5u],
          "Bump refinement did not remap every public value endpoint");
  const auto refined_plan = plan_surface_value_storage(
      *expanded.program, std::vector<bool>(refined_values.size(), true), [&] {
        auto outputs = std::vector<bool>(refined_values.size(), false);
        outputs[expanded.root_values[6u].value] = true;
        return outputs;
      }());
  require(refined_plan.valid,
          "expanded Bump graph could not be colored as one typed stream: " +
              refined_plan.diagnostic);
  const auto refined_scene = build_surface_value_executable_scene(
      std::vector{SurfaceValueExecutionInput{.program = expanded.program.get(),
                                             .storage = &refined_plan}});
  const auto refined_bump_samples = std::count_if(
      refined_scene.values.instructions.begin(),
      refined_scene.values.instructions.end(), [](const auto &instruction) {
        return surface_value_operation(instruction) ==
               ValueOperation::bump_samples;
      });
  const auto refined_recursive_bumps = std::count_if(
      refined_scene.values.instructions.begin(),
      refined_scene.values.instructions.end(), [](const auto &instruction) {
        return surface_value_operation(instruction) == ValueOperation::bump;
      });
  require(refined_scene.valid && refined_scene.values.programs.size() == 1u &&
              refined_scene.values.instructions.size() ==
                  refined_plan.instructions.size() &&
              refined_scene.instruction_variants.size() ==
                  refined_scene.values.instructions.size() &&
              refined_bump_samples == 1u && refined_recursive_bumps == 0u,
          "expanded Bump graph did not lower directly to one executable "
          "stream");

  // Nest a position-dependent Bump inside another height expression. The
  // formal context law C+(w,0)/(0,w) requires an explicit scalar add for at
  // least one second-order sample; this guards against accidentally
  // overwriting the outer context with the inner one.
  auto compositional_values = expanded_values;
  compositional_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::vector_to_scalar,
      .source_node = NodeId{108u},
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::unary>(
          {{value_operand::unary::input, ValueExpressionId{6u}}})});
  compositional_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::add,
      .source_node = NodeId{109u},
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::binary>(
          {{value_operand::binary::a, ValueExpressionId{5u}},
           {value_operand::binary::b, ValueExpressionId{7u}}})});
  auto outer_bump = expanded_values[6u];
  outer_bump.source_node = NodeId{110u};
  outer_bump.operands[value_operand::bump::height] = ValueExpressionId{8u};
  compositional_values.emplace_back(std::move(outer_bump));
  const SurfaceProgram compositional_source{
      23u, expanded_parameters,  std::move(compositional_values), {}, {}, {},
      {},  ValueExpressionId{9u}};
  const auto compositional = expand_surface_bump_program(compositional_source);
  require(compositional.valid && compositional.program &&
              compositional.program->surface_normal_root() ==
                  compositional.root_values[9u],
          "nested Bump refinement failed to produce one graph");
  auto composed_offset_found = false;
  for (const auto &instruction : compositional.program->value_instructions()) {
    require(instruction.operation != ValueOperation::bump,
            "nested Bump refinement retained a recursive opcode");
    if (instruction.operation != ValueOperation::sampled_surface_position) {
      continue;
    }
    const auto dx = instruction.operand(value_operand::sampled_nullary::dx);
    const auto dy = instruction.operand(value_operand::sampled_nullary::dy);
    composed_offset_found |=
        compositional.program->value_instructions()[dx.value].operation ==
            ValueOperation::add ||
        compositional.program->value_instructions()[dy.value].operation ==
            ValueOperation::add;
  }
  require(composed_offset_found,
          "nested Bump refinement replaced rather than composed sample "
          "contexts");

  auto malformed_image = image;
  malformed_image.instructions.front().control |= 1u << 31u;
  const auto malformed_scene = build_surface_value_scene_image(
      std::vector{malformed_image, metadata_image});
  require(!malformed_scene.valid && malformed_scene.programs.empty() &&
              malformed_scene.instructions.empty() &&
              malformed_scene.diagnostic.find(
                  "without an immediate contract") != std::string::npos,
          "scene aggregation accepted or partially committed a malformed "
          "instruction stream");
  auto malformed_arity = image;
  malformed_arity.instructions.front().control ^=
      1u << surface_value_operand_count_shift;
  const auto malformed_arity_scene =
      build_surface_value_scene_image(std::vector{malformed_arity});
  require(!malformed_arity_scene.valid && malformed_arity_scene.diagnostic.find(
                                              "arity") != std::string::npos,
          "scene aggregation accepted an opcode/arity disagreement");

  std::vector<ValueInstruction> invalid_values;
  invalid_values.emplace_back(ValueInstruction{
      .operation = ValueOperation::add,
      .result_type = SocketType::floating,
      .operands = make_value_operands<value_operand::binary>(
          {{value_operand::binary::a, ValueExpressionId{1u}},
           {value_operand::binary::b, ValueExpressionId{1u}}})});
  invalid_values.emplace_back(make_parameter_value(0u));
  const SurfaceProgram invalid_program{
      2u, {make_parameter(0u)}, std::move(invalid_values), {}, {}};
  const auto invalid_plan =
      plan_surface_value_storage(invalid_program, std::vector<bool>(2u, true),
                                 std::vector<bool>{true, false});
  require(!invalid_plan.valid &&
              invalid_plan.diagnostic.find("topological") != std::string::npos,
          "storage planning accepted a forward value dependency");
  const auto invalid_expansion = expand_surface_bump_program(invalid_program);
  require(!invalid_expansion.valid && invalid_expansion.diagnostic.find(
                                          "topological") != std::string::npos,
          "Bump refinement accepted a forward value dependency");
}

} // namespace

int main() {
  try {
    psycles::test_support::test_surface_closure_metadata();
    test_surface_value_storage_plan();
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "Surface-program metadata test failure: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
