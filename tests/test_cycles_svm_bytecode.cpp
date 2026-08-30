#include <psycles/compiler/cycles_svm_bytecode.h>

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>

namespace {

using namespace psycles::compiler::cycles_svm;

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void require_words(std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected,
                   const char *message) {
  require(actual.size() == expected.size(), message);
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    require(actual[index] == expected[index], message);
  }
}

void test_typed_stream() {
  BytecodeBuilder builder;
  const auto jump_at = builder.add_node(
      NODE_SHADER_JUMP,
      SVMNodeShaderJump{.offset_surface = 4,
                        .offset_volume = 17,
                        .offset_displacement = 29});
  require(jump_at == 0u, "shader jump must begin at word zero");

  const auto value_at = builder.add_node(
      NODE_VALUE_F,
      SVMNodeValueF{.value = 1.25f, .out_offset = 7u, ._pad = {0u, 0u, 0u}});
  require(value_at == 4u, "typed payload must advance the word PC");

  static constexpr std::uint32_t expected[] = {
      1u, 4u, 17u, 29u,
      17u, 0x3fa00000u, 7u,
  };
  require_words(builder.words(), expected,
                "typed Cycles SVM stream differs from the 5.2.1 oracle");
  require(builder.node_types_used()[NODE_SHADER_JUMP] &&
              builder.node_types_used()[NODE_VALUE_F] &&
              !builder.node_types_used()[NODE_CLOSURE_BSDF],
          "node usage domain must be the exact opcode domain");

  builder.set_word(1u, 8u);
  require(builder.words()[1u] == 8u,
          "forward jump patch must address a word in the same stream");
  builder.clear();
  require(builder.empty() && !builder.node_types_used()[NODE_VALUE_F],
          "clear must reset both Cycles compiler products");
}

void test_inputs() {
  require(input_float(2.0f).bits == 0x40000000u,
          "float immediate bit pattern differs from Cycles");
  require(input_float(SVMStackOffset{37u}).bits == 0x7fc00025u,
          "float stack tag differs from Cycles");

  const auto vector_immediate = input_float3(1.0f, -2.0f, 0.5f);
  require(vector_immediate.x.bits == 0x3f800000u &&
              vector_immediate.y.bits == 0xc0000000u &&
              vector_immediate.z.bits == 0x3f000000u,
          "float3 immediate bit pattern differs from Cycles");
  const auto vector_stack = input_float3(SVMStackOffset{14u});
  require(vector_stack.x.bits == 0x7fc0000eu &&
              vector_stack.y.bits == 0u && vector_stack.z.bits == 0u,
          "float3 stack tag differs from Cycles");

  require(input_float(std::numeric_limits<float>::infinity()).bits == 0u &&
              input_float(std::numeric_limits<float>::quiet_NaN()).bits == 0u,
          "non-finite scalar defaults must be canonical Cycles zero");
  const auto invalid_vector =
      input_float3(1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f);
  require(invalid_vector.x.bits == 0u && invalid_vector.y.bits == 0u &&
              invalid_vector.z.bits == 0u,
          "one non-finite vector component must zero the full Cycles default");

  const auto immediate_int = input_int(-17);
  require(immediate_int.value == -17 &&
              immediate_int.offset == SVM_STACK_INVALID,
          "integer immediate differs from Cycles");
  const auto linked_int = input_int(SVMStackOffset{9u}, 41);
  require(linked_int.value == 41 && linked_int.offset == 9u,
          "integer stack input differs from Cycles");
}

} // namespace

int main() {
  test_typed_stream();
  test_inputs();
  return 0;
}
