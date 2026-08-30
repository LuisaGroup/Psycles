#include <psycles/compiler/cycles_svm_bytecode.h>

#include <cstdlib>

namespace psycles::compiler::cycles_svm {

std::size_t BytecodeBuilder::size() const noexcept { return _words.size(); }

bool BytecodeBuilder::empty() const noexcept { return _words.empty(); }

std::span<const std::uint32_t> BytecodeBuilder::words() const noexcept {
  return _words;
}

const std::array<bool, NODE_NUM> &
BytecodeBuilder::node_types_used() const noexcept {
  return _node_types_used;
}

void BytecodeBuilder::clear() noexcept {
  _words.clear();
  _node_types_used.fill(false);
}

std::size_t BytecodeBuilder::add_node(ShaderNodeType type) {
  const auto index = static_cast<std::size_t>(type);
  if (index >= _node_types_used.size()) {
    std::abort();
  }
  const auto begin = _words.size();
  _words.emplace_back(static_cast<std::uint32_t>(type));
  _node_types_used[index] = true;
  return begin;
}

void BytecodeBuilder::add_node_data_float(float value) {
  _words.emplace_back(std::bit_cast<std::uint32_t>(value));
}

void BytecodeBuilder::add_node_data_float4(float x, float y, float z, float w) {
  add_node_data_float(x);
  add_node_data_float(y);
  add_node_data_float(z);
  add_node_data_float(w);
}

void BytecodeBuilder::set_word(std::size_t index,
                               std::uint32_t value) noexcept {
  if (index >= _words.size()) {
    std::abort();
  }
  _words[index] = value;
}

SVMInputFloat input_float(float value) noexcept {
  // Cycles filters non-finite unlinked defaults because their NaN payload can
  // collide with the stack-offset tag.
  if (!std::isfinite(value)) {
    value = 0.0f;
  }
  return SVMInputFloat{std::bit_cast<std::uint32_t>(value)};
}

SVMInputFloat input_float(SVMStackOffset offset) noexcept {
  return SVMInputFloat{SVM_INPUT_STACK_OFFSET_MASK |
                       static_cast<std::uint32_t>(offset)};
}

SVMInputFloat3 input_float3(float x, float y, float z) noexcept {
  if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z))) {
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
  }
  return SVMInputFloat3{input_float(x), input_float(y), input_float(z)};
}

SVMInputFloat3 input_float3(SVMStackOffset offset) noexcept {
  return SVMInputFloat3{input_float(offset), SVMInputFloat{0u},
                        SVMInputFloat{0u}};
}

SVMInputInt input_int(int value) noexcept {
  return SVMInputInt{value, SVM_STACK_INVALID, {0u, 0u, 0u}};
}

SVMInputInt input_int(SVMStackOffset offset, int fallback) noexcept {
  return SVMInputInt{fallback, offset, {0u, 0u, 0u}};
}

} // namespace psycles::compiler::cycles_svm
