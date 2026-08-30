#pragma once

#include <psycles/compiler/cycles_svm_node_types.h>

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

namespace psycles::compiler::cycles_svm {

template<typename T>
concept SvmPayload = std::is_class_v<T> &&
                     std::is_standard_layout_v<T> &&
                     std::is_trivially_copyable_v<T> &&
                     alignof(T) <= alignof(std::uint32_t) &&
                     sizeof(T) % sizeof(std::uint32_t) == 0u;

// Exact host-side counterpart of Cycles 5.2.1 SVMCompiler::add_node and
// add_node_data. The stream is measured in uint32 words; no side payload or
// alternative instruction header exists.
class BytecodeBuilder {
private:
  std::vector<std::uint32_t> _words;
  std::array<bool, NODE_NUM> _node_types_used{};

public:
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::span<const std::uint32_t> words() const noexcept;
  [[nodiscard]] const std::array<bool, NODE_NUM> &node_types_used() const noexcept;

  void clear() noexcept;

  // Returns the word index of the opcode. This is the same index Cycles saves
  // when it later patches a forward closure jump.
  [[nodiscard]] std::size_t add_node(ShaderNodeType type);

  template<SvmPayload T>
  [[nodiscard]] std::size_t add_node(ShaderNodeType type, const T &payload) {
    const auto begin = add_node(type);
    add_node_data(payload);
    return begin;
  }

  template<SvmPayload T>
  void add_node_data(const T &payload) {
    const auto old_size = _words.size();
    constexpr auto word_count = sizeof(T) / sizeof(std::uint32_t);
    _words.resize(old_size + word_count);
    std::memcpy(_words.data() + old_size, &payload, sizeof(T));
  }

  void add_node_data_float(float value);
  void add_node_data_float4(float x, float y, float z, float w);

  // Cycles patches jump payload words after emitting the guarded sub-tree.
  // Out-of-range writes are programmer errors and terminate, matching the
  // unchecked indexed write in its compiler rather than changing the stream.
  void set_word(std::size_t index, std::uint32_t value) noexcept;
};

[[nodiscard]] SVMInputFloat input_float(float value) noexcept;
[[nodiscard]] SVMInputFloat input_float(SVMStackOffset offset) noexcept;
[[nodiscard]] SVMInputFloat3 input_float3(float x, float y, float z) noexcept;
[[nodiscard]] SVMInputFloat3 input_float3(SVMStackOffset offset) noexcept;
[[nodiscard]] SVMInputInt input_int(int value) noexcept;
[[nodiscard]] SVMInputInt input_int(SVMStackOffset offset, int fallback) noexcept;

} // namespace psycles::compiler::cycles_svm
