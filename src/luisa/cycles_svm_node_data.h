/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

// Projection of Cycles svm_node_get<T>: capture the typed payload address
// and advance the PC once, without loading any fields. Accessors record
// reads at the caller's original control-flow location, including after
// caustic/allocation guards. In Direct-JIT mode (_static_base != nullptr),
// all payload words resolve to host compile-time literals with zero device
// word-buffer loads.
template <typename Payload>
class NodeDataView final {
  static_assert(std::is_standard_layout_v<Payload>);
  static_assert(sizeof(Payload) % sizeof(std::uint32_t) == 0u);
  luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> _words;
  luisa::compute::UInt _offset{0u};
  const std::uint32_t *_static_base{nullptr};
  const luisa::compute::Constant<luisa::uint> *_param_lut{nullptr};
  const luisa::compute::UInt *_param_base{nullptr};
  const std::uint32_t *_param_indices_base{nullptr};

 public:
  explicit NodeDataView(Cursor& cursor) noexcept
      : _words{cursor._words} {
    constexpr auto payload_words =
        static_cast<std::uint32_t>(sizeof(Payload) / sizeof(std::uint32_t));
    if (cursor.is_static()) {
      LUISA_ASSERT(cursor._static_index + payload_words <= cursor._static_count,
                   "Static SVM Cursor overflow in NodeDataView.");
      _static_base = cursor._static_words + cursor._static_index;
      if (cursor._param_indices != nullptr && cursor._param_lut != nullptr &&
          cursor._param_base != nullptr) {
        _param_lut = cursor._param_lut;
        _param_base = cursor._param_base;
        _param_indices_base = cursor._param_indices + cursor._static_index;
      }
      cursor._static_index += payload_words;
    } else {
      _offset = *cursor._offset;
      cursor.advance(payload_words);
    }
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::Expr<std::uint32_t> word() const noexcept {
    static_assert(ByteOffset % sizeof(std::uint32_t) == 0u);
    static_assert(ByteOffset + sizeof(std::uint32_t) <= sizeof(Payload));
    constexpr auto word_idx =
        static_cast<std::uint32_t>(ByteOffset / sizeof(std::uint32_t));
    if (_static_base != nullptr) {
      if (_param_indices_base != nullptr && _param_indices_base[word_idx] != ~0u) {
        return (*_param_lut)[*_param_base + _param_indices_base[word_idx]];
      }
      return luisa::compute::Expr<std::uint32_t>{_static_base[word_idx]};
    }
    return luisa::compute::Expr<std::uint32_t>{
        _words.read(_offset + word_idx)};
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::Expr<std::uint32_t>
  stack_offset() const noexcept {
    static_assert(ByteOffset < sizeof(Payload));
    constexpr auto word_idx =
        static_cast<std::uint32_t>((ByteOffset - ByteOffset % 4u) / 4u);
    constexpr auto shift = static_cast<std::uint32_t>(8u * (ByteOffset % 4u));
    if (_static_base != nullptr &&
        (_param_indices_base == nullptr || _param_indices_base[word_idx] == ~0u)) {
      return luisa::compute::Expr<std::uint32_t>{
          (_static_base[word_idx] >> shift) & 0xffu};
    }
    return (word<ByteOffset - ByteOffset % 4u>() >> shift) & 0xffu;
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::Float load_float(Stack& stack) const noexcept {
    return stack_load_input_float(stack, word<ByteOffset>());
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::Float3 load_float3(
      Stack& stack) const noexcept {
    using namespace luisa::compute;
    if (_static_base != nullptr) {
      return stack_load_input_float3(stack, word<ByteOffset>(),
                                     word<ByteOffset + 4u>(),
                                     word<ByteOffset + 8u>());
    }
    const auto x_bits = word<ByteOffset>();
    Float3 result = make_float3(0.0f);
    $if((x_bits >> 8u) ==
        (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      result = stack_load_float3(stack, x_bits & 0xffu);
    }
    $else {
      result = make_float3(
          x_bits.template bitcast<float>(),
          word<ByteOffset + 4u>().template bitcast<float>(),
          word<ByteOffset + 8u>().template bitcast<float>());
    };
    return result;
  }
};
}  // namespace psycles::luisa_backend::cycles_svm::detail
