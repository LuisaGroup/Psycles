/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstddef>
#include <type_traits>

#include "cycles_svm_internal.h"

namespace psycles::luisa_backend::cycles_svm::detail {

// Projection of Cycles svm_node_get<T>: capture the typed payload address
// and advance the PC once, without loading any fields. Accessors record
// reads at the caller's original control-flow location, including after
// caustic/allocation guards. This adds no device-side object or dispatch.
template <typename Payload>
class NodeDataView final {
  static_assert(std::is_standard_layout_v<Payload>);
  static_assert(sizeof(Payload) % sizeof(std::uint32_t) == 0u);
  luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> _words;
  luisa::compute::UInt _offset;

 public:
  explicit NodeDataView(Cursor& cursor) noexcept
      : _words{cursor._words}, _offset{cursor._offset} {
    cursor.advance(
        static_cast<std::uint32_t>(sizeof(Payload) / sizeof(std::uint32_t)));
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::UInt word() const noexcept {
    static_assert(ByteOffset % sizeof(std::uint32_t) == 0u);
    static_assert(ByteOffset + sizeof(std::uint32_t) <= sizeof(Payload));
    return _words.read(_offset + static_cast<std::uint32_t>(
                                     ByteOffset / sizeof(std::uint32_t)));
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::UInt stack_offset() const noexcept {
    static_assert(ByteOffset < sizeof(Payload));
    return (word<ByteOffset - ByteOffset % 4u>() >> (8u * (ByteOffset % 4u))) &
           0xffu;
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::Float load_float(Stack& stack) const noexcept {
    return stack_load_input_float(stack, word<ByteOffset>());
  }

  template <std::size_t ByteOffset>
  [[nodiscard]] luisa::compute::Float3 load_float3(
      Stack& stack) const noexcept {
    // Cycles tests the first lane before loading the remaining literal lanes.
    // Besides matching the SVM input contract, keeping the stack path inside
    // this branch avoids three unnecessary word-buffer reads for dynamic
    // stack-backed vector inputs.
    using namespace luisa::compute;
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
