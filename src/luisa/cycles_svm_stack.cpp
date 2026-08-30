/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

Cursor::Cursor(const luisa::compute::BufferUInt &words,
               luisa::compute::UInt &offset) noexcept
    : _words{words}, _offset{offset} {}

luisa::compute::UInt Cursor::word() noexcept {
  auto value = _words.read(_offset);
  _offset += 1u;
  return value;
}

luisa::compute::Float Cursor::floating() noexcept {
  return word().bitcast<float>();
}

void Cursor::advance(
    luisa::compute::Expr<std::uint32_t> word_count) noexcept {
  _offset += word_count;
}

luisa::compute::UInt Cursor::byte(
    luisa::compute::Expr<std::uint32_t> word_value,
    std::uint32_t byte_index) const noexcept {
  return (word_value >> (byte_index * 8u)) & 0xffu;
}

luisa::compute::Float stack_load_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept {
  return stack[offset];
}

luisa::compute::Float stack_load_float_default(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<float> value) noexcept {
  using namespace luisa::compute;
  Float result = value;
  $if (offset != static_cast<std::uint32_t>(
                     compiler::cycles_svm::SVM_STACK_INVALID)) {
    result = stack_load_float(stack, offset);
  };
  return result;
}

luisa::compute::Float3 stack_load_float3(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept {
  using namespace luisa::compute;
  return make_float3(stack[offset], stack[offset + 1u], stack[offset + 2u]);
}

luisa::compute::Float3 stack_load_float3_default(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<luisa::float3> value) noexcept {
  using namespace luisa::compute;
  Float3 result = value;
  $if (offset != static_cast<std::uint32_t>(
                     compiler::cycles_svm::SVM_STACK_INVALID)) {
    result = stack_load_float3(stack, offset);
  };
  return result;
}

luisa::compute::Int stack_load_int(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept {
  return stack[offset].bitcast<std::int32_t>();
}

luisa::compute::Float stack_load_input_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> bits) noexcept {
  using namespace luisa::compute;
  Float result = bits.bitcast<float>();
  $if ((bits >> 8u) ==
       (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    result = stack_load_float(stack, bits & 0xffu);
  };
  return result;
}

luisa::compute::Float3 stack_load_input_float3(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> x_bits,
    luisa::compute::Expr<std::uint32_t> y_bits,
    luisa::compute::Expr<std::uint32_t> z_bits) noexcept {
  using namespace luisa::compute;
  Float3 result = make_float3(x_bits.bitcast<float>(),
                              y_bits.bitcast<float>(),
                              z_bits.bitcast<float>());
  $if ((x_bits >> 8u) ==
       (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    result = stack_load_float3(stack, x_bits & 0xffu);
  };
  return result;
}

void stack_store_float(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       luisa::compute::Expr<float> value) noexcept {
  stack[offset] = value;
}

void stack_store_float3(Stack &stack,
                        luisa::compute::Expr<std::uint32_t> offset,
                        luisa::compute::Expr<luisa::float3> value) noexcept {
  stack[offset] = value.x;
  stack[offset + 1u] = value.y;
  stack[offset + 2u] = value.z;
}

void stack_store_dual3(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       const Dual3 &value) noexcept {
  stack_store_float3(stack, offset, value.val);
  stack_store_float3(stack, offset + 3u, value.dx);
  stack_store_float3(stack, offset + 6u, value.dy);
}

void stack_store_int(Stack &stack,
                     luisa::compute::Expr<std::uint32_t> offset,
                     luisa::compute::Expr<std::int32_t> value) noexcept {
  stack[offset] = value.bitcast<float>();
}

} // namespace psycles::luisa_backend::cycles_svm::detail
