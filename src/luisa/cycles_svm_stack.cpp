/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <bit>

#include <luisa/ast/expression.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

std::optional<std::uint32_t>
try_const_uint(luisa::compute::Expr<std::uint32_t> expr) noexcept {
  const auto *raw = expr.expression();
  if (raw != nullptr &&
      raw->tag() == luisa::compute::Expression::Tag::LITERAL) {
    const auto &val =
        static_cast<const luisa::compute::LiteralExpr *>(raw)->value();
    if (const auto *u = luisa::get_if<luisa::uint>(&val)) {
      return static_cast<std::uint32_t>(*u);
    }
  }
  return std::nullopt;
}

std::optional<std::int32_t>
try_const_int(luisa::compute::Expr<std::int32_t> expr) noexcept {
  const auto *raw = expr.expression();
  if (raw != nullptr &&
      raw->tag() == luisa::compute::Expression::Tag::LITERAL) {
    const auto &val =
        static_cast<const luisa::compute::LiteralExpr *>(raw)->value();
    if (const auto *i = luisa::get_if<int>(&val)) {
      return static_cast<std::int32_t>(*i);
    }
  }
  return std::nullopt;
}

Cursor::Cursor(
    luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> words,
    luisa::compute::UInt &offset) noexcept
    : _words{words}, _offset{&offset} {}

Cursor::Cursor(
    luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> words,
    std::span<const std::uint32_t> static_words,
    std::uint32_t global_base_offset) noexcept
    : _words{words},
      _offset{nullptr},
      _static_words{static_words.data()},
      _static_count{static_words.size()},
      _static_index{0u},
      _global_base_offset{global_base_offset} {}

luisa::compute::Expr<std::uint32_t> Cursor::word() noexcept {
  if (_static_words != nullptr) {
    LUISA_ASSERT(_static_index < _static_count,
                 "Static SVM Cursor word() out of bounds ({} >= {}).",
                 _static_index, _static_count);
    return luisa::compute::Expr<std::uint32_t>{_static_words[_static_index++]};
  }
  auto value = _words.read(*_offset);
  *_offset += 1u;
  return luisa::compute::Expr<std::uint32_t>{value};
}

luisa::compute::Expr<std::uint32_t> Cursor::word_at(
    luisa::compute::Expr<std::uint32_t> relative_word) const noexcept {
  if (_static_words != nullptr) {
    if (const auto rel = try_const_uint(relative_word)) {
      const auto idx = _static_index + static_cast<std::size_t>(*rel);
      LUISA_ASSERT(idx < _static_count,
                   "Static SVM Cursor word_at({}) out of bounds ({} >= {}).",
                   *rel, idx, _static_count);
      return luisa::compute::Expr<std::uint32_t>{_static_words[idx]};
    }
    const auto base =
        _global_base_offset + static_cast<std::uint32_t>(_static_index);
    return luisa::compute::Expr<std::uint32_t>{
        _words.read(base + relative_word)};
  }
  return luisa::compute::Expr<std::uint32_t>{
      _words.read(*_offset + relative_word)};
}

luisa::compute::Expr<float> Cursor::floating() noexcept {
  if (_static_words != nullptr) {
    LUISA_ASSERT(_static_index < _static_count,
                 "Static SVM Cursor floating() out of bounds ({} >= {}).",
                 _static_index, _static_count);
    return luisa::compute::Expr<float>{
        std::bit_cast<float>(_static_words[_static_index++])};
  }
  return word().bitcast<float>();
}

luisa::compute::Expr<float> Cursor::floating_at(
    luisa::compute::Expr<std::uint32_t> relative_word) const noexcept {
  if (_static_words != nullptr) {
    if (const auto rel = try_const_uint(relative_word)) {
      const auto idx = _static_index + static_cast<std::size_t>(*rel);
      LUISA_ASSERT(
          idx < _static_count,
          "Static SVM Cursor floating_at({}) out of bounds ({} >= {}).", *rel,
          idx, _static_count);
      return luisa::compute::Expr<float>{
          std::bit_cast<float>(_static_words[idx])};
    }
    const auto base =
        _global_base_offset + static_cast<std::uint32_t>(_static_index);
    return _words.read(base + relative_word).bitcast<float>();
  }
  return _words.read(*_offset + relative_word).bitcast<float>();
}

void Cursor::advance(
    luisa::compute::Expr<std::uint32_t> word_count) noexcept {
  if (_static_words != nullptr) {
    if (const auto count = try_const_uint(word_count)) {
      _static_index += static_cast<std::size_t>(*count);
      LUISA_ASSERT(_static_index <= _static_count,
                   "Static SVM Cursor advance({}) out of bounds ({} > {}).",
                   *count, _static_index, _static_count);
      return;
    }
    // Dynamic table advance at the end of a ramp/curve instruction. Each IR
    // instruction owns an isolated Cursor, so subsequent instructions do not
    // depend on advancing past a runtime-sized tail.
    return;
  }
  *_offset += word_count;
}

luisa::compute::Expr<std::uint32_t> Cursor::byte(
    luisa::compute::Expr<std::uint32_t> word_value,
    std::uint32_t byte_index) const noexcept {
  if (const auto w = try_const_uint(word_value)) {
    return luisa::compute::Expr<std::uint32_t>{(*w >> (byte_index * 8u)) &
                                               0xffu};
  }
  return (word_value >> (byte_index * 8u)) & 0xffu;
}

luisa::compute::Float stack_load_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept {
  if (const auto c = try_const_uint(offset)) {
    return stack[*c];
  }
  return stack[offset];
}

luisa::compute::Float stack_load_float_default(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<float> value) noexcept {
  using namespace luisa::compute;
  if (const auto c = try_const_uint(offset)) {
    if (*c == static_cast<std::uint32_t>(
                  compiler::cycles_svm::SVM_STACK_INVALID)) {
      return value;
    }
    return stack[*c];
  }
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
  if (const auto c = try_const_uint(offset)) {
    return make_float3(stack[*c], stack[*c + 1u], stack[*c + 2u]);
  }
  return make_float3(stack[offset], stack[offset + 1u], stack[offset + 2u]);
}

luisa::compute::Float3 stack_load_float3_default(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<luisa::float3> value) noexcept {
  using namespace luisa::compute;
  if (const auto c = try_const_uint(offset)) {
    if (*c == static_cast<std::uint32_t>(
                  compiler::cycles_svm::SVM_STACK_INVALID)) {
      return value;
    }
    return make_float3(stack[*c], stack[*c + 1u], stack[*c + 2u]);
  }
  Float3 result = value;
  $if (offset != static_cast<std::uint32_t>(
                     compiler::cycles_svm::SVM_STACK_INVALID)) {
    result = stack_load_float3(stack, offset);
  };
  return result;
}

luisa::compute::Int stack_load_int(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept {
  if (const auto c = try_const_uint(offset)) {
    return stack[*c].bitcast<std::int32_t>();
  }
  return stack[offset].bitcast<std::int32_t>();
}

luisa::compute::Float stack_load_input_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> bits) noexcept {
  using namespace luisa::compute;
  if (const auto b = try_const_uint(bits)) {
    if ((*b >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      return stack[*b & 0xffu];
    }
    return std::bit_cast<float>(*b);
  }
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
  const auto cx = try_const_uint(x_bits);
  const auto cy = try_const_uint(y_bits);
  const auto cz = try_const_uint(z_bits);
  if (cx && cy && cz) {
    if ((*cx >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      const auto off = *cx & 0xffu;
      return make_float3(stack[off], stack[off + 1u], stack[off + 2u]);
    }
    return make_float3(std::bit_cast<float>(*cx),
                       std::bit_cast<float>(*cy),
                       std::bit_cast<float>(*cz));
  }
  Float3 result = make_float3(x_bits.bitcast<float>(),
                              y_bits.bitcast<float>(),
                              z_bits.bitcast<float>());
  $if ((x_bits >> 8u) ==
       (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    result = stack_load_float3(stack, x_bits & 0xffu);
  };
  return result;
}

Dual1 stack_load_input_dual_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> bits) noexcept {
  using namespace luisa::compute;
  if (const auto b = try_const_uint(bits)) {
    if ((*b >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      const auto off = *b & 0xffu;
      return {.val = stack[off],
              .dx = stack[off + 1u],
              .dy = stack[off + 2u]};
    }
    return {.val = std::bit_cast<float>(*b), .dx = 0.0f, .dy = 0.0f};
  }
  Dual1 result{.val = bits.bitcast<float>(), .dx = 0.0f, .dy = 0.0f};
  $if ((bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    const UInt offset = bits & 0xffu;
    result.val = stack_load_float(stack, offset);
    result.dx = stack_load_float(stack, offset + 1u);
    result.dy = stack_load_float(stack, offset + 2u);
  };
  return result;
}

Dual3 stack_load_input_dual_float3(
    Stack &stack, luisa::compute::Expr<std::uint32_t> x_bits,
    luisa::compute::Expr<std::uint32_t> y_bits,
    luisa::compute::Expr<std::uint32_t> z_bits) noexcept {
  using namespace luisa::compute;
  const auto cx = try_const_uint(x_bits);
  const auto cy = try_const_uint(y_bits);
  const auto cz = try_const_uint(z_bits);
  if (cx && cy && cz) {
    if ((*cx >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      const auto off = *cx & 0xffu;
      return {.val = make_float3(stack[off], stack[off + 1u], stack[off + 2u]),
              .dx = make_float3(stack[off + 3u], stack[off + 4u],
                                stack[off + 5u]),
              .dy = make_float3(stack[off + 6u], stack[off + 7u],
                                stack[off + 8u])};
    }
    return {.val = make_float3(std::bit_cast<float>(*cx),
                               std::bit_cast<float>(*cy),
                               std::bit_cast<float>(*cz)),
            .dx = make_float3(0.0f),
            .dy = make_float3(0.0f)};
  }
  Dual3 result{.val = make_float3(x_bits.bitcast<float>(),
                                  y_bits.bitcast<float>(),
                                  z_bits.bitcast<float>()),
               .dx = make_float3(0.0f),
               .dy = make_float3(0.0f)};
  $if ((x_bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    const UInt offset = x_bits & 0xffu;
    result.val = stack_load_float3(stack, offset);
    result.dx = stack_load_float3(stack, offset + 3u);
    result.dy = stack_load_float3(stack, offset + 6u);
  };
  return result;
}

void stack_store_float(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       luisa::compute::Expr<float> value) noexcept {
  if (const auto c = try_const_uint(offset)) {
    stack[*c] = value;
    return;
  }
  stack[offset] = value;
}

void stack_store_float3(Stack &stack,
                        luisa::compute::Expr<std::uint32_t> offset,
                        luisa::compute::Expr<luisa::float3> value) noexcept {
  if (const auto c = try_const_uint(offset)) {
    stack[*c] = value.x;
    stack[*c + 1u] = value.y;
    stack[*c + 2u] = value.z;
    return;
  }
  stack[offset] = value.x;
  stack[offset + 1u] = value.y;
  stack[offset + 2u] = value.z;
}

void stack_store_dual1(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       const Dual1 &value) noexcept {
  if (const auto c = try_const_uint(offset)) {
    stack[*c] = value.val;
    stack[*c + 1u] = value.dx;
    stack[*c + 2u] = value.dy;
    return;
  }
  stack_store_float(stack, offset, value.val);
  stack_store_float(stack, offset + 1u, value.dx);
  stack_store_float(stack, offset + 2u, value.dy);
}

void stack_store_dual3(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       const Dual3 &value) noexcept {
  if (const auto c = try_const_uint(offset)) {
    stack_store_float3(stack, *c, value.val);
    stack_store_float3(stack, *c + 3u, value.dx);
    stack_store_float3(stack, *c + 6u, value.dy);
    return;
  }
  stack_store_float3(stack, offset, value.val);
  stack_store_float3(stack, offset + 3u, value.dx);
  stack_store_float3(stack, offset + 6u, value.dy);
}

void stack_store_int(Stack &stack,
                     luisa::compute::Expr<std::uint32_t> offset,
                     luisa::compute::Expr<std::int32_t> value) noexcept {
  if (const auto c = try_const_uint(offset)) {
    stack[*c] = value.bitcast<float>();
    return;
  }
  stack[offset] = value.bitcast<float>();
}

} // namespace psycles::luisa_backend::cycles_svm::detail
