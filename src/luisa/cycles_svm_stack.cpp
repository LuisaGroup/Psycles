/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <bit>

#include <luisa/ast/expression.h>
#include <luisa/ast/function_builder.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using psycles::compiler::cycles_svm::SVMStackOffset;

namespace {

[[nodiscard]] std::optional<std::uint32_t>
eval_const_uint_expr(const luisa::compute::Expression *raw) noexcept {
  using namespace luisa::compute;
  if (raw == nullptr) {
    return std::nullopt;
  }
  switch (raw->tag()) {
    case Expression::Tag::LITERAL: {
      const auto &val = static_cast<const LiteralExpr *>(raw)->value();
      if (const auto *u = luisa::get_if<luisa::uint>(&val)) {
        return static_cast<std::uint32_t>(*u);
      }
      if (const auto *i = luisa::get_if<int>(&val)) {
        return static_cast<std::uint32_t>(*i);
      }
      return std::nullopt;
    }
    case Expression::Tag::BINARY: {
      const auto *bin = static_cast<const BinaryExpr *>(raw);
      const auto lhs = eval_const_uint_expr(bin->lhs());
      const auto rhs = eval_const_uint_expr(bin->rhs());
      if (!lhs || !rhs) {
        return std::nullopt;
      }
      switch (bin->op()) {
        case BinaryOp::ADD: return *lhs + *rhs;
        case BinaryOp::SUB: return *lhs - *rhs;
        case BinaryOp::BIT_AND: return *lhs & *rhs;
        case BinaryOp::BIT_OR: return *lhs | *rhs;
        case BinaryOp::BIT_XOR: return *lhs ^ *rhs;
        case BinaryOp::SHL: return (*rhs < 32u) ? (*lhs << *rhs) : 0u;
        case BinaryOp::SHR: return (*rhs < 32u) ? (*lhs >> *rhs) : 0u;
        default: return std::nullopt;
      }
    }
    case Expression::Tag::CAST: {
      const auto *cast = static_cast<const CastExpr *>(raw);
      if (cast->op() == CastOp::STATIC) {
        return eval_const_uint_expr(cast->expression());
      }
      return std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

} // namespace

std::optional<std::uint32_t>
try_const_uint(luisa::compute::Expr<std::uint32_t> expr) noexcept {
  return eval_const_uint_expr(expr.expression());
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

std::optional<float>
try_const_float(luisa::compute::Expr<float> expr) noexcept {
  const auto *raw = expr.expression();
  if (raw != nullptr &&
      raw->tag() == luisa::compute::Expression::Tag::LITERAL) {
    const auto &val =
        static_cast<const luisa::compute::LiteralExpr *>(raw)->value();
    if (const auto *f = luisa::get_if<float>(&val)) {
      return *f;
    }
  }
  return std::nullopt;
}

luisa::compute::Expr<std::uint32_t>
stack_offset_add(luisa::compute::Expr<std::uint32_t> base,
                 std::uint32_t delta) noexcept {
  if (delta == 0u) {
    return base;
  }
  if (const auto cb = try_const_uint(base)) {
    return luisa::compute::Expr<std::uint32_t>{*cb + delta};
  }
  return base + delta;
}

luisa::compute::Expr<std::uint32_t>
stack_offset_add(luisa::compute::Expr<std::uint32_t> base,
                 luisa::compute::Expr<std::uint32_t> delta) noexcept {
  const auto cb = try_const_uint(base);
  const auto cd = try_const_uint(delta);
  if (cb && cd) {
    return luisa::compute::Expr<std::uint32_t>{*cb + *cd};
  }
  if (cd && *cd == 0u) {
    return base;
  }
  if (cb && *cb == 0u) {
    return delta;
  }
  return base + delta;
}

Stack::Stack(std::size_t size, bool direct_jit_mode, bool spectral) noexcept
    : _size{size}, _direct_jit{direct_jit_mode} {
  if (_direct_jit) {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    _owner_fb = fb;
    _root_scope = fb != nullptr ? fb->current_scope() : nullptr;
    _slots.resize(_size);
  } else {
    _array.emplace(size);
  }
  if (spectral) {
    _spectral_slots.emplace(std::array<SpectralSlot, 2u>{
        SpectralSlot{.offset = static_cast<std::uint32_t>(SVM_STACK_INVALID),
                     .spd = luisa::compute::make_float4(0.0f)},
        SpectralSlot{.offset = static_cast<std::uint32_t>(SVM_STACK_INVALID),
                     .spd = luisa::compute::make_float4(0.0f)}});
  }
}

Stack::Stack(std::size_t size, bool spectral) noexcept
    : Stack{size, false, spectral} {}

Stack Stack::direct_jit(std::size_t size, bool spectral) noexcept {
  return Stack{size, true, spectral};
}

void Stack::record_spectral_slot(
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<luisa::float4> spd) noexcept {
  using namespace luisa::compute;
  if (!_spectral_slots.has_value()) {
    return;
  }
  _spectral_slot_recorded = true;
  auto &slots = *_spectral_slots;
  const UInt target{offset};
  const Bool hit0 = slots[0u].offset == target;
  const Bool hit1 = !hit0 & (slots[1u].offset == target);
  const Bool miss = !(hit0 | hit1);
  const std::size_t victim = _next_spectral_slot ? 1u : 0u;
  _next_spectral_slot = !_next_spectral_slot;
  const Bool write0 = victim == 0u ? (hit0 | miss) : hit0;
  const Bool write1 = victim == 1u ? (hit1 | miss) : hit1;
  slots[0u].offset = select(slots[0u].offset, target, write0);
  slots[0u].spd = select(slots[0u].spd, spd, write0);
  slots[1u].offset = select(slots[1u].offset, target, write1);
  slots[1u].spd = select(slots[1u].spd, spd, write1);
}

void Stack::invalidate_spectral_slot(
    luisa::compute::Expr<std::uint32_t> offset) noexcept {
  using namespace luisa::compute;
  if (!_spectral_slots.has_value() ||
      (_direct_jit && !_spectral_slot_recorded)) {
    return;
  }
  const UInt off{offset};
  const UInt invalid = static_cast<std::uint32_t>(SVM_STACK_INVALID);
  for (auto &slot : *_spectral_slots) {
    const Bool overlap = (slot.offset != invalid) &
                         (slot.offset <= off) &
                         (off < slot.offset + 3u);
    slot.offset = select(slot.offset, invalid, overlap);
  }
}

luisa::compute::Bool Stack::try_load_spectral_slot(
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Float4 &out_spd) const noexcept {
  using namespace luisa::compute;
  if (!_spectral_slots.has_value() ||
      (_direct_jit && !_spectral_slot_recorded)) {
    return false;
  }
  const auto &slots = *_spectral_slots;
  const UInt target{offset};
  const Bool valid_target =
      target != static_cast<std::uint32_t>(SVM_STACK_INVALID);
  const Bool hit0 = valid_target & (slots[0u].offset == target);
  const Bool hit1 = valid_target & !hit0 & (slots[1u].offset == target);
  out_spd = select(out_spd, slots[1u].spd, hit1);
  out_spd = select(out_spd, slots[0u].spd, hit0);
  return hit0 | hit1;
}

bool Stack::at_root_scope() const noexcept {
  const auto *fb = luisa::compute::detail::FunctionBuilder::current();
  return fb != nullptr && fb == _owner_fb &&
         fb->current_scope() == _root_scope;
}

std::optional<float>
Stack::try_const_slot(std::uint32_t offset) const noexcept {
  if (!_direct_jit || offset >= _size) {
    return std::nullopt;
  }
  const auto &slot = _slots[offset];
  if (!slot.use_var && slot.literal) {
    return slot.literal;
  }
  return std::nullopt;
}

void Stack::materialize_all_to_vars() noexcept {
  using namespace luisa::compute;
  if (!_direct_jit) {
    return;
  }
  for (auto &slot : _slots) {
    if (slot.use_var) {
      continue;
    }
    if (slot.literal) {
      if (slot.var_expr == nullptr) {
        Float v = *slot.literal;
        slot.var_expr = v.expression();
      } else {
        dsl::assign(Float{slot.var_expr}, Expr<float>{*slot.literal});
      }
      slot.literal.reset();
      slot.use_var = true;
    } else if (slot.ssa_expr != nullptr) {
      if (slot.var_expr == nullptr) {
        Float v = Expr<float>{slot.ssa_expr};
        slot.var_expr = v.expression();
      } else {
        dsl::assign(Float{slot.var_expr}, Expr<float>{slot.ssa_expr});
      }
      slot.ssa_expr = nullptr;
      slot.use_var = true;
    }
  }
}

luisa::compute::Expr<float>
Stack::load_const_expr(std::uint32_t offset) noexcept {
  using namespace luisa::compute;
  if (!_direct_jit) {
    return Expr<float>{(*_array)[offset]};
  }
  if (offset >= _size) {
    return Expr<float>{0.0f};
  }
  const auto &slot = _slots[offset];
  if (slot.use_var && slot.var_expr != nullptr) {
    return Expr<float>{slot.var_expr};
  }
  if (slot.literal) {
    return Expr<float>{*slot.literal};
  }
  if (slot.ssa_expr != nullptr) {
    return Expr<float>{slot.ssa_expr};
  }
  return Expr<float>{0.0f};
}

luisa::compute::Float Stack::load_const(std::uint32_t offset) noexcept {
  if (!_direct_jit) {
    return (*_array)[offset];
  }
  return luisa::compute::Float{load_const_expr(offset)};
}

luisa::compute::Float
Stack::load_dynamic(luisa::compute::Expr<std::uint32_t> offset) noexcept {
  LUISA_ASSERT(!_direct_jit,
               "Dynamic SVM stack load is not supported in DirectJit mode.");
  return (*_array)[offset];
}

void Stack::store_const(std::uint32_t offset,
                        luisa::compute::Expr<float> value) noexcept {
  using namespace luisa::compute;
  if (_spectral_slots.has_value()) {
    invalidate_spectral_slot(offset);
  }
  if (!_direct_jit) {
    (*_array)[offset] = value;
    return;
  }
  if (offset >= _size) {
    return;
  }
  auto &slot = _slots[offset];
  if (at_root_scope()) {
    if (const auto lit = try_const_float(value)) {
      slot.literal = *lit;
      slot.ssa_expr = nullptr;
      slot.use_var = false;
      return;
    }
    Float ssa = value;
    slot.ssa_expr = ssa.expression();
    slot.literal.reset();
    slot.use_var = false;
    return;
  }
  if (!slot.use_var) {
    auto *fb = luisa::compute::detail::FunctionBuilder::current();
    LUISA_ASSERT(fb != nullptr && fb == _owner_fb && _root_scope != nullptr &&
                     !_root_scope->statements().empty(),
                 "DirectJit Stack nested write requires a valid enclosing root scope.");
    // Materialize the pre-branch slot value right before the enclosing
    // statement in _root_scope so any non-taken sibling branch preserves the
    // exact root-scope value (or 0.0f if previously unwritten) with zero
    // uninitialized local reads across all backends.
    const Expr<float> init_val =
        slot.literal ? Expr<float>{*slot.literal}
                     : (slot.ssa_expr != nullptr
                            ? Expr<float>{slot.ssa_expr}
                            : Expr<float>{0.0f});
    const auto *enclosing = _root_scope->pop();
    fb->with(_root_scope, [&]() noexcept {
      if (slot.var_expr == nullptr) {
        Float v = init_val;
        slot.var_expr = v.expression();
      } else {
        dsl::assign(Float{slot.var_expr}, init_val);
      }
    });
    _root_scope->append(enclosing);
    slot.literal.reset();
    slot.ssa_expr = nullptr;
    slot.use_var = true;
  }
  dsl::assign(Float{slot.var_expr}, value);
}

void Stack::store_dynamic(luisa::compute::Expr<std::uint32_t> offset,
                          luisa::compute::Expr<float> value) noexcept {
  LUISA_ASSERT(!_direct_jit,
               "Dynamic SVM stack store is not supported in DirectJit mode.");
  if (_spectral_slots.has_value()) {
    invalidate_spectral_slot(offset);
  }
  (*_array)[offset] = value;
}

Cursor::Cursor(
    luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> words,
    luisa::compute::UInt &offset) noexcept
    : _words{words}, _offset{&offset} {}

Cursor::Cursor(
    luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> words,
    std::span<const std::uint32_t> static_words,
    std::uint32_t global_base_offset,
    const luisa::compute::Constant<luisa::uint> *param_lut,
    const luisa::compute::UInt *param_base,
    const std::uint32_t *param_indices,
    std::optional<std::uint32_t> direct_image_ch_mode) noexcept
    : _words{words},
      _offset{nullptr},
      _static_words{static_words.data()},
      _static_count{static_words.size()},
      _static_index{0u},
      _global_base_offset{global_base_offset},
      _param_lut{param_lut},
      _param_base{param_base},
      _param_indices{param_indices},
      _direct_image_ch_mode{direct_image_ch_mode} {}

std::optional<std::uint32_t>
Cursor::static_word_at(std::uint32_t relative_word) const noexcept {
  if (_static_words == nullptr) {
    return std::nullopt;
  }
  const auto idx = _static_index + static_cast<std::size_t>(relative_word);
  if (idx >= _static_count) {
    return std::nullopt;
  }
  return _static_words[idx];
}

luisa::compute::Expr<std::uint32_t> Cursor::word() noexcept {
  if (_static_words != nullptr) {
    LUISA_ASSERT(_static_index < _static_count,
                 "Static SVM Cursor word() out of bounds ({} >= {}).",
                 _static_index, _static_count);
    const auto idx = _static_index++;
    if (_param_indices != nullptr && _param_lut != nullptr &&
        _param_base != nullptr && _param_indices[idx] != ~0u) {
      return (*_param_lut)[*_param_base + _param_indices[idx]];
    }
    return luisa::compute::Expr<std::uint32_t>{_static_words[idx]};
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
      if (_param_indices != nullptr && _param_lut != nullptr &&
          _param_base != nullptr && _param_indices[idx] != ~0u) {
        return (*_param_lut)[*_param_base + _param_indices[idx]];
      }
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
    const auto idx = _static_index++;
    if (_param_indices != nullptr && _param_lut != nullptr &&
        _param_base != nullptr && _param_indices[idx] != ~0u) {
      return (*_param_lut)[*_param_base + _param_indices[idx]].bitcast<float>();
    }
    return luisa::compute::Expr<float>{
        std::bit_cast<float>(_static_words[idx])};
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
      if (_param_indices != nullptr && _param_lut != nullptr &&
          _param_base != nullptr && _param_indices[idx] != ~0u) {
        return (*_param_lut)[*_param_base + _param_indices[idx]].bitcast<float>();
      }
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
    return stack.load_const(*c);
  }
  return stack.load_dynamic(offset);
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
    return stack.load_const(*c);
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
    return make_float3(stack.load_const_expr(*c),
                       stack.load_const_expr(*c + 1u),
                       stack.load_const_expr(*c + 2u));
  }
  return make_float3(stack.load_dynamic(offset),
                     stack.load_dynamic(offset + 1u),
                     stack.load_dynamic(offset + 2u));
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
    return make_float3(stack.load_const_expr(*c),
                       stack.load_const_expr(*c + 1u),
                       stack.load_const_expr(*c + 2u));
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
    return stack.load_const_expr(*c).bitcast<std::int32_t>();
  }
  return stack.load_dynamic(offset).bitcast<std::int32_t>();
}

luisa::compute::Float stack_load_input_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> bits) noexcept {
  using namespace luisa::compute;
  if (const auto b = try_const_uint(bits)) {
    if ((*b >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      return stack.load_const(*b & 0xffu);
    }
    return std::bit_cast<float>(*b);
  }
  if (stack.is_direct_jit()) {
    // Structural dedup parameterizes literal float words only; connected
    // stack inputs remain invariant compile-time constants caught above.
    return bits.bitcast<float>();
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
  if (cx && (*cx >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    const auto off = *cx & 0xffu;
    return make_float3(stack.load_const_expr(off),
                       stack.load_const_expr(off + 1u),
                       stack.load_const_expr(off + 2u));
  }
  if (cx && cy && cz) {
    return make_float3(std::bit_cast<float>(*cx),
                       std::bit_cast<float>(*cy),
                       std::bit_cast<float>(*cz));
  }
  const auto component_float = [](const std::optional<std::uint32_t> &c,
                                  Expr<std::uint32_t> raw) noexcept -> Float {
    return c ? Float{std::bit_cast<float>(*c)} : raw.bitcast<float>();
  };
  if (stack.is_direct_jit()) {
    return make_float3(component_float(cx, x_bits),
                       component_float(cy, y_bits),
                       component_float(cz, z_bits));
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
      return {.val = stack.load_const(off),
              .dx = stack.load_const(off + 1u),
              .dy = stack.load_const(off + 2u)};
    }
    return {.val = std::bit_cast<float>(*b), .dx = 0.0f, .dy = 0.0f};
  }
  if (stack.is_direct_jit()) {
    return {.val = bits.bitcast<float>(), .dx = 0.0f, .dy = 0.0f};
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
  if (cx && (*cx >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    const auto off = *cx & 0xffu;
    return {.val = make_float3(stack.load_const_expr(off),
                               stack.load_const_expr(off + 1u),
                               stack.load_const_expr(off + 2u)),
            .dx = make_float3(stack.load_const_expr(off + 3u),
                              stack.load_const_expr(off + 4u),
                              stack.load_const_expr(off + 5u)),
            .dy = make_float3(stack.load_const_expr(off + 6u),
                              stack.load_const_expr(off + 7u),
                              stack.load_const_expr(off + 8u))};
  }
  if (cx && cy && cz) {
    return {.val = make_float3(std::bit_cast<float>(*cx),
                               std::bit_cast<float>(*cy),
                               std::bit_cast<float>(*cz)),
            .dx = make_float3(0.0f),
            .dy = make_float3(0.0f)};
  }
  const auto component_float = [](const std::optional<std::uint32_t> &c,
                                  Expr<std::uint32_t> raw) noexcept -> Float {
    return c ? Float{std::bit_cast<float>(*c)} : raw.bitcast<float>();
  };
  if (stack.is_direct_jit()) {
    return {.val = make_float3(component_float(cx, x_bits),
                               component_float(cy, y_bits),
                               component_float(cz, z_bits)),
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
    stack.store_const(*c, value);
    return;
  }
  stack.store_dynamic(offset, value);
}

void stack_store_float_if_valid(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<float> value) noexcept {
  using namespace luisa::compute;
  if (const auto c = try_const_uint(offset)) {
    if (*c != static_cast<std::uint32_t>(
                  compiler::cycles_svm::SVM_STACK_INVALID)) {
      stack.store_const(*c, value);
    }
    return;
  }
  $if (offset != static_cast<std::uint32_t>(
                     compiler::cycles_svm::SVM_STACK_INVALID)) {
    stack_store_float(stack, offset, value);
  };
}

void stack_store_float3(Stack &stack,
                        luisa::compute::Expr<std::uint32_t> offset,
                        luisa::compute::Expr<luisa::float3> value) noexcept {
  using namespace luisa::compute;
  const auto *raw = value.expression();
  if (const auto c = try_const_uint(offset)) {
    if (raw != nullptr && raw->tag() == Expression::Tag::LITERAL) {
      const auto &lv = static_cast<const LiteralExpr *>(raw)->value();
      if (const auto *v3 = luisa::get_if<luisa::float3>(&lv)) {
        stack.store_const(*c, Expr<float>{v3->x});
        stack.store_const(*c + 1u, Expr<float>{v3->y});
        stack.store_const(*c + 2u, Expr<float>{v3->z});
        return;
      }
    }
    if (raw != nullptr && raw->tag() == Expression::Tag::REF) {
      stack.store_const(*c, value.x);
      stack.store_const(*c + 1u, value.y);
      stack.store_const(*c + 2u, value.z);
      return;
    }
    const Float3 v = value;
    stack.store_const(*c, v.x);
    stack.store_const(*c + 1u, v.y);
    stack.store_const(*c + 2u, v.z);
    return;
  }
  const Float3 v = value;
  stack.store_dynamic(offset, v.x);
  stack.store_dynamic(offset + 1u, v.y);
  stack.store_dynamic(offset + 2u, v.z);
}

void stack_store_float3_if_valid(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<luisa::float3> value) noexcept {
  using namespace luisa::compute;
  if (const auto c = try_const_uint(offset)) {
    if (*c != static_cast<std::uint32_t>(
                  compiler::cycles_svm::SVM_STACK_INVALID)) {
      stack_store_float3(stack, *c, value);
    }
    return;
  }
  $if (offset != static_cast<std::uint32_t>(
                     compiler::cycles_svm::SVM_STACK_INVALID)) {
    stack_store_float3(stack, offset, value);
  };
}

void stack_store_dual1(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       const Dual1 &value) noexcept {
  if (const auto c = try_const_uint(offset)) {
    stack.store_const(*c, value.val);
    stack.store_const(*c + 1u, value.dx);
    stack.store_const(*c + 2u, value.dy);
    return;
  }
  stack_store_float(stack, offset, value.val);
  stack_store_float(stack, offset + 1u, value.dx);
  stack_store_float(stack, offset + 2u, value.dy);
}

void stack_store_dual1_if_valid(Stack &stack,
                                luisa::compute::Expr<std::uint32_t> offset,
                                const Dual1 &value) noexcept {
  using namespace luisa::compute;
  if (const auto c = try_const_uint(offset)) {
    if (*c != static_cast<std::uint32_t>(
                  compiler::cycles_svm::SVM_STACK_INVALID)) {
      stack_store_dual1(stack, *c, value);
    }
    return;
  }
  $if (offset != static_cast<std::uint32_t>(
                     compiler::cycles_svm::SVM_STACK_INVALID)) {
    stack_store_dual1(stack, offset, value);
  };
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

void stack_store_dual3_if_valid(Stack &stack,
                                luisa::compute::Expr<std::uint32_t> offset,
                                const Dual3 &value) noexcept {
  using namespace luisa::compute;
  if (const auto c = try_const_uint(offset)) {
    if (*c != static_cast<std::uint32_t>(
                  compiler::cycles_svm::SVM_STACK_INVALID)) {
      stack_store_dual3(stack, *c, value);
    }
    return;
  }
  $if (offset != static_cast<std::uint32_t>(
                     compiler::cycles_svm::SVM_STACK_INVALID)) {
    stack_store_dual3(stack, offset, value);
  };
}

void stack_store_int(Stack &stack,
                     luisa::compute::Expr<std::uint32_t> offset,
                     luisa::compute::Expr<std::int32_t> value) noexcept {
  if (const auto c = try_const_uint(offset)) {
    stack.store_const(*c, value.bitcast<float>());
    return;
  }
  stack.store_dynamic(offset, value.bitcast<float>());
}

} // namespace psycles::luisa_backend::cycles_svm::detail
