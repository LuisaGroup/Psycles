/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Float4 rgb_ramp_lookup_clamped(
    Cursor &cursor, Expr<float> factor, Expr<std::uint32_t> interpolate_word,
    Expr<std::uint32_t> table_size) noexcept {
  if (const auto ci = try_const_uint(interpolate_word)) {
    if (*ci == 0u) {
      static auto nearest_callable = []() noexcept {
        Callable result{[](BufferVar<luisa::uint> words, UInt base_offset,
                           Float factor, UInt table_size) noexcept {
          const UInt last = table_size - 1u;
          const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
          const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
          const UInt idx = base_offset + cast<uint>(index) * 4u;
          return make_float4(words.read(idx).bitcast<float>(),
                             words.read(idx + 1u).bitcast<float>(),
                             words.read(idx + 2u).bitcast<float>(),
                             words.read(idx + 3u).bitcast<float>());
        }};
        result.set_name("cycles_svm_rgb_ramp_nearest");
        return result;
      }();
      return nearest_callable(cursor.words_buffer(),
                              cursor.current_word_offset(), factor, table_size);
    }
    static auto linear_callable = []() noexcept {
      Callable result{[](BufferVar<luisa::uint> words, UInt base_offset,
                         Float factor, UInt table_size) noexcept {
        auto read_rgba = [&](Expr<std::uint32_t> element) noexcept {
          const UInt idx = base_offset + element * 4u;
          return make_float4(words.read(idx).bitcast<float>(),
                             words.read(idx + 1u).bitcast<float>(),
                             words.read(idx + 2u).bitcast<float>(),
                             words.read(idx + 3u).bitcast<float>());
        };
        const UInt last = table_size - 1u;
        const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
        const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
        const UInt element = cast<uint>(index);
        const Float t = scaled - cast<float>(index);
        const Float4 c0 = read_rgba(element);
        const Float4 c1 = read_rgba(min(element + 1u, last));
        return (1.0f - t) * c0 + t * c1;
      }};
      result.set_name("cycles_svm_rgb_ramp_linear");
      return result;
    }();
    return linear_callable(cursor.words_buffer(), cursor.current_word_offset(),
                           factor, table_size);
  }
  static auto callable = []() noexcept {
    Callable result{
        [](BufferVar<luisa::uint> words, UInt base_offset, Float factor,
           Bool interpolate, UInt table_size) noexcept {
          auto read_rgba = [&](Expr<std::uint32_t> element) noexcept {
            const UInt idx = base_offset + element * 4u;
            return make_float4(words.read(idx).bitcast<float>(),
                               words.read(idx + 1u).bitcast<float>(),
                               words.read(idx + 2u).bitcast<float>(),
                               words.read(idx + 3u).bitcast<float>());
          };
          const UInt last = table_size - 1u;
          const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
          const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
          const UInt element = cast<uint>(index);
          const Float t = select(0.0f, scaled - cast<float>(index), interpolate);
          const Float4 c0 = read_rgba(element);
          const Float4 c1 = read_rgba(min(element + 1u, last));
          return (1.0f - t) * c0 + t * c1;
        }};
    result.set_name("cycles_svm_rgb_ramp_lookup");
    return result;
  }();
  return callable(cursor.words_buffer(), cursor.current_word_offset(), factor,
                  interpolate_word != 0u, table_size);
}

[[nodiscard]] Float curve_channel_lookup_clamped(
    Cursor &cursor, Expr<float> factor, Expr<std::uint32_t> channel,
    Expr<std::uint32_t> table_size) noexcept {
  static auto callable = []() noexcept {
    Callable result{
        [](BufferVar<luisa::uint> words, UInt base_offset, Float factor,
           UInt channel, UInt table_size) noexcept {
          const UInt last = table_size - 1u;
          const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
          const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
          const UInt element = cast<uint>(index);
          const Float t = scaled - cast<float>(index);
          const Float c0 =
              words.read(base_offset + element * 4u + channel).bitcast<float>();
          const Float c1 =
              words.read(base_offset + min(element + 1u, last) * 4u + channel)
                  .bitcast<float>();
          return (1.0f - t) * c0 + t * c1;
        }};
    result.set_name("cycles_svm_curve_channel_clamped");
    return result;
  }();
  return callable(cursor.words_buffer(), cursor.current_word_offset(), factor,
                  channel, table_size);
}

[[nodiscard]] Float curve_channel_lookup(
    Cursor &cursor, Expr<float> factor, Expr<std::uint32_t> channel,
    Expr<bool> extrapolate, bool may_extrapolate,
    Expr<std::uint32_t> table_size) noexcept {
  if (!may_extrapolate) {
    return curve_channel_lookup_clamped(cursor, factor, channel, table_size);
  }
  static auto callable = []() noexcept {
    Callable result{
        [](BufferVar<luisa::uint> words, UInt base_offset, Float factor,
           UInt channel, Bool extrapolate, UInt table_size) noexcept {
          const UInt last = table_size - 1u;
          Float out = 0.0f;
          $if(((factor < 0.0f) | (factor > 1.0f)) & extrapolate) {
            const Bool low = factor < 0.0f;
            const UInt i0 = select(last, 0u, low);
            const UInt i1 = select(last - 1u, 1u, low);
            const Float value =
                words.read(base_offset + i0 * 4u + channel).bitcast<float>();
            const Float delta =
                value -
                words.read(base_offset + i1 * 4u + channel).bitcast<float>();
            const Float distance = select(factor - 1.0f, -factor, low);
            out = value + delta * distance * cast<float>(last);
          }
          $else {
            const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
            const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
            const UInt element = cast<uint>(index);
            const Float t = scaled - cast<float>(index);
            const Float c0 = words.read(base_offset + element * 4u + channel)
                                 .bitcast<float>();
            const Float c1 =
                words.read(base_offset + min(element + 1u, last) * 4u + channel)
                    .bitcast<float>();
            out = (1.0f - t) * c0 + t * c1;
          };
          return out;
        }};
    result.set_name("cycles_svm_curve_channel_extrap");
    return result;
  }();
  return callable(cursor.words_buffer(), cursor.current_word_offset(), factor,
                  channel, extrapolate, table_size);
}

[[nodiscard]] Float
float_ramp_lookup_extrapolated(Cursor &cursor, Expr<float> factor,
                               Expr<bool> interpolate, Expr<bool> extrapolate,
                               bool may_extrapolate,
                               Expr<std::uint32_t> table_size) noexcept {
  if (!may_extrapolate) {
    static auto clamped_callable = []() noexcept {
      Callable result{[](BufferVar<luisa::uint> words, UInt base_offset,
                         Float factor, Bool interpolate,
                         UInt table_size) noexcept {
        const UInt last = table_size - 1u;
        const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
        const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
        const UInt element = cast<uint>(index);
        const Float t = select(0.0f, scaled - cast<float>(index), interpolate);
        const Float c0 = words.read(base_offset + element).bitcast<float>();
        const Float c1 =
            words.read(base_offset + min(element + 1u, last)).bitcast<float>();
        return (1.0f - t) * c0 + t * c1;
      }};
      result.set_name("cycles_svm_float_ramp_clamped");
      return result;
    }();
    return clamped_callable(cursor.words_buffer(), cursor.current_word_offset(),
                            factor, interpolate, table_size);
  }
  static auto callable = []() noexcept {
    Callable result{
        [](BufferVar<luisa::uint> words, UInt base_offset, Float factor,
           Bool interpolate, Bool extrapolate, UInt table_size) noexcept {
          const UInt last = table_size - 1u;
          Float out = 0.0f;
          $if(((factor < 0.0f) | (factor > 1.0f)) & extrapolate) {
            const Bool low = factor < 0.0f;
            const UInt i0 = select(last, 0u, low);
            const UInt i1 = select(last - 1u, 1u, low);
            const Float value = words.read(base_offset + i0).bitcast<float>();
            const Float delta =
                value - words.read(base_offset + i1).bitcast<float>();
            const Float distance = select(factor - 1.0f, -factor, low);
            out = value + delta * distance * cast<float>(last);
          }
          $else {
            const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
            const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
            const UInt element = cast<uint>(index);
            const Float t =
                select(0.0f, scaled - cast<float>(index), interpolate);
            const Float c0 = words.read(base_offset + element).bitcast<float>();
            const Float c1 =
                words.read(base_offset + min(element + 1u, last))
                    .bitcast<float>();
            out = (1.0f - t) * c0 + t * c1;
          };
          return out;
        }};
    result.set_name("cycles_svm_float_ramp_extrap");
    return result;
  }();
  return callable(cursor.words_buffer(), cursor.current_word_offset(), factor,
                  interpolate, extrapolate, table_size);
}

} // namespace

void node_rgb_ramp(Cursor &cursor, Stack &stack) noexcept {
  const auto table_size = cursor.word();
  const Float factor = stack_load_input_float(stack, cursor.word());
  const auto packed = cursor.word();
  const auto interpolate = cursor.byte(packed, 0u);
  const auto color_offset = cursor.byte(packed, 1u);
  const auto alpha_offset = cursor.byte(packed, 2u);

  const Float4 color = rgb_ramp_lookup_clamped(
      cursor, factor, interpolate, table_size);

  stack_store_float3_if_valid(stack, color_offset, color.xyz());
  stack_store_float_if_valid(stack, alpha_offset, color.w);
  cursor.advance(table_size * 4u);
}

void node_curves(Cursor &cursor, Stack &stack, bool may_extrapolate) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const Float3 color =
      stack_load_input_float3(stack, color_x, color_y, color_z);
  const Float factor = stack_load_input_float(stack, cursor.word());
  const auto min_x = cursor.floating();
  const auto max_x = cursor.floating();
  const auto table_size = cursor.word();
  const auto packed = cursor.word();
  const auto extrapolate_word = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 1u);
  bool effective_extrapolate = may_extrapolate;
  if (const auto ce = try_const_uint(extrapolate_word)) {
    effective_extrapolate = may_extrapolate && (*ce != 0u);
  }
  const auto extrapolate = extrapolate_word != 0u;

  const Float range_x = max_x - min_x;
  const Float3 relative = (color - min_x) / range_x;
  const Float3 mapped = make_float3(
      curve_channel_lookup(cursor, relative.x, 0u, extrapolate,
                           effective_extrapolate, table_size),
      curve_channel_lookup(cursor, relative.y, 1u, extrapolate,
                           effective_extrapolate, table_size),
      curve_channel_lookup(cursor, relative.z, 2u, extrapolate,
                           effective_extrapolate, table_size));
  stack_store_float3(stack, output_offset,
                     (1.0f - factor) * color + factor * mapped);
  cursor.advance(table_size * 4u);
}

void node_float_curve(Cursor &cursor, Stack &stack) noexcept {
  const Float factor = stack_load_input_float(stack, cursor.word());
  const Float input = stack_load_input_float(stack, cursor.word());
  const auto min_x = cursor.floating();
  const auto max_x = cursor.floating();
  const auto table_size = cursor.word();
  const auto packed = cursor.word();
  const auto extrapolate_word = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 1u);
  bool effective_extrapolate = true;
  if (const auto ce = try_const_uint(extrapolate_word)) {
    effective_extrapolate = (*ce != 0u);
  }
  const auto extrapolate = extrapolate_word != 0u;

  const Float relative = (input - min_x) / (max_x - min_x);
  const Float value = float_ramp_lookup_extrapolated(
      cursor, relative, true, extrapolate, effective_extrapolate, table_size);
  stack_store_float(stack, output_offset,
                    (1.0f - factor) * input + factor * value);
  cursor.advance(table_size);
}

} // namespace psycles::luisa_backend::cycles_svm::detail
