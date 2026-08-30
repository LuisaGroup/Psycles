/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using Stack = luisa::compute::ArrayFloat<SVM_STACK_SIZE>;

class Cursor final {
private:
  const luisa::compute::BufferUInt &_words;
  luisa::compute::UInt &_offset;

public:
  Cursor(const luisa::compute::BufferUInt &words,
         luisa::compute::UInt &offset) noexcept;

  [[nodiscard]] luisa::compute::UInt word() noexcept;
  [[nodiscard]] luisa::compute::Float floating() noexcept;
  void advance(luisa::compute::Expr<std::uint32_t> word_count) noexcept;
  [[nodiscard]] luisa::compute::UInt byte(
      luisa::compute::Expr<std::uint32_t> word,
      std::uint32_t byte_index) const noexcept;
};

[[nodiscard]] luisa::compute::Float stack_load_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float stack_load_float_default(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<float> value) noexcept;
[[nodiscard]] luisa::compute::Float3 stack_load_float3(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float stack_load_input_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> bits) noexcept;
[[nodiscard]] luisa::compute::Float3 stack_load_input_float3(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> x_bits,
    luisa::compute::Expr<std::uint32_t> y_bits,
    luisa::compute::Expr<std::uint32_t> z_bits) noexcept;

void stack_store_float(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       luisa::compute::Expr<float> value) noexcept;
void stack_store_float3(Stack &stack,
                        luisa::compute::Expr<std::uint32_t> offset,
                        luisa::compute::Expr<luisa::float3> value) noexcept;

[[nodiscard]] luisa::compute::Float svm_math(
    luisa::compute::Expr<std::uint32_t> type,
    luisa::compute::Expr<float> a,
    luisa::compute::Expr<float> b,
    luisa::compute::Expr<float> c) noexcept;

void node_value_f(Cursor &cursor, Stack &stack) noexcept;
void node_value_v(Cursor &cursor, Stack &stack) noexcept;
void node_geometry(Cursor &cursor, Stack &stack, ShaderData &shader_data,
                   luisa::compute::Bool &supported) noexcept;
void node_light_path(Cursor &cursor, Stack &stack,
                     const ShaderData &shader_data,
                     const PathState &path_state,
                     std::uint32_t node_feature_mask) noexcept;
void node_math(Cursor &cursor, Stack &stack) noexcept;
void node_hsv(Cursor &cursor, Stack &stack) noexcept;
void node_gamma(Cursor &cursor, Stack &stack) noexcept;
void node_brightness(Cursor &cursor, Stack &stack) noexcept;
void node_invert(Cursor &cursor, Stack &stack) noexcept;
void node_mix(Cursor &cursor, Stack &stack) noexcept;
void node_separate_color(Cursor &cursor, Stack &stack) noexcept;
void node_combine_color(Cursor &cursor, Stack &stack) noexcept;
void node_clamp(Cursor &cursor, Stack &stack) noexcept;

void node_closure_set_weight(Cursor &cursor,
                             luisa::compute::Float3 &closure_weight) noexcept;
void node_closure_weight(Cursor &cursor, Stack &stack,
                         luisa::compute::Float3 &closure_weight) noexcept;
void node_emission_weight(Cursor &cursor, Stack &stack,
                          luisa::compute::Float3 &closure_weight) noexcept;
void node_mix_closure(Cursor &cursor, Stack &stack) noexcept;
void node_closure_emission(Cursor &cursor, Stack &stack,
                           luisa::compute::Expr<luisa::float3> closure_weight,
                           ShaderData &shader_data,
                           luisa::compute::Bool &supported) noexcept;
void node_closure_bsdf_skip(Cursor &cursor,
                            luisa::compute::Expr<std::uint32_t> closure_type)
    noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
