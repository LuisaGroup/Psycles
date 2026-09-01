/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/luisa/cycles_svm.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using Stack = luisa::compute::ArrayFloat<SVM_STACK_SIZE>;

struct Differential3 {
  luisa::compute::Float3 dx;
  luisa::compute::Float3 dy;
};

[[nodiscard]] Differential3
differential_from_compact(luisa::compute::Expr<luisa::float3> direction,
                          luisa::compute::Expr<float> differential) noexcept;

class Cursor final {
private:
  const luisa::compute::BufferUInt &_words;
  luisa::compute::UInt &_offset;

public:
  Cursor(const luisa::compute::BufferUInt &words,
         luisa::compute::UInt &offset) noexcept;

  [[nodiscard]] luisa::compute::UInt word() noexcept;
  [[nodiscard]] luisa::compute::Float floating() noexcept;
  [[nodiscard]] luisa::compute::Float floating_at(
      luisa::compute::Expr<std::uint32_t> relative_word) const noexcept;
  void advance(luisa::compute::Expr<std::uint32_t> word_count) noexcept;
  [[nodiscard]] luisa::compute::UInt byte(
      luisa::compute::Expr<std::uint32_t> word,
      std::uint32_t byte_index) const noexcept;
};

[[nodiscard]] luisa::compute::Float4x4 transform_from_rows(
    luisa::compute::Expr<luisa::float4> x,
    luisa::compute::Expr<luisa::float4> y,
    luisa::compute::Expr<luisa::float4> z) noexcept;
[[nodiscard]] luisa::compute::Float4x4 packed_transform(
    Cursor &cursor) noexcept;
[[nodiscard]] Dual3 transform_point(
    luisa::compute::Expr<luisa::float4x4> transform,
    const Dual3 &value) noexcept;

[[nodiscard]] luisa::compute::Float stack_load_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float stack_load_float_default(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<float> value) noexcept;
[[nodiscard]] luisa::compute::Float3 stack_load_float3(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float3 stack_load_float3_default(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<luisa::float3> value) noexcept;
[[nodiscard]] luisa::compute::Int stack_load_int(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float stack_load_input_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> bits) noexcept;
[[nodiscard]] luisa::compute::Float3 stack_load_input_float3(
    Stack &stack,
    luisa::compute::Expr<std::uint32_t> x_bits,
    luisa::compute::Expr<std::uint32_t> y_bits,
    luisa::compute::Expr<std::uint32_t> z_bits) noexcept;
[[nodiscard]] Dual1 stack_load_input_dual_float(
    Stack &stack, luisa::compute::Expr<std::uint32_t> bits) noexcept;
[[nodiscard]] Dual3 stack_load_input_dual_float3(
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
void stack_store_dual1(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       const Dual1 &value) noexcept;
void stack_store_dual3(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> offset,
                       const Dual3 &value) noexcept;
void stack_store_int(Stack &stack,
                     luisa::compute::Expr<std::uint32_t> offset,
                     luisa::compute::Expr<std::int32_t> value) noexcept;

[[nodiscard]] luisa::compute::Float3
normalize_cycles(luisa::compute::Expr<luisa::float3> value) noexcept;
[[nodiscard]] luisa::compute::Float3
safe_normalize_cycles(luisa::compute::Expr<luisa::float3> value) noexcept;
[[nodiscard]] Dual3 safe_normalize_dual(const Dual3 &value) noexcept;
void object_position_transform(luisa::compute::Float3 &value,
                               const TransformState &transform_state,
                               const ShaderData &shader_data,
                               bool object_motion_enabled) noexcept;
void object_inverse_normal_transform(luisa::compute::Float3 &value,
                                     const TransformState &transform_state,
                                     const ShaderData &shader_data,
                                     bool object_motion_enabled) noexcept;
void object_normal_transform(luisa::compute::Float3 &value,
                             const TransformState &transform_state,
                             const ShaderData &shader_data,
                             bool object_motion_enabled) noexcept;
void object_inverse_dir_transform(luisa::compute::Float3 &value,
                                  const TransformState &transform_state,
                                  const ShaderData &shader_data,
                                  bool object_motion_enabled) noexcept;

[[nodiscard]] luisa::compute::Float svm_math(
    luisa::compute::Expr<std::uint32_t> type,
    luisa::compute::Expr<float> a,
    luisa::compute::Expr<float> b,
    luisa::compute::Expr<float> c) noexcept;

void node_value_f(Cursor &cursor, Stack &stack) noexcept;
void node_value_v(Cursor &cursor, Stack &stack) noexcept;
void node_geometry(Cursor &cursor, Stack &stack,
                   const KernelGlobals &kernel_globals,
                   const ShaderData &shader_data,
                   bool use_derivatives) noexcept;
void node_object_info(Cursor &cursor, Stack &stack,
                      const InfoServices &services,
                      const ShaderData &shader_data) noexcept;
void node_particle_info(Cursor &cursor, Stack &stack,
                        const InfoServices &services,
                        const ShaderData &shader_data) noexcept;
void node_hair_info(Cursor &cursor, Stack &stack,
                    const InfoServices *services,
                    const ShaderData &shader_data,
                    luisa::compute::Bool &supported) noexcept;
void node_point_info(Cursor &cursor, Stack &stack,
                     const InfoServices &services,
                     const ShaderData &shader_data) noexcept;
void node_normal(Cursor &cursor, Stack &stack) noexcept;
void node_camera(Cursor &cursor, Stack &stack,
                 const TransformState &transform_state,
                 const ShaderData &shader_data) noexcept;
void node_fresnel(Cursor &cursor, Stack &stack,
                  const ShaderData &shader_data) noexcept;
void node_layer_weight(Cursor &cursor, Stack &stack,
                       const ShaderData &shader_data) noexcept;
void node_tex_coord(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals,
                    const TransformState &transform_state,
                    const ShaderData &shader_data,
                    const PathState &path_state,
                    bool use_derivatives,
                    bool volume_enabled,
                    bool object_motion_enabled) noexcept;
void node_tex_image(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals,
                    ShaderData &shader_data,
                    bool use_derivatives) noexcept;
void node_tex_image_box(Cursor &cursor, Stack &stack,
                        const KernelGlobals &kernel_globals,
                        const TransformState &transform_state,
                        ShaderData &shader_data,
                        bool use_derivatives,
                        bool object_motion_enabled) noexcept;
void node_tex_environment(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          ShaderData &shader_data,
                          bool use_derivatives) noexcept;
void node_mapping(Cursor &cursor, Stack &stack,
                  bool use_derivatives) noexcept;
void node_texture_mapping(Cursor &cursor, Stack &stack,
                          bool use_derivatives) noexcept;
void node_min_max(Cursor &cursor, Stack &stack) noexcept;
void node_vector_math_mapping_normalize(Cursor &cursor, Stack &stack,
                                        bool use_derivatives) noexcept;
void node_tex_noise(Cursor &cursor, Stack &stack) noexcept;
void node_tex_white_noise(Cursor &cursor, Stack &stack) noexcept;
void node_tex_gradient(Cursor &cursor, Stack &stack) noexcept;
void node_rgb_ramp(Cursor &cursor, Stack &stack) noexcept;
void node_curves(Cursor &cursor, Stack &stack) noexcept;
void node_float_curve(Cursor &cursor, Stack &stack) noexcept;
void node_attr_surface(Cursor &cursor, Stack &stack,
                       const KernelGlobals &kernel_globals,
                       const ShaderData &shader_data) noexcept;
void node_attr_derivative(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          const ShaderData &shader_data) noexcept;
void node_attr_volume(Cursor &cursor, Stack &stack,
                      const KernelGlobals &kernel_globals,
                      ShaderData &shader_data) noexcept;
void node_vertex_color(Cursor &cursor, Stack &stack,
                       const KernelGlobals &kernel_globals,
                       const ShaderData &shader_data) noexcept;
void node_vertex_color_derivative(Cursor &cursor, Stack &stack,
                                  const KernelGlobals &kernel_globals,
                                  const ShaderData &shader_data) noexcept;
void node_convert(Cursor &cursor, Stack &stack,
                  const KernelGlobals &kernel_globals,
                  bool use_derivatives) noexcept;
void node_light_path(Cursor &cursor, Stack &stack,
                     const ShaderData &shader_data,
                     const PathState &path_state,
                     std::uint32_t node_feature_mask) noexcept;
void node_light_falloff(Cursor &cursor, Stack &stack,
                        const ShaderData &shader_data) noexcept;
void node_math(Cursor &cursor, Stack &stack) noexcept;
void node_hsv(Cursor &cursor, Stack &stack) noexcept;
void node_gamma(Cursor &cursor, Stack &stack) noexcept;
void node_brightness(Cursor &cursor, Stack &stack) noexcept;
void node_invert(Cursor &cursor, Stack &stack) noexcept;
void node_mix(Cursor &cursor, Stack &stack) noexcept;
void node_mix_color(Cursor &cursor, Stack &stack) noexcept;
void node_mix_float(Cursor &cursor, Stack &stack) noexcept;
void node_mix_vector(Cursor &cursor, Stack &stack) noexcept;
void node_mix_vector_non_uniform(Cursor &cursor, Stack &stack) noexcept;
void node_separate_color(Cursor &cursor, Stack &stack) noexcept;
void node_combine_color(Cursor &cursor, Stack &stack) noexcept;
void node_separate_vector(Cursor &cursor, Stack &stack,
                          bool use_derivatives) noexcept;
void node_combine_vector(Cursor &cursor, Stack &stack,
                         bool use_derivatives) noexcept;
void node_vector_rotate(Cursor &cursor, Stack &stack) noexcept;
void node_vector_transform(Cursor &cursor, Stack &stack,
                           const TransformState &transform_state,
                           const ShaderData &shader_data,
                           bool object_motion_enabled) noexcept;
void node_wireframe(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals,
                    const TransformState &transform_state,
                    const ShaderData &shader_data,
                    bool require_triangle_primitive,
                    bool object_motion_enabled) noexcept;
void node_set_bump(Cursor &cursor, Stack &stack,
                   const TransformState &transform_state,
                   const ShaderData &shader_data,
                   bool bump_feature_enabled,
                   bool object_motion_enabled) noexcept;
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
void node_closure_background(
    Cursor &cursor, Stack &stack,
    luisa::compute::Expr<luisa::float3> closure_weight,
    ShaderData &shader_data) noexcept;
void node_closure_bsdf_skip(Cursor &cursor,
                            luisa::compute::Expr<std::uint32_t> closure_type)
    noexcept;

} // namespace psycles::luisa_backend::cycles_svm::detail
