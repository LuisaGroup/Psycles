/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <psycles/luisa/cycles_svm.h>

#include <optional>
#include <span>

#include <luisa/dsl/builtin.h>
#include <luisa/dsl/local.h>

namespace psycles::luisa_backend::cycles_svm::detail {

// Cycles stack addressing is unchanged; only the physical local extent is a
// host/JIT input. The maximum-sized default keeps direct node fixtures and
// callers without a compiled scene valid. Bytecode VM mode uses a Local<float>
// array to support runtime-indexed stack slots; DirectJit mode uses per-case
// virtual SSA slots + lazy mutable scalar fallback for conditional blocks,
// eliminating the function-wide stack array alloca and cross-case PHIs.
class Stack final {
private:
  struct SlotState {
    std::optional<float> literal;
    const luisa::compute::Expression *ssa_expr{nullptr};
    const luisa::compute::Expression *var_expr{nullptr};
    bool use_var{false};
  };

  luisa::optional<luisa::compute::Local<float>> _array;
  luisa::compute::detail::FunctionBuilder *_owner_fb{nullptr};
  luisa::compute::ScopeStmt *_root_scope{nullptr};
  std::size_t _size{0u};
  bool _direct_jit{false};
  luisa::vector<SlotState> _slots;

  Stack(std::size_t size, bool direct_jit_mode) noexcept;

public:
  explicit Stack(std::size_t size = SVM_STACK_SIZE) noexcept;
  [[nodiscard]] static Stack direct_jit(std::size_t size) noexcept;

  [[nodiscard]] bool is_direct_jit() const noexcept { return _direct_jit; }
  [[nodiscard]] std::size_t size() const noexcept { return _size; }
  [[nodiscard]] bool at_root_scope() const noexcept;
  [[nodiscard]] std::optional<float>
  try_const_slot(std::uint32_t offset) const noexcept;
  void materialize_all_to_vars() noexcept;

  [[nodiscard]] luisa::compute::Expr<float>
  load_const_expr(std::uint32_t offset) noexcept;
  [[nodiscard]] luisa::compute::Float
  load_const(std::uint32_t offset) noexcept;
  [[nodiscard]] luisa::compute::Float
  load_dynamic(luisa::compute::Expr<std::uint32_t> offset) noexcept;
  void store_const(std::uint32_t offset,
                   luisa::compute::Expr<float> value) noexcept;
  void store_dynamic(luisa::compute::Expr<std::uint32_t> offset,
                     luisa::compute::Expr<float> value) noexcept;
};

/* A diagnostic evaluator reports unsupported dynamic transitions through a
 * device Bool. The renderer's compiler-validated path instead makes the same
 * transitions unreachable, matching Cycles' release SVM contract without
 * carrying a per-node status lane through the interpreter merge. */
class EvaluationTransition final {
private:
  luisa::compute::Bool *_diagnostic_supported;

public:
  explicit EvaluationTransition(
      luisa::compute::Bool *diagnostic_supported) noexcept
      : _diagnostic_supported{diagnostic_supported} {}

  void unsupported() const noexcept {
    if (_diagnostic_supported != nullptr) {
      *_diagnostic_supported = false;
    } else {
      luisa::compute::dsl::unreachable(
          "unsupported transition in compiler-validated Cycles SVM stream");
    }
  }
};

struct Differential3 {
  luisa::compute::Float3 dx;
  luisa::compute::Float3 dy;
};

[[nodiscard]] Differential3
differential_from_compact(luisa::compute::Expr<luisa::float3> direction,
                          luisa::compute::Expr<float> differential) noexcept;

template<typename Payload> class NodeDataView;

class Cursor final {
  template<typename Payload> friend class NodeDataView;

private:
  luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> _words;
  luisa::compute::UInt *_offset{nullptr};
  const std::uint32_t *_static_words{nullptr};
  std::size_t _static_count{};
  std::size_t _static_index{};
  std::uint32_t _global_base_offset{};

public:
  Cursor(luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> words,
         luisa::compute::UInt &offset) noexcept;
  Cursor(luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> words,
         std::span<const std::uint32_t> static_words,
         std::uint32_t global_base_offset) noexcept;

  [[nodiscard]] bool is_static() const noexcept {
    return _static_words != nullptr;
  }
  [[nodiscard]] luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>>
  words_buffer() const noexcept {
    return _words;
  }
  [[nodiscard]] luisa::compute::Expr<std::uint32_t>
  current_word_offset() const noexcept {
    if (_static_words != nullptr) {
      return luisa::compute::Expr<std::uint32_t>{
          _global_base_offset + static_cast<std::uint32_t>(_static_index)};
    }
    return *_offset;
  }

  [[nodiscard]] luisa::compute::Expr<std::uint32_t> word() noexcept;
  [[nodiscard]] luisa::compute::Expr<std::uint32_t>
  word_at(luisa::compute::Expr<std::uint32_t> relative_word) const noexcept;
  [[nodiscard]] luisa::compute::Expr<float> floating() noexcept;
  [[nodiscard]] luisa::compute::Expr<float>
  floating_at(luisa::compute::Expr<std::uint32_t> relative_word) const noexcept;
  void advance(luisa::compute::Expr<std::uint32_t> word_count) noexcept;
  [[nodiscard]] luisa::compute::Expr<std::uint32_t>
  byte(luisa::compute::Expr<std::uint32_t> word,
       std::uint32_t byte_index) const noexcept;
};

class DirectJitCaseCacheGuard final {
private:
  void *_prev{nullptr};
  void *_owned{nullptr};

public:
  DirectJitCaseCacheGuard() noexcept;
  ~DirectJitCaseCacheGuard() noexcept;
  DirectJitCaseCacheGuard(const DirectJitCaseCacheGuard &) = delete;
  DirectJitCaseCacheGuard &operator=(const DirectJitCaseCacheGuard &) = delete;
};

[[nodiscard]] std::optional<luisa::compute::Float3>
lookup_cached_surface_attr(const ShaderData &shader_data,
                           std::uint64_t attr_id,
                           std::uint32_t output_type) noexcept;
void store_cached_surface_attr(const ShaderData &shader_data,
                               std::uint64_t attr_id,
                               std::uint32_t output_type,
                               const luisa::compute::Float3 &value) noexcept;
[[nodiscard]] std::optional<Dual3>
lookup_cached_deriv_attr(const ShaderData &shader_data,
                         std::uint64_t attr_id,
                         std::uint32_t output_type) noexcept;
void store_cached_deriv_attr(const ShaderData &shader_data,
                             std::uint64_t attr_id,
                             std::uint32_t output_type,
                             const Dual3 &value) noexcept;

[[nodiscard]] std::optional<std::uint32_t>
try_const_uint(luisa::compute::Expr<std::uint32_t> expr) noexcept;
[[nodiscard]] std::optional<std::int32_t>
try_const_int(luisa::compute::Expr<std::int32_t> expr) noexcept;
[[nodiscard]] std::optional<float>
try_const_float(luisa::compute::Expr<float> expr) noexcept;
[[nodiscard]] luisa::compute::Expr<std::uint32_t>
stack_offset_add(luisa::compute::Expr<std::uint32_t> base,
                 std::uint32_t delta) noexcept;
[[nodiscard]] luisa::compute::Expr<std::uint32_t>
stack_offset_add(luisa::compute::Expr<std::uint32_t> base,
                 luisa::compute::Expr<std::uint32_t> delta) noexcept;

[[nodiscard]] luisa::compute::Float4x4
transform_from_rows(luisa::compute::Expr<luisa::float4> x,
                    luisa::compute::Expr<luisa::float4> y,
                    luisa::compute::Expr<luisa::float4> z) noexcept;
[[nodiscard]] luisa::compute::Float4x4
packed_transform(Cursor &cursor) noexcept;
[[nodiscard]] Dual3
transform_point(luisa::compute::Expr<luisa::float4x4> transform,
                const Dual3 &value) noexcept;

[[nodiscard]] luisa::compute::Float
stack_load_float(Stack &stack,
                 luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float
stack_load_float_default(Stack &stack,
                         luisa::compute::Expr<std::uint32_t> offset,
                         luisa::compute::Expr<float> value) noexcept;
[[nodiscard]] luisa::compute::Float3
stack_load_float3(Stack &stack,
                  luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float3
stack_load_float3_default(Stack &stack,
                          luisa::compute::Expr<std::uint32_t> offset,
                          luisa::compute::Expr<luisa::float3> value) noexcept;
[[nodiscard]] luisa::compute::Int
stack_load_int(Stack &stack,
               luisa::compute::Expr<std::uint32_t> offset) noexcept;
[[nodiscard]] luisa::compute::Float
stack_load_input_float(Stack &stack,
                       luisa::compute::Expr<std::uint32_t> bits) noexcept;
[[nodiscard]] luisa::compute::Float3
stack_load_input_float3(Stack &stack,
                        luisa::compute::Expr<std::uint32_t> x_bits,
                        luisa::compute::Expr<std::uint32_t> y_bits,
                        luisa::compute::Expr<std::uint32_t> z_bits) noexcept;
[[nodiscard]] Dual1
stack_load_input_dual_float(Stack &stack,
                            luisa::compute::Expr<std::uint32_t> bits) noexcept;
[[nodiscard]] Dual3 stack_load_input_dual_float3(
    Stack &stack, luisa::compute::Expr<std::uint32_t> x_bits,
    luisa::compute::Expr<std::uint32_t> y_bits,
    luisa::compute::Expr<std::uint32_t> z_bits) noexcept;

void stack_store_float(Stack &stack, luisa::compute::Expr<std::uint32_t> offset,
                       luisa::compute::Expr<float> value) noexcept;
void stack_store_float_if_valid(Stack &stack,
                                luisa::compute::Expr<std::uint32_t> offset,
                                luisa::compute::Expr<float> value) noexcept;
void stack_store_float3(Stack &stack,
                        luisa::compute::Expr<std::uint32_t> offset,
                        luisa::compute::Expr<luisa::float3> value) noexcept;
void stack_store_float3_if_valid(
    Stack &stack, luisa::compute::Expr<std::uint32_t> offset,
    luisa::compute::Expr<luisa::float3> value) noexcept;
void stack_store_dual1(Stack &stack, luisa::compute::Expr<std::uint32_t> offset,
                       const Dual1 &value) noexcept;
void stack_store_dual1_if_valid(Stack &stack,
                                luisa::compute::Expr<std::uint32_t> offset,
                                const Dual1 &value) noexcept;
void stack_store_dual3(Stack &stack, luisa::compute::Expr<std::uint32_t> offset,
                       const Dual3 &value) noexcept;
void stack_store_dual3_if_valid(Stack &stack,
                                luisa::compute::Expr<std::uint32_t> offset,
                                const Dual3 &value) noexcept;
void stack_store_int(Stack &stack, luisa::compute::Expr<std::uint32_t> offset,
                     luisa::compute::Expr<std::int32_t> value) noexcept;

[[nodiscard]] luisa::compute::Float3
normalize_cycles(luisa::compute::Expr<luisa::float3> value) noexcept;
[[nodiscard]] luisa::compute::Float3
safe_normalize_cycles(luisa::compute::Expr<luisa::float3> value) noexcept;
[[nodiscard]] Dual3 safe_normalize_dual(const Dual3 &value) noexcept;
[[nodiscard]] Dual3 normalize_dual_cycles(const Dual3 &value) noexcept;
[[nodiscard]] Dual3
shading_position_dual(const ShaderData &shader_data) noexcept;
[[nodiscard]] TriangleNormals
triangle_normals(const KernelGlobals &kernel_globals,
                 const ShaderData &shader_data) noexcept;
[[nodiscard]] luisa::compute::Float3
triangle_smooth_normal_unnormalized_object_space(
    const KernelGlobals &kernel_globals, const TransformState &transform_state,
    const ShaderData &shader_data, bool object_motion_enabled) noexcept;
[[nodiscard]] luisa::compute::Float3
rec709_to_rgb(const KernelGlobals &kernel_globals,
              luisa::compute::Expr<luisa::float3> rec709) noexcept;
[[nodiscard]] luisa::compute::Float3
xyz_to_rgb(const KernelGlobals &kernel_globals,
           luisa::compute::Expr<luisa::float3> xyz) noexcept;
void object_position_transform(luisa::compute::Float3 &value,
                               const TransformState &transform_state,
                               const ShaderData &shader_data,
                               bool object_motion_enabled) noexcept;
void object_position_transform(Dual3 &value,
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
void object_normal_transform(Dual3 &value,
                             const TransformState &transform_state,
                             const ShaderData &shader_data,
                             bool object_motion_enabled) noexcept;
void object_dir_transform(luisa::compute::Float3 &value,
                          const TransformState &transform_state,
                          const ShaderData &shader_data,
                          bool object_motion_enabled) noexcept;
void object_inverse_dir_transform(luisa::compute::Float3 &value,
                                  const TransformState &transform_state,
                                  const ShaderData &shader_data,
                                  bool object_motion_enabled) noexcept;

[[nodiscard]] luisa::compute::Float
svm_math(luisa::compute::Expr<std::uint32_t> type,
         luisa::compute::Expr<float> a, luisa::compute::Expr<float> b,
         luisa::compute::Expr<float> c) noexcept;

void node_value_f(Cursor &cursor, Stack &stack, bool use_derivatives) noexcept;
void node_value_v(Cursor &cursor, Stack &stack, bool use_derivatives) noexcept;
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
void node_hair_info(Cursor &cursor, Stack &stack, const InfoServices *services,
                    const ShaderData &shader_data,
                    const EvaluationTransition &transition) noexcept;
void node_point_info(Cursor &cursor, Stack &stack, const InfoServices &services,
                     const ShaderData &shader_data) noexcept;
void node_ambient_occlusion(Cursor &cursor, Stack &stack,
                            const AmbientOcclusionServices &services,
                            ShaderData &shader_data) noexcept;
void node_normal(Cursor &cursor, Stack &stack) noexcept;
void node_normal_map(Cursor &cursor, Stack &stack,
                     const KernelGlobals &kernel_globals,
                     const TransformState &transform_state,
                     const ShaderData &shader_data,
                     bool object_motion_enabled) noexcept;
void node_tangent(Cursor &cursor, Stack &stack,
                  const KernelGlobals &kernel_globals,
                  const TransformState &transform_state,
                  const ShaderData &shader_data, bool use_derivatives,
                  bool object_motion_enabled) noexcept;
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
                    const ShaderData &shader_data, const PathState &path_state,
                    bool use_derivatives, bool volume_enabled,
                    bool object_motion_enabled,
                    std::uint32_t tex_coord_mask = ~0u) noexcept;
void node_tex_image(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals,
                    ShaderData &shader_data, const PathState &path_state,
                    bool use_derivatives) noexcept;
void node_tex_image_box(Cursor &cursor, Stack &stack,
                        const KernelGlobals &kernel_globals,
                        const TransformState &transform_state,
                        ShaderData &shader_data, const PathState &path_state,
                        bool use_derivatives,
                        bool object_motion_enabled) noexcept;
void node_tex_environment(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          ShaderData &shader_data,
                          bool use_derivatives) noexcept;
void node_tex_sky(Cursor &cursor, Stack &stack,
                  const KernelGlobals &kernel_globals,
                  ShaderData &shader_data,
                  const PathState &path_state) noexcept;
void node_mapping(Cursor &cursor, Stack &stack, bool use_derivatives,
                  MappingStaticOptions options = {}) noexcept;
void node_texture_mapping(Cursor &cursor, Stack &stack,
                          bool use_derivatives) noexcept;
void node_min_max(Cursor &cursor, Stack &stack) noexcept;
void node_vector_math(Cursor &cursor, Stack &stack, bool use_derivatives,
                      std::uint64_t type_mask = ~0ull) noexcept;
[[nodiscard]] AttributeDescriptor find_attribute_from_base_offset(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    luisa::compute::Expr<std::uint32_t> base_attr_offset,
    luisa::compute::Expr<luisa::ulong> id) noexcept;

void node_tex_noise(Cursor &cursor, Stack &stack,
                    compiler::cycles_svm::NoiseUsage usage = {},
                    NoiseStaticOptions options = {}) noexcept;
void node_tex_white_noise(Cursor &cursor, Stack &stack) noexcept;
void node_tex_gradient(Cursor &cursor, Stack &stack) noexcept;
void node_tex_voronoi(Cursor &cursor, Stack &stack,
                      bool voronoi_extra_enabled,
                      VoronoiStaticOptions options = {}) noexcept;
void node_tex_gabor(Cursor &cursor, Stack &stack) noexcept;
void node_radial_tiling(Cursor &cursor, Stack &stack) noexcept;
void node_tex_wave(Cursor &cursor, Stack &stack,
                   WaveStaticOptions options = {}) noexcept;
void node_tex_magic(Cursor &cursor, Stack &stack) noexcept;
void node_tex_checker(Cursor &cursor, Stack &stack) noexcept;
void node_tex_brick(Cursor &cursor, Stack &stack) noexcept;
void node_rgb_ramp(Cursor &cursor, Stack &stack) noexcept;
void node_curves(Cursor &cursor, Stack &stack,
                 bool may_extrapolate = true) noexcept;
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
                     const ShaderData &shader_data, const PathState &path_state,
                     std::uint32_t node_feature_mask) noexcept;
void node_light_falloff(Cursor &cursor, Stack &stack,
                        const ShaderData &shader_data) noexcept;
void node_ies(Cursor &cursor, Stack &stack,
              const KernelGlobals &kernel_globals) noexcept;
void node_math(Cursor &cursor, Stack &stack,
               std::uint64_t type_mask = ~0ull) noexcept;
void node_hsv(Cursor &cursor, Stack &stack) noexcept;
void node_gamma(Cursor &cursor, Stack &stack) noexcept;
void node_brightness(Cursor &cursor, Stack &stack) noexcept;
void node_invert(Cursor &cursor, Stack &stack) noexcept;
void node_mix(Cursor &cursor, Stack &stack,
              std::uint32_t blend_mask = ~0u) noexcept;
void node_mix_color(Cursor &cursor, Stack &stack,
                    std::uint32_t blend_mask = ~0u) noexcept;
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
                   const ShaderData &shader_data, bool bump_feature_enabled,
                   bool object_motion_enabled, bool bump_state_enabled = true,
                   bool bump_object_space_enabled = true) noexcept;
void node_set_normal(Cursor &cursor, Stack &stack,
                     ShaderData &shader_data) noexcept;
void node_enter_bump_eval(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          const TransformState &transform_state,
                          ShaderData &shader_data,
                          bool object_motion_enabled,
                          bool may_have_undisplaced_position = true) noexcept;
void node_leave_bump_eval(Cursor &cursor, Stack &stack,
                          ShaderData &shader_data) noexcept;
void node_set_displacement(Cursor &cursor, Stack &stack,
                           ShaderData &shader_data,
                           bool bump_feature_enabled) noexcept;
void node_displacement(Cursor &cursor, Stack &stack,
                       const TransformState &transform_state,
                       const ShaderData &shader_data,
                       bool bump_feature_enabled,
                       bool object_motion_enabled) noexcept;
void node_vector_displacement(Cursor &cursor, Stack &stack,
                              const KernelGlobals &kernel_globals,
                              const TransformState &transform_state,
                              const ShaderData &shader_data,
                              bool bump_feature_enabled,
                              bool object_motion_enabled) noexcept;
void node_clamp(Cursor &cursor, Stack &stack) noexcept;
void node_map_range(Cursor &cursor, Stack &stack,
                    std::uint32_t type_mask = ~0u) noexcept;
void node_vector_map_range(Cursor &cursor, Stack &stack,
                           std::uint32_t type_mask = ~0u) noexcept;
void node_blackbody(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals) noexcept;
void node_wavelength(Cursor &cursor, Stack &stack,
                     const KernelGlobals &kernel_globals) noexcept;

void node_closure_set_weight(Cursor &cursor,
                             luisa::compute::Float3 &closure_weight) noexcept;
void node_closure_weight(Cursor &cursor, Stack &stack,
                         luisa::compute::Float3 &closure_weight) noexcept;
void node_emission_weight(Cursor &cursor, Stack &stack,
                          luisa::compute::Float3 &closure_weight) noexcept;
void node_mix_closure(Cursor &cursor, Stack &stack) noexcept;
void node_closure_holdout(Cursor &cursor, Stack &stack,
                          luisa::compute::Expr<luisa::float3> closure_weight,
                          ShaderData &shader_data) noexcept;
void node_closure_emission(const KernelGlobals &kernel_globals,
                           Cursor &cursor, Stack &stack,
                           luisa::compute::Expr<luisa::float3> closure_weight,
                           ShaderData &shader_data,
                           const EvaluationTransition &transition) noexcept;
void node_closure_background(Cursor &cursor, Stack &stack,
                             luisa::compute::Expr<luisa::float3> closure_weight,
                             ShaderData &shader_data) noexcept;
void node_closure_bsdf(const KernelGlobals &kernel_globals, Cursor &cursor,
                       Stack &stack,
                       luisa::compute::Expr<luisa::float3> closure_weight,
                       compiler::cycles_svm::ShaderType shader_type,
                       std::uint32_t kernel_features,
                       std::uint32_t node_feature_mask, ShaderData &shader_data,
                       const PathState &path_state,
                       const EvaluationTransition &transition) noexcept;
void node_closure_bsdf_skip(
    Cursor &cursor, luisa::compute::Expr<std::uint32_t> closure_type,
    std::uint64_t svm_bsdf_node_mask = ~0ull) noexcept;

// Forwarding wrapper around KernelGlobals so scoped AST specializations
// (e.g. per-instruction or per-attribute) override only the queried metadata.
class DelegatingKernelGlobals : public KernelGlobals {
protected:
  const KernelGlobals &_inner;

public:
  explicit DelegatingKernelGlobals(const KernelGlobals &inner) noexcept
      : _inner{inner} {}

  [[nodiscard]] luisa::compute::Float cycles_bsdf_data(
      luisa::compute::Expr<std::uint32_t> index) const noexcept override {
    return _inner.cycles_bsdf_data(index);
  }
  [[nodiscard]] luisa::compute::Bool caustics_reflective() const noexcept override {
    return _inner.caustics_reflective();
  }
  [[nodiscard]] luisa::compute::Bool caustics_refractive() const noexcept override {
    return _inner.caustics_refractive();
  }
  [[nodiscard]] luisa::compute::Float
  transparent_roughness_squared_threshold() const noexcept override {
    return _inner.transparent_roughness_squared_threshold();
  }
  [[nodiscard]] luisa::compute::Bool
  background_use_sun_guiding() const noexcept override {
    return _inner.background_use_sun_guiding();
  }
  [[nodiscard]] bool
  has_shadow_terminator_shading_offset() const noexcept override {
    return _inner.has_shadow_terminator_shading_offset();
  }
  [[nodiscard]] bool may_have_thin_film() const noexcept override {
    return _inner.may_have_thin_film();
  }
  [[nodiscard]] compiler::cycles_svm::ClosureUsage
  closure_usage() const noexcept override {
    return _inner.closure_usage();
  }
  [[nodiscard]] SurfaceAttributeSummary
  surface_attribute_summary() const noexcept override {
    return _inner.surface_attribute_summary();
  }
  [[nodiscard]] SurfaceAttributeSummary
  attribute_summary_for_id(std::uint64_t id) const noexcept override {
    return _inner.attribute_summary_for_id(id);
  }
  [[nodiscard]] NormalMapUsage normal_map_usage() const noexcept override {
    return _inner.normal_map_usage();
  }
  [[nodiscard]] std::uint32_t mix_blend_mask() const noexcept override {
    return _inner.mix_blend_mask();
  }
  [[nodiscard]] NoiseStaticOptions
  noise_static_options() const noexcept override {
    return _inner.noise_static_options();
  }
  [[nodiscard]] std::uint32_t tex_coord_mask() const noexcept override {
    return _inner.tex_coord_mask();
  }
  [[nodiscard]] bool bump_uses_object_space() const noexcept override {
    return _inner.bump_uses_object_space();
  }
  [[nodiscard]] bool curves_may_extrapolate() const noexcept override {
    return _inner.curves_may_extrapolate();
  }
  [[nodiscard]] std::uint64_t math_type_mask() const noexcept override {
    return _inner.math_type_mask();
  }
  [[nodiscard]] std::uint64_t vector_math_type_mask() const noexcept override {
    return _inner.vector_math_type_mask();
  }
  [[nodiscard]] MappingStaticOptions
  mapping_static_options() const noexcept override {
    return _inner.mapping_static_options();
  }
  [[nodiscard]] std::uint32_t map_range_type_mask() const noexcept override {
    return _inner.map_range_type_mask();
  }
  [[nodiscard]] VoronoiStaticOptions
  voronoi_static_options() const noexcept override {
    return _inner.voronoi_static_options();
  }
  [[nodiscard]] WaveStaticOptions
  wave_static_options() const noexcept override {
    return _inner.wave_static_options();
  }
  [[nodiscard]] bool has_curves() const noexcept override {
    return _inner.has_curves();
  }
  [[nodiscard]] luisa::compute::Float object_shadow_terminator_shading_offset(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_shadow_terminator_shading_offset(object);
  }
  [[nodiscard]] std::optional<luisa::compute::Float> object_volume_density(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_volume_density(object);
  }
  [[nodiscard]] luisa::compute::Float
  ies(luisa::compute::Expr<std::uint32_t> offset) const noexcept override {
    return _inner.ies(offset);
  }
  [[nodiscard]] const InfoServices *info_services() const noexcept override {
    return _inner.info_services();
  }
  [[nodiscard]] const AmbientOcclusionServices *
  ambient_occlusion_services() const noexcept override {
    return _inner.ambient_occlusion_services();
  }
  [[nodiscard]] TriangleVertices triangle_vertices(
      luisa::compute::Expr<std::uint32_t> object,
      luisa::compute::Expr<std::uint32_t> prim) const noexcept override {
    return _inner.triangle_vertices(object, prim);
  }
  [[nodiscard]] TriangleVertices
  motion_triangle_vertices(luisa::compute::Expr<std::uint32_t> object,
                           luisa::compute::Expr<std::uint32_t> prim,
                           luisa::compute::Expr<float> time) const noexcept override {
    return _inner.motion_triangle_vertices(object, prim, time);
  }
  [[nodiscard]] luisa::compute::Float3 film_rgb_to_y() const noexcept override {
    return _inner.film_rgb_to_y();
  }
  [[nodiscard]] luisa::compute::Float3
  primitive_tangent(const ShaderData &shader_data) const noexcept override {
    return _inner.primitive_tangent(shader_data);
  }
  [[nodiscard]] Dual3 primitive_tangent_derivative(
      const ShaderData &shader_data) const noexcept override {
    return _inner.primitive_tangent_derivative(shader_data);
  }
  [[nodiscard]] luisa::compute::UInt object_attribute_map_offset(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_attribute_map_offset(object);
  }
  [[nodiscard]] luisa::compute::Var<compiler::cycles_svm::AttributeMap>
  attribute_map(
      luisa::compute::Expr<std::uint32_t> offset) const noexcept override {
    return _inner.attribute_map(offset);
  }
  [[nodiscard]] std::optional<luisa::compute::UInt2>
  find_attribute_map_offset(
      luisa::compute::Expr<std::uint32_t> start_offset,
      luisa::compute::Expr<luisa::ulong> id) const noexcept override {
    return _inner.find_attribute_map_offset(start_offset, id);
  }
  [[nodiscard]] std::optional<SurfaceAttributeBuffers>
  surface_attribute_buffers() const noexcept override {
    return _inner.surface_attribute_buffers();
  }
  [[nodiscard]] bool prefers_outlined_loops() const noexcept override {
    return _inner.prefers_outlined_loops();
  }
  [[nodiscard]] luisa::compute::Float attribute_float(
      luisa::compute::Expr<std::int32_t> offset) const noexcept override {
    return _inner.attribute_float(offset);
  }
  [[nodiscard]] luisa::compute::Float2 attribute_float2(
      luisa::compute::Expr<std::int32_t> offset) const noexcept override {
    return _inner.attribute_float2(offset);
  }
  [[nodiscard]] luisa::compute::Var<compiler::cycles_svm::packed_float3>
  attribute_float3(
      luisa::compute::Expr<std::int32_t> offset) const noexcept override {
    return _inner.attribute_float3(offset);
  }
  [[nodiscard]] luisa::compute::Float4 attribute_float4(
      luisa::compute::Expr<std::int32_t> offset) const noexcept override {
    return _inner.attribute_float4(offset);
  }
  [[nodiscard]] luisa::compute::Var<compiler::cycles_svm::uchar4>
  attribute_uchar4(
      luisa::compute::Expr<std::int32_t> offset) const noexcept override {
    return _inner.attribute_uchar4(offset);
  }
  [[nodiscard]] luisa::compute::Var<compiler::cycles_svm::packed_normal>
  attribute_normal(
      luisa::compute::Expr<std::int32_t> offset) const noexcept override {
    return _inner.attribute_normal(offset);
  }
  [[nodiscard]] luisa::compute::UInt3 triangle_vertex_indices(
      luisa::compute::Expr<std::uint32_t> prim) const noexcept override {
    return _inner.triangle_vertex_indices(prim);
  }
  [[nodiscard]] luisa::compute::Int object_normal_offset(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_normal_offset(object);
  }
  [[nodiscard]] luisa::compute::UInt object_num_geom_steps(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_num_geom_steps(object);
  }
  [[nodiscard]] luisa::compute::Int object_num_vertices(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_num_vertices(object);
  }
  [[nodiscard]] luisa::compute::Int object_num_primitives(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_num_primitives(object);
  }
  [[nodiscard]] luisa::compute::Float3 object_dupli_generated(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_dupli_generated(object);
  }
  [[nodiscard]] luisa::compute::Float3 object_dupli_uv(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_dupli_uv(object);
  }
  [[nodiscard]] luisa::compute::UInt camera_type() const noexcept override {
    return _inner.camera_type();
  }
  [[nodiscard]] luisa::compute::Float camera_width() const noexcept override {
    return _inner.camera_width();
  }
  [[nodiscard]] luisa::compute::Float camera_height() const noexcept override {
    return _inner.camera_height();
  }
  [[nodiscard]] luisa::compute::Float3 camera_world_to_ndc(
      const ShaderData &shader_data,
      luisa::compute::Expr<luisa::float3> position) const noexcept override {
    return _inner.camera_world_to_ndc(shader_data, position);
  }
  [[nodiscard]] luisa::compute::Var<compiler::cycles_svm::KernelCurve>
  curve(luisa::compute::Expr<std::uint32_t> prim) const noexcept override {
    return _inner.curve(prim);
  }
  [[nodiscard]] luisa::compute::Int object_position_offset(
      luisa::compute::Expr<std::uint32_t> object) const noexcept override {
    return _inner.object_position_offset(object);
  }
  [[nodiscard]] luisa::compute::Float4
  curve_key(luisa::compute::Expr<std::int32_t> key) const noexcept override {
    return _inner.curve_key(key);
  }
  [[nodiscard]] luisa::compute::Bool film_is_rec709() const noexcept override {
    return _inner.film_is_rec709();
  }
  [[nodiscard]] luisa::compute::Float3 film_xyz_to_r() const noexcept override {
    return _inner.film_xyz_to_r();
  }
  [[nodiscard]] luisa::compute::Float3 film_xyz_to_g() const noexcept override {
    return _inner.film_xyz_to_g();
  }
  [[nodiscard]] luisa::compute::Float3 film_xyz_to_b() const noexcept override {
    return _inner.film_xyz_to_b();
  }
  [[nodiscard]] luisa::compute::Float3
  film_rec709_to_r() const noexcept override {
    return _inner.film_rec709_to_r();
  }
  [[nodiscard]] luisa::compute::Float3
  film_rec709_to_g() const noexcept override {
    return _inner.film_rec709_to_g();
  }
  [[nodiscard]] luisa::compute::Float3
  film_rec709_to_b() const noexcept override {
    return _inner.film_rec709_to_b();
  }
  [[nodiscard]] luisa::compute::Float3
  object_inverse_position_transform_if_object(
      const ShaderData &shader_data,
      luisa::compute::Expr<luisa::float3> value) const noexcept override {
    return _inner.object_inverse_position_transform_if_object(shader_data,
                                                              value);
  }
  [[nodiscard]] Dual3 object_inverse_position_transform_if_object_derivative(
      const ShaderData &shader_data, const Dual3 &value) const noexcept override {
    return _inner.object_inverse_position_transform_if_object_derivative(
        shader_data, value);
  }
  [[nodiscard]] luisa::compute::Float3 object_inverse_position_transform(
      const ShaderData &shader_data,
      luisa::compute::Expr<luisa::float3> value) const noexcept override {
    return _inner.object_inverse_position_transform(shader_data, value);
  }
  [[nodiscard]] luisa::compute::Float4 kernel_image_interp_with_udim(
      ShaderData &shader_data,
      luisa::compute::Expr<std::int32_t> image_texture_id,
      const Dual2 &uv) const noexcept override {
    return _inner.kernel_image_interp_with_udim(shader_data, image_texture_id,
                                                uv);
  }
  [[nodiscard]] luisa::compute::Float4 kernel_image_interp_3d(
      ShaderData &shader_data,
      luisa::compute::Expr<std::int32_t> image_texture_id,
      luisa::compute::Expr<luisa::float3> position,
      luisa::compute::Expr<std::int32_t> interpolation,
      luisa::compute::Expr<bool> stochastic) const noexcept override {
    return _inner.kernel_image_interp_3d(shader_data, image_texture_id,
                                         position, interpolation, stochastic);
  }
};

} // namespace psycles::luisa_backend::cycles_svm::detail
