#pragma once

#include <psycles/compiler/cycles_svm_closure_usage.h>
#include <psycles/compiler/cycles_svm_node_types.h>
#include <psycles/compiler/cycles_svm_noise_usage.h>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace psycles::compiler::cycles_svm {

// Per-instruction static specialization facts captured directly by the unified
// Cycles SVM frontend. Backend translators (e.g. DirectJitTranslator) consume
// these exact facts to specialize each emitted node without runtime switches.
struct SvmInstructionSpecialization {
  bool may_have_thin_film{true};
  ClosureUsage closure_usage{};
  std::uint32_t mix_blend_mask{~0u};
  std::uint32_t normal_map_space_mask{~0u};
  bool normal_map_uses_original_base{true};
  NoiseUsage noise_usage{};
  bool noise_may_have_distortion{true};
  bool noise_may_need_color{true};
  std::uint32_t tex_coord_mask{~0u};
  bool bump_uses_object_space{true};
  bool curves_may_extrapolate{true};
  std::uint64_t math_type_mask{~0ull};
  std::uint64_t vector_math_type_mask{~0ull};
  std::uint32_t mapping_type_mask{~0u};
  bool mapping_may_rotate{true};
  std::uint32_t map_range_type_mask{~0u};
  std::uint32_t voronoi_dimension_mask{~0u};
  std::uint32_t voronoi_feature_mask{~0u};
  std::uint32_t voronoi_metric_mask{~0u};
  bool voronoi_may_have_detail{true};
  bool voronoi_may_normalize{true};
  bool voronoi_may_need_color{true};
  bool voronoi_may_need_position{true};
  bool voronoi_may_need_radius{true};
  std::uint32_t wave_type_mask{~0u};
  std::uint32_t wave_profile_mask{~0u};
  bool wave_may_have_distortion{true};

  friend constexpr bool operator==(const SvmInstructionSpecialization &,
                                   const SvmInstructionSpecialization &) = default;
};

struct SvmInstructionIr {
  ShaderNodeType opcode{NODE_END};
  // Offset in the owning word image immediately following the opcode word:
  // local to ShaderImage::words prior to linking, and relocated to
  // ShaderTableImage::words by link_shader_table().
  std::uint32_t payload_offset{};
  std::uint32_t payload_word_count{};
  SvmInstructionSpecialization specialization{};

  friend constexpr bool operator==(const SvmInstructionIr &,
                                   const SvmInstructionIr &) = default;
};

// Unified node-based SVM IR element: either a primitive SVM instruction or a
// structured forward conditional subtree (NODE_JUMP_IF_ONE / NODE_JUMP_IF_ZERO).
struct SvmOpIr {
  bool is_conditional{};
  // Meaningful when is_conditional == true:
  //   true  -> NODE_JUMP_IF_ONE  (executes children when stack[offset] < 1.0f)
  //   false -> NODE_JUMP_IF_ZERO (executes children when stack[offset] > 0.0f)
  bool jump_if_one{};
  std::uint8_t cond_stack_offset{SVM_STACK_INVALID};
  SvmInstructionIr instruction{};
  std::vector<SvmOpIr> children;
};

[[nodiscard]] bool operator==(const SvmOpIr &lhs, const SvmOpIr &rhs) noexcept;

using SvmBlockIr = std::vector<SvmOpIr>;

// Three canonical entry routines matching ShaderType (SURFACE [including any
// BUMP prefix], VOLUME, DISPLACEMENT).
struct SvmShaderIr {
  std::array<SvmBlockIr, 3u> routines{};

  [[nodiscard]] const SvmBlockIr &routine_for(ShaderType type) const noexcept {
    const auto idx = static_cast<std::size_t>(type);
    return routines[idx < routines.size() ? idx : 0u];
  }
};

[[nodiscard]] bool operator==(const SvmShaderIr &lhs,
                              const SvmShaderIr &rhs) noexcept;

struct SvmTableIr {
  bool valid{};
  std::vector<SvmShaderIr> shaders;
};

// Relocate local ShaderImage::words payload offsets to linked global offsets.
void relocate_svm_shader_ir(SvmShaderIr &ir, std::uint32_t add_offset,
                            std::uint32_t subtract_offset) noexcept;

// Structural and payload-value equivalence of two linked routines, independent
// of their distinct global payload_offset positions in words.
[[nodiscard]] bool same_linked_svm_block(
    std::span<const std::uint32_t> words, const SvmBlockIr &lhs,
    const SvmBlockIr &rhs) noexcept;

} // namespace psycles::compiler::cycles_svm
