#pragma once

#include <psycles/compiler/cycles_svm_compiler.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace psycles::compiler::cycles_svm {

// Exact host image produced by Cycles 5.2.1
// SVMShaderManager::device_update_specific: one dense ShaderJump table
// followed by every shader tail in the same order. Shader identity is the
// input position; no topology interning or material pre-evaluation occurs at
// this boundary.
struct ShaderTableImage {
  bool valid{};
  std::string diagnostic;
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types_used{};
  std::uint32_t peak_stack_usage{};
  std::uint32_t shader_count{};
};

// Link per-shader local images emitted by compile_shader into the exact global
// Cycles SVM layout. For local shader i with tail base G_i and local entry L,
// the global entry is G_i + (L - jump_node_word_count). All tail words are
// copied verbatim, so relative closure-control jumps retain their semantics.
[[nodiscard]] ShaderTableImage
link_shader_table(std::span<const ShaderImage> shaders);

} // namespace psycles::compiler::cycles_svm
