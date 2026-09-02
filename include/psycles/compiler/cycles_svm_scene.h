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

struct ShaderTableCompileUnit {
  std::uint32_t shader_index{};
  const ShaderProgram *shader{};
  ShaderCompileContext context{};
};

// One transaction containing every scene-global identity allocated while
// compiling the shader table. Named attributes preserve Cycles' assigned SVM
// ids; image and IES arrays are indexed directly by bytecode payloads.
struct CompiledShaderTable {
  ShaderTableImage table;
  // Per source shader, including inert holes. Cycles derives geometry
  // attribute demand from each shader rather than from the scene-wide opcode
  // union; object/particle packing must make the same distinction.
  std::vector<std::array<bool, NODE_NUM>> shader_node_types_used;
  std::vector<std::pair<std::string, std::uint64_t>> named_attributes;
  std::vector<ImageBinding> images;
  std::vector<float> ies;
};

// Link per-shader local images emitted by compile_shader into the exact global
// Cycles SVM layout. For local shader i with tail base G_i and local entry L,
// the global entry is G_i + (L - jump_node_word_count). All tail words are
// copied verbatim, so relative closure-control jumps retain their semantics.
[[nodiscard]] ShaderTableImage
link_shader_table(std::span<const ShaderImage> shaders);

// Compile shader units through one scene-wide resource interning domain and
// link them by their original Cycles shader indices. Missing indices become
// unreachable END-only entries; duplicate indices are accepted only when
// their complete local word images and specialization metadata agree.
[[nodiscard]] CompiledShaderTable
compile_shader_table(std::span<const ShaderTableCompileUnit> shaders);

} // namespace psycles::compiler::cycles_svm
