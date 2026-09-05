#pragma once

#include <psycles/compiler/cycles_svm_compiler.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
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

struct ShaderKernelSettings {
  std::string_view name;
  bool use_transparent_shadow{true};
  bool use_bump_map_correction{true};
  contract::EmissionSampling emission_sampling{
      contract::EmissionSampling::automatic};
  contract::VolumeSampling volume_sampling{
      contract::VolumeSampling::multiple_importance};
  contract::VolumeInterpolation volume_interpolation{
      contract::VolumeInterpolation::linear};
  std::int32_t pass_id{};
};

struct ShaderTableCompileUnit {
  std::uint32_t shader_index{};
  const ShaderProgram *shader{};
  ShaderCompileContext context{};
  ShaderKernelSettings kernel;
};

// One transaction containing every scene-global identity allocated while
// compiling the shader table. Named attributes preserve Cycles' assigned SVM
// ids; image and IES arrays are indexed directly by bytecode payloads.
struct CompiledShaderTable {
  ShaderTableImage table;
  // Exact shader-owned portion of Cycles Scene::kernel_features, including
  // its unconditional BSDF/Emission base. Geometry/integrator/film features
  // remain owned by their respective scene components.
  std::uint32_t kernel_features{
      kernel_feature_node_bsdf | kernel_feature_node_emission};
  // Parallel native DeviceScene::shaders image in the identical dense shader
  // index domain. Unrepresented source holes are byte-zero and unreachable.
  std::vector<KernelShader> kernel_shaders;
  // Per source shader, including inert holes. Cycles derives geometry
  // attribute demand from each shader rather than from the scene-wide opcode
  // union; object/particle packing must make the same distinction.
  std::vector<std::array<bool, NODE_NUM>> shader_node_types_used;
  // Exact first-insertion-wins Shader::attributes order. Geometry merges
  // these vectors in used-shader order before emitting its attribute map.
  std::vector<std::vector<std::uint64_t>>
      shader_attribute_ids_in_request_order;
  // Sorted unique projection retained solely for membership queries. It is
  // not a valid source for geometry attribute-map packing.
  std::vector<std::vector<std::uint64_t>> shader_attribute_ids_used;
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
