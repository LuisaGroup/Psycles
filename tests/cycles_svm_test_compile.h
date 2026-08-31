#pragma once

#include <psycles/compiler/cycles_svm_compiler.h>

// Convenience overload restricted to isolated one-material tests. Production
// compilation must retain one AttributeIDMap for the whole scene, matching
// Cycles ShaderManager::unique_attribute_id.
namespace psycles::compiler::cycles_svm {

[[nodiscard]] inline ShaderImage
compile_shader(const ShaderProgram &shader) {
  AttributeIDMap attribute_ids;
  return compile_shader(shader, attribute_ids);
}

} // namespace psycles::compiler::cycles_svm
