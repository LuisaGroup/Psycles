#pragma once

#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/shader_program.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace psycles::compiler::cycles_svm {

struct ShaderImage {
  bool valid{};
  std::string diagnostic;
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types_used{};
  std::uint32_t peak_stack_usage{};
};

// Compile all Cycles routines for one material into its local word stream:
// four-word ShaderJump, optional bump routine, surface, volume, displacement.
// Unsupported Cycles node families reject the image; they never select the
// previous Psycles execution plan.
[[nodiscard]] ShaderImage compile_shader(const ShaderProgram &shader);

} // namespace psycles::compiler::cycles_svm
