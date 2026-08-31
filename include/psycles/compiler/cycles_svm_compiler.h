#pragma once

#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/shader_program.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace psycles::compiler::cycles_svm {

struct ShaderImage {
  bool valid{};
  std::string diagnostic;
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types_used{};
  std::uint32_t peak_stack_usage{};
};

// Exact scene-owned SVMCompiler mode which Cycles sets from Shader::is_background.
// It is deliberately explicit: Texture Coordinate compilation cannot infer
// world/background semantics from the graph topology.
struct ShaderCompileContext {
  bool background{};
};

// Scene-wide named-attribute identifier state corresponding to Cycles 5.2.1
// ShaderManager::unique_attribute_id. The first request for a name assigns
// ATTR_STD_NUM + current size; subsequent requests return the same identifier.
class AttributeIDMap final {
private:
  std::mutex _attribute_lock;
  std::unordered_map<std::string, std::uint64_t> _unique_attribute_id;

public:
  [[nodiscard]] std::uint64_t get_attribute_id(std::string_view name);
  [[nodiscard]] static constexpr std::uint64_t
  get_attribute_id(AttributeStandard standard) noexcept {
    return static_cast<std::uint64_t>(standard);
  }
};

// Compile all Cycles routines for one material into its local word stream:
// four-word ShaderJump, optional bump routine, surface, volume, displacement.
// Unsupported Cycles node families reject the image; they never select the
// previous Psycles execution plan.
[[nodiscard]] ShaderImage compile_shader(const ShaderProgram &shader,
                                         AttributeIDMap &attribute_ids,
                                         ShaderCompileContext context);

} // namespace psycles::compiler::cycles_svm
