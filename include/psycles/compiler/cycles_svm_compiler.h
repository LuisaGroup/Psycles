#pragma once

#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/shader_program.h>

#include <array>
#include <compare>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace psycles::compiler::cycles_svm {

// Normalized Cycles ImageParams identity. A Psycles resource_id already
// selects one exported source together with its color-space, alpha and frame
// metadata; interpolation and extension are the remaining node-varying
// fields. Their numeric values match util/types_image.h in Cycles 5.2.1.
enum class ImageInterpolation : std::uint8_t {
  linear = 0u,
  closest = 1u,
  cubic = 2u,
  smart = 3u,
};

enum class ImageExtension : std::uint8_t {
  repeat = 0u,
  extend = 1u,
  clip = 2u,
  mirror = 3u,
};

struct ImageBinding {
  std::uint64_t resource_id{};
  ImageInterpolation interpolation{ImageInterpolation::linear};
  ImageExtension extension{ImageExtension::repeat};

  auto operator<=>(const ImageBinding &) const = default;
};

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

// Scene-wide equivalent of Cycles ImageManager handle interning for the SVM
// visible fields. The same source image with a different sampler is a distinct
// handle; equal triples are deduplicated across materials.
class ImageIDMap final {
private:
  mutable std::mutex _image_lock;
  std::map<ImageBinding, std::int32_t> _image_ids;
  std::vector<ImageBinding> _bindings;

public:
  [[nodiscard]] std::int32_t get_image_id(ImageBinding binding);
  [[nodiscard]] std::vector<ImageBinding> bindings() const;
};

// Compile all Cycles routines for one material into its local word stream:
// four-word ShaderJump, optional bump routine, surface, volume, displacement.
// Unsupported Cycles node families reject the image; they never select the
// previous Psycles execution plan.
[[nodiscard]] ShaderImage compile_shader(const ShaderProgram &shader,
                                         AttributeIDMap &attribute_ids,
                                         ImageIDMap &image_ids,
                                         ShaderCompileContext context);

// Compatibility boundary for node families that do not reference images.
// Production scene compilation must pass one ImageIDMap for the whole scene.
[[nodiscard]] ShaderImage compile_shader(const ShaderProgram &shader,
                                         AttributeIDMap &attribute_ids,
                                         ShaderCompileContext context);

} // namespace psycles::compiler::cycles_svm
