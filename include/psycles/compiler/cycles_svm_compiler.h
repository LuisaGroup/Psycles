#pragma once

#include <psycles/compiler/cycles_svm_attribute_request.h>
#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/contract/scene.h>

#include <array>
#include <bit>
#include <compare>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
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

// Exact identity of a generated Cycles SkyLoader image. The key deliberately
// stores IEEE words instead of ordering float values: ImageManager equality is
// exact field equality, and NaNs must not violate std::map's strict ordering.
// Sun rotation, size, intensity and disc visibility are absent because they do
// not affect the generated sky image in Cycles 5.2.1.
struct NishitaImageBinding {
  bool multiple_scattering{};
  std::array<std::uint32_t, 5u> parameter_bits{};

  auto operator<=>(const NishitaImageBinding &) const = default;

  [[nodiscard]] static constexpr NishitaImageBinding encode(
      bool multiple, float sun_elevation, float altitude, float air_density,
      float aerosol_density, float ozone_density) noexcept {
    return {.multiple_scattering = multiple,
            .parameter_bits = {
                std::bit_cast<std::uint32_t>(sun_elevation),
                std::bit_cast<std::uint32_t>(altitude),
                std::bit_cast<std::uint32_t>(air_density),
                std::bit_cast<std::uint32_t>(aerosol_density),
                std::bit_cast<std::uint32_t>(ozone_density)}};
  }

  [[nodiscard]] constexpr float sun_elevation() const noexcept {
    return std::bit_cast<float>(parameter_bits[0u]);
  }
  [[nodiscard]] constexpr float altitude() const noexcept {
    return std::bit_cast<float>(parameter_bits[1u]);
  }
  [[nodiscard]] constexpr float air_density() const noexcept {
    return std::bit_cast<float>(parameter_bits[2u]);
  }
  [[nodiscard]] constexpr float aerosol_density() const noexcept {
    return std::bit_cast<float>(parameter_bits[3u]);
  }
  [[nodiscard]] constexpr float ozone_density() const noexcept {
    return std::bit_cast<float>(parameter_bits[4u]);
  }
};

struct ImageBinding {
  std::uint64_t resource_id{};
  ImageInterpolation interpolation{ImageInterpolation::linear};
  ImageExtension extension{ImageExtension::repeat};
  // Empty selects an exported scene image identified by resource_id. A value
  // selects the generated SkyLoader resource and leaves resource_id unused.
  std::optional<NishitaImageBinding> nishita;

  auto operator<=>(const ImageBinding &) const = default;
};

// Shader facts populated by the same Cycles-graph traversal that emits the
// local SVM image. These are not inferred from the resulting opcode stream:
// node virtuals and output topology carry source semantics which bytecode
// alone does not preserve.
struct ShaderCompileMetadata {
  bool has_surface{};
  bool has_surface_transparent{};
  bool has_surface_raytrace{};
  bool has_surface_bssrdf{};
  bool has_surface_spatial_varying{};
  bool has_volume{};
  bool has_volume_connected{};
  bool has_volume_spatial_varying{};
  bool has_volume_attribute_dependency{};
  bool has_bump_from_surface{};
  bool has_bump_from_displacement{};
  bool has_bssrdf_bump{};
  bool has_light_path_node{};
  bool emission_is_constant{true};
  bool emission_from_auto_conversion{};
  Vec3f emission_estimate{};
};

struct ShaderImage {
  bool valid{};
  std::string diagnostic;
  std::vector<std::uint32_t> words;
  std::array<bool, NODE_NUM> node_types_used{};
  // Exact, canonical Shader::attributes image collected before Cycles graph
  // finalization. Requests stay symbolic here so hidden geometry dependencies
  // cannot perturb the named IDs allocated by SVM bytecode compilation.
  std::vector<AttributeRequest> attribute_requests;
  std::uint32_t peak_stack_usage{};
  ShaderCompileMetadata metadata;
};

// Exact scene-owned SVMCompiler mode which Cycles sets from Shader::is_background.
// It is deliberately explicit: Texture Coordinate compilation cannot infer
// world/background semantics from the graph topology.
struct ShaderCompileContext {
  bool background{};
  contract::DisplacementMethod displacement_method{
      contract::DisplacementMethod::bump};
  // Cycles' Blackbody constant folder evaluates in Rec.709 and projects the
  // result through ShaderManager's active scene-linear transform. Keeping the
  // same scene-owned value here prevents host folding from silently assuming
  // Rec.709 when the device interpreter would use another working space.
  contract::ShaderColorSpace color_space{};
};

// Scene-wide named-attribute identifier state corresponding to Cycles 5.2.1
// ShaderManager::unique_attribute_id. The first request for a name assigns
// ATTR_STD_NUM + current size; subsequent requests return the same identifier.
class AttributeIDMap final {
private:
  mutable std::mutex _attribute_lock;
  std::unordered_map<std::string, std::uint64_t> _unique_attribute_id;

public:
  [[nodiscard]] std::uint64_t get_attribute_id(std::string_view name);
  [[nodiscard]] std::vector<std::pair<std::string, std::uint64_t>>
  bindings() const;
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

// Scene-wide equivalent of Cycles 5.2.1 LightManager's IES slot table. Slot
// identity is the complete raw IES byte string; parsing never changes shader
// topology and the packed table is material data consumed by NODE_IES.
class IESIDMap final {
private:
  mutable std::mutex _ies_lock;
  std::map<std::string, std::uint32_t, std::less<>> _ies_slots;
  std::vector<std::vector<float>> _packed_profiles;

public:
  [[nodiscard]] std::uint32_t get_ies_slot(std::string_view content);
  [[nodiscard]] std::vector<float> packed_data() const;
  [[nodiscard]] std::size_t slot_count() const;
};

// Compile all Cycles routines for one material into its local word stream:
// four-word ShaderJump, optional bump routine, surface, volume, displacement.
// Unsupported Cycles node families reject the image; they never select the
// previous Psycles execution plan.
[[nodiscard]] ShaderImage compile_shader(const ShaderProgram &shader,
                                         AttributeIDMap &attribute_ids,
                                         ImageIDMap &image_ids,
                                         IESIDMap &ies_ids,
                                         ShaderCompileContext context);

// Compatibility boundary for node families that do not need to inspect the
// scene-wide IES table. Production scene compilation must retain one IESIDMap
// across every material, just as it retains ImageIDMap.
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
