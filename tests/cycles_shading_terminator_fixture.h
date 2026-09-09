#pragma once

#include <array>
#include <cstdint>

namespace psycles::test_support::shading_terminator {

// Authored function-boundary inputs only. The separate original-Cycles HIP
// observer produces expected BSDF values, PDFs, directions and labels.
// frequency is the finalized KernelObject value, not the Blender UI offset.
using Vector = std::array<float, 3u>;
enum class Kind : std::uint32_t { diffuse, translucent, none };
inline constexpr std::uint32_t bump_map_correction = 1u << 15u;

struct Input {
  Kind kind{Kind::diffuse};
  float frequency{1.0f};
  std::uint32_t flags{};
  Vector shader_normal{0.0f, 0.0f, 1.0f};
  Vector geometric_normal{0.0f, 0.0f, 1.0f};
  Vector closure_normal{0.0f, 0.0f, 1.0f};
  Vector incoming{0.0f, 0.0f, 1.0f};
  Vector evaluation_direction{0.6f, 0.0f, 0.8f};
  Vector random{0.125f, 0.5f, 0.75f};
};

inline constexpr std::array<const char *, 6u> names{
    "diffuse_identity_frequency",
    "diffuse_positive_frequency",
    "translucent_identity_frequency",
    "translucent_positive_frequency",
    "label_none_positive_frequency",
    "diffuse_bump_identity_frequency"};

inline constexpr auto inputs = [] {
  std::array<Input, names.size()> result{};
  result[1u].frequency = 2.0f;
  result[2u].kind = Kind::translucent;
  result[2u].evaluation_direction = {0.6f, 0.0f, -0.8f};
  result[3u] = result[2u];
  result[3u].frequency = 2.0f;
  result[4u].kind = Kind::none;
  result[4u].frequency = 2.0f;
  result[5u].flags = bump_map_correction;
  result[5u].closure_normal = {0.8f, 0.0f, 0.6f};
  return result;
}();

// Every row uses a valid object index equal to the case index. Original
// bsdf_eval/reflection sample directly dereference objects[sd.object], so an
// OBJECT_NONE row there would not be a defined Cycles oracle input. Test the
// Psycles KernelGlobals sentinel adapter in a separate service regression.
enum FloatRow : std::uint32_t {
  shader_normal_frequency,
  geometric_normal,
  closure_normal_sample_weight,
  sampled_value_pdf,
  sampled_direction_eta,
  sampled_roughness,
  evaluated_value_pdf,
  evaluation_direction,
  // x/y: original bump terms for sampled/evaluation directions; z/w unused.
  bump_terms,
  float_rows
};
// uint4[0/1]: sd.flag, num_closure, num_closure_left, closure type before/after.
// uint4[2]: sample label, valid object index, primitive type, fixture Kind.
inline constexpr std::uint32_t integer_rows = 3u;
inline constexpr std::uint32_t format_version = 1u;

} // namespace psycles::test_support::shading_terminator
