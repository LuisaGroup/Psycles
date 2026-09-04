#pragma once

#include <cstdint>

#include <psycles/compiler/surface_program.h>

namespace psycles::compiler {

// Device value opcodes are execution families, not authored graph-node
// identities. The order follows the corresponding Cycles 5.2 SVM families
// where Psycles implements them. One family owns one fixed bytecode ABI and
// may select a finite semantic subtype from the instruction payload.
enum class SurfaceSvmValueOpcode : std::uint8_t {
  value,
  geometry,
  geometry_derivative,
  convert,
  tex_coord,
  tex_coord_derivative,
  attribute,
  attribute_derivative,
  tex_image,
  tex_image_box,
  tex_environment,
  bump,
  bump_support,
  hsv,
  fresnel,
  layer_weight,
  math,
  vector_math,
  rgb_ramp,
  gamma,
  brightness_contrast,
  light_path,
  object_info,
  particle_info,
  hair_info,
  tangent,
  mapping,
  gradient,
  voronoi,
  wave,
  magic,
  checker,
  brick,
  white_noise,
  noise,
  normal_map,
  normal_map_derivative,
  invert,
  mix_color,
  separate_color,
  combine_color,
  wavelength,
  blackbody,
  map_range,
  vector_map_range,
  clamp,
  rgb_curve,
  mix_float,
  mix_vector,
  mix_vector_non_uniform,
  sky,
  ambient_occlusion,
  displacement,
  light_falloff,
  vector_displacement,
  count,
  invalid = 0xffu,
};

inline constexpr std::uint32_t surface_svm_value_opcode_count =
    static_cast<std::uint32_t>(SurfaceSvmValueOpcode::count);
inline constexpr std::uint32_t surface_value_operation_count =
    static_cast<std::uint32_t>(ValueOperation::vector_displacement) + 1u;
inline constexpr std::uint32_t surface_svm_opcode_image_projection_shift = 12u;
inline constexpr std::uint32_t surface_svm_opcode_image_projection_mask =
    0x3u << surface_svm_opcode_image_projection_shift;
inline constexpr std::uint32_t surface_svm_opcode_mix_vector_non_uniform_bit =
    1u;

[[nodiscard]] constexpr bool
surface_svm_value_opcode_valid(SurfaceSvmValueOpcode opcode) noexcept {
  return static_cast<std::uint32_t>(opcode) < surface_svm_value_opcode_count;
}

// Total semantic-operation -> execution-family projection before the two
// execution-shape refinements encoded in the compact immediate. Multiple
// operations map to one family exactly when one runtime subtype can select
// their behavior without changing the handler ABI.
[[nodiscard]] constexpr SurfaceSvmValueOpcode
surface_svm_value_base_opcode(ValueOperation operation) noexcept {
  switch (operation) {
  case ValueOperation::parameter:
    return SurfaceSvmValueOpcode::value;
  case ValueOperation::passthrough:
  case ValueOperation::scalar_to_color:
  case ValueOperation::scalar_to_boolean:
  case ValueOperation::color_to_scalar:
  case ValueOperation::vector_to_scalar:
    return SurfaceSvmValueOpcode::convert;
  case ValueOperation::add:
  case ValueOperation::subtract:
  case ValueOperation::multiply:
  case ValueOperation::divide:
  case ValueOperation::minimum:
  case ValueOperation::maximum:
  case ValueOperation::power:
  case ValueOperation::math:
  case ValueOperation::absolute:
    return SurfaceSvmValueOpcode::math;
  case ValueOperation::clamp01:
  case ValueOperation::clamp_range:
    return SurfaceSvmValueOpcode::clamp;
  case ValueOperation::map_range_float:
    return SurfaceSvmValueOpcode::map_range;
  case ValueOperation::map_range_vector:
    return SurfaceSvmValueOpcode::vector_map_range;
  case ValueOperation::vector_math_value:
  case ValueOperation::vector_math_vector:
    return SurfaceSvmValueOpcode::vector_math;
  case ValueOperation::mix_float:
    return SurfaceSvmValueOpcode::mix_float;
  case ValueOperation::mix_vector:
    return SurfaceSvmValueOpcode::mix_vector;
  case ValueOperation::mix:
  case ValueOperation::multiply_color:
    return SurfaceSvmValueOpcode::mix_color;
  case ValueOperation::hue_saturation:
    return SurfaceSvmValueOpcode::hsv;
  case ValueOperation::invert:
    return SurfaceSvmValueOpcode::invert;
  case ValueOperation::gamma:
    return SurfaceSvmValueOpcode::gamma;
  case ValueOperation::brightness_contrast:
    return SurfaceSvmValueOpcode::brightness_contrast;
  case ValueOperation::blackbody:
    return SurfaceSvmValueOpcode::blackbody;
  case ValueOperation::wavelength:
    return SurfaceSvmValueOpcode::wavelength;
  case ValueOperation::surface_position:
  case ValueOperation::shading_normal:
  case ValueOperation::geometric_normal:
  case ValueOperation::incoming:
  case ValueOperation::pointiness:
    return SurfaceSvmValueOpcode::geometry;
  case ValueOperation::tangent:
    return SurfaceSvmValueOpcode::tangent;
  case ValueOperation::uv:
  case ValueOperation::generated:
  case ValueOperation::reflection:
  case ValueOperation::object_position:
  case ValueOperation::object_position_with_transform:
    return SurfaceSvmValueOpcode::tex_coord;
  case ValueOperation::object_location:
  case ValueOperation::object_random:
  case ValueOperation::random_per_island:
    return SurfaceSvmValueOpcode::object_info;
  case ValueOperation::particle_index:
  case ValueOperation::particle_random:
    return SurfaceSvmValueOpcode::particle_info;
  case ValueOperation::back_facing:
  case ValueOperation::path_is_camera:
  case ValueOperation::path_is_shadow:
  case ValueOperation::path_is_diffuse:
  case ValueOperation::path_is_glossy:
  case ValueOperation::path_is_singular:
  case ValueOperation::path_is_reflection:
  case ValueOperation::path_is_transmission:
  case ValueOperation::path_is_volume_scatter:
  case ValueOperation::path_ray_length:
  case ValueOperation::path_ray_depth:
  case ValueOperation::path_diffuse_depth:
  case ValueOperation::path_glossy_depth:
  case ValueOperation::path_transparent_depth:
  case ValueOperation::path_transmission_depth:
  case ValueOperation::path_portal_depth:
    return SurfaceSvmValueOpcode::light_path;
  case ValueOperation::curve_is_strand:
  case ValueOperation::curve_intercept:
  case ValueOperation::curve_length:
  case ValueOperation::curve_thickness:
  case ValueOperation::curve_tangent_normal:
  case ValueOperation::curve_random:
    return SurfaceSvmValueOpcode::hair_info;
  case ValueOperation::fresnel:
    return SurfaceSvmValueOpcode::fresnel;
  case ValueOperation::layer_weight_fresnel:
  case ValueOperation::layer_weight_facing:
    return SurfaceSvmValueOpcode::layer_weight;
  case ValueOperation::mapping:
    return SurfaceSvmValueOpcode::mapping;
  case ValueOperation::image_color:
  case ValueOperation::image_alpha:
    return SurfaceSvmValueOpcode::tex_image;
  case ValueOperation::environment_color:
  case ValueOperation::environment_alpha:
    return SurfaceSvmValueOpcode::tex_environment;
  case ValueOperation::attribute_color:
  case ValueOperation::attribute_factor:
  case ValueOperation::attribute_alpha:
    return SurfaceSvmValueOpcode::attribute;
  case ValueOperation::normal_map:
    return SurfaceSvmValueOpcode::normal_map;
  case ValueOperation::bump:
    return SurfaceSvmValueOpcode::bump;
  case ValueOperation::bump_offset_zero:
  case ValueOperation::bump_filter_width:
  case ValueOperation::bump_samples:
    return SurfaceSvmValueOpcode::bump_support;
  case ValueOperation::sampled_surface_position:
  case ValueOperation::sampled_pointiness:
    return SurfaceSvmValueOpcode::geometry_derivative;
  case ValueOperation::sampled_uv:
  case ValueOperation::sampled_generated:
  case ValueOperation::sampled_object_position:
  case ValueOperation::sampled_object_position_with_transform:
    return SurfaceSvmValueOpcode::tex_coord_derivative;
  case ValueOperation::sampled_attribute_color:
  case ValueOperation::sampled_attribute_factor:
  case ValueOperation::sampled_attribute_alpha:
    return SurfaceSvmValueOpcode::attribute_derivative;
  case ValueOperation::sampled_normal_map:
    return SurfaceSvmValueOpcode::normal_map_derivative;
  case ValueOperation::noise_factor:
  case ValueOperation::noise_color:
    return SurfaceSvmValueOpcode::noise;
  case ValueOperation::white_noise_value:
  case ValueOperation::white_noise_color:
    return SurfaceSvmValueOpcode::white_noise;
  case ValueOperation::checker_color:
  case ValueOperation::checker_factor:
    return SurfaceSvmValueOpcode::checker;
  case ValueOperation::brick_color:
  case ValueOperation::brick_factor:
    return SurfaceSvmValueOpcode::brick;
  case ValueOperation::magic_color:
  case ValueOperation::magic_factor:
    return SurfaceSvmValueOpcode::magic;
  case ValueOperation::wave_color:
  case ValueOperation::wave_factor:
    return SurfaceSvmValueOpcode::wave;
  case ValueOperation::voronoi_distance:
  case ValueOperation::voronoi_color:
  case ValueOperation::voronoi_position:
  case ValueOperation::voronoi_w:
  case ValueOperation::voronoi_radius:
    return SurfaceSvmValueOpcode::voronoi;
  case ValueOperation::gradient:
    return SurfaceSvmValueOpcode::gradient;
  case ValueOperation::color_ramp:
    return SurfaceSvmValueOpcode::rgb_ramp;
  case ValueOperation::rgb_curve:
    return SurfaceSvmValueOpcode::rgb_curve;
  case ValueOperation::separate_r:
  case ValueOperation::separate_g:
  case ValueOperation::separate_b:
    return SurfaceSvmValueOpcode::separate_color;
  case ValueOperation::combine_color:
    return SurfaceSvmValueOpcode::combine_color;
  case ValueOperation::hosek_wilkie_sky:
  case ValueOperation::nishita_sky:
    return SurfaceSvmValueOpcode::sky;
  case ValueOperation::ambient_occlusion:
    return SurfaceSvmValueOpcode::ambient_occlusion;
  case ValueOperation::displacement:
    return SurfaceSvmValueOpcode::displacement;
  case ValueOperation::vector_displacement:
    return SurfaceSvmValueOpcode::vector_displacement;
  case ValueOperation::light_falloff:
    return SurfaceSvmValueOpcode::light_falloff;
  }
  return SurfaceSvmValueOpcode::invalid;
}

// The only family refinements are the same typed execution distinctions used
// by Cycles: box-projected images issue up to three texture samples, and a
// non-uniform vector mix reads a float3 factor. All remaining authored modes
// are runtime subtype data inside their base family.
[[nodiscard]] constexpr SurfaceSvmValueOpcode
surface_svm_value_opcode(ValueOperation operation,
                         std::uint32_t svm_immediate) noexcept {
  if ((operation == ValueOperation::image_color ||
       operation == ValueOperation::image_alpha) &&
      ((svm_immediate & surface_svm_opcode_image_projection_mask) >>
       surface_svm_opcode_image_projection_shift) == 1u) {
    return SurfaceSvmValueOpcode::tex_image_box;
  }
  if (operation == ValueOperation::mix_vector &&
      (svm_immediate & surface_svm_opcode_mix_vector_non_uniform_bit) != 0u) {
    return SurfaceSvmValueOpcode::mix_vector_non_uniform;
  }
  return surface_svm_value_base_opcode(operation);
}

[[nodiscard]] consteval bool surface_svm_value_opcode_contract_holds() {
  for (auto raw = std::uint32_t{}; raw < surface_value_operation_count; ++raw) {
    const auto operation = static_cast<ValueOperation>(raw);
    if (!surface_svm_value_opcode_valid(
            surface_svm_value_base_opcode(operation)) ||
        !surface_svm_value_opcode_valid(
            surface_svm_value_opcode(operation, 0u))) {
      return false;
    }
  }
  return surface_svm_value_opcode(ValueOperation::image_color, 0u) ==
             SurfaceSvmValueOpcode::tex_image &&
         surface_svm_value_opcode(
             ValueOperation::image_color,
             1u << surface_svm_opcode_image_projection_shift) ==
             SurfaceSvmValueOpcode::tex_image_box &&
         surface_svm_value_opcode(
             ValueOperation::image_color,
             2u << surface_svm_opcode_image_projection_shift) ==
             SurfaceSvmValueOpcode::tex_image &&
         surface_svm_value_opcode(ValueOperation::mix_vector, 0u) ==
             SurfaceSvmValueOpcode::mix_vector &&
         surface_svm_value_opcode(
             ValueOperation::mix_vector,
             surface_svm_opcode_mix_vector_non_uniform_bit) ==
             SurfaceSvmValueOpcode::mix_vector_non_uniform;
}

static_assert(surface_svm_value_opcode_contract_holds());

} // namespace psycles::compiler
