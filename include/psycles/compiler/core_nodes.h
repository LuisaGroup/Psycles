#pragma once

#include <psycles/contract/shader_graph.h>

namespace psycles::compiler {

namespace node_type {
inline constexpr auto constant_float = "psycles.constant.float";
inline constexpr auto constant_color = "psycles.constant.color";
inline constexpr auto geometry = "psycles.geometry";
inline constexpr auto texture_coordinate = "psycles.texture_coordinate";
inline constexpr auto mapping = "psycles.vector.mapping";
inline constexpr auto image_texture = "psycles.texture.image";
inline constexpr auto noise_texture = "psycles.texture.noise";
inline constexpr auto brick_texture = "psycles.texture.brick";
inline constexpr auto gradient_texture = "psycles.texture.gradient";
inline constexpr auto nishita_sky = "psycles.texture.nishita_sky";
inline constexpr auto add_float = "psycles.math.add";
inline constexpr auto subtract_float = "psycles.math.subtract";
inline constexpr auto multiply_float = "psycles.math.multiply";
inline constexpr auto divide_float = "psycles.math.divide";
inline constexpr auto minimum_float = "psycles.math.minimum";
inline constexpr auto maximum_float = "psycles.math.maximum";
inline constexpr auto power_float = "psycles.math.power";
inline constexpr auto absolute_float = "psycles.math.absolute";
inline constexpr auto clamp_float = "psycles.math.clamp";
inline constexpr auto clamp_range = "psycles.math.clamp_range";
inline constexpr auto scalar_to_color = "psycles.convert.scalar_to_color";
inline constexpr auto color_to_scalar = "psycles.convert.color_to_scalar";
inline constexpr auto vector_to_scalar = "psycles.convert.vector_to_scalar";
inline constexpr auto vector_to_color = "psycles.convert.vector_to_color";
inline constexpr auto color_to_vector = "psycles.convert.color_to_vector";
inline constexpr auto vector_to_normal = "psycles.convert.vector_to_normal";
inline constexpr auto normal_to_vector = "psycles.convert.normal_to_vector";
inline constexpr auto object_info = "psycles.object_info";
inline constexpr auto light_path = "psycles.light_path";
inline constexpr auto layer_weight = "psycles.layer_weight";
inline constexpr auto mix_color = "psycles.color.mix";
inline constexpr auto multiply_color = "psycles.color.multiply";
inline constexpr auto hue_saturation = "psycles.color.hue_saturation";
inline constexpr auto invert_color = "psycles.color.invert";
inline constexpr auto gamma_color = "psycles.color.gamma";
inline constexpr auto brightness_contrast =
    "psycles.color.brightness_contrast";
inline constexpr auto color_ramp = "psycles.color.ramp";
inline constexpr auto rgb_curve = "psycles.color.rgb_curve";
inline constexpr auto separate_color = "psycles.color.separate";
inline constexpr auto combine_color = "psycles.color.combine";
inline constexpr auto normal_map = "psycles.normal_map";
inline constexpr auto bump = "psycles.bump";
inline constexpr auto vertex_color = "psycles.attribute.vertex_color";
inline constexpr auto diffuse_bsdf = "psycles.closure.diffuse";
inline constexpr auto principled_bsdf = "psycles.closure.principled";
inline constexpr auto glossy_bsdf = "psycles.closure.glossy";
inline constexpr auto emission = "psycles.closure.emission";
inline constexpr auto transparent_bsdf = "psycles.closure.transparent";
inline constexpr auto add_closure = "psycles.closure.add";
inline constexpr auto mix_closure = "psycles.closure.mix";
}// namespace node_type

[[nodiscard]] contract::NodeRegistry make_core_node_registry();

}// namespace psycles::compiler
