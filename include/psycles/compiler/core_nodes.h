#pragma once

#include <psycles/contract/shader_graph.h>

namespace psycles::compiler {

namespace node_type {
inline constexpr auto constant_float = "psycles.constant.float";
inline constexpr auto constant_color = "psycles.constant.color";
inline constexpr auto geometry = "psycles.geometry";
inline constexpr auto texture_coordinate = "psycles.texture_coordinate";
inline constexpr auto uv_map = "psycles.uv_map";
inline constexpr auto mapping = "psycles.vector.mapping";
inline constexpr auto image_texture = "psycles.texture.image";
inline constexpr auto environment_texture =
    "psycles.texture.environment";
inline constexpr auto noise_texture = "psycles.texture.noise";
inline constexpr auto white_noise_texture =
    "psycles.texture.white_noise";
inline constexpr auto checker_texture =
    "psycles.texture.checker";
inline constexpr auto brick_texture = "psycles.texture.brick";
inline constexpr auto magic_texture = "psycles.texture.magic";
inline constexpr auto wave_texture = "psycles.texture.wave";
inline constexpr auto voronoi_texture = "psycles.texture.voronoi";
inline constexpr auto gradient_texture = "psycles.texture.gradient";
inline constexpr auto nishita_sky = "psycles.texture.nishita_sky";
inline constexpr auto hosek_wilkie_sky =
    "psycles.texture.hosek_wilkie_sky";
inline constexpr auto math = "psycles.math.cycles";
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
inline constexpr auto map_range = "psycles.math.map_range";
inline constexpr auto vector_math = "psycles.math.vector";
inline constexpr auto vector_rotate = "psycles.vector.rotate";
inline constexpr auto vector_transform = "psycles.vector.transform";
inline constexpr auto wireframe = "psycles.wireframe";
inline constexpr auto scalar_to_color = "psycles.convert.scalar_to_color";
inline constexpr auto scalar_to_boolean =
    "psycles.convert.scalar_to_boolean";
inline constexpr auto color_to_scalar = "psycles.convert.color_to_scalar";
inline constexpr auto vector_to_scalar = "psycles.convert.vector_to_scalar";
inline constexpr auto vector_to_color = "psycles.convert.vector_to_color";
inline constexpr auto color_to_vector = "psycles.convert.color_to_vector";
inline constexpr auto point_to_vector = "psycles.convert.point_to_vector";
inline constexpr auto float3_to_vector = "psycles.convert.float3_to_vector";
inline constexpr auto vector_to_normal = "psycles.convert.vector_to_normal";
inline constexpr auto normal_to_vector = "psycles.convert.normal_to_vector";
inline constexpr auto object_info = "psycles.object_info";
inline constexpr auto particle_info = "psycles.particle_info";
inline constexpr auto hair_info = "psycles.hair_info";
inline constexpr auto light_path = "psycles.light_path";
inline constexpr auto light_falloff = "psycles.light_falloff";
inline constexpr auto layer_weight = "psycles.layer_weight";
inline constexpr auto fresnel = "psycles.fresnel";
inline constexpr auto ambient_occlusion = "psycles.ambient_occlusion";
inline constexpr auto mix_float = "psycles.value.mix_float";
inline constexpr auto mix_vector = "psycles.value.mix_vector";
inline constexpr auto mix_vector_nonuniform =
    "psycles.value.mix_vector_nonuniform";
inline constexpr auto legacy_mix_color = "psycles.color.mix_legacy";
inline constexpr auto mix_color = "psycles.color.mix";
inline constexpr auto multiply_color = "psycles.color.multiply";
inline constexpr auto hue_saturation = "psycles.color.hue_saturation";
inline constexpr auto invert_color = "psycles.color.invert";
inline constexpr auto gamma_color = "psycles.color.gamma";
inline constexpr auto brightness_contrast =
    "psycles.color.brightness_contrast";
inline constexpr auto blackbody = "psycles.color.blackbody";
inline constexpr auto wavelength = "psycles.color.wavelength";
inline constexpr auto color_ramp = "psycles.color.ramp";
inline constexpr auto rgb_curve = "psycles.color.rgb_curve";
inline constexpr auto separate_color = "psycles.color.separate";
inline constexpr auto combine_color = "psycles.color.combine";
inline constexpr auto separate_xyz = "psycles.vector.separate_xyz";
inline constexpr auto combine_xyz = "psycles.vector.combine_xyz";
inline constexpr auto normal_map = "psycles.normal_map";
inline constexpr auto bump = "psycles.bump";
inline constexpr auto displacement = "psycles.displacement";
inline constexpr auto vertex_color = "psycles.attribute.vertex_color";
inline constexpr auto attribute = "psycles.attribute.named";
inline constexpr auto diffuse_bsdf = "psycles.closure.diffuse";
inline constexpr auto translucent_bsdf = "psycles.closure.translucent";
inline constexpr auto principled_bsdf = "psycles.closure.principled";
inline constexpr auto subsurface_scattering =
    "psycles.closure.subsurface_scattering";
inline constexpr auto glossy_bsdf = "psycles.closure.glossy";
inline constexpr auto metallic_bsdf = "psycles.closure.metallic";
inline constexpr auto sheen_bsdf = "psycles.closure.sheen";
inline constexpr auto hair_bsdf = "psycles.closure.hair";
inline constexpr auto glass_bsdf = "psycles.closure.glass";
inline constexpr auto refraction_bsdf = "psycles.closure.refraction";
inline constexpr auto emission = "psycles.closure.emission";
inline constexpr auto background = "psycles.closure.background";
inline constexpr auto transparent_bsdf = "psycles.closure.transparent";
inline constexpr auto null_closure = "psycles.closure.null";
inline constexpr auto add_closure = "psycles.closure.add";
inline constexpr auto mix_closure = "psycles.closure.mix";
inline constexpr auto volume_absorption =
    "psycles.volume.absorption";
inline constexpr auto volume_scatter = "psycles.volume.scatter";
inline constexpr auto volume_coefficients =
    "psycles.volume.coefficients";
inline constexpr auto volume_emission = "psycles.volume.emission";
inline constexpr auto principled_volume =
    "psycles.volume.principled";
inline constexpr auto null_volume = "psycles.volume.null";
inline constexpr auto add_volume = "psycles.volume.add";
inline constexpr auto mix_volume = "psycles.volume.mix";
}// namespace node_type

[[nodiscard]] contract::NodeRegistry make_core_node_registry();

}// namespace psycles::compiler
