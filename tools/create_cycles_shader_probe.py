"""Create canonical Blender scenes for focused Cycles shader-node probes.

Usage:

    blender --background --python create_cycles_shader_probe.py -- \
        output.blend probe-name

The generated .blend is an input to both ``render_cycles_golden.py`` and
``export_psycles_scene.py``. Probe scenes contain no baked shader result.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable
from typing import Any

import bpy


_TOOL_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(_TOOL_DIRECTORY) not in sys.path:
    # Blender's --python and runpy do not share a reliable script-directory
    # import contract, so make the sibling implementation package explicit.
    sys.path.insert(0, str(_TOOL_DIRECTORY))

from cycles_shader_probe import (  # noqa: E402
    closures,
    environment_inputs,
    geometry_inputs,
    lights_camera,
    magic_inputs,
    procedural_textures,
    refraction_closures,
    subsurface_closures,
    support,
    texture_inputs,
    values,
    volume_closures,
    voronoi_inputs,
    wave_inputs,
)


_PROBES: dict[str, Callable[[Any], None]] = {
    "add_shader_emission": closures._add_shader_emission,
    "area_light": lights_camera._area_light,
    "area_light_ellipse": lights_camera._area_light_ellipse,
    "area_light_spread": lights_camera._area_light_spread,
    "camera_blackman_harris_filter": lights_camera._camera_blackman_harris_filter,
    "camera_dof_disk": lights_camera._camera_dof_disk,
    "flat_light_distribution": lights_camera._flat_light_distribution,
    "triangle_light_solid_angle": lights_camera._triangle_light_solid_angle,
    "background_world": values._background_world,
    "blackbody_matrix": texture_inputs._blackbody_matrix,
    "bump_matrix": closures._bump_matrix,
    "bump_nested_matrix": closures._bump_nested_matrix,
    "bump_surface": closures._bump_surface,
    "brightness_contrast": values._brightness_contrast,
    "brick_texture": procedural_textures._brick_texture,
    "brick_texture_constants": procedural_textures._brick_texture_constants,
    "checker_texture_matrix": texture_inputs._checker_texture_matrix,
    "clamp": values._clamp,
    "color_ramp_alpha_modes": texture_inputs._color_ramp_alpha_modes,
    "color_ramp_modes": texture_inputs._color_ramp_modes,
    "color_ramp_rgb": texture_inputs._color_ramp_rgb,
    "combine_color_modes": values._combine_color_modes,
    "diffuse_bsdf_matrix": closures._diffuse_bsdf_matrix,
    "diffuse_surface": closures._diffuse_surface,
    "emission_surface": lights_camera._emission_surface,
    "environment_texture_projection_modes": (
        environment_inputs._environment_texture_projection_modes
    ),
    "environment_texture_sampling_modes": (
        environment_inputs._environment_texture_sampling_modes
    ),
    "environment_texture_world_default": (
        environment_inputs._environment_texture_world_default
    ),
    "fresnel_matrix": texture_inputs._fresnel_matrix,
    "gamma_color": values._gamma_color,
    "glass_transport": closures._glass_transport,
    "geometry_attribute_outputs": (
        geometry_inputs._geometry_attribute_outputs
    ),
    "geometry_position_color_conversion": (
        geometry_inputs._geometry_position_color_conversion
    ),
    "geometry_pointiness": geometry_inputs._geometry_pointiness,
    "gradient_spherical": texture_inputs._gradient_spherical,
    "gradient_matrix": values._gradient_matrix,
    "hosek_wilkie_diffuse_transport": (
        closures._hosek_wilkie_diffuse_transport
    ),
    "hue_saturation_value": values._hue_saturation_value,
    "image_texture_srgb": texture_inputs._image_texture_srgb,
    "image_texture_sampling_modes": texture_inputs._image_texture_sampling_modes,
    "image_texture_projection_modes": texture_inputs._image_texture_projection_modes,
    "indirect_diffuse": closures._indirect_diffuse,
    "indirect_principled": closures._indirect_principled,
    "integrator_clamp_direct": lights_camera._integrator_clamp_direct,
    "invert_color_matrix": texture_inputs._invert_color_matrix,
    "legacy_separate_combine_matrix": values._legacy_separate_combine_matrix,
    "layer_weight_matrix": texture_inputs._layer_weight_matrix,
    "map_range_matrix": texture_inputs._map_range_matrix,
    "math_edge_cases": values._math_edge_cases,
    "math_operations": values._math_operations,
    "mapping_modes": texture_inputs._mapping_modes,
    "magic_texture_matrix": magic_inputs._magic_texture_matrix,
    "mix_color_modes": values._mix_color_modes,
    "mix_color_edge_cases": values._mix_color_edge_cases,
    "mix_data_types": values._mix_data_types,
    "mix_rgb_legacy_modes": values._mix_rgb_legacy_modes,
    "mix_shader_emission": closures._mix_shader_emission,
    "negative_scale_surface": closures._negative_scale_surface,
    "nishita_diffuse_transport": closures._nishita_diffuse_transport,
    "node_group_color": values._node_group_color,
    "noise_bump_object": texture_inputs._noise_bump_object,
    "noise_color_3d": texture_inputs._noise_color_3d,
    "noise_factor_2d": texture_inputs._noise_factor_2d,
    "noise_fbm_matrix": texture_inputs._noise_fbm_matrix,
    "noise_hetero_terrain_matrix": texture_inputs._noise_hetero_terrain_matrix,
    "noise_hybrid_multifractal_matrix": (
        texture_inputs._noise_hybrid_multifractal_matrix
    ),
    "noise_multifractal_matrix": texture_inputs._noise_multifractal_matrix,
    "noise_ridged_multifractal_matrix": (
        texture_inputs._noise_ridged_multifractal_matrix
    ),
    "normal_map_surface": closures._normal_map_surface,
    "normal_map_matrix": closures._normal_map_matrix,
    "normal_map_named_uv_matrix": closures._normal_map_named_uv_matrix,
    "particle_random_instances": procedural_textures._particle_random_instances,
    "particle_random_nonparticle": procedural_textures._particle_random_nonparticle,
    "point_light": lights_camera._point_light,
    "point_light_light_path": lights_camera._point_light_light_path,
    "point_light_nodes": lights_camera._point_light_nodes,
    "point_light_shadow_light_path": lights_camera._point_light_shadow_light_path,
    "point_light_shadow_limit": lights_camera._point_light_shadow_limit,
    "point_light_soft_disk": lights_camera._point_light_soft_disk,
    "point_light_soft_sphere": lights_camera._point_light_soft_sphere,
    "principled_alpha_surface": closures._principled_alpha_surface,
    "principled_bump_glossy": closures._principled_bump_glossy,
    "principled_coat_surface": closures._principled_coat_surface,
    "principled_emission": closures._principled_emission,
    "principled_emission_layers": closures._principled_emission_layers,
    "principled_sheen_surface": closures._principled_sheen_surface,
    "principled_surface": closures._principled_surface,
    "principled_transmission_surface": closures._principled_transmission_surface,
    "refraction_bsdf_matrix": refraction_closures._refraction_bsdf_matrix,
    "random_walk_transport": (
        subsurface_closures._random_walk_transport
    ),
    "rgb_emission": lights_camera._rgb_emission,
    "rgb_curve_matrix": texture_inputs._rgb_curve_matrix,
    "rgb_to_bw": values._rgb_to_bw,
    "separate_color_modes": values._separate_color_modes,
    "spot_light": lights_camera._spot_light,
    "spot_light_soft": lights_camera._spot_light_soft,
    "sun_light": lights_camera._sun_light,
    "sun_light_disk": lights_camera._sun_light_disk,
    "transparent_mix": closures._transparent_mix,
    "transparent_data_pass": closures._transparent_data_pass,
    "translucent_bsdf_matrix": closures._translucent_bsdf_matrix,
    "translucent_surface": closures._translucent_surface,
    "value_emission": lights_camera._value_emission,
    "volume_emission_transport": (
        volume_closures._volume_emission_transport
    ),
    "vector_math_matrix": texture_inputs._vector_math_matrix,
    "voronoi_texture_distance": voronoi_inputs._voronoi_texture_distance,
    "voronoi_texture_edges": voronoi_inputs._voronoi_texture_edges,
    "voronoi_texture_fractal": voronoi_inputs._voronoi_texture_fractal,
    "wavelength_matrix": texture_inputs._wavelength_matrix,
    "wave_texture_distortion": wave_inputs._wave_texture_distortion,
    "wave_texture_modes": wave_inputs._wave_texture_modes,
    "white_noise_dimensions": procedural_textures._white_noise_dimensions,
}


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit(
            "expected: output.blend probe-name; probes: "
            + ", ".join(sorted(_PROBES))
        )
    output = pathlib.Path(args[0]).resolve()
    probe_name = args[1]
    try:
        create = _PROBES[probe_name]
    except KeyError as error:
        raise SystemExit(
            f"unknown probe {probe_name!r}; probes: "
            + ", ".join(sorted(_PROBES))
        ) from error

    support._clear()
    scene = bpy.context.scene
    scene.name = f"Psycles Probe - {probe_name}"
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    image_settings = scene.render.image_settings
    if hasattr(image_settings, "media_type"):
        # Blender 5.2 separates the media category from the concrete file
        # format; selecting the category makes OPEN_EXR_MULTILAYER writable.
        image_settings.media_type = "MULTI_LAYER_IMAGE"
    image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_depth = "32"
    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 0x51A7
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_light_tree = False
    scene["psycles_probe"] = probe_name
    support._camera(scene)
    if probe_name != "background_world":
        support._world(scene, (0.0, 0.0, 0.0, 1.0), 0.0)
    create(scene)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(f"Created Psycles Cycles probe {probe_name}: {output}")


if __name__ == "__main__":
    _main()
