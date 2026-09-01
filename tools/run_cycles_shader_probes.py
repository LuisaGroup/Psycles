#!/usr/bin/env python3
"""Run canonical shader probes through Cycles and Psycles-Luisa.

The runner creates each .blend, renders the authoritative Cycles multilayer
EXR, exports the unchanged node graph, renders it with Psycles, and writes the
linear pass differential report.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import os


_ALL_PROBES = (
    "add_shader_emission",
    "ambient_occlusion_matrix",
    "area_light",
    "area_light_ellipse",
    "area_light_spread",
    "background_world",
    "background_world_zero",
    "background_world_linked",
    "background_world_mix",
    "blackbody_matrix",
    "bump_matrix",
    "bump_nested_matrix",
    "bump_surface",
    "brightness_contrast",
    "brick_texture",
    "brick_texture_constants",
    "camera_blackman_harris_filter",
    "camera_data",
    "camera_dof_disk",
    "checker_texture_matrix",
    "clamp",
    "color_ramp_alpha_modes",
    "color_ramp_modes",
    "color_ramp_rgb",
    "combine_color_modes",
    "diffuse_bsdf_matrix",
    "diffuse_surface",
    "dynamic_mix_shader",
    "emission_surface",
    "environment_texture_projection_modes",
    "environment_texture_sampling_modes",
    "environment_texture_world_default",
    "float_curve_matrix",
    "fresnel_matrix",
    "flat_light_distribution",
    "gamma_color",
    "glass_transport",
    "geometry_attribute_outputs",
    "geometry_displacement_methods",
    "geometry_position_color_conversion",
    "geometry_pointiness",
    "glossy_bsdf_matrix",
    "gradient_matrix",
    "gradient_mapping_constant_fold",
    "gradient_opcode_matrix",
    "gradient_spherical",
    "hair_bsdf_matrix",
    "hosek_wilkie_diffuse_transport",
    "hue_saturation_value",
    "image_texture_srgb",
    "image_texture_node_mapping",
    "image_texture_sampling_modes",
    "image_texture_projection_modes",
    "indirect_diffuse",
    "indirect_principled",
    "info_nodes_matrix",
    "integrator_clamp_direct",
    "invert_color_matrix",
    "legacy_separate_combine_matrix",
    "layer_weight_matrix",
    "light_falloff_matrix",
    "light_path_matrix",
    "map_range_matrix",
    "math_edge_cases",
    "math_operations",
    "mapping_modes",
    "magic_texture_matrix",
    "metallic_bsdf_matrix",
    "mix_color_modes",
    "mix_color_edge_cases",
    "mix_data_types",
    "mix_rgb_legacy_modes",
    "mix_shader_emission",
    "negative_scale_surface",
    "nishita_diffuse_transport",
    "node_group_color",
    "noise_bump_object",
    "noise_color_3d",
    "noise_factor_2d",
    "noise_fbm_matrix",
    "noise_hetero_terrain_matrix",
    "noise_hybrid_multifractal_matrix",
    "noise_multifractal_matrix",
    "noise_ridged_multifractal_matrix",
    "normal_node_matrix",
    "normal_map_surface",
    "normal_map_matrix",
    "normal_map_named_uv_matrix",
    "normal_map_displacement_matrix",
    "particle_random_instances",
    "particle_random_nonparticle",
    "point_light",
    "point_light_light_path",
    "point_light_nodes",
    "point_light_shadow_light_path",
    "point_light_shadow_limit",
    "point_light_soft_disk",
    "point_light_soft_sphere",
    "transmission_light_path_visibility",
    "principled_alpha_surface",
    "principled_bump_glossy",
    "principled_coat_surface",
    "principled_emission",
    "principled_emission_layers",
    "principled_sheen_surface",
    "principled_surface",
    "principled_thin_wall_surface",
    "principled_transmission_surface",
    "random_walk_transport",
    "refraction_bsdf_matrix",
    "rgb_emission",
    "rgb_curve_matrix",
    "rgb_to_bw",
    "separate_color_modes",
    "sheen_bsdf_matrix",
    "spot_light",
    "spot_light_soft",
    "sun_light",
    "sun_light_disk",
    "svm_color_constant_fold",
    "svm_color_pipeline",
    "svm_combsep_color_constant_fold",
    "svm_combsep_color_pipeline",
    "svm_legacy_mix_constant_matrix",
    "svm_legacy_mix_matrix",
    "svm_math_constant_fold",
    "svm_math_dedup",
    "svm_mix_closure_fold",
    "svm_modern_mix_color_matrix",
    "svm_modern_mix_constant_matrix",
    "svm_modern_mix_data_matrix",
    "svm_modern_mix_fold_edges",
    "svm_modern_mix_import_chain",
    "svm_sepcomb_vector_constant_fold",
    "svm_sepcomb_vector_pipeline",
    "svm_vector_rotate_matrix",
    "svm_vector_transform_matrix",
    "svm_vector_transform_zero_normal",
    "svm_vector_transform_no_object_from_camera",
    "svm_vector_transform_no_object_to_camera",
    "svm_bump_constant_fold",
    "svm_geometry_attributes",
    "svm_geometry_bump_offsets",
    "svm_vertex_color",
    "svm_wireframe_matrix",
    "svm_wireframe_bump",
    "thin_film_surface",
    "transparent_mix",
    "transparent_data_pass",
    "translucent_bsdf_matrix",
    "translucent_surface",
    "triangle_light_solid_angle",
    "texture_coordinate_object_transform",
    "svm_texture_coordinate",
    "svm_texture_coordinate_background_generated",
    "svm_texture_coordinate_background_reflection",
    "value_emission",
    "vector_curve_matrix",
    "volume_emission_transport",
    "vector_math_matrix",
    "voronoi_texture_distance",
    "voronoi_texture_edges",
    "voronoi_texture_fractal",
    "wavelength_matrix",
    "wave_texture_distortion",
    "wave_texture_modes",
    "white_noise_constant_fold",
    "white_noise_dimensions",
    "white_noise_matrix",
)

_REPORT_PASSES = (
    "Combined",
    "Normal",
    "DiffCol",
    "GlossCol",
    "TransCol",
    "DiffDir",
    "DiffInd",
    "GlossDir",
    "GlossInd",
    "TransDir",
    "TransInd",
    "Emit",
    "Env",
)

# These gates cover renderer defects that the canonical probe was introduced
# to catch. Energy gates are appropriate when multiple unbiased direction
# mappings are valid. A probe may additionally require a relative-RMSE gate
# when its purpose is to preserve Cycles' exact sample mapping.
_PROBE_RATIO_GATES = {
    "ambient_occlusion_matrix": {
        # The probe shares Cycles' branched Sobol dimensions and therefore
        # compares the full spatial estimator, not merely expected energy.
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "geometry_attribute_outputs": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "geometry_displacement_methods": {
        # This probe deliberately includes silhouette pixels, where Cycles
        # CPU and the Luisa raster-independent path can differ at the last
        # few filter samples. The energy gate remains tight enough to reject
        # either true-displacement mode being evaluated as bump-only.
        "Combined": (0.9995, 1.0005),
        "DiffCol": (0.9995, 1.0005),
        "Normal": (0.9995, 1.0005),
    },
    "normal_map_displacement_matrix": {
        # Combined/Emit encode the signed Normal Map result directly. Normal
        # is itself signed and has a small luminance mean, so its ratio needs
        # the same stable signed-vector envelope as the displacement probe.
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
        "Normal": (0.9995, 1.0005),
    },
    "normal_node_matrix": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "geometry_position_color_conversion": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "geometry_pointiness": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "texture_coordinate_object_transform": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "image_texture_node_mapping": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "image_texture_projection_modes": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "magic_texture_matrix": {
        "Combined": (0.9999, 1.0001),
        "Emit": (0.9999, 1.0001),
    },
    "wave_texture_distortion": {
        # Cycles CPU and Cycles HIP differ by 3.41e-5 relative RMSE on this
        # deliberately high-frequency saw/triangle matrix. Keep one oracle
        # (Cycles CPU) for every Luisa backend, while allowing only the same
        # device-level float32 envelope rather than hiding structural errors.
        "Combined": (0.99997, 1.00003),
        "Emit": (0.99997, 1.00003),
    },
    "wave_texture_modes": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "volume_emission_transport": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "voronoi_texture_distance": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "voronoi_texture_fractal": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "camera_blackman_harris_filter": {
        "Combined": (0.9995, 1.0005),
    },
    "camera_dof_disk": {
        "Combined": (0.9995, 1.0005),
    },
    "environment_texture_projection_modes": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "environment_texture_sampling_modes": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "environment_texture_world_default": {
        "Combined": (0.99999, 1.00001),
        "Env": (0.99999, 1.00001),
    },
    "fresnel_matrix": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "light_falloff_matrix": {
        # Six equal-area cells make any selector, smoothing, or linked-input
        # error structural rather than statistically sparse.
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "light_path_matrix": {
        # This is a deterministic primary-ray state matrix. Energy and image
        # shape should agree; only isolated reconstruction-boundary pixels
        # receive a small backend-independent float envelope.
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "glass_transport": {
        "Combined": (0.9999, 1.0001),
        "GlossCol": (0.9999, 1.0001),
        "GlossDir": (0.9999, 1.0001),
        "TransCol": (0.9999, 1.0001),
        "TransDir": (0.9999, 1.0001),
    },
    "glossy_bsdf_matrix": {
        "Combined": (0.9999, 1.0001),
        "GlossCol": (0.99999, 1.00001),
        "GlossDir": (0.9999, 1.0001),
        "Normal": (0.99999, 1.00001),
    },
    "indirect_principled": {
        "DiffInd": (0.98, 1.02),
        "GlossInd": (0.98, 1.02),
    },
    "nishita_diffuse_transport": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "hosek_wilkie_diffuse_transport": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "point_light_light_path": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "point_light_shadow_light_path": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "point_light_shadow_limit": {
        "Combined": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
    },
    "transmission_light_path_visibility": {
        "Combined": (0.99999, 1.00001),
        "TransDir": (0.99999, 1.00001),
    },
    "principled_alpha_surface": {
        "Combined": (0.999999, 1.000001),
        "Env": (0.999999, 1.000001),
    },
    "principled_coat_surface": {
        "Combined": (0.99999, 1.00001),
        "DiffDir": (0.99999, 1.00001),
        "GlossDir": (0.99999, 1.00001),
        "DiffCol": (0.99999, 1.00001),
        "GlossCol": (0.99999, 1.00001),
        "Normal": (0.99999, 1.00001),
    },
    "principled_transmission_surface": {
        "Combined": (0.99999, 1.00001),
        "DiffCol": (0.99999, 1.00001),
        "GlossCol": (0.99999, 1.00001),
        "TransCol": (0.99999, 1.00001),
        "TransDir": (0.99999, 1.00001),
        "Normal": (0.99999, 1.00001),
    },
    "principled_thin_wall_surface": {
        # This matrix preserves Cycles' event-level bump-shadowing rule for
        # selected thin-transmission samples. Diffuse Direct is deliberately
        # excluded: one zero-reference, zero-DiffCol pixel makes that divided
        # AOV unstable without changing Combined or any closure-color pass.
        "Combined": (0.9999, 1.0001),
        "DiffCol": (0.99999, 1.00001),
        "GlossCol": (0.99999, 1.00001),
        "TransCol": (0.99999, 1.00001),
        "GlossDir": (0.99998, 1.00002),
        "TransDir": (0.99998, 1.00002),
        "Normal": (0.9999, 1.0001),
    },
    "thin_film_surface": {
        # A single zero-angle Sun makes every nonzero pass a deterministic
        # closure evaluation rather than a Monte Carlo energy estimate. Keep
        # a small backend-transcendental envelope for the spectral Fresnel
        # calculation while rejecting any missing or misclassified lobe.
        # TransDir/TransInd are deliberately excluded: the reference signal
        # is numerical zero (about 1e-8/2e-5), so ratios are not meaningful.
        "Combined": (0.9995, 1.0005),
        "GlossCol": (0.9995, 1.0005),
        "GlossDir": (0.9995, 1.0005),
        "TransCol": (0.9995, 1.0005),
        "Normal": (0.9995, 1.0005),
    },
    "metallic_bsdf_matrix": {
        # The zero-angle Sun turns all nonzero passes into deterministic
        # closure evaluation. These gates reject an incorrect Fresnel model,
        # distribution, anisotropic axis, film branch, or input saturation
        # in any of the 16 equal-area cells while allowing float32 backend
        # transcendental differences.
        "Combined": (0.9995, 1.0005),
        "GlossCol": (0.9995, 1.0005),
        "GlossDir": (0.9995, 1.0005),
        "Normal": (0.9995, 1.0005),
    },
    "sheen_bsdf_matrix": {
        # The zero-angle Sun produces deterministic direct evaluation. The
        # matrix also makes the deliberately different Cycles pass policy
        # observable: Microfiber is diffuse while Ashikhmin is glossy.
        "Combined": (0.9995, 1.0005),
        "DiffCol": (0.9995, 1.0005),
        "DiffDir": (0.9995, 1.0005),
        "GlossCol": (0.9995, 1.0005),
        "GlossDir": (0.9995, 1.0005),
        "Normal": (0.9995, 1.0005),
    },
    "refraction_bsdf_matrix": {
        "Combined": (0.99998, 1.00002),
        "GlossCol": (0.0, 0.0),
        "TransCol": (0.99999, 1.00001),
        "TransDir": (0.99998, 1.00002),
        "Normal": (0.99999, 1.00001),
    },
    "principled_emission": {
        "Combined": (0.999999, 1.000001),
        "Emit": (0.999999, 1.000001),
    },
    "principled_emission_layers": {
        "Combined": (0.99999, 1.00001),
        "Emit": (0.99999, 1.00001),
    },
    "principled_sheen_surface": {
        "Combined": (0.99999, 1.00001),
        "DiffDir": (0.99999, 1.00001),
        "DiffCol": (0.99999, 1.00001),
        "Normal": (0.99999, 1.00001),
    },
    "triangle_light_solid_angle": {
        "Combined": (0.995, 1.005),
        "DiffDir": (0.995, 1.005),
    },
}

_PROBE_RELATIVE_RMSE_GATES = {
    "ambient_occlusion_matrix": {
        "Combined": 0.000001,
        "Emit": 0.000001,
    },
    "geometry_attribute_outputs": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "geometry_displacement_methods": {
        "Combined": 0.00005,
        "DiffCol": 0.000001,
        "Normal": 0.00005,
    },
    "normal_map_displacement_matrix": {
        # Reject collapsing true displacement into bump, merging ORIGINAL
        # and DISPLACED Mikk frames, losing named-UV handedness, or ignoring
        # the OpenGL/DirectX convention on any Luisa backend.
        "Combined": 0.00005,
        "Emit": 0.00005,
        "Normal": 0.00005,
    },
    "normal_node_matrix": {
        # At 64 spp, four isolated material-boundary pixels differ by exactly
        # one camera sample (maximum 1/64); every cell interior is exact and
        # the mean-energy error is below 4e-7. Bound the global contribution
        # here and pair it with the p99 gate below so no authored Normal-node
        # case can hide behind those ray/triangle edge ties.
        "Combined": 0.0001,
        "Emit": 0.0001,
    },
    "geometry_position_color_conversion": {
        "Combined": 0.000001,
        "Emit": 0.000001,
    },
    "geometry_pointiness": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "texture_coordinate_object_transform": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "image_texture_node_mapping": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "image_texture_projection_modes": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "magic_texture_matrix": {
        "Combined": 0.0001,
        "Emit": 0.0001,
    },
    "wave_texture_distortion": {
        "Combined": 0.00006,
        "Emit": 0.00006,
    },
    "wave_texture_modes": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "volume_emission_transport": {
        "Combined": 0.000001,
        "Emit": 0.000001,
    },
    "voronoi_texture_distance": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "voronoi_texture_edges": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "voronoi_texture_fractal": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "camera_blackman_harris_filter": {
        "Combined": 0.0005,
    },
    "camera_dof_disk": {
        "Combined": 0.0005,
    },
    "environment_texture_projection_modes": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "environment_texture_sampling_modes": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "environment_texture_world_default": {
        "Combined": 0.00001,
        "Env": 0.00001,
    },
    "fresnel_matrix": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "light_falloff_matrix": {
        "Combined": 0.0001,
        "Emit": 0.0001,
    },
    "light_path_matrix": {
        "Combined": 0.00005,
        "Emit": 0.00005,
    },
    "glass_transport": {
        "Combined": 0.0002,
        "GlossCol": 0.0001,
        "GlossDir": 0.0001,
        "TransCol": 0.0001,
        "TransDir": 0.0002,
    },
    "glossy_bsdf_matrix": {
        "Combined": 0.0002,
        "GlossCol": 0.00001,
        "GlossDir": 0.0002,
        "Normal": 0.00001,
    },
    "nishita_diffuse_transport": {
        "Combined": 0.0005,
        "DiffDir": 0.0005,
    },
    "point_light_light_path": {
        "Combined": 0.000005,
        "DiffDir": 0.000005,
    },
    "point_light_shadow_light_path": {
        "Combined": 0.000005,
        "DiffDir": 0.000005,
    },
    "point_light_shadow_limit": {
        "Combined": 0.000005,
        "DiffDir": 0.000005,
    },
    "transmission_light_path_visibility": {
        "Combined": 0.00001,
        "TransDir": 0.00001,
    },
    "principled_alpha_surface": {
        "Combined": 0.000001,
        "Env": 0.000001,
    },
    "principled_coat_surface": {
        "Combined": 0.0001,
        "DiffDir": 0.0001,
        "GlossDir": 0.0001,
        "DiffCol": 0.0001,
        "GlossCol": 0.0001,
        "Normal": 0.0001,
    },
    "principled_transmission_surface": {
        "Combined": 0.000005,
        "DiffCol": 0.0000001,
        "GlossCol": 0.000001,
        "TransCol": 0.0000001,
        "TransDir": 0.000002,
        "Normal": 0.0000001,
    },
    "principled_thin_wall_surface": {
        "Combined": 0.0001,
        "DiffCol": 0.00002,
        "GlossCol": 0.00002,
        "TransCol": 0.00002,
        "GlossDir": 0.00003,
        "TransDir": 0.00005,
        "Normal": 0.00002,
    },
    "refraction_bsdf_matrix": {
        "Combined": 0.00001,
        "GlossCol": 0.0,
        "TransCol": 0.00005,
        "TransDir": 0.00001,
        "Normal": 0.0000001,
    },
    "principled_emission": {
        "Combined": 0.000001,
        "Emit": 0.000001,
    },
    "principled_emission_layers": {
        "Combined": 0.00001,
        "Emit": 0.00001,
    },
    "principled_sheen_surface": {
        "Combined": 0.000002,
        "DiffDir": 0.000002,
        "DiffCol": 0.0000001,
        "Normal": 0.000001,
    },
    "triangle_light_solid_angle": {
        "Combined": 0.005,
        "DiffDir": 0.005,
    },
}

# A 99th-percentile error gate is appropriate for equal-area material
# matrices whose purpose is closure evaluation rather than camera sampling.
# Every authored case in these matrices occupies at least 1/16 of the image,
# so a structural error in any case necessarily reaches the 99th percentile.
# In contrast, isolated ray/triangle boundary choices occupy less than 1% and
# must not dominate the otherwise deterministic direct-light comparison.
# Normalize by Cycles RMS so the gate remains independent of exposure and Sun
# energy.
_PROBE_NORMALIZED_P99_RMSE_GATES = {
    "light_falloff_matrix": {
        "Combined": 0.000001,
        "Emit": 0.000001,
    },
    "normal_node_matrix": {
        "Combined": 0.000001,
        "Emit": 0.000001,
    },
    "thin_film_surface": {
        "Combined": 0.0001,
        "GlossCol": 0.0001,
        "GlossDir": 0.0001,
        "TransCol": 0.0001,
        "Normal": 0.0001,
    },
    "metallic_bsdf_matrix": {
        "Combined": 0.0001,
        "GlossCol": 0.0001,
        "GlossDir": 0.0001,
        "Normal": 0.0001,
    },
    "sheen_bsdf_matrix": {
        "Combined": 0.0001,
        "DiffCol": 0.0001,
        "DiffDir": 0.0001,
        "GlossCol": 0.0001,
        "GlossDir": 0.0001,
        "Normal": 0.0001,
    },
    "hair_bsdf_matrix": {
        # Cycles CPU versus HIP itself reaches 7.04e-4 relative RMSE in the
        # narrow Hair Reflection peak on this matrix. Keep a 1e-3 normalized
        # p99 envelope: it covers backend trig/triangle-edge rounding while
        # rejecting the former wrong dPdv contract by three orders of
        # magnitude (0.93 Combined, 2.09 GlossDir, 2.67 TransDir).
        "Combined": 0.001,
        "GlossCol": 0.001,
        "GlossDir": 0.001,
        "TransCol": 0.001,
        "TransDir": 0.001,
        "Normal": 0.001,
    },
}

# A non-finite Cycles reference sample is retained as oracle evidence but must
# never excuse a Psycles NaN/Inf. This is intentionally per probe: older
# reports predate the source-attributed counters and remain readable.
_PROBE_ACTUAL_INVALID_PIXEL_GATES = {
    "light_falloff_matrix": {
        "Combined": 0,
        "Emit": 0,
    },
    "hair_bsdf_matrix": {
        "Combined": 0,
        "GlossCol": 0,
        "GlossDir": 0,
        "TransCol": 0,
        "TransDir": 0,
        "Normal": 0,
    },
    "sheen_bsdf_matrix": {
        "Combined": 0,
        "DiffCol": 0,
        "DiffDir": 0,
        "GlossCol": 0,
        "GlossDir": 0,
        "Normal": 0,
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=pathlib.Path, required=True)
    parser.add_argument(
        "--psycles-render", type=pathlib.Path, required=True
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--backend", default="fallback")
    parser.add_argument("--cycles-device", default="CPU")
    parser.add_argument("--cycles-device-name", default="")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument(
        "probes",
        nargs="*",
        choices=_ALL_PROBES,
        default=list(_ALL_PROBES),
    )
    result = parser.parse_args()
    if result.width <= 0 or result.height <= 0 or result.samples <= 0:
        parser.error("width, height, and samples must be positive")
    return result


def _run(command: list[str], environment: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def _psycles_environment(
    backend: str,
    environment: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Return the explicit backend contract for a Psycles probe process."""
    if backend != "vk":
        return None
    result = dict(os.environ if environment is None else environment)
    # A Vulkan shader probe is a native XIR -> SPIR-V canary. Assign rather
    # than setdefault: a stale parent-shell override must not silently turn a
    # correctness run into the legacy DXC route.
    result["LUISA_VULKAN_USE_XIR"] = "1"
    result["LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV"] = "1"
    result["LUISA_VULKAN_DISABLE_DXC"] = "1"
    return result


def _cycles_golden_command(
    blender: str,
    blend: pathlib.Path,
    golden_script: pathlib.Path,
    cycles: pathlib.Path,
    *,
    width: int,
    height: int,
    samples: int,
    device: str,
    device_name: str,
) -> list[str]:
    # Do not pass a seed override. The canonical probe .blend and the
    # exported Psycles scene must consume the same scene-owned seed.
    return [
        blender,
        str(blend),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(golden_script),
        "--",
        str(cycles),
        str(width),
        str(height),
        str(samples),
        "--cycles-device",
        device,
        "--device-name",
        device_name,
    ]


def _comparison_command(
    python: str,
    compare_script: pathlib.Path,
    cycles: pathlib.Path,
    report: pathlib.Path,
    triptych_dir: pathlib.Path,
    psycles_exr: pathlib.Path,
    cycles_metadata: pathlib.Path,
    scene_metadata: pathlib.Path,
) -> list[str]:
    # Comparison is a standalone Python/OIIO/Pillow tool. It must not inherit
    # Blender's private Python environment, which need not contain Pillow.
    return [
        python,
        str(compare_script),
        str(cycles),
        str(report),
        "--triptych-dir",
        str(triptych_dir),
        "--reference-metadata",
        str(cycles_metadata),
        "--actual-metadata",
        str(scene_metadata),
        *(
            f"{pass_name}={psycles_exr}"
            for pass_name in _REPORT_PASSES
        ),
    ]


def _probe_gate_failures(
    probe: str,
    report_payload: dict[str, object],
) -> list[str]:
    ratio_gates = _PROBE_RATIO_GATES.get(probe, {})
    rmse_gates = _PROBE_RELATIVE_RMSE_GATES.get(probe, {})
    p99_gates = _PROBE_NORMALIZED_P99_RMSE_GATES.get(probe, {})
    invalid_gates = _PROBE_ACTUAL_INVALID_PIXEL_GATES.get(probe, {})
    passes = report_payload.get("passes")
    if not isinstance(passes, dict):
        return [f"{probe}: report has no pass dictionary"]
    failures: list[str] = []
    for pass_name, (minimum, maximum) in ratio_gates.items():
        pass_report = passes.get(pass_name)
        if not isinstance(pass_report, dict):
            failures.append(f"{probe}: missing {pass_name}")
            continue
        ratio = pass_report.get("luminance_mean_ratio")
        if not isinstance(ratio, (float, int)):
            failures.append(
                f"{probe}: {pass_name} has no numeric energy ratio"
            )
        elif not minimum <= float(ratio) <= maximum:
            failures.append(
                f"{probe}: {pass_name} energy ratio {float(ratio):.6f} "
                f"is outside [{minimum:.6f}, {maximum:.6f}]"
            )
    for pass_name, maximum in rmse_gates.items():
        pass_report = passes.get(pass_name)
        if not isinstance(pass_report, dict):
            if pass_name not in ratio_gates:
                failures.append(f"{probe}: missing {pass_name}")
            continue
        relative_rmse = pass_report.get("relative_rmse")
        if not isinstance(relative_rmse, (float, int)):
            failures.append(
                f"{probe}: {pass_name} has no numeric relative RMSE"
            )
        elif float(relative_rmse) > maximum:
            failures.append(
                f"{probe}: {pass_name} relative RMSE "
                f"{float(relative_rmse):.6f} exceeds {maximum:.6f}"
            )
    for pass_name, maximum in p99_gates.items():
        pass_report = passes.get(pass_name)
        if not isinstance(pass_report, dict):
            if pass_name not in ratio_gates and pass_name not in rmse_gates:
                failures.append(f"{probe}: missing {pass_name}")
            continue
        p99_rmse = pass_report.get("p99_pixel_rmse")
        cycles_rms = pass_report.get("cycles_rms")
        if not isinstance(p99_rmse, (float, int)):
            failures.append(
                f"{probe}: {pass_name} has no numeric p99 pixel RMSE"
            )
            continue
        if not isinstance(cycles_rms, (float, int)):
            failures.append(
                f"{probe}: {pass_name} has no numeric Cycles RMS"
            )
            continue
        normalized_p99_rmse = float(p99_rmse) / max(
            abs(float(cycles_rms)), 1.0e-20
        )
        if normalized_p99_rmse > maximum:
            failures.append(
                f"{probe}: {pass_name} normalized p99 pixel RMSE "
                f"{normalized_p99_rmse:.6f} exceeds {maximum:.6f}"
            )
    for pass_name, maximum in invalid_gates.items():
        pass_report = passes.get(pass_name)
        if not isinstance(pass_report, dict):
            continue
        actual_invalid = pass_report.get("actual_invalid_pixels")
        if not isinstance(actual_invalid, int):
            failures.append(
                f"{probe}: {pass_name} has no integer Psycles invalid-pixel "
                "count"
            )
        elif actual_invalid > maximum:
            failures.append(
                f"{probe}: {pass_name} has {actual_invalid} non-finite "
                f"Psycles pixels, maximum is {maximum}"
            )
    return failures


def _main() -> int:
    arguments = _arguments()
    root = pathlib.Path(__file__).resolve().parent.parent
    create_script = root / "tools/create_cycles_shader_probe.py"
    golden_script = root / "tools/render_cycles_golden.py"
    export_script = root / "tools/export_psycles_scene.py"
    compare_script = root / "tools/compare_cycles.py"
    blender = str(arguments.blender.resolve())
    renderer = str(arguments.psycles_render.resolve())
    output_root = arguments.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for probe in arguments.probes:
        probe_root = output_root / probe
        probe_root.mkdir(parents=True, exist_ok=True)
        blend = probe_root / f"{probe}.blend"
        cycles = probe_root / f"{probe}-cycles.exr"
        bundle = probe_root / "export"
        preview = probe_root / f"{probe}-psycles.ppm"
        stem = preview.with_suffix("")
        psycles_exr = stem.with_suffix(".exr")
        report = probe_root / f"{probe}-diff.json"
        try:
            _run_environment = _psycles_environment(arguments.backend)
            _run(
                [
                    blender,
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(create_script),
                    "--",
                    str(blend),
                    probe,
                ]
            )
            _run(
                _cycles_golden_command(
                    blender,
                    blend,
                    golden_script,
                    cycles,
                    width=arguments.width,
                    height=arguments.height,
                    samples=arguments.samples,
                    device=arguments.cycles_device,
                    device_name=arguments.cycles_device_name,
                )
            )
            _run(
                [
                    blender,
                    str(blend),
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(export_script),
                    "--",
                    str(bundle),
                ]
            )
            _run(
                [
                    renderer,
                    str(bundle),
                    str(preview),
                    arguments.backend,
                    str(arguments.width),
                    str(arguments.height),
                    str(arguments.samples),
                ],
                environment=_run_environment,
            )
            _run(
                _comparison_command(
                    sys.executable,
                    compare_script,
                    cycles,
                    report,
                    probe_root / "triptychs",
                    psycles_exr,
                    cycles.with_suffix(".json"),
                    bundle / "scene.json",
                )
            )
            gate_failures = _probe_gate_failures(
                probe,
                json.loads(report.read_text(encoding="utf-8")),
            )
            if gate_failures:
                failures.append(probe)
                for failure in gate_failures:
                    print(f"probe gate failed: {failure}", file=sys.stderr)
        except subprocess.CalledProcessError:
            failures.append(probe)
            print(f"probe failed: {probe}", file=sys.stderr)

    if failures:
        print(
            "failed probes: " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    print(
        f"completed {len(arguments.probes)} shader probes: {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
