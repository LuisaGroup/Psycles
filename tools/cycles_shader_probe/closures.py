"""Surface-closure, normal, bump, and indirect-transport probe scenes."""

from __future__ import annotations

from typing import Any

import bpy
from mathutils import Vector

from .support import (
    _bsdf_matrix_sun,
    _input,
    _linked_vector,
    _material,
    _material_matrix,
    _output,
    _plane,
    _sphere,
    _world,
)


def _add_shader_emission(scene: Any) -> None:
    """Cover Add Shader with both, either, and neither closure connected."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = []
    for index, (connect_first, connect_second) in enumerate(
        ((True, True), (True, False), (False, True), (False, False))
    ):
        material, tree, output = _material(
            f"Add Shader Probe {index:02d}"
        )
        first = tree.nodes.new("ShaderNodeEmission")
        first.name = f"Red Emission {index:02d}"
        _input(first, "Color").default_value = (
            0.8,
            0.1,
            0.03,
            1.0,
        )
        _input(first, "Strength").default_value = 0.75
        second = tree.nodes.new("ShaderNodeEmission")
        second.name = f"Blue Emission {index:02d}"
        _input(second, "Color").default_value = (
            0.02,
            0.2,
            0.9,
            1.0,
        )
        _input(second, "Strength").default_value = 1.25
        add = tree.nodes.new("ShaderNodeAddShader")
        add.name = f"Add Shader {index:02d}"
        if connect_first:
            tree.links.new(_output(first, "Emission"), add.inputs[0])
        if connect_second:
            tree.links.new(_output(second, "Emission"), add.inputs[1])
        tree.links.new(
            _output(add, "Shader"), _input(output, "Surface")
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=1,
        name="Add Shader Matrix",
    )


def _mix_shader_emission(scene: Any) -> None:
    """Cover Mix Shader factor clamping and empty closure branches."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (-1.0, True, True),
        (0.0, True, True),
        (0.37, True, True),
        (1.0, True, True),
        (2.0, True, True),
        (0.25, False, True),
        (0.75, True, False),
        (0.5, False, False),
    )
    materials = []
    for index, (
        factor,
        connect_first,
        connect_second,
    ) in enumerate(cases):
        material, tree, output = _material(
            f"Mix Shader Probe {index:02d}"
        )
        first = tree.nodes.new("ShaderNodeEmission")
        first.name = f"First Emission {index:02d}"
        _input(first, "Color").default_value = (
            0.9,
            0.04,
            0.12,
            1.0,
        )
        _input(first, "Strength").default_value = 1.4
        second = tree.nodes.new("ShaderNodeEmission")
        second.name = f"Second Emission {index:02d}"
        _input(second, "Color").default_value = (
            0.03,
            0.72,
            0.2,
            1.0,
        )
        _input(second, "Strength").default_value = 0.65
        factor_node = tree.nodes.new("ShaderNodeValue")
        factor_node.name = f"Mix Factor {index:02d}"
        _output(factor_node, "Value").default_value = factor
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.name = f"Mix Shader {index:02d}"
        tree.links.new(
            _output(factor_node, "Value"), _input(mix, "Fac")
        )
        if connect_first:
            tree.links.new(
                _output(first, "Emission"), mix.inputs[1]
            )
        if connect_second:
            tree.links.new(
                _output(second, "Emission"), mix.inputs[2]
            )
        tree.links.new(
            _output(mix, "Shader"), _input(output, "Surface")
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=8,
        rows=1,
        name="Mix Shader Matrix",
    )


def _dynamic_mix_shader(scene: Any) -> None:
    """Keep Mix Shader's factor linked so Cycles emits both jump nodes."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    material, tree, output = _material("Dynamic Mix Probe")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Geometry"
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.name = "Transparent BSDF"
    _input(transparent, "Color").default_value = (
        0.75,
        0.9,
        0.6,
        1.0,
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (
        0.85,
        0.08,
        0.03,
        1.0,
    )
    _input(emission, "Strength").default_value = 1.2
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "Dynamic Mix Shader"
    tree.links.new(
        _output(geometry, "Backfacing"),
        _input(mix, "Fac"),
    )
    tree.links.new(
        _output(transparent, "BSDF"), mix.inputs[1]
    )
    tree.links.new(
        _output(emission, "Emission"), mix.inputs[2]
    )
    tree.links.new(
        _output(mix, "Shader"), _input(output, "Surface")
    )
    _plane(material)


def _transparent_mix(scene: Any) -> None:
    _world(scene, (0.04, 0.22, 0.7, 1.0), 1.8)
    material, tree, output = _material("Transparent Probe")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.name = "Transparent BSDF"
    _input(transparent, "Color").default_value = (
        0.75,
        0.9,
        0.6,
        1.0,
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (0.85, 0.08, 0.03, 1.0)
    _input(emission, "Strength").default_value = 1.2
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "Mix Shader"
    _input(mix, "Fac").default_value = 0.62
    tree.links.new(_output(transparent, "BSDF"), mix.inputs[1])
    tree.links.new(_output(emission, "Emission"), mix.inputs[2])
    tree.links.new(
        _output(mix, "Shader"), _input(output, "Surface")
    )
    _plane(material)


def _transparent_data_pass(scene: Any) -> None:
    """Verify Normal/DiffCol traversal through a transparent camera hit."""
    _world(scene, (0.3, 0.42, 0.65, 1.0), 0.8)

    foreground, tree, output = _material(
        "Transparent Data Pass Foreground"
    )
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.name = "Transparent BSDF"
    _input(transparent, "Color").default_value = (
        0.72,
        0.9,
        0.61,
        1.0,
    )
    tree.links.new(
        _output(transparent, "BSDF"),
        _input(output, "Surface"),
    )
    bpy.ops.mesh.primitive_plane_add(
        size=8.0,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, 0.0),
        rotation=(0.31, 0.0, 0.0),
    )
    front_plane = bpy.context.object
    front_plane.name = "Transparent Foreground"
    front_plane.data.materials.append(foreground)

    background, tree, output = _material(
        "Transparent Data Pass Background"
    )
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.63,
        0.17,
        0.08,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.27
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    bpy.ops.mesh.primitive_plane_add(
        size=16.0,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, -3.0),
    )
    back_plane = bpy.context.object
    back_plane.name = "Diffuse Background"
    back_plane.data.materials.append(background)
    bpy.context.view_layer.pass_alpha_threshold = 0.5


def _diffuse_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Diffuse Probe")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (0.68, 0.24, 0.09, 1.0)
    _input(diffuse, "Roughness").default_value = 0.43
    tree.links.new(
        _output(diffuse, "BSDF"), _input(output, "Surface")
    )
    _sphere(material)


def _glass_transport(scene: Any) -> None:
    """Exercise smooth and rough Glass reflection/refraction transport."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    _world(scene, (0.04, 0.16, 0.72, 1.0), 1.2)

    glass_cases = (
        ("BECKMANN", 0.0, 1.45, (1.0, 1.0, 1.0)),
        ("GGX", 0.0, 1.5, (0.72, 0.94, 1.0)),
        ("BECKMANN", 0.17320508, 1.5, (1.0, 1.0, 1.0)),
        ("GGX", 0.32, 1.33, (1.0, 0.72, 0.48)),
    )
    glass_materials = []
    for index, (distribution, roughness, ior, color) in enumerate(
        glass_cases
    ):
        material, tree, output = _material(
            f"Glass Transport {index:02d}"
        )
        glass = tree.nodes.new("ShaderNodeBsdfGlass")
        glass.name = f"Glass BSDF {index:02d}"
        glass.distribution = distribution
        _input(glass, "Color").default_value = (*color, 1.0)
        _input(glass, "Roughness").default_value = roughness
        _input(glass, "IOR").default_value = ior
        tree.links.new(
            _output(glass, "BSDF"), _input(output, "Surface")
        )
        glass_materials.append(material)
    foreground = _material_matrix(
        scene,
        glass_materials,
        columns=4,
        rows=1,
        name="Glass Transport Foreground",
    )
    foreground.location.z = 0.0

    background_materials = []
    for index, color in enumerate(
        (
            (0.92, 0.08, 0.025),
            (0.03, 0.72, 0.12),
            (0.92, 0.42, 0.035),
            (0.42, 0.04, 0.88),
        )
    ):
        material, tree, output = _material(
            f"Glass Background {index:02d}"
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Background Emission {index:02d}"
        _input(emission, "Color").default_value = (*color, 1.0)
        _input(emission, "Strength").default_value = 1.8
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        background_materials.append(material)
    background = _material_matrix(
        scene,
        background_materials,
        columns=4,
        rows=1,
        name="Glass Transport Background",
    )
    background.location.z = -1.0


def _diffuse_bsdf_matrix(scene: Any) -> None:
    """Cover Diffuse color allocation, Oren-Nayar, and normal handling."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    _bsdf_matrix_sun(scene, transmission=False)
    cases = (
        ((0.68, 0.24, 0.09), 0.0, None, False),
        ((0.68, 0.24, 0.09), 1.0e-6, None, False),
        ((0.68, 0.24, 0.09), 0.25, None, False),
        ((0.68, 0.24, 0.09), 1.0, None, False),
        ((0.68, 0.24, 0.09), 2.0, None, False),
        ((0.68, 0.24, 0.09), -1.0, None, False),
        ((1.4, 0.2, 2.2), 0.43, None, False),
        ((-0.5, 0.6, 1.2), 0.43, None, False),
        ((-0.5, -0.2, -1.0), 0.43, None, False),
        ((1.0e-6, 1.0e-6, 1.0e-6), 0.43, None, False),
        ((3.0e-5, 0.0, 0.0), 0.43, None, False),
        ((0.7, 0.3, 0.1), 0.43, (0.6, 0.0, 0.8), False),
        ((0.7, 0.3, 0.1), 0.43, (0.3, 0.0, 0.4), False),
        ((0.7, 0.3, 0.1), 0.43, (0.0, 0.0, 0.0), False),
        ((0.7, 0.3, 0.1), 0.43, (1.0, 0.0, 0.0), False),
        ((0.7, 0.3, 0.1), 0.43, None, True),
    )
    materials = []
    backfacing: set[int] = set()
    for index, (color, roughness, normal, backface) in enumerate(
        cases
    ):
        material, tree, output = _material(
            f"Diffuse BSDF Matrix {index:02d}"
        )
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.name = f"Diffuse BSDF {index:02d}"
        tree.links.new(
            _linked_vector(tree, f"Diffuse Color {index:02d}", color),
            _input(diffuse, "Color"),
        )
        roughness_node = tree.nodes.new("ShaderNodeValue")
        roughness_node.name = f"Diffuse Roughness {index:02d}"
        _output(roughness_node, "Value").default_value = roughness
        tree.links.new(
            _output(roughness_node, "Value"),
            _input(diffuse, "Roughness"),
        )
        if normal is not None:
            tree.links.new(
                _linked_vector(
                    tree, f"Diffuse Normal {index:02d}", normal
                ),
                _input(diffuse, "Normal"),
            )
        tree.links.new(
            _output(diffuse, "BSDF"), _input(output, "Surface")
        )
        if backface:
            backfacing.add(index)
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Diffuse BSDF Matrix",
        backfacing=backfacing,
    )
    # This probe isolates closure evaluation. Excluding the coplanar matrix
    # from shadow rays avoids triangle-edge self-intersection on transmitted
    # directions without changing camera visibility.
    surface.visible_shadow = False


def _glossy_bsdf_matrix(scene: Any) -> None:
    """Cover standalone Glossy allocation and constant-Fresnel transport."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    _bsdf_matrix_sun(scene, transmission=False)
    bpy.data.objects["BSDF Matrix Sun"].rotation_euler[1] = 0.92
    cases = (
        ((0.68, 0.24, 0.09), 0.40, 1.00, None),
        ((0.68, 0.24, 0.09), 0.40, 0.37, None),
        ((1.40, 0.20, 2.20), 0.40, 0.63, None),
        ((-0.50, 0.60, 1.20), 0.40, 0.63, None),
        ((-0.50, -0.20, -1.00), 0.40, 0.63, None),
        ((1.0e-6, 1.0e-6, 1.0e-6), 0.40, 1.00, None),
        ((3.0e-5, 0.0, 0.0), 0.40, 1.00, None),
        ((0.70, 0.30, 0.10), 0.05, 0.81, None),
        ((0.70, 0.30, 0.10), 0.20, 0.81, None),
        ((0.70, 0.30, 0.10), 0.70, 0.81, None),
        ((0.70, 0.30, 0.10), 1.00, 0.81, None),
        ((0.70, 0.30, 0.10), 2.00, 0.81, None),
        ((0.70, 0.30, 0.10), -1.00, 0.81, None),
        ((0.70, 0.30, 0.10), 0.40, 0.81, (0.60, 0.0, 0.80)),
        ((0.70, 0.30, 0.10), 0.40, 0.81, (0.30, 0.0, 0.40)),
        ((0.70, 0.30, 0.10), 0.40, 0.81, (0.0, 0.0, 0.0)),
    )
    materials = []
    for index, (color, roughness, mix_factor, normal) in enumerate(cases):
        material, tree, output = _material(
            f"Glossy BSDF Matrix {index:02d}"
        )
        glossy = tree.nodes.new("ShaderNodeBsdfAnisotropic")
        glossy.name = f"Glossy BSDF {index:02d}"
        # The roughness sweep locks all standalone microfacet families:
        # Beckmann visible-normal sampling, ordinary GGX, and MULTI_GGX's
        # constant-Fresnel energy-preservation path.
        if index == 8:
            glossy.distribution = "BECKMANN"
        elif index in {9, 10, 11}:
            glossy.distribution = "MULTI_GGX"
        else:
            glossy.distribution = "GGX"
        tree.links.new(
            _linked_vector(tree, f"Glossy Color {index:02d}", color),
            _input(glossy, "Color"),
        )
        roughness_node = tree.nodes.new("ShaderNodeValue")
        roughness_node.name = f"Glossy Roughness {index:02d}"
        _output(roughness_node, "Value").default_value = roughness
        tree.links.new(
            _output(roughness_node, "Value"),
            _input(glossy, "Roughness"),
        )
        if normal is not None:
            tree.links.new(
                _linked_vector(
                    tree, f"Glossy Normal {index:02d}", normal
                ),
                _input(glossy, "Normal"),
            )
        factor = tree.nodes.new("ShaderNodeValue")
        factor.name = f"Glossy Mix Factor {index:02d}"
        _output(factor, "Value").default_value = mix_factor
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.name = f"Glossy Mix {index:02d}"
        tree.links.new(_output(factor, "Value"), _input(mix, "Fac"))
        tree.links.new(_output(glossy, "BSDF"), mix.inputs[2])
        tree.links.new(_output(mix, "Shader"), _input(output, "Surface"))
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Glossy BSDF Matrix",
    )
    surface.visible_shadow = False


def _indirect_diffuse(scene: Any) -> None:
    """Exercise a second diffuse surface before reaching the world."""
    _world(scene, (0.64, 0.78, 1.0, 1.0), 0.7)

    floor, floor_tree, floor_output = _material(
        "Indirect Diffuse Floor"
    )
    floor_diffuse = floor_tree.nodes.new("ShaderNodeBsdfDiffuse")
    floor_diffuse.name = "Floor Diffuse"
    _input(floor_diffuse, "Color").default_value = (
        0.62,
        0.41,
        0.23,
        1.0,
    )
    _input(floor_diffuse, "Roughness").default_value = 0.0
    floor_tree.links.new(
        _output(floor_diffuse, "BSDF"),
        _input(floor_output, "Surface"),
    )
    _plane(floor)

    wall, wall_tree, wall_output = _material(
        "Indirect Diffuse Wall"
    )
    wall_diffuse = wall_tree.nodes.new("ShaderNodeBsdfDiffuse")
    wall_diffuse.name = "Wall Diffuse"
    _input(wall_diffuse, "Color").default_value = (
        0.18,
        0.72,
        0.27,
        1.0,
    )
    _input(wall_diffuse, "Roughness").default_value = 0.0
    wall_tree.links.new(
        _output(wall_diffuse, "BSDF"),
        _input(wall_output, "Surface"),
    )
    bpy.ops.mesh.primitive_plane_add(
        size=8.0,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 1.1, 2.0),
        rotation=(1.5707963267948966, 0.0, 0.0),
    )
    wall_object = bpy.context.object
    wall_object.name = "Indirect Bounce Wall"
    wall_object.data.materials.append(wall)

    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0


def _indirect_principled(scene: Any) -> None:
    """Exercise multiple bounces through mixed diffuse/glossy closures."""
    _world(scene, (0.64, 0.78, 1.0, 1.0), 0.7)
    # Pin the path-dependent Cycles closure filter. It is applied only after a
    # low-PDF prior bounce, so this probe distinguishes closure setup state
    # from the effective microfacet alpha used for subsequent evaluation.
    scene.cycles.blur_glossy = 1.0

    floor, floor_tree, floor_output = _material(
        "Indirect Principled Floor"
    )
    floor_bsdf = floor_tree.nodes.new(
        "ShaderNodeBsdfPrincipled"
    )
    floor_bsdf.name = "Floor Principled"
    floor_bsdf.distribution = "GGX"
    _input(floor_bsdf, "Base Color").default_value = (
        0.62,
        0.41,
        0.23,
        1.0,
    )
    _input(floor_bsdf, "Metallic").default_value = 0.15
    _input(floor_bsdf, "Roughness").default_value = 0.35
    _input(floor_bsdf, "Diffuse Roughness").default_value = 0.0
    _input(floor_bsdf, "IOR").default_value = 1.45
    _input(floor_bsdf, "Specular IOR Level").default_value = 0.5
    floor_tree.links.new(
        _output(floor_bsdf, "BSDF"),
        _input(floor_output, "Surface"),
    )
    _plane(floor)

    wall, wall_tree, wall_output = _material(
        "Indirect Principled Wall"
    )
    wall_bsdf = wall_tree.nodes.new(
        "ShaderNodeBsdfPrincipled"
    )
    wall_bsdf.name = "Wall Principled"
    wall_bsdf.distribution = "GGX"
    _input(wall_bsdf, "Base Color").default_value = (
        0.18,
        0.72,
        0.27,
        1.0,
    )
    _input(wall_bsdf, "Metallic").default_value = 0.05
    _input(wall_bsdf, "Roughness").default_value = 0.52
    _input(wall_bsdf, "Diffuse Roughness").default_value = 0.0
    _input(wall_bsdf, "IOR").default_value = 1.45
    _input(wall_bsdf, "Specular IOR Level").default_value = 0.5
    wall_tree.links.new(
        _output(wall_bsdf, "BSDF"),
        _input(wall_output, "Surface"),
    )
    bpy.ops.mesh.primitive_plane_add(
        size=8.0,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 1.1, 2.0),
        rotation=(1.5707963267948966, 0.0, 0.0),
    )
    wall_object = bpy.context.object
    wall_object.name = "Indirect Principled Bounce Wall"
    wall_object.data.materials.append(wall)

    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 0


def _nishita_diffuse_transport(scene: Any) -> None:
    """Measure the integrated Nishita sky and solar-disc transport."""
    world = bpy.data.worlds.new("Nishita Transport World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    sky = tree.nodes.new("ShaderNodeTexSky")
    sky.name = "Nishita Sky"
    sky.sky_type = "SINGLE_SCATTERING"
    sky.sun_disc = True
    sky.sun_elevation = 0.9250245094299316
    sky.sun_rotation = 2.6179938316345215
    sky.sun_size = 0.01745329238474369
    sky.sun_intensity = 1.0
    sky.altitude = 0.0
    sky.air_density = 1.0
    sky.aerosol_density = 1.0
    sky.ozone_density = 1.0
    background = tree.nodes.new("ShaderNodeBackground")
    background.name = "Background"
    _input(background, "Strength").default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.name = "World Output"
    tree.links.new(
        _output(sky, "Color"),
        _input(background, "Color"),
    )
    tree.links.new(
        _output(background, "Background"),
        _input(output, "Surface"),
    )
    scene.world = world

    material, material_tree, material_output = _material(
        "Nishita Diffuse Receiver"
    )
    diffuse = material_tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse Receiver"
    _input(diffuse, "Color").default_value = (
        0.62,
        0.41,
        0.23,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    material_tree.links.new(
        _output(diffuse, "BSDF"),
        _input(material_output, "Surface"),
    )
    _plane(material)

    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.use_light_tree = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"


def _hosek_wilkie_diffuse_transport(scene: Any) -> None:
    """Measure legacy Hosek-Wilkie sky transport without baking it."""
    world = bpy.data.worlds.new("Hosek-Wilkie Transport World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    sky = tree.nodes.new("ShaderNodeTexSky")
    sky.name = "Hosek-Wilkie Sky"
    sky.sky_type = "HOSEK_WILKIE"
    sky.sun_direction = (
        -0.9461538195610046,
        0.0615384615957737,
        0.31781429052352905,
    )
    sky.turbidity = 2.9
    sky.ground_albedo = 0.3
    background = tree.nodes.new("ShaderNodeBackground")
    background.name = "Background"
    _input(background, "Strength").default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.name = "World Output"
    tree.links.new(
        _output(sky, "Color"),
        _input(background, "Color"),
    )
    tree.links.new(
        _output(background, "Background"),
        _input(output, "Surface"),
    )
    scene.world = world

    material, material_tree, material_output = _material(
        "Hosek-Wilkie Diffuse Receiver"
    )
    diffuse = material_tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse Receiver"
    _input(diffuse, "Color").default_value = (
        0.62,
        0.41,
        0.23,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    material_tree.links.new(
        _output(diffuse, "BSDF"),
        _input(material_output, "Surface"),
    )
    _plane(material)

    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.use_light_tree = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"


def _translucent_surface(scene: Any) -> None:
    """Exercise Cycles' diffuse-transmission hemisphere and event labels."""
    _world(scene, (0.31, 0.56, 0.82, 1.0), 1.0)
    material, tree, output = _material("Translucent Probe")
    translucent = tree.nodes.new("ShaderNodeBsdfTranslucent")
    translucent.name = "Translucent BSDF"
    _input(translucent, "Color").default_value = (
        0.73,
        0.28,
        0.11,
        1.0,
    )
    tree.links.new(
        _output(translucent, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _translucent_bsdf_matrix(scene: Any) -> None:
    """Cover Translucent allocation, transmission normal, and pass labels."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    _bsdf_matrix_sun(scene, transmission=True)
    cases = (
        ((0.73, 0.28, 0.11), None, False),
        ((1.4, 0.2, 2.2), None, False),
        ((-0.5, 0.6, 1.2), None, False),
        ((-0.5, -0.2, -1.0), None, False),
        ((1.0e-6, 1.0e-6, 1.0e-6), None, False),
        ((3.0e-5, 0.0, 0.0), None, False),
        ((0.7, 0.3, 0.1), (0.6, 0.0, 0.8), False),
        ((0.7, 0.3, 0.1), (0.3, 0.0, 0.4), False),
        ((0.7, 0.3, 0.1), (0.0, 0.0, 0.0), False),
        ((0.7, 0.3, 0.1), (1.0, 0.0, 0.0), False),
        ((0.7, 0.3, 0.1), (-0.6, 0.0, -0.8), False),
        ((0.7, 0.3, 0.1), (0.0, 1.0, 0.0), False),
        ((0.7, 0.3, 0.1), None, True),
        ((0.7, 0.3, 0.1), (0.6, 0.0, 0.8), True),
        ((0.7, 0.3, 0.1), (1.0, 0.0, 0.0), True),
        ((0.7, 0.3, 0.1), (0.0, 0.0, 0.0), True),
    )
    materials = []
    backfacing: set[int] = set()
    for index, (color, normal, backface) in enumerate(cases):
        material, tree, output = _material(
            f"Translucent BSDF Matrix {index:02d}"
        )
        translucent = tree.nodes.new("ShaderNodeBsdfTranslucent")
        translucent.name = f"Translucent BSDF {index:02d}"
        tree.links.new(
            _linked_vector(
                tree, f"Translucent Color {index:02d}", color
            ),
            _input(translucent, "Color"),
        )
        if normal is not None:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Translucent Normal {index:02d}",
                    normal,
                ),
                _input(translucent, "Normal"),
            )
        tree.links.new(
            _output(translucent, "BSDF"),
            _input(output, "Surface"),
        )
        if backface:
            backfacing.add(index)
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Translucent BSDF Matrix",
        backfacing=backfacing,
    )
    surface.visible_shadow = False
    surface.visible_transmission = False


def _principled_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Principled Probe")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Principled BSDF"
    principled.distribution = "GGX"
    _input(principled, "Base Color").default_value = (
        0.32,
        0.12,
        0.06,
        1.0,
    )
    _input(principled, "Metallic").default_value = 0.35
    _input(principled, "Roughness").default_value = 0.28
    _input(principled, "Diffuse Roughness").default_value = 0.0
    _input(principled, "IOR").default_value = 1.45
    _input(principled, "Specular IOR Level").default_value = 0.5
    _input(principled, "Specular Tint").default_value = (
        0.7,
        0.9,
        1.0,
        1.0,
    )
    tree.links.new(
        _output(principled, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _principled_sheen_surface(scene: Any) -> None:
    """Isolate current Cycles Principled Sheen setup and LTC sampling."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.use_light_tree = False

    world = bpy.data.worlds.new("Directional Sheen World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "World Direction"
    absolute = tree.nodes.new("ShaderNodeVectorMath")
    absolute.name = "Absolute World Direction"
    absolute.operation = "ABSOLUTE"
    background = tree.nodes.new("ShaderNodeBackground")
    background.name = "Directional Background"
    _input(background, "Strength").default_value = 1.4
    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.name = "World Output"
    tree.links.new(
        _output(coordinates, "Normal"), absolute.inputs[0]
    )
    tree.links.new(
        _output(absolute, "Vector"), _input(background, "Color")
    )
    tree.links.new(
        _output(background, "Background"), _input(output, "Surface")
    )
    world.cycles.sampling_method = "NONE"
    scene.world = world

    cases = (
        (-1.0, 0.5, (1.0, 1.0, 1.0), None),
        (0.0, 0.5, (1.0, 1.0, 1.0), None),
        (0.5e-5, 0.5, (1.0, 1.0, 1.0), None),
        (1.0e-5, 0.5, (1.0, 1.0, 1.0), None),
        (2.0e-5, 0.5, (1.0, 1.0, 1.0), None),
        (0.25, -0.3, (0.8, 0.3, 0.1), None),
        (0.45, 0.0, (0.9, 0.5, 0.2), None),
        (0.55, 0.001, (0.3, 0.8, 0.4), None),
        (0.65, 0.05, (1.0, 0.25, 0.08), None),
        (0.75, 0.37, (0.8, 0.4, 0.2), None),
        (0.85, 0.95, (0.15, 0.65, 1.2), None),
        (0.9, 1.0, (0.7, 0.9, 0.25), None),
        (0.9, 1.4, (0.6, 0.2, 1.1), None),
        (0.8, 0.6, (-0.2, 0.5, 1.4), None),
        (0.8, 0.32, (0.8, 0.4, 0.2), (0.3, -0.2, 0.9)),
        (0.8, 0.72, (0.2, 1.1, 0.45), (0.8, 0.0, 0.6)),
    )
    materials = []
    for index, (weight, roughness, tint, normal) in enumerate(cases):
        material, material_tree, material_output = _material(
            f"Principled Sheen {index:02d}"
        )
        principled = material_tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = f"Isolated Principled Sheen {index:02d}"
        principled.distribution = "GGX"
        _input(principled, "Base Color").default_value = (
            0.0,
            0.0,
            0.0,
            1.0,
        )
        _input(principled, "Metallic").default_value = 0.0
        _input(principled, "Roughness").default_value = 0.27
        _input(principled, "IOR").default_value = 1.0
        _input(principled, "Specular IOR Level").default_value = 0.5
        _input(principled, "Alpha").default_value = 1.0
        _input(principled, "Coat Weight").default_value = 0.0
        weight_node = material_tree.nodes.new("ShaderNodeValue")
        weight_node.name = f"Linked Sheen Weight {index:02d}"
        _output(weight_node, "Value").default_value = weight
        roughness_node = material_tree.nodes.new("ShaderNodeValue")
        roughness_node.name = f"Linked Sheen Roughness {index:02d}"
        _output(roughness_node, "Value").default_value = roughness
        material_tree.links.new(
            _output(weight_node, "Value"),
            _input(principled, "Sheen Weight"),
        )
        material_tree.links.new(
            _output(roughness_node, "Value"),
            _input(principled, "Sheen Roughness"),
        )
        material_tree.links.new(
            _linked_vector(
                material_tree, f"Linked Sheen Tint {index:02d}", tint
            ),
            _input(principled, "Sheen Tint"),
        )
        if normal is not None:
            material_tree.links.new(
                _linked_vector(
                    material_tree,
                    f"Linked Sheen Normal {index:02d}",
                    normal,
                ),
                _input(principled, "Normal"),
            )
        material_tree.links.new(
            _output(principled, "BSDF"),
            _input(material_output, "Surface"),
        )
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Principled Sheen Surface Matrix",
        frame_bleed=0.02,
    )
    surface.visible_shadow = False


def _principled_coat_surface(scene: Any) -> None:
    """Exercise Cycles' physical Principled Coat and ordered attenuation."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    _bsdf_matrix_sun(scene, transmission=False)
    cases = (
        (-1.0, 0.30, 1.50, (1.0, 1.0, 1.0), None),
        (0.0, 0.30, 1.50, (1.0, 1.0, 1.0), None),
        (0.5e-5, 0.30, 1.50, (1.0, 1.0, 1.0), None),
        (1.0e-5, 0.30, 1.50, (1.0, 1.0, 1.0), None),
        (2.0e-5, 0.30, 1.50, (1.0, 1.0, 1.0), None),
        (0.70, -0.30, 1.50, (1.0, 1.0, 1.0), None),
        (0.70, 0.0, 1.50, (1.0, 1.0, 1.0), None),
        (0.70, 0.001, 1.50, (1.0, 1.0, 1.0), None),
        (0.70, 0.01, 1.50, (1.0, 1.0, 1.0), None),
        (0.70, 0.20, 1.0, (1.0, 1.0, 1.0), None),
        (0.70, 0.20, 1.33, (1.0, 1.0, 1.0), None),
        (0.70, 0.20, 2.0, (0.2, 0.55, 0.9), None),
        (1.30, 0.31, 1.45, (0.45, 0.8, 0.2), None),
        (0.65, 0.27, 1.60, (1.0, 1.0, 1.0), (0.6, 0.0, 0.8)),
        (0.65, 0.27, 1.60, (1.0, 1.0, 1.0), (0.3, 0.0, 0.4)),
        (0.65, 0.27, 1.60, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)),
    )
    materials = []
    for index, (weight, roughness, ior, tint, normal) in enumerate(cases):
        material, tree, output = _material(
            f"Principled Coat {index:02d}"
        )
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = f"Physical Principled Coat {index:02d}"
        principled.distribution = "GGX"
        _input(principled, "Base Color").default_value = (
            0.38,
            0.16,
            0.07,
            1.0,
        )
        _input(principled, "Metallic").default_value = 0.0
        _input(principled, "Roughness").default_value = 0.4
        _input(principled, "IOR").default_value = 1.0
        _input(principled, "Specular IOR Level").default_value = 0.5
        _input(principled, "Sheen Weight").default_value = 0.0
        _input(principled, "Alpha").default_value = 1.0
        weight_node = tree.nodes.new("ShaderNodeValue")
        weight_node.name = f"Linked Coat Weight {index:02d}"
        _output(weight_node, "Value").default_value = weight
        roughness_node = tree.nodes.new("ShaderNodeValue")
        roughness_node.name = f"Linked Coat Roughness {index:02d}"
        _output(roughness_node, "Value").default_value = roughness
        ior_node = tree.nodes.new("ShaderNodeValue")
        ior_node.name = f"Linked Coat IOR {index:02d}"
        _output(ior_node, "Value").default_value = ior
        tree.links.new(
            _output(weight_node, "Value"),
            _input(principled, "Coat Weight"),
        )
        tree.links.new(
            _output(roughness_node, "Value"),
            _input(principled, "Coat Roughness"),
        )
        tree.links.new(
            _output(ior_node, "Value"),
            _input(principled, "Coat IOR"),
        )
        tree.links.new(
            _linked_vector(tree, f"Linked Coat Tint {index:02d}", tint),
            _input(principled, "Coat Tint"),
        )
        if normal is not None:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Linked Coat Normal {index:02d}",
                    normal,
                ),
                _input(principled, "Coat Normal"),
            )
        tree.links.new(
            _output(principled, "BSDF"),
            _input(output, "Surface"),
        )
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Principled Coat Surface Matrix",
        frame_bleed=0.02,
    )
    surface.visible_shadow = False


def _principled_transmission_surface(scene: Any) -> None:
    """Exercise Cycles' thick Principled generalized-Schlick glass."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    _bsdf_matrix_sun(scene, transmission=True)
    # distribution, weight, roughness, IOR, base color, specular tint,
    # metallic, explicit normal, backface
    cases = (
        ("GGX", -1.0, 0.30, 1.50, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 0.0, 0.30, 1.50, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 0.5e-5, 0.30, 1.50, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 1.0e-5, 0.30, 1.50, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 2.0e-5, 0.30, 1.50, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 0.25, 0.05, 1.50, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 0.70, 0.20, 1.00, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 0.70, 0.20, 1.33, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("GGX", 0.70, 0.20, 2.00, (0.81, 0.25, 0.04),
         (0.2, 0.7, 1.0), 0.0, None, False),
        ("GGX", 1.30, 0.31, 1.45, (0.16, 0.81, 0.36),
         (0.8, 0.3, 1.0), 0.0, None, False),
        ("GGX", 0.65, 0.27, 1.60, (0.64, 0.09, 0.49),
         (0.4, 1.0, 0.2), 0.0, (0.3, 0.0, 0.4), False),
        ("GGX", 0.65, 0.27, 1.60, (0.64, 0.09, 0.49),
         (0.4, 1.0, 0.2), 0.0, None, True),
        ("MULTI_GGX", 0.70, 0.30, 1.50, (0.36, 0.64, 1.00),
         (1.0, 1.0, 1.0), 0.0, None, False),
        ("MULTI_GGX", 1.00, 0.60, 1.33, (0.81, 0.25, 0.04),
         (0.2, 0.7, 1.0), 0.0, None, False),
        ("MULTI_GGX", 0.85, 0.42, 1.80, (0.09, 0.49, 0.81),
         (1.0, 0.4, 0.2), 0.0, (0.6, 0.0, 0.8), False),
        ("GGX", 0.90, 0.40, 1.50, (0.49, 0.81, 0.16),
         (0.3, 0.8, 1.0), 0.35, None, False),
    )
    materials = []
    backfacing: set[int] = set()
    for index, case in enumerate(cases):
        (
            distribution,
            transmission,
            roughness,
            ior,
            base_color,
            specular_tint,
            metallic,
            normal,
            backface,
        ) = case
        material, tree, output = _material(
            f"Principled Transmission {index:02d}"
        )
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = f"Physical Principled Transmission {index:02d}"
        principled.distribution = distribution
        _input(principled, "Diffuse Roughness").default_value = 0.0
        _input(principled, "Specular IOR Level").default_value = 0.5
        _input(principled, "Sheen Weight").default_value = 0.0
        _input(principled, "Coat Weight").default_value = 0.0
        _input(principled, "Alpha").default_value = 1.0
        tree.links.new(
            _linked_vector(tree, f"Linked Base Color {index:02d}", base_color),
            _input(principled, "Base Color"),
        )
        tree.links.new(
            _linked_vector(tree, f"Linked Specular Tint {index:02d}", specular_tint),
            _input(principled, "Specular Tint"),
        )
        for label, socket, value in (
            ("Transmission Weight", "Transmission Weight", transmission),
            ("Roughness", "Roughness", roughness),
            ("IOR", "IOR", ior),
            ("Metallic", "Metallic", metallic),
        ):
            value_node = tree.nodes.new("ShaderNodeValue")
            value_node.name = f"Linked {label} {index:02d}"
            _output(value_node, "Value").default_value = value
            tree.links.new(
                _output(value_node, "Value"),
                _input(principled, socket),
            )
        if normal is not None:
            tree.links.new(
                _linked_vector(tree, f"Linked Normal {index:02d}", normal),
                _input(principled, "Normal"),
            )
        tree.links.new(
            _output(principled, "BSDF"),
            _input(output, "Surface"),
        )
        if backface:
            backfacing.add(index)
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Principled Transmission Surface Matrix",
        backfacing=backfacing,
        frame_bleed=0.02,
    )
    surface.visible_shadow = False


def _principled_thin_wall_surface(scene: Any) -> None:
    """Exercise Cycles' zero-thin-film Principled Thin Wall expansion."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 3
    scene.cycles.light_sampling_threshold = 0.0
    _world(scene, (0.035, 0.08, 0.16, 1.0), 0.45)

    def add_sun(
        name: str,
        direction: tuple[float, float, float],
        color: tuple[float, float, float],
        energy: float,
    ) -> None:
        light_data = bpy.data.lights.new(name, type="SUN")
        light_data.color = color
        light_data.energy = energy
        light_data.angle = 0.0
        light_data.normalize = True
        light_data.use_shadow = True
        light = bpy.data.objects.new(name, light_data)
        light.rotation_euler = Vector(direction).to_track_quat(
            "Z", "Y"
        ).to_euler()
        scene.collection.objects.link(light)

    # The two oblique delta lights expose reflection and transmission without
    # conflating the thin-wall direction law with a normal-incidence special
    # case. The low constant world also makes singular transmission visible.
    add_sun(
        "Thin Wall Front Sun",
        (0.42, -0.23, 0.88),
        (1.0, 0.38, 0.12),
        1.6,
    )
    add_sun(
        "Thin Wall Back Sun",
        (-0.31, 0.27, -0.91),
        (0.11, 0.48, 1.0),
        2.1,
    )

    # mode, thin wall, linked Thin Wall, roughness, IOR, base color,
    # diffuse roughness, subsurface anisotropy, explicit normal, backface,
    # bump-map correction
    cases = (
        ("glass", False, False, 0.28, 1.45, (0.36, 0.64, 1.00),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.28, 1.45, (0.36, 0.64, 1.00),
         0.0, 0.0, None, False, True),
        ("glass", True, True, 0.28, 1.45, (0.36, 0.64, 1.00),
         0.0, 0.0, None, False, True),
        ("glass", False, True, 0.28, 1.45, (0.36, 0.64, 1.00),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.0, 1.0, (0.81, 0.25, 0.04),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.0, 1.33, (0.16, 0.81, 0.36),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.0, 2.0, (0.64, 0.09, 0.49),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.05, 1.5, (0.81, 0.49, 0.09),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.35, 1.33, (0.09, 0.49, 0.81),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.7, 1.8, (0.49, 0.81, 0.16),
         0.0, 0.0, None, False, True),
        ("glass", True, False, 0.27, 1.6, (0.64, 0.09, 0.49),
         0.0, 0.0, (0.3, 0.0, 0.4), False, True),
        ("glass", True, False, 0.27, 1.6, (0.64, 0.09, 0.49),
         0.0, 0.0, None, True, True),
        ("subsurface", True, False, 0.32, 1.4, (0.72, 0.24, 0.08),
         0.0, -0.8, None, False, True),
        ("subsurface", True, False, 0.32, 1.4, (0.12, 0.52, 0.72),
         0.18, -0.45, None, False, True),
        ("subsurface", True, False, 0.32, 1.4, (0.18, 0.70, 0.20),
         0.55, 0.0, None, False, True),
        ("subsurface", True, True, 0.32, 1.4, (0.68, 0.22, 0.10),
         0.9, 0.75, (0.6, 0.0, 0.8), False, True),
        # Paired controls make conversion, reflection/transmission splitting,
        # and bump-map correction independently observable.
        ("subsurface", True, False, 0.32, 1.4, (0.68, 0.22, 0.10),
         0.9, 0.75, (0.6, 0.0, 0.8), False, True),
        ("subsurface", True, True, 0.32, 1.4, (0.68, 0.22, 0.10),
         0.9, -1.0, (0.6, 0.0, 0.8), False, True),
        ("subsurface", True, True, 0.32, 1.4, (0.68, 0.22, 0.10),
         0.9, 1.0, (0.6, 0.0, 0.8), False, True),
        ("subsurface", True, True, 0.32, 1.4, (0.68, 0.22, 0.10),
         0.9, 0.75, (0.6, 0.0, 0.8), False, False),
    )
    materials = []
    backfacing: set[int] = set()
    for index, case in enumerate(cases):
        (
            mode,
            thin_wall,
            linked_thin_wall,
            roughness,
            ior,
            base_color,
            diffuse_roughness,
            subsurface_anisotropy,
            normal,
            backface,
            bump_correction,
        ) = case
        material, tree, output = _material(
            f"Principled Thin Wall {index:02d}"
        )
        material.cycles.use_bump_map_correction = bump_correction
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = f"Physical Principled Thin Wall {index:02d}"
        principled.distribution = "GGX"
        _input(principled, "Base Color").default_value = (
            *base_color,
            1.0,
        )
        _input(principled, "Metallic").default_value = 0.0
        _input(principled, "Roughness").default_value = roughness
        _input(principled, "Diffuse Roughness").default_value = (
            diffuse_roughness
        )
        _input(principled, "IOR").default_value = ior
        _input(principled, "Specular IOR Level").default_value = 0.5
        _input(principled, "Specular Tint").default_value = (
            0.4,
            0.8,
            1.0,
            1.0,
        )
        _input(principled, "Transmission Weight").default_value = (
            1.0 if mode == "glass" else 0.0
        )
        _input(principled, "Subsurface Weight").default_value = (
            1.0 if mode == "subsurface" else 0.0
        )
        _input(principled, "Subsurface Anisotropy").default_value = (
            subsurface_anisotropy
        )
        _input(principled, "Sheen Weight").default_value = 0.0
        _input(principled, "Coat Weight").default_value = 0.0
        _input(principled, "Alpha").default_value = 1.0
        # Thin-film interference is deliberately outside this checkpoint.
        # Pinning the authored thickness to exact zero keeps this scene on the
        # analytically implemented no-film branch in both renderers.
        _input(principled, "Thin Film Thickness").default_value = 0.0
        if linked_thin_wall:
            thin_wall_value = tree.nodes.new("ShaderNodeValue")
            thin_wall_value.name = f"Linked Thin Wall {index:02d}"
            _output(thin_wall_value, "Value").default_value = (
                1.0 if thin_wall else 0.0
            )
            tree.links.new(
                _output(thin_wall_value, "Value"),
                _input(principled, "Thin Wall"),
            )
        else:
            _input(principled, "Thin Wall").default_value = thin_wall
        if normal is not None:
            tree.links.new(
                _linked_vector(tree, f"Linked Normal {index:02d}", normal),
                _input(principled, "Normal"),
            )
        tree.links.new(
            _output(principled, "BSDF"),
            _input(output, "Surface"),
        )
        if backface:
            backfacing.add(index)
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=5,
        rows=4,
        name="Principled Thin Wall Surface Matrix",
        backfacing=backfacing,
        frame_bleed=0.02,
    )
    surface.visible_shadow = False


def _principled_emission(scene: Any) -> None:
    """Isolate raw Principled Emission Color/Strength evaluation."""
    material, tree, output = _material("Principled Emission Probe")
    color = tree.nodes.new("ShaderNodeRGB")
    color.name = "Linked Principled Emission Color"
    _output(color, "Color").default_value = (
        0.17,
        0.43,
        0.91,
        1.0,
    )
    strength = tree.nodes.new("ShaderNodeValue")
    strength.name = "Linked Principled Emission Strength"
    _output(strength, "Value").default_value = 2.75
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Raw Principled Emission"
    _input(principled, "Base Color").default_value = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    _input(principled, "Metallic").default_value = 0.0
    _input(principled, "Roughness").default_value = 0.5
    _input(principled, "Sheen Weight").default_value = 0.0
    _input(principled, "Coat Weight").default_value = 0.0
    _input(principled, "Alpha").default_value = 1.0
    tree.links.new(
        _output(color, "Color"),
        _input(principled, "Emission Color"),
    )
    tree.links.new(
        _output(strength, "Value"),
        _input(principled, "Emission Strength"),
    )
    tree.links.new(
        _output(principled, "BSDF"),
        _input(output, "Surface"),
    )
    _plane(material)


def _principled_emission_layers(scene: Any) -> None:
    """Exercise Cycles' ordered Alpha, Sheen, Coat, and emission relation."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    defaults = {
        "alpha": 1.0,
        "sheen_weight": 0.0,
        "sheen_roughness": 0.5,
        "sheen_tint": (1.0, 1.0, 1.0),
        "coat_weight": 0.0,
        "coat_roughness": 0.03,
        "coat_ior": 1.5,
        "coat_tint": (1.0, 1.0, 1.0),
        "coat_normal": None,
    }
    cases = (
        {},
        {"alpha": 0.37},
        {"alpha": -0.25},
        {"alpha": 1.4},
        {
            "sheen_weight": 0.65,
            "sheen_roughness": 0.05,
            "sheen_tint": (1.0, 0.2, 0.05),
        },
        {
            "sheen_weight": 1.2,
            "sheen_roughness": 0.95,
            "sheen_tint": (0.1, 0.8, 1.4),
        },
        {
            # N and Coat Normal cancel at the 0.5 interpolation point.
            # Cycles' sheen path keeps that exact zero normal rather than
            # substituting the shading normal.
            "sheen_weight": 0.8,
            "coat_weight": 0.5,
            "coat_normal": (0.0, 0.0, -1.0),
        },
        {
            "sheen_weight": 0.8,
            "sheen_roughness": -0.3,
            "sheen_tint": (0.7, 0.4, 0.2),
        },
        {"coat_weight": 0.8, "coat_roughness": 0.0},
        {
            "coat_weight": 0.8,
            "coat_roughness": 0.42,
            "coat_ior": 1.33,
        },
        {
            "coat_weight": 1.0,
            "coat_roughness": 0.15,
            "coat_ior": 1.0,
        },
        {
            "coat_weight": 0.7,
            "coat_roughness": 0.2,
            "coat_ior": 2.0,
            "coat_tint": (0.2, 0.55, 0.9),
        },
        {
            "coat_weight": 0.65,
            "coat_roughness": 0.27,
            "coat_ior": 1.6,
            "coat_normal": (0.6, 0.0, 0.8),
        },
        {
            "alpha": 0.73,
            "sheen_weight": 0.55,
            "sheen_roughness": 0.38,
            "sheen_tint": (0.9, 0.35, 0.12),
            "coat_weight": 0.75,
            "coat_roughness": 0.23,
            "coat_ior": 1.7,
            "coat_tint": (0.3, 0.7, 0.95),
        },
        {
            "coat_weight": 1.3,
            "coat_roughness": 0.31,
            "coat_ior": 1.45,
            "coat_tint": (0.45, 0.8, 0.2),
        },
        {
            "alpha": 0.58,
            "sheen_weight": 0.9,
            "sheen_roughness": 0.72,
            "sheen_tint": (0.25, 1.1, 0.6),
            "coat_weight": 0.9,
            "coat_roughness": 0.08,
            "coat_ior": 1.8,
            "coat_tint": (0.8, 0.25, 0.5),
            "coat_normal": (0.8, 0.0, 0.6),
        },
    )
    materials = []
    for index, overrides in enumerate(cases):
        case = defaults | overrides
        material, tree, output = _material(
            f"Principled Emission Layers {index:02d}"
        )
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = f"Layered Principled Emission {index:02d}"
        _input(principled, "Base Color").default_value = (0.0, 0.0, 0.0, 1.0)
        _input(principled, "Metallic").default_value = 0.0
        _input(principled, "Roughness").default_value = 0.5
        _input(principled, "Alpha").default_value = case["alpha"]
        _input(principled, "Sheen Weight").default_value = case[
            "sheen_weight"
        ]
        _input(principled, "Sheen Roughness").default_value = case[
            "sheen_roughness"
        ]
        _input(principled, "Sheen Tint").default_value = (
            *case["sheen_tint"],
            1.0,
        )
        _input(principled, "Coat Weight").default_value = case[
            "coat_weight"
        ]
        _input(principled, "Coat Roughness").default_value = case[
            "coat_roughness"
        ]
        _input(principled, "Coat IOR").default_value = case[
            "coat_ior"
        ]
        _input(principled, "Coat Tint").default_value = (
            *case["coat_tint"],
            1.0,
        )
        _input(principled, "Emission Color").default_value = (
            0.17,
            0.43,
            0.91,
            1.0,
        )
        _input(principled, "Emission Strength").default_value = 2.75
        if case["coat_normal"] is not None:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Linked Coat Normal {index:02d}",
                    case["coat_normal"],
                ),
                _input(principled, "Coat Normal"),
            )
        tree.links.new(
            _output(principled, "BSDF"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Principled Emission Layer Matrix",
    )


def _principled_alpha_surface(scene: Any) -> None:
    """Isolate Principled Alpha and Cycles' merged transparent closure."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    _world(scene, (0.12, 0.37, 0.83, 1.0), 1.7)
    cases = (
        {"alpha": -1.0},
        {"alpha": 0.0},
        {"alpha": 0.25},
        {"alpha": 0.5},
        {"alpha": 0.75},
        {"alpha": 0.999},
        {"alpha": 1.0},
        {"alpha": 2.0},
        {"alpha": 0.2, "linked": True},
        {"alpha": 0.4, "linked": True},
        {"alpha": 0.6, "linked": True},
        {"alpha": 0.8, "linked": True},
        {"alpha": 1.0 - 0.5e-5},
        {"alpha": 1.0 - 1.0e-5},
        {"alpha": 1.0 - 2.0e-5},
        {"alpha": 0.6, "merged": True},
    )
    materials = []
    for index, case in enumerate(cases):
        material, tree, output = _material(
            f"Principled Alpha Surface {index:02d}"
        )
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = f"Principled Alpha {index:02d}"
        _input(principled, "Base Color").default_value = (
            0.0,
            0.0,
            0.0,
            1.0,
        )
        _input(principled, "Metallic").default_value = 0.0
        _input(principled, "Roughness").default_value = 0.5
        _input(principled, "IOR").default_value = 1.0
        _input(principled, "Specular IOR Level").default_value = 0.5
        _input(principled, "Emission Strength").default_value = 0.0
        if case.get("linked", False):
            alpha = tree.nodes.new("ShaderNodeValue")
            alpha.name = f"Linked Alpha {index:02d}"
            _output(alpha, "Value").default_value = case["alpha"]
            tree.links.new(
                _output(alpha, "Value"),
                _input(principled, "Alpha"),
            )
        else:
            _input(principled, "Alpha").default_value = case["alpha"]

        surface = _output(principled, "BSDF")
        if case.get("merged", False):
            transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
            transparent.name = "Standalone transparency to merge"
            _input(transparent, "Color").default_value = (
                0.15,
                0.15,
                0.15,
                1.0,
            )
            add = tree.nodes.new("ShaderNodeAddShader")
            add.name = "Merged transparency order"
            tree.links.new(
                _output(transparent, "BSDF"),
                add.inputs[0],
            )
            tree.links.new(surface, add.inputs[1])
            surface = _output(add, "Shader")
        tree.links.new(surface, _input(output, "Surface"))
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Principled Alpha Surface Matrix",
    )


def _principled_bump_glossy(scene: Any) -> None:
    """Stress Cycles' default glossy bump-map correction path."""
    _world(scene, (0.31, 0.52, 0.79, 1.0), 1.2)
    material, tree, output = _material(
        "Principled Bump Glossy Probe"
    )
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.name = "Strong Tangent Normal"
    normal_map.space = "TANGENT"
    _input(normal_map, "Strength").default_value = 1.0
    _input(normal_map, "Color").default_value = (
        1.0,
        0.0,
        0.5,
        1.0,
    )
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Metallic Principled"
    principled.distribution = "GGX"
    _input(principled, "Base Color").default_value = (
        0.68,
        0.27,
        0.08,
        1.0,
    )
    _input(principled, "Metallic").default_value = 1.0
    _input(principled, "Roughness").default_value = 0.32
    _input(principled, "IOR").default_value = 1.45
    _input(principled, "Specular IOR Level").default_value = 0.5
    tree.links.new(
        _output(normal_map, "Normal"),
        _input(principled, "Normal"),
    )
    tree.links.new(
        _output(principled, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _negative_scale_surface(scene: Any) -> None:
    """Exercise Cycles' object-space normal transform under reflection."""
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Negative Scale Probe")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.51,
        0.19,
        0.07,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.27
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    sphere = _sphere(material)
    sphere.scale = (-1.0, 1.0, 1.0)


def _bump_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Bump Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    gradient = tree.nodes.new("ShaderNodeTexGradient")
    gradient.name = "Gradient Texture"
    gradient.gradient_type = "LINEAR"
    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = "Bump"
    _input(bump, "Strength").default_value = 1.0
    _input(bump, "Distance").default_value = 0.2
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.5,
        0.22,
        0.08,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(coordinates, "Generated"),
        _input(gradient, "Vector"),
    )
    tree.links.new(
        _output(gradient, "Fac"),
        _input(bump, "Height"),
    )
    tree.links.new(
        _output(bump, "Normal"),
        _input(diffuse, "Normal"),
    )
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _bump_matrix(scene: Any) -> None:
    """Expose Bump normals without lighting or silhouette variance."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (0.0, 0.2, 0.1, False, (1.0, 0.0, 0.0), None),
        (1.0, 0.0, 0.1, False, (1.0, 0.0, 0.0), None),
        (1.0, 0.2, 0.1, False, (1.0, 0.0, 0.0), None),
        (1.0, 0.2, 0.1, False, (0.0, 1.0, 0.0), None),
        (1.0, 0.2, 0.1, False, (0.7, -0.4, 0.0), None),
        (0.25, 0.2, 0.1, False, (1.0, 0.0, 0.0), None),
        (2.0, 0.2, 0.1, False, (1.0, 0.0, 0.0), None),
        (-1.0, 0.2, 0.1, False, (1.0, 0.0, 0.0), None),
        (1.0, -0.2, 0.1, False, (1.0, 0.0, 0.0), None),
        (1.0, 0.2, 0.1, True, (1.0, 0.0, 0.0), None),
        (1.0, 0.2, 0.01, False, (0.7, -0.4, 0.0), None),
        (1.0, 0.2, 1.0, False, (0.7, -0.4, 0.0), None),
        (1.0, 0.2, 0.1, False, (1.0, 0.0, 0.0), None),
        (1.0, 0.2, 0.1, False, (1.0, 0.0, 0.0), (0.3, 0.0, 0.4)),
        (0.65, 0.47, 0.37, False, (2.0, -3.0, 0.0), (0.0, 0.0, 0.0)),
        (1.3, 0.07, 0.63, True, (-1.7, 2.1, 0.0), (1.0, 0.0, 0.0)),
    )
    materials = []
    for index, (
        strength,
        distance,
        filter_width,
        invert,
        gradient,
        normal,
    ) in enumerate(cases):
        material, tree, output = _material(
            f"Bump Matrix {index:02d}"
        )
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        coordinates.name = f"Coordinates {index:02d}"
        height = tree.nodes.new("ShaderNodeVectorMath")
        height.name = f"Height Dot {index:02d}"
        height.operation = "DOT_PRODUCT"
        _input(height, "Vector").default_value = gradient
        # The second Vector socket is identified by index because Blender
        # gives repeated Vector inputs the same display name.
        height.inputs[1].default_value = gradient
        tree.links.new(
            _output(coordinates, "Generated"),
            height.inputs[0],
        )

        bump = tree.nodes.new("ShaderNodeBump")
        bump.name = f"Bump {index:02d}"
        _input(bump, "Strength").default_value = strength
        _input(bump, "Distance").default_value = distance
        _input(bump, "Filter Width").default_value = filter_width
        bump.invert = invert
        tree.links.new(
            _output(height, "Value"),
            _input(bump, "Height"),
        )
        if normal is not None:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Bump Normal {index:02d}",
                    normal,
                ),
                _input(bump, "Normal"),
            )

        add = tree.nodes.new("ShaderNodeMixRGB")
        add.name = f"Normal Bias {index:02d}"
        add.blend_type = "ADD"
        _input(add, "Fac").default_value = 1.0
        _input(add, "Color2").default_value = (1.0, 1.0, 1.0, 1.0)
        scale = tree.nodes.new("ShaderNodeMixRGB")
        scale.name = f"Normal Scale {index:02d}"
        scale.blend_type = "MULTIPLY"
        _input(scale, "Fac").default_value = 1.0
        _input(scale, "Color2").default_value = (0.5, 0.5, 0.5, 1.0)
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Bump Emission {index:02d}"
        tree.links.new(
            _output(bump, "Normal"),
            _input(add, "Color1"),
        )
        tree.links.new(
            _output(add, "Color"),
            _input(scale, "Color1"),
        )
        tree.links.new(
            _output(scale, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Bump Matrix",
        backfacing={12},
    )

    # Keep the world-space cells identical while exercising the object-space
    # derivative path under a non-uniform, rotated instance transform. A
    # unit object transform would not catch confusing world dP with object
    # coordinate/Generated differentials in the Bump height subgraph.
    surface.rotation_euler = (0.19, -0.27, 0.41)
    surface.scale = (1.7, 0.65, 1.3)
    bpy.context.view_layer.update()
    world_to_object = surface.matrix_world.inverted()
    for vertex in surface.data.vertices:
        vertex.co = world_to_object @ vertex.co

def _bump_nested_matrix(scene: Any) -> None:
    """Exercise a Bump output as the explicit Normal of another Bump."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    # inner gradient, outer gradient, inner/outer strength,
    # inner/outer distance, inner/outer filter width, inner/outer invert
    cases = (
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0, 1.0, 0.2, 0.2, 0.1, 0.1, False, False),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.0, 1.0, 0.2, 0.2, 0.1, 0.1, False, False),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0, 0.0, 0.2, 0.2, 0.1, 0.1, False, False),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0, 1.0, 0.2, 0.2, 0.1, 0.1, True, False),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0, 1.0, 0.2, 0.2, 0.1, 0.1, False, True),
        ((0.7, -0.4, 0.0), (-0.3, 0.8, 0.0), 1.0, 1.0, -0.2, 0.2, 0.1, 0.1, False, False),
        ((0.7, -0.4, 0.0), (-0.3, 0.8, 0.0), 1.0, 1.0, 0.2, -0.2, 0.1, 0.1, False, False),
        ((0.7, -0.4, 0.0), (-0.3, 0.8, 0.0), 1.0, 1.0, 0.2, 0.2, 0.01, 1.0, False, False),
        ((0.7, -0.4, 0.0), (-0.3, 0.8, 0.0), 2.0, 0.25, 0.2, 0.47, 0.37, 0.63, False, False),
        ((1.0, 1.0, 0.0), (1.0, 1.0, 0.0), 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, False, False),
        ((-1.7, 2.1, 0.0), (2.0, -3.0, 0.0), 0.65, 1.3, 0.47, 0.07, 0.37, 0.63, True, False),
        ((0.0, 0.0, 0.0), (0.7, -0.4, 0.0), 1.0, 1.0, 0.2, 0.2, 0.1, 0.1, False, False),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0, 1.0, 0.2, 0.2, 0.1, 0.1, False, False),
        ((0.7, -0.4, 0.0), (-0.3, 0.8, 0.0), 1.0, 1.0, 0.2, 0.2, 0.1, 0.1, True, True),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1.0, 1.0, 0.2, 0.2, 0.1, 0.1, False, False),
        ((-1.7, 2.1, 0.0), (2.0, -3.0, 0.0), 1.3, 0.65, 0.07, 0.47, 0.63, 0.37, False, True),
    )
    materials = []
    for index, case in enumerate(cases):
        (
            inner_gradient,
            outer_gradient,
            inner_strength,
            outer_strength,
            inner_distance,
            outer_distance,
            inner_filter,
            outer_filter,
            inner_invert,
            outer_invert,
        ) = case
        material, tree, output = _material(
            f"Nested Bump Matrix {index:02d}"
        )
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        coordinates.name = f"Coordinates {index:02d}"

        def height_dot(label: str, gradient: tuple[float, float, float]) -> Any:
            node = tree.nodes.new("ShaderNodeVectorMath")
            node.name = f"{label} Height Dot {index:02d}"
            node.operation = "DOT_PRODUCT"
            node.inputs[1].default_value = gradient
            tree.links.new(_output(coordinates, "Generated"), node.inputs[0])
            return node

        inner_height = height_dot("Inner", inner_gradient)
        outer_height = height_dot("Outer", outer_gradient)
        inner = tree.nodes.new("ShaderNodeBump")
        inner.name = f"Inner Bump {index:02d}"
        _input(inner, "Strength").default_value = inner_strength
        _input(inner, "Distance").default_value = inner_distance
        _input(inner, "Filter Width").default_value = inner_filter
        inner.invert = inner_invert
        tree.links.new(
            _output(inner_height, "Value"),
            _input(inner, "Height"),
        )
        outer = tree.nodes.new("ShaderNodeBump")
        outer.name = f"Outer Bump {index:02d}"
        _input(outer, "Strength").default_value = outer_strength
        _input(outer, "Distance").default_value = outer_distance
        _input(outer, "Filter Width").default_value = outer_filter
        outer.invert = outer_invert
        tree.links.new(
            _output(outer_height, "Value"),
            _input(outer, "Height"),
        )
        tree.links.new(
            _output(inner, "Normal"),
            _input(outer, "Normal"),
        )

        add = tree.nodes.new("ShaderNodeMixRGB")
        add.name = f"Nested Normal Bias {index:02d}"
        add.blend_type = "ADD"
        _input(add, "Fac").default_value = 1.0
        _input(add, "Color2").default_value = (1.0, 1.0, 1.0, 1.0)
        scale = tree.nodes.new("ShaderNodeMixRGB")
        scale.name = f"Nested Normal Scale {index:02d}"
        scale.blend_type = "MULTIPLY"
        _input(scale, "Fac").default_value = 1.0
        _input(scale, "Color2").default_value = (0.5, 0.5, 0.5, 1.0)
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Nested Bump Emission {index:02d}"
        tree.links.new(_output(outer, "Normal"), _input(add, "Color1"))
        tree.links.new(_output(add, "Color"), _input(scale, "Color1"))
        tree.links.new(_output(scale, "Color"), _input(emission, "Color"))
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Nested Bump Matrix",
        backfacing={12, 13},
    )
    surface.rotation_euler = (0.19, -0.27, 0.41)
    surface.scale = (1.7, 0.65, 1.3)
    bpy.context.view_layer.update()
    world_to_object = surface.matrix_world.inverted()
    for vertex in surface.data.vertices:
        vertex.co = world_to_object @ vertex.co
