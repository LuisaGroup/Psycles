"""Surface-closure, normal, bump, and indirect-transport probe scenes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import (
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


def _bsdf_matrix_sun(
    scene: Any,
    *,
    transmission: bool,
) -> None:
    """Add a zero-angle Sun for variance-free BSDF evaluation."""
    data = bpy.data.lights.new("BSDF Matrix Sun", type="SUN")
    data.color = (0.36, 0.72, 1.0)
    data.energy = 1.7
    data.normalize = True
    data.angle = 0.0
    data.use_shadow = True
    light = bpy.data.objects.new(data.name, data)
    if transmission:
        light.rotation_euler = (3.141592653589793, 0.0, 0.0)
    scene.collection.objects.link(light)
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 1
    # Tiny closure-allocation threshold cases must not be randomized by
    # Cycles' direct-light sample roulette.
    scene.cycles.light_sampling_threshold = 0.0


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


def _normal_map_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Normal Map Probe")
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.name = "Normal Map"
    normal_map.space = "TANGENT"
    _input(normal_map, "Strength").default_value = 0.7
    _input(normal_map, "Color").default_value = (
        0.65,
        0.35,
        0.95,
        1.0,
    )
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
        _output(normal_map, "Normal"),
        _input(diffuse, "Normal"),
    )
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _normal_map_matrix(scene: Any) -> None:
    """Expose Normal Map output for spaces, strength, signs, and backsides."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("TANGENT", 1.0, (0.5, 0.5, 1.0), False),
        ("TANGENT", 0.0, (0.7, 0.2, 0.9), False),
        ("TANGENT", 0.5, (0.7, 0.2, 0.9), False),
        ("TANGENT", 2.0, (0.7, 0.2, 0.9), False),
        ("TANGENT", -1.0, (0.7, 0.2, 0.9), False),
        ("TANGENT", 0.7, (0.5, 0.5, 0.5), False),
        ("TANGENT", 1.0, (0.65, 0.35, 0.95), True),
        ("TANGENT", 1.0, (1.0, 0.0, 0.5), True),
        ("OBJECT", 1.0, (0.8, 0.3, 0.9), False),
        ("WORLD", 1.0, (0.8, 0.3, 0.9), False),
        ("BLENDER_OBJECT", 1.0, (0.8, 0.3, 0.9), False),
        ("BLENDER_WORLD", 1.0, (0.8, 0.3, 0.9), False),
        ("OBJECT", 0.0, (0.2, 0.7, 0.4), False),
        ("WORLD", 0.5, (0.2, 0.7, 0.4), False),
        ("BLENDER_OBJECT", 2.0, (0.2, 0.7, 0.4), False),
        ("BLENDER_WORLD", -1.0, (0.2, 0.7, 0.4), False),
    )
    materials = []
    for index, (space, strength, color, _mirrored) in enumerate(cases):
        material, tree, output = _material(
            f"Normal Map Matrix {index:02d} {space}"
        )
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.name = f"Normal Map {index:02d} {space}"
        normal_map.space = space
        _input(normal_map, "Strength").default_value = strength
        _input(normal_map, "Color").default_value = (*color, 1.0)
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Normal Map Emission {index:02d}"
        tree.links.new(
            _output(normal_map, "Normal"),
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
        name="Normal Map Matrix",
        backfacing=set(range(12, 16)),
    )

    # Preserve the exact world-space cell rectangles while retaining a
    # nontrivial object transform. This distinguishes OBJECT from WORLD
    # without introducing reconstruction-filter differences at cell edges.
    surface.rotation_euler[2] = 0.41
    surface.scale = (1.7, 0.65, 1.3)
    bpy.context.view_layer.update()
    world_to_object = surface.matrix_world.inverted()
    for vertex in surface.data.vertices:
        vertex.co = world_to_object @ vertex.co

    uv_layer = surface.data.uv_layers.new(name="UVMap")
    for polygon in surface.data.polygons:
        mirrored = cases[polygon.material_index][3]
        coordinates = (
            ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0))
            if mirrored
            else ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        )
        for loop_index, coordinate in zip(
            polygon.loop_indices, coordinates, strict=True
        ):
            uv_layer.data[loop_index].uv = coordinate


def _normal_map_named_uv_matrix(scene: Any) -> None:
    """Select distinct MikkTSpace frames by Normal Map UV layer name."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("", 1.0, (0.75, 0.35, 0.9)),
        ("UV_A", 1.0, (0.75, 0.35, 0.9)),
        ("UV_B", 1.0, (0.75, 0.35, 0.9)),
        ("Missing", 1.0, (0.75, 0.35, 0.9)),
        ("", 0.0, (0.2, 0.8, 0.6)),
        ("UV_A", 0.5, (0.2, 0.8, 0.6)),
        ("UV_B", 0.5, (0.2, 0.8, 0.6)),
        ("Missing", 0.5, (0.2, 0.8, 0.6)),
        ("", 2.0, (1.0, 0.0, 0.5)),
        ("UV_A", -1.0, (1.0, 0.0, 0.5)),
        ("UV_B", 2.0, (1.0, 0.0, 0.5)),
        ("Missing", -1.0, (1.0, 0.0, 0.5)),
        ("", 1.0, (0.65, 0.35, 0.95)),
        ("UV_A", 1.0, (0.65, 0.35, 0.95)),
        ("UV_B", 1.0, (0.65, 0.35, 0.95)),
        ("Missing", 1.0, (0.65, 0.35, 0.95)),
    )
    materials = []
    for index, (uv_map, strength, color) in enumerate(cases):
        material, tree, output = _material(
            f"Named Normal Map Matrix {index:02d}"
        )
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.name = f"Named Normal Map {index:02d}"
        normal_map.space = "TANGENT"
        normal_map.uv_map = uv_map
        _input(normal_map, "Strength").default_value = strength
        _input(normal_map, "Color").default_value = (*color, 1.0)
        add = tree.nodes.new("ShaderNodeMixRGB")
        add.name = f"Named Normal Bias {index:02d}"
        add.blend_type = "ADD"
        _input(add, "Fac").default_value = 1.0
        _input(add, "Color2").default_value = (1.0, 1.0, 1.0, 1.0)
        scale = tree.nodes.new("ShaderNodeMixRGB")
        scale.name = f"Named Normal Scale {index:02d}"
        scale.blend_type = "MULTIPLY"
        _input(scale, "Fac").default_value = 1.0
        _input(scale, "Color2").default_value = (0.5, 0.5, 0.5, 1.0)
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Named Normal Emission {index:02d}"
        tree.links.new(
            _output(normal_map, "Normal"),
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
        name="Named Normal Map Matrix",
        backfacing=set(range(12, 16)),
    )
    uv_a = surface.data.uv_layers.new(name="UV_A", do_init=False)
    uv_b = surface.data.uv_layers.new(name="UV_B", do_init=False)
    surface.data.uv_layers.active = uv_a
    coordinates_a = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    coordinates_b = (
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )
    for polygon in surface.data.polygons:
        for loop_index, coordinate_a, coordinate_b in zip(
            polygon.loop_indices,
            coordinates_a,
            coordinates_b,
            strict=True,
        ):
            uv_a.data[loop_index].uv = coordinate_a
            uv_b.data[loop_index].uv = coordinate_b

    surface.rotation_euler = (0.19, -0.27, 0.41)
    surface.scale = (1.7, 0.65, 1.3)
    bpy.context.view_layer.update()
    world_to_object = surface.matrix_world.inverted()
    for vertex in surface.data.vertices:
        vertex.co = world_to_object @ vertex.co
