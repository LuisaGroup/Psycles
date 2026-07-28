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
import tempfile
from collections.abc import Callable
from typing import Any

import bpy
import numpy as np
import OpenImageIO as oiio


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.meshes,
        bpy.data.node_groups,
        bpy.data.worlds,
    ):
        for datablock in list(collection):
            collection.remove(datablock)


def _input(node: Any, name: str) -> Any:
    socket = node.inputs.get(name)
    if socket is None:
        raise RuntimeError(
            f"{node.bl_idname} has no input {name!r}; "
            f"available: {[item.name for item in node.inputs]}"
        )
    return socket


def _output(node: Any, name: str) -> Any:
    socket = node.outputs.get(name)
    if socket is None:
        raise RuntimeError(
            f"{node.bl_idname} has no output {name!r}; "
            f"available: {[item.name for item in node.outputs]}"
        )
    return socket


def _input_identifier(node: Any, identifier: str) -> Any:
    for socket in node.inputs:
        if socket.identifier == identifier:
            return socket
    raise RuntimeError(
        f"{node.bl_idname} has no input identifier {identifier!r}; "
        f"available: {[item.identifier for item in node.inputs]}"
    )


def _output_identifier(node: Any, identifier: str) -> Any:
    for socket in node.outputs:
        if socket.identifier == identifier:
            return socket
    raise RuntimeError(
        f"{node.bl_idname} has no output identifier {identifier!r}; "
        f"available: {[item.identifier for item in node.outputs]}"
    )


def _camera(scene: Any) -> None:
    data = bpy.data.cameras.new("Probe Camera")
    data.type = "ORTHO"
    data.ortho_scale = 2.2
    data.lens = 50.0
    camera = bpy.data.objects.new("Probe Camera", data)
    camera.location = (0.0, 0.0, 3.0)
    scene.collection.objects.link(camera)
    scene.camera = camera


def _plane(material: Any) -> None:
    bpy.ops.mesh.primitive_plane_add(
        # Fill the complete orthographic frame so value/closure probes have
        # no silhouette pixels whose reconstruction-filter coverage would
        # obscure the node formula being tested.
        size=8.0,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, 0.0),
    )
    plane = bpy.context.object
    plane.name = "Probe Surface"
    plane.data.materials.append(material)


def _material_matrix(
    scene: Any,
    materials: list[Any],
    columns: int,
    rows: int,
    name: str,
    *,
    backfacing: set[int] | None = None,
) -> Any:
    """Fill the orthographic frame with one material per contiguous cell."""
    if len(materials) != columns * rows:
        raise ValueError(
            f"{name}: expected {columns * rows} materials, "
            f"got {len(materials)}"
        )
    # The orthographic probe camera spans exactly [-1.1, 1.1]. Cell
    # boundaries therefore land on integer pixels whenever the render
    # dimensions are divisible by the matrix dimensions.
    extent = 1.1
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for index in range(len(materials)):
        column = index % columns
        row = index // columns
        x0 = -extent + 2.0 * extent * column / columns
        x1 = -extent + 2.0 * extent * (column + 1) / columns
        y0 = -extent + 2.0 * extent * row / rows
        y1 = -extent + 2.0 * extent * (row + 1) / rows
        first = len(vertices)
        vertices.extend(
            (
                (x0, y0, 0.0),
                (x1, y0, 0.0),
                (x1, y1, 0.0),
                (x0, y1, 0.0),
            )
        )
        face = (first, first + 1, first + 2, first + 3)
        if backfacing is not None and index in backfacing:
            face = tuple(reversed(face))
        faces.append(face)

    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(vertices, [], faces)
    for material in materials:
        mesh.materials.append(material)
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = index
    surface = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(surface)
    return surface


def _sphere(material: Any) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64,
        ring_count=32,
        radius=0.82,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, 0.0),
    )
    sphere = bpy.context.object
    sphere.name = "Probe Surface"
    sphere.data.materials.append(material)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    return sphere


def _material(name: str) -> tuple[Any, Any, Any]:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.name = "Material Output"
    return material, tree, output


def _world(
    scene: Any,
    color: tuple[float, float, float, float],
    strength: float,
) -> tuple[Any, Any, Any]:
    world = bpy.data.worlds.new("Probe World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    background = tree.nodes.new("ShaderNodeBackground")
    background.name = "Background"
    _input(background, "Color").default_value = color
    _input(background, "Strength").default_value = strength
    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.name = "World Output"
    tree.links.new(
        _output(background, "Background"),
        _input(output, "Surface"),
    )
    scene.world = world
    return world, tree, background


def _analytic_light_probe(
    scene: Any,
    light_type: str,
) -> Any:
    material, tree, output = _material(
        f"{light_type.title()} Light Probe Surface"
    )
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse"
    _input(diffuse, "Color").default_value = (
        0.62,
        0.41,
        0.23,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _plane(material)

    data = bpy.data.lights.new(
        f"{light_type.title()} Probe Light",
        type=light_type,
    )
    data.color = (0.36, 0.72, 1.0)
    data.use_shadow = True
    data.normalize = True
    light = bpy.data.objects.new(data.name, data)
    light.location = (0.37, -0.21, 1.4)
    scene.collection.objects.link(light)

    if light_type == "POINT":
        data.energy = 37.0
        data.shadow_soft_size = 0.0
    elif light_type == "SPOT":
        data.energy = 37.0
        data.shadow_soft_size = 0.0
        data.spot_size = 0.92
        data.spot_blend = 0.37
    elif light_type == "AREA":
        data.energy = 37.0
        data.shape = "RECTANGLE"
        data.size = 0.91
        data.size_y = 0.47
        data.spread = 3.141592653589793
    elif light_type == "SUN":
        data.energy = 1.7
        data.angle = 0.0
    else:
        raise ValueError(f"unsupported analytic light probe: {light_type}")

    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    return data


def _area_light(scene: Any) -> None:
    _analytic_light_probe(scene, "AREA")


def _area_light_ellipse(scene: Any) -> None:
    data = _analytic_light_probe(scene, "AREA")
    data.shape = "ELLIPSE"


def _area_light_spread(scene: Any) -> None:
    data = _analytic_light_probe(scene, "AREA")
    data.spread = 0.73


def _flat_light_distribution(scene: Any) -> None:
    """Exercise Cycles' mixed triangle/lamp flat CDF on one receiver."""
    world = scene.world
    if world is not None:
        world.cycles.sampling_method = "NONE"

    receiver, tree, output = _material(
        "Flat Distribution Receiver"
    )
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse"
    _input(diffuse, "Color").default_value = (
        0.52,
        0.31,
        0.17,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _plane(receiver)

    for index, (x, size, color, strength) in enumerate(
        (
            (-2.2, 0.5, (1.0, 0.18, 0.06, 1.0), 8.0),
            (2.2, 1.0, (0.08, 0.22, 1.0, 1.0), 3.0),
        )
    ):
        material, emitter_tree, emitter_output = _material(
            f"Flat Distribution Emitter {index}"
        )
        material.cycles.emission_sampling = "FRONT"
        emission = emitter_tree.nodes.new("ShaderNodeEmission")
        emission.name = "Emission"
        _input(emission, "Color").default_value = color
        _input(emission, "Strength").default_value = strength
        emitter_tree.links.new(
            _output(emission, "Emission"),
            _input(emitter_output, "Surface"),
        )
        bpy.ops.mesh.primitive_plane_add(
            size=size,
            enter_editmode=False,
            align="WORLD",
            location=(x, 0.0, 3.0),
            rotation=(3.141592653589793, 0.0, 0.0),
        )
        emitter = bpy.context.object
        emitter.name = f"Flat Distribution Emitter {index}"
        emitter.data.materials.append(material)

    data = bpy.data.lights.new(
        "Flat Distribution Point", type="POINT"
    )
    data.color = (0.19, 1.0, 0.27)
    data.energy = 24.0
    data.shadow_soft_size = 0.0
    light = bpy.data.objects.new(data.name, data)
    light.location = (0.0, 1.7, 2.4)
    scene.collection.objects.link(light)

    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0


def _point_light(scene: Any) -> None:
    _analytic_light_probe(scene, "POINT")


def _point_light_nodes(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.use_nodes = True
    emission = data.node_tree.nodes.get("Emission")
    if emission is None:
        raise RuntimeError("point light has no default Emission node")
    _input(emission, "Color").default_value = (
        0.77,
        0.31,
        0.58,
        1.0,
    )
    _input(emission, "Strength").default_value = 0.63


def _point_light_soft_disk(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.shadow_soft_size = 0.19
    data.use_soft_falloff = True


def _point_light_soft_sphere(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.shadow_soft_size = 0.19
    data.use_soft_falloff = False


def _spot_light(scene: Any) -> None:
    _analytic_light_probe(scene, "SPOT")


def _spot_light_soft(scene: Any) -> None:
    data = _analytic_light_probe(scene, "SPOT")
    data.shadow_soft_size = 0.13
    data.use_soft_falloff = False


def _sun_light(scene: Any) -> None:
    _analytic_light_probe(scene, "SUN")


def _sun_light_disk(scene: Any) -> None:
    data = _analytic_light_probe(scene, "SUN")
    data.angle = 0.17


def _rgb_emission(scene: Any) -> None:
    material, tree, output = _material("RGB Probe")
    rgb = tree.nodes.new("ShaderNodeRGB")
    rgb.name = "RGB"
    _output(rgb, "Color").default_value = (0.12, 0.37, 0.83, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Strength").default_value = 1.7
    tree.links.new(_output(rgb, "Color"), _input(emission, "Color"))
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _value_emission(scene: Any) -> None:
    material, tree, output = _material("Value Probe")
    value = tree.nodes.new("ShaderNodeValue")
    value.name = "Value"
    _output(value, "Value").default_value = 2.25
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (0.18, 0.61, 0.29, 1.0)
    tree.links.new(
        _output(value, "Value"), _input(emission, "Strength")
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _emission_surface(scene: Any) -> None:
    material, tree, output = _material("Emission Probe")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (0.73, 0.21, 0.08, 1.0)
    _input(emission, "Strength").default_value = 3.5
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _integrator_clamp_direct(scene: Any) -> None:
    """Exercise Blender UI clamp to Cycles device-clamp conversion."""
    material, tree, output = _material("Direct Clamp Probe")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (10.0, 10.0, 10.0, 1.0)
    _input(emission, "Strength").default_value = 1.0
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    scene.cycles.sample_clamp_direct = 2.0
    scene.cycles.sample_clamp_indirect = 0.0
    _plane(material)


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


def _linked_vector(
    tree: Any,
    name: str,
    value: tuple[float, float, float],
) -> Any:
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    node.name = name
    _input(node, "X").default_value = value[0]
    _input(node, "Y").default_value = value[1]
    _input(node, "Z").default_value = value[2]
    return _output(node, "Vector")


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


def _noise_color_3d(scene: Any) -> None:
    material, tree, output = _material("Noise Color 3D Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 3D"
    noise.noise_dimensions = "3D"
    noise.noise_type = "FBM"
    noise.normalize = True
    _input(noise, "Scale").default_value = 1.7
    _input(noise, "Detail").default_value = 2.35
    _input(noise, "Roughness").default_value = 0.61
    _input(noise, "Lacunarity").default_value = 2.2
    _input(noise, "Distortion").default_value = 0.37
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _noise_factor_2d(scene: Any) -> None:
    material, tree, output = _material("Noise Factor 2D Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 2D"
    noise.noise_dimensions = "2D"
    noise.noise_type = "FBM"
    noise.normalize = False
    _input(noise, "Scale").default_value = 2.3
    _input(noise, "Detail").default_value = 1.75
    _input(noise, "Roughness").default_value = 0.43
    _input(noise, "Lacunarity").default_value = 1.8
    _input(noise, "Distortion").default_value = 0.0
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Fac"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _noise_bump_object(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Noise Bump Object Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 3D"
    noise.noise_dimensions = "3D"
    noise.noise_type = "FBM"
    noise.normalize = True
    _input(noise, "Scale").default_value = 20.0
    _input(noise, "Detail").default_value = 2.0
    _input(noise, "Roughness").default_value = 0.5
    _input(noise, "Lacunarity").default_value = 2.0
    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = "Bump"
    _input(bump, "Strength").default_value = 0.2
    _input(bump, "Distance").default_value = 0.005
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.5,
        0.22,
        0.08,
        1.0,
    )
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Fac"),
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
    _plane(material)


def _noise_type_matrix(scene: Any, noise_type: str) -> None:
    """Tile every dimension/normalize/output combination for one noise type."""
    # The matrix intentionally contains discontinuities at exact pixel
    # boundaries. A narrow box filter keeps this a shader-formula comparison
    # instead of measuring Cycles' reconstruction filter against Psycles'
    # current film filter.
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials: list[Any] = []
    for normalize in (False, True):
        for dimensions in range(1, 5):
            for output_name in ("Fac", "Color"):
                material, tree, output = _material(
                    "Noise "
                    f"{noise_type} {dimensions}D "
                    f"{'Normalized' if normalize else 'Raw'} "
                    f"{output_name}"
                )
                noise = tree.nodes.new("ShaderNodeTexNoise")
                noise.name = (
                    f"Noise {noise_type} {dimensions}D "
                    f"{output_name}"
                )
                noise.noise_dimensions = f"{dimensions}D"
                noise.noise_type = noise_type
                noise.normalize = normalize
                if dimensions != 1:
                    vector = tree.nodes.new("ShaderNodeRGB")
                    vector.name = "Constant Noise Coordinates"
                    _output(vector, "Color").default_value = (
                        0.173,
                        -0.625,
                        1.375,
                        1.0,
                    )
                    tree.links.new(
                        _output(vector, "Color"),
                        _input_identifier(noise, "Vector"),
                    )
                _input_identifier(noise, "W").default_value = -0.437
                _input_identifier(noise, "Scale").default_value = 2.35
                _input_identifier(noise, "Detail").default_value = 2.375
                _input_identifier(
                    noise, "Roughness"
                ).default_value = 0.63
                _input_identifier(
                    noise, "Lacunarity"
                ).default_value = 2.17
                _input_identifier(noise, "Offset").default_value = 0.37
                _input_identifier(noise, "Gain").default_value = 1.11
                _input_identifier(
                    noise, "Distortion"
                ).default_value = 0.42
                emission = tree.nodes.new("ShaderNodeEmission")
                emission.name = "Emission"
                tree.links.new(
                    _output(noise, output_name),
                    _input(emission, "Color"),
                )
                tree.links.new(
                    _output(emission, "Emission"),
                    _input(output, "Surface"),
                )
                materials.append(material)

    # A single contiguous mesh eliminates reconstruction-filter background
    # edges. Each face has a separate material so a wrong mode remains
    # spatially visible instead of disappearing inside an aggregate.
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name=f"Noise {noise_type} Matrix",
    )


def _noise_fbm_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "FBM")


def _noise_multifractal_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "MULTIFRACTAL")


def _noise_ridged_multifractal_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "RIDGED_MULTIFRACTAL")


def _noise_hybrid_multifractal_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "HYBRID_MULTIFRACTAL")


def _noise_hetero_terrain_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "HETERO_TERRAIN")


def _gradient_spherical(scene: Any) -> None:
    material, tree, output = _material("Spherical Gradient Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.name = "Point Mapping"
    mapping.vector_type = "POINT"
    _input(mapping, "Scale").default_value = (0.6, 0.6, 0.6)
    gradient = tree.nodes.new("ShaderNodeTexGradient")
    gradient.name = "Spherical Gradient"
    gradient.gradient_type = "SPHERICAL"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(mapping, "Vector"),
    )
    tree.links.new(
        _output(mapping, "Vector"),
        _input(gradient, "Vector"),
    )
    tree.links.new(
        _output(gradient, "Fac"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _packed_rgba_image(
    name: str,
    pixels: np.ndarray,
    *,
    colorspace: str,
    alpha_mode: str,
) -> Any:
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False
    ) as temporary:
        texture_path = pathlib.Path(temporary.name)
    output_image = oiio.ImageOutput.create(str(texture_path))
    if output_image is None:
        raise RuntimeError("could not create probe PNG")
    try:
        height, width, channels = pixels.shape
        if channels != 4:
            raise ValueError("packed probe image must have four channels")
        if not output_image.open(
            str(texture_path),
            oiio.ImageSpec(width, height, channels, oiio.UINT8),
        ):
            raise RuntimeError("could not open probe PNG")
        if not output_image.write_image(pixels):
            raise RuntimeError("could not write probe PNG")
    finally:
        output_image.close()

    try:
        image = bpy.data.images.load(
            str(texture_path), check_existing=False
        )
        image.name = name
        image.colorspace_settings.name = colorspace
        image.alpha_mode = alpha_mode
        image.pack()
    finally:
        texture_path.unlink(missing_ok=True)
    return image


def _image_texture_srgb(scene: Any) -> None:
    """Exercise sRGB-before-filtering and straight-alpha sampling."""
    image = _packed_rgba_image(
        "Probe sRGB Texture",
        np.asarray(
            [
                [
                    (16, 64, 240, 32),
                    (240, 32, 16, 224),
                ],
                [
                    (32, 220, 64, 96),
                    (180, 128, 200, 160),
                ],
            ],
            dtype=np.uint8,
        ),
        colorspace="sRGB",
        alpha_mode="STRAIGHT",
    )

    material, tree, output = _material("sRGB Image Texture")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.name = "Image Texture"
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "EXTEND"
    separate = tree.nodes.new("ShaderNodeSeparateColor")
    separate.name = "Separate Image Color"
    separate.mode = "RGB"
    tree.links.new(
        _output(texture, "Color"),
        _input(separate, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Color and Alpha"
    combine.mode = "RGB"
    tree.links.new(
        _output(separate, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(separate, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(texture, "Alpha"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _image_texture_sampling_modes(scene: Any) -> None:
    """Exercise Cycles' 2D interpolation and extension cross-product."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    image = _packed_rgba_image(
        "Probe Raw Sampling Texture",
        np.asarray(
            [
                [
                    (7, 19, 233, 31),
                    (41, 211, 67, 83),
                    (173, 29, 109, 149),
                    (251, 137, 11, 223),
                    (97, 59, 197, 47),
                ],
                [
                    (227, 73, 17, 199),
                    (61, 151, 239, 101),
                    (131, 239, 53, 181),
                    (19, 97, 157, 59),
                    (199, 31, 79, 241),
                ],
                [
                    (83, 229, 127, 113),
                    (149, 43, 193, 167),
                    (239, 181, 23, 71),
                    (53, 7, 211, 137),
                    (167, 113, 61, 211),
                ],
                [
                    (31, 109, 179, 251),
                    (211, 67, 97, 43),
                    (107, 197, 37, 191),
                    (71, 241, 139, 89),
                    (233, 17, 151, 157),
                ],
            ],
            dtype=np.uint8,
        ),
        colorspace="Non-Color",
        alpha_mode="STRAIGHT",
    )
    coordinates = {
        "REPEAT": (-0.173, 1.217, 0.37),
        "EXTEND": (1.127, -0.083, 0.37),
        "CLIP": (-0.037, 0.617, 0.37),
        "MIRROR": (-0.283, 1.193, 0.37),
    }
    materials = []
    for interpolation in ("Closest", "Linear", "Cubic", "Smart"):
        for extension in ("REPEAT", "EXTEND", "CLIP", "MIRROR"):
            name = f"Image {interpolation} {extension}"
            material, tree, output = _material(name)
            mapping = tree.nodes.new("ShaderNodeMapping")
            mapping.name = f"{name} Coordinate"
            mapping.vector_type = "POINT"
            _input(mapping, "Vector").default_value = coordinates[extension]
            texture = tree.nodes.new("ShaderNodeTexImage")
            texture.name = name
            texture.image = image
            texture.interpolation = interpolation
            texture.extension = extension
            texture.projection = "FLAT"
            tree.links.new(
                _output(mapping, "Vector"),
                _input(texture, "Vector"),
            )
            channels = tree.nodes.new("ShaderNodeSeparateColor")
            channels.name = f"{name} Channels"
            channels.mode = "RGB"
            tree.links.new(
                _output(texture, "Color"),
                _input(channels, "Color"),
            )
            packed = tree.nodes.new("ShaderNodeCombineColor")
            packed.name = f"{name} RGB Alpha"
            packed.mode = "RGB"
            tree.links.new(
                _output(channels, "Red"),
                _input(packed, "Red"),
            )
            tree.links.new(
                _output(channels, "Green"),
                _input(packed, "Green"),
            )
            tree.links.new(
                _output(texture, "Alpha"),
                _input(packed, "Blue"),
            )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"{name} Emission"
            tree.links.new(
                _output(packed, "Color"),
                _input(emission, "Color"),
            )
            tree.links.new(
                _output(emission, "Emission"),
                _input(output, "Surface"),
            )
            materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Image Sampling Modes Matrix",
    )


def _image_texture_projection_modes(scene: Any) -> None:
    """Exercise flat, spherical, tube, and blended box projection."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    image = _packed_rgba_image(
        "Probe Projection Texture",
        np.asarray(
            [
                [
                    (11, 47, 229, 37),
                    (83, 197, 29, 113),
                    (179, 17, 131, 193),
                    (241, 109, 61, 73),
                ],
                [
                    (59, 223, 151, 167),
                    (211, 71, 101, 43),
                    (127, 239, 19, 227),
                    (31, 137, 199, 97),
                ],
                [
                    (197, 31, 79, 211),
                    (103, 181, 233, 59),
                    (53, 97, 173, 149),
                    (229, 251, 41, 181),
                ],
                [
                    (139, 89, 13, 251),
                    (17, 157, 217, 83),
                    (251, 61, 107, 137),
                    (73, 209, 163, 199),
                ],
            ],
            dtype=np.uint8,
        ),
        colorspace="Non-Color",
        alpha_mode="STRAIGHT",
    )
    cases = (
        ("FLAT", 0.0, (0.173, 0.617, 0.83), (0.0, 0.0, 1.0)),
        ("FLAT", 0.0, (-0.213, 1.137, 0.31), (0.0, 0.0, 1.0)),
        ("FLAT", 0.0, (0.917, 0.081, 0.57), (0.0, 0.0, 1.0)),
        ("FLAT", 0.0, (1.271, -0.191, 0.49), (0.0, 0.0, 1.0)),
        ("SPHERE", 0.0, (0.83, 0.71, 0.26), (0.0, 0.0, 1.0)),
        ("SPHERE", 0.0, (0.19, 0.87, 0.63), (0.0, 0.0, 1.0)),
        ("SPHERE", 0.0, (0.51, 0.49, 0.93), (0.0, 0.0, 1.0)),
        ("SPHERE", 0.0, (0.5, 0.5, 0.5), (0.0, 0.0, 1.0)),
        ("TUBE", 0.0, (0.83, 0.71, 0.26), (0.0, 0.0, 1.0)),
        ("TUBE", 0.0, (0.19, 0.87, 0.63), (0.0, 0.0, 1.0)),
        ("TUBE", 0.0, (0.51, 0.49, 0.93), (0.0, 0.0, 1.0)),
        ("TUBE", 0.0, (0.5, 0.5, 0.5), (0.0, 0.0, 1.0)),
        ("BOX", 0.0, (0.21, 0.73, 0.42), (0.93, 0.21, 0.30)),
        ("BOX", 0.2, (0.67, 0.18, 0.91), (-0.31, 0.89, 0.34)),
        ("BOX", 0.55, (0.37, 0.82, 0.14), (0.41, -0.52, 0.75)),
        ("BOX", 1.0, (0.76, 0.29, 0.58), (-0.58, -0.49, 0.65)),
    )
    materials = []
    normals = []
    for index, (projection, blend, coordinate, normal) in enumerate(cases):
        name = f"Image Projection {index:02d} {projection}"
        material, tree, output = _material(name)
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.name = f"{name} Coordinate"
        mapping.vector_type = "POINT"
        _input(mapping, "Vector").default_value = coordinate
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.name = name
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "REPEAT"
        texture.projection = projection
        texture.projection_blend = blend
        tree.links.new(
            _output(mapping, "Vector"),
            _input(texture, "Vector"),
        )
        channels = tree.nodes.new("ShaderNodeSeparateColor")
        channels.name = f"{name} Channels"
        channels.mode = "RGB"
        tree.links.new(
            _output(texture, "Color"),
            _input(channels, "Color"),
        )
        packed = tree.nodes.new("ShaderNodeCombineColor")
        packed.name = f"{name} RGB Alpha"
        packed.mode = "RGB"
        tree.links.new(
            _output(channels, "Red"),
            _input(packed, "Red"),
        )
        tree.links.new(
            _output(channels, "Green"),
            _input(packed, "Green"),
        )
        tree.links.new(
            _output(texture, "Alpha"),
            _input(packed, "Blue"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"{name} Emission"
        tree.links.new(
            _output(packed, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
        length = sum(component * component for component in normal) ** 0.5
        normals.append(tuple(component / length for component in normal))

    extent = 1.1
    vertices = []
    faces = []
    for index in range(len(materials)):
        column = index % 4
        row = index // 4
        x0 = -extent + 2.0 * extent * column / 4
        x1 = -extent + 2.0 * extent * (column + 1) / 4
        y0 = -extent + 2.0 * extent * row / 4
        y1 = -extent + 2.0 * extent * (row + 1) / 4
        first = len(vertices)
        vertices.extend(
            (
                (x0, y0, 0.0),
                (x1, y0, 0.0),
                (x1, y1, 0.0),
                (x0, y1, 0.0),
            )
        )
        faces.append((first, first + 1, first + 2, first + 3))
    mesh = bpy.data.meshes.new("Image Projection Modes Matrix Mesh")
    mesh.from_pydata(vertices, [], faces)
    for material in materials:
        mesh.materials.append(material)
    custom_normals = []
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = index
        polygon.use_smooth = True
        custom_normals.extend([normals[index]] * polygon.loop_total)
    mesh.normals_split_custom_set(custom_normals)
    surface = bpy.data.objects.new(
        "Image Projection Modes Matrix", mesh
    )
    scene.collection.objects.link(surface)


def _color_ramp_rgb(scene: Any) -> None:
    """Exercise scene-reachable linear and constant RGB ramps."""
    material, tree, output = _material("RGB Color Ramp Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    gradient = tree.nodes.new("ShaderNodeTexGradient")
    gradient.name = "Linear Gradient"
    gradient.gradient_type = "LINEAR"
    tree.links.new(
        _output(coordinates, "Generated"),
        _input(gradient, "Vector"),
    )

    linear = tree.nodes.new("ShaderNodeValToRGB")
    linear.name = "Linear RGB Ramp"
    linear.color_ramp.color_mode = "RGB"
    linear.color_ramp.interpolation = "LINEAR"
    linear.color_ramp.elements[0].position = 0.13
    linear.color_ramp.elements[0].color = (0.04, 0.17, 0.73, 0.21)
    middle = linear.color_ramp.elements.new(0.47)
    middle.color = (0.82, 0.09, 0.24, 0.68)
    linear.color_ramp.elements[1].position = 0.82
    linear.color_ramp.elements[1].color = (0.15, 0.91, 0.36, 0.94)
    tree.links.new(
        _output(gradient, "Fac"),
        _input(linear, "Fac"),
    )

    constant = tree.nodes.new("ShaderNodeValToRGB")
    constant.name = "Constant RGB Ramp"
    constant.color_ramp.color_mode = "RGB"
    constant.color_ramp.interpolation = "CONSTANT"
    constant.color_ramp.elements[0].position = 0.19
    constant.color_ramp.elements[0].color = (0.91, 0.11, 0.07, 0.32)
    middle = constant.color_ramp.elements.new(0.58)
    middle.color = (0.13, 0.77, 0.31, 0.61)
    constant.color_ramp.elements[1].position = 0.88
    constant.color_ramp.elements[1].color = (0.22, 0.36, 0.95, 0.86)
    tree.links.new(
        _output(gradient, "Fac"),
        _input(constant, "Fac"),
    )

    linear_channels = tree.nodes.new("ShaderNodeSeparateColor")
    linear_channels.mode = "RGB"
    constant_channels = tree.nodes.new("ShaderNodeSeparateColor")
    constant_channels.mode = "RGB"
    tree.links.new(
        _output(linear, "Color"),
        _input(linear_channels, "Color"),
    )
    tree.links.new(
        _output(constant, "Color"),
        _input(constant_channels, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.mode = "RGB"
    tree.links.new(
        _output(linear_channels, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(constant_channels, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(linear, "Alpha"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _color_ramp_material(
    name: str,
    factor: float,
    *,
    color_mode: str,
    interpolation: str,
    hue_interpolation: str = "NEAR",
    output_name: str = "Color",
) -> Any:
    material, tree, output = _material(name)
    value = tree.nodes.new("ShaderNodeValue")
    value.name = f"{name} Factor"
    _output(value, "Value").default_value = factor

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.name = name
    ramp.color_ramp.color_mode = color_mode
    ramp.color_ramp.interpolation = interpolation
    ramp.color_ramp.hue_interpolation = hue_interpolation
    elements = ramp.color_ramp.elements
    elements[0].position = 0.11
    elements[0].color = (0.97, 0.025, 0.18, 0.13)
    middle_a = elements.new(0.39)
    middle_a.color = (0.03, 0.88, 0.74, 0.88)
    middle_b = elements.new(0.71)
    middle_b.color = (0.12, 0.055, 0.96, 0.31)
    elements[1].position = 0.91
    elements[1].color = (0.91, 0.72, 0.035, 0.76)
    tree.links.new(
        _output(value, "Value"),
        _input(ramp, "Fac"),
    )

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(ramp, output_name),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _color_ramp_modes(scene: Any) -> None:
    """Exercise every Blender Color Ramp color/interpolation mode."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = [
        ("RGB", interpolation, "NEAR", factor)
        for interpolation, factor in zip(
            ("LINEAR", "CONSTANT", "EASE", "CARDINAL", "B_SPLINE"),
            (0.173, 0.287, 0.443, 0.619, 0.823),
            strict=True,
        )
    ]
    cases.extend(
        ("HSV", "LINEAR", hue, factor)
        for hue, factor in zip(
            ("NEAR", "FAR", "CW", "CCW"),
            (0.241, 0.367, 0.557, 0.779),
            strict=True,
        )
    )
    cases.extend(
        ("HSL", "LINEAR", hue, factor)
        for hue, factor in zip(
            ("NEAR", "FAR", "CW", "CCW"),
            (0.197, 0.331, 0.593, 0.857),
            strict=True,
        )
    )
    # The final three cells verify Cycles' clamping at both ends and a
    # factor exactly on a 1/256 normalized lookup-table sample.
    cases.extend(
        (
            ("RGB", "LINEAR", "NEAR", -0.25),
            ("RGB", "EASE", "NEAR", 1.25),
            ("HSV", "LINEAR", "FAR", 127.0 / 256.0),
        )
    )
    materials = [
        _color_ramp_material(
            f"Color Ramp {index:02d} "
            f"{color_mode} {interpolation} {hue}",
            factor,
            color_mode=color_mode,
            interpolation=interpolation,
            hue_interpolation=hue,
        )
        for index, (
            color_mode,
            interpolation,
            hue,
            factor,
        ) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Color Ramp Modes Matrix",
    )


def _color_ramp_alpha_modes(scene: Any) -> None:
    """Exercise sampled alpha interpolation, clamp, and table boundaries."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("LINEAR", 0.173),
        ("CONSTANT", 0.287),
        ("EASE", 0.443),
        ("CARDINAL", 0.619),
        ("B_SPLINE", 0.823),
        ("LINEAR", -0.25),
        ("EASE", 1.25),
        ("CARDINAL", 129.0 / 256.0),
    )
    materials = [
        _color_ramp_material(
            f"Color Ramp Alpha {index:02d} {interpolation}",
            factor,
            color_mode="RGB",
            interpolation=interpolation,
            output_name="Alpha",
        )
        for index, (interpolation, factor) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="Color Ramp Alpha Matrix",
    )


def _rgb_curve_material(
    name: str,
    *,
    factor: float,
    color: tuple[float, float, float, float],
    extend: str,
    wide_domain: bool,
) -> Any:
    material, tree, output = _material(name)
    factor_node = tree.nodes.new("ShaderNodeValue")
    factor_node.name = f"{name} Factor"
    _output(factor_node, "Value").default_value = factor
    color_node = tree.nodes.new("ShaderNodeRGB")
    color_node.name = f"{name} Color"
    _output(color_node, "Color").default_value = color

    curves = tree.nodes.new("ShaderNodeRGBCurve")
    curves.name = name
    mapping = curves.mapping
    mapping.extend = extend
    mapping.use_clip = False
    if wide_domain:
        endpoints = (
            (-0.25, 1.15),
            (-0.05, 1.05),
            (-0.10, 1.30),
            (-0.20, 1.20),
        )
    else:
        endpoints = ((0.0, 1.0),) * 4
    shapes = (
        ((0.08, 0.87), (0.31, 0.76), (0.74, 0.21)),
        ((0.17, 0.94), (0.29, 0.16), (0.68, 0.83)),
        ((0.04, 0.72), (0.43, 0.91), (0.79, 0.12)),
        ((0.11, 0.89), (0.37, 0.24), (0.63, 0.78)),
    )
    for curve, (domain_min, domain_max), shape in zip(
        mapping.curves,
        endpoints,
        shapes,
        strict=True,
    ):
        curve.points[0].location = (domain_min, shape[0][1])
        curve.points[-1].location = (domain_max, shape[-1][1])
        domain_range = domain_max - domain_min
        for relative_x, y in shape[1:-1]:
            point = curve.points.new(
                domain_min + relative_x * domain_range,
                y,
            )
            point.handle_type = "AUTO"
    mapping.update()

    tree.links.new(
        _output(factor_node, "Value"),
        _input(curves, "Fac"),
    )
    tree.links.new(
        _output(color_node, "Color"),
        _input(curves, "Color"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(curves, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _rgb_curve_matrix(scene: Any) -> None:
    """Cover normalized tables, domains, extrapolation, and Fac mixing."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (1.0, (0.17, 0.43, 0.81, 1.0), "EXTRAPOLATED", False),
        (0.37, (0.71, 0.22, 0.49, 1.0), "EXTRAPOLATED", False),
        (0.0, (0.29, 0.63, 0.11, 1.0), "EXTRAPOLATED", False),
        (1.4, (0.57, 0.08, 0.92, 1.0), "EXTRAPOLATED", False),
        (-0.4, (0.33, 0.77, 0.26, 1.0), "HORIZONTAL", False),
        (1.0, (-0.61, 1.72, 0.46, 1.0), "HORIZONTAL", True),
        (1.0, (-0.61, 1.72, 0.46, 1.0), "EXTRAPOLATED", True),
        (0.58, (1.41, -0.32, 0.73, 1.0), "EXTRAPOLATED", True),
    )
    materials = [
        _rgb_curve_material(
            f"RGB Curve {index:02d} {extend}",
            factor=factor,
            color=color,
            extend=extend,
            wide_domain=wide_domain,
        )
        for index, (
            factor,
            color,
            extend,
            wide_domain,
        ) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="RGB Curve Matrix",
    )


def _mapping_material(
    name: str,
    vector_type: str,
    *,
    edge_scale: bool,
) -> Any:
    material, tree, output = _material(name)
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.name = name
    mapping.vector_type = vector_type
    _input_identifier(mapping, "Vector").default_value = (
        0.17,
        0.31,
        0.47,
    )
    _input_identifier(mapping, "Location").default_value = (
        0.05,
        0.04,
        0.03,
    )
    _input_identifier(mapping, "Rotation").default_value = (
        0.27,
        -0.19,
        0.33,
    )
    _input_identifier(mapping, "Scale").default_value = (
        (0.0, -0.7, 1.3)
        if edge_scale
        else (1.1, 0.8, 1.3)
    )

    separate = tree.nodes.new("ShaderNodeSeparateColor")
    separate.name = f"{name} Separate"
    separate.mode = "RGB"
    tree.links.new(
        _output(mapping, "Vector"),
        _input(separate, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = f"{name} Encode"
    combine.mode = "RGB"
    for channel in ("Red", "Green", "Blue"):
        encode = tree.nodes.new("ShaderNodeMath")
        encode.name = f"{name} Encode {channel}"
        encode.operation = "MULTIPLY_ADD"
        tree.links.new(
            _output(separate, channel),
            _input(encode, "Value"),
        )
        encode.inputs[1].default_value = 0.25
        encode.inputs[2].default_value = 0.5
        tree.links.new(
            _output(encode, "Value"),
            _input(combine, channel),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _mapping_modes(scene: Any) -> None:
    """Cover all Mapping vector types and zero/negative scale semantics."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = [
        _mapping_material(
            f"Mapping {vector_type} "
            f"{'Edge' if edge_scale else 'Regular'}",
            vector_type,
            edge_scale=edge_scale,
        )
        for edge_scale in (False, True)
        for vector_type in ("POINT", "TEXTURE", "VECTOR", "NORMAL")
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="Mapping Modes Matrix",
    )


def _checker_texture_matrix(scene: Any) -> None:
    """Cover Checker precision correction, signs, scale, Color, and Fac."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ((0.0, 0.0, 0.0), 1.0),
        ((1.0, 0.0, 0.0), 1.0),
        ((-1.0, 0.0, 0.0), 1.0),
        ((0.0, 0.0, 1.0), 1.0),
        ((0.5, 0.5, 0.5), 2.0),
        ((-0.5, 0.5, 0.5), 2.0),
        ((1.25, -2.5, 3.75), 0.8),
        ((-1.25, 2.5, -3.75), 0.8),
        ((0.999999, 1.000001, -0.999999), 1.0),
        ((17.0, 18.0, 19.0), 0.25),
        ((-17.0, -18.0, -19.0), 0.25),
        ((0.13, 0.37, 0.91), 7.3),
        ((0.13, 0.37, 0.91), -7.3),
        ((41.0, -29.0, 13.0), 0.0),
        ((1024.25, -2048.5, 4096.75), 0.125),
        ((-0.000001, 0.000001, -0.000001), 1000000.0),
    )
    materials = []
    for index, (coordinate, scale) in enumerate(cases):
        material, tree, output = _material(
            f"Checker {index:02d}"
        )
        coordinate_node = tree.nodes.new("ShaderNodeRGB")
        coordinate_node.name = f"Checker Coordinates {index:02d}"
        _output(coordinate_node, "Color").default_value = (
            *coordinate,
            1.0,
        )
        checker = tree.nodes.new("ShaderNodeTexChecker")
        checker.name = f"Checker {index:02d}"
        tree.links.new(
            _output(coordinate_node, "Color"),
            _input(checker, "Vector"),
        )
        _input(checker, "Color1").default_value = (
            0.13,
            0.37,
            0.79,
            1.0,
        )
        _input(checker, "Color2").default_value = (
            0.83,
            0.61,
            0.17,
            1.0,
        )
        _input(checker, "Scale").default_value = scale
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Checker Separate {index:02d}"
        separate.mode = "RGB"
        tree.links.new(
            _output(checker, "Color"),
            _input(separate, "Color"),
        )
        combine = tree.nodes.new("ShaderNodeCombineColor")
        combine.name = f"Checker Encode {index:02d}"
        combine.mode = "RGB"
        tree.links.new(
            _output(separate, "Red"),
            _input(combine, "Red"),
        )
        tree.links.new(
            _output(separate, "Green"),
            _input(combine, "Green"),
        )
        tree.links.new(
            _output(checker, "Fac"),
            _input(combine, "Blue"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Checker Emission {index:02d}"
        tree.links.new(
            _output(combine, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Checker Texture Matrix",
    )


def _fresnel_matrix(scene: Any) -> None:
    """Cover Fresnel IOR, grazing angles, clamp, and backfacing eta."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (1.5, (0.0, 0.0, 1.0)),
        (1.0, (0.0, 0.0, 1.0)),
        (0.0, (0.0, 0.0, 1.0)),
        (0.5, (0.0, 0.0, 1.0)),
        (1.33, (0.9682458, 0.0, 0.25)),
        (2.5, (0.8660254, 0.0, 0.5)),
        (1.1, (0.4358899, 0.0, 0.9)),
        (10.0, (1.0, 0.0, 0.0)),
        (1.5, (0.0, 0.0, 1.0)),
        (0.5, (0.0, 0.0, 1.0)),
        (1.33, (0.9682458, 0.0, 0.25)),
        (2.5, (0.8660254, 0.0, 0.5)),
        (1.0, (1.0, 0.0, 0.0)),
        (0.0, (0.4358899, 0.0, 0.9)),
        (-2.0, (0.9682458, 0.0, 0.25)),
        (10.0, (0.9999995, 0.0, 0.001)),
    )
    materials = []
    for index, (ior, normal) in enumerate(cases):
        material, tree, output = _material(
            f"Fresnel {index:02d}"
        )
        normal_node = tree.nodes.new("ShaderNodeRGB")
        normal_node.name = f"Fresnel Normal {index:02d}"
        _output(normal_node, "Color").default_value = (
            *normal,
            1.0,
        )
        fresnel = tree.nodes.new("ShaderNodeFresnel")
        fresnel.name = f"Fresnel {index:02d}"
        _input(fresnel, "IOR").default_value = ior
        tree.links.new(
            _output(normal_node, "Color"),
            _input(fresnel, "Normal"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Fresnel Emission {index:02d}"
        tree.links.new(
            _output(fresnel, "Fac"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Fresnel Matrix",
        backfacing=set(range(8, 16)),
    )


def _layer_weight_matrix(scene: Any) -> None:
    """Cover both outputs, blend branches, normals, and backfacing hits."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (-2.0, None, False),
        (-0.5, (0.0, 0.0, 1.0), False),
        (0.0, (0.0, 0.0, 1.0), False),
        (0.25, (0.4358899, 0.0, 0.9), False),
        (0.499, (0.21650635, 0.0, 0.125), False),
        (0.5, (0.9682458, 0.0, 0.25), False),
        (0.75, (0.9999995, 0.0, 0.001), False),
        (0.99999, (1.0, 0.0, 0.0), False),
        (1.0, (0.0, 0.0, 1.0), False),
        (2.0, (0.4358899, 0.0, 0.9), False),
        (0.25, (0.0, 0.0, 0.0), False),
        (0.75, (-0.9682458, 0.0, -0.25), False),
        (0.0, None, True),
        (0.25, (0.4358899, 0.0, 0.9), True),
        (0.75, (0.9682458, 0.0, 0.25), True),
        (1.0, (0.0, 0.0, 1.0), True),
    )
    materials = []
    backfacing: set[int] = set()
    for case_index, (blend, normal, is_backfacing) in enumerate(cases):
        for output_name in ("Fresnel", "Facing"):
            index = len(materials)
            material, tree, output = _material(
                f"Layer Weight {case_index:02d} {output_name}"
            )
            blend_node = tree.nodes.new("ShaderNodeValue")
            blend_node.name = f"Layer Blend {index:02d}"
            _output(blend_node, "Value").default_value = blend
            layer = tree.nodes.new("ShaderNodeLayerWeight")
            layer.name = f"Layer Weight {index:02d}"
            tree.links.new(
                _output(blend_node, "Value"),
                _input(layer, "Blend"),
            )
            if normal is not None:
                normal_node = tree.nodes.new("ShaderNodeCombineXYZ")
                normal_node.name = f"Layer Normal {index:02d}"
                _input(normal_node, "X").default_value = normal[0]
                _input(normal_node, "Y").default_value = normal[1]
                _input(normal_node, "Z").default_value = normal[2]
                tree.links.new(
                    _output(normal_node, "Vector"),
                    _input(layer, "Normal"),
                )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"Layer Emission {index:02d}"
            tree.links.new(
                _output(layer, output_name),
                _input(emission, "Color"),
            )
            tree.links.new(
                _output(emission, "Emission"),
                _input(output, "Surface"),
            )
            if is_backfacing:
                backfacing.add(index)
            materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=8,
        rows=4,
        name="Layer Weight Matrix",
        backfacing=backfacing,
    )


def _map_range_matrix(scene: Any) -> None:
    """Cover scalar/vector Map Range modes, guards, steps, and clamp."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scalar_cases = (
        ("LINEAR", False, 0.25, 0.0, 1.0, 0.1, 0.9, 4.0),
        ("LINEAR", True, 1.5, 0.0, 1.0, 0.8, 0.2, 4.0),
        ("LINEAR", True, 5.0, 2.0, 2.0, 0.3, 0.7, 4.0),
        ("STEPPED", False, 0.37, 0.0, 1.0, 0.1, 0.9, 4.0),
        ("STEPPED", False, 0.7, 0.0, 1.0, 0.2, 0.8, 0.0),
        ("STEPPED", True, -0.2, 0.0, 1.0, 0.2, 0.8, -3.0),
        ("SMOOTHSTEP", False, 1.5, 0.0, 1.0, 0.1, 0.9, 4.0),
        ("SMOOTHERSTEP", True, 0.25, 1.0, -1.0, 0.9, 0.1, 5.0),
    )
    vector_cases = (
        (
            "LINEAR",
            False,
            (0.25, 0.5, 0.75),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 2.0),
            (0.1, 0.2, 0.3),
            (0.9, 0.8, 0.7),
            (4.0, 4.0, 4.0),
        ),
        (
            "LINEAR",
            True,
            (2.0, -1.0, 0.5),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.8, 0.1, 0.9),
            (0.2, 0.7, 0.3),
            (4.0, 4.0, 4.0),
        ),
        (
            "STEPPED",
            False,
            (0.37, 0.7, 0.7),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.1, 0.2, 0.3),
            (0.9, 0.8, 0.7),
            (4.0, 0.0, -2.0),
        ),
        (
            "STEPPED",
            True,
            (-0.2, 1.4, 0.6),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.2, 0.8, 0.1),
            (0.8, 0.2, 0.9),
            (3.0, 2.0, 5.0),
        ),
        (
            "SMOOTHSTEP",
            False,
            (-1.0, 0.5, 2.0),
            (0.0, 1.0, 2.0),
            (1.0, -1.0, 2.0),
            (0.1, 0.2, 0.3),
            (0.9, 0.8, 0.7),
            (4.0, 4.0, 4.0),
        ),
        (
            "SMOOTHERSTEP",
            True,
            (0.2, 0.5, 0.8),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.9, 0.1, 0.8),
            (0.1, 0.7, 0.2),
            (4.0, 4.0, 4.0),
        ),
        (
            "LINEAR",
            False,
            (0.25, -0.5, 2.0),
            (1.0, -1.0, 3.0),
            (-1.0, 1.0, 1.0),
            (0.2, 0.3, 0.4),
            (0.8, 0.7, 0.6),
            (4.0, 4.0, 4.0),
        ),
        (
            "SMOOTHERSTEP",
            False,
            (-2.0, 0.35, 3.0),
            (0.0, 0.0, 2.0),
            (1.0, 1.0, -2.0),
            (0.8, 0.1, 0.9),
            (0.2, 0.7, 0.3),
            (4.0, 4.0, 4.0),
        ),
    )

    materials = []
    for index, (
        interpolation,
        use_clamp,
        value,
        from_min,
        from_max,
        to_min,
        to_max,
        steps,
    ) in enumerate(scalar_cases):
        material, tree, output = _material(
            f"Map Range Scalar {index:02d}"
        )
        map_range = tree.nodes.new("ShaderNodeMapRange")
        map_range.name = f"Map Range Scalar {index:02d}"
        map_range.data_type = "FLOAT"
        map_range.interpolation_type = interpolation
        map_range.clamp = use_clamp
        for identifier, default in (
            ("Value", value),
            ("From Min", from_min),
            ("From Max", from_max),
            ("To Min", to_min),
            ("To Max", to_max),
            ("Steps", steps),
        ):
            _input_identifier(
                map_range, identifier
            ).default_value = default
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Map Range Scalar Emission {index:02d}"
        tree.links.new(
            _output_identifier(map_range, "Result"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    for local_index, (
        interpolation,
        use_clamp,
        value,
        from_min,
        from_max,
        to_min,
        to_max,
        steps,
    ) in enumerate(vector_cases):
        index = len(scalar_cases) + local_index
        material, tree, output = _material(
            f"Map Range Vector {local_index:02d}"
        )
        map_range = tree.nodes.new("ShaderNodeMapRange")
        map_range.name = f"Map Range Vector {local_index:02d}"
        map_range.data_type = "FLOAT_VECTOR"
        map_range.interpolation_type = interpolation
        map_range.clamp = use_clamp
        for identifier, default in (
            ("Vector", value),
            ("From_Min_FLOAT3", from_min),
            ("From_Max_FLOAT3", from_max),
            ("To_Min_FLOAT3", to_min),
            ("To_Max_FLOAT3", to_max),
            ("Steps_FLOAT3", steps),
        ):
            _input_identifier(
                map_range, identifier
            ).default_value = default
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Map Range Vector Emission {local_index:02d}"
        tree.links.new(
            _output_identifier(map_range, "Vector"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Map Range Matrix",
    )


def _vector_math_matrix(scene: Any) -> None:
    """Cover every Cycles Vector Math operation and guarded edge paths."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    operations = (
        "ADD",
        "SUBTRACT",
        "MULTIPLY",
        "DIVIDE",
        "MULTIPLY_ADD",
        "CROSS_PRODUCT",
        "PROJECT",
        "REFLECT",
        "REFRACT",
        "FACEFORWARD",
        "DOT_PRODUCT",
        "DISTANCE",
        "LENGTH",
        "SCALE",
        "NORMALIZE",
        "ABSOLUTE",
        "POWER",
        "SIGN",
        "MINIMUM",
        "MAXIMUM",
        "FLOOR",
        "CEIL",
        "FRACTION",
        "MODULO",
        "WRAP",
        "SNAP",
        "SINE",
        "COSINE",
        "TANGENT",
    )
    base_a = (0.17, -0.63, 1.2)
    base_b = (0.7, -0.2, 0.5)
    base_c = (-0.4, 0.8, 0.1)
    base_scale = 0.73
    operation_inputs = {
        "POWER": (
            (-2.0, -4.0, 8.0),
            (3.0, 2.0, 0.5),
            base_c,
            base_scale,
        ),
        "SIGN": (
            (-2.0, 0.0, 3.0),
            base_b,
            base_c,
            base_scale,
        ),
        "MODULO": (
            (-0.7, 0.7, 1.5),
            (0.2, -0.2, 0.5),
            base_c,
            base_scale,
        ),
        "WRAP": (
            (-0.2, 2.4, 0.75),
            (1.0, 1.0, 1.0),
            (0.0, -1.0, 0.25),
            base_scale,
        ),
        "SNAP": (
            (0.73, -0.73, 1.4),
            (0.25, -0.2, 0.5),
            base_c,
            base_scale,
        ),
        "REFRACT": (
            (0.1, 0.0, -0.9949874),
            (0.0, 0.0, 2.0),
            base_c,
            0.67,
        ),
    }
    cases = [
        (
            operation,
            operation,
            *operation_inputs.get(
                operation,
                (base_a, base_b, base_c, base_scale),
            ),
        )
        for operation in operations
    ]
    cases.extend(
        (
            (
                "DIVIDE_ZERO_COMPONENTS",
                "DIVIDE",
                (1.0, -2.0, 4.0),
                (0.0, 2.0, 0.0),
                base_c,
                base_scale,
            ),
            (
                "PROJECT_ZERO_AXIS",
                "PROJECT",
                base_a,
                (0.0, 0.0, 0.0),
                base_c,
                base_scale,
            ),
            (
                "REFLECT_ZERO_NORMAL",
                "REFLECT",
                base_a,
                (0.0, 0.0, 0.0),
                base_c,
                base_scale,
            ),
            (
                "REFRACT_TOTAL_INTERNAL_REFLECTION",
                "REFRACT",
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                base_c,
                1.5,
            ),
            (
                "FACEFORWARD_OPPOSITE",
                "FACEFORWARD",
                base_a,
                (0.5, 0.25, 0.75),
                (0.5, 0.25, 0.75),
                base_scale,
            ),
            (
                "NORMALIZE_ZERO",
                "NORMALIZE",
                (0.0, 0.0, 0.0),
                base_b,
                base_c,
                base_scale,
            ),
            (
                "POWER_INVALID_NEGATIVE",
                "POWER",
                (-2.0, -4.0, -8.0),
                (0.5, -0.5, 0.25),
                base_c,
                base_scale,
            ),
            (
                "MODULO_ZERO_DIVISORS",
                "MODULO",
                base_a,
                (0.0, -0.2, 0.0),
                base_c,
                base_scale,
            ),
            (
                "WRAP_ZERO_RANGE",
                "WRAP",
                base_a,
                (1.0, 0.5, -0.25),
                (1.0, -0.5, -0.25),
                base_scale,
            ),
            (
                "SNAP_ZERO_INCREMENT",
                "SNAP",
                base_a,
                (0.0, -0.2, 0.0),
                base_c,
                base_scale,
            ),
            (
                "CROSS_PARALLEL",
                "CROSS_PRODUCT",
                (0.5, -1.0, 2.0),
                (1.0, -2.0, 4.0),
                base_c,
                base_scale,
            ),
        )
    )
    if len(cases) != 40:
        raise RuntimeError(
            f"Vector Math matrix must contain 40 cells, got {len(cases)}"
        )

    scalar_operations = {"DOT_PRODUCT", "DISTANCE", "LENGTH"}
    materials = []
    for index, (label, operation, a, b, c, scale) in enumerate(cases):
        material, tree, output = _material(
            f"Vector Math Matrix {index:02d} {label}"
        )
        vector_math = tree.nodes.new("ShaderNodeVectorMath")
        vector_math.name = f"Vector Math {index:02d} {label}"
        vector_math.operation = operation
        _input_identifier(vector_math, "Vector").default_value = a
        _input_identifier(
            vector_math, "Vector_001"
        ).default_value = b
        _input_identifier(
            vector_math, "Vector_002"
        ).default_value = c
        _input_identifier(vector_math, "Scale").default_value = scale
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Vector Math Emission {index:02d}"
        result_identifier = (
            "Value" if operation in scalar_operations else "Vector"
        )
        tree.links.new(
            _output_identifier(vector_math, result_identifier),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=8,
        rows=5,
        name="Vector Math Matrix",
    )


def _blackbody_matrix(scene: Any) -> None:
    """Cover every Cycles blackbody polynomial interval and clamp."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    temperatures = (
        -100.0,
        0.0,
        799.0,
        800.0,
        964.0,
        965.0,
        1166.0,
        1167.0,
        1449.0,
        1902.0,
        3315.0,
        6365.0,
        6500.0,
        11999.0,
        12000.0,
        20000.0,
    )
    materials = []
    for index, temperature in enumerate(temperatures):
        material, tree, output = _material(
            f"Blackbody Matrix {index:02d} {temperature:g}K"
        )
        value = tree.nodes.new("ShaderNodeValue")
        value.name = f"Temperature {index:02d}"
        _output(value, "Value").default_value = temperature
        blackbody = tree.nodes.new("ShaderNodeBlackbody")
        blackbody.name = f"Blackbody {index:02d}"
        tree.links.new(
            _output(value, "Value"),
            _input(blackbody, "Temperature"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Blackbody Emission {index:02d}"
        tree.links.new(
            _output(blackbody, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Blackbody Matrix",
    )


def _wavelength_matrix(scene: Any) -> None:
    """Cover Cycles CIE interpolation, truncation, and range guards."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    wavelengths = (
        -100.0,
        374.9,
        375.0,
        379.0,
        380.0,
        382.5,
        385.0,
        400.0,
        445.0,
        500.0,
        520.1,
        555.0,
        600.0,
        650.0,
        700.0,
        775.0,
        779.999,
        780.0,
        781.0,
        1000.0,
    )
    materials = []
    for index, wavelength in enumerate(wavelengths):
        material, tree, output = _material(
            f"Wavelength Matrix {index:02d} {wavelength:g}nm"
        )
        value = tree.nodes.new("ShaderNodeValue")
        value.name = f"Wavelength Value {index:02d}"
        _output(value, "Value").default_value = wavelength
        wavelength_node = tree.nodes.new("ShaderNodeWavelength")
        wavelength_node.name = f"Wavelength {index:02d}"
        tree.links.new(
            _output(value, "Value"),
            _input(wavelength_node, "Wavelength"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Wavelength Emission {index:02d}"
        tree.links.new(
            _output(wavelength_node, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=5,
        rows=4,
        name="Wavelength Matrix",
    )


def _invert_color_matrix(scene: Any) -> None:
    """Cover linked unbounded factors and signed HDR color inputs."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (-2.0, (0.2, 0.4, 0.8, 1.0)),
        (-1.0, (0.2, 0.4, 0.8, 1.0)),
        (0.0, (0.2, 0.4, 0.8, 1.0)),
        (0.25, (0.2, 0.4, 0.8, 1.0)),
        (0.5, (0.2, 0.4, 0.8, 1.0)),
        (1.0, (0.2, 0.4, 0.8, 1.0)),
        (2.0, (0.2, 0.4, 0.8, 1.0)),
        (3.0, (0.2, 0.4, 0.8, 1.0)),
        (-0.5, (-0.3, 1.4, 2.1, 0.2)),
        (0.0, (-0.3, 1.4, 2.1, 0.2)),
        (0.25, (-0.3, 1.4, 2.1, 0.2)),
        (0.75, (-0.3, 1.4, 2.1, 0.2)),
        (1.0, (-0.3, 1.4, 2.1, 0.2)),
        (1.5, (-0.3, 1.4, 2.1, 0.2)),
        (-3.0, (0.2, 0.4, 0.8, 1.0)),
        (4.0, (0.2, 0.4, 0.8, 1.0)),
    )
    materials = []
    for index, (factor, color) in enumerate(cases):
        material, tree, output = _material(
            f"Invert Color Matrix {index:02d}"
        )
        factor_node = tree.nodes.new("ShaderNodeValue")
        factor_node.name = f"Invert Factor {index:02d}"
        _output(factor_node, "Value").default_value = factor
        invert = tree.nodes.new("ShaderNodeInvert")
        invert.name = f"Invert Color {index:02d}"
        _input(invert, "Color").default_value = color
        tree.links.new(
            _output(factor_node, "Value"),
            _input(invert, "Fac"),
        )
        bias = tree.nodes.new("ShaderNodeVectorMath")
        bias.name = f"Invert Positive Bias {index:02d}"
        bias.operation = "ADD"
        _input_identifier(
            bias, "Vector_001"
        ).default_value = (10.0, 10.0, 10.0)
        tree.links.new(
            _output(invert, "Color"),
            _input_identifier(bias, "Vector"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Invert Emission {index:02d}"
        tree.links.new(
            _output(bias, "Vector"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Invert Color Matrix",
    )


def _brick_texture(scene: Any) -> None:
    """Exercise both Brick Color and Fac with non-default row controls."""
    material, tree, output = _material("Brick Texture Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    brick = tree.nodes.new("ShaderNodeTexBrick")
    brick.name = "Brick Texture"
    brick.offset = 0.37
    brick.offset_frequency = 3
    brick.squash = 0.72
    brick.squash_frequency = 2
    _input(brick, "Color1").default_value = (0.78, 0.08, 0.03, 1.0)
    _input(brick, "Color2").default_value = (0.12, 0.43, 0.91, 1.0)
    _input(brick, "Mortar").default_value = (0.07, 0.21, 0.13, 1.0)
    _input(brick, "Scale").default_value = 5.3
    _input(brick, "Mortar Size").default_value = 0.036
    _input(brick, "Mortar Smooth").default_value = 0.017
    _input(brick, "Bias").default_value = -0.14
    _input(brick, "Brick Width").default_value = 0.53
    _input(brick, "Row Height").default_value = 0.19
    tree.links.new(
        _output(coordinates, "Generated"),
        _input(brick, "Vector"),
    )
    channels = tree.nodes.new("ShaderNodeSeparateColor")
    channels.mode = "RGB"
    tree.links.new(
        _output(brick, "Color"),
        _input(channels, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.mode = "RGB"
    tree.links.new(
        _output(channels, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(channels, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(brick, "Fac"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _brick_texture_constants(scene: Any) -> None:
    """Compare Brick numerics without reconstruction-filter edge noise."""
    material, tree, output = _material("Brick Texture Constant Probe")

    def brick_at(
        name: str, coordinate: tuple[float, float, float]
    ) -> Any:
        coordinate_node = tree.nodes.new("ShaderNodeRGB")
        coordinate_node.name = f"{name} Coordinates"
        _output(coordinate_node, "Color").default_value = (
            *coordinate,
            1.0,
        )
        brick = tree.nodes.new("ShaderNodeTexBrick")
        brick.name = name
        brick.offset = 0.37
        brick.offset_frequency = 3
        brick.squash = 0.72
        brick.squash_frequency = 2
        tree.links.new(
            _output(coordinate_node, "Color"),
            _input(brick, "Vector"),
        )
        _input(brick, "Color1").default_value = (
            0.78,
            0.08,
            0.03,
            1.0,
        )
        _input(brick, "Color2").default_value = (
            0.12,
            0.43,
            0.91,
            1.0,
        )
        _input(brick, "Mortar").default_value = (
            0.07,
            0.21,
            0.13,
            1.0,
        )
        _input(brick, "Scale").default_value = 5.3
        _input(brick, "Mortar Size").default_value = 0.036
        _input(brick, "Mortar Smooth").default_value = 0.017
        _input(brick, "Bias").default_value = -0.14
        _input(brick, "Brick Width").default_value = 0.53
        _input(brick, "Row Height").default_value = 0.19
        return brick

    brick_red = brick_at("Brick Red Sample", (0.113, 0.257, 0.0))
    brick_green = brick_at(
        "Brick Green Sample", (0.471, 0.379, 0.0)
    )
    brick_factor = brick_at(
        "Brick Mortar Sample", (0.811, 0.541, 0.0)
    )
    red_channels = tree.nodes.new("ShaderNodeSeparateColor")
    red_channels.name = "Separate Red Sample"
    red_channels.mode = "RGB"
    tree.links.new(
        _output(brick_red, "Color"),
        _input(red_channels, "Color"),
    )
    green_channels = tree.nodes.new("ShaderNodeSeparateColor")
    green_channels.name = "Separate Green Sample"
    green_channels.mode = "RGB"
    tree.links.new(
        _output(brick_green, "Color"),
        _input(green_channels, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Brick Samples"
    combine.mode = "RGB"
    tree.links.new(
        _output(red_channels, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(green_channels, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(brick_factor, "Fac"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _white_noise_dimensions(scene: Any) -> None:
    material, tree, output = _material(
        "White Noise Dimensions Probe"
    )
    values: list[Any] = []
    colors: list[dict[str, Any]] = []
    constants = (
        ((0.17, -2.3, 5.1), 0.37),
        ((-0.25, 1.75, 3.5), -4.25),
        ((2.125, -0.75, 0.03125), 8.5),
        ((-6.0, 0.625, 4.75), 1.125),
    )
    for dimensions, (vector, w) in enumerate(
        constants, start=1
    ):
        white = tree.nodes.new("ShaderNodeTexWhiteNoise")
        white.name = f"White Noise {dimensions}D"
        white.noise_dimensions = f"{dimensions}D"
        if dimensions != 1:
            _input(white, "Vector").default_value = vector
        if dimensions in (1, 4):
            _input(white, "W").default_value = w
        values.append(_output(white, "Value"))
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate {dimensions}D Color"
        separate.mode = "RGB"
        tree.links.new(
            _output(white, "Color"),
            _input(separate, "Color"),
        )
        colors.append(
            {
                channel: _output(separate, channel)
                for channel in ("Red", "Green", "Blue")
            }
        )

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Add {index}"
            add.operation = "ADD"
            tree.links.new(current, _input(add, "Value"))
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, _input(scale, "Value"))
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    red = average(
        "Red",
        [values[0], values[3], colors[1]["Red"]],
    )
    green = average(
        "Green",
        [values[1], colors[2]["Green"], colors[0]["Green"]],
    )
    blue = average(
        "Blue",
        [values[2], colors[3]["Blue"], colors[0]["Blue"]],
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Combine White Noise Coverage"
    combine.mode = "RGB"
    tree.links.new(red, _input(combine, "Red"))
    tree.links.new(green, _input(combine, "Green"))
    tree.links.new(blue, _input(combine, "Blue"))
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _particle_random_nonparticle(scene: Any) -> None:
    material, tree, output = _material("Particle Random Probe")
    particle = tree.nodes.new("ShaderNodeParticleInfo")
    particle.name = "Particle Info"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(particle, "Random"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _particle_random_instances(scene: Any) -> None:
    material, tree, output = _material("Particle Instance Random Probe")
    particle = tree.nodes.new("ShaderNodeParticleInfo")
    particle.name = "Particle Info"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(particle, "Random"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2,
        radius=0.11,
        enter_editmode=False,
        location=(10.0, 10.0, 0.0),
    )
    instance = bpy.context.object
    instance.name = "Particle Instance"
    instance.data.materials.append(material)

    emitter_material, emitter_tree, emitter_output = _material(
        "Particle Emitter"
    )
    emitter_diffuse = emitter_tree.nodes.new("ShaderNodeBsdfDiffuse")
    _input(emitter_diffuse, "Color").default_value = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    emitter_tree.links.new(
        _output(emitter_diffuse, "BSDF"),
        _input(emitter_output, "Surface"),
    )
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=5,
        y_subdivisions=5,
        size=1.8,
        enter_editmode=False,
        location=(0.0, 0.0, -0.12),
    )
    emitter = bpy.context.object
    emitter.name = "Particle Emitter"
    emitter.data.materials.append(emitter_material)
    bpy.context.view_layer.objects.active = emitter
    emitter.select_set(True)
    bpy.ops.object.particle_system_add()
    system = emitter.particle_systems[-1]
    settings = system.settings
    settings.type = "HAIR"
    settings.count = 25
    settings.hair_length = 0.12
    settings.render_type = "OBJECT"
    settings.instance_object = instance
    settings.particle_size = 1.0
    settings.size_random = 0.0
    settings.emit_from = "VERT"
    settings.use_modifier_stack = True


def _background_world(scene: Any) -> None:
    _world(scene, (0.16, 0.48, 0.77, 1.0), 2.3)


def _node_group_color(scene: Any) -> None:
    material, tree, output = _material("Node Group Probe")

    group = bpy.data.node_groups.new(
        "Generic Color Transform", "ShaderNodeTree"
    )
    group.interface.new_socket(
        name="Color",
        in_out="INPUT",
        socket_type="NodeSocketColor",
    )
    group.interface.new_socket(
        name="Color",
        in_out="OUTPUT",
        socket_type="NodeSocketColor",
    )
    group_input = group.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"
    group_output = group.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    invert = group.nodes.new("ShaderNodeInvert")
    invert.name = "Invert"
    _input(invert, "Fac").default_value = 0.25
    group.links.new(
        _output(group_input, "Color"),
        _input(invert, "Color"),
    )
    group.links.new(
        _output(invert, "Color"),
        _input(group_output, "Color"),
    )

    outer = bpy.data.node_groups.new(
        "Nested Color Wrapper", "ShaderNodeTree"
    )
    outer.interface.new_socket(
        name="Color",
        in_out="INPUT",
        socket_type="NodeSocketColor",
    )
    outer.interface.new_socket(
        name="Color",
        in_out="OUTPUT",
        socket_type="NodeSocketColor",
    )
    outer_input = outer.nodes.new("NodeGroupInput")
    outer_input.name = "Group Input"
    outer_output = outer.nodes.new("NodeGroupOutput")
    outer_output.name = "Group Output"
    nested = outer.nodes.new("ShaderNodeGroup")
    nested.name = "Nested Arbitrary Instance"
    nested.node_tree = group
    outer.links.new(
        _output(outer_input, "Color"),
        _input(nested, "Color"),
    )
    outer.links.new(
        _output(nested, "Color"),
        _input(outer_output, "Color"),
    )

    source = tree.nodes.new("ShaderNodeRGB")
    source.name = "Group Source"
    _output(source, "Color").default_value = (
        0.12,
        0.47,
        0.81,
        1.0,
    )
    instance = tree.nodes.new("ShaderNodeGroup")
    instance.name = "Arbitrarily Named Group Instance"
    instance.node_tree = outer
    tree.links.new(
        _output(source, "Color"),
        _input(instance, "Color"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Strength").default_value = 2.0
    tree.links.new(
        _output(instance, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _rgb_to_bw(scene: Any) -> None:
    material, tree, output = _material("RGB to BW")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Luminance Weights"
    combine.mode = "RGB"
    for color, target in (
        ((1.0, 0.0, 0.0, 1.0), "Red"),
        ((0.0, 1.0, 0.0, 1.0), "Green"),
        ((0.0, 0.0, 1.0, 1.0), "Blue"),
    ):
        convert = tree.nodes.new("ShaderNodeRGBToBW")
        convert.name = "RGB to BW"
        _input(convert, "Color").default_value = color
        tree.links.new(
            _output(convert, "Val"),
            _input(combine, target),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _gamma_color(scene: Any) -> None:
    material, tree, output = _material("Gamma")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Gamma Branches"
    combine.mode = "RGB"
    for index, (color, gamma, source, target) in enumerate(
        (
            ((0.2, 0.5, 0.9, 1.0), 0.0, "Red", "Red"),
            ((0.18, 0.5, 0.87, 1.0), 2.2, "Green", "Green"),
            ((0.0, 0.25, 0.25, 1.0), -0.5, "Blue", "Blue"),
        )
    ):
        node = tree.nodes.new("ShaderNodeGamma")
        node.name = f"Gamma Branch {index}"
        _input(node, "Color").default_value = color
        _input(node, "Gamma").default_value = gamma
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select Gamma Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(node, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, source),
            _input(combine, target),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _brightness_contrast(scene: Any) -> None:
    material, tree, output = _material("Brightness Contrast")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Brightness/Contrast Branches"
    combine.mode = "RGB"
    for index, (color, bright, contrast, source, target) in enumerate(
        (
            ((0.14, 0.52, 0.88, 1.0), 0.0, 0.0, "Red", "Red"),
            (
                (0.09, 0.44, 0.81, 1.0),
                0.17,
                -0.35,
                "Green",
                "Green",
            ),
            (
                (0.08, 0.37, 0.08, 1.0),
                -0.22,
                0.48,
                "Blue",
                "Blue",
            ),
        )
    ):
        node = tree.nodes.new("ShaderNodeBrightContrast")
        node.name = f"Brightness/Contrast Branch {index}"
        _input(node, "Color").default_value = color
        _input(node, "Bright").default_value = bright
        _input(node, "Contrast").default_value = contrast
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select Brightness Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(node, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, source),
            _input(combine, target),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _hue_saturation_value(scene: Any) -> None:
    """Cover Cycles HSV adjustment, hue wrapping, and factor blending."""
    material, tree, output = _material("Hue Saturation Value")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack HSV Branches"
    combine.mode = "RGB"
    branches = (
        (
            (0.12, 0.65, 0.9, 1.0),
            0.3,
            1.4,
            0.75,
            1.0,
            "Red",
        ),
        (
            (0.83, 0.2, 0.06, 1.0),
            0.82,
            0.25,
            1.7,
            0.37,
            "Green",
        ),
        (
            (0.1, 0.4, 0.7, 1.0),
            0.05,
            2.2,
            0.4,
            0.85,
            "Blue",
        ),
    )
    for index, (
        color,
        hue,
        saturation,
        value,
        factor,
        channel,
    ) in enumerate(branches):
        node = tree.nodes.new("ShaderNodeHueSaturation")
        node.name = f"HSV Branch {index}"
        _input(node, "Color").default_value = color
        _input(node, "Hue").default_value = hue
        _input(node, "Saturation").default_value = saturation
        _input(node, "Value").default_value = value
        _input(node, "Fac").default_value = factor
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select HSV Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(node, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, channel),
            _input(combine, channel),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _clamp(scene: Any) -> None:
    material, tree, output = _material("Clamp")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Clamp Branches"
    combine.mode = "RGB"
    for index, (mode, value, minimum, maximum, target) in enumerate(
        (
            ("MINMAX", 0.5, 0.8, 0.2, "Red"),
            ("RANGE", 0.5, 0.8, 0.2, "Green"),
            ("MINMAX", 2.0, -0.2, 0.7, "Blue"),
        )
    ):
        node = tree.nodes.new("ShaderNodeClamp")
        node.name = f"Clamp Branch {index}"
        node.clamp_type = mode
        _input(node, "Value").default_value = value
        _input(node, "Min").default_value = minimum
        _input(node, "Max").default_value = maximum
        tree.links.new(
            _output(node, "Result"),
            _input(combine, target),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _math_operations(scene: Any) -> None:
    """Exercise every Blender 4.5.10 ShaderNodeMath operation."""
    material, tree, output = _material("Math Operations Probe")
    cases = (
        ("ADD", 0.12, 0.23, 0.5),
        ("SUBTRACT", 0.72, 0.19, 0.5),
        ("MULTIPLY", 0.45, 0.55, 0.5),
        ("DIVIDE", 0.24, 0.8, 0.5),
        ("MULTIPLY_ADD", 0.3, 0.5, 0.2),
        ("POWER", 0.64, 0.5, 0.5),
        ("LOGARITHM", 0.5, 0.25, 0.5),
        ("SQRT", 0.36, 0.5, 0.5),
        ("INVERSE_SQRT", 4.0, 0.5, 0.5),
        ("ABSOLUTE", -0.47, 0.5, 0.5),
        ("EXPONENT", -0.7, 0.5, 0.5),
        ("MINIMUM", 0.33, 0.71, 0.5),
        ("MAXIMUM", 0.33, 0.71, 0.5),
        ("LESS_THAN", 0.2, 0.3, 0.5),
        ("GREATER_THAN", 0.8, 0.3, 0.5),
        ("SIGN", 0.4, 0.5, 0.5),
        ("COMPARE", 0.4, 0.42, 0.03),
        ("SMOOTH_MIN", 0.4, 0.6, 0.3),
        ("SMOOTH_MAX", 0.4, 0.6, 0.3),
        ("ROUND", 0.51, 0.5, 0.5),
        ("FLOOR", 1.8, 0.5, 0.5),
        ("CEIL", 0.2, 0.5, 0.5),
        ("TRUNC", 1.8, 0.5, 0.5),
        ("FRACT", 1.37, 0.5, 0.5),
        ("MODULO", 1.3, 0.8, 0.5),
        ("FLOORED_MODULO", -0.3, 0.8, 0.5),
        ("WRAP", 1.4, 1.0, 0.2),
        ("SNAP", 0.74, 0.2, 0.5),
        ("PINGPONG", 0.73, 0.6, 0.5),
        ("SINE", 0.5, 0.5, 0.5),
        ("COSINE", 0.7, 0.5, 0.5),
        ("TANGENT", 0.3, 0.5, 0.5),
        ("ARCSINE", 0.5, 0.5, 0.5),
        ("ARCCOSINE", 0.8, 0.5, 0.5),
        ("ARCTANGENT", 0.6, 0.5, 0.5),
        ("ARCTAN2", 0.4, 0.8, 0.5),
        ("SINH", 0.5, 0.5, 0.5),
        ("COSH", 0.0, 0.5, 0.5),
        ("TANH", 0.7, 0.5, 0.5),
        ("RADIANS", 45.0, 0.5, 0.5),
        ("DEGREES", 0.01, 0.5, 0.5),
    )
    channels: list[list[Any]] = [[], [], []]
    for index, (operation, a, b, c) in enumerate(cases):
        math = tree.nodes.new("ShaderNodeMath")
        math.name = f"Math {index:02d} {operation}"
        math.operation = operation
        math.use_clamp = False
        math.inputs[0].default_value = a
        math.inputs[1].default_value = b
        math.inputs[2].default_value = c
        channels[index % 3].append(_output(math, "Value"))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Math Operations"
    combine.mode = "RGB"
    for name, sockets in zip(
        ("Red", "Green", "Blue"),
        channels,
        strict=True,
    ):
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _math_edge_cases(scene: Any) -> None:
    """Cover Cycles guards, signed behavior, and Math output clamping."""
    material, tree, output = _material("Math Edge Cases Probe")
    cases = (
        ("DIVIDE", 0.7, 0.0, 0.5, False),
        ("POWER", -0.5, 0.5, 0.5, False),
        ("POWER", -0.5, 3.0, 0.5, False),
        ("POWER", 0.0, 0.0, 0.5, False),
        ("LOGARITHM", -2.0, 10.0, 0.5, False),
        ("LOGARITHM", 0.5, 1.0, 0.5, False),
        ("SQRT", -4.0, 0.5, 0.5, False),
        ("INVERSE_SQRT", 0.0, 0.5, 0.5, False),
        ("ARCSINE", 2.0, 0.5, 0.5, False),
        ("ARCCOSINE", -2.0, 0.5, 0.5, False),
        ("ARCTAN2", 0.0, 0.0, 0.5, False),
        ("SIGN", 0.0, 0.5, 0.5, False),
        ("MODULO", -1.3, 0.8, 0.5, False),
        ("FLOORED_MODULO", -1.3, 0.8, 0.5, False),
        ("MODULO", 0.7, 0.0, 0.5, False),
        ("WRAP", 0.7, 0.4, 0.4, False),
        ("WRAP", 0.7, 0.2, 0.8, False),
        ("SNAP", 0.7, 0.0, 0.5, False),
        ("PINGPONG", 0.7, 0.0, 0.5, False),
        ("PINGPONG", -0.7, 0.6, 0.5, False),
        ("SMOOTH_MIN", 0.4, 0.6, 0.0, False),
        ("SMOOTH_MAX", 0.4, 0.6, 0.0, False),
        ("COMPARE", 1.0, 1.0000001192092896, -1.0, False),
        ("COMPARE", 1.0, 1.000000238418579, 0.0, False),
        ("ROUND", -1.5, 0.5, 0.5, False),
        ("TRUNC", -1.8, 0.5, 0.5, False),
        ("FRACT", -1.3, 0.5, 0.5, False),
        ("ADD", 0.8, 0.7, 0.5, True),
        ("SUBTRACT", 0.2, 0.7, 0.5, True),
    )
    channels: list[list[Any]] = [[], [], []]
    for index, (operation, a, b, c, use_clamp) in enumerate(cases):
        math = tree.nodes.new("ShaderNodeMath")
        math.name = f"Math Edge {index:02d} {operation}"
        math.operation = operation
        math.use_clamp = use_clamp
        math.inputs[0].default_value = a
        math.inputs[1].default_value = b
        math.inputs[2].default_value = c
        channels[index % 3].append(_output(math, "Value"))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Edge Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Edge Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Math Edge Cases"
    combine.mode = "RGB"
    for name, sockets in zip(
        ("Red", "Green", "Blue"),
        channels,
        strict=True,
    ):
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _mix_color_modes(scene: Any) -> None:
    """Cover every Cycles color blend mode of the modern Mix node."""
    material, tree, output = _material("Mix Color Modes Probe")
    modes = (
        "MIX",
        "DARKEN",
        "MULTIPLY",
        "BURN",
        "LIGHTEN",
        "SCREEN",
        "DODGE",
        "ADD",
        "OVERLAY",
        "SOFT_LIGHT",
        "LINEAR_LIGHT",
        "DIFFERENCE",
        "EXCLUSION",
        "SUBTRACT",
        "DIVIDE",
        "HUE",
        "SATURATION",
        "COLOR",
        "VALUE",
    )
    channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, mode in enumerate(modes):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Color {index:02d} {mode}"
        mix.data_type = "RGBA"
        mix.blend_type = mode
        mix.clamp_factor = True
        mix.clamp_result = False
        _input_identifier(
            mix, "Factor_Float"
        ).default_value = 0.37
        _input_identifier(mix, "A_Color").default_value = (
            0.17,
            0.63,
            0.89,
            1.0,
        )
        _input_identifier(mix, "B_Color").default_value = (
            0.82,
            0.24,
            0.51,
            1.0,
        )
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate {mode}"
        separate.mode = "RGB"
        tree.links.new(
            _output_identifier(mix, "Result_Color"),
            _input(separate, "Color"),
        )
        for channel in channels:
            channels[channel].append(_output(separate, channel))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Mix Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Mix Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Mix Color Modes"
    combine.mode = "RGB"
    for name, sockets in channels.items():
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _mix_data_types(scene: Any) -> None:
    """Cover float, uniform-vector, and non-uniform-vector Mix paths."""
    material, tree, output = _material("Mix Data Types Probe")

    float_outputs: list[Any] = []
    for index, (factor, clamp_factor, a, b) in enumerate(
        (
            (1.4, True, 0.2, 0.8),
            (1.4, False, 0.2, 0.8),
            (-0.3, True, 0.2, 0.8),
        )
    ):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Float {index}"
        mix.data_type = "FLOAT"
        mix.clamp_factor = clamp_factor
        _input_identifier(mix, "Factor_Float").default_value = factor
        _input_identifier(mix, "A_Float").default_value = a
        _input_identifier(mix, "B_Float").default_value = b
        float_outputs.append(
            _output_identifier(mix, "Result_Float")
        )

    vector_outputs: list[Any] = []
    vector_cases = (
        ("UNIFORM", 1.3, (0.5, 0.5, 0.5), True),
        ("UNIFORM", -0.25, (0.5, 0.5, 0.5), False),
        ("NON_UNIFORM", 0.5, (-0.2, 0.5, 1.4), True),
        ("NON_UNIFORM", 0.5, (-0.2, 0.5, 1.4), False),
    )
    for index, (factor_mode, scalar_factor, vector_factor, clamp) in enumerate(
        vector_cases
    ):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Vector {index} {factor_mode}"
        mix.data_type = "VECTOR"
        mix.factor_mode = factor_mode
        mix.clamp_factor = clamp
        _input_identifier(
            mix, "Factor_Float"
        ).default_value = scalar_factor
        _input_identifier(
            mix, "Factor_Vector"
        ).default_value = vector_factor
        _input_identifier(mix, "A_Vector").default_value = (
            0.1,
            0.7,
            -0.2,
        )
        _input_identifier(mix, "B_Vector").default_value = (
            0.9,
            -0.1,
            0.6,
        )
        vector_outputs.append(
            _output_identifier(mix, "Result_Vector")
        )

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Data Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Data Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    vector_channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, vector_output in enumerate(vector_outputs):
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Mix Vector {index}"
        separate.mode = "RGB"
        tree.links.new(vector_output, _input(separate, "Color"))
        for channel, component in zip(
            vector_channels,
            ("Red", "Green", "Blue"),
            strict=True,
        ):
            vector_channels[channel].append(
                _output(separate, component)
            )

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Mix Data Types"
    combine.mode = "RGB"
    tree.links.new(
        average("Float", float_outputs),
        _input(combine, "Red"),
    )
    tree.links.new(
        average("Vector Y", vector_channels["Green"]),
        _input(combine, "Green"),
    )
    tree.links.new(
        average("Vector Z", vector_channels["Blue"]),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _mix_color_edge_cases(scene: Any) -> None:
    """Cover guarded branches and factor/result clamping of color Mix."""
    material, tree, output = _material("Mix Color Edge Cases Probe")
    cases = (
        (
            "DIVIDE",
            0.7,
            (0.2, 0.7, 0.9, 1.0),
            (0.0, 0.2, 0.0, 1.0),
            True,
            False,
        ),
        (
            "DODGE",
            0.8,
            (0.0, 0.5, 0.9, 1.0),
            (0.3, 3.0, 0.9, 1.0),
            True,
            False,
        ),
        (
            "BURN",
            1.0,
            (0.5, 0.1, 0.9, 1.0),
            (0.0, 0.2, 2.0, 1.0),
            True,
            False,
        ),
        (
            "HUE",
            0.6,
            (0.2, 0.7, 0.4, 1.0),
            (0.5, 0.5, 0.5, 1.0),
            True,
            False,
        ),
        (
            "SATURATION",
            0.6,
            (0.5, 0.5, 0.5, 1.0),
            (0.1, 0.8, 0.3, 1.0),
            True,
            False,
        ),
        (
            "COLOR",
            0.6,
            (0.2, 0.7, 0.4, 1.0),
            (0.5, 0.5, 0.5, 1.0),
            True,
            False,
        ),
        (
            "MIX",
            1.4,
            (0.2, 0.3, 0.4, 1.0),
            (0.8, 0.7, 0.6, 1.0),
            False,
            False,
        ),
        (
            "ADD",
            1.4,
            (0.7, 0.8, 0.9, 1.0),
            (0.8, 0.7, 0.6, 1.0),
            True,
            False,
        ),
        (
            "LINEAR_LIGHT",
            0.9,
            (0.1, 0.9, 0.5, 1.0),
            (0.0, 1.0, 0.8, 1.0),
            True,
            True,
        ),
        (
            "SUBTRACT",
            1.0,
            (0.1, 0.7, 0.3, 1.0),
            (0.8, 0.2, 0.9, 1.0),
            True,
            True,
        ),
    )
    channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, (
        mode,
        factor,
        color_a,
        color_b,
        clamp_factor,
        clamp_result,
    ) in enumerate(cases):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Edge {index:02d} {mode}"
        mix.data_type = "RGBA"
        mix.blend_type = mode
        mix.clamp_factor = clamp_factor
        mix.clamp_result = clamp_result
        _input_identifier(
            mix, "Factor_Float"
        ).default_value = factor
        _input_identifier(mix, "A_Color").default_value = color_a
        _input_identifier(mix, "B_Color").default_value = color_b
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Mix Edge {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output_identifier(mix, "Result_Color"),
            _input(separate, "Color"),
        )
        for channel in channels:
            channels[channel].append(_output(separate, channel))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Edge Mix Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Edge Mix Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Mix Color Edge Cases"
    combine.mode = "RGB"
    for name, sockets in channels.items():
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _gradient_matrix(scene: Any) -> None:
    """Cover every Cycles Gradient Texture mode and saturation branch."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("LINEAR", (-2.0, 0.0, 0.0)),
        ("LINEAR", (0.37, 2.0, -1.0)),
        ("LINEAR", (2.0, 0.0, 0.0)),
        ("QUADRATIC", (-1.0, 0.0, 0.0)),
        ("QUADRATIC", (0.5, 0.0, 0.0)),
        ("EASING", (-0.5, 0.0, 0.0)),
        ("EASING", (0.3, 0.0, 0.0)),
        ("EASING", (1.5, 0.0, 0.0)),
        ("DIAGONAL", (-1.0, 0.2, 0.0)),
        ("DIAGONAL", (1.4, 0.8, 0.0)),
        ("RADIAL", (1.0, 0.0, 0.0)),
        ("RADIAL", (0.0, -1.0, 0.0)),
        ("SPHERICAL", (0.0, 0.0, 0.0)),
        ("SPHERICAL", (1.0, 0.0, 0.0)),
        ("QUADRATIC_SPHERE", (0.0, 0.0, 0.0)),
        ("QUADRATIC_SPHERE", (0.5, 0.5, 0.5)),
    )
    materials = []
    for index, (gradient_type, vector) in enumerate(cases):
        material, tree, output = _material(
            f"Gradient Matrix {index:02d} {gradient_type}"
        )
        gradient = tree.nodes.new("ShaderNodeTexGradient")
        gradient.name = f"Gradient {index:02d} {gradient_type}"
        gradient.gradient_type = gradient_type
        tree.links.new(
            _linked_vector(
                tree,
                f"Gradient Vector {index:02d}",
                vector,
            ),
            _input(gradient, "Vector"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Gradient Emission {index:02d}"
        tree.links.new(
            _output(gradient, "Fac"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Gradient Matrix",
    )


def _mix_rgb_legacy_modes(scene: Any) -> None:
    """Cover every Cycles blend mode of the legacy MixRGB node."""
    material, tree, output = _material("Legacy MixRGB Modes Probe")
    modes = (
        "MIX",
        "DARKEN",
        "MULTIPLY",
        "BURN",
        "LIGHTEN",
        "SCREEN",
        "DODGE",
        "ADD",
        "OVERLAY",
        "SOFT_LIGHT",
        "LINEAR_LIGHT",
        "DIFFERENCE",
        "EXCLUSION",
        "SUBTRACT",
        "DIVIDE",
        "HUE",
        "SATURATION",
        "COLOR",
        "VALUE",
    )
    channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, mode in enumerate(modes):
        mix = tree.nodes.new("ShaderNodeMixRGB")
        mix.name = f"Legacy MixRGB {index:02d} {mode}"
        mix.blend_type = mode
        mix.use_alpha = index % 2 == 0
        mix.use_clamp = False
        _input(mix, "Fac").default_value = 0.37
        _input(mix, "Color1").default_value = (
            0.17,
            0.63,
            0.89,
            0.2,
        )
        _input(mix, "Color2").default_value = (
            0.82,
            0.24,
            0.51,
            0.9,
        )
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Legacy {mode}"
        separate.mode = "RGB"
        tree.links.new(
            _output(mix, "Color"),
            _input(separate, "Color"),
        )
        for channel in channels:
            channels[channel].append(_output(separate, channel))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Legacy Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Legacy Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Legacy MixRGB Modes"
    combine.mode = "RGB"
    for name, sockets in channels.items():
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _separate_color_modes(scene: Any) -> None:
    material, tree, output = _material("Separate Color Modes")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Separate Modes"
    combine.mode = "RGB"
    for index, (mode, source, target) in enumerate(
        (
            ("RGB", "Red", "Red"),
            ("HSV", "Green", "Green"),
            ("HSL", "Red", "Blue"),
        )
    ):
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Color {mode} {index}"
        separate.mode = mode
        _input(separate, "Color").default_value = (
            0.13,
            0.47,
            0.82,
            1.0,
        )
        tree.links.new(
            _output(separate, source),
            _input(combine, target),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _combine_color_modes(scene: Any) -> None:
    material, tree, output = _material("Combine Color Modes")
    packed = tree.nodes.new("ShaderNodeCombineColor")
    packed.name = "Pack Combine Modes"
    packed.mode = "RGB"
    for index, (mode, channels, source, target) in enumerate(
        (
            ("RGB", (0.2, 0.6, 0.9), "Red", "Red"),
            ("HSV", (0.73, 0.61, 0.84), "Green", "Green"),
            ("HSL", (0.13, 0.55, 0.36), "Blue", "Blue"),
        )
    ):
        combine = tree.nodes.new("ShaderNodeCombineColor")
        combine.name = f"Combine Color {mode} {index}"
        combine.mode = mode
        for name, value in zip(
            ("Red", "Green", "Blue"),
            channels,
            strict=True,
        ):
            _input(combine, name).default_value = value
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select Combine Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(combine, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, source),
            _input(packed, target),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(packed, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _legacy_separate_combine_matrix(scene: Any) -> None:
    """Cover all sockets of legacy RGB, HSV, and XYZ split/pack nodes."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    value_sets = (
        (
            (0.13, 0.47, 0.91),
            (0.13, 0.70, 0.80),
            (-0.70, 0.25, 1.30),
        ),
        (
            (1.20, -0.30, 0.50),
            (0.83, 0.20, 1.30),
            (2.00, -3.00, 4.00),
        ),
        (
            (0.00, 1.00, 2.00),
            (1.20, 0.90, 0.40),
            (0.00, 0.00, 0.00),
        ),
        (
            (-1.50, 0.25, 3.00),
            (0.00, 0.00, 2.00),
            (0.0001, 0.999, -1.25),
        ),
    )
    materials = []
    for set_index, (rgb, hsv, xyz) in enumerate(value_sets):
        components = (0, 1, 2, set_index % 3)
        for local_index, component in enumerate(components):
            index = set_index * 4 + local_index
            material, tree, output = _material(
                f"Legacy Split Pack {index:02d}"
            )

            combine_rgb = tree.nodes.new("ShaderNodeCombineRGB")
            combine_rgb.name = f"Combine RGB {index:02d}"
            for socket, value in zip(
                ("R", "G", "B"), rgb, strict=True
            ):
                _input(combine_rgb, socket).default_value = value
            separate_rgb = tree.nodes.new("ShaderNodeSeparateRGB")
            separate_rgb.name = f"Separate RGB {index:02d}"
            tree.links.new(
                _output(combine_rgb, "Image"),
                _input(separate_rgb, "Image"),
            )

            combine_hsv = tree.nodes.new("ShaderNodeCombineHSV")
            combine_hsv.name = f"Combine HSV {index:02d}"
            for socket, value in zip(
                ("H", "S", "V"), hsv, strict=True
            ):
                _input(combine_hsv, socket).default_value = value
            separate_hsv = tree.nodes.new("ShaderNodeSeparateHSV")
            separate_hsv.name = f"Separate HSV {index:02d}"
            tree.links.new(
                _output(combine_hsv, "Color"),
                _input(separate_hsv, "Color"),
            )

            combine_xyz = tree.nodes.new("ShaderNodeCombineXYZ")
            combine_xyz.name = f"Combine XYZ {index:02d}"
            for socket, value in zip(
                ("X", "Y", "Z"), xyz, strict=True
            ):
                _input(combine_xyz, socket).default_value = value
            separate_xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
            separate_xyz.name = f"Separate XYZ {index:02d}"
            tree.links.new(
                _output(combine_xyz, "Vector"),
                _input(separate_xyz, "Vector"),
            )

            packed = tree.nodes.new("ShaderNodeCombineColor")
            packed.name = f"Pack Legacy Results {index:02d}"
            packed.mode = "RGB"
            tree.links.new(
                _output(
                    separate_rgb,
                    ("R", "G", "B")[component],
                ),
                _input(packed, "Red"),
            )
            tree.links.new(
                _output(
                    separate_hsv,
                    ("H", "S", "V")[component],
                ),
                _input(packed, "Green"),
            )
            tree.links.new(
                _output(
                    separate_xyz,
                    ("X", "Y", "Z")[component],
                ),
                _input(packed, "Blue"),
            )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"Legacy Emission {index:02d}"
            tree.links.new(
                _output(packed, "Color"),
                _input(emission, "Color"),
            )
            tree.links.new(
                _output(emission, "Emission"),
                _input(output, "Surface"),
            )
            materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Legacy Separate Combine Matrix",
    )


_PROBES: dict[str, Callable[[Any], None]] = {
    "add_shader_emission": _add_shader_emission,
    "area_light": _area_light,
    "area_light_ellipse": _area_light_ellipse,
    "area_light_spread": _area_light_spread,
    "flat_light_distribution": _flat_light_distribution,
    "background_world": _background_world,
    "blackbody_matrix": _blackbody_matrix,
    "bump_matrix": _bump_matrix,
    "bump_nested_matrix": _bump_nested_matrix,
    "bump_surface": _bump_surface,
    "brightness_contrast": _brightness_contrast,
    "brick_texture": _brick_texture,
    "brick_texture_constants": _brick_texture_constants,
    "checker_texture_matrix": _checker_texture_matrix,
    "clamp": _clamp,
    "color_ramp_alpha_modes": _color_ramp_alpha_modes,
    "color_ramp_modes": _color_ramp_modes,
    "color_ramp_rgb": _color_ramp_rgb,
    "combine_color_modes": _combine_color_modes,
    "diffuse_bsdf_matrix": _diffuse_bsdf_matrix,
    "diffuse_surface": _diffuse_surface,
    "emission_surface": _emission_surface,
    "fresnel_matrix": _fresnel_matrix,
    "gamma_color": _gamma_color,
    "gradient_spherical": _gradient_spherical,
    "gradient_matrix": _gradient_matrix,
    "hue_saturation_value": _hue_saturation_value,
    "image_texture_srgb": _image_texture_srgb,
    "image_texture_sampling_modes": _image_texture_sampling_modes,
    "image_texture_projection_modes": _image_texture_projection_modes,
    "indirect_diffuse": _indirect_diffuse,
    "integrator_clamp_direct": _integrator_clamp_direct,
    "invert_color_matrix": _invert_color_matrix,
    "legacy_separate_combine_matrix": (
        _legacy_separate_combine_matrix
    ),
    "layer_weight_matrix": _layer_weight_matrix,
    "map_range_matrix": _map_range_matrix,
    "math_edge_cases": _math_edge_cases,
    "math_operations": _math_operations,
    "mapping_modes": _mapping_modes,
    "mix_color_modes": _mix_color_modes,
    "mix_color_edge_cases": _mix_color_edge_cases,
    "mix_data_types": _mix_data_types,
    "mix_rgb_legacy_modes": _mix_rgb_legacy_modes,
    "mix_shader_emission": _mix_shader_emission,
    "negative_scale_surface": _negative_scale_surface,
    "node_group_color": _node_group_color,
    "noise_bump_object": _noise_bump_object,
    "noise_color_3d": _noise_color_3d,
    "noise_factor_2d": _noise_factor_2d,
    "noise_fbm_matrix": _noise_fbm_matrix,
    "noise_hetero_terrain_matrix": _noise_hetero_terrain_matrix,
    "noise_hybrid_multifractal_matrix": (
        _noise_hybrid_multifractal_matrix
    ),
    "noise_multifractal_matrix": _noise_multifractal_matrix,
    "noise_ridged_multifractal_matrix": (
        _noise_ridged_multifractal_matrix
    ),
    "normal_map_surface": _normal_map_surface,
    "normal_map_matrix": _normal_map_matrix,
    "normal_map_named_uv_matrix": _normal_map_named_uv_matrix,
    "particle_random_instances": _particle_random_instances,
    "particle_random_nonparticle": _particle_random_nonparticle,
    "point_light": _point_light,
    "point_light_nodes": _point_light_nodes,
    "point_light_soft_disk": _point_light_soft_disk,
    "point_light_soft_sphere": _point_light_soft_sphere,
    "principled_bump_glossy": _principled_bump_glossy,
    "principled_surface": _principled_surface,
    "rgb_emission": _rgb_emission,
    "rgb_curve_matrix": _rgb_curve_matrix,
    "rgb_to_bw": _rgb_to_bw,
    "separate_color_modes": _separate_color_modes,
    "spot_light": _spot_light,
    "spot_light_soft": _spot_light_soft,
    "sun_light": _sun_light,
    "sun_light_disk": _sun_light_disk,
    "transparent_mix": _transparent_mix,
    "transparent_data_pass": _transparent_data_pass,
    "translucent_bsdf_matrix": _translucent_bsdf_matrix,
    "translucent_surface": _translucent_surface,
    "value_emission": _value_emission,
    "vector_math_matrix": _vector_math_matrix,
    "wavelength_matrix": _wavelength_matrix,
    "white_noise_dimensions": _white_noise_dimensions,
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

    _clear()
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
    _camera(scene)
    if probe_name != "background_world":
        _world(scene, (0.0, 0.0, 0.0, 1.0), 0.0)
    create(scene)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(f"Created Psycles Cycles probe {probe_name}: {output}")


if __name__ == "__main__":
    _main()
