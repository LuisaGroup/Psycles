"""Normal Map probe scenes and tangent-space matrix oracles."""

from __future__ import annotations

from typing import Any

import bpy

from .support import (
    _input,
    _material,
    _material_matrix,
    _output,
    _sphere,
    _world,
)


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
