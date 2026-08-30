"""Nested Bump probe scenes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import _input, _material, _material_matrix, _output


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
