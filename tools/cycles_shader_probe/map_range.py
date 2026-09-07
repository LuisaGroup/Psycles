"""Original Blender Map Range graphs, with no baked shader outputs."""

from __future__ import annotations

from typing import Any

from .support import (
    _input,
    _input_identifier,
    _material,
    _material_matrix,
    _output,
)


def _shader(vector: bool, mode: str, clamp: bool, variant: str) -> Any:
    name = f"Map Range {'Vector' if vector else 'Float'} {mode} {int(clamp)} {variant}"
    material, tree, output = _material(name)
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Geometry"
    remap = tree.nodes.new("ShaderNodeMapRange")
    remap.name = "Map Range"
    remap.data_type = "FLOAT_VECTOR" if vector else "FLOAT"
    remap.interpolation_type = mode
    remap.clamp = clamp
    if vector:
        tree.links.new(_output(geometry, "Position"), _input(remap, "Vector"))
        for socket, value in (
            ("From_Min_FLOAT3", (-1.0, 2.0, 0.5)),
            ("From_Max_FLOAT3", (2.0, -1.0, 0.5)),
            ("To_Min_FLOAT3", (0.25, 1.5, 0.75)),
            ("To_Max_FLOAT3", (1.75, -0.5, 1.25)),
            ("Steps_FLOAT3", (3.0, 0.0, -2.0)),
        ):
            _input_identifier(remap, socket).default_value = value
        value_output = _output(remap, "Vector")
    else:
        split = tree.nodes.new("ShaderNodeSeparateXYZ")
        split.name = "Separate XYZ"
        tree.links.new(_output(geometry, "Position"), _input(split, "Vector"))
        tree.links.new(_output(split, "X"), _input(remap, "Value"))
        for socket, value in (
            ("From Min", 2.0),
            ("From Max", 2.0 if variant == "equal_from" else -1.0),
            ("To Min", 1.5), ("To Max", 0.25),
            ("Steps", {"zero_steps": 0.0, "negative_steps": -2.0}.get(variant, 3.0)),
        ):
            _input_identifier(remap, socket).default_value = value
        if variant == "linked":
            # Every scalar argument is stack-addressed. The Clamp expansion
            # must share the linked To Min/Max producers with Map Range.
            for socket, channel in (
                ("From Min", "Y"), ("From Max", "Z"),
                ("To Min", "Z"), ("To Max", "Y"), ("Steps", "X"),
            ):
                tree.links.new(_output(split, channel), _input_identifier(remap, socket))
        value_output = _output(remap, "Result")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(value_output, _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def matrix(scene: Any) -> None:
    cases = [
        (vector, mode, clamp, "standard")
        for vector in (False, True)
        for mode in ("LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP")
        for clamp in (False, True)
    ]
    cases.extend(
        (False, "LINEAR" if variant == "equal_from" else "STEPPED", clamp, variant)
        for variant in ("equal_from", "zero_steps", "negative_steps", "linked")
        for clamp in (False, True)
    )
    materials = [_shader(*case) for case in cases]
    _material_matrix(scene, materials, columns=6, rows=4, name="Map Range Matrix")
    scene.cycles.max_bounces = 1
    scene.cycles.use_light_tree = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
