"""Cycles 5.2 IES Light node probes with unmodified photometric sources."""

from __future__ import annotations

import math
import pathlib
from typing import Any

import bpy

from .support import _input, _material, _material_matrix, _output


# Nonuniform angles and deliberately asymmetric values exercise both cubic
# dimensions. The final Type C row repeats the first as required for 360-degree
# wrapping. These bytes are the renderer input, not preprocessed lookup data.
_TYPE_C = """IESNA:LM-63-2002
[TEST] PSYCLES TYPE C
TILT=NONE
1 1000 1 5 5 1 1 0 0 0 1 1 100
0 20 75 120 180
0 45 130 250 360
1 2 5 9 12
2 4 8 13 17
4 7 11 16 22
3 6 10 15 20
1 2 5 9 12
"""

_TYPE_B = """IESNA:LM-63-2002
[TEST] PSYCLES TYPE B
TILT=NONE
1 1000 0.75 4 3 2 1 0 0 0 1 1 100
0 30 60 90
0 45 90
2 3 5 8
4 7 11 16
6 10 15 21
"""

_TYPE_A = """IESNA:LM-63-2002
[TEST] PSYCLES TYPE A
TILT=NONE
1 1000 0.5 4 3 3 1 0 0 0 1 1 100
-90 -30 30 90
0 45 90
3 5 8 13
4 7 11 17
6 10 15 22
"""


def _ies_material(
    index: int,
    content: str | None,
    *,
    external_path: pathlib.Path | None = None,
    implicit_vector: bool = False,
    linked_strength: bool = False,
) -> Any:
    material, tree, output = _material(f"IES Light {index:02d}")
    ies = tree.nodes.new("ShaderNodeTexIES")
    ies.name = f"IES Light {index:02d}"
    _input(ies, "Strength").default_value = 0.4
    if external_path is not None:
        ies.mode = "EXTERNAL"
        ies.filepath = "//" + external_path.name
    else:
        ies.mode = "INTERNAL"
        if content is not None:
            text = bpy.data.texts.new(f"IES Profile {index:02d}.ies")
            text.write(content)
            ies.ies = text

    if not implicit_vector:
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        geometry.name = f"Geometry {index:02d}"
        tree.links.new(_output(geometry, "Position"), _input(ies, "Vector"))
    if linked_strength:
        light_path = tree.nodes.new("ShaderNodeLightPath")
        light_path.name = f"IES Strength {index:02d}"
        tree.links.new(
            _output(light_path, "Ray Length"), _input(ies, "Strength")
        )

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"Emission {index:02d}"
    tree.links.new(_output(ies, "Factor"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _ies_light_matrix(scene: Any) -> None:
    """Exercise Cycles IES A/B/C conversion, source modes and slot dedup."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01

    output = pathlib.Path(str(scene["psycles_probe_output"]))
    external = output.with_name("psycles_type_c_external.ies")
    external.parent.mkdir(parents=True, exist_ok=True)
    external.write_bytes(_TYPE_C.encode("utf-8"))

    materials = [
        _ies_material(0, _TYPE_C, implicit_vector=True),
        _ies_material(1, _TYPE_C, external_path=external),
        _ies_material(2, _TYPE_B),
        _ies_material(3, _TYPE_A),
        _ies_material(4, None),
        _ies_material(
            5,
            _TYPE_C,
            implicit_vector=True,
            linked_strength=True,
        ),
    ]
    _material_matrix(
        scene,
        materials,
        columns=3,
        rows=2,
        name="IES Light Matrix",
        frame_bleed=0.01,
    )


def _direction(horizontal: float, vertical: float) -> tuple[float, float, float]:
    radius = math.sin(vertical)
    return (
        radius * math.sin(horizontal - math.pi),
        radius * math.cos(horizontal - math.pi),
        -math.cos(vertical),
    )


def _ies_value_material(
    index: int,
    content: str | None,
    vector: tuple[float, float, float],
    strength: float,
) -> Any:
    material, tree, output = _material(f"IES Value {index:02d}")
    ies = tree.nodes.new("ShaderNodeTexIES")
    ies.name = f"IES Value {index:02d}"
    _input(ies, "Strength").default_value = strength
    if content is not None:
        text = bpy.data.texts.new(f"IES Value Profile {index:02d}.ies")
        text.write(content)
        ies.ies = text

    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = f"Direction {index:02d}"
    _input(combine, "X").default_value = vector[0]
    _input(combine, "Y").default_value = vector[1]
    _input(combine, "Z").default_value = vector[2]
    tree.links.new(_output(combine, "Vector"), _input(ies, "Vector"))

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"Emission {index:02d}"
    tree.links.new(_output(ies, "Factor"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _ies_light_values(scene: Any) -> None:
    """Flat cells provide a numerical Cycles oracle for IES interpolation."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (_TYPE_C, _direction(0.05, 0.5), 0.4),
        (_TYPE_C, _direction(2.0, 1.0), 0.4),
        (_TYPE_C, _direction(5.8, 2.8), 0.4),
        (_TYPE_B, _direction(1.1, 1.3), 0.6),
        (_TYPE_A, _direction(3.2, 1.4), 0.8),
        (_TYPE_A, _direction(0.5, 1.2), 0.7),
        (None, _direction(2.4, 1.7), 0.25),
        (_TYPE_C, (0.0, 0.0, 0.0), 0.3),
    )
    materials = [
        _ies_value_material(index, content, vector, strength)
        for index, (content, vector, strength) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="IES Light Values",
        frame_bleed=0.01,
    )
