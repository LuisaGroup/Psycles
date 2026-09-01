"""Focused color-operation probe scenes."""

from __future__ import annotations

from typing import Any

from .support import (
    _input,
    _input_identifier,
    _material,
    _material_matrix,
    _output,
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
