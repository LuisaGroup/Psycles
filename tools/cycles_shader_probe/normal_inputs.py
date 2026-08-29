"""Blender Normal-node probes against the Cycles SVM definition."""

from __future__ import annotations

from typing import Any

import bpy

from .support import _input, _material, _material_matrix, _output


def _linked_vector(tree: Any, name: str, value: tuple[float, float, float]) -> Any:
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = name
    _input(combine, "X").default_value = value[0]
    _input(combine, "Y").default_value = value[1]
    _input(combine, "Z").default_value = value[2]
    return _output(combine, "Vector")


def _normal_node_matrix(scene: Any) -> None:
    """Cover both outputs, normalization, zero vectors, and linked inputs.

    Cycles defines ``D = normalize(direction)``, ``Normal = D``, and
    ``Dot = dot(D, normalize(input Normal))``. The authored direction lives on
    the node's Normal *output* socket; it is not the input default. Every input
    below is linked so this probe also rejects replacing Dot with its UI
    default, which was the production-scene failure that introduced the test.
    """

    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("DOT", (0.0, 0.0, 2.0), (0.0, 0.0, 4.0)),
        ("DOT", (2.0, 0.0, 0.0), (-3.0, 0.0, 0.0)),
        ("DOT", (1.0, 2.0, 3.0), (4.0, -2.0, 1.0)),
        ("DOT", (0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
        ("DOT", (1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
        ("NORMAL", (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)),
        ("NORMAL", (0.0, -3.0, 4.0), (8.0, 1.0, -2.0)),
        ("NORMAL", (0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
        ("BOTH", (1.0, -2.0, 3.0), (-4.0, 5.0, 1.0)),
    )
    materials = []
    for index, (mode, direction, input_normal) in enumerate(cases):
        material, tree, output = _material(
            f"Normal Node {index:02d} {mode}"
        )
        normal = tree.nodes.new("ShaderNodeNormal")
        normal.name = f"Normal {index:02d} {mode}"
        _output(normal, "Normal").default_value = direction
        tree.links.new(
            _linked_vector(
                tree,
                f"Linked Input {index:02d}",
                input_normal,
            ),
            _input(normal, "Normal"),
        )

        if mode == "DOT":
            scale = tree.nodes.new("ShaderNodeMath")
            scale.name = f"Encode Dot Scale {index:02d}"
            scale.operation = "MULTIPLY"
            _input(scale, "Value_001").default_value = 0.5
            tree.links.new(_output(normal, "Dot"), _input(scale, "Value"))
            bias = tree.nodes.new("ShaderNodeMath")
            bias.name = f"Encode Dot Bias {index:02d}"
            bias.operation = "ADD"
            _input(bias, "Value_001").default_value = 0.5
            tree.links.new(_output(scale, "Value"), _input(bias, "Value"))
            encoded = _output(bias, "Value")
        else:
            vector_scale = tree.nodes.new("ShaderNodeVectorMath")
            vector_scale.name = f"Encode Normal Scale {index:02d}"
            vector_scale.operation = "SCALE"
            _input(vector_scale, "Scale").default_value = 0.5
            tree.links.new(
                _output(normal, "Normal"),
                _input(vector_scale, "Vector"),
            )
            if mode == "BOTH":
                dot_scale = tree.nodes.new("ShaderNodeMath")
                dot_scale.name = f"Shared Dot Scale {index:02d}"
                dot_scale.operation = "MULTIPLY"
                _input(dot_scale, "Value_001").default_value = 0.25
                tree.links.new(
                    _output(normal, "Dot"),
                    _input(dot_scale, "Value"),
                )
                tree.links.new(
                    _output(dot_scale, "Value"),
                    _input(vector_scale, "Scale"),
                )
            bias = tree.nodes.new("ShaderNodeVectorMath")
            bias.name = f"Encode Normal Bias {index:02d}"
            bias.operation = "ADD"
            _input(bias, "Vector_001").default_value = (0.5, 0.5, 0.5)
            tree.links.new(
                _output(vector_scale, "Vector"),
                _input(bias, "Vector"),
            )
            encoded = _output(bias, "Vector")

        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Normal Node Emission {index:02d}"
        tree.links.new(encoded, _input(emission, "Color"))
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=3,
        rows=3,
        name="Normal Node Matrix",
    )
