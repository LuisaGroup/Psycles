"""Cycles 5.2 SVM Separate XYZ and Combine XYZ probes."""

from __future__ import annotations

from typing import Any

from .support import _input, _material, _material_matrix, _output


def _svm_sepcomb_vector_pipeline(scene: Any) -> None:
    """Keep all three Cycles vector split/pack transitions dynamic."""
    material, tree, output = _material("SVM Separate Combine Vector")

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Geometry"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = "Separate XYZ"
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = "Combine XYZ"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"

    tree.links.new(_output(geometry, "Normal"), _input(separate, "Vector"))
    tree.links.new(_output(separate, "Z"), _input(combine, "X"))
    tree.links.new(_output(separate, "X"), _input(combine, "Y"))
    tree.links.new(_output(separate, "Y"), _input(combine, "Z"))
    tree.links.new(_output(combine, "Vector"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))

    surface = _material_matrix(
        scene,
        [material, material],
        columns=2,
        rows=1,
        name="SVM Separate Combine Vector",
        backfacing={1},
        frame_bleed=0.1,
    )
    surface.rotation_euler = (0.31, -0.27, 0.19)


def _svm_sepcomb_vector_constant_fold(scene: Any) -> None:
    """Force Cycles' SeparateXYZNode and CombineXYZNode constant folders."""
    material, tree, output = _material(
        "SVM Separate Combine Vector Constant"
    )

    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = "Combine XYZ Constant"
    for socket, value in zip(
        ("X", "Y", "Z"), (-0.7, 0.25, 1.3), strict=True
    ):
        scalar = tree.nodes.new("ShaderNodeValue")
        scalar.name = f"{socket} Constant"
        _output(scalar, "Value").default_value = value
        tree.links.new(_output(scalar, "Value"), _input(combine, socket))

    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = "Separate XYZ Constant"
    packed = tree.nodes.new("ShaderNodeCombineColor")
    packed.name = "Pack Constant Result"
    packed.mode = "RGB"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"

    tree.links.new(_output(combine, "Vector"), _input(separate, "Vector"))
    tree.links.new(_output(separate, "Z"), _input(packed, "Red"))
    tree.links.new(_output(separate, "X"), _input(packed, "Green"))
    tree.links.new(_output(separate, "Y"), _input(packed, "Blue"))
    tree.links.new(_output(packed, "Color"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))

    _material_matrix(
        scene,
        [material],
        columns=1,
        rows=1,
        name="SVM Separate Combine Vector Constant",
        frame_bleed=0.1,
    )
