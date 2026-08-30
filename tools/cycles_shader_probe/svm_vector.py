"""Cycles 5.2 SVM vector-node probes."""

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


def _svm_vector_rotate_matrix(scene: Any) -> None:
    """Exercise every Cycles Vector Rotate mode and its zero-axis branch."""
    cases = [
        ("AXIS_ANGLE", False, (0.29, 0.73, -0.41), "Forward"),
        ("AXIS_ANGLE", True, (0.29, 0.73, -0.41), "Inverse"),
        ("X_AXIS", False, (0.29, 0.73, -0.41), "Forward"),
        ("X_AXIS", True, (0.29, 0.73, -0.41), "Inverse"),
        ("Y_AXIS", False, (0.29, 0.73, -0.41), "Forward"),
        ("Y_AXIS", True, (0.29, 0.73, -0.41), "Inverse"),
        ("Z_AXIS", False, (0.29, 0.73, -0.41), "Forward"),
        ("Z_AXIS", True, (0.29, 0.73, -0.41), "Inverse"),
        ("EULER_XYZ", False, (0.29, 0.73, -0.41), "Forward"),
        ("EULER_XYZ", True, (0.29, 0.73, -0.41), "Inverse"),
        ("AXIS_ANGLE", False, (0.0, 0.0, 0.0), "Zero Axis"),
        ("AXIS_ANGLE", True, (0.0, 0.0, 0.0), "Zero Axis Inverse"),
    ]
    materials = []
    for index, (rotate_type, invert, axis, label) in enumerate(cases):
        material, tree, output = _material(
            f"SVM Vector Rotate {index:02d} {rotate_type} {label}"
        )
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        geometry.name = "Geometry"
        rotate = tree.nodes.new("ShaderNodeVectorRotate")
        rotate.name = "Vector Rotate"
        rotate.rotation_type = rotate_type
        rotate.invert = invert
        _input(rotate, "Center").default_value = (0.17, -0.23, 0.31)
        if rotate_type == "EULER_XYZ":
            _input(rotate, "Rotation").default_value = (0.31, -0.52, 0.27)
        else:
            _input(rotate, "Angle").default_value = 0.71
            if rotate_type == "AXIS_ANGLE":
                _input(rotate, "Axis").default_value = axis
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "Emission"

        tree.links.new(_output(geometry, "Normal"), _input(rotate, "Vector"))
        tree.links.new(_output(rotate, "Vector"), _input(emission, "Color"))
        tree.links.new(
            _output(emission, "Emission"), _input(output, "Surface")
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=6,
        rows=2,
        name="SVM Vector Rotate Matrix",
        frame_bleed=0.1,
    )
