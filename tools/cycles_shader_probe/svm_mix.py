"""Cycles 5.2 SVM Mix-node bytecode and evaluation probes."""

from __future__ import annotations

from typing import Any

from .support import (
    _input,
    _input_identifier,
    _linked_vector,
    _material,
    _material_matrix,
    _output,
    _output_identifier,
)


MODERN_MIX_MODES = (
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


def _dynamic_factor(tree: Any, name: str) -> Any:
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = f"{name} Geometry"
    factor = tree.nodes.new("ShaderNodeMath")
    factor.name = f"{name} Factor"
    factor.operation = "MULTIPLY_ADD"
    tree.links.new(_output(geometry, "Backfacing"), factor.inputs[0])
    factor.inputs[1].default_value = 1.5
    factor.inputs[2].default_value = -0.25
    return _output(factor, "Value")


def _emission(tree: Any, output: Any, name: str) -> Any:
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return emission


def _svm_modern_mix_color_matrix(scene: Any) -> None:
    """Keep all Cycles 5.2 MixColor opcodes and clamp fields dynamic."""
    materials = []
    for index, mode in enumerate(MODERN_MIX_MODES):
        material, tree, output = _material(
            f"SVM Modern Mix Color {index:02d} {mode}"
        )
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Modern Mix Color {index:02d} {mode}"
        mix.data_type = "RGBA"
        mix.blend_type = mode
        mix.clamp_factor = index % 2 == 0
        mix.clamp_result = index % 3 == 0
        tree.links.new(
            _dynamic_factor(tree, f"Modern Mix Color {index:02d}"),
            _input_identifier(mix, "Factor_Float"),
        )
        _input_identifier(mix, "A_Color").default_value = (
            0.17,
            0.63,
            0.89,
            0.2,
        )
        _input_identifier(mix, "B_Color").default_value = (
            0.82,
            0.24,
            0.51,
            0.9,
        )
        emission = _emission(tree, output, f"Modern Mix Color {index:02d}")
        tree.links.new(
            _output_identifier(mix, "Result_Color"),
            _input(emission, "Color"),
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials + materials,
        columns=len(MODERN_MIX_MODES),
        rows=2,
        name="SVM Modern Mix Color Matrix",
        backfacing=set(range(len(materials), len(materials) * 2)),
        frame_bleed=0.1,
    )


def _svm_modern_mix_data_matrix(scene: Any) -> None:
    """Keep Cycles 5.2 MixFloat and both MixVector forms dynamic."""
    materials = []
    for use_clamp in (False, True):
        material, tree, output = _material(
            f"SVM Modern Mix Float clamp={int(use_clamp)}"
        )
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = "Modern Mix Float"
        mix.data_type = "FLOAT"
        mix.clamp_factor = use_clamp
        tree.links.new(
            _dynamic_factor(tree, "Modern Mix Float"),
            _input_identifier(mix, "Factor_Float"),
        )
        _input_identifier(mix, "A_Float").default_value = 0.2
        _input_identifier(mix, "B_Float").default_value = 0.8
        emission = _emission(tree, output, "Modern Mix Float")
        _input(emission, "Color").default_value = (0.31, 0.57, 0.83, 1.0)
        tree.links.new(
            _output_identifier(mix, "Result_Float"),
            _input(emission, "Strength"),
        )
        materials.append(material)

    for factor_mode in ("UNIFORM", "NON_UNIFORM"):
        for use_clamp in (False, True):
            material, tree, output = _material(
                "SVM Modern Mix Vector "
                f"{factor_mode} clamp={int(use_clamp)}"
            )
            mix = tree.nodes.new("ShaderNodeMix")
            mix.name = f"Modern Mix Vector {factor_mode}"
            mix.data_type = "VECTOR"
            mix.factor_mode = factor_mode
            mix.clamp_factor = use_clamp
            geometry = tree.nodes.new("ShaderNodeNewGeometry")
            geometry.name = f"Modern Mix Vector {factor_mode} Geometry"
            if factor_mode == "UNIFORM":
                tree.links.new(
                    _dynamic_factor(tree, "Modern Mix Vector Uniform"),
                    _input_identifier(mix, "Factor_Float"),
                )
            else:
                tree.links.new(
                    _output(geometry, "Normal"),
                    _input_identifier(mix, "Factor_Vector"),
                )
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
            emission = _emission(
                tree, output, f"Modern Mix Vector {factor_mode}"
            )
            tree.links.new(
                _output_identifier(mix, "Result_Vector"),
                _input(emission, "Color"),
            )
            materials.append(material)

    _material_matrix(
        scene,
        materials + materials,
        columns=len(materials),
        rows=2,
        name="SVM Modern Mix Data Matrix",
        backfacing=set(range(len(materials), len(materials) * 2)),
        frame_bleed=0.1,
    )


def _linked_float(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return _output(node, "Value")


def _linked_color(
    tree: Any, name: str, value: tuple[float, float, float, float]
) -> Any:
    node = tree.nodes.new("ShaderNodeRGB")
    node.name = name
    _output(node, "Color").default_value = value
    return _output(node, "Color")


def _svm_modern_mix_constant_matrix(scene: Any) -> None:
    """Force Cycles 5.2 Mix-node constant folders for every typed form."""
    materials = []
    for index, mode in enumerate(MODERN_MIX_MODES):
        material, tree, output = _material(
            f"SVM Modern Mix Color Constant {index:02d} {mode}"
        )
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Modern Mix Color Constant {index:02d} {mode}"
        mix.data_type = "RGBA"
        mix.blend_type = mode
        mix.clamp_factor = index % 2 == 0
        mix.clamp_result = index % 3 == 0
        tree.links.new(
            _linked_float(tree, f"Factor {index:02d}", 1.4),
            _input_identifier(mix, "Factor_Float"),
        )
        tree.links.new(
            _linked_color(
                tree, f"A {index:02d}", (0.17, 0.63, 0.89, 0.2)
            ),
            _input_identifier(mix, "A_Color"),
        )
        tree.links.new(
            _linked_color(
                tree, f"B {index:02d}", (0.82, 0.24, 0.51, 0.9)
            ),
            _input_identifier(mix, "B_Color"),
        )
        emission = _emission(
            tree, output, f"Modern Mix Color Constant {index:02d}"
        )
        tree.links.new(
            _output_identifier(mix, "Result_Color"),
            _input(emission, "Color"),
        )
        materials.append(material)

    for data_type, factor_mode in (
        ("FLOAT", "UNIFORM"),
        ("VECTOR", "UNIFORM"),
        ("VECTOR", "NON_UNIFORM"),
    ):
        for use_clamp in (False, True):
            name = f"{data_type} {factor_mode} clamp={int(use_clamp)}"
            material, tree, output = _material(
                f"SVM Modern Mix Constant {name}"
            )
            mix = tree.nodes.new("ShaderNodeMix")
            mix.name = f"Modern Mix Constant {name}"
            mix.data_type = data_type
            mix.clamp_factor = use_clamp
            if data_type == "FLOAT":
                tree.links.new(
                    _linked_float(tree, f"{name} Factor", 1.4),
                    _input_identifier(mix, "Factor_Float"),
                )
                tree.links.new(
                    _linked_float(tree, f"{name} A", 0.2),
                    _input_identifier(mix, "A_Float"),
                )
                tree.links.new(
                    _linked_float(tree, f"{name} B", 0.8),
                    _input_identifier(mix, "B_Float"),
                )
                emission = _emission(tree, output, name)
                _input(emission, "Color").default_value = (
                    0.31,
                    0.57,
                    0.83,
                    1.0,
                )
                tree.links.new(
                    _output_identifier(mix, "Result_Float"),
                    _input(emission, "Strength"),
                )
            else:
                mix.factor_mode = factor_mode
                if factor_mode == "UNIFORM":
                    tree.links.new(
                        _linked_float(tree, f"{name} Factor", 1.4),
                        _input_identifier(mix, "Factor_Float"),
                    )
                else:
                    tree.links.new(
                        _linked_vector(tree, f"{name} Factor", (-0.2, 0.5, 1.4)),
                        _input_identifier(mix, "Factor_Vector"),
                    )
                tree.links.new(
                    _linked_vector(tree, f"{name} A", (0.1, 0.7, -0.2)),
                    _input_identifier(mix, "A_Vector"),
                )
                tree.links.new(
                    _linked_vector(tree, f"{name} B", (0.9, -0.1, 0.6)),
                    _input_identifier(mix, "B_Vector"),
                )
                emission = _emission(tree, output, name)
                tree.links.new(
                    _output_identifier(mix, "Result_Vector"),
                    _input(emission, "Color"),
                )
            materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=len(materials),
        rows=1,
        name="SVM Modern Mix Constant Matrix",
        frame_bleed=0.1,
    )


def _svm_modern_mix_import_chain(scene: Any) -> None:
    """Exercise the exact Blender-to-Cycles socket mapping for every Mix type."""
    material, tree, output = _material("SVM Modern Mix Import Chain")

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Geometry"

    mix_float = tree.nodes.new("ShaderNodeMix")
    mix_float.name = "Mix Float"
    mix_float.data_type = "FLOAT"
    mix_float.clamp_factor = False
    tree.links.new(
        _output(geometry, "Backfacing"),
        _input_identifier(mix_float, "Factor_Float"),
    )
    _input_identifier(mix_float, "A_Float").default_value = 0.2
    _input_identifier(mix_float, "B_Float").default_value = 0.8

    mix_uniform = tree.nodes.new("ShaderNodeMix")
    mix_uniform.name = "Mix Vector Uniform"
    mix_uniform.data_type = "VECTOR"
    mix_uniform.factor_mode = "UNIFORM"
    mix_uniform.clamp_factor = True
    tree.links.new(
        _output_identifier(mix_float, "Result_Float"),
        _input_identifier(mix_uniform, "Factor_Float"),
    )
    _input_identifier(mix_uniform, "A_Vector").default_value = (
        0.1,
        0.7,
        -0.2,
    )
    _input_identifier(mix_uniform, "B_Vector").default_value = (
        0.9,
        -0.1,
        0.6,
    )

    mix_nonuniform = tree.nodes.new("ShaderNodeMix")
    mix_nonuniform.name = "Mix Vector Non Uniform"
    mix_nonuniform.data_type = "VECTOR"
    mix_nonuniform.factor_mode = "NON_UNIFORM"
    mix_nonuniform.clamp_factor = False
    tree.links.new(
        _output(geometry, "Normal"),
        _input_identifier(mix_nonuniform, "Factor_Vector"),
    )
    _input_identifier(mix_nonuniform, "A_Vector").default_value = (
        0.3,
        -0.4,
        0.8,
    )
    _input_identifier(mix_nonuniform, "B_Vector").default_value = (
        -0.2,
        0.6,
        0.1,
    )

    mix_color = tree.nodes.new("ShaderNodeMix")
    mix_color.name = "Mix Color"
    mix_color.data_type = "RGBA"
    mix_color.blend_type = "OVERLAY"
    mix_color.clamp_factor = False
    mix_color.clamp_result = True
    tree.links.new(
        _output(geometry, "Backfacing"),
        _input_identifier(mix_color, "Factor_Float"),
    )
    tree.links.new(
        _output_identifier(mix_uniform, "Result_Vector"),
        _input_identifier(mix_color, "A_Color"),
    )
    tree.links.new(
        _output_identifier(mix_nonuniform, "Result_Vector"),
        _input_identifier(mix_color, "B_Color"),
    )

    emission = _emission(tree, output, "Modern Mix Import Chain")
    tree.links.new(
        _output_identifier(mix_color, "Result_Color"),
        _input(emission, "Color"),
    )

    _material_matrix(
        scene,
        [material, material],
        columns=2,
        rows=1,
        name="SVM Modern Mix Import Chain",
        backfacing={1},
        frame_bleed=0.1,
    )


def _dynamic_color(tree: Any, name: str) -> Any:
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = f"{name} Geometry"
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = f"{name} Combine"
    combine.mode = "RGB"
    tree.links.new(_output(geometry, "Backfacing"), _input(combine, "Red"))
    _input(combine, "Green").default_value = -0.4
    _input(combine, "Blue").default_value = 1.2
    return _output(combine, "Color")


def _svm_modern_mix_fold_edges(scene: Any) -> None:
    """Force Cycles' linked-input modern Mix constant-fold edge paths."""
    materials = []

    material, tree, output = _material("SVM MixColor Same Link Clamp")
    color = _dynamic_color(tree, "Same Link")
    mix = tree.nodes.new("ShaderNodeMix")
    mix.name = "Same Link Mix Color"
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.clamp_factor = True
    mix.clamp_result = True
    _input_identifier(mix, "Factor_Float").default_value = 0.37
    tree.links.new(color, _input_identifier(mix, "A_Color"))
    tree.links.new(color, _input_identifier(mix, "B_Color"))
    emission = _emission(tree, output, "Same Link")
    tree.links.new(
        _output_identifier(mix, "Result_Color"),
        _input(emission, "Color"),
    )
    materials.append(material)

    material, tree, output = _material("SVM MixColor Factor Zero")
    color = _dynamic_color(tree, "Factor Zero")
    mix = tree.nodes.new("ShaderNodeMix")
    mix.name = "Factor Zero Mix Color"
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.clamp_factor = True
    mix.clamp_result = False
    _input_identifier(mix, "Factor_Float").default_value = 0.0
    tree.links.new(color, _input_identifier(mix, "A_Color"))
    _input_identifier(mix, "B_Color").default_value = (0.8, 0.2, 0.6, 1.0)
    emission = _emission(tree, output, "Factor Zero")
    tree.links.new(
        _output_identifier(mix, "Result_Color"),
        _input(emission, "Color"),
    )
    materials.append(material)

    material, tree, output = _material("SVM MixFloat Factor One")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Factor One Geometry"
    mix = tree.nodes.new("ShaderNodeMix")
    mix.name = "Factor One Mix Float"
    mix.data_type = "FLOAT"
    mix.clamp_factor = True
    _input_identifier(mix, "Factor_Float").default_value = 1.0
    _input_identifier(mix, "A_Float").default_value = 0.2
    tree.links.new(
        _output(geometry, "Backfacing"),
        _input_identifier(mix, "B_Float"),
    )
    emission = _emission(tree, output, "Factor One")
    _input(emission, "Color").default_value = (0.31, 0.57, 0.83, 1.0)
    tree.links.new(
        _output_identifier(mix, "Result_Float"),
        _input(emission, "Strength"),
    )
    materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=len(materials),
        rows=1,
        name="SVM Modern Mix Fold Edges",
        frame_bleed=0.1,
    )
