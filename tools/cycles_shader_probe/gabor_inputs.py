"""Gabor Texture probes against raw Blender/Cycles node graphs."""

from __future__ import annotations

from typing import Any

from .support import (
    _input,
    _input_identifier,
    _material,
    _material_matrix,
    _output,
)


def _gabor_material(
    name: str,
    *,
    gabor_type: str,
    output_name: str,
    coordinates: tuple[float, float, float],
    scale: float,
    frequency: float,
    anisotropy: float,
    orientation_2d: float,
    orientation_3d: tuple[float, float, float],
) -> Any:
    material, tree, output = _material(name)
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = f"{name} Geometry"
    coordinate_add = tree.nodes.new("ShaderNodeVectorMath")
    coordinate_add.name = f"{name} Runtime Coordinates"
    coordinate_add.operation = "ADD"
    _input_identifier(coordinate_add, "Vector_001").default_value = coordinates

    gabor = tree.nodes.new("ShaderNodeTexGabor")
    gabor.name = f"{name} Gabor"
    gabor.gabor_type = gabor_type
    _input(gabor, "Scale").default_value = scale
    _input(gabor, "Frequency").default_value = frequency
    _input(gabor, "Anisotropy").default_value = anisotropy
    _input_identifier(gabor, "Orientation 2D").default_value = orientation_2d
    _input_identifier(gabor, "Orientation 3D").default_value = orientation_3d

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    tree.links.new(
        _output(geometry, "Normal"),
        _input_identifier(coordinate_add, "Vector"),
    )
    tree.links.new(
        _output(coordinate_add, "Vector"),
        _input(gabor, "Vector"),
    )
    tree.links.new(_output(gabor, output_name), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _svm_gabor_matrix(scene: Any) -> None:
    """Force all Gabor variants and outputs through Cycles' SVM handler."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("2D", "Value", (-0.73, 0.41, -1.17), 2.3, 0.0, 0.0, -1.2, (1.0, 2.0, 0.3)),
        ("2D", "Phase", (0.27, -0.91, 0.36), -1.7, 0.73, 0.35, 0.0, (0.2, -0.4, 1.3)),
        ("2D", "Intensity", (1.19, 0.05, -0.42), 0.75, 2.0, 1.0, 2.4, (-2.0, 0.5, 0.7)),
        ("2D", "Value", (-1.37, 1.11, 0.22), 4.1, 5.7, 0.68, 5.2, (0.0, 1.0, 0.0)),
        ("2D", "Phase", (0.0, 0.0, 0.0), 0.0, 1.3, 0.12, -3.0, (1.0, 0.0, 0.0)),
        ("2D", "Intensity", (2.03, -1.77, 0.9), 1.9, 11.0, 0.92, 0.79, (0.3, 0.8, -0.1)),
        ("3D", "Value", (-0.83, 0.29, -1.41), 2.3, 0.0, 0.0, 0.0, (1.0, 2.0, 0.3)),
        ("3D", "Phase", (0.47, -1.21, 0.66), -1.7, 0.73, 0.35, 1.1, (0.2, -0.4, 1.3)),
        ("3D", "Intensity", (1.49, 0.15, -0.72), 0.75, 2.0, 1.0, -2.7, (-2.0, 0.5, 0.7)),
        ("3D", "Value", (-1.07, 1.31, 0.52), 4.1, 5.7, 0.68, 2.2, (0.4, 1.1, -0.6)),
        ("3D", "Phase", (0.2, -0.3, 0.4), 0.0, 1.3, 0.12, -0.2, (1.0, 0.0, 0.0)),
        ("3D", "Intensity", (2.33, -1.47, 1.2), 1.9, 11.0, 0.92, 0.39, (0.3, 0.8, -0.1)),
    )
    materials = []
    for index, case in enumerate(cases):
        (
            gabor_type,
            output_name,
            coordinates,
            scale,
            frequency,
            anisotropy,
            orientation_2d,
            orientation_3d,
        ) = case
        materials.append(
            _gabor_material(
                f"SVM Gabor {index:02d} {gabor_type} {output_name}",
                gabor_type=gabor_type,
                output_name=output_name,
                coordinates=coordinates,
                scale=scale,
                frequency=frequency,
                anisotropy=anisotropy,
                orientation_2d=orientation_2d,
                orientation_3d=orientation_3d,
            )
        )
    _material_matrix(
        scene,
        materials,
        columns=6,
        rows=2,
        name="SVM Gabor Matrix",
    )
