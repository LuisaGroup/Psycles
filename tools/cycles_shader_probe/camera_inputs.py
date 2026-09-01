"""Cycles 5.2 camera and path-state node probes."""

from __future__ import annotations

from typing import Any

from mathutils import Vector

from .support import (
    _input,
    _input_identifier,
    _material,
    _material_matrix,
    _output,
    _plane,
)


_LIGHT_PATH_OUTPUTS = (
    "Is Camera Ray",
    "Is Shadow Ray",
    "Is Diffuse Ray",
    "Is Glossy Ray",
    "Is Singular Ray",
    "Is Reflection Ray",
    "Is Transmission Ray",
    "Is Volume Scatter Ray",
    "Ray Length",
    "Ray Depth",
    "Diffuse Depth",
    "Glossy Depth",
    "Transparent Depth",
    "Transmission Depth",
    "Portal Depth",
)


def _camera_data(scene: Any) -> None:
    """Keep all three Camera Data outputs live under a nontrivial transform."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01

    camera = scene.camera
    camera.location = (1.1, -0.8, 3.2)
    direction = Vector((0.0, 0.0, 0.0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    material, tree, output = _material("Camera Data")
    data = tree.nodes.new("ShaderNodeCameraData")
    data.name = "Camera Data"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = "Separate View Vector"
    tree.links.new(_output(data, "View Vector"), _input(separate, "Vector"))

    view_x = tree.nodes.new("ShaderNodeMath")
    view_x.name = "Map View X"
    view_x.operation = "MULTIPLY_ADD"
    _input_identifier(view_x, "Value_001").default_value = 0.5
    _input_identifier(view_x, "Value_002").default_value = 0.5
    tree.links.new(
        _output(separate, "X"), _input_identifier(view_x, "Value")
    )

    z_depth = tree.nodes.new("ShaderNodeMath")
    z_depth.name = "Scale Z Depth"
    z_depth.operation = "MULTIPLY"
    _input_identifier(z_depth, "Value_001").default_value = -0.2
    tree.links.new(
        _output(data, "View Z Depth"),
        _input_identifier(z_depth, "Value"),
    )

    distance = tree.nodes.new("ShaderNodeMath")
    distance.name = "Scale View Distance"
    distance.operation = "MULTIPLY"
    _input_identifier(distance, "Value_001").default_value = 0.2
    tree.links.new(
        _output(data, "View Distance"),
        _input_identifier(distance, "Value"),
    )

    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = "Camera Data RGB"
    tree.links.new(_output(view_x, "Value"), _input(combine, "X"))
    tree.links.new(_output(z_depth, "Value"), _input(combine, "Y"))
    tree.links.new(_output(distance, "Value"), _input(combine, "Z"))

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(_output(combine, "Vector"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    _plane(material)


def _light_path_matrix(scene: Any) -> None:
    """Expose every Cycles 5.2 Light Path output on primary camera rays."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = []
    for index, output_name in enumerate(_LIGHT_PATH_OUTPUTS):
        material, tree, output = _material(
            f"Light Path {index:02d} {output_name}"
        )
        light_path = tree.nodes.new("ShaderNodeLightPath")
        light_path.name = "Light Path"
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "Emission"
        tree.links.new(
            _output(light_path, output_name),
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
        columns=5,
        rows=3,
        name="Light Path Matrix",
        frame_bleed=0.01,
    )
