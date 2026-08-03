"""Environment Texture probes against the raw Blender/Cycles node graph."""

from __future__ import annotations

from typing import Any

import numpy as np

from .support import (
    _input,
    _material,
    _material_matrix,
    _output,
    _world,
)
from .texture_inputs import _packed_rgba_image


def _direction_image(name: str) -> Any:
    height = 5
    width = 7
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixels[y, x] = (
                (17 + 53 * x + 29 * y) % 256,
                (37 + 19 * x + 71 * y) % 256,
                (239 + 83 * x + 11 * y) % 256,
                255,
            )
    return _packed_rgba_image(
        name,
        pixels,
        colorspace="Non-Color",
        alpha_mode="STRAIGHT",
    )


def _environment_texture_projection_modes(scene: Any) -> None:
    """Exercise Cycles equirectangular and mirror-ball direction maps."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    image = _direction_image("Probe Environment Texture")
    directions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (2.0, -3.0, 4.0),
    )
    materials = []
    for projection in ("EQUIRECTANGULAR", "MIRROR_BALL"):
        for index, direction in enumerate(directions):
            name = f"Environment {projection} {index:02d}"
            material, tree, output = _material(name)
            mapping = tree.nodes.new("ShaderNodeMapping")
            mapping.name = f"{name} Direction"
            mapping.vector_type = "POINT"
            _input(mapping, "Vector").default_value = direction
            texture = tree.nodes.new("ShaderNodeTexEnvironment")
            texture.name = name
            texture.image = image
            texture.interpolation = "Linear"
            texture.projection = projection
            tree.links.new(
                _output(mapping, "Vector"),
                _input(texture, "Vector"),
            )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"{name} Emission"
            tree.links.new(
                _output(texture, "Color"),
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
        name="Environment Projection Modes Matrix",
    )


def _environment_texture_sampling_modes(scene: Any) -> None:
    """Exercise every exposed interpolation with sRGB RGBA storage."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    image = _packed_rgba_image(
        "Probe sRGB Environment Texture",
        np.asarray(
            [
                [
                    (7, 19, 233, 31),
                    (41, 211, 67, 83),
                    (173, 29, 109, 149),
                    (251, 137, 11, 223),
                    (97, 59, 197, 47),
                ],
                [
                    (227, 73, 17, 199),
                    (61, 151, 239, 101),
                    (131, 239, 53, 181),
                    (19, 97, 157, 59),
                    (199, 31, 79, 241),
                ],
                [
                    (83, 229, 127, 113),
                    (149, 43, 193, 167),
                    (239, 181, 23, 71),
                    (53, 7, 211, 137),
                    (167, 113, 61, 211),
                ],
                [
                    (31, 109, 179, 251),
                    (211, 67, 97, 43),
                    (107, 197, 37, 191),
                    (71, 241, 139, 89),
                    (233, 17, 151, 157),
                ],
            ],
            dtype=np.uint8,
        ),
        colorspace="sRGB",
        alpha_mode="STRAIGHT",
    )
    materials = []
    for projection, direction in (
        ("EQUIRECTANGULAR", (0.37, -0.61, 0.71)),
        ("MIRROR_BALL", (-0.43, 0.29, 0.83)),
    ):
        for interpolation in ("Closest", "Linear", "Cubic", "Smart"):
            name = f"Environment {projection} {interpolation}"
            material, tree, output = _material(name)
            mapping = tree.nodes.new("ShaderNodeMapping")
            mapping.name = f"{name} Direction"
            mapping.vector_type = "POINT"
            _input(mapping, "Vector").default_value = direction
            texture = tree.nodes.new("ShaderNodeTexEnvironment")
            texture.name = name
            texture.image = image
            texture.interpolation = interpolation
            texture.projection = projection
            tree.links.new(
                _output(mapping, "Vector"),
                _input(texture, "Vector"),
            )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"{name} Emission"
            tree.links.new(
                _output(texture, "Color"),
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
        rows=2,
        name="Environment Sampling Modes Matrix",
    )


def _environment_texture_world_default(scene: Any) -> None:
    """Exercise Cycles' implicit world-ray Vector input without baking."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.camera.data.type = "PERSP"
    scene.camera.data.lens = 35.0
    image = _direction_image("Probe World Environment Texture")
    _, tree, background = _world(
        scene,
        (0.0, 0.0, 0.0, 1.0),
        1.0,
    )
    texture = tree.nodes.new("ShaderNodeTexEnvironment")
    texture.name = "Unlinked World Environment"
    texture.image = image
    texture.interpolation = "Linear"
    texture.projection = "EQUIRECTANGULAR"
    tree.links.new(
        _output(texture, "Color"),
        _input(background, "Color"),
    )
