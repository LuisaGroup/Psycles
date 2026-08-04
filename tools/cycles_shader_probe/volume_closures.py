"""Closed-boundary Cycles volume-closure transport probes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import _input, _material, _output


def _volume_emission_transport(scene: Any) -> None:
    """Exercise an Emission closure through the material Volume domain."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.light_sampling_threshold = 0.0

    material, tree, output = _material("Volume Emission")
    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.name = "Volume Light Path"
    strength = tree.nodes.new("ShaderNodeMath")
    strength.name = "Camera Emission Strength"
    strength.operation = "MULTIPLY"
    _input(strength, "Value_001").default_value = 0.7
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Volume Emission"
    _input(emission, "Color").default_value = (0.11, 0.37, 0.83, 1.0)
    tree.links.new(
        _output(light_path, "Is Camera Ray"),
        _input(strength, "Value"),
    )
    tree.links.new(
        _output(strength, "Value"),
        _input(emission, "Strength"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Volume"),
    )

    bpy.ops.mesh.primitive_cube_add(
        size=1.6,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, 0.0),
    )
    volume = bpy.context.object
    volume.name = "Closed Emission Volume"
    volume.scale = (1.0, 0.72, 0.58)
    volume.data.materials.append(material)
