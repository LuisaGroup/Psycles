"""Closed-boundary Cycles volume-closure transport probes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import _input, _material, _output


def _volume_coefficients_svm(scene: Any) -> None:
    """Original Cycles typed coefficient payloads, including linked weights."""
    phases = ("HENYEY_GREENSTEIN", "FOURNIER_FORAND", "DRAINE", "RAYLEIGH", "MIE")
    for index, phase in enumerate((*phases, "HENYEY_GREENSTEIN", "HENYEY_GREENSTEIN")):
        label = phase if index < len(phases) else ("Linked" if index == len(phases) else "Repeated")
        material, tree, output = _material(f"Volume Coefficients {label}")
        volume = tree.nodes.new("ShaderNodeVolumeCoefficients")
        volume.phase = phase
        values = {
            "Scatter Coefficients": (0.17, 0.43, 0.79),
            "Absorption Coefficients": (0.11, 0.29, 0.53),
            "Emission Coefficients": (0.07, 0.13, 0.31),
        } | {
            "HENYEY_GREENSTEIN": {"Anisotropy": -0.25},
            "FOURNIER_FORAND": {"IOR": 1.37, "Backscatter": 0.16},
            "DRAINE": {"Anisotropy": -0.25, "Alpha": 0.46},
            "RAYLEIGH": {},
            "MIE": {"Diameter": 16.0},
        }[phase]
        for name, value in values.items():
            _input(volume, name).default_value = value
        terminal = _output(volume, "Volume")
        if label == "Linked":
            path = tree.nodes.new("ShaderNodeLightPath")
            vector = tree.nodes.new("ShaderNodeCombineXYZ")
            _input(vector, "Y").default_value = 0.3
            _input(vector, "Z").default_value = 0.7
            tree.links.new(_output(path, "Is Camera Ray"), _input(vector, "X"))
            tree.links.new(_output(vector, "Vector"), _input(volume, "Scatter Coefficients"))
            tree.links.new(_output(path, "Ray Length"), _input(volume, "Anisotropy"))
            absorption = tree.nodes.new("ShaderNodeVolumeAbsorption")
            mix = tree.nodes.new("ShaderNodeMixShader")
            tree.links.new(_output(path, "Is Camera Ray"), mix.inputs[0])
            tree.links.new(terminal, mix.inputs[1])
            tree.links.new(_output(absorption, "Volume"), mix.inputs[2])
            terminal = _output(mix, "Shader")
        elif label == "Repeated":
            duplicate = tree.nodes.new("ShaderNodeVolumeCoefficients")
            duplicate.phase = phase
            for name, value in values.items():
                _input(duplicate, name).default_value = value
            add = tree.nodes.new("ShaderNodeAddShader")
            tree.links.new(terminal, add.inputs[0])
            tree.links.new(_output(duplicate, "Volume"), add.inputs[1])
            terminal = _output(add, "Shader")
        tree.links.new(terminal, _input(output, "Volume"))
        bpy.ops.mesh.primitive_cube_add(size=0.4, location=(-1.25 + 0.5 * index, 0.0, 0.0))
        bpy.context.object.data.materials.append(material)


def _principled_volume_svm(scene: Any, variant: str = "Default") -> None:
    """Keep symbolic grid names and dynamic inputs in an external word oracle."""
    material, tree, output = _material(f"Principled Volume {variant}")
    volume = tree.nodes.new("ShaderNodeVolumePrincipled")
    for name, value in {
        "Color": (0.17, 0.43, 0.79, 1.0),
        "Absorption Color": (0.04, 0.25, 0.81, 1.0),
        "Emission Color": (0.1, 0.25, 0.6, 1.0),
        "Blackbody Tint": (0.75, 0.5, 0.25, 1.0),
        "Density": 0.75,
        "Anisotropy": -0.2,
        "Emission Strength": 0.8,
        "Blackbody Intensity": 0.65,
        "Temperature": 1800.0,
    }.items():
        _input(volume, name).default_value = value
    if variant == "Named":
        _input(volume, "Density Attribute").default_value = "probe_density"
        _input(volume, "Color Attribute").default_value = "probe_color"
        _input(volume, "Temperature Attribute").default_value = "probe_temperature"
    terminal = _output(volume, "Volume")
    if variant == "Linked":
        path = tree.nodes.new("ShaderNodeLightPath")
        tree.links.new(_output(path, "Is Camera Ray"), _input(volume, "Density"))
        tree.links.new(_output(path, "Ray Length"), _input(volume, "Color"))
        tree.links.new(_output(path, "Is Shadow Ray"), _input(volume, "Blackbody Intensity"))
        absorption = tree.nodes.new("ShaderNodeVolumeAbsorption")
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(_output(path, "Is Camera Ray"), mix.inputs[0])
        tree.links.new(terminal, mix.inputs[1])
        tree.links.new(_output(absorption, "Volume"), mix.inputs[2])
        terminal = _output(mix, "Shader")
    tree.links.new(terminal, _input(output, "Volume"))
    bpy.ops.mesh.primitive_cube_add(size=1.5)
    bpy.context.object.data.materials.append(material)


def _principled_volume_named_svm(scene: Any) -> None:
    _principled_volume_svm(scene, "Named")


def _principled_volume_linked_svm(scene: Any) -> None:
    _principled_volume_svm(scene, "Linked")


def _volume_scatter_svm(scene: Any) -> None:
    """Emit every Cycles 5.2 Scatter Volume phase payload."""
    phases = (
        "HENYEY_GREENSTEIN",
        "FOURNIER_FORAND",
        "DRAINE",
        "RAYLEIGH",
        "MIE",
    )
    for index, phase in enumerate(phases):
        material, tree, output = _material(f"Volume Scatter {phase}")
        scatter = tree.nodes.new("ShaderNodeVolumeScatter")
        scatter.name = f"Scatter {phase}"
        scatter.phase = phase
        _input(scatter, "Color").default_value = (
            0.17 + 0.05 * index,
            0.43,
            0.79 - 0.04 * index,
            1.0,
        )
        _input(scatter, "Density").default_value = 0.7 + 0.1 * index
        if phase == "HENYEY_GREENSTEIN":
            _input(scatter, "Anisotropy").default_value = -0.25
        elif phase == "FOURNIER_FORAND":
            _input(scatter, "IOR").default_value = 1.37
            _input(scatter, "Backscatter").default_value = 0.16
        elif phase == "DRAINE":
            _input(scatter, "Anisotropy").default_value = -0.05
            _input(scatter, "Alpha").default_value = 0.46
        elif phase == "MIE":
            _input(scatter, "Diameter").default_value = 16.0
        tree.links.new(
            _output(scatter, "Volume"),
            _input(output, "Volume"),
        )
        bpy.ops.mesh.primitive_cube_add(
            size=0.5,
            enter_editmode=False,
            align="WORLD",
            location=(-1.2 + 0.6 * index, 0.0, 0.0),
        )
        bpy.context.object.data.materials.append(material)


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
