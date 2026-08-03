"""Wave Texture probes against raw Blender/Cycles node graphs."""

from __future__ import annotations

from typing import Any

from .support import (
    _input,
    _material,
    _material_matrix,
    _output,
    _output_identifier,
)


def _constant_vector(tree: Any, name: str, value: tuple[float, ...]) -> Any:
    node = tree.nodes.new("ShaderNodeRGB")
    node.name = name
    _output(node, "Color").default_value = (*value, 1.0)
    return node


def _linked_value(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return node


def _wave_material(
    name: str,
    *,
    wave_type: str,
    direction: str,
    profile: str,
    coordinates: tuple[float, float, float] | None,
    scale: float,
    distortion: float,
    detail: float,
    detail_scale: float,
    detail_roughness: float,
    phase: float,
    output_name: str,
    link_scalars: bool = False,
) -> Any:
    material, tree, output = _material(name)
    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.name = name
    wave.wave_type = wave_type
    if wave_type == "BANDS":
        wave.bands_direction = direction
    else:
        wave.rings_direction = direction
    wave.wave_profile = profile

    if coordinates is not None:
        vector = _constant_vector(
            tree,
            f"{name} Coordinates",
            coordinates,
        )
        tree.links.new(
            _output(vector, "Color"),
            _input(wave, "Vector"),
        )

    scalar_inputs = (
        ("Scale", scale),
        ("Distortion", distortion),
        ("Detail", detail),
        ("Detail Scale", detail_scale),
        ("Detail Roughness", detail_roughness),
        ("Phase Offset", phase),
    )
    for input_name, value in scalar_inputs:
        if link_scalars:
            source = _linked_value(
                tree,
                f"{name} {input_name}",
                value,
            )
            tree.links.new(
                _output(source, "Value"),
                _input(wave, input_name),
            )
        else:
            _input(wave, input_name).default_value = value

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    wave_output = (
        _output_identifier(wave, "Fac")
        if output_name == "Fac"
        else _output(wave, "Color")
    )
    tree.links.new(wave_output, _input(emission, "Color"))
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _wave_texture_modes(scene: Any) -> None:
    """Cover both types, every direction/profile, and both outputs."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = []
    index = 0
    directions = {
        "BANDS": ("X", "Y", "Z", "DIAGONAL"),
        "RINGS": ("X", "Y", "Z", "SPHERICAL"),
    }
    for wave_type, type_directions in directions.items():
        for direction_index, direction in enumerate(type_directions):
            for profile_index, profile in enumerate(("SIN", "SAW", "TRI")):
                name = (
                    f"Wave {index:02d} {wave_type} "
                    f"{direction} {profile}"
                )
                coordinates = (
                    0.173 + 0.071 * index,
                    -0.625 + 0.037 * direction_index,
                    1.375 - 0.043 * profile_index,
                )
                materials.append(
                    _wave_material(
                        name,
                        wave_type=wave_type,
                        direction=direction,
                        profile=profile,
                        coordinates=coordinates,
                        scale=(-1.7 if index % 5 == 0 else 2.35),
                        distortion=0.0,
                        detail=2.0,
                        detail_scale=1.0,
                        detail_roughness=0.5,
                        phase=-0.83 + 0.19 * profile_index,
                        output_name="Color" if index % 2 == 0 else "Fac",
                    )
                )
                index += 1

    # Eight additional cells pin the precision correction at signed unit and
    # zero coordinates, negative/zero scale, and the implicit Generated input.
    edge_cases = (
        ("BANDS", "X", "SAW", (1.0, 0.0, 0.0), 1.0, 0.0, "Color"),
        ("BANDS", "Y", "TRI", (0.0, -1.0, 0.0), 1.0, 0.0, "Fac"),
        ("BANDS", "Z", "SIN", (0.0, 0.0, 0.0), 0.0, 1.57079637, "Color"),
        ("BANDS", "DIAGONAL", "SAW", (-2.0, 3.0, -4.0), -0.5, -2.3, "Fac"),
        ("RINGS", "X", "TRI", (0.0, 0.0, 0.0), 3.0, 0.25, "Color"),
        ("RINGS", "Y", "SIN", (10000.0, -0.5, 0.25), 0.001, -0.7, "Fac"),
        ("RINGS", "Z", "SAW", None, 3.75, 0.31, "Color"),
        ("RINGS", "SPHERICAL", "TRI", None, -2.25, -1.13, "Fac"),
    )
    for edge_index, (
        wave_type,
        direction,
        profile,
        coordinates,
        scale,
        phase,
        output_name,
    ) in enumerate(edge_cases):
        name = f"Wave Edge {edge_index:02d} {wave_type} {direction} {profile}"
        materials.append(
            _wave_material(
                name,
                wave_type=wave_type,
                direction=direction,
                profile=profile,
                coordinates=coordinates,
                scale=scale,
                distortion=0.0,
                detail=2.0,
                detail_scale=1.0,
                detail_roughness=0.5,
                phase=phase,
                output_name=output_name,
            )
        )

    _material_matrix(
        scene,
        materials,
        columns=8,
        rows=4,
        name="Wave Texture Modes Matrix",
    )


def _wave_texture_distortion(scene: Any) -> None:
    """Cover Cycles' exact normalized 3D fBm distortion recurrence."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (0.0, 0.0, 1.0, 0.0, 0.0),
        # A linked negative fractional Detail pins Cycles' truncation-based
        # octave bound (`i <= float_to_int(detail)`), which differs from floor.
        (0.25, -0.5, 0.73, 0.5, 0.3),
        (-0.75, 1.25, -1.7, 0.2, -0.9),
        (1.5, 2.375, 2.17, 0.63, 1.1),
        (4.855, 16.0, 2.1, 0.5, 1.57079637),
        (0.94, 4.0, 1.0, 0.5, 1.57079637),
        (0.465, 4.0, 1.0, 0.5, 1.57079637),
        (-2.25, 3.75, -0.875, 1.0, -2.4),
        (3.0, 7.5, 0.125, 0.0, 0.75),
        (0.001, 15.0, 8.0, 0.9, -1.25),
        (8.0, 0.25, -3.0, 0.35, 2.75),
        (-0.125, 12.875, 0.0, 0.75, -0.5),
        (2.0, 1.0, 100.0, 0.1, 0.0),
        (1.0, 5.5, -100.0, 0.85, 3.14159274),
        (-4.0, 9.25, 0.333, 0.55, -3.14159274),
        (0.5, 14.5, 1.999, 0.42, 6.28318548),
    )
    materials = []
    for index, (
        distortion,
        detail,
        detail_scale,
        roughness,
        phase,
    ) in enumerate(cases):
        wave_type = "BANDS" if index % 2 == 0 else "RINGS"
        direction = (
            ("X", "Y", "Z", "DIAGONAL")[index % 4]
            if wave_type == "BANDS"
            else ("X", "Y", "Z", "SPHERICAL")[index % 4]
        )
        profile = ("SIN", "SAW", "TRI")[index % 3]
        name = f"Wave Distortion {index:02d}"
        materials.append(
            _wave_material(
                name,
                wave_type=wave_type,
                direction=direction,
                profile=profile,
                coordinates=(
                    0.173 + 0.113 * index,
                    -0.625 + 0.071 * index,
                    1.375 - 0.097 * index,
                ),
                scale=(12.0, 64.0, 80.0, -7.5)[index % 4],
                distortion=distortion,
                detail=detail,
                detail_scale=detail_scale,
                detail_roughness=roughness,
                phase=phase,
                output_name="Color" if index % 2 == 0 else "Fac",
                link_scalars=True,
            )
        )

    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Wave Texture Distortion Matrix",
    )
