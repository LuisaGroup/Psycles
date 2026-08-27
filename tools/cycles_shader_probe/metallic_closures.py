"""Cycles 5.2 standalone Metallic BSDF probe scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bpy

from .support import (
    _bsdf_matrix_sun,
    _input,
    _linked_vector,
    _material,
    _material_matrix,
    _output,
)


@dataclass(frozen=True)
class _MetallicCase:
    fresnel: str
    distribution: str
    color_or_ior: tuple[float, float, float]
    tint_or_extinction: tuple[float, float, float]
    roughness: float
    anisotropy: float = 0.0
    rotation: float = 0.0
    film_thickness: float = 0.0
    film_ior: float = 1.33
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tangent: tuple[float, float, float] = (1.0, 0.0, 0.0)


def _linked_float(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return _output(node, "Value")


def _metallic_bsdf_matrix(scene: Any) -> None:
    """Exercise the unmodified Cycles 5.2 standalone Metallic closure.

    The 16 equal-area cells form a finite product cover of the two static
    Fresnel models, all three microfacet distributions, anisotropy/rotation,
    input-domain saturation, and both thin-film branches. Every authored
    numeric socket is linked through a typed Blender node. This prevents
    constant-input pruning while preserving the raw closure graph: neither
    Blender/Cycles nor the exporter converts the material to another BSDF or
    pre-bakes a Fresnel result.
    """
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    _bsdf_matrix_sun(scene, transmission=False)
    # Oblique deterministic illumination distinguishes Fresnel models,
    # distributions, anisotropic axes, and tangent rotation without adding a
    # sampled finite emitter.
    bpy.data.objects["BSDF Matrix Sun"].rotation_euler[1] = 0.92

    cases = (
        # F82-tint: ordinary distributions, multiple scattering, saturation,
        # anisotropic orientation, and both zero/active film controls.
        _MetallicCase("F82", "GGX", (0.72, 0.14, 0.035),
                      (0.92, 0.42, 0.12), 0.18),
        _MetallicCase("F82", "BECKMANN", (0.16, 0.62, 0.82),
                      (0.74, 0.86, 0.98), 0.35),
        _MetallicCase("F82", "MULTI_GGX", (0.58, 0.48, 0.08),
                      (0.92, 0.78, 0.34), 0.72),
        _MetallicCase("F82", "GGX", (-0.25, 0.45, 1.20),
                      (1.30, -0.10, 0.70), 0.42),
        _MetallicCase("F82", "GGX", (0.68, 0.24, 0.09),
                      (0.96, 0.56, 0.24), 0.31, 0.78, 0.0),
        _MetallicCase("F82", "GGX", (0.68, 0.24, 0.09),
                      (0.96, 0.56, 0.24), 0.31, 0.78, 0.25),
        _MetallicCase("F82", "MULTI_GGX", (0.18, 0.54, 0.88),
                      (0.64, 0.90, 0.99), 0.63, 0.90, 0.125,
                      tangent=(0.6, 0.8, 0.0)),
        _MetallicCase("F82", "MULTI_GGX", (0.62, 0.22, 0.06),
                      (0.96, 0.48, 0.18), 0.46, 0.55, 0.18,
                      420.0, 1.52, tangent=(0.8, 0.6, 0.0)),
        # Physical complex IOR: the same semantic axes with n/k stored in the
        # original Vector sockets rather than converted to F0/edge tint.
        _MetallicCase("PHYSICAL_CONDUCTOR", "GGX",
                      (0.27, 0.68, 1.32), (3.61, 2.62, 1.91), 0.18),
        _MetallicCase("PHYSICAL_CONDUCTOR", "BECKMANN",
                      (1.45, 0.84, 0.38), (1.92, 2.58, 3.42), 0.35),
        _MetallicCase("PHYSICAL_CONDUCTOR", "MULTI_GGX",
                      (2.76, 2.51, 2.23), (3.87, 3.40, 3.01), 0.72),
        _MetallicCase("PHYSICAL_CONDUCTOR", "GGX",
                      (-0.40, 0.70, 2.10), (3.00, -0.20, 1.40), 0.42),
        _MetallicCase("PHYSICAL_CONDUCTOR", "GGX",
                      (0.33, 0.91, 1.77), (3.42, 2.35, 1.28),
                      0.31, 0.78, 0.0),
        _MetallicCase("PHYSICAL_CONDUCTOR", "GGX",
                      (0.33, 0.91, 1.77), (3.42, 2.35, 1.28),
                      0.31, 0.78, 0.25),
        _MetallicCase("PHYSICAL_CONDUCTOR", "MULTI_GGX",
                      (1.20, 0.72, 0.31), (2.10, 2.84, 3.72),
                      0.63, 0.90, 0.125, tangent=(0.6, 0.8, 0.0)),
        _MetallicCase("PHYSICAL_CONDUCTOR", "MULTI_GGX",
                      (0.42, 0.88, 1.64), (3.72, 2.54, 1.36),
                      0.46, 0.55, 0.18, 610.0, 1.70,
                      normal=(0.24, 0.0, 0.97),
                      tangent=(0.8, 0.6, 0.0)),
    )

    materials = []
    for index, case in enumerate(cases):
        material, tree, output = _material(
            f"Metallic BSDF Matrix {index:02d}"
        )
        closure = tree.nodes.new("ShaderNodeBsdfMetallic")
        closure.name = f"Raw Metallic BSDF {index:02d}"
        closure.fresnel_type = case.fresnel
        closure.distribution = case.distribution

        if case.fresnel == "F82":
            tree.links.new(
                _linked_vector(tree, f"Linked Base Color {index:02d}",
                               case.color_or_ior),
                _input(closure, "Base Color"),
            )
            tree.links.new(
                _linked_vector(tree, f"Linked Edge Tint {index:02d}",
                               case.tint_or_extinction),
                _input(closure, "Edge Tint"),
            )
        elif case.fresnel == "PHYSICAL_CONDUCTOR":
            tree.links.new(
                _linked_vector(tree, f"Linked IOR {index:02d}",
                               case.color_or_ior),
                _input(closure, "IOR"),
            )
            tree.links.new(
                _linked_vector(tree, f"Linked Extinction {index:02d}",
                               case.tint_or_extinction),
                _input(closure, "Extinction"),
            )
        else:
            raise AssertionError(f"unknown Metallic Fresnel {case.fresnel!r}")

        for socket_name, value in (
            ("Roughness", case.roughness),
            ("Anisotropy", case.anisotropy),
            ("Rotation", case.rotation),
            ("Thin Film Thickness", case.film_thickness),
            ("Thin Film IOR", case.film_ior),
        ):
            tree.links.new(
                _linked_float(
                    tree, f"Linked {socket_name} {index:02d}", value
                ),
                _input(closure, socket_name),
            )
        tree.links.new(
            _linked_vector(
                tree, f"Linked Metallic Normal {index:02d}", case.normal
            ),
            _input(closure, "Normal"),
        )
        if case.anisotropy != 0.0:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Linked Metallic Tangent {index:02d}",
                    case.tangent,
                ),
                _input(closure, "Tangent"),
            )
        tree.links.new(_output(closure, "BSDF"), _input(output, "Surface"))
        materials.append(material)

    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Cycles 5.2 Metallic BSDF Matrix",
        frame_bleed=0.02,
    )
    surface.visible_shadow = False
