"""Standalone Refraction closure probe scenes."""

from __future__ import annotations

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


def _refraction_bsdf_matrix(scene: Any) -> None:
    """Exercise Cycles' standalone, pure-transmission Refraction BSDF."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    _bsdf_matrix_sun(scene, transmission=True)
    # Keep the zero-solid-angle light deterministic while moving it away
    # from the microfacet peak. This exercises the full refractive Jacobian
    # without one near-singular normal-incidence value dominating the matrix.
    bpy.data.objects["BSDF Matrix Sun"].rotation_euler = (
        3.141592653589793,
        0.38,
        0.17,
    )
    # Stop after camera-hit direct-light evaluation. Sampled refraction,
    # including eta=1 and TIR singular cases, is covered deterministically by
    # the three-backend Luisa closure regression.
    scene.cycles.max_bounces = 0

    # distribution, roughness, IOR, color, explicit normal, backface,
    # linked color. The first cell is Barbershop's disinfectant material.
    cases = (
        (
            "BECKMANN",
            0.137102127,
            1.159999967,
            (0.382274, 0.651278, 0.868007),
            None,
            False,
            False,
        ),
        (
            "GGX",
            0.137102127,
            1.159999967,
            (0.382274, 0.651278, 0.868007),
            None,
            False,
            False,
        ),
        ("GGX", 0.0, 1.5, (1.0, 1.0, 1.0), None, False, False),
        (
            "BECKMANN",
            0.0,
            1.0,
            (0.72, 0.94, 1.0),
            None,
            False,
            False,
        ),
        ("GGX", -0.03, 1.33, (1.4, 0.2, 2.2), None, False, False),
        (
            "BECKMANN",
            0.32,
            2.0,
            (1.0, 0.72, 0.48),
            None,
            False,
            False,
        ),
        ("GGX", 1.0, 0.67, (0.16, 0.81, 0.36), None, False, False),
        (
            "BECKMANN",
            0.45,
            1.5,
            (0.64, 0.09, 0.49),
            None,
            True,
            False,
        ),
        (
            "GGX",
            0.2,
            1.5,
            (0.7, 0.3, 0.1),
            (0.6, 0.0, 0.8),
            False,
            False,
        ),
        (
            "GGX",
            0.2,
            1.5,
            (0.7, 0.3, 0.1),
            (0.3, 0.0, 0.4),
            False,
            False,
        ),
        (
            "BECKMANN",
            0.2,
            1.5,
            (0.7, 0.3, 0.1),
            (0.0, 0.0, 0.0),
            False,
            False,
        ),
        (
            "BECKMANN",
            0.2,
            1.5,
            (0.7, 0.3, 0.1),
            (0.99, 0.0, 0.14),
            True,
            False,
        ),
        ("GGX", 0.25, 1.5, (-0.5, 0.6, 1.2), None, False, True),
        ("GGX", 0.25, 1.5, (-0.5, -0.2, -1.0), None, False, True),
        (
            "GGX",
            0.25,
            1.5,
            (1.0e-6, 1.0e-6, 1.0e-6),
            None,
            False,
            True,
        ),
        ("GGX", 0.25, 1.5, (6.0e-5, 0.0, 0.0), None, False, True),
    )
    materials = []
    backfacing: set[int] = set()
    for index, case in enumerate(cases):
        (
            distribution,
            roughness,
            ior,
            color,
            normal,
            backface,
            linked_color,
        ) = case
        material, tree, output = _material(
            f"Refraction BSDF Matrix {index:02d}"
        )
        refraction = tree.nodes.new("ShaderNodeBsdfRefraction")
        refraction.name = f"Raw Refraction BSDF {index:02d}"
        refraction.distribution = distribution
        if linked_color:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Linked Refraction Color {index:02d}",
                    color,
                ),
                _input(refraction, "Color"),
            )
        else:
            _input(refraction, "Color").default_value = (*color, 1.0)
        for label, socket, value in (
            ("Roughness", "Roughness", roughness),
            ("IOR", "IOR", ior),
        ):
            value_node = tree.nodes.new("ShaderNodeValue")
            value_node.name = f"Linked Refraction {label} {index:02d}"
            _output(value_node, "Value").default_value = value
            tree.links.new(
                _output(value_node, "Value"),
                _input(refraction, socket),
            )
        if normal is not None:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Linked Refraction Normal {index:02d}",
                    normal,
                ),
                _input(refraction, "Normal"),
            )
        tree.links.new(
            _output(refraction, "BSDF"),
            _input(output, "Surface"),
        )
        if backface:
            backfacing.add(index)
        materials.append(material)
    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Refraction BSDF Matrix",
        backfacing=backfacing,
        frame_bleed=0.02,
    )
    surface.visible_shadow = False
