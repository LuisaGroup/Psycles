"""Cycles 5.2 thin-film surface closure probe scenes."""

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


def _linked_float(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return _output(node, "Value")


def _thin_film_surface(scene: Any) -> None:
    """Exercise the raw Cycles 5.2 closures that accept thin film.

    A single zero-angle Sun makes direct closure evaluation deterministic:
    neither renderer has to choose between lights or sample a finite emitter.
    All film inputs, including the zero-film controls, are linked so the
    exported graph preserves the authored runtime dependency instead of
    relying on Cycles' constant-input pruning.
    """
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    _bsdf_matrix_sun(scene, transmission=False)

    # kind, film thickness [nm], film IOR, roughness, material IOR,
    # base color, metallic, transmission, thin wall, explicit normal
    cases = (
        ("principled", 0.0, 1.33, 0.24, 1.50,
         (0.0, 0.0, 0.0), 0.0, 0.0, False, None),
        ("principled", 100.0, 1.33, 0.24, 1.50,
         (0.0, 0.0, 0.0), 0.0, 0.0, False, None),
        ("principled", 300.0, 1.33, 0.24, 1.50,
         (0.0, 0.0, 0.0), 0.0, 0.0, False, None),
        ("principled", 600.0, 1.70, 0.24, 1.50,
         (0.0, 0.0, 0.0), 0.0, 0.0, False, (0.6, 0.0, 0.8)),
        ("principled", 0.0, 1.33, 0.28, 1.50,
         (0.72, 0.14, 0.035), 1.0, 0.0, False, None),
        ("principled", 150.0, 1.33, 0.28, 1.50,
         (0.72, 0.14, 0.035), 1.0, 0.0, False, None),
        ("principled", 450.0, 1.50, 0.28, 1.50,
         (0.16, 0.62, 0.82), 1.0, 0.0, False, None),
        ("principled", 900.0, 2.00, 0.28, 1.50,
         (0.58, 0.48, 0.08), 1.0, 0.0, False, (0.6, 0.0, 0.8)),
        ("principled", 0.0, 1.33, 0.22, 1.45,
         (0.36, 0.64, 1.0), 0.0, 1.0, False, None),
        ("principled", 275.0, 1.33, 0.22, 1.45,
         (0.36, 0.64, 1.0), 0.0, 1.0, False, None),
        ("principled", 0.0, 1.33, 0.22, 1.45,
         (0.36, 0.64, 1.0), 0.0, 1.0, True, None),
        ("principled", 275.0, 1.33, 0.22, 1.45,
         (0.36, 0.64, 1.0), 0.0, 1.0, True, None),
        ("glass", 0.0, 1.33, 0.20, 1.50,
         (0.85, 0.85, 0.85), 0.0, 1.0, False, None),
        ("glass", 180.0, 1.15, 0.20, 1.50,
         (0.85, 0.85, 0.85), 0.0, 1.0, False, None),
        ("glass", 350.0, 1.33, 0.20, 1.50,
         (0.85, 0.85, 0.85), 0.0, 1.0, False, None),
        ("glass", 650.0, 1.70, 0.20, 1.50,
         (0.85, 0.85, 0.85), 0.0, 1.0, False,
         (0.6, 0.0, 0.8)),
    )

    materials = []
    for index, case in enumerate(cases):
        (
            kind,
            film_thickness,
            film_ior,
            roughness,
            material_ior,
            base_color,
            metallic,
            transmission,
            thin_wall,
            normal,
        ) = case
        material, tree, output = _material(f"Thin Film {index:02d}")
        if kind == "principled":
            closure = tree.nodes.new("ShaderNodeBsdfPrincipled")
            closure.name = f"Raw Principled Thin Film {index:02d}"
            closure.distribution = "GGX"
            _input(closure, "Base Color").default_value = (
                *base_color,
                1.0,
            )
            _input(closure, "Metallic").default_value = metallic
            _input(closure, "Roughness").default_value = roughness
            _input(closure, "Diffuse Roughness").default_value = 0.0
            _input(closure, "IOR").default_value = material_ior
            _input(closure, "Specular IOR Level").default_value = 0.5
            _input(closure, "Specular Tint").default_value = (
                1.0,
                1.0,
                1.0,
                1.0,
            )
            _input(closure, "Transmission Weight").default_value = (
                transmission
            )
            _input(closure, "Thin Wall").default_value = thin_wall
            _input(closure, "Sheen Weight").default_value = 0.0
            _input(closure, "Coat Weight").default_value = 0.0
            _input(closure, "Alpha").default_value = 1.0
        elif kind == "glass":
            closure = tree.nodes.new("ShaderNodeBsdfGlass")
            closure.name = f"Raw Glass Thin Film {index:02d}"
            closure.distribution = "GGX"
            _input(closure, "Color").default_value = (*base_color, 1.0)
            _input(closure, "Roughness").default_value = roughness
            _input(closure, "IOR").default_value = material_ior
        else:
            raise AssertionError(f"unknown thin-film closure kind {kind!r}")

        tree.links.new(
            _linked_float(
                tree,
                f"Linked Thin Film Thickness {index:02d}",
                film_thickness,
            ),
            _input(closure, "Thin Film Thickness"),
        )
        tree.links.new(
            _linked_float(
                tree,
                f"Linked Thin Film IOR {index:02d}",
                film_ior,
            ),
            _input(closure, "Thin Film IOR"),
        )
        if normal is not None:
            tree.links.new(
                _linked_vector(
                    tree,
                    f"Linked Thin Film Normal {index:02d}",
                    normal,
                ),
                _input(closure, "Normal"),
            )
        tree.links.new(
            _output(closure, "BSDF"),
            _input(output, "Surface"),
        )
        materials.append(material)

    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Cycles 5.2 Thin Film Surface Matrix",
        frame_bleed=0.02,
    )
    surface.visible_shadow = False
