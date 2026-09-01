"""Cycles 5.2 standalone Sheen BSDF probe scenes."""

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
class _SheenCase:
    distribution: str
    color: tuple[float, float, float]
    roughness: float
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


def _linked_float(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return _output(node, "Value")


def _standalone_sheen_svm_oracle(
    scene: Any,
    *,
    name: str,
    distribution: str,
    color: tuple[float, float, float],
    roughness: float,
) -> None:
    """Build one literal standalone Cycles Sheen word-image oracle."""
    material, tree, output = _material(name)
    closure = tree.nodes.new("ShaderNodeBsdfSheen")
    closure.name = name
    closure.distribution = distribution
    _input(closure, "Color").default_value = (*color, 1.0)
    _input(closure, "Roughness").default_value = roughness
    tree.links.new(_output(closure, "BSDF"), _input(output, "Surface"))
    _material_matrix(scene, [material], columns=1, rows=1, name=name)


def _standalone_sheen_microfiber_svm_oracle(scene: Any) -> None:
    """Isolate Cycles 5.2's standalone Microfiber Sheen transition."""
    _standalone_sheen_svm_oracle(
        scene,
        name="Standalone Microfiber Sheen SVM Oracle",
        distribution="MICROFIBER",
        color=(0.38, 0.77, 0.16),
        roughness=0.43,
    )


def _standalone_sheen_ashikhmin_svm_oracle(scene: Any) -> None:
    """Isolate Cycles 5.2's standalone Ashikhmin Velvet transition."""
    _standalone_sheen_svm_oracle(
        scene,
        name="Standalone Ashikhmin Sheen SVM Oracle",
        distribution="ASHIKHMIN",
        color=(0.82, 0.19, 0.57),
        roughness=0.24,
    )


def _sheen_bsdf_matrix(scene: Any) -> None:
    """Exercise both unmodified Cycles 5.2 standalone Sheen closures.

    The 16 equal-area cells form a finite product cover of the two static
    distributions, their distinct lower roughness bounds, the shared input
    saturation, lower-only closure-color clamp, and authored normal. Numeric
    sockets are linked through typed Blender nodes so neither Blender nor the
    exporter can replace the raw closure with a precomputed result. The
    internal Weight socket is deliberately untouched: Cycles owns that graph
    compiler input and it is not part of the Blender material contract.
    """
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    _bsdf_matrix_sun(scene, transmission=False)
    # An oblique zero-angle Sun keeps direct closure evaluation deterministic
    # while distinguishing the LTC and Ashikhmin angular responses.
    bpy.data.objects["BSDF Matrix Sun"].rotation_euler[1] = 0.92

    cases = (
        # Microfiber: SVM saturates roughness and setup then applies the
        # stricter 1e-3 LTC coordinate bound. Include both sides of those
        # boundaries, ordinary table coordinates, two tilted normals, and an
        # over-one color that must not be clamped on its upper side.
        _SheenCase("MICROFIBER", (0.82, 0.16, 0.04), -0.25),
        _SheenCase("MICROFIBER", (0.10, 0.62, 0.92), 0.0),
        _SheenCase("MICROFIBER", (0.92, 0.72, 0.12), 0.001),
        _SheenCase("MICROFIBER", (0.42, 0.10, 0.76), 0.18),
        _SheenCase("MICROFIBER", (0.08, 0.86, 0.38), 0.50),
        _SheenCase(
            "MICROFIBER", (1.28, 0.36, 0.08), 0.82, (0.24, 0.0, 0.97)
        ),
        _SheenCase(
            "MICROFIBER", (0.18, 0.48, 0.96), 1.0, (-0.18, 0.23, 0.96)
        ),
        _SheenCase("MICROFIBER", (-0.30, 0.58, 1.20), 1.25),
        # Ashikhmin: SVM first saturates sigma and setup applies max(0.01).
        # Keep the same semantic input axes while changing values so a stale
        # Microfiber dispatch or a shared roughness transform cannot pass by
        # accidental cell equality.
        _SheenCase("ASHIKHMIN", (0.76, 0.08, 0.18), -0.25),
        _SheenCase("ASHIKHMIN", (0.10, 0.70, 0.84), 0.0),
        _SheenCase("ASHIKHMIN", (0.88, 0.60, 0.10), 0.01),
        _SheenCase("ASHIKHMIN", (0.52, 0.12, 0.78), 0.12),
        _SheenCase("ASHIKHMIN", (0.06, 0.80, 0.34), 0.35),
        _SheenCase(
            "ASHIKHMIN", (1.18, 0.30, 0.06), 0.65, (0.24, 0.0, 0.97)
        ),
        _SheenCase(
            "ASHIKHMIN", (0.14, 0.42, 0.92), 1.0, (-0.18, 0.23, 0.96)
        ),
        _SheenCase("ASHIKHMIN", (-0.25, 0.52, 1.16), 1.30),
    )

    materials = []
    for index, case in enumerate(cases):
        material, tree, output = _material(
            f"Sheen BSDF Matrix {index:02d}"
        )
        closure = tree.nodes.new("ShaderNodeBsdfSheen")
        closure.name = f"Raw Sheen BSDF {index:02d}"
        closure.distribution = case.distribution
        tree.links.new(
            _linked_vector(
                tree, f"Linked Sheen Color {index:02d}", case.color
            ),
            _input(closure, "Color"),
        )
        tree.links.new(
            _linked_float(
                tree, f"Linked Sheen Roughness {index:02d}", case.roughness
            ),
            _input(closure, "Roughness"),
        )
        tree.links.new(
            _linked_vector(
                tree, f"Linked Sheen Normal {index:02d}", case.normal
            ),
            _input(closure, "Normal"),
        )
        tree.links.new(_output(closure, "BSDF"), _input(output, "Surface"))
        materials.append(material)

    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Cycles 5.2 Sheen BSDF Matrix",
        frame_bleed=0.02,
    )
    surface.visible_shadow = False
