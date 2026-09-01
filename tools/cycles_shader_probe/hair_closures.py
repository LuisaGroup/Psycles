"""Cycles 5.2 legacy Hair BSDF probe scenes."""

from __future__ import annotations

import math
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
class _HairCase:
    component: str
    color: tuple[float, float, float]
    offset: float
    roughness_u: float
    roughness_v: float
    tangent: tuple[float, float, float] | None = None


def _linked_float(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return _output(node, "Value")


def _standalone_hair_reflection_svm_oracle(scene: Any) -> None:
    """Keep Blender 5.2's legacy Hair Reflection socket defaults intact."""
    name = "Standalone Hair Reflection SVM Oracle"
    material, tree, output = _material(name)
    closure = tree.nodes.new("ShaderNodeBsdfHair")
    closure.name = name
    closure.component = "Reflection"
    tree.links.new(_output(closure, "BSDF"), _input(output, "Surface"))
    _material_matrix(scene, [material], columns=1, rows=1, name=name)


def _standalone_hair_transmission_svm_oracle(scene: Any) -> None:
    """Exercise both roughness clamps, signed offset, and linked Tangent."""
    name = "Standalone Hair Transmission SVM Oracle"
    material, tree, output = _material(name)
    closure = tree.nodes.new("ShaderNodeBsdfHair")
    closure.name = name
    closure.component = "Transmission"
    _input(closure, "Color").default_value = (0.83, 0.17, 0.52, 1.0)
    _input(closure, "Offset").default_value = 0.27
    _input(closure, "RoughnessU").default_value = 0.0002
    _input(closure, "RoughnessV").default_value = 1.4
    tree.links.new(
        _linked_vector(tree, "Authored Hair Tangent", (0.3, 0.4, 0.0)),
        _input(closure, "Tangent"),
    )
    tree.links.new(_output(closure, "BSDF"), _input(output, "Surface"))
    _material_matrix(scene, [material], columns=1, rows=1, name=name)


def _hair_bsdf_matrix(scene: Any) -> None:
    """Exercise both unmodified Cycles 5.2 legacy Hair closures.

    The two rows per component cover lower/upper roughness clamps, signed
    offset, unlinked triangle tangent derivation, and two linked tangents.
    Every authored numeric input is routed through a typed Blender node. The
    internal Weight socket remains untouched because it belongs to Cycles'
    graph compiler rather than the Blender material ABI.
    """
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.use_light_tree = False
    _bsdf_matrix_sun(scene, transmission=False)

    # Hair Reflection and Hair Transmission occupy opposite hemispheres. Two
    # zero-angle Suns provide variance-free direct evaluation of both laws in
    # one image; a closure rejects the Sun on the other side by construction.
    front = bpy.data.objects["BSDF Matrix Sun"]
    front.rotation_euler = (0.0, 0.72, 0.18)
    back_data = bpy.data.lights.new("BSDF Matrix Back Sun", type="SUN")
    back_data.color = (0.36, 0.72, 1.0)
    back_data.energy = 1.7
    back_data.normalize = True
    back_data.angle = 0.0
    back_data.use_shadow = True
    back = bpy.data.objects.new(back_data.name, back_data)
    back.rotation_euler = (math.pi, -0.72, -0.18)
    scene.collection.objects.link(back)

    semantic_cases = (
        # Unlinked Tangent uses triangle dPdv and forces offset to zero.
        ((0.82, 0.16, 0.04), 0.31, 0.10, 1.00, None),
        # Both physical roughness axes clamp independently to 1e-3.
        ((0.10, 0.62, 0.92), -0.22, 0.0001, 0.0002, None),
        # Both axes clamp independently to one; upper Color stays unbounded.
        ((1.24, 0.36, 0.08), 0.17, 1.40, 1.70, None),
        ((0.92, 0.72, 0.12), 0.28, 0.08, 0.31, (1.0, 0.0, 0.0)),
        ((0.42, 0.10, 0.76), -0.35, 0.22, 0.54, (0.6, 0.8, 0.0)),
        ((0.08, 0.86, 0.38), 0.0, 0.65, 0.12, (-0.8, 0.6, 0.0)),
        # Closure allocation clamps only the lower side of Color.
        ((-0.30, 0.58, 1.20), 0.42, 0.34, 0.77, (0.8, 0.6, 0.0)),
        ((0.18, 0.48, 0.96), -0.12, 0.47, 0.19, (-0.6, 0.8, 0.0)),
    )
    cases = tuple(
        _HairCase(component, color, offset, roughness_u, roughness_v, tangent)
        for component in ("Reflection", "Transmission")
        for color, offset, roughness_u, roughness_v, tangent in semantic_cases
    )

    materials = []
    for index, case in enumerate(cases):
        material, tree, output = _material(
            f"Hair BSDF Matrix {index:02d}"
        )
        closure = tree.nodes.new("ShaderNodeBsdfHair")
        closure.name = f"Raw Hair BSDF {index:02d}"
        closure.component = case.component
        tree.links.new(
            _linked_vector(
                tree, f"Linked Hair Color {index:02d}", case.color
            ),
            _input(closure, "Color"),
        )
        for socket_name, value in (
            ("Offset", case.offset),
            ("RoughnessU", case.roughness_u),
            ("RoughnessV", case.roughness_v),
        ):
            tree.links.new(
                _linked_float(
                    tree, f"Linked Hair {socket_name} {index:02d}", value
                ),
                _input(closure, socket_name),
            )
        if case.tangent is not None:
            tree.links.new(
                _linked_vector(
                    tree, f"Linked Hair Tangent {index:02d}", case.tangent
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
        name="Cycles 5.2 Hair BSDF Matrix",
        frame_bleed=0.02,
    )
    surface.visible_shadow = False
