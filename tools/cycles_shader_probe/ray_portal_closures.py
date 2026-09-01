"""Cycles 5.2 standalone Ray Portal BSDF oracle scenes."""

from __future__ import annotations

from typing import Any

from .support import _input, _material, _material_matrix, _output


def _ray_portal(
    name: str,
    color: tuple[float, float, float],
) -> tuple[Any, Any, Any]:
    material, tree, output = _material(name)
    closure = tree.nodes.new("ShaderNodeBsdfRayPortal")
    closure.name = name
    _input(closure, "Color").default_value = (*color, 1.0)
    tree.links.new(_output(closure, "BSDF"), _input(output, "Surface"))
    return material, tree, closure


def _standalone_ray_portal_default_svm_oracle(scene: Any) -> None:
    """Keep Cycles' implicit Geometry.Position and zero-Direction fallback."""
    name = "Standalone Default Ray Portal SVM Oracle"
    material, _tree, _closure = _ray_portal(name, (0.26, 0.71, 0.43))
    _material_matrix(scene, [material], columns=1, rows=1, name=name)


def _standalone_ray_portal_authored_svm_oracle(scene: Any) -> None:
    """Exercise an explicit Position stack value and non-unit Direction."""
    name = "Standalone Authored Ray Portal SVM Oracle"
    material, tree, closure = _ray_portal(name, (0.83, 0.17, 0.52))
    position = tree.nodes.new("ShaderNodeCombineXYZ")
    position.name = "Authored Ray Portal Position"
    _input(position, "X").default_value = 1.25
    _input(position, "Y").default_value = -0.75
    _input(position, "Z").default_value = 2.5
    tree.links.new(_output(position, "Vector"), _input(closure, "Position"))
    _input(closure, "Direction").default_value = (0.3, -0.4, 1.2)
    _material_matrix(scene, [material], columns=1, rows=1, name=name)
