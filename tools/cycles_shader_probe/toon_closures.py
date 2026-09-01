"""Cycles 5.2 standalone Toon BSDF oracle scenes."""

from __future__ import annotations

from typing import Any

from .support import _input, _material, _material_matrix, _output


def _standalone_toon_svm_oracle(
    scene: Any,
    *,
    name: str,
    component: str,
    color: tuple[float, float, float],
    size: float,
    smooth: float,
) -> None:
    """Build one literal standalone Cycles Toon word-image oracle."""
    material, tree, output = _material(name)
    closure = tree.nodes.new("ShaderNodeBsdfToon")
    closure.name = name
    closure.component = component
    _input(closure, "Color").default_value = (*color, 1.0)
    _input(closure, "Size").default_value = size
    _input(closure, "Smooth").default_value = smooth
    tree.links.new(_output(closure, "BSDF"), _input(output, "Surface"))
    _material_matrix(scene, [material], columns=1, rows=1, name=name)


def _standalone_toon_diffuse_svm_oracle(scene: Any) -> None:
    """Isolate Cycles 5.2's standalone Diffuse Toon transition."""
    _standalone_toon_svm_oracle(
        scene,
        name="Standalone Diffuse Toon SVM Oracle",
        component="DIFFUSE",
        color=(0.31, 0.73, 0.19),
        size=0.37,
        smooth=0.21,
    )


def _standalone_toon_glossy_svm_oracle(scene: Any) -> None:
    """Isolate Cycles 5.2's standalone Glossy Toon transition."""
    _standalone_toon_svm_oracle(
        scene,
        name="Standalone Glossy Toon SVM Oracle",
        component="GLOSSY",
        color=(0.84, 0.22, 0.56),
        size=0.63,
        smooth=0.14,
    )
