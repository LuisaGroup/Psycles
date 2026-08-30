"""Cycles-compatible Blender particle-hair extraction primitives.

These helpers model Blender/Cycles host-side contracts. Geometry serialization
stays in ``export_psycles_scene.py`` so this module has no knowledge of the
Psycles scene format.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _cycles_srgb_channel_to_linear(value: np.float32) -> np.float32:
    """Match Cycles color_srgb_to_linear in float32."""

    if value < np.float32(0.04045):
        return (
            np.float32(0.0)
            if value < np.float32(0.0)
            else np.float32(value * np.float32(1.0 / 12.92))
        )
    base = np.float32(
        np.float32(value + np.float32(0.055))
        * np.float32(1.0 / 1.055)
    )
    return np.float32(np.power(base, np.float32(2.4)))


def cycles_particle_hair_color_value(
    srgb: np.ndarray[Any, np.dtype[np.float32]],
) -> tuple[float, float, float, float]:
    """Return Cycles' curve-domain linear-Rec.709 particle color.

    `BKE_particle_mcol_on_emitter` returns the emitter's byte-color sample in
    sRGB. Cycles converts it to a float4 on the host with
    `color_srgb_to_linear_v4`. Unlike a mesh BYTE_COLOR fetch, the resulting
    curve attribute is not tagged as byte data, so the kernel does not apply
    `rec709_to_rgb` a second time.
    """

    linear = tuple(
        _cycles_srgb_channel_to_linear(np.float32(component))
        for component in srgb
    )
    return tuple(float(component) for component in linear) + (1.0,)


def _node_tree_named_color_attributes(
    tree: Any,
    result: set[str],
    visited: set[int],
) -> None:
    """Collect static named color-attribute requests from a shader tree."""

    if tree is None:
        return
    identity = int(tree.as_pointer())
    if identity in visited:
        return
    visited.add(identity)
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeVertexColor":
            name = str(getattr(node, "layer_name", ""))
            if name:
                result.add(name)
        elif (
            node.bl_idname == "ShaderNodeAttribute"
            and str(getattr(node, "attribute_type", "GEOMETRY"))
            == "GEOMETRY"
        ):
            name = str(getattr(node, "attribute_name", ""))
            if name:
                result.add(name)
        nested = getattr(node, "node_tree", None)
        if nested is not None:
            _node_tree_named_color_attributes(nested, result, visited)


_NAMED_COLOR_ATTRIBUTE_CACHE: dict[int, frozenset[str]] = {}


def material_named_color_attributes(material: Any) -> frozenset[str]:
    """Return raw curve-color attributes demanded by one material.

    This is a geometry-residency query only. Shader nodes and closures remain
    untouched and are still evaluated by the Luisa SVM. Cycles performs the
    same demand filtering before projecting emitter byte colors to the curve
    domain.
    """

    if material is None or not material.use_nodes:
        return frozenset()
    identity = int(material.as_pointer())
    cached = _NAMED_COLOR_ATTRIBUTE_CACHE.get(identity)
    if cached is not None:
        return cached
    result: set[str] = set()
    _node_tree_named_color_attributes(material.node_tree, result, set())
    frozen = frozenset(result)
    _NAMED_COLOR_ATTRIBUTE_CACHE[identity] = frozen
    return frozen


def particle_hair_systems(evaluated: Any) -> list[tuple[Any, Any, Any]]:
    """Return the final-render legacy hair systems consumed by Cycles."""

    systems: list[tuple[Any, Any, Any]] = []
    for modifier in evaluated.modifiers:
        if modifier.type != "PARTICLE_SYSTEM" or not modifier.show_render:
            continue
        particle_system = getattr(modifier, "particle_system", None)
        if particle_system is None:
            continue
        settings = particle_system.settings
        if settings.type == "HAIR" and settings.render_type == "PATH":
            systems.append((modifier, particle_system, settings))
    return systems


def cycles_shape_radius(
    shape: np.float32,
    root: np.float32,
    tip: np.float32,
    intercept: np.float32,
) -> np.float32:
    """Evaluate Cycles' particle-hair radius curve in float32."""

    radius = np.float32(np.float32(1.0) - intercept)
    if shape < np.float32(0.0):
        radius = np.float32(
            np.power(radius, np.float32(np.float32(1.0) + shape))
        )
    elif shape > np.float32(0.0):
        radius = np.float32(
            np.power(
                radius,
                np.float32(
                    np.float32(1.0)
                    / np.float32(np.float32(1.0) - shape)
                ),
            )
        )
    return np.float32(
        np.float32(radius * np.float32(root - tip)) + tip
    )


def cycles_particle_hair_positions(
    particle_system: Any,
    evaluated: Any,
    particle_no: int,
    key_count: int,
) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
    """Recover Cycles' prefix-preserving BKE_particle_co_hair contract.

    A particle cache defines keys on the prefix ``[0, segments]``. The BKE
    function leaves its output untouched outside that prefix, and Cycles seeds
    the output with the preceding coordinate before every call. Blender's RNA
    wrapper instead zero-initializes the output-only parameter, so a shortened
    path appears to jump to world origin. Because the unwritten domain is a
    suffix, propagate the last written coordinate only across the trailing
    all-zero RNA results. Interior zero coordinates remain observable.
    """

    positions = [
        np.asarray(
            particle_system.co_hair(
                object=evaluated,
                particle_no=particle_no,
                step=step,
            ),
            dtype=np.float32,
        )
        for step in range(key_count)
    ]
    suffix_begin = len(positions)
    while suffix_begin > 0 and not np.any(
        positions[suffix_begin - 1] != np.float32(0.0)
    ):
        suffix_begin -= 1
    if 0 < suffix_begin < len(positions):
        positions[suffix_begin:] = [
            positions[suffix_begin - 1].copy()
            for _ in range(len(positions) - suffix_begin)
        ]
    return positions


def cycles_particle_hair_uv(
    modifier: Any,
    particle_system: Any,
    particle_no: int,
    uv_no: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Evaluate Cycles' BKE_particle_uv_on_emitter argument contract.

    Cycles advances its ParticleData pointer only while visiting parents. For
    child strands it keeps the pointer at the first parent and lets
    ``particle_no`` select the child in BKE. Blender's RNA wrapper requires a
    non-null pointer, so valid child systems use that same first-parent value.
    """

    parents = particle_system.particles
    if not parents:
        return np.zeros(2, dtype=np.float32)
    particle = (
        parents[particle_no]
        if particle_no < len(parents)
        else parents[0]
    )
    return np.asarray(
        particle_system.uv_on_emitter(
            modifier=modifier,
            particle=particle,
            particle_no=particle_no,
            uv_no=uv_no,
        ),
        dtype=np.float32,
    )


def cycles_particle_hair_color(
    modifier: Any,
    particle_system: Any,
    particle_no: int,
    color_no: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Evaluate Cycles' BKE_particle_mcol_on_emitter contract.

    Cycles uses the same parent-pointer progression for particle color and UV
    extraction. The RNA wrapper exposes the RGB payload written by BKE; the
    alpha component is the unchanged one initialized by Cycles and is added by
    the scene exporter after the sRGB-to-linear conversion.
    """

    parents = particle_system.particles
    if not parents:
        return np.zeros(3, dtype=np.float32)
    particle = (
        parents[particle_no]
        if particle_no < len(parents)
        else parents[0]
    )
    return np.asarray(
        particle_system.mcol_on_emitter(
            modifier=modifier,
            particle=particle,
            particle_no=particle_no,
            vcol_no=color_no,
        ),
        dtype=np.float32,
    )
