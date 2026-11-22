"""Shader graph and OCIO transform extraction helpers for Psycles scene sync."""

from __future__ import annotations

import sys
from typing import Any

import bpy
import numpy as np

from . import blender_scene_manifest as manifest

_CYCLES_DEFAULT_SHADER_COUNT = 5
CYCLES_BACKGROUND_SHADER_INDEX = 3


def cycles_shader_color_space() -> dict[str, Any]:
    """Reproduce Cycles ShaderManager::init_xyz_transforms from OCIO."""
    xyz_to_rec709 = np.asarray(
        (
            (3.2404542, -1.5371385, -0.4985314),
            (-0.9692660, 1.8760108, 0.0415560),
            (0.0556434, -0.2040259, 1.0572252),
        ),
        dtype=np.float32,
    )
    xyz_to_rgb = xyz_to_rec709.copy()
    blender_luma = [0.2126, 0.7152, 0.0722]
    try:
        import PyOpenColorIO as ocio

        config = ocio.GetCurrentConfig()
        blender_luma = list(config.getDefaultLumaCoefs())
        roles = dict(config.getRoles())
        if "aces_interchange" in roles:
            processor = config.getProcessor(
                "scene_linear", "aces_interchange"
            ).getDefaultCPUProcessor()
            transformed_basis = np.asarray(
                [
                    processor.applyRGB([1.0, 0.0, 0.0]),
                    processor.applyRGB([0.0, 1.0, 0.0]),
                    processor.applyRGB([0.0, 0.0, 1.0]),
                ],
                dtype=np.float32,
            )
            aces_ap0_to_xyz_d65 = np.asarray(
                (
                    (0.938280, -0.004451, 0.016628),
                    (0.337369, 0.729522, -0.066890),
                    (0.001174, -0.003711, 1.091595),
                ),
                dtype=np.float32,
            )
            xyz_to_rgb = (
                np.linalg.inv(transformed_basis.T)
                @ np.linalg.inv(aces_ap0_to_xyz_d65)
            ).astype(np.float32)
        elif "XYZ" in roles:
            processor = config.getProcessor(
                "scene_linear", "XYZ"
            ).getDefaultCPUProcessor()
            transformed_basis = np.asarray(
                [
                    processor.applyRGB([1.0, 0.0, 0.0]),
                    processor.applyRGB([0.0, 1.0, 0.0]),
                    processor.applyRGB([0.0, 0.0, 1.0]),
                ],
                dtype=np.float32,
            )
            xyz_to_rgb = np.linalg.inv(
                transformed_basis.T
            ).astype(np.float32)
    except Exception as error:
        print(
            "Warning: could not derive Cycles OCIO shader transforms; "
            f"using Rec.709 defaults: {error}",
            file=sys.stderr,
        )

    rec709_to_rgb = (
        xyz_to_rgb @ np.linalg.inv(xyz_to_rec709)
    ).astype(np.float32)
    return {
        "blender_luma": [float(value) for value in blender_luma],
        "xyz_to_rgb": [
            [float(component) for component in row]
            for row in xyz_to_rgb
        ],
        "rec709_to_rgb": [
            [float(component) for component in row]
            for row in rec709_to_rgb
        ],
    }


def _socket_links(tree: Any) -> dict[tuple[str, str], tuple[str, str]]:
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for link in tree.links:
        if (
            not link.is_valid
            or link.is_muted
            or link.from_socket.is_unavailable
            or link.to_socket.is_unavailable
        ):
            continue
        result[(link.to_node.name, link.to_socket.identifier)] = (
            link.from_node.name,
            link.from_socket.identifier,
        )
    return result


def _cycles_output(
    tree: Any,
    world: bool = False,
    light: bool = False,
) -> Any | None:
    expected = (
        "ShaderNodeOutputLight"
        if light
        else ("ShaderNodeOutputWorld" if world else "ShaderNodeOutputMaterial")
    )
    node = tree.get_output_node("CYCLES")
    if node is None or node.bl_idname != expected:
        return None
    return node


def _output_root(
    output: Any | None,
    links: dict[tuple[str, str], tuple[str, str]],
    socket: str,
) -> dict[str, str] | None:
    if output is None:
        return None
    source = links.get((output.name, socket))
    if source is None:
        return None
    return {"node": source[0], "socket": source[1]}


def serialize_shader_tree(
    tree: Any,
    world: bool = False,
    light: bool = False,
) -> dict[str, Any] | None:
    data = manifest._node_tree_manifest(tree)
    if data is not None:
        output = _cycles_output(tree, world, light)
        links = _socket_links(tree)
        data["surface_root"] = _output_root(output, links, "Surface")
        data["volume_root"] = None if light else _output_root(output, links, "Volume")
        data["displacement_root"] = (
            None if (world or light) else _output_root(output, links, "Displacement")
        )
    return data


def _node_tree_uses_pointiness(
    tree: Any,
    visited: set[int] | None = None,
) -> bool:
    if tree is None:
        return False
    if visited is None:
        visited = set()
    identity = int(tree.as_pointer())
    if identity in visited:
        return False
    visited.add(identity)
    for node in tree.nodes:
        if node.bl_idname == "ShaderNodeNewGeometry":
            output = node.outputs.get("Pointiness")
            if output is not None and output.is_linked:
                return True
        nested = getattr(node, "node_tree", None)
        if nested is not None and _node_tree_uses_pointiness(nested, visited):
            return True
    return False


def material_uses_pointiness(
    material: Any,
    cache: dict[int, bool] | None = None,
) -> bool:
    if material is None:
        return False
    identity = int(material.as_pointer())
    if cache is not None and identity in cache:
        return cache[identity]
    result = bool(
        material is not None
        and material.use_nodes
        and _node_tree_uses_pointiness(material.node_tree)
    )
    if cache is not None:
        cache[identity] = result
    return result


def original_id(value: Any) -> Any:
    original = getattr(value, "original", None)
    return original if original is not None else value


def id_key(value: Any) -> int:
    return int(original_id(value).as_pointer())


def cycles_shader_indices(
    depsgraph: Any,
) -> tuple[dict[int, int], dict[int, int]]:
    next_index = _CYCLES_DEFAULT_SHADER_COUNT
    light_indices: dict[int, int] = {}
    material_indices: dict[int, int] = {}
    for expected_type, indices in (
        (bpy.types.Light, light_indices),
        (bpy.types.Material, material_indices),
    ):
        for dependency_id in depsgraph.ids:
            original = original_id(dependency_id)
            if not isinstance(original, expected_type):
                continue
            key = id_key(original)
            if key in indices:
                continue
            indices[key] = next_index
            next_index += 1
    return light_indices, material_indices
