"""Helpers for extracting Geometry Nodes attributes and evaluated materials."""

from __future__ import annotations

import array
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np


def _walk_node_tree_attributes(
    node_tree: Any,
    required: set[str],
    visited: set[int],
) -> None:
    if node_tree is None:
        return
    ptr = int(node_tree.as_pointer())
    if ptr in visited:
        return
    visited.add(ptr)
    for node in node_tree.nodes:
        if node.type == "ATTRIBUTE":
            if (
                getattr(node, "attribute_type", "GEOMETRY") == "GEOMETRY"
                and getattr(node, "attribute_name", "")
            ):
                required.add(str(node.attribute_name))
        else:
            nested = getattr(node, "node_tree", None)
            if nested is not None:
                _walk_node_tree_attributes(nested, required, visited)


def collect_required_geometry_attributes(materials: Iterable[Any]) -> set[str]:
    """Return every GEOMETRY-type Attribute name referenced by scene shaders."""
    required: set[str] = set()
    visited: set[int] = set()
    for material in materials:
        if getattr(material, "use_nodes", False) and material.node_tree is not None:
            _walk_node_tree_attributes(material.node_tree, required, visited)
    return required


def resolved_material_slots(evaluated_obj: Any, mesh: Any) -> list[Any]:
    """Resolve effective material slots across object overrides and evaluated data."""
    obj_slots = getattr(evaluated_obj, "material_slots", ())
    mesh_mats = getattr(mesh, "materials", ()) if mesh is not None else ()
    count = max(len(obj_slots), len(mesh_mats))
    resolved: list[Any] = []
    for index in range(count):
        slot = obj_slots[index] if index < len(obj_slots) else None
        if slot is not None and getattr(slot, "link", "DATA") == "OBJECT":
            resolved.append(slot.material)
        elif index < len(mesh_mats):
            resolved.append(mesh_mats[index])
        elif slot is not None:
            resolved.append(slot.material)
        else:
            resolved.append(None)
    return resolved


_TYPE_INFO: dict[str, tuple[str, int, str, Any]] = {
    "FLOAT": ("FLOAT", 1, "value", np.float32),
    "BOOLEAN": ("FLOAT", 1, "value", np.bool_),
    "INT": ("FLOAT", 1, "value", np.int32),
    "INT8": ("FLOAT", 1, "value", np.int32),
    "FLOAT2": ("FLOAT2", 2, "vector", np.float32),
    "FLOAT_VECTOR": ("FLOAT3", 3, "vector", np.float32),
}


def extract_generic_mesh_attributes(
    mesh: Any,
    required_names: set[str],
    excluded_keys: set[tuple[str, str]],
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
    color_converter: Callable[[Any, int, Any], tuple[float, float, float, float]],
    write_array_fn: Callable[[Any, Any], dict[str, int]],
    stream: Any,
) -> list[dict[str, Any]]:
    """Extract shader-referenced generic mesh attributes in Cycles domain order."""
    attributes = getattr(mesh, "attributes", None)
    if attributes is None or not required_names:
        return []

    point_count = len(mesh.vertices)
    face_count = len(mesh.polygons)
    corner_count = len(mesh.loops)
    tri_corners: np.ndarray | None = None
    tri_faces: np.ndarray | None = None
    domain_expected = {
        "POINT": point_count,
        "CORNER": corner_count,
        "FACE": face_count,
    }

    extracted: list[dict[str, Any]] = []
    for attr in attributes:
        name = str(attr.name)
        if name.startswith(".") or name not in required_names:
            continue
        domain = str(attr.domain)
        if domain not in domain_expected or (name, domain) in excluded_keys:
            continue
        data_type = str(attr.data_type)
        src_len = domain_expected[domain]
        if len(attr.data) < src_len:
            continue

        if data_type in _TYPE_INFO:
            out_type, stride, prop, np_dtype = _TYPE_INFO[data_type]
            raw = np.empty(src_len * stride, dtype=np_dtype)
            if src_len > 0:
                attr.data.foreach_get(prop, raw)
            flat = raw.astype(np.float32, copy=False).reshape(src_len, stride)
            if domain == "POINT":
                ordered = flat
            elif domain == "CORNER":
                if tri_corners is None:
                    tri_corners = np.fromiter(
                        (li for tri in mesh.loop_triangles for li in tri.loops),
                        dtype=np.intp,
                        count=len(mesh.loop_triangles) * 3,
                    )
                ordered = flat[tri_corners] if len(tri_corners) > 0 else flat[:0]
            else:
                if tri_faces is None:
                    tri_faces = np.fromiter(
                        (tri.polygon_index for tri in mesh.loop_triangles),
                        dtype=np.intp,
                        count=len(mesh.loop_triangles),
                    )
                ordered = flat[tri_faces] if len(tri_faces) > 0 else flat[:0]
            section = write_array_fn(
                stream, np.ascontiguousarray(ordered, dtype="<f4")
            )
            extracted.append(
                {
                    "name": name,
                    "domain": domain,
                    "type": out_type,
                    "values": section,
                }
            )
        elif data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
            out_vals = array.array("f")
            if domain == "POINT":
                indices = range(point_count)
            elif domain == "CORNER":
                indices = (li for tri in mesh.loop_triangles for li in tri.loops)
            else:
                indices = (tri.polygon_index for tri in mesh.loop_triangles)
            for idx in indices:
                out_vals.extend(color_converter(attr, idx, rec709_to_rgb))
            extracted.append(
                {
                    "name": name,
                    "domain": domain,
                    "type": "RGBA",
                    "values": write_array_fn(stream, out_vals),
                }
            )
    return extracted
