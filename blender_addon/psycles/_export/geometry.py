"""Evaluated mesh, hair, instance, light, and camera extraction for Psycles."""

from __future__ import annotations

import array
import copy
import io
import sys
from typing import Any

import bpy
import numpy as np

from . import cycles_hash
from .blender_cycles_object import (
    light_group as _cycles_light_group,
    object_ao_distance as _cycles_object_ao_distance,
    object_properties as _cycles_object_properties,
    object_random_id as _cycles_object_random_id,
    particle_parent_index as _cycles_particle_index,
    shadow_catcher as _cycles_shadow_catcher,
)
from .blender_particle_hair import (
    cycles_particle_hair_color as _cycles_particle_hair_color,
    cycles_particle_hair_color_value as _cycles_particle_hair_color_value,
    cycles_particle_hair_positions as _cycles_particle_hair_positions,
    cycles_particle_hair_uv as _cycles_particle_hair_uv,
    cycles_shape_radius as _cycles_shape_radius,
    material_named_color_attributes as _material_named_color_attributes,
    particle_hair_systems as _particle_hair_systems,
)
from .shader_nodes import (
    material_uses_pointiness,
    original_id,
    serialize_shader_tree,
)


def _cycles_uint_to_float(value: int) -> float:
    return float(cycles_hash.u32(value)) / float(cycles_hash.UINT32_MASK)


def column_major(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


def write_array(stream: Any, values: array.array[Any]) -> dict[str, int]:
    if sys.byteorder != "little":
        values.byteswap()
    offset = stream.tell()
    payload = memoryview(values).cast("B")
    written = stream.write(payload)
    if written != len(payload):
        raise OSError(
            f"short geometry write: expected {len(payload)} bytes, wrote {written}"
        )
    return {"offset": offset, "bytes": written}


def write_ndarray(stream: Any, arr: np.ndarray[Any, Any]) -> dict[str, int]:
    contiguous = np.ascontiguousarray(arr)
    if sys.byteorder != "little" and contiguous.dtype.itemsize > 1:
        contiguous = contiguous.astype(
            contiguous.dtype.newbyteorder("<"), copy=False
        )
    offset = stream.tell()
    payload = memoryview(contiguous).cast("B")
    written = stream.write(payload)
    if written != len(payload):
        raise OSError(
            f"short geometry write: expected {len(payload)} bytes, wrote {written}"
        )
    return {"offset": offset, "bytes": written}


def _rebase_section(section: dict[str, int] | None, delta: int) -> None:
    if section is not None:
        section["offset"] += delta


def rebase_geometry_sections(mesh_dict: dict[str, Any], base_offset: int) -> dict[str, Any]:
    if base_offset != 0:
        for key in (
            "positions",
            "normals",
            "uv",
            "uv_tangents",
            "generated",
            "indices",
            "triangle_material_slots",
            "triangle_smooth",
            "triangle_random_per_island",
        ):
            _rebase_section(mesh_dict.get(key), base_offset)
        psrc = mesh_dict.get("pointiness_source")
        if psrc is not None:
            _rebase_section(psrc.get("point_normals"), base_offset)
            _rebase_section(psrc.get("edges"), base_offset)
        iedges = mesh_dict.get("island_edges")
        if iedges is not None:
            _rebase_section(iedges.get("edges"), base_offset)
        for layer in mesh_dict.get("uv_layers", ()):
            _rebase_section(layer.get("values"), base_offset)
            _rebase_section(layer.get("tangents"), base_offset)
        for attr in mesh_dict.get("color_attributes", ()):
            _rebase_section(attr.get("values"), base_offset)
            _rebase_section(attr.get("byte_values"), base_offset)
    return mesh_dict


def geometry_cache_key(
    obj: Any,
    scene: Any,
    depsgraph_mode: str = "RENDER",
) -> tuple[Any, ...]:
    settings_mode = "PREVIEW" if depsgraph_mode == "VIEWPORT" else "RENDER"
    if (
        obj.type == "MESH"
        and obj.data is not None
        and not obj.is_modified(scene, settings_mode)
    ):
        material_slots = tuple(
            slot.material.as_pointer() if slot.material is not None else 0
            for slot in obj.material_slots
        )
        return ("mesh", obj.data.as_pointer(), material_slots)
    return ("object", obj.as_pointer())


def _cycles_uses_adaptive_subdivision(
    obj: Any,
    is_viewport: bool = False,
) -> bool:
    modifiers = list(obj.modifiers)
    if not modifiers:
        return False
    modifier = modifiers[-1]
    enabled = modifier.show_viewport if is_viewport else modifier.show_render
    return bool(
        modifier.type == "SUBSURF"
        and enabled
        and getattr(modifier, "use_adaptive_subdivision", False)
    )


def _extract_mesh_geometry_to_local_stream(
    obj: Any,
    depsgraph: Any,
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
    *,
    is_viewport: bool,
    pointiness_cache: dict[int, bool] | None,
) -> tuple[dict[str, Any], bytes] | None:
    evaluated = obj.evaluated_get(depsgraph)
    try:
        mesh = evaluated.to_mesh(
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
    except RuntimeError:
        return None
    if mesh is None:
        return None
    local_stream = io.BytesIO()
    try:
        mesh.calc_loop_triangles()
        n_tris = len(mesh.loop_triangles)
        if n_tris == 0:
            return None
        n_verts = len(mesh.vertices)
        n_edges = len(mesh.edges)
        n_loops = len(mesh.loops)
        n_corners = n_tris * 3

        needs_pointiness = any(
            material_uses_pointiness(slot.material, pointiness_cache)
            for slot in obj.material_slots
        )

        positions = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", positions)

        texspace_location = mesh.texspace_location
        texspace_size = mesh.texspace_size
        generated_scale = [
            0.5 / float(size) if float(size) != 0.0 else 0.0
            for size in texspace_size
        ]
        generated_location = [
            float(texspace_location[axis]) * generated_scale[axis] - 0.5
            for axis in range(3)
        ]
        generated_transform = [
            generated_scale[0], 0.0, 0.0, 0.0,
            0.0, generated_scale[1], 0.0, 0.0,
            0.0, 0.0, generated_scale[2], 0.0,
            -generated_location[0], -generated_location[1], -generated_location[2], 1.0,
        ]
        scale64 = np.asarray(generated_scale, dtype=np.float64)
        loc64 = np.asarray(generated_location, dtype=np.float64)
        generated = (
            positions.reshape(n_verts, 3).astype(np.float64) * scale64 - loc64
        ).astype(np.float32)

        blender_normal_domain = str(getattr(mesh, "normals_domain", "CORNER"))
        use_corner_normals = blender_normal_domain == "CORNER"
        normal_domain = "CORNER" if use_corner_normals else "POINT"

        point_normals: np.ndarray[Any, np.dtype[np.float32]] | None = None
        if needs_pointiness or normal_domain == "POINT":
            point_normals = np.empty(n_verts * 3, dtype=np.float32)
            mesh.vertices.foreach_get("normal", point_normals)

        edges_i32 = np.empty(n_edges * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges_i32)
        edges_u32 = np.ascontiguousarray(edges_i32.view(np.uint32))

        tri_loops_i32 = np.empty(n_corners, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops_i32)
        tri_loops = tri_loops_i32

        tri_mats_i32 = np.empty(n_tris, dtype=np.int32)
        mesh.loop_triangles.foreach_get("material_index", tri_mats_i32)
        materials = np.maximum(tri_mats_i32, 0).astype(np.uint32, copy=False)

        loop_vidx_i32 = np.empty(n_loops, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_vidx_i32)
        indices = np.ascontiguousarray(
            loop_vidx_i32[tri_loops], dtype=np.uint32
        )

        if normal_domain == "CORNER":
            loop_normals = np.empty(n_loops * 3, dtype=np.float32)
            mesh.loops.foreach_get("normal", loop_normals)
            normals = np.ascontiguousarray(
                loop_normals.reshape(n_loops, 3)[tri_loops], dtype=np.float32
            )
            smooth = np.ones(n_tris, dtype=np.uint32)
        else:
            assert point_normals is not None
            normals = point_normals
            sharp_face_attribute = mesh.attributes.get("sharp_face")
            if sharp_face_attribute is not None:
                n_sharp = len(sharp_face_attribute.data)
                sharp = np.empty(n_sharp, dtype=bool)
                sharp_face_attribute.data.foreach_get("value", sharp)
                tri_polys = np.empty(n_tris, dtype=np.int32)
                mesh.loop_triangles.foreach_get("polygon_index", tri_polys)
                smooth = (~sharp[tri_polys]).astype(np.uint32)
            else:
                smooth_val = 1 if blender_normal_domain != "FACE" else 0
                smooth = np.full(n_tris, smooth_val, dtype=np.uint32)

        positions_sec = write_ndarray(local_stream, positions)
        normals_sec = write_ndarray(local_stream, normals)
        generated_sec = write_ndarray(local_stream, generated)
        edges_sec = write_ndarray(local_stream, edges_u32)
        indices_sec = write_ndarray(local_stream, indices)
        materials_sec = write_ndarray(local_stream, materials)
        smooth_sec = write_ndarray(local_stream, smooth)

        pointiness_source = None
        if needs_pointiness:
            assert point_normals is not None
            pointiness_source = {
                "point_normals": write_ndarray(local_stream, point_normals),
                "edge_count": n_edges,
                "edges": dict(edges_sec),
            }

        uv_layer_names = [layer.name for layer in mesh.uv_layers]
        default_uv_name = (
            mesh.uv_layers.active_render.name
            if mesh.uv_layers.active_render is not None
            else None
        )
        uv_sections: dict[str, dict[str, int]] = {}
        for uv_layer_name in uv_layer_names:
            uv_layer = mesh.uv_layers.get(uv_layer_name)
            if uv_layer is None:
                continue
            raw_uv = np.empty(n_loops * 2, dtype=np.float32)
            uv_layer.data.foreach_get("uv", raw_uv)
            corner_uv = np.ascontiguousarray(
                raw_uv.reshape(n_loops, 2)[tri_loops], dtype=np.float32
            )
            uv_sections[uv_layer_name] = write_ndarray(local_stream, corner_uv)

        default_uv_available = default_uv_name in uv_sections
        if default_uv_available and default_uv_name is not None:
            uv_sec = dict(uv_sections[default_uv_name])
        else:
            uv_sec = write_ndarray(
                local_stream, np.zeros((n_corners, 2), dtype=np.float32)
            )

        color_layers = [
            attribute
            for attribute in mesh.color_attributes
            if attribute.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}
            and attribute.domain in {"CORNER", "POINT"}
        ]
        mat_rec709_T = np.asarray(rec709_to_rgb, dtype=np.float64).T
        color_attr_entries: list[dict[str, Any]] = []
        for attribute in color_layers:
            if attribute.domain == "POINT":
                raw_c = np.empty(n_verts * 4, dtype=np.float32)
                attribute.data.foreach_get("color", raw_c)
                values_sec = write_ndarray(local_stream, raw_c)
                byte_sec = None
            else:
                raw_c = np.empty(n_loops * 4, dtype=np.float32)
                attribute.data.foreach_get("color", raw_c)
                c_tri = np.ascontiguousarray(
                    raw_c.reshape(n_loops, 4)[tri_loops], dtype=np.float32
                )
                if attribute.data_type == "BYTE_COLOR":
                    rgb = (
                        c_tri[:, :3].astype(np.float64) @ mat_rec709_T
                    ).astype(np.float32)
                    c_out = np.empty((n_corners, 4), dtype=np.float32)
                    c_out[:, :3] = rgb
                    c_out[:, 3] = c_tri[:, 3]
                    values_sec = write_ndarray(local_stream, c_out)

                    raw_srgb = np.empty(n_loops * 4, dtype=np.float32)
                    attribute.data.foreach_get("color_srgb", raw_srgb)
                    srgb_tri = raw_srgb.reshape(n_loops, 4)[tri_loops]
                    byte_vals = np.clip(
                        np.floor(srgb_tri.astype(np.float64) * 255.0 + 0.5),
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                    byte_sec = write_ndarray(local_stream, byte_vals)
                else:
                    values_sec = write_ndarray(local_stream, c_tri)
                    byte_sec = None
            color_attr_entries.append({
                "name": attribute.name,
                "domain": attribute.domain,
                "data_type": attribute.data_type,
                "values": values_sec,
                "byte_values": byte_sec,
            })

        mesh_dict = {
            "name": obj.name,
            "uses_adaptive_subdivision": _cycles_uses_adaptive_subdivision(
                obj, is_viewport=is_viewport
            ),
            "point_count": n_verts,
            "corner_count": n_corners,
            "triangle_count": n_tris,
            "positions": positions_sec,
            "normal_domain": normal_domain,
            "normals": normals_sec,
            "uv_domain": "CORNER",
            "default_uv_available": default_uv_available,
            "uv": uv_sec,
            "uv_tangent_domain": "CORNER",
            "uv_layers": [
                {
                    "name": name,
                    "domain": "CORNER",
                    "values": sec,
                }
                for name, sec in uv_sections.items()
            ],
            "generated_domain": "POINT",
            "generated": generated_sec,
            "generated_transform": generated_transform,
            "default_color_attribute": (
                mesh.color_attributes.default_color_name or None
            ),
            "color_attributes": color_attr_entries,
            "pointiness_source": pointiness_source,
            "island_edges": {
                "edge_count": n_edges,
                "edges": dict(edges_sec),
            },
            "indices": indices_sec,
            "triangle_material_slots": materials_sec,
            "triangle_smooth": smooth_sec,
            "material_slots": [
                slot.material.name if slot.material else None
                for slot in obj.material_slots
            ],
        }
        return mesh_dict, local_stream.getvalue()
    finally:
        evaluated.to_mesh_clear()


def extract_mesh_geometry(
    obj: Any,
    depsgraph: Any,
    stream: Any,
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
    *,
    is_viewport: bool = False,
    pointiness_cache: dict[int, bool] | None = None,
    mesh_cache: Any = None,
    scene: Any = None,
) -> dict[str, Any] | None:
    if mesh_cache is not None and scene is not None and obj.type == "MESH" and obj.data is not None:
        settings_mode = "PREVIEW" if is_viewport else "RENDER"
        needs_pointiness = any(
            material_uses_pointiness(slot.material, pointiness_cache)
            for slot in obj.material_slots
        )
        material_names = tuple(
            slot.material.name if slot.material is not None else None
            for slot in obj.material_slots
        )
        mesh_ptr = (
            obj.data.original.as_pointer()
            if getattr(obj.data, "original", None)
            else obj.data.as_pointer()
        )
        adaptive_subdiv = _cycles_uses_adaptive_subdivision(obj, is_viewport=is_viewport)
        if not obj.is_modified(scene, settings_mode):
            cache_key: tuple[Any, ...] = (
                "mesh",
                mesh_ptr,
                settings_mode,
                needs_pointiness,
                rec709_to_rgb,
                material_names,
                adaptive_subdiv,
            )
        else:
            obj_ptr = (
                obj.original.as_pointer()
                if getattr(obj, "original", None)
                else obj.as_pointer()
            )
            mod_sig = tuple(
                (
                    m.type,
                    m.name,
                    bool(m.show_viewport if is_viewport else m.show_render),
                    int(getattr(m, "levels", 0))
                    if is_viewport
                    else int(getattr(m, "render_levels", 0)),
                )
                for m in obj.modifiers
            )
            cache_key = (
                "object",
                obj_ptr,
                mesh_ptr,
                settings_mode,
                needs_pointiness,
                rec709_to_rgb,
                material_names,
                adaptive_subdiv,
                mod_sig,
            )

        cached = mesh_cache.get(cache_key)
        if cached is not None:
            template_dict, raw_bytes = cached
            mesh_dict = copy.deepcopy(template_dict)
            mesh_dict["name"] = obj.name
            base_offset = stream.tell()
            stream.write(raw_bytes)
            return rebase_geometry_sections(mesh_dict, base_offset)

        extracted = _extract_mesh_geometry_to_local_stream(
            obj,
            depsgraph,
            rec709_to_rgb,
            is_viewport=is_viewport,
            pointiness_cache=pointiness_cache,
        )
        if extracted is None:
            return None
        template_dict, raw_bytes = extracted
        mesh_cache.put(cache_key, template_dict, raw_bytes)
        mesh_dict = copy.deepcopy(template_dict)
        base_offset = stream.tell()
        stream.write(raw_bytes)
        return rebase_geometry_sections(mesh_dict, base_offset)

    extracted = _extract_mesh_geometry_to_local_stream(
        obj,
        depsgraph,
        rec709_to_rgb,
        is_viewport=is_viewport,
        pointiness_cache=pointiness_cache,
    )
    if extracted is None:
        return None
    mesh_dict, raw_bytes = extracted
    base_offset = stream.tell()
    stream.write(raw_bytes)
    return rebase_geometry_sections(mesh_dict, base_offset)


def object_has_particle_hair(
    obj: Any,
    depsgraph: Any,
    is_viewport: bool = False,
) -> bool:
    if obj.type != "MESH":
        return False
    return bool(
        _particle_hair_systems(obj.evaluated_get(depsgraph), is_viewport=is_viewport)
    )


def extract_particle_hair_geometry(
    obj: Any,
    depsgraph: Any,
    stream: Any,
    shape: str,
    subdivisions: int,
    *,
    is_viewport: bool = False,
) -> dict[str, Any] | None:
    evaluated = obj.evaluated_get(depsgraph)
    systems = _particle_hair_systems(evaluated, is_viewport=is_viewport)
    if not systems:
        return None

    requested_color_attributes = set().union(
        *(
            _material_named_color_attributes(slot.material)
            for slot in obj.material_slots
        )
    )
    emitter_mesh = evaluated.to_mesh(
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    try:
        uv_layer_names = [layer.name for layer in emitter_mesh.uv_layers]
        eligible_color_attributes = [
            attribute.name
            for attribute in emitter_mesh.attributes
            if attribute.data_type == "BYTE_COLOR" and attribute.domain == "CORNER"
        ]
        color_attributes = [
            (name, index)
            for index, name in enumerate(eligible_color_attributes)
            if name in requested_color_attributes
        ]
        default_uv_layer = (
            emitter_mesh.uv_layers.active_render.name
            if emitter_mesh.uv_layers.active_render is not None
            else None
        )
    finally:
        evaluated.to_mesh_clear()

    object_to_world = np.asarray(evaluated.matrix_world, dtype=np.float32)
    world_to_object = np.linalg.inv(object_to_world).astype(np.float32, copy=False)
    keys = array.array("f")
    curve_first_key = array.array("I")
    curve_material_slots = array.array("I")
    intercepts = array.array("f")
    lengths = array.array("f")
    randoms = array.array("f")
    curve_uv_layers = {name: array.array("f") for name in uv_layer_names}
    curve_color_attributes = {name: array.array("f") for name, _ in color_attributes}
    segment_count = 0
    curve_index = 0
    material_slot_count = max(len(obj.material_slots), 1)

    for modifier, particle_system, settings in systems:
        parent_count = len(particle_system.particles)
        child_count = len(particle_system.child_particles)
        first_particle = (
            parent_count
            if settings.child_type != "NONE" and child_count != 0
            else 0
        )
        particle_end = parent_count + child_count
        step_attr = "display_step" if is_viewport else "render_step"
        key_count = (1 << int(getattr(settings, step_attr, settings.render_step))) + 1
        if settings.kink == "SPIRAL":
            key_count += int(settings.kink_extra_steps)
        if particle_end <= first_particle or key_count < 2:
            continue

        base_radius = np.float32(np.float32(settings.radius_scale) * np.float32(0.5))
        root_radius = np.float32(base_radius * np.float32(settings.root_radius))
        tip_radius = np.float32(base_radius * np.float32(settings.tip_radius))
        radius_shape = np.float32(settings.shape)
        close_tip = bool(settings.use_close_tip)
        material_slot = min(max(int(settings.material) - 1, 0), material_slot_count - 1)

        for particle_no in range(first_particle, particle_end):
            curve_first_key.append(len(keys) // 4)
            for uv_no, name in enumerate(uv_layer_names):
                uv = _cycles_particle_hair_uv(modifier, particle_system, particle_no, uv_no)
                curve_uv_layers[name].extend((float(uv[0]), float(uv[1])))
            for name, color_no in color_attributes:
                color = _cycles_particle_hair_color(
                    modifier, particle_system, particle_no, color_no
                )
                curve_color_attributes[name].extend(
                    _cycles_particle_hair_color_value(color)
                )
            world_positions = _cycles_particle_hair_positions(
                particle_system, evaluated, particle_no, key_count
            )
            positions: list[np.ndarray[Any, np.dtype[np.float32]]] = []
            curve_times: list[np.float32] = []
            curve_length = np.float32(0.0)
            previous: np.ndarray[Any, np.dtype[np.float32]] | None = None
            for coordinate in world_positions:
                world = np.asarray(
                    (float(coordinate[0]), float(coordinate[1]), float(coordinate[2]), 1.0),
                    dtype=np.float32,
                )
                local = np.asarray((world_to_object @ world)[:3], dtype=np.float32)
                if previous is not None:
                    delta = np.asarray(local - previous, dtype=np.float32)
                    squared_length = np.float32(np.dot(delta, delta))
                    curve_length = np.float32(
                        curve_length + np.float32(np.sqrt(squared_length))
                    )
                positions.append(local)
                curve_times.append(curve_length)
                previous = local

            for key, (position, curve_time) in enumerate(
                zip(positions, curve_times, strict=True)
            ):
                intercept = np.float32(
                    np.float32(curve_time / curve_length)
                    if curve_length > np.float32(0.0)
                    else np.float32(0.0)
                )
                radius = _cycles_shape_radius(
                    radius_shape, root_radius, tip_radius, intercept
                )
                if close_tip and key + 1 == key_count:
                    radius = np.float32(0.0)
                keys.extend(
                    (float(position[0]), float(position[1]), float(position[2]), float(radius))
                )
                intercepts.append(float(intercept))

            curve_material_slots.append(material_slot)
            lengths.append(float(curve_length))
            randoms.append(_cycles_uint_to_float(cycles_hash.hash_uint2(curve_index, 0)))
            segment_count += key_count - 1
            curve_index += 1

    if curve_index == 0:
        return None
    return {
        "name": f"{obj.name}.particle_hair",
        "shape": {"RIBBONS": "RIBBON", "THICK": "THICK", "THICK_LINEAR": "THICK_LINEAR"}[
            shape
        ],
        "subdivisions": int(subdivisions),
        "key_count": len(keys) // 4,
        "curve_count": curve_index,
        "segment_count": segment_count,
        "keys": write_array(stream, keys),
        "curve_first_key": write_array(stream, curve_first_key),
        "curve_material_slots": write_array(stream, curve_material_slots),
        "default_uv_layer": default_uv_layer,
        "uv_layers": [
            {"name": name, "values": write_array(stream, values)}
            for name, values in curve_uv_layers.items()
        ],
        "color_attributes": [
            {"name": name, "values": write_array(stream, values)}
            for name, values in curve_color_attributes.items()
        ],
        "intercept": write_array(stream, intercepts),
        "length": write_array(stream, lengths),
        "random": write_array(stream, randoms),
        "material_slots": [
            slot.material.name if slot.material else None for slot in obj.material_slots
        ],
    }


def instance_ray_visibility(object_instance: Any) -> dict[str, bool]:
    obj = object_instance.object
    parent = object_instance.parent if object_instance.is_instance else None

    def visible(property_name: str) -> bool:
        result = bool(getattr(obj, property_name))
        if parent is not None and parent != obj:
            result = result and bool(getattr(parent, property_name))
        return result

    return {
        "camera": visible("visible_camera"),
        "diffuse": visible("visible_diffuse"),
        "glossy": visible("visible_glossy"),
        "transmission": visible("visible_transmission"),
        "shadow": visible("visible_shadow"),
        "volume_scatter": visible("visible_volume_scatter"),
    }


def cycles_object_is_geometry(object_instance: Any) -> bool:
    if not object_instance.show_self:
        return False
    obj = object_instance.object
    if obj.data is None:
        return False
    is_cycles_geometry = isinstance(obj.data, bpy.types.Mesh) or obj.type in {
        "VOLUME",
        "CURVES",
        "POINTCLOUD",
        "LIGHT",
    }
    return is_cycles_geometry and any(instance_ray_visibility(object_instance).values())


def extract_light(
    obj: Any,
    view_layer: Any,
    *,
    object_instance: Any,
    particle_source: dict[str, Any] | None = None,
    transform: Any | None = None,
    visibility: dict[str, bool] | None = None,
    is_shadow_catcher: bool = False,
    cycles_sync: dict[str, int] | None = None,
) -> dict[str, Any]:
    light = obj.data
    cycles = getattr(light, "cycles", None)
    use_temperature = bool(getattr(light, "use_temperature", False))
    temperature_color = (
        list(light.temperature_color) if use_temperature else [1.0, 1.0, 1.0]
    )
    size = (
        float(light.shadow_soft_size)
        if light.type in {"POINT", "SPOT"}
        else float(getattr(light, "size", 0.0))
    )
    return {
        "name": obj.name,
        "shader_name": original_id(light).name,
        "type": light.type,
        "transform": column_major(obj.matrix_world if transform is None else transform),
        "color": list(light.color),
        "temperature_color": temperature_color,
        "energy": float(light.energy),
        "exposure": float(getattr(light, "exposure", 0.0)),
        "normalize": bool(getattr(light, "normalize", True)),
        "shape": getattr(light, "shape", "POINT"),
        "size": size,
        "size_y": float(getattr(light, "size_y", 0.0)),
        "spread": float(getattr(light, "spread", 3.141592653589793)),
        "spot_size": float(getattr(light, "spot_size", 0.7853981633974483)),
        "spot_blend": float(getattr(light, "spot_blend", 0.15)),
        "angle": float(getattr(light, "angle", 0.0)),
        "use_soft_falloff": bool(getattr(light, "use_soft_falloff", False)),
        "use_multiple_importance_sampling": bool(
            getattr(cycles, "use_multiple_importance_sampling", True)
        ),
        "is_portal": bool(getattr(cycles, "is_portal", False)),
        "max_bounces": int(getattr(cycles, "max_bounces", 1024)),
        "cast_shadow": bool(getattr(light, "use_shadow", True)),
        "visibility": (
            visibility
            if visibility is not None
            else {
                "camera": bool(getattr(obj, "visible_camera", True)),
                "diffuse": bool(getattr(obj, "visible_diffuse", True)),
                "glossy": bool(getattr(obj, "visible_glossy", True)),
                "transmission": bool(getattr(obj, "visible_transmission", True)),
                "shadow": bool(getattr(obj, "visible_shadow", True)),
                "volume_scatter": bool(getattr(obj, "visible_volume_scatter", True)),
            }
        ),
        "is_shadow_catcher": is_shadow_catcher,
        **_cycles_object_properties(object_instance, view_layer),
        "random_id": int(
            object_instance.random_id
            if object_instance.is_instance
            else _cycles_object_random_id(obj.name)
        )
        & cycles_hash.UINT32_MASK,
        "particle_index": _cycles_particle_index(object_instance),
        "cycles_particle_source": particle_source,
        "cycles_sync": cycles_sync,
        "node_tree": (
            serialize_shader_tree(light.node_tree, light=True)
            if light.use_nodes
            else None
        ),
    }


def extract_geometry_instance(
    object_instance: Any,
    geometry_type: str,
    geometry_index: int,
    cycles_object_index: int,
    light_groups: dict[str, int],
    particle_source: dict[str, Any] | None,
    view_layer: Any,
) -> dict[str, Any]:
    return {
        "name": object_instance.object.name,
        "geometry_type": geometry_type,
        "geometry": geometry_index,
        "transform": column_major(object_instance.matrix_world),
        "persistent_id": list(object_instance.persistent_id),
        "is_instance": bool(object_instance.is_instance),
        "particle_index": _cycles_particle_index(object_instance),
        "cycles_particle_source": particle_source,
        "random_id": int(
            object_instance.random_id
            if object_instance.is_instance
            else _cycles_object_random_id(object_instance.object.name)
        )
        & cycles_hash.UINT32_MASK,
        **_cycles_object_properties(object_instance, view_layer),
        "ambient_occlusion_distance": _cycles_object_ao_distance(object_instance),
        "visibility": instance_ray_visibility(object_instance),
        "is_shadow_catcher": _cycles_shadow_catcher(object_instance),
        "cycles_sync": {
            "object_index": cycles_object_index,
            "light_group": _cycles_light_group(object_instance, light_groups),
        },
    }


def camera_focus_distance(camera: Any, depsgraph: Any) -> float:
    dof = camera.data.dof
    focus_object = dof.focus_object
    if focus_object is None:
        return max(float(dof.focus_distance), 1.0e-5)

    focus_object = focus_object.evaluated_get(depsgraph)
    focus_transform = focus_object.matrix_world
    focus_subtarget = str(getattr(dof, "focus_subtarget", ""))
    pose = getattr(focus_object, "pose", None)
    if pose is not None and focus_subtarget:
        pose_bone = pose.bones.get(focus_subtarget)
        if pose_bone is not None:
            focus_transform = focus_transform @ pose_bone.matrix

    camera_transform = camera.matrix_world
    view_direction = camera_transform.col[2].xyz.normalized()
    camera_to_focus = camera_transform.translation - focus_transform.translation
    return max(abs(float(view_direction.dot(camera_to_focus))), 1.0e-5)


def normalized_transform(matrix: Any) -> Any:
    transform = matrix.copy()
    for column_index in range(3):
        column = transform.col[column_index].xyz.normalized()
        for row_index in range(3):
            transform[row_index][column_index] = column[row_index]
    return transform
