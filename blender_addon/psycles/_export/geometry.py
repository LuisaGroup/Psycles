"""Evaluated mesh, hair, instance, light, and camera extraction for Psycles."""

from __future__ import annotations

import array
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


def _cycles_color_attribute_value(
    attribute: Any,
    index: int,
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float, float]:
    value = tuple(float(component) for component in attribute.data[index].color)
    if attribute.data_type != "BYTE_COLOR" or attribute.domain != "CORNER":
        return value
    rgb = value[:3]
    return tuple(
        sum(row[channel] * rgb[channel] for channel in range(3))
        for row in rec709_to_rgb
    ) + (value[3],)


def _cycles_corner_byte_color_value(
    attribute: Any,
    index: int,
) -> tuple[int, int, int, int]:
    return tuple(
        max(0, min(255, int(float(component) * 255.0 + 0.5)))
        for component in attribute.data[index].color_srgb
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


def extract_mesh_geometry(
    obj: Any,
    depsgraph: Any,
    stream: Any,
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
    *,
    is_viewport: bool = False,
    pointiness_cache: dict[int, bool] | None = None,
) -> dict[str, Any] | None:
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
    try:
        needs_pointiness = any(
            material_uses_pointiness(slot.material, pointiness_cache)
            for slot in obj.material_slots
        )
        pointiness_normals = array.array("f")
        pointiness_edges = array.array("I")
        if needs_pointiness:
            pointiness_normals.extend(
                component
                for vertex in mesh.vertices
                for component in vertex.normal
            )
            pointiness_edges.extend(
                int(vertex)
                for edge in mesh.edges
                for vertex in edge.vertices
            )
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            return None
        uv_layer_names = [layer.name for layer in mesh.uv_layers]
        default_uv_name = (
            mesh.uv_layers.active_render.name
            if mesh.uv_layers.active_render is not None
            else None
        )
        uv_by_layer: dict[str, list[tuple[float, float]]] = {}
        tangent_by_layer: dict[str, list[tuple[float, float, float, float]]] = {}
        for uv_layer_name in uv_layer_names:
            uv_layer = mesh.uv_layers.get(uv_layer_name)
            if uv_layer is None:
                continue
            uv_by_layer[uv_layer_name] = [
                tuple(float(value) for value in item.uv)
                for item in uv_layer.data
            ]
            tangent_by_layer[uv_layer_name] = [
                (0.0, 0.0, 0.0, 0.0) for _loop in mesh.loops
            ]
        positions = array.array("f")
        normals = array.array("f")
        uvs = array.array("f")
        uv_tangents = array.array("f")
        uv_layer_values = {name: array.array("f") for name in uv_by_layer}
        uv_tangent_values = {name: array.array("f") for name in uv_by_layer}
        generated = array.array("f")
        color_layers = [
            attribute
            for attribute in mesh.color_attributes
            if attribute.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}
            and attribute.domain in {"CORNER", "POINT"}
        ]
        color_values = {attribute.name: array.array("f") for attribute in color_layers}
        byte_color_values = {
            attribute.name: array.array("B")
            for attribute in color_layers
            if attribute.data_type == "BYTE_COLOR" and attribute.domain == "CORNER"
        }
        indices = array.array("I")
        materials = array.array("I")
        smooth = array.array("I")
        random_per_island = array.array("f")

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
        generated_by_vertex = [
            tuple(
                float(vertex.co[axis]) * generated_scale[axis] - generated_location[axis]
                for axis in range(3)
            )
            for vertex in mesh.vertices
        ]
        positions.extend(
            component for vertex in mesh.vertices for component in vertex.co
        )
        generated.extend(
            component for value in generated_by_vertex for component in value
        )
        blender_normal_domain = str(getattr(mesh, "normals_domain", "CORNER"))
        use_corner_normals = blender_normal_domain == "CORNER"
        sharp_face_attribute = mesh.attributes.get("sharp_face")
        if use_corner_normals:
            smooth_by_polygon = None
            smooth_without_attribute = True
        elif sharp_face_attribute is not None:
            smooth_by_polygon = tuple(
                not bool(item.value) for item in sharp_face_attribute.data
            )
            smooth_without_attribute = False
        else:
            smooth_by_polygon = None
            smooth_without_attribute = blender_normal_domain != "FACE"
        normal_domain = "CORNER" if use_corner_normals else "POINT"
        if normal_domain == "POINT":
            normals.extend(
                component for vertex in mesh.vertices for component in vertex.normal
            )
        for attribute in color_layers:
            if attribute.domain != "POINT":
                continue
            for vertex in mesh.vertices:
                color_values[attribute.name].extend(
                    _cycles_color_attribute_value(attribute, vertex.index, rec709_to_rgb)
                )

        parents = list(range(len(mesh.vertices)))
        ranks = [0] * len(mesh.vertices)

        def find_root(vertex_index: int) -> int:
            root = vertex_index
            while parents[root] != root:
                root = parents[root]
            while parents[vertex_index] != root:
                parent = parents[vertex_index]
                parents[vertex_index] = root
                vertex_index = parent
            return root

        def join_vertices(first: int, second: int) -> None:
            root_first = find_root(first)
            root_second = find_root(second)
            if root_first == root_second:
                return
            if ranks[root_first] < ranks[root_second]:
                root_first, root_second = root_second, root_first
            parents[root_second] = root_first
            if ranks[root_first] == ranks[root_second]:
                ranks[root_first] += 1

        for edge in mesh.edges:
            join_vertices(int(edge.vertices[0]), int(edge.vertices[1]))

        for triangle in mesh.loop_triangles:
            for loop_index in triangle.loops:
                loop = mesh.loops[loop_index]
                if normal_domain == "CORNER":
                    normals.extend(float(value) for value in loop.normal)
                for attribute in color_layers:
                    if attribute.domain != "CORNER":
                        continue
                    color_values[attribute.name].extend(
                        _cycles_color_attribute_value(attribute, loop_index, rec709_to_rgb)
                    )
                    if attribute.name in byte_color_values:
                        byte_color_values[attribute.name].extend(
                            _cycles_corner_byte_color_value(attribute, loop_index)
                        )
                if default_uv_name not in uv_by_layer:
                    uvs.extend((0.0, 0.0))
                else:
                    uvs.extend(uv_by_layer[default_uv_name][loop_index])
                if default_uv_name not in tangent_by_layer:
                    uv_tangents.extend((0.0, 0.0, 0.0, 0.0))
                else:
                    uv_tangents.extend(tangent_by_layer[default_uv_name][loop_index])
                for layer_name, values in uv_by_layer.items():
                    uv_layer_values[layer_name].extend(values[loop_index])
                    uv_tangent_values[layer_name].extend(
                        tangent_by_layer[layer_name][loop_index]
                    )
                indices.append(int(loop.vertex_index))
            materials.append(int(triangle.material_index))
            smooth.append(
                int(
                    smooth_without_attribute
                    if smooth_by_polygon is None
                    else smooth_by_polygon[triangle.polygon_index]
                )
            )
            first_vertex = mesh.loops[triangle.loops[0]].vertex_index
            random_per_island.append(
                _cycles_uint_to_float(cycles_hash.hash_uint(find_root(first_vertex)))
            )
        pointiness_source = None
        if needs_pointiness:
            pointiness_source = {
                "point_normals": write_array(stream, pointiness_normals),
                "edge_count": len(pointiness_edges) // 2,
                "edges": write_array(stream, pointiness_edges),
            }
        return {
            "name": obj.name,
            "uses_adaptive_subdivision": _cycles_uses_adaptive_subdivision(
                obj, is_viewport=is_viewport
            ),
            "point_count": len(mesh.vertices),
            "corner_count": len(mesh.loop_triangles) * 3,
            "triangle_count": len(mesh.loop_triangles),
            "positions": write_array(stream, positions),
            "normal_domain": normal_domain,
            "normals": write_array(stream, normals),
            "uv_domain": "CORNER",
            "default_uv_available": default_uv_name in uv_by_layer,
            "uv": write_array(stream, uvs),
            "uv_tangent_domain": "CORNER",
            "uv_tangents": write_array(stream, uv_tangents),
            "uv_layers": [
                {
                    "name": name,
                    "domain": "CORNER",
                    "values": write_array(stream, uv_layer_values[name]),
                    "tangents": write_array(stream, uv_tangent_values[name]),
                }
                for name in uv_layer_values
            ],
            "generated_domain": "POINT",
            "generated": write_array(stream, generated),
            "generated_transform": generated_transform,
            "default_color_attribute": (
                mesh.color_attributes.default_color_name or None
            ),
            "color_attributes": [
                {
                    "name": attribute.name,
                    "domain": attribute.domain,
                    "data_type": attribute.data_type,
                    "values": write_array(stream, color_values[attribute.name]),
                    "byte_values": (
                        write_array(stream, byte_color_values[attribute.name])
                        if attribute.name in byte_color_values
                        else None
                    ),
                }
                for attribute in color_layers
            ],
            "pointiness_source": pointiness_source,
            "indices": write_array(stream, indices),
            "triangle_material_slots": write_array(stream, materials),
            "triangle_smooth": write_array(stream, smooth),
            "triangle_random_per_island": write_array(stream, random_per_island),
            "material_slots": [
                slot.material.name if slot.material else None
                for slot in obj.material_slots
            ],
        }
    finally:
        evaluated.to_mesh_clear()


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
