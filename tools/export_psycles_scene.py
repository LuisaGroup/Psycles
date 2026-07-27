"""Export evaluated Blender scene data for the Psycles Luisa runtime.

Usage:

    blender scene.blend --background --python export_psycles_scene.py -- out

The output directory contains:

* scene.json: cameras, instances, lights, normalized node-tree data and
  byte ranges for geometry;
* geometry.bin: little-endian, tightly packed vertex/triangle streams;
* textures/: exact packed image payloads (or external source copies).

This exporter never evaluates, bakes, or replaces a shader graph. Blender and
Cycles remain the semantic source; node evaluation is lowered to Luisa DSL.
"""

from __future__ import annotations

import array
import hashlib
import json
import pathlib
import shutil
import struct
import sys
from typing import Any

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blender_scene_manifest as manifest  # noqa: E402


_UINT32_MASK = 0xFFFFFFFF


def _u32(value: int) -> int:
    return value & _UINT32_MASK


def _rotate_left_u32(value: int, bits: int) -> int:
    value = _u32(value)
    return _u32((value << bits) | (value >> (32 - bits)))


def _cycles_hash_final(a: int, b: int, c: int) -> tuple[int, int, int]:
    # Jenkins lookup3 final(), copied arithmetically from Cycles util/hash.h.
    c = _u32((c ^ b) - _rotate_left_u32(b, 14))
    a = _u32((a ^ c) - _rotate_left_u32(c, 11))
    b = _u32((b ^ a) - _rotate_left_u32(a, 25))
    c = _u32((c ^ b) - _rotate_left_u32(b, 16))
    a = _u32((a ^ c) - _rotate_left_u32(c, 4))
    b = _u32((b ^ a) - _rotate_left_u32(a, 14))
    c = _u32((c ^ b) - _rotate_left_u32(b, 24))
    return a, b, c


def _cycles_hash_uint(value: int) -> int:
    a = b = c = _u32(0xDEADBEEF + (1 << 2) + 13)
    a = _u32(a + value)
    return _cycles_hash_final(a, b, c)[2]


def _cycles_hash_uint2(x: int, y: int) -> int:
    a = b = c = _u32(0xDEADBEEF + (2 << 2) + 13)
    b = _u32(b + y)
    a = _u32(a + x)
    return _cycles_hash_final(a, b, c)[2]


def _cycles_hash_string(value: str) -> int:
    result = 0
    for byte in value.encode("utf-8"):
        result = _u32(result * 37 + byte)
    return result


def _cycles_uint_to_float(value: int) -> float:
    return float(_u32(value)) / float(_UINT32_MASK)


def _cycles_object_random_id(name: str) -> int:
    return _cycles_hash_uint2(_cycles_hash_string(name), 0)


def _column_major(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


def _safe_name(name: str) -> str:
    clean = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in name
    )
    return clean[:120] or "unnamed"


def _socket_links(tree: Any) -> dict[tuple[str, str], tuple[str, str]]:
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for link in tree.links:
        result[(link.to_node.name, link.to_socket.identifier)] = (
            link.from_node.name,
            link.from_socket.identifier,
        )
    return result


def _active_output(
    tree: Any,
    socket: str,
    world: bool = False,
) -> dict[str, str] | None:
    expected = "ShaderNodeOutputWorld" if world else "ShaderNodeOutputMaterial"
    for node in tree.nodes:
        if node.bl_idname == expected and getattr(node, "is_active_output", True):
            links = _socket_links(tree)
            source = links.get((node.name, socket))
            if source:
                return {"node": source[0], "socket": source[1]}
    return None


def _tree(tree: Any, world: bool = False) -> dict[str, Any] | None:
    data = manifest._node_tree_manifest(tree)
    if data is not None:
        data["surface_root"] = _active_output(tree, "Surface", world)
        data["volume_root"] = _active_output(tree, "Volume", world)
        data["displacement_root"] = (
            None
            if world
            else _active_output(tree, "Displacement", world)
        )
    return data


def _write_array(stream: Any, values: array.array[Any]) -> dict[str, int]:
    if sys.byteorder != "little":
        values.byteswap()
    offset = stream.tell()
    values.tofile(stream)
    return {"offset": offset, "bytes": stream.tell() - offset}


def _geometry(
    obj: Any,
    depsgraph: Any,
    stream: Any,
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
    owned_mesh = None
    try:
        # Mesh.calc_tangents rejects n-gons, while Cycles and Psycles both
        # consume triangles. Triangulate only n-gons on Blender's temporary
        # evaluated mesh so tangent generation and exported Accel topology
        # share the same corner-domain UV/color data. Existing triangles and
        # quads are left untouched.
        if any(len(polygon.vertices) > 4 for polygon in mesh.polygons):
            # Never edit the depsgraph-owned temporary mesh in place: doing
            # so can leave its CustomData layer pointers stale. An owned ID
            # copy keeps UV/color corner layers valid through BMesh writeback.
            owned_mesh = mesh.copy()
            mesh = owned_mesh
            edit_mesh = bmesh.new()
            try:
                edit_mesh.from_mesh(mesh)
                ngon_faces = [
                    face
                    for face in edit_mesh.faces
                    if len(face.verts) > 4
                ]
                if ngon_faces:
                    bmesh.ops.triangulate(
                        edit_mesh,
                        faces=ngon_faces,
                        quad_method="FIXED",
                        ngon_method="BEAUTY",
                    )
                    edit_mesh.to_mesh(mesh)
                    mesh.update()
            finally:
                edit_mesh.free()
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            return None
        uv_layer = mesh.uv_layers.active
        has_uv_tangents = False
        if uv_layer is not None:
            try:
                mesh.calc_tangents(uvmap=uv_layer.name)
                has_uv_tangents = True
            except RuntimeError:
                # Match Cycles' missing tangent-attribute behavior. The
                # exported zero tangent is consumed as "use sd->N".
                has_uv_tangents = False
        # calc_tangents may reallocate loop CustomData. Never retain the
        # pre-call Python layer wrapper, which can become a dangling pointer
        # on complex evaluated meshes.
        uv_layer = mesh.uv_layers.active
        positions = array.array("f")
        normals = array.array("f")
        uvs = array.array("f")
        uv_tangents = array.array("f")
        generated = array.array("f")
        color_layers = [
            attribute
            for attribute in mesh.color_attributes
            if attribute.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}
            and attribute.domain in {"CORNER", "POINT"}
        ]
        color_values = {
            attribute.name: array.array("f")
            for attribute in color_layers
        }
        indices = array.array("I")
        materials = array.array("I")
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
        generated_by_vertex = [
            tuple(
                float(vertex.co[axis]) * generated_scale[axis]
                - generated_location[axis]
                for axis in range(3)
            )
            for vertex in mesh.vertices
        ]

        # Match Blender's DisjointSet union-by-rank and Cycles'
        # attr_create_random_per_island. The result is a face-domain value,
        # deliberately not a vertex attribute, so it remains constant across
        # each triangle.
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

        next_index = 0
        for triangle in mesh.loop_triangles:
            for loop_index in triangle.loops:
                loop = mesh.loops[loop_index]
                vertex = mesh.vertices[loop.vertex_index]
                positions.extend(float(value) for value in vertex.co)
                normals.extend(float(value) for value in loop.normal)
                generated.extend(generated_by_vertex[loop.vertex_index])
                for attribute in color_layers:
                    attribute_index = (
                        loop_index
                        if attribute.domain == "CORNER"
                        else loop.vertex_index
                    )
                    # Blender's .color accessor is scene-linear for both
                    # FLOAT_COLOR and BYTE_COLOR. Cycles likewise decodes
                    # ATTR_ELEMENT_CORNER_BYTE from sRGB before shading.
                    color_values[attribute.name].extend(
                        float(value)
                        for value in attribute.data[
                            attribute_index
                        ].color
                    )
                if uv_layer is None:
                    uvs.extend((0.0, 0.0))
                else:
                    uvs.extend(
                        float(value) for value in uv_layer.data[loop_index].uv
                    )
                if has_uv_tangents:
                    uv_tangents.extend(
                        float(value) for value in loop.tangent
                    )
                    uv_tangents.append(float(loop.bitangent_sign))
                else:
                    uv_tangents.extend((0.0, 0.0, 0.0, 0.0))
                indices.append(next_index)
                next_index += 1
            materials.append(int(triangle.material_index))
            first_vertex = mesh.loops[triangle.loops[0]].vertex_index
            random_per_island.append(
                _cycles_uint_to_float(
                    _cycles_hash_uint(find_root(first_vertex))
                )
            )
        return {
            "name": obj.name,
            "vertex_count": next_index,
            "triangle_count": len(mesh.loop_triangles),
            "positions": _write_array(stream, positions),
            "normals": _write_array(stream, normals),
            "uv": _write_array(stream, uvs),
            "uv_tangents": _write_array(stream, uv_tangents),
            "generated": _write_array(stream, generated),
            "color_attributes": [
                {
                    "name": attribute.name,
                    "domain": attribute.domain,
                    "data_type": attribute.data_type,
                    "values": _write_array(
                        stream, color_values[attribute.name]
                    ),
                }
                for attribute in color_layers
            ],
            "indices": _write_array(stream, indices),
            "triangle_material_slots": _write_array(stream, materials),
            "triangle_random_per_island": _write_array(
                stream, random_per_island
            ),
            "material_slots": [
                slot.material.name if slot.material else None
                for slot in obj.material_slots
            ],
        }
    finally:
        if owned_mesh is not None:
            bpy.data.meshes.remove(owned_mesh)
        evaluated.to_mesh_clear()


def _image_extension(image: Any) -> str:
    suffix = pathlib.Path(image.filepath).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".tga", ".bmp", ".hdr"}:
        return suffix
    return {
        "JPEG": ".jpg",
        "PNG": ".png",
        "TARGA": ".tga",
        "BMP": ".bmp",
        "HDR": ".hdr",
    }.get(image.file_format, ".bin")


def _export_images(output: pathlib.Path) -> list[dict[str, Any]]:
    texture_directory = output / "textures"
    texture_directory.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    for index, image in enumerate(
        sorted(bpy.data.images, key=lambda candidate: candidate.name)
    ):
        if image.type in {"RENDER_RESULT", "COMPOSITING"} or min(image.size) <= 0:
            continue
        extension = _image_extension(image)
        destination = texture_directory / (
            f"{index:03d}-{_safe_name(image.name)}{extension}"
        )
        if image.packed_file is not None:
            destination.write_bytes(bytes(image.packed_file.data))
        else:
            source = pathlib.Path(bpy.path.abspath(image.filepath))
            if not source.is_file():
                raise FileNotFoundError(
                    f"external image is missing: {image.name}: {source}"
                )
            shutil.copyfile(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        result.append(
            {
                "name": image.name,
                "path": destination.relative_to(output).as_posix(),
                "width": int(image.size[0]),
                "height": int(image.size[1]),
                "source": image.source,
                "colorspace": image.colorspace_settings.name,
                "alpha_mode": image.alpha_mode,
                "sha256": digest,
            }
        )
    return result


def _light(obj: Any) -> dict[str, Any]:
    light = obj.data
    return {
        "name": obj.name,
        "type": light.type,
        "transform": _column_major(obj.matrix_world),
        "color": list(light.color),
        "energy": float(light.energy),
        "shape": getattr(light, "shape", "POINT"),
        "size": float(getattr(light, "size", 0.0)),
        "size_y": float(getattr(light, "size_y", 0.0)),
        "spread": float(getattr(light, "spread", 3.141592653589793)),
        "spot_size": float(getattr(light, "spot_size", 0.7853981633974483)),
        "spot_blend": float(getattr(light, "spot_blend", 0.15)),
        "node_tree": _tree(light.node_tree) if light.use_nodes else None,
    }


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected one output directory after '--'")
    output = pathlib.Path(args[0]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene

    # ``evaluated_depsgraph_get`` returns the viewport dependency graph when
    # invoked from a background Python script.  Render-only objects are
    # therefore absent when either the object or one of its collections has
    # ``hide_viewport`` enabled, even if Cycles renders it because
    # ``hide_render`` is disabled.  Build an equivalent render-visible
    # dependency graph for export.  The loaded .blend is never saved, so these
    # temporary visibility changes cannot modify the source scene.
    for collection in bpy.data.collections:
        collection.hide_viewport = collection.hide_render
    for obj in scene.objects:
        obj.hide_viewport = obj.hide_render
        if not obj.hide_render:
            obj.hide_set(False)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    geometries: list[dict[str, Any]] = []
    geometry_by_object: dict[str, int] = {}
    instances: list[dict[str, Any]] = []
    with (output / "geometry.bin").open("wb") as stream:
        stream.write(b"PSYGEO1\0")
        stream.write(struct.pack("<II", 1, 0))
        for object_instance in depsgraph.object_instances:
            obj = object_instance.object
            original = obj.original if obj.original is not None else obj
            if original.hide_render:
                continue
            if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
                continue
            key = original.name_full
            geometry_index = geometry_by_object.get(key)
            if geometry_index is None:
                geometry_data = _geometry(original, depsgraph, stream)
                if geometry_data is None:
                    continue
                geometry_index = len(geometries)
                geometry_by_object[key] = geometry_index
                geometries.append(geometry_data)
            instances.append(
                {
                    "name": object_instance.object.name,
                    "geometry": geometry_index,
                    "transform": _column_major(object_instance.matrix_world),
                    "persistent_id": list(object_instance.persistent_id),
                    "is_instance": bool(object_instance.is_instance),
                    # Cycles stores particle data for both parent and child
                    # particles and exposes that data index through Particle
                    # Info. Blender's dependency-graph persistent ID is the
                    # corresponding particle index for particle duplis.
                    "particle_index": int(
                        object_instance.persistent_id[0]
                        if object_instance.particle_system is not None
                        else 0
                    )
                    & _UINT32_MASK,
                    # Cycles uses the dependency-graph random id for duplis,
                    # and hash_uint2(hash_string(object name), 0) otherwise.
                    "random_id": int(
                        object_instance.random_id
                        if object_instance.is_instance
                        else _cycles_object_random_id(
                            object_instance.object.name
                        )
                    )
                    & _UINT32_MASK,
                    "visibility": {
                        "camera": bool(original.visible_camera),
                        "diffuse": bool(original.visible_diffuse),
                        "glossy": bool(original.visible_glossy),
                        "transmission": bool(original.visible_transmission),
                        "shadow": bool(original.visible_shadow),
                        "volume_scatter": bool(
                            original.visible_volume_scatter
                        ),
                    },
                }
            )

    geometry_size = (output / "geometry.bin").stat().st_size
    for geometry in geometries:
        sections = [
            geometry[name]
            for name in (
                "positions",
                "normals",
                "uv",
                "uv_tangents",
                "generated",
                "indices",
                "triangle_material_slots",
                "triangle_random_per_island",
            )
        ]
        sections.extend(
            attribute["values"]
            for attribute in geometry["color_attributes"]
        )
        for section in sections:
            end = int(section["offset"]) + int(section["bytes"])
            if (
                int(section["offset"]) < 0
                or int(section["bytes"]) < 0
                or end > geometry_size
            ):
                raise RuntimeError(
                    f"incomplete geometry.bin section for "
                    f"{geometry['name']!r}: {section}, "
                    f"file size {geometry_size}"
                )

    materials = []
    for material in sorted(bpy.data.materials, key=lambda item: item.name):
        materials.append(
            {
                "name": material.name,
                "surface_render_method": getattr(
                    material, "surface_render_method", None
                ),
                "node_tree": (
                    _tree(material.node_tree)
                    if material.use_nodes and material.node_tree
                    else None
                ),
            }
        )

    camera = scene.camera
    camera_data = None
    if camera is not None:
        camera_data = {
            "name": camera.name,
            "type": camera.data.type,
            "transform": _column_major(camera.matrix_world),
            "angle": float(camera.data.angle),
            "angle_x": float(camera.data.angle_x),
            "angle_y": float(camera.data.angle_y),
            "sensor_fit": str(camera.data.sensor_fit),
            "lens": float(camera.data.lens),
            "ortho_scale": float(camera.data.ortho_scale),
            "shift_x": float(camera.data.shift_x),
            "shift_y": float(camera.data.shift_y),
            "clip_start": float(camera.data.clip_start),
            "clip_end": float(camera.data.clip_end),
            "dof": {
                "enabled": bool(camera.data.dof.use_dof),
                "focus_distance": float(camera.data.dof.focus_distance),
                "fstop": float(camera.data.dof.aperture_fstop),
                "blades": int(camera.data.dof.aperture_blades),
                "rotation": float(camera.data.dof.aperture_rotation),
                "ratio": float(camera.data.dof.aperture_ratio),
            },
        }

    payload = {
        "schema": "psycles.blender-scene.v1",
        "source": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "frame": scene.frame_current,
        "camera": camera_data,
        "render": {
            "width": scene.render.resolution_x,
            "height": scene.render.resolution_y,
            "percentage": scene.render.resolution_percentage,
            "transparent": scene.render.film_transparent,
            "pixel_filter_type": scene.cycles.pixel_filter_type,
            "filter_width": float(scene.cycles.filter_width),
            "pass_alpha_threshold": float(
                bpy.context.view_layer.pass_alpha_threshold
            ),
            "cycles": manifest._cycles_settings(scene),
        },
        "geometries": geometries,
        "instances": instances,
        "materials": materials,
        "lights": [
            _light(obj)
            for obj in sorted(bpy.data.objects, key=lambda item: item.name)
            if obj.type == "LIGHT" and not obj.hide_render
        ],
        "world": (
            {
                "name": scene.world.name,
                "color": list(scene.world.color),
                "node_tree": (
                    _tree(scene.world.node_tree, world=True)
                    if scene.world.use_nodes
                    else None
                ),
            }
            if scene.world
            else None
        ),
        # World shader resources are generated by Psycles/Luisa. The exporter
        # must never call Cycles to bake a world texture.
        "world_environment": None,
        "node_groups": [
            _tree(group)
            for group in sorted(
                bpy.data.node_groups, key=lambda item: item.name
            )
            if group.bl_idname == "ShaderNodeTree"
        ],
        "images": _export_images(output),
    }
    (output / "scene.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(geometries)} geometries, {len(instances)} "
        f"instances, {len(materials)} materials, "
        f"{len(payload['images'])} images to {output}"
    )


if __name__ == "__main__":
    _main()
