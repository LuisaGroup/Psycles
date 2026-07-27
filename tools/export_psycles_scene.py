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
import math
import pathlib
import shutil
import struct
import sys
import tempfile
from typing import Any

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix, Vector

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


def _active_output(tree: Any, world: bool = False) -> dict[str, str] | None:
    expected = "ShaderNodeOutputWorld" if world else "ShaderNodeOutputMaterial"
    for node in tree.nodes:
        if node.bl_idname == expected and getattr(node, "is_active_output", True):
            links = _socket_links(tree)
            source = links.get((node.name, "Surface"))
            if source:
                return {"node": source[0], "socket": source[1]}
    return None


def _tree(tree: Any, world: bool = False) -> dict[str, Any] | None:
    data = manifest._node_tree_manifest(tree)
    if data is not None:
        data["surface_root"] = _active_output(tree, world)
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
    try:
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:
            return None
        uv_layer = mesh.uv_layers.active
        positions = array.array("f")
        normals = array.array("f")
        uvs = array.array("f")
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


def _render_result_linear(scene: Any) -> np.ndarray:
    bpy.ops.render.render(scene=scene.name)
    result = bpy.data.images.get("Render Result")
    if result is None:
        raise RuntimeError("Cycles did not produce a Render Result")
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as file:
        temporary_path = pathlib.Path(file.name)
    try:
        result.save_render(str(temporary_path), scene=scene)
        source = oiio.ImageInput.open(str(temporary_path))
        if source is None:
            raise RuntimeError("could not open temporary world EXR")
        try:
            pixels = source.read_image(format=oiio.FLOAT)
            if pixels is None:
                raise RuntimeError("could not read temporary world EXR")
            return np.asarray(pixels, dtype=np.float32)[:, :, :3].copy()
        finally:
            source.close()
    finally:
        temporary_path.unlink(missing_ok=True)


def _cycles_sun_direction(node: Any) -> Vector:
    elevation = math.fmod(float(node.sun_elevation), 2.0 * math.pi)
    rotation = float(node.sun_rotation)
    if abs(elevation) >= math.pi:
        elevation -= math.copysign(2.0 * math.pi, elevation)
    if elevation >= 0.5 * math.pi or elevation <= -0.5 * math.pi:
        elevation = math.copysign(math.pi, elevation) - elevation
        rotation += math.pi
    rotation = math.fmod(rotation, 2.0 * math.pi)
    if rotation < 0.0:
        rotation += 2.0 * math.pi
    rotation = 2.0 * math.pi - rotation
    return Vector(
        (
            -math.cos(elevation) * math.sin(rotation),
            math.cos(elevation) * math.cos(rotation),
            math.sin(elevation),
        )
    )


def _export_world_environment(
    output: pathlib.Path,
    source_scene: Any,
) -> dict[str, Any] | None:
    world = source_scene.world
    if world is None or not world.use_nodes or world.node_tree is None:
        return None
    sky_nodes = [
        node
        for node in world.node_tree.nodes
        if node.bl_idname == "ShaderNodeTexSky"
        and node.sky_type == "NISHITA"
    ]
    sun_disc_states = {
        node.name: bool(node.sun_disc) for node in sky_nodes
    }
    for node in sky_nodes:
        node.sun_disc = False

    render_scene = bpy.data.scenes.new("__psycles_world_precompute__")
    camera_data = bpy.data.cameras.new(
        "__psycles_world_precompute_camera_data__"
    )
    camera = bpy.data.objects.new(
        "__psycles_world_precompute_camera__", camera_data
    )
    try:
        render_scene.world = world
        render_scene.collection.objects.link(camera)
        render_scene.camera = camera
        render_scene.render.engine = "CYCLES"
        render_scene.cycles.device = "CPU"
        render_scene.cycles.use_denoising = False
        render_scene.render.resolution_percentage = 100
        render_scene.render.film_transparent = False
        render_scene.render.image_settings.file_format = "OPEN_EXR"
        render_scene.render.image_settings.color_depth = "32"
        render_scene.use_nodes = False

        camera_data.type = "PANO"
        # Blender 4.5 exposes the Cycles panorama projection directly on
        # Camera RNA. Older releases used a camera.cycles property group.
        camera_data.panorama_type = "EQUIRECTANGULAR"
        # Cycles panorama directions use X as the camera forward axis while
        # Blender cameras look down local -Z. An identity Blender camera maps
        # a Cycles direction (x, y, z) to (-y, z, -x), as verified by the
        # official-Cycles direction probe. Apply its inverse so the exported map
        # is indexed directly by world-space Cycles equirectangular
        # coordinates rather than carrying a camera-coordinate convention.
        camera.matrix_world = Matrix(
            (
                (0.0, 0.0, -1.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        render_scene.cycles.samples = 1
        render_scene.render.resolution_x = 512
        render_scene.render.resolution_y = 256
        pixels = _render_result_linear(render_scene)
        environment_path = output / "world-environment.float3"
        contiguous = np.ascontiguousarray(pixels, dtype="<f4")
        environment_path.write_bytes(contiguous.tobytes(order="C"))

        suns: list[dict[str, Any]] = []
        camera_data.type = "PERSP"
        camera_data.angle = 1.0e-4
        render_scene.cycles.samples = 16
        render_scene.render.resolution_x = 4
        render_scene.render.resolution_y = 4
        for node in sky_nodes:
            if not sun_disc_states[node.name]:
                continue
            direction = _cycles_sun_direction(node).normalized()
            camera.matrix_world = direction.to_track_quat(
                "-Z", "Y"
            ).to_matrix().to_4x4()
            baseline = np.mean(
                _render_result_linear(render_scene), axis=(0, 1)
            )
            node.sun_disc = True
            with_sun = np.mean(
                _render_result_linear(render_scene), axis=(0, 1)
            )
            node.sun_disc = False
            # Cycles applies radial limb darkening
            # 0.4 + 0.6 * sqrt(1-r^2). Its disk-area mean is 0.8.
            radiance = np.maximum(
                (with_sun - baseline) * 0.8, 0.0
            )
            suns.append(
                {
                    "name": node.name,
                    "direction": [float(value) for value in direction],
                    "radiance": [float(value) for value in radiance],
                    "angular_radius": 0.5 * float(node.sun_size),
                }
            )

        return {
            "name": world.name,
            "path": environment_path.relative_to(output).as_posix(),
            "width": int(pixels.shape[1]),
            "height": int(pixels.shape[0]),
            "format": "float32_rgb_le",
            "mapping": "cycles_equirectangular",
            "sun_discs_excluded": True,
            "suns": suns,
        }
    finally:
        for node in sky_nodes:
            node.sun_disc = sun_disc_states[node.name]
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(camera_data, do_unlink=True)
        bpy.data.scenes.remove(render_scene, do_unlink=True)


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

    world_environment = _export_world_environment(output, scene)
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
        "world_environment": world_environment,
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
