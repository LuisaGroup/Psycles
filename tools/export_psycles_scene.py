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
import traceback
from typing import Any

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blender_scene_manifest as manifest  # noqa: E402
import cycles_hash  # noqa: E402


_CYCLES_DEFAULT_SHADER_COUNT = 5
_CYCLES_BACKGROUND_SHADER_INDEX = 3


def _cycles_shader_color_space() -> dict[str, list[list[float]]]:
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
    try:
        import PyOpenColorIO as ocio

        config = ocio.GetCurrentConfig()
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
        "xyz_to_rgb": [
            [float(component) for component in row]
            for row in xyz_to_rgb
        ],
        "rec709_to_rgb": [
            [float(component) for component in row]
            for row in rec709_to_rgb
        ],
    }


def _cycles_color_attribute_value(
    attribute: Any,
    index: int,
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float, float]:
    """Return the scene-linear value uploaded by Cycles for this domain."""

    value = tuple(float(component) for component in attribute.data[index].color)
    # Cycles preserves CORNER/BYTE_COLOR as ATTR_ELEMENT_CORNER_BYTE. Its
    # device fetch decodes sRGB and then converts linear Rec.709 into the
    # active scene-linear working space. Blender's `.color` accessor has
    # already performed the former, so apply exactly the latter here. Other
    # domains/types take Cycles' float-attribute upload path unchanged.
    if attribute.data_type != "BYTE_COLOR" or attribute.domain != "CORNER":
        return value
    rgb = value[:3]
    return tuple(
        sum(row[channel] * rgb[channel] for channel in range(3))
        for row in rec709_to_rgb
    ) + (value[3],)


def _cycles_uint_to_float(value: int) -> float:
    return float(cycles_hash.u32(value)) / float(cycles_hash.UINT32_MASK)


def _cycles_object_random_id(name: str) -> int:
    return cycles_hash.hash_uint2(cycles_hash.hash_string(name), 0)


def _cycles_particle_index(object_instance: Any) -> int:
    """Return the Particle Info index exposed by Cycles for a dupli.

    Cycles' ``BlenderSync::sync_dupli_particle`` only creates particle data
    for parent particles. Interpolated/simple child particles have a
    dependency-graph persistent ID at or above ``ParticleSystem.totpart``;
    Cycles rejects those IDs and leaves the object on particle-table entry
    zero. The SVM Particle Info node then reads that entry's index, so child
    particles use the same zero sentinel as non-particle objects.

    ``ParticleSystem.particles`` is Blender's RNA view of the parent-particle
    array and therefore supplies the same bound as ``totpart``. Keep this
    boundary explicit instead of treating every dependency-graph ID as a
    Cycles particle-table index.
    """

    particle_system = object_instance.particle_system
    if particle_system is None:
        return 0
    persistent_index = int(object_instance.persistent_id[0])
    parent_count = len(particle_system.particles)
    if persistent_index < 0 or persistent_index >= parent_count:
        return 0
    return persistent_index & cycles_hash.UINT32_MASK


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
    light: bool = False,
) -> dict[str, str] | None:
    expected = (
        "ShaderNodeOutputLight"
        if light
        else (
            "ShaderNodeOutputWorld"
            if world
            else "ShaderNodeOutputMaterial"
        )
    )
    for node in tree.nodes:
        if node.bl_idname == expected and getattr(node, "is_active_output", True):
            links = _socket_links(tree)
            source = links.get((node.name, socket))
            if source:
                return {"node": source[0], "socket": source[1]}
    return None


def _tree(
    tree: Any,
    world: bool = False,
    light: bool = False,
) -> dict[str, Any] | None:
    data = manifest._node_tree_manifest(tree)
    if data is not None:
        data["surface_root"] = _active_output(
            tree, "Surface", world, light
        )
        data["volume_root"] = (
            None
            if light
            else _active_output(tree, "Volume", world)
        )
        data["displacement_root"] = (
            None
            if world or light
            else _active_output(tree, "Displacement", world)
        )
    return data


def _node_tree_uses_pointiness(
    tree: Any,
    visited: set[int] | None = None,
) -> bool:
    """Match Cycles' linked Geometry.Pointiness attribute request.

    The test is intentionally structural. It decides whether to retain the
    evaluated point normals and original edges in the scene contract; it does
    not evaluate or rewrite any part of the shader graph.
    """

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
        if nested is not None and _node_tree_uses_pointiness(
            nested, visited
        ):
            return True
    return False


_POINTINESS_MATERIAL_CACHE: dict[int, bool] = {}


def _material_uses_pointiness(material: Any) -> bool:
    if material is None:
        return False
    identity = int(material.as_pointer())
    cached = _POINTINESS_MATERIAL_CACHE.get(identity)
    if cached is not None:
        return cached
    result = bool(
        material is not None
        and material.use_nodes
        and _node_tree_uses_pointiness(material.node_tree)
    )
    _POINTINESS_MATERIAL_CACHE[identity] = result
    return result


def _write_array(stream: Any, values: array.array[Any]) -> dict[str, int]:
    if sys.byteorder != "little":
        values.byteswap()
    offset = stream.tell()
    # Do not mix ``array.tofile`` with writes to Python's buffered stream.
    # Some Blender/Python builds let ``tofile`` bypass the buffered writer;
    # the logical file position then advances while the last physical write
    # is absent when the section table is validated.  A byte memoryview keeps
    # every write on the same stream path without copying large meshes.
    payload = memoryview(values).cast("B")
    written = stream.write(payload)
    if written != len(payload):
        raise OSError(
            f"short geometry write: expected {len(payload)} bytes, "
            f"wrote {written}"
        )
    return {"offset": offset, "bytes": written}


def _geometry_cache_key(obj: Any, scene: Any) -> tuple[Any, ...]:
    # Reusing an evaluated mesh is only safe when Blender reports that the
    # render geometry is identical to the source Mesh datablock. Modifier
    # evaluation can depend on object-local settings and referenced objects:
    # Lone Monk's arch.005--arch.008 share one Mesh datablock, but their Mirror
    # modifiers produce four different render meshes.
    if (
        obj.type == "MESH"
        and obj.data is not None
        and not obj.is_modified(scene, "RENDER")
    ):
        material_slots = tuple(
            slot.material.as_pointer() if slot.material is not None else 0
            for slot in obj.material_slots
        )
        return ("mesh", obj.data.as_pointer(), material_slots)
    return ("object", obj.as_pointer())


def _cycles_uses_adaptive_subdivision(obj: Any) -> bool:
    """Match Cycles' render-time object_subdivision_type gate."""

    modifiers = list(obj.modifiers)
    if not modifiers:
        return False
    modifier = modifiers[-1]
    return bool(
        modifier.type == "SUBSURF"
        and modifier.show_render
        and getattr(modifier, "use_adaptive_subdivision", False)
    )


def _geometry(
    obj: Any,
    depsgraph: Any,
    stream: Any,
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
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
        needs_pointiness = any(
            _material_uses_pointiness(slot.material)
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
        uv_layer_names = [layer.name for layer in mesh.uv_layers]
        # Cycles maps ATTR_STD_UV to Mesh::default_uv_map_name(), exposed
        # through Blender's active_render UV layer. The UI/editing-active
        # layer is an independent choice and may intentionally differ.
        default_uv_name = (
            mesh.uv_layers.active_render.name
            if mesh.uv_layers.active_render is not None
            else None
        )
        uv_by_layer: dict[str, list[tuple[float, float]]] = {}
        tangent_by_layer: dict[
            str, list[tuple[float, float, float, float]]
        ] = {}
        for uv_layer_name in uv_layer_names:
            has_tangents = False
            try:
                mesh.calc_tangents(uvmap=uv_layer_name)
                has_tangents = True
            except RuntimeError:
                # Match Cycles' missing tangent-attribute behavior. A zero
                # tangent is consumed as "use sd->N" by Normal Map.
                pass
            # calc_tangents may reallocate loop CustomData. Reacquire the
            # layer by name after every call before copying its values.
            uv_layer = mesh.uv_layers.get(uv_layer_name)
            if uv_layer is None:
                continue
            uv_by_layer[uv_layer_name] = [
                tuple(float(value) for value in item.uv)
                for item in uv_layer.data
            ]
            tangent_by_layer[uv_layer_name] = [
                (
                    float(loop.tangent[0]),
                    float(loop.tangent[1]),
                    float(loop.tangent[2]),
                    float(loop.bitangent_sign),
                )
                if has_tangents
                else (0.0, 0.0, 0.0, 0.0)
                for loop in mesh.loops
            ]
        positions = array.array("f")
        normals = array.array("f")
        uvs = array.array("f")
        uv_tangents = array.array("f")
        uv_layer_values = {
            name: array.array("f") for name in uv_by_layer
        }
        uv_tangent_values = {
            name: array.array("f") for name in uv_by_layer
        }
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
            generated_scale[0],
            0.0,
            0.0,
            0.0,
            0.0,
            generated_scale[1],
            0.0,
            0.0,
            0.0,
            0.0,
            generated_scale[2],
            0.0,
            -generated_location[0],
            -generated_location[1],
            -generated_location[2],
            1.0,
        ]
        generated_by_vertex = [
            tuple(
                float(vertex.co[axis]) * generated_scale[axis]
                - generated_location[axis]
                for axis in range(3)
            )
            for vertex in mesh.vertices
        ]
        positions.extend(
            component
            for vertex in mesh.vertices
            for component in vertex.co
        )
        generated.extend(
            component
            for value in generated_by_vertex
            for component in value
        )
        blender_normal_domain = str(
            getattr(mesh, "normals_domain", "CORNER")
        )
        use_corner_normals = blender_normal_domain == "CORNER"
        # Match Cycles create_mesh() and Mesh::pack_shaders(), rather than
        # MeshPolygon.use_smooth. The latter only reflects sharp_face and can
        # still report true when sharp edges make Blender's effective normal
        # domain FACE. Conversely, Cycles always enables interpolated shading
        # when corner normals exist because flatness is already encoded in the
        # three corner values.
        sharp_face_attribute = mesh.attributes.get("sharp_face")
        if use_corner_normals:
            smooth_by_polygon = None
            smooth_without_attribute = True
        elif sharp_face_attribute is not None:
            smooth_by_polygon = tuple(
                not bool(item.value)
                for item in sharp_face_attribute.data
            )
            smooth_without_attribute = False
        else:
            smooth_by_polygon = None
            smooth_without_attribute = blender_normal_domain != "FACE"
        normal_domain = "CORNER" if use_corner_normals else "POINT"
        if normal_domain == "POINT":
            normals.extend(
                component
                for vertex in mesh.vertices
                for component in vertex.normal
            )
        for attribute in color_layers:
            if attribute.domain != "POINT":
                continue
            for vertex in mesh.vertices:
                color_values[attribute.name].extend(
                    _cycles_color_attribute_value(
                        attribute,
                        vertex.index,
                        rec709_to_rgb,
                    )
                )

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

        for triangle in mesh.loop_triangles:
            for loop_index in triangle.loops:
                loop = mesh.loops[loop_index]
                if normal_domain == "CORNER":
                    normals.extend(float(value) for value in loop.normal)
                for attribute in color_layers:
                    if attribute.domain != "CORNER":
                        continue
                    color_values[attribute.name].extend(
                        _cycles_color_attribute_value(
                            attribute,
                            loop_index,
                            rec709_to_rgb,
                        )
                    )
                if default_uv_name not in uv_by_layer:
                    uvs.extend((0.0, 0.0))
                else:
                    uvs.extend(
                        uv_by_layer[default_uv_name][loop_index]
                    )
                if default_uv_name not in tangent_by_layer:
                    uv_tangents.extend((0.0, 0.0, 0.0, 0.0))
                else:
                    uv_tangents.extend(
                        tangent_by_layer[default_uv_name][loop_index]
                    )
                for layer_name, values in uv_by_layer.items():
                    uv_layer_values[layer_name].extend(
                        values[loop_index]
                    )
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
                _cycles_uint_to_float(
                    cycles_hash.hash_uint(find_root(first_vertex))
                )
            )
        pointiness_source = None
        if needs_pointiness:
            pointiness_source = {
                "point_normals": _write_array(
                    stream, pointiness_normals
                ),
                "edge_count": len(pointiness_edges) // 2,
                "edges": _write_array(stream, pointiness_edges),
            }
        return {
            "name": obj.name,
            "uses_adaptive_subdivision": (
                _cycles_uses_adaptive_subdivision(obj)
            ),
            "point_count": len(mesh.vertices),
            "corner_count": len(mesh.loop_triangles) * 3,
            "triangle_count": len(mesh.loop_triangles),
            "positions": _write_array(stream, positions),
            "normal_domain": normal_domain,
            "normals": _write_array(stream, normals),
            "uv_domain": "CORNER",
            "default_uv_available": default_uv_name in uv_by_layer,
            "uv": _write_array(stream, uvs),
            "uv_tangent_domain": "CORNER",
            "uv_tangents": _write_array(stream, uv_tangents),
            "uv_layers": [
                {
                    "name": name,
                    "domain": "CORNER",
                    "values": _write_array(
                        stream, uv_layer_values[name]
                    ),
                    "tangents": _write_array(
                        stream, uv_tangent_values[name]
                    ),
                }
                for name in uv_layer_values
            ],
            "generated_domain": "POINT",
            "generated": _write_array(stream, generated),
            # Cycles' ATTR_STD_GENERATED_TRANSFORM. Volume shading points
            # use this object-space affine transform directly because there
            # is no surface primitive whose Generated values can be
            # interpolated.
            "generated_transform": generated_transform,
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
            "pointiness_source": pointiness_source,
            "indices": _write_array(stream, indices),
            "triangle_material_slots": _write_array(stream, materials),
            "triangle_smooth": _write_array(stream, smooth),
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


def _particle_hair_systems(evaluated: Any) -> list[tuple[Any, Any]]:
    """Return the final-render legacy hair systems consumed by Cycles.

    Cycles walks evaluated particle-system modifiers, checks the modifier's
    render bit, and accepts only HAIR/PATH systems. Keeping this predicate in
    one place makes object-index assignment and geometry extraction agree.
    """

    systems: list[tuple[Any, Any]] = []
    for modifier in evaluated.modifiers:
        if modifier.type != "PARTICLE_SYSTEM" or not modifier.show_render:
            continue
        particle_system = getattr(modifier, "particle_system", None)
        if particle_system is None:
            continue
        settings = particle_system.settings
        if settings.type == "HAIR" and settings.render_type == "PATH":
            systems.append((particle_system, settings))
    return systems


def _object_has_particle_hair(obj: Any, depsgraph: Any) -> bool:
    if obj.type != "MESH":
        return False
    return bool(_particle_hair_systems(obj.evaluated_get(depsgraph)))


def _cycles_shape_radius(
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


def _particle_hair_geometry(
    obj: Any,
    depsgraph: Any,
    stream: Any,
    shape: str,
    subdivisions: int,
) -> dict[str, Any] | None:
    """Extract Cycles legacy particle hair without tessellating or baking it."""

    evaluated = obj.evaluated_get(depsgraph)
    systems = _particle_hair_systems(evaluated)
    if not systems:
        return None

    object_to_world = np.asarray(evaluated.matrix_world, dtype=np.float32)
    world_to_object = np.linalg.inv(object_to_world).astype(
        np.float32, copy=False
    )
    keys = array.array("f")
    curve_first_key = array.array("I")
    curve_material_slots = array.array("I")
    intercepts = array.array("f")
    lengths = array.array("f")
    randoms = array.array("f")
    segment_count = 0
    curve_index = 0
    material_slot_count = max(len(obj.material_slots), 1)

    for particle_system, settings in systems:
        parent_count = len(particle_system.particles)
        child_count = len(particle_system.child_particles)
        first_particle = (
            parent_count
            if settings.child_type != "NONE" and child_count != 0
            else 0
        )
        particle_end = parent_count + child_count
        key_count = (1 << int(settings.render_step)) + 1
        if settings.kink == "SPIRAL":
            key_count += int(settings.kink_extra_steps)
        if particle_end <= first_particle or key_count < 2:
            continue

        base_radius = np.float32(
            np.float32(settings.radius_scale) * np.float32(0.5)
        )
        root_radius = np.float32(
            base_radius * np.float32(settings.root_radius)
        )
        tip_radius = np.float32(
            base_radius * np.float32(settings.tip_radius)
        )
        radius_shape = np.float32(settings.shape)
        close_tip = bool(settings.use_close_tip)
        material_slot = min(
            max(int(settings.material) - 1, 0),
            material_slot_count - 1,
        )

        for particle_no in range(first_particle, particle_end):
            curve_first_key.append(len(keys) // 4)
            positions: list[np.ndarray[Any, np.dtype[np.float32]]] = []
            curve_times: list[np.float32] = []
            curve_length = np.float32(0.0)
            previous: np.ndarray[Any, np.dtype[np.float32]] | None = None
            for step in range(key_count):
                coordinate = particle_system.co_hair(
                    object=evaluated,
                    particle_no=particle_no,
                    step=step,
                )
                world = np.asarray(
                    (
                        float(coordinate[0]),
                        float(coordinate[1]),
                        float(coordinate[2]),
                        1.0,
                    ),
                    dtype=np.float32,
                )
                local = np.asarray(
                    (world_to_object @ world)[:3],
                    dtype=np.float32,
                )
                if previous is not None:
                    delta = np.asarray(local - previous, dtype=np.float32)
                    squared_length = np.float32(np.dot(delta, delta))
                    curve_length = np.float32(
                        curve_length
                        + np.float32(np.sqrt(squared_length))
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
                    radius_shape,
                    root_radius,
                    tip_radius,
                    intercept,
                )
                if close_tip and key + 1 == key_count:
                    radius = np.float32(0.0)
                keys.extend(
                    (
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        float(radius),
                    )
                )
                intercepts.append(float(intercept))

            curve_material_slots.append(material_slot)
            lengths.append(float(curve_length))
            randoms.append(
                _cycles_uint_to_float(
                    cycles_hash.hash_uint2(curve_index, 0)
                )
            )
            segment_count += key_count - 1
            curve_index += 1

    if curve_index == 0:
        return None
    return {
        "name": f"{obj.name}.particle_hair",
        "shape": {
            "RIBBONS": "RIBBON",
            "THICK": "THICK",
            "THICK_LINEAR": "THICK_LINEAR",
        }[shape],
        "subdivisions": int(subdivisions),
        "key_count": len(keys) // 4,
        "curve_count": curve_index,
        "segment_count": segment_count,
        "keys": _write_array(stream, keys),
        "curve_first_key": _write_array(stream, curve_first_key),
        "curve_material_slots": _write_array(
            stream, curve_material_slots
        ),
        "intercept": _write_array(stream, intercepts),
        "length": _write_array(stream, lengths),
        "random": _write_array(stream, randoms),
        "material_slots": [
            slot.material.name if slot.material else None
            for slot in obj.material_slots
        ],
    }


def _image_extension(image: Any) -> str:
    if image.source == "GENERATED":
        return ".png"
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


def _external_image_path(image: Any) -> pathlib.Path:
    # Linked image datablocks retain paths relative to the library .blend
    # that owns them, not the currently open main file. This distinction is
    # observable in Blender's official Classroom scene, whose linked assets
    # use paths such as ``//../../textures/_baseTextures/...``. Delegate the
    # base selection to Blender so export and Cycles resolve the same file.
    return pathlib.Path(
        bpy.path.abspath(
            image.filepath,
            library=image.library,
        )
    )


def _save_generated_image(
    image: Any,
    destination: pathlib.Path,
) -> None:
    # GENERATED images have no backing file, but their pixel buffer is a real
    # Cycles texture input. Save a temporary datablock copy through Blender's
    # image codec so the original filepath/format and source scene remain
    # untouched. ``Image.save`` applies the datablock color-space encoding;
    # Psycles decodes that declared space when sampling the exported texture.
    encoded = image.copy()
    try:
        pixels = array.array("f", [0.0]) * len(image.pixels)
        image.pixels.foreach_get(pixels)
        encoded.pixels.foreach_set(pixels)
        encoded.update()
        encoded.filepath_raw = str(destination)
        encoded.file_format = "PNG"
        encoded.save()
    finally:
        bpy.data.images.remove(encoded)


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
        elif image.source == "GENERATED":
            _save_generated_image(image, destination)
        else:
            source = _external_image_path(image)
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


def _original_id(value: Any) -> Any:
    original = getattr(value, "original", None)
    return original if original is not None else value


def _id_key(value: Any) -> int:
    return int(_original_id(value).as_pointer())


def _cycles_shader_indices(
    depsgraph: Any,
) -> tuple[dict[int, int], dict[int, int]]:
    """Reproduce the Cycles scene shader-vector insertion order.

    ShaderManager::add_default inserts five shaders. BlenderSync then reuses
    the default background shader, inserts one shader per dependency-graph
    Light ID, and finally one per dependency-graph Material ID. The exported
    material array may remain name-sorted for stable JSON; source shader
    identity must not be reconstructed from that array order.
    """

    next_index = _CYCLES_DEFAULT_SHADER_COUNT
    light_indices: dict[int, int] = {}
    material_indices: dict[int, int] = {}
    for expected_type, indices in (
        (bpy.types.Light, light_indices),
        (bpy.types.Material, material_indices),
    ):
        for dependency_id in depsgraph.ids:
            original = _original_id(dependency_id)
            if not isinstance(original, expected_type):
                continue
            key = _id_key(original)
            if key in indices:
                continue
            indices[key] = next_index
            next_index += 1
    return light_indices, material_indices


def _cycles_object_is_geometry(object_instance: Any) -> bool:
    if not object_instance.show_self:
        return False
    obj = object_instance.object
    if obj.data is None:
        return False
    # BlenderSync::object_is_geometry accepts ordinary surface geometry only
    # when the evaluated object-data ID is a Mesh. Legacy Curve/Surface/Font
    # path objects can still be converted by Object.to_mesh(), but Cycles does
    # not create an Object for them while their evaluated data remains Curve.
    # Counting those objects shifts every later source object identity.
    is_cycles_geometry = (
        isinstance(obj.data, bpy.types.Mesh)
        or obj.type in {"VOLUME", "CURVES", "POINTCLOUD", "LIGHT"}
    )
    return is_cycles_geometry and any(
        _instance_ray_visibility(object_instance).values()
    )


def _cycles_light_group(
    object_instance: Any,
    light_groups: dict[str, int],
) -> int:
    obj = object_instance.object
    name = str(getattr(obj, "lightgroup", ""))
    if not name and object_instance.is_instance:
        parent = object_instance.parent
        if parent is not None and parent != obj:
            name = str(getattr(parent, "lightgroup", ""))
    return light_groups.get(name, -1)


def _cycles_shadow_catcher(object_instance: Any) -> bool:
    obj = object_instance.object
    result = bool(getattr(obj, "is_shadow_catcher", False))
    if object_instance.is_instance:
        parent = object_instance.parent
        if parent is not None and parent != obj:
            result = result or bool(
                getattr(parent, "is_shadow_catcher", False)
            )
    return result


def _light(
    obj: Any,
    *,
    transform: Any | None = None,
    visibility: dict[str, bool] | None = None,
    is_shadow_catcher: bool = False,
    cycles_sync: dict[str, int] | None = None,
) -> dict[str, Any]:
    light = obj.data
    cycles = getattr(light, "cycles", None)
    use_temperature = bool(getattr(light, "use_temperature", False))
    temperature_color = (
        list(light.temperature_color)
        if use_temperature
        else [1.0, 1.0, 1.0]
    )
    size = (
        float(light.shadow_soft_size)
        if light.type in {"POINT", "SPOT"}
        else float(getattr(light, "size", 0.0))
    )
    return {
        "name": obj.name,
        "type": light.type,
        "transform": _column_major(
            obj.matrix_world if transform is None else transform
        ),
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
        "use_soft_falloff": bool(
            getattr(light, "use_soft_falloff", False)
        ),
        "use_multiple_importance_sampling": bool(
            getattr(cycles, "use_multiple_importance_sampling", True)
        ),
        "cast_shadow": bool(getattr(light, "use_shadow", True)),
        "visibility": (
            visibility
            if visibility is not None
            else {
                "camera": bool(getattr(obj, "visible_camera", True)),
                "diffuse": bool(getattr(obj, "visible_diffuse", True)),
                "glossy": bool(getattr(obj, "visible_glossy", True)),
                "transmission": bool(
                    getattr(obj, "visible_transmission", True)
                ),
                "shadow": bool(getattr(obj, "visible_shadow", True)),
                "volume_scatter": bool(
                    getattr(obj, "visible_volume_scatter", True)
                ),
            }
        ),
        "is_shadow_catcher": is_shadow_catcher,
        "cycles_sync": cycles_sync,
        "node_tree": (
            _tree(light.node_tree, light=True)
            if light.use_nodes
            else None
        ),
    }


def _instance_ray_visibility(
    object_instance: Any,
) -> dict[str, bool]:
    """Match Cycles' child-and-instancer ray visibility intersection."""

    obj = object_instance.object
    parent = (
        object_instance.parent
        if object_instance.is_instance
        else None
    )

    def visible(property_name: str) -> bool:
        result = bool(getattr(obj, property_name))
        if parent is not None and parent != obj:
            result = result and bool(
                getattr(parent, property_name)
            )
        return result

    return {
        "camera": visible("visible_camera"),
        "diffuse": visible("visible_diffuse"),
        "glossy": visible("visible_glossy"),
        "transmission": visible("visible_transmission"),
        "shadow": visible("visible_shadow"),
        "volume_scatter": visible("visible_volume_scatter"),
    }


def _geometry_instance(
    object_instance: Any,
    original: Any,
    geometry_type: str,
    geometry_index: int,
    cycles_object_index: int,
    light_groups: dict[str, int],
) -> dict[str, Any]:
    return {
        "name": object_instance.object.name,
        "geometry_type": geometry_type,
        "geometry": geometry_index,
        "transform": _column_major(object_instance.matrix_world),
        "persistent_id": list(object_instance.persistent_id),
        "is_instance": bool(object_instance.is_instance),
        # Match Cycles' parent-only particle table. Child particles retain
        # the zero sentinel for both surface and hair objects.
        "particle_index": _cycles_particle_index(object_instance),
        # Cycles uses the dependency-graph random id for duplis and hashes
        # the object name for ordinary objects. Its separate hair object
        # intentionally receives the same random id.
        "random_id": int(
            object_instance.random_id
            if object_instance.is_instance
            else _cycles_object_random_id(object_instance.object.name)
        )
        & cycles_hash.UINT32_MASK,
        "shadow_terminator_geometry_offset": float(
            original.cycles.shadow_terminator_geometry_offset
        ),
        "visibility": _instance_ray_visibility(object_instance),
        "is_shadow_catcher": _cycles_shadow_catcher(object_instance),
        "cycles_sync": {
            "object_index": cycles_object_index,
            "light_group": _cycles_light_group(
                object_instance, light_groups
            ),
        },
    }


def _export_scene(
    output: pathlib.Path,
    depsgraph: Any,
    source_camera: Any | None,
) -> None:
    scene = bpy.context.scene
    if depsgraph.mode != "RENDER":
        raise RuntimeError(
            "Psycles scene export requires Blender's RENDER dependency "
            f"graph, received {depsgraph.mode!r}"
        )

    shader_color_space = _cycles_shader_color_space()
    rec709_to_rgb = tuple(
        tuple(float(component) for component in row)
        for row in shader_color_space["rec709_to_rgb"]
    )
    geometries: list[dict[str, Any]] = []
    geometry_by_source: dict[tuple[Any, ...], int] = {}
    curve_geometries: list[dict[str, Any]] = []
    curve_geometry_by_source: dict[tuple[Any, ...], int] = {}
    instances: list[dict[str, Any]] = []
    lights: list[dict[str, Any]] = []
    light_shader_indices, material_shader_indices = (
        _cycles_shader_indices(depsgraph)
    )
    light_groups = {
        group.name: index
        for index, group in enumerate(bpy.context.view_layer.lightgroups)
    }
    cycles_object_index = 0
    cycles_primitive_offset = 0
    cycles_curve_offset = 0
    cycles_curve_segment_offset = 0
    curve_shape = str(scene.cycles_curves.shape)
    curve_subdivisions = int(scene.cycles_curves.subdivisions)
    with (output / "geometry.bin").open("wb") as stream:
        stream.write(b"PSYGEO2\0")
        stream.write(struct.pack("<II", 2, 0))
        for object_instance in depsgraph.object_instances:
            obj = object_instance.object
            original = obj.original if obj.original is not None else obj
            source_object_index: int | None = None
            if _cycles_object_is_geometry(object_instance):
                source_object_index = cycles_object_index
                cycles_object_index += 1
            hair_object_index: int | None = None
            if (
                bool(object_instance.show_particles)
                and _object_has_particle_hair(original, depsgraph)
            ):
                hair_object_index = cycles_object_index
                cycles_object_index += 1

            if original.hide_render:
                continue
            if obj.type == "LIGHT":
                if (
                    not object_instance.show_self
                    or source_object_index is None
                ):
                    continue
                light_data = _original_id(obj.data)
                light_shader_index = light_shader_indices.get(
                    _id_key(light_data)
                )
                if light_shader_index is None:
                    raise RuntimeError(
                        "Cycles dependency graph did not expose the shader "
                        f"identity for light data {light_data.name!r}"
                    )
                lights.append(
                    _light(
                        obj,
                        transform=object_instance.matrix_world,
                        visibility=_instance_ray_visibility(
                            object_instance
                        ),
                        is_shadow_catcher=_cycles_shadow_catcher(
                            object_instance
                        ),
                        cycles_sync={
                            "object_index": source_object_index,
                            "light_group": _cycles_light_group(
                                object_instance, light_groups
                            ),
                            "shader_index": light_shader_index,
                        },
                    )
                )
                continue
            if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
                continue
            # Cycles syncs the emitter surface first, then its independent
            # hair object. OB_VISIBLE_SELF gates only the former.
            if (
                object_instance.show_self
                and source_object_index is not None
            ):
                key = _geometry_cache_key(original, scene)
                geometry_index = geometry_by_source.get(key)
                if geometry_index is None:
                    geometry_data = _geometry(
                        original,
                        depsgraph,
                        stream,
                        rec709_to_rgb,
                    )
                    if geometry_data is not None:
                        geometry_data["cycles_sync"] = {
                            "primitive_offset": cycles_primitive_offset,
                        }
                        cycles_primitive_offset += int(
                            geometry_data["triangle_count"]
                        )
                        geometry_index = len(geometries)
                        geometry_by_source[key] = geometry_index
                        geometries.append(geometry_data)
                if geometry_index is not None:
                    instances.append(
                        _geometry_instance(
                            object_instance,
                            original,
                            "MESH",
                            geometry_index,
                            source_object_index,
                            light_groups,
                        )
                    )

            if hair_object_index is not None:
                hair_key = (
                    "particle_hair",
                    original.as_pointer(),
                    curve_shape,
                    curve_subdivisions,
                )
                curve_geometry_index = curve_geometry_by_source.get(
                    hair_key
                )
                if curve_geometry_index is None:
                    curve_geometry = _particle_hair_geometry(
                        original,
                        depsgraph,
                        stream,
                        curve_shape,
                        curve_subdivisions,
                    )
                    if curve_geometry is not None:
                        curve_geometry["cycles_sync"] = {
                            "curve_offset": cycles_curve_offset,
                            "segment_offset": (
                                cycles_curve_segment_offset
                            ),
                        }
                        cycles_curve_offset += int(
                            curve_geometry["curve_count"]
                        )
                        cycles_curve_segment_offset += int(
                            curve_geometry["segment_count"]
                        )
                        curve_geometry_index = len(curve_geometries)
                        curve_geometry_by_source[hair_key] = (
                            curve_geometry_index
                        )
                        curve_geometries.append(curve_geometry)
                if curve_geometry_index is not None:
                    instances.append(
                        _geometry_instance(
                            object_instance,
                            original,
                            "CURVE",
                            curve_geometry_index,
                            hair_object_index,
                            light_groups,
                        )
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
        for uv_layer in geometry["uv_layers"]:
            sections.append(uv_layer["values"])
            sections.append(uv_layer["tangents"])
        pointiness_source = geometry.get("pointiness_source")
        if pointiness_source is not None:
            sections.append(pointiness_source["point_normals"])
            sections.append(pointiness_source["edges"])
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
    for geometry in curve_geometries:
        for name in (
            "keys",
            "curve_first_key",
            "curve_material_slots",
            "intercept",
            "length",
            "random",
        ):
            section = geometry[name]
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
        cycles = getattr(material, "cycles", None)
        shader_index = material_shader_indices.get(_id_key(material))
        materials.append(
            {
                "name": material.name,
                "use_bump_map_correction": bool(
                    getattr(cycles, "use_bump_map_correction", True)
                ),
                "emission_sampling": str(
                    getattr(cycles, "emission_sampling", "AUTO")
                ),
                "volume_sampling": str(
                    getattr(
                        cycles,
                        "volume_sampling",
                        "MULTIPLE_IMPORTANCE",
                    )
                ),
                "surface_render_method": getattr(
                    material, "surface_render_method", None
                ),
                # Blender 4.1 moved this setting from the Cycles custom
                # properties onto Material itself. Keep the older location
                # as a compatibility fallback for supported Blender builds.
                "displacement_method": str(
                    getattr(
                        material,
                        "displacement_method",
                        getattr(cycles, "displacement_method", "BUMP"),
                    )
                ),
                "cycles_sync": (
                    {"shader_index": shader_index}
                    if shader_index is not None
                    else None
                ),
                "node_tree": (
                    _tree(material.node_tree)
                    if material.use_nodes and material.node_tree
                    else None
                ),
            }
        )

    camera = source_camera
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

    pixel_filter_type = str(scene.cycles.pixel_filter_type)
    # Cycles hard-codes a one-pixel support for BOX and does not synchronize
    # the UI filter_width property in that mode. Export the effective render
    # contract rather than a dormant property value.
    filter_width = (
        1.0
        if pixel_filter_type == "BOX"
        else float(scene.cycles.filter_width)
    )

    payload = {
        "schema": "psycles.blender-scene.v2",
        "source": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "frame": scene.frame_current,
        "camera": camera_data,
        "render": {
            "width": scene.render.resolution_x,
            "height": scene.render.resolution_y,
            "percentage": scene.render.resolution_percentage,
            "transparent": scene.render.film_transparent,
            "pixel_filter_type": pixel_filter_type,
            "filter_width": filter_width,
            "pass_alpha_threshold": float(
                bpy.context.view_layer.pass_alpha_threshold
            ),
            "color_management": {
                "display_device": scene.display_settings.display_device,
                "view_transform": scene.view_settings.view_transform,
                "look": scene.view_settings.look,
                "exposure": float(scene.view_settings.exposure),
                "gamma": float(scene.view_settings.gamma),
                "use_curve_mapping": bool(
                    scene.view_settings.use_curve_mapping
                ),
                "sequencer_color_space": (
                    scene.sequencer_colorspace_settings.name
                ),
                "shader_transforms": shader_color_space,
            },
            "cycles": manifest._cycles_settings(scene),
        },
        "geometries": geometries,
        "curve_geometries": curve_geometries,
        "instances": instances,
        "materials": materials,
        "lights": lights,
        "world": (
            {
                "name": scene.world.name,
                "color": list(scene.world.color),
                "sampling_method": str(
                    getattr(
                        getattr(scene.world, "cycles", None),
                        "sampling_method",
                        "AUTOMATIC",
                    )
                ),
                "sample_map_resolution": int(
                    getattr(
                        getattr(scene.world, "cycles", None),
                        "sample_map_resolution",
                        1024,
                    )
                ),
                "use_shadows": bool(
                    getattr(scene.world.cycles, "use_shadows", True)
                ),
                "visibility": {
                    "camera": bool(
                        getattr(
                            scene.world.cycles_visibility,
                            "camera",
                            True,
                        )
                    ),
                    "diffuse": bool(
                        getattr(
                            scene.world.cycles_visibility,
                            "diffuse",
                            True,
                        )
                    ),
                    "glossy": bool(
                        getattr(
                            scene.world.cycles_visibility,
                            "glossy",
                            True,
                        )
                    ),
                    "transmission": bool(
                        getattr(
                            scene.world.cycles_visibility,
                            "transmission",
                            True,
                        )
                    ),
                    "shadow": bool(
                        getattr(
                            scene.world.cycles_visibility,
                            "shadow",
                            True,
                        )
                    ),
                    "volume_scatter": bool(
                        getattr(
                            scene.world.cycles_visibility,
                            "scatter",
                            True,
                        )
                    ),
                },
                "cycles_sync": {
                    "shader_index": _CYCLES_BACKGROUND_SHADER_INDEX,
                    # BlenderSync creates the background-light object after
                    # dependency-graph geometry and analytic lights.
                    "object_index": cycles_object_index,
                    "light_group": light_groups.get(
                        str(getattr(scene.world, "lightgroup", "")),
                        -1,
                    ),
                },
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
        f"Exported {len(geometries)} meshes, "
        f"{len(curve_geometries)} curve geometries, "
        f"{len(instances)} "
        f"instances, {len(materials)} materials, "
        f"{len(payload['images'])} images to {output}"
    )


_EXPORT_OUTPUT: pathlib.Path | None = None
_EXPORT_CAMERA: Any | None = None
_EXPORT_COMPLETED = False
_EXPORT_ERROR: BaseException | None = None


class _PsyclesExportRenderEngine(bpy.types.RenderEngine):
    """Obtain the same final-render dependency graph consumed by Cycles."""

    bl_idname = "PSYCLES_SCENE_EXPORT"
    bl_label = "Psycles Scene Export"
    bl_use_preview = False

    def render(self, depsgraph: Any) -> None:
        global _EXPORT_COMPLETED, _EXPORT_ERROR
        try:
            if _EXPORT_OUTPUT is None:
                raise RuntimeError("Psycles export output was not initialized")
            _export_scene(
                _EXPORT_OUTPUT,
                depsgraph,
                _EXPORT_CAMERA,
            )
            _EXPORT_COMPLETED = True
        except BaseException as error:
            _EXPORT_ERROR = error
            traceback.print_exc()
            raise


def _main() -> None:
    global _EXPORT_OUTPUT, _EXPORT_CAMERA
    global _EXPORT_COMPLETED, _EXPORT_ERROR

    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected one output directory after '--'")
    output = pathlib.Path(args[0]).resolve()
    output.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    original_engine = scene.render.engine
    original_camera = scene.camera
    temporary_camera = None
    _EXPORT_OUTPUT = output
    _EXPORT_CAMERA = original_camera
    _EXPORT_COMPLETED = False
    _EXPORT_ERROR = None

    # Blender's final-render operator requires a camera before it constructs
    # the RENDER dependency graph. Preserve camera-less scene semantics in the
    # payload by keeping the original camera separately and removing this
    # temporary object immediately after export.
    if original_camera is None:
        camera_data = bpy.data.cameras.new(
            "__Psycles Export Temporary Camera"
        )
        temporary_camera = bpy.data.objects.new(
            "__Psycles Export Temporary Camera",
            camera_data,
        )
        scene.collection.objects.link(temporary_camera)
        scene.camera = temporary_camera

    bpy.utils.register_class(_PsyclesExportRenderEngine)
    try:
        scene.render.engine = _PsyclesExportRenderEngine.bl_idname
        bpy.ops.render.render()
    finally:
        scene.render.engine = original_engine
        scene.camera = original_camera
        if temporary_camera is not None:
            camera_data = temporary_camera.data
            bpy.data.objects.remove(temporary_camera, do_unlink=True)
            bpy.data.cameras.remove(camera_data)
        bpy.utils.unregister_class(_PsyclesExportRenderEngine)

    if _EXPORT_ERROR is not None:
        raise _EXPORT_ERROR
    if not _EXPORT_COMPLETED:
        raise RuntimeError(
            "Blender did not invoke the Psycles final-render export engine"
        )


if __name__ == "__main__":
    _main()
