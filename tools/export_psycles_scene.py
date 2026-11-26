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
import json
import pathlib
import struct
import sys
import traceback
from typing import Any

import bpy
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import blender_scene_manifest as manifest  # noqa: E402
import blender_build_identity  # noqa: E402
from blender_image_export import export_images as _export_images  # noqa: E402
from blender_mesh_attributes import (  # noqa: E402
    collect_required_geometry_attributes as _collect_required_geometry_attributes,
    collect_required_uv_map_names as _collect_required_uv_map_names,
    extract_generic_mesh_attributes as _extract_generic_mesh_attributes,
    resolved_material_slots as _resolved_material_slots,
)
from blender_cycles_object import (  # noqa: E402
    ParticleSourceRegistry as _ParticleSourceRegistry,
    light_group as _cycles_light_group,
    object_ao_distance as _cycles_object_ao_distance,
    object_properties as _cycles_object_properties,
    object_random_id as _cycles_object_random_id,
    particle_parent_index as _cycles_particle_index,
    shadow_catcher as _cycles_shadow_catcher,
    uses_light_linking as _cycles_object_uses_light_linking,
)
from blender_particle_hair import (  # noqa: E402
    cycles_particle_hair_color as _cycles_particle_hair_color,
    cycles_particle_hair_color_value as _cycles_particle_hair_color_value,
    cycles_particle_hair_positions as _cycles_particle_hair_positions,
    cycles_particle_hair_uv as _cycles_particle_hair_uv,
    cycles_shape_radius as _cycles_shape_radius,
    material_named_color_attributes as _material_named_color_attributes,
    particle_hair_systems as _particle_hair_systems,
)
import cycles_hash  # noqa: E402
import exporter_identity  # noqa: E402


_CYCLES_DEFAULT_SHADER_COUNT = 5
_CYCLES_BACKGROUND_SHADER_INDEX = 3
_EXPORTER_IDENTITY = exporter_identity.current(pathlib.Path(__file__))


def _cycles_shader_color_space() -> dict[str, Any]:
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


def _cycles_corner_byte_color_value(
    attribute: Any,
    index: int,
) -> tuple[int, int, int, int]:
    """Return the exact uchar4 payload retained by Cycles.

    Blender's ``color_srgb`` accessor is the normalized view of the native
    BYTE_COLOR storage. Quantizing it back to 8 bits is lossless and avoids
    attempting an inverse transfer from the separate scene-linear legacy
    projection.
    """

    return tuple(
        max(0, min(255, int(float(component) * 255.0 + 0.5)))
        for component in attribute.data[index].color_srgb
    )


def _cycles_uint_to_float(value: int) -> float:
    return float(cycles_hash.u32(value)) / float(cycles_hash.UINT32_MASK)


def _column_major(matrix: Any) -> list[float]:
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


def _socket_links(tree: Any) -> dict[tuple[str, str], tuple[str, str]]:
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for link in tree.links:
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
        else (
            "ShaderNodeOutputWorld"
            if world
            else "ShaderNodeOutputMaterial"
        )
    )
    # This is the same target-aware selection primitive used by Cycles through
    # ntreeShaderOutputNode(..., SHD_OUTPUT_CYCLES): an exact CYCLES target is
    # preferred over an ALL target, and is_active_output only breaks ties
    # between nodes with the same target. Looking for the globally active node
    # instead makes versioned files silently export their EEVEE graph.
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


def _tree(
    tree: Any,
    world: bool = False,
    light: bool = False,
) -> dict[str, Any] | None:
    data = manifest._node_tree_manifest(tree)
    if data is not None:
        output = _cycles_output(tree, world, light)
        links = _socket_links(tree)
        data["surface_root"] = _output_root(output, links, "Surface")
        data["volume_root"] = (
            None
            if light
            else _output_root(output, links, "Volume")
        )
        data["displacement_root"] = (
            None
            if world or light
            else _output_root(output, links, "Displacement")
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


def _write_array(stream: Any, values: Any) -> dict[str, int]:
    if isinstance(values, np.ndarray):
        return _write_ndarray(stream, values)
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


def _write_ndarray(stream: Any, values: np.ndarray[Any, Any]) -> dict[str, int]:
    contiguous = np.ascontiguousarray(values)
    if sys.byteorder != "little" and contiguous.dtype.itemsize > 1:
        contiguous = contiguous.byteswap()
    offset = stream.tell()
    payload = memoryview(contiguous).cast("B")
    written = stream.write(payload)
    if written != len(payload):
        raise OSError(
            f"short geometry write: expected {len(payload)} bytes, "
            f"wrote {written}"
        )
    return {"offset": offset, "bytes": written}


def _geometry_cache_key(
    evaluated_obj: Any,
    original: Any,
    scene: Any,
    is_instance: bool,
) -> tuple[Any, ...]:
    # Reusing an evaluated mesh is only safe when Blender reports that the
    # render geometry is identical to the source Mesh datablock and material
    # bindings match. Geometry Nodes / curve / grease pencil instances carry
    # evaluated Mesh datablocks directly on `evaluated_obj.data`.
    mesh_data = (
        evaluated_obj.data
        if isinstance(getattr(evaluated_obj, "data", None), bpy.types.Mesh)
        else None
    )
    material_slots = tuple(
        mat.as_pointer() if mat is not None else 0
        for mat in _resolved_material_slots(evaluated_obj, mesh_data)
    )
    if (
        not is_instance
        and original.type == "MESH"
        and original.data is not None
        and not original.is_modified(scene, "RENDER")
    ):
        return ("mesh", original.data.as_pointer(), material_slots)
    if is_instance and mesh_data is not None:
        return (
            "inst_mesh",
            mesh_data.as_pointer(),
            material_slots,
            _cycles_uses_adaptive_subdivision(original),
        )
    return (
        "object",
        original.as_pointer(),
        evaluated_obj.data.as_pointer() if evaluated_obj.data is not None else 0,
        material_slots,
    )


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
    evaluated_obj: Any,
    original: Any,
    depsgraph: Any,
    stream: Any,
    rec709_to_rgb: tuple[tuple[float, float, float], ...],
    required_geometry_attributes: set[str],
    required_uv_map_names: set[str],
) -> dict[str, Any] | None:
    try:
        mesh = evaluated_obj.to_mesh(
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
    except RuntimeError:
        return None
    if mesh is None:
        return None
    try:
        resolved_slots = _resolved_material_slots(evaluated_obj, mesh)
        # Cycles copies Blender's evaluated Mesh::corner_tris verbatim. The
        # Python loop-triangle cache is the same ordered topology; mutating
        # n-gons through BMesh here would change both triangle identity and
        # the tessellation consumed by true displacement.
        mesh.calc_loop_triangles()
        tri_count = len(mesh.loop_triangles)
        if tri_count == 0:
            return None
        needs_pointiness = any(
            _material_uses_pointiness(mat)
            for mat in resolved_slots
        )
        pointiness_source = None
        v_count = len(mesh.vertices)
        l_count = len(mesh.loops)
        c_count = tri_count * 3
        if needs_pointiness:
            e_count = len(mesh.edges)
            p_normals = np.empty(v_count * 3, dtype=np.float32)
            mesh.vertices.foreach_get("normal", p_normals)
            p_edges = np.empty(e_count * 2, dtype=np.uint32)
            mesh.edges.foreach_get("vertices", p_edges)
            pointiness_source = {
                "point_normals": _write_ndarray(stream, p_normals),
                "edge_count": e_count,
                "edges": _write_ndarray(stream, p_edges),
            }

        tri_loops = np.empty(c_count, dtype=np.int32)
        mesh.loop_triangles.foreach_get("loops", tri_loops)
        indices = np.empty(c_count, dtype=np.uint32)
        mesh.loop_triangles.foreach_get("vertices", indices)
        materials = np.empty(tri_count, dtype=np.uint32)
        mesh.loop_triangles.foreach_get("material_index", materials)

        uv_layer_names = [layer.name for layer in mesh.uv_layers]
        # Cycles maps ATTR_STD_UV to Mesh::default_uv_map_name(), exposed
        # through Blender's active_render UV layer. The UI/editing-active
        # layer is an independent choice and may intentionally differ.
        default_uv_name = (
            mesh.uv_layers.active_render.name
            if mesh.uv_layers.active_render is not None
            else None
        )
        uv_layer_values: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
        for uv_layer_name in uv_layer_names:
            uv_layer = mesh.uv_layers.get(uv_layer_name)
            if uv_layer is None:
                continue
            raw_uv = np.empty(l_count * 2, dtype=np.float32)
            if l_count > 0:
                uv_layer.data.foreach_get("uv", raw_uv)
            uv_layer_values[uv_layer_name] = raw_uv.reshape(l_count, 2)[tri_loops]

        promoted_uv_attrs: set[tuple[str, str]] = set()
        if required_uv_map_names and getattr(mesh, "attributes", None) is not None:
            for attr in mesh.attributes:
                name = str(attr.name)
                if (
                    name.startswith(".")
                    or name not in required_uv_map_names
                    or name in uv_layer_values
                    or str(attr.data_type) != "FLOAT2"
                ):
                    continue
                domain = str(attr.domain)
                if domain == "CORNER" and len(attr.data) >= l_count:
                    raw = np.empty(l_count * 2, dtype=np.float32)
                    if l_count > 0:
                        attr.data.foreach_get("vector", raw)
                    uv_layer_values[name] = raw.reshape(l_count, 2)[tri_loops]
                    promoted_uv_attrs.add((name, "CORNER"))
                elif domain == "POINT" and len(attr.data) >= v_count:
                    raw = np.empty(v_count * 2, dtype=np.float32)
                    if v_count > 0:
                        attr.data.foreach_get("vector", raw)
                    uv_layer_values[name] = raw.reshape(v_count, 2)[indices]
                    promoted_uv_attrs.add((name, "POINT"))

        if default_uv_name is not None and default_uv_name in uv_layer_values:
            uvs = uv_layer_values[default_uv_name]
        else:
            uvs = np.zeros((c_count, 2), dtype=np.float32)

        texspace_location = mesh.texspace_location
        texspace_size = mesh.texspace_size
        generated_scale = np.asarray(
            [
                0.5 / float(size) if float(size) != 0.0 else 0.0
                for size in texspace_size
            ],
            dtype=np.float64,
        )
        generated_location = (
            np.asarray([float(v) for v in texspace_location], dtype=np.float64)
            * generated_scale
            - 0.5
        )
        generated_transform = [
            float(generated_scale[0]),
            0.0,
            0.0,
            0.0,
            0.0,
            float(generated_scale[1]),
            0.0,
            0.0,
            0.0,
            0.0,
            float(generated_scale[2]),
            0.0,
            float(-generated_location[0]),
            float(-generated_location[1]),
            float(-generated_location[2]),
            1.0,
        ]
        pos_flat = np.empty(v_count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", pos_flat)
        positions = pos_flat.reshape(v_count, 3)
        generated = (
            positions.astype(np.float64) * generated_scale - generated_location
        ).astype(np.float32)

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
            smooth = np.ones(tri_count, dtype=np.uint32)
        elif sharp_face_attribute is not None:
            poly_count = len(mesh.polygons)
            sharp_vals = np.empty(poly_count, dtype=np.float32)
            sharp_face_attribute.data.foreach_get("value", sharp_vals)
            tri_polys = np.empty(tri_count, dtype=np.int32)
            mesh.loop_triangles.foreach_get("polygon_index", tri_polys)
            smooth = (sharp_vals[tri_polys] == 0.0).astype(np.uint32)
        else:
            fill_val = 1 if blender_normal_domain != "FACE" else 0
            smooth = np.full(tri_count, fill_val, dtype=np.uint32)

        normal_domain = "CORNER" if use_corner_normals else "POINT"
        if normal_domain == "POINT":
            nrm_flat = np.empty(v_count * 3, dtype=np.float32)
            mesh.vertices.foreach_get("normal", nrm_flat)
            normals = nrm_flat
        else:
            nrm_flat = np.empty(l_count * 3, dtype=np.float32)
            mesh.loops.foreach_get("normal", nrm_flat)
            normals = nrm_flat.reshape(l_count, 3)[tri_loops]

        color_layers = [
            attribute
            for attribute in mesh.color_attributes
            if attribute.data_type in {"BYTE_COLOR", "FLOAT_COLOR"}
            and attribute.domain in {"CORNER", "POINT"}
        ]
        rec709_mat = np.asarray(rec709_to_rgb, dtype=np.float64)
        color_values: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
        byte_color_values: dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
        for attribute in color_layers:
            dom_cnt = v_count if attribute.domain == "POINT" else l_count
            c_flat = np.empty(dom_cnt * 4, dtype=np.float32)
            attribute.data.foreach_get("color", c_flat)
            c_rgba = c_flat.reshape(dom_cnt, 4)
            if attribute.domain == "CORNER":
                c_rgba = c_rgba[tri_loops]
            if attribute.data_type == "BYTE_COLOR" and attribute.domain == "CORNER":
                rgb_lin = (c_rgba[:, :3].astype(np.float64) @ rec709_mat.T).astype(
                    np.float32
                )
                c_rgba = np.concatenate([rgb_lin, c_rgba[:, 3:4]], axis=1)
                srgb_flat = np.empty(l_count * 4, dtype=np.float32)
                attribute.data.foreach_get("color_srgb", srgb_flat)
                srgb_corner = srgb_flat.reshape(l_count, 4)[tri_loops]
                byte_color_values[attribute.name] = np.clip(
                    np.floor(srgb_corner.astype(np.float64) * 255.0 + 0.5),
                    0,
                    255,
                ).astype(np.uint8)
            color_values[attribute.name] = c_rgba

        excluded_generic = (
            {(name, "CORNER") for name in uv_layer_values}
            | promoted_uv_attrs
            | {(attribute.name, attribute.domain) for attribute in color_layers}
        )
        generic_attributes = _extract_generic_mesh_attributes(
            mesh,
            required_geometry_attributes,
            excluded_generic,
            rec709_to_rgb,
            _cycles_color_attribute_value,
            _write_array,
            stream,
        )
        edge_count = len(mesh.edges)
        edge_vertices = np.empty(edge_count * 2, dtype=np.uint32)
        mesh.edges.foreach_get("vertices", edge_vertices)
        return {
            "name": evaluated_obj.name,
            "uses_adaptive_subdivision": (
                _cycles_uses_adaptive_subdivision(original)
            ),
            "point_count": v_count,
            "corner_count": c_count,
            "triangle_count": tri_count,
            "positions": _write_ndarray(stream, positions),
            "normal_domain": normal_domain,
            "normals": _write_ndarray(stream, normals),
            "uv_domain": "CORNER",
            "default_uv_available": default_uv_name in uv_layer_values,
            "uv": _write_ndarray(stream, uvs),
            "uv_tangent_domain": "CORNER",
            "uv_tangents": None,
            "uv_layers": [
                {
                    "name": name,
                    "domain": "CORNER",
                    "values": _write_ndarray(
                        stream, uv_layer_values[name]
                    ),
                    "tangents": None,
                }
                for name in uv_layer_values
            ],
            "generated_domain": "POINT",
            "generated": _write_ndarray(stream, generated),
            # Cycles' ATTR_STD_GENERATED_TRANSFORM. Volume shading points
            # use this object-space affine transform directly because there
            # is no surface primitive whose Generated values can be
            # interpolated.
            "generated_transform": generated_transform,
            "default_color_attribute": (
                mesh.color_attributes.default_color_name or None
            ),
            "color_attributes": [
                {
                    "name": attribute.name,
                    "domain": attribute.domain,
                    "data_type": attribute.data_type,
                    "values": _write_ndarray(
                        stream, color_values[attribute.name]
                    ),
                    "byte_values": (
                        _write_ndarray(
                            stream,
                            byte_color_values[attribute.name],
                        )
                        if attribute.name in byte_color_values
                        else None
                    ),
                }
                for attribute in color_layers
            ],
            "generic_attributes": generic_attributes,
            "pointiness_source": pointiness_source,
            "indices": _write_ndarray(stream, indices),
            "triangle_material_slots": _write_ndarray(stream, materials),
            "triangle_smooth": _write_ndarray(stream, smooth),
            "triangle_random_per_island": None,
            "island_edges": {
                "edge_count": edge_count,
                "edges": _write_ndarray(stream, edge_vertices),
            },
            "material_slots": [
                mat.name if mat else None
                for mat in resolved_slots
            ],
        }
    finally:
        evaluated_obj.to_mesh_clear()


def _object_has_particle_hair(obj: Any, depsgraph: Any) -> bool:
    if obj.type != "MESH":
        return False
    return bool(_particle_hair_systems(obj.evaluated_get(depsgraph)))


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
        # Cycles numbers color attributes in the complete filtered
        # CORNER/BYTE_COLOR image, then exports only shader-demanded names.
        # Preserve the original indices passed to mcol_on_emitter even when
        # unrelated layers are omitted from the bundle.
        eligible_color_attributes = [
            attribute.name
            for attribute in emitter_mesh.attributes
            if attribute.data_type == "BYTE_COLOR"
            and attribute.domain == "CORNER"
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
    world_to_object = np.linalg.inv(object_to_world).astype(
        np.float32, copy=False
    )
    keys = array.array("f")
    curve_first_key = array.array("I")
    curve_material_slots = array.array("I")
    intercepts = array.array("f")
    lengths = array.array("f")
    randoms = array.array("f")
    curve_uv_layers = {
        name: array.array("f") for name in uv_layer_names
    }
    curve_color_attributes = {
        name: array.array("f") for name, _ in color_attributes
    }
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
            for uv_no, name in enumerate(uv_layer_names):
                uv = _cycles_particle_hair_uv(
                    modifier,
                    particle_system,
                    particle_no,
                    uv_no,
                )
                curve_uv_layers[name].extend(
                    (float(uv[0]), float(uv[1]))
                )
            for name, color_no in color_attributes:
                color = _cycles_particle_hair_color(
                    modifier,
                    particle_system,
                    particle_no,
                    color_no,
                )
                curve_color_attributes[name].extend(
                    _cycles_particle_hair_color_value(color)
                )
            world_positions = _cycles_particle_hair_positions(
                particle_system,
                evaluated,
                particle_no,
                key_count,
            )
            positions: list[np.ndarray[Any, np.dtype[np.float32]]] = []
            curve_times: list[np.float32] = []
            curve_length = np.float32(0.0)
            previous: np.ndarray[Any, np.dtype[np.float32]] | None = None
            for coordinate in world_positions:
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
        "default_uv_layer": default_uv_layer,
        "uv_layers": [
            {
                "name": name,
                "values": _write_array(stream, values),
            }
            for name, values in curve_uv_layers.items()
        ],
        "color_attributes": [
            {
                "name": name,
                "values": _write_array(stream, values),
            }
            for name, values in curve_color_attributes.items()
        ],
        "intercept": _write_array(stream, intercepts),
        "length": _write_array(stream, lengths),
        "random": _write_array(stream, randoms),
        "material_slots": [
            slot.material.name if slot.material else None
            for slot in obj.material_slots
        ],
    }


def _vec_hash_uint2_zero_to_float(count: int) -> np.ndarray[Any, Any]:
    """Vectorized hash_uint2(i, 0) / UINT32_MASK matching cycles_hash.py."""
    seed = np.uint32(0xDEADBEEF + (2 << 2) + 13)
    a = seed + np.arange(count, dtype=np.uint32)
    b = np.full(count, seed, dtype=np.uint32)
    c = np.full(count, seed, dtype=np.uint32)
    with np.errstate(over="ignore"):
        def rotl(v: np.ndarray[Any, Any], k: int) -> np.ndarray[Any, Any]:
            return (v << np.uint32(k)) | (v >> np.uint32(32 - k))

        c = (c ^ b) - rotl(b, 14)
        a = (a ^ c) - rotl(c, 11)
        b = (b ^ a) - rotl(a, 25)
        c = (c ^ b) - rotl(b, 16)
        a = (a ^ c) - rotl(c, 4)
        b = (b ^ a) - rotl(a, 14)
        c = (c ^ b) - rotl(b, 24)
    return (c.astype(np.float64) / np.float64(cycles_hash.UINT32_MASK)).astype(
        np.float32
    )


def _curves_object_geometry(
    obj: Any,
    depsgraph: Any,
    stream: Any,
    shape: str,
    subdivisions: int,
) -> dict[str, Any] | None:
    """Export an evaluated Blender CURVES object vectorized with NumPy."""
    evaluated = obj.evaluated_get(depsgraph)
    curves_data = evaluated.data
    if curves_data is None:
        return None
    curve_count = len(curves_data.curves)
    point_count = len(curves_data.points)
    if curve_count == 0 or point_count == 0:
        return None

    points_length = np.empty(curve_count, dtype=np.int32)
    curves_data.curves.foreach_get("points_length", points_length)

    pos_attr = curves_data.attributes.get("position")
    if pos_attr is None:
        return None
    positions = np.empty(point_count * 3, dtype=np.float32)
    pos_attr.data.foreach_get("vector", positions)
    positions = positions.reshape(point_count, 3)

    rad_attr = curves_data.attributes.get("radius")
    if rad_attr is not None and rad_attr.domain == "POINT":
        radii = np.empty(point_count, dtype=np.float32)
        rad_attr.data.foreach_get("value", radii)
    elif rad_attr is not None and rad_attr.domain == "CURVE":
        curve_radii = np.empty(curve_count, dtype=np.float32)
        rad_attr.data.foreach_get("value", curve_radii)
        radii = np.repeat(curve_radii, points_length)
    else:
        radii = np.full(point_count, 0.005, dtype=np.float32)

    material_slot_count = max(len(obj.material_slots), 1)
    mat_attr = curves_data.attributes.get("material_index")
    if mat_attr is not None and mat_attr.domain == "CURVE":
        curve_mat_slots = np.empty(curve_count, dtype=np.int32)
        mat_attr.data.foreach_get("value", curve_mat_slots)
        np.clip(curve_mat_slots, 0, material_slot_count - 1, out=curve_mat_slots)
    else:
        curve_mat_slots = np.zeros(curve_count, dtype=np.int32)

    valid_curves = points_length >= 2
    if not np.all(valid_curves):
        if not np.any(valid_curves):
            return None
        keep_pts = np.repeat(valid_curves, points_length)
        positions = positions[keep_pts]
        radii = radii[keep_pts]
        points_length = points_length[valid_curves]
        curve_mat_slots = curve_mat_slots[valid_curves]
        curve_count = int(points_length.shape[0])
        point_count = int(positions.shape[0])
    else:
        valid_curves = None

    curve_first_key = np.empty(curve_count, dtype=np.uint32)
    curve_first_key[0] = 0
    if curve_count > 1:
        np.cumsum(points_length[:-1], out=curve_first_key[1:])

    keys = np.empty((point_count, 4), dtype=np.float32)
    keys[:, :3] = positions
    keys[:, 3] = radii

    deltas = positions[1:] - positions[:-1]
    seg_len = np.sqrt(np.sum(deltas * deltas, axis=1, dtype=np.float32), dtype=np.float32)
    if curve_count > 1:
        seg_len[curve_first_key[1:].astype(np.int64) - 1] = 0.0
    cum = np.empty(point_count, dtype=np.float64)
    cum[0] = 0.0
    if point_count > 1:
        np.cumsum(seg_len, out=cum[1:])

    first_i64 = curve_first_key.astype(np.int64)
    last_i64 = np.empty(curve_count, dtype=np.int64)
    if curve_count > 1:
        last_i64[:-1] = first_i64[1:] - 1
    last_i64[-1] = point_count - 1
    curve_start = cum[first_i64]
    lengths = (cum[last_i64] - curve_start).astype(np.float32)

    start_per_pt = np.repeat(curve_start, points_length)
    len_per_pt = np.repeat(lengths, points_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        intercepts = np.where(
            len_per_pt > 0.0, (cum - start_per_pt) / len_per_pt, 0.0
        ).astype(np.float32)

    randoms = _vec_hash_uint2_zero_to_float(curve_count)
    segment_count = int(point_count - curve_count)

    requested_color_attributes = set().union(
        *(
            _material_named_color_attributes(slot.material)
            for slot in obj.material_slots
        )
    )
    uv_layers_data: list[dict[str, Any]] = []
    color_attributes_data: list[dict[str, Any]] = []
    for attr in curves_data.attributes:
        if attr.name.startswith(".") or attr.domain != "CURVE":
            continue
        if attr.data_type == "FLOAT2":
            buf = np.empty(len(curves_data.curves) * 2, dtype=np.float32)
            attr.data.foreach_get("vector", buf)
            uv = buf.reshape(-1, 2)
            if valid_curves is not None:
                uv = uv[valid_curves]
            uv_layers_data.append(
                {
                    "name": attr.name,
                    "values": _write_ndarray(stream, uv),
                }
            )
        elif (
            attr.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}
            and attr.name in requested_color_attributes
        ):
            buf = np.empty(len(curves_data.curves) * 4, dtype=np.float32)
            attr.data.foreach_get("color", buf)
            col = buf.reshape(-1, 4)
            if valid_curves is not None:
                col = col[valid_curves]
            color_attributes_data.append(
                {
                    "name": attr.name,
                    "values": _write_ndarray(stream, col),
                }
            )

    uv_names = [layer["name"] for layer in uv_layers_data]
    default_uv_layer = str(getattr(curves_data, "surface_uv_map", "") or "")
    if default_uv_layer not in uv_names:
        if "surface_uv_coordinate" in uv_names:
            default_uv_layer = "surface_uv_coordinate"
        elif "UVMap" in uv_names:
            default_uv_layer = "UVMap"
        else:
            default_uv_layer = uv_names[0] if uv_names else None

    return {
        "name": f"{obj.name}.curves",
        "shape": {
            "RIBBONS": "RIBBON",
            "THICK": "THICK",
            "THICK_LINEAR": "THICK_LINEAR",
        }[shape],
        "subdivisions": int(subdivisions),
        "key_count": int(point_count),
        "curve_count": int(curve_count),
        "segment_count": int(segment_count),
        "keys": _write_ndarray(stream, keys),
        "curve_first_key": _write_ndarray(stream, curve_first_key),
        "curve_material_slots": _write_ndarray(
            stream, curve_mat_slots.astype(np.uint32, copy=False)
        ),
        "default_uv_layer": default_uv_layer,
        "uv_layers": uv_layers_data,
        "color_attributes": color_attributes_data,
        "intercept": _write_ndarray(stream, intercepts),
        "length": _write_ndarray(stream, lengths),
        "random": _write_ndarray(stream, randoms),
        "material_slots": [
            slot.material.name if slot.material else None
            for slot in obj.material_slots
        ],
    }


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


def _light(
    obj: Any,
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
        # BlenderSync keys lamp shaders by the Light datablock ID and uses
        # that ID's name for Shader::name. Keep this distinct from the object
        # name/asset identity: multiple light objects may share one datablock
        # and therefore one Cycles shader-vector entry.
        "shader_name": _original_id(light).name,
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
        **_cycles_object_properties(object_instance, bpy.context.view_layer),
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
    geometry_type: str,
    geometry_index: int,
    cycles_object_index: int,
    light_groups: dict[str, int],
    particle_source: dict[str, Any] | None,
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
        "cycles_particle_source": particle_source,
        # Cycles uses the dependency-graph random id for duplis and hashes
        # the object name for ordinary objects. Its separate hair object
        # intentionally receives the same random id.
        "random_id": int(
            object_instance.random_id
            if object_instance.is_instance
            else _cycles_object_random_id(object_instance.object.name)
        )
        & cycles_hash.UINT32_MASK,
        **_cycles_object_properties(object_instance, bpy.context.view_layer),
        "ambient_occlusion_distance": _cycles_object_ao_distance(
            object_instance
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


def _camera_focus_distance(camera: Any, depsgraph: Any) -> float:
    """Return the effective Cycles focal distance for the render camera.

    Blender stores an authored scalar distance and an optional focus object.
    Cycles ignores the scalar when the object is present and projects the
    camera-to-target displacement onto the camera's local Z axis.  A bone
    subtarget replaces the focus object's origin when it resolves.
    """
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
    camera_to_focus = (
        camera_transform.translation - focus_transform.translation
    )
    return max(abs(float(view_direction.dot(camera_to_focus))), 1.0e-5)


def _camera_transform_without_scale(camera: Any) -> Any:
    """Match Cycles' per-axis normalization of the camera transform."""
    transform = camera.matrix_world.copy()
    for column_index in range(3):
        column = transform.col[column_index].xyz.normalized()
        for row_index in range(3):
            transform[row_index][column_index] = column[row_index]
    return transform


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
    particle_sources = _ParticleSourceRegistry(scene)
    cycles_uses_light_linking = False
    required_geometry_attributes = _collect_required_geometry_attributes(
        bpy.data.materials
    )
    required_uv_map_names = _collect_required_uv_map_names(
        bpy.data.materials
    )
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
                cycles_uses_light_linking = (
                    cycles_uses_light_linking
                    or _cycles_object_uses_light_linking(object_instance)
                )
            hair_object_index: int | None = None
            if (
                bool(object_instance.show_particles)
                and _object_has_particle_hair(original, depsgraph)
            ):
                hair_object_index = cycles_object_index
                cycles_object_index += 1
                cycles_uses_light_linking = (
                    cycles_uses_light_linking
                    or _cycles_object_uses_light_linking(object_instance)
                )

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
                        object_instance=object_instance,
                        particle_source=particle_sources.capture(
                            object_instance
                        ),
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
                            # BlenderSync only assigns Shader::pass_id to
                            # material shaders. Lamp shaders retain Cycles'
                            # default zero pass ID.
                            "pass_id": 0,
                        },
                    )
                )
                continue
            if obj.type == "CURVES":
                if (
                    not object_instance.show_self
                    or source_object_index is None
                ):
                    continue
                material_slots = tuple(
                    slot.material.as_pointer()
                    if slot.material is not None
                    else 0
                    for slot in obj.material_slots
                )
                curves_key = (
                    "curves",
                    obj.data.as_pointer()
                    if obj.data is not None
                    else obj.as_pointer(),
                    material_slots,
                    curve_shape,
                    curve_subdivisions,
                )
                curve_geometry_index = curve_geometry_by_source.get(curves_key)
                if curve_geometry_index is None:
                    curve_geometry = _curves_object_geometry(
                        obj,
                        depsgraph,
                        stream,
                        curve_shape,
                        curve_subdivisions,
                    )
                    if curve_geometry is not None:
                        curve_geometry["cycles_sync"] = {
                            "curve_offset": cycles_curve_offset,
                            "segment_offset": cycles_curve_segment_offset,
                        }
                        cycles_curve_offset += int(
                            curve_geometry["curve_count"]
                        )
                        cycles_curve_segment_offset += int(
                            curve_geometry["segment_count"]
                        )
                        curve_geometry_index = len(curve_geometries)
                        curve_geometry_by_source[curves_key] = (
                            curve_geometry_index
                        )
                        curve_geometries.append(curve_geometry)
                if curve_geometry_index is not None:
                    instances.append(
                        _geometry_instance(
                            object_instance,
                            "CURVE",
                            curve_geometry_index,
                            source_object_index,
                            light_groups,
                            particle_sources.capture(object_instance),
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
                key = _geometry_cache_key(
                    obj, original, scene, object_instance.is_instance
                )
                geometry_index = geometry_by_source.get(key)
                if geometry_index is None:
                    geometry_data = _geometry(
                        obj,
                        original,
                        depsgraph,
                        stream,
                        rec709_to_rgb,
                        required_geometry_attributes,
                        required_uv_map_names,
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
                            "MESH",
                            geometry_index,
                            source_object_index,
                            light_groups,
                            particle_sources.capture(object_instance),
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
                            "CURVE",
                            curve_geometry_index,
                            hair_object_index,
                            light_groups,
                            particle_sources.capture(object_instance),
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
                "generated",
                "indices",
                "triangle_material_slots",
            )
            if geometry.get(name) is not None
        ]
        for optional_name in ("uv_tangents", "triangle_random_per_island"):
            if geometry.get(optional_name) is not None:
                sections.append(geometry[optional_name])
        island_edges = geometry.get("island_edges")
        if island_edges is not None:
            sections.append(island_edges["edges"])
        sections.extend(
            attribute["values"]
            for attribute in geometry["color_attributes"]
        )
        sections.extend(
            attribute["byte_values"]
            for attribute in geometry["color_attributes"]
            if attribute.get("byte_values") is not None
        )
        sections.extend(
            attribute["values"]
            for attribute in geometry.get("generic_attributes", ())
        )
        for uv_layer in geometry["uv_layers"]:
            sections.append(uv_layer["values"])
            if uv_layer.get("tangents") is not None:
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
        for uv_layer in geometry["uv_layers"]:
            section = uv_layer["values"]
            end = int(section["offset"]) + int(section["bytes"])
            if (
                int(section["offset"]) < 0
                or int(section["bytes"]) < 0
                or end > geometry_size
            ):
                raise RuntimeError(
                    f"incomplete geometry.bin UV section for "
                    f"{geometry['name']!r}: {section}, "
                    f"file size {geometry_size}"
                )
        for attribute in geometry["color_attributes"]:
            section = attribute["values"]
            end = int(section["offset"]) + int(section["bytes"])
            if (
                int(section["offset"]) < 0
                or int(section["bytes"]) < 0
                or end > geometry_size
            ):
                raise RuntimeError(
                    "incomplete geometry.bin curve-color section for "
                    f"{geometry['name']!r}: {section}, "
                    f"file size {geometry_size}"
                )

    # The exported geometry records are the complete surface root set. A
    # Blender material outside their slot image cannot be dereferenced by any
    # exported instance; serializing its graph would make unrelated datablocks
    # alter SVM size, diagnostics, and shader-cache identity.
    used_material_names = {
        name
        for geometry in (*geometries, *curve_geometries)
        for name in geometry["material_slots"]
        if name is not None
    }
    materials = []
    for material in sorted(bpy.data.materials, key=lambda item: item.name):
        if material.name not in used_material_names:
            continue
        cycles = getattr(material, "cycles", None)
        shader_index = material_shader_indices.get(_id_key(material))
        materials.append(
            {
                "name": material.name,
                "use_transparent_shadow": bool(
                    getattr(material, "use_transparent_shadow", True)
                ),
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
                "volume_interpolation": str(
                    getattr(cycles, "volume_interpolation", "LINEAR")
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
                    {
                        "shader_index": shader_index,
                        "pass_id": int(material.pass_index),
                    }
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

    camera = (
        source_camera.evaluated_get(depsgraph)
        if source_camera is not None
        else None
    )
    camera_data = None
    if camera is not None:
        camera_data = {
            "name": camera.name,
            "type": camera.data.type,
            "transform": _column_major(
                _camera_transform_without_scale(camera)
            ),
            "angle": float(camera.data.angle),
            "angle_x": float(camera.data.angle_x),
            "angle_y": float(camera.data.angle_y),
            "sensor_fit": str(camera.data.sensor_fit),
            "lens": float(camera.data.lens),
            "sensor_width": float(camera.data.sensor_width),
            "sensor_height": float(camera.data.sensor_height),
            "pixel_aspect_x": float(scene.render.pixel_aspect_x),
            "pixel_aspect_y": float(scene.render.pixel_aspect_y),
            "ortho_scale": float(camera.data.ortho_scale),
            "shift_x": float(camera.data.shift_x),
            "shift_y": float(camera.data.shift_y),
            "clip_start": float(camera.data.clip_start),
            "clip_end": float(camera.data.clip_end),
            "dof": {
                "enabled": bool(camera.data.dof.use_dof),
                "focus_distance": _camera_focus_distance(
                    camera, depsgraph
                ),
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
        "cycles_sync": {
            "object_count": cycles_object_index + (
                1 if scene.world is not None else 0
            ),
            "uses_light_linking": cycles_uses_light_linking,
        },
        "source": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "blender_build": blender_build_identity.current(bpy.app),
        "exporter": _EXPORTER_IDENTITY,
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
                "volume_sampling": str(
                    getattr(
                        getattr(scene.world, "cycles", None),
                        "volume_sampling",
                        "MULTIPLE_IMPORTANCE",
                    )
                ),
                "volume_interpolation": str(
                    getattr(
                        getattr(scene.world, "cycles", None),
                        "volume_interpolation",
                        "LINEAR",
                    )
                ),
                "sample_map_resolution": int(
                    getattr(
                        getattr(scene.world, "cycles", None),
                        "sample_map_resolution",
                        1024,
                    )
                ),
                "max_bounces": int(
                    getattr(scene.world.cycles, "max_bounces", 1024)
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
                    "pass_id": 0,
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
