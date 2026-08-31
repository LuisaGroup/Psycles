"""Emit a deterministic, machine-readable manifest for a Blender scene.

Run with:

    blender scene.blend --background --python blender_scene_manifest.py -- out.json

The manifest is intentionally descriptive rather than a Psycles scene export.
It is used to drive feature coverage before any lowering policy is applied.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import Any

import bpy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cycles_hash  # noqa: E402


def _column_major(matrix: Any) -> list[float]:
    return [
        float(matrix[row][column])
        for column in range(4)
        for row in range(4)
    ]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_list"):
        return value.to_list()
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            return [_json_value(item) for item in value]
        except (TypeError, ValueError):
            pass
    return str(value)


def _socket_manifest(socket: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "identifier": socket.identifier,
        "name": socket.name,
        "type": socket.bl_rna.identifier,
        "linked": socket.is_linked,
    }
    if hasattr(socket, "default_value"):
        result["default"] = _json_value(socket.default_value)
    if hasattr(socket, "default_attribute_name"):
        result["default_attribute_name"] = socket.default_attribute_name
    return result


def _node_properties(node: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    ignored = {
        "bl_idname",
        "bl_label",
        "bl_description",
        "bl_icon",
        "bl_static_type",
        "color",
        "dimensions",
        "height",
        "hide",
        "inputs",
        "internal_links",
        "label",
        "location",
        "name",
        "outputs",
        "parent",
        "rna_type",
        "select",
        "show_options",
        "show_preview",
        "show_texture",
        "type",
        "use_custom_color",
        "width",
        "width_hidden",
    }
    for prop in node.bl_rna.properties:
        identifier = prop.identifier
        if (
            identifier.startswith("bl_")
            or identifier in ignored
            or identifier
            in {"location_absolute", "mute", "warning_propagation"}
            or prop.is_readonly
            or prop.type in {"POINTER", "COLLECTION"}
        ):
            continue
        try:
            result[identifier] = _json_value(getattr(node, identifier))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return result


def _node_special_data(node: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if node.bl_idname == "ShaderNodeTexCoord":
        coordinate_object = getattr(node, "object", None)
        if coordinate_object is not None:
            # Cycles stores the explicitly referenced object's object-to-world
            # transform in TextureCoordinateNode::ob_tfm and performs the
            # inverse while compiling NODE_TEXCO_OBJECT_WITH_TRANSFORM.
            # Preserve that raw authored value; do not pre-bake SVM behavior in
            # the Blender exporter.
            result["object_coordinates"] = {
                "object": coordinate_object.name,
                "object_to_world": _column_major(
                    coordinate_object.matrix_world
                ),
            }
    if hasattr(node, "texture_mapping"):
        mapping = node.texture_mapping
        translation = tuple(float(value) for value in mapping.translation)
        rotation = tuple(float(value) for value in mapping.rotation)
        scale = tuple(float(value) for value in mapping.scale)
        axes = (
            mapping.mapping_x,
            mapping.mapping_y,
            mapping.mapping_z,
        )
        # Cycles attaches this legacy TexMapping to every TextureNode and
        # elides it only when translation/rotation/scale and the axis map are
        # exactly identity. Old production .blend files still use it even
        # though the modern UI normally exposes a separate Mapping node.
        if (
            translation != (0.0, 0.0, 0.0)
            or rotation != (0.0, 0.0, 0.0)
            or scale != (1.0, 1.0, 1.0)
            or axes != ("X", "Y", "Z")
        ):
            result["texture_mapping"] = {
                "vector_type": mapping.vector_type,
                "translation": list(translation),
                "rotation": list(rotation),
                "scale": list(scale),
                "mapping_x": axes[0],
                "mapping_y": axes[1],
                "mapping_z": axes[2],
            }
    if hasattr(node, "color_ramp"):
        ramp = node.color_ramp
        result["color_ramp"] = {
            "color_mode": ramp.color_mode,
            "interpolation": ramp.interpolation,
            "hue_interpolation": ramp.hue_interpolation,
            # Cycles' Blender adapter evaluates ColorBand at i / 256 for
            # i=[0, 256], then the device linearly (or constantly) looks up
            # that 257-entry table. Export the same normalized pre-SVM data
            # rather than reimplementing Blender's HSV/HSL/Cardinal/B-Spline
            # ColorBand evaluator in the renderer.
            "samples": [
                list(ramp.evaluate(index / 256.0))
                for index in range(257)
            ],
            "elements": [
                {
                    "position": element.position,
                    "color": list(element.color),
                }
                for element in ramp.elements
            ],
        }
    if hasattr(node, "mapping"):
        mapping = node.mapping
        mapping.update()
        curves = list(mapping.curves)
        result["curve_mapping"] = {
            "black_level": list(mapping.black_level),
            "white_level": list(mapping.white_level),
            "clip_min": [mapping.clip_min_x, mapping.clip_min_y],
            "clip_max": [mapping.clip_max_x, mapping.clip_max_y],
            "use_clip": mapping.use_clip,
            "extrapolate": mapping.extend == "EXTRAPOLATED",
            "curves": [
                {
                    "points": [
                        {
                            "location": list(point.location),
                            "handle_type": point.handle_type,
                        }
                        for point in curve.points
                    ]
                }
                for curve in curves
            ],
        }
        curve_count = {
            "ShaderNodeFloatCurve": 1,
            "ShaderNodeVectorCurve": 3,
            "ShaderNodeRGBCurve": 4,
        }.get(node.bl_idname)
        if curve_count is not None and len(curves) >= curve_count:
            # This is the exact normalized data route used by Cycles'
            # Blender adapter: find the common domain of the curves consumed
            # by this node, then sample the 257 endpoints that become inline
            # SVM table data. RGB Curves additionally applies its fourth
            # common curve before the per-channel curves.
            min_x = min(
                float(curve.points[0].location[0])
                for curve in curves[:curve_count]
            )
            max_x = max(
                float(curve.points[-1].location[0])
                for curve in curves[:curve_count]
            )
            curve_mapping = result["curve_mapping"]
            curve_mapping["min_x"] = min_x
            curve_mapping["max_x"] = max_x
            curve_mapping["samples"] = []
            for index in range(257):
                x = min_x + index / 256.0 * (max_x - min_x)
                if node.bl_idname == "ShaderNodeFloatCurve":
                    sample: float | list[float] = mapping.evaluate(
                        curves[0], x
                    )
                else:
                    input_x = (
                        mapping.evaluate(curves[3], x)
                        if node.bl_idname == "ShaderNodeRGBCurve"
                        else x
                    )
                    sample = [
                        mapping.evaluate(curves[channel], input_x)
                        for channel in range(3)
                    ]
                curve_mapping["samples"].append(sample)
    if hasattr(node, "image_user"):
        image_user = node.image_user
        result["image_user"] = {
            "frame_current": image_user.frame_current,
            "frame_duration": image_user.frame_duration,
            "frame_offset": image_user.frame_offset,
            "frame_start": image_user.frame_start,
            "tile": image_user.tile,
            "use_auto_refresh": image_user.use_auto_refresh,
            "use_cyclic": image_user.use_cyclic,
        }
    return result


def _node_tree_manifest(node_tree: Any) -> dict[str, Any] | None:
    if node_tree is None:
        return None
    nodes = []
    for node in sorted(node_tree.nodes, key=lambda item: item.name):
        internal_links = [
            {
                "from_socket": link.from_socket.identifier,
                "to_socket": link.to_socket.identifier,
            }
            for link in node.internal_links
        ]
        internal_links.sort(
            key=lambda item: (
                item["to_socket"],
                item["from_socket"],
            )
        )
        nodes.append(
            {
                "name": node.name,
                "label": node.label,
                "bl_idname": node.bl_idname,
                "type": node.type,
                # Cycles replaces every muted node with the runtime-owned
                # internal input-to-output links. Preserve that graph
                # relation explicitly: evaluating the concrete node while
                # merely retaining its external links is observably wrong.
                "mute": bool(node.mute),
                "internal_links": internal_links,
                "properties": _node_properties(node),
                "special": _node_special_data(node),
                "inputs": [_socket_manifest(socket) for socket in node.inputs],
                "outputs": [_socket_manifest(socket) for socket in node.outputs],
                "node_tree": (
                    node.node_tree.name
                    if hasattr(node, "node_tree") and node.node_tree
                    else None
                ),
                "image": (
                    node.image.name
                    if hasattr(node, "image") and node.image
                    else None
                ),
            }
        )
    links = []
    for link in node_tree.links:
        # Match Cycles add_nodes_inlined(): invalid, explicitly muted, and
        # unavailable-socket links do not participate in the shader graph.
        if (
            not link.is_valid
            or link.is_muted
            or link.from_socket.is_unavailable
            or link.to_socket.is_unavailable
        ):
            continue
        links.append(
            {
                "from_node": link.from_node.name,
                "from_socket": link.from_socket.identifier,
                "to_node": link.to_node.name,
                "to_socket": link.to_socket.identifier,
            }
        )
    links.sort(
        key=lambda item: (
            item["to_node"],
            item["to_socket"],
            item["from_node"],
            item["from_socket"],
        )
    )
    return {
        "name": node_tree.name,
        "nodes": nodes,
        "links": links,
    }


def _material_manifest(material: Any) -> dict[str, Any]:
    return {
        "name": material.name,
        "use_nodes": material.use_nodes,
        "surface_render_method": getattr(
            material, "surface_render_method", None
        ),
        "use_transparency_overlap": getattr(
            material, "use_transparency_overlap", None
        ),
        "node_tree": _node_tree_manifest(material.node_tree),
    }


def _object_manifest(obj: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": obj.name,
        "type": obj.type,
        "hide_render": obj.hide_render,
        "matrix_world": [
            value for row in obj.matrix_world for value in row
        ],
        "materials": [
            slot.material.name if slot.material else None
            for slot in obj.material_slots
        ],
        "modifiers": [
            {
                "name": modifier.name,
                "type": modifier.type,
                "show_render": modifier.show_render,
            }
            for modifier in obj.modifiers
        ],
    }
    if obj.type == "MESH":
        result["mesh"] = {
            "name": obj.data.name,
            "vertices": len(obj.data.vertices),
            "edges": len(obj.data.edges),
            "polygons": len(obj.data.polygons),
            "uv_layers": [layer.name for layer in obj.data.uv_layers],
            "color_attributes": [
                {
                    "name": attribute.name,
                    "domain": attribute.domain,
                    "data_type": attribute.data_type,
                }
                for attribute in obj.data.color_attributes
            ],
            "has_custom_normals": obj.data.has_custom_normals,
        }
    elif obj.type == "LIGHT":
        light = obj.data
        result["light"] = {
            "name": light.name,
            "type": light.type,
            "color": list(light.color),
            "energy": light.energy,
            "use_shadow": light.use_shadow,
            "shape": getattr(light, "shape", None),
            "size": getattr(light, "size", None),
            "size_y": getattr(light, "size_y", None),
            "spread": getattr(light, "spread", None),
            "angle": getattr(light, "angle", None),
            "spot_size": getattr(light, "spot_size", None),
            "spot_blend": getattr(light, "spot_blend", None),
            "use_nodes": light.use_nodes,
            "node_tree": _node_tree_manifest(light.node_tree),
        }
    elif obj.type == "CAMERA":
        camera = obj.data
        result["camera"] = {
            "name": camera.name,
            "type": camera.type,
            "lens": camera.lens,
            "sensor_fit": camera.sensor_fit,
            "sensor_width": camera.sensor_width,
            "sensor_height": camera.sensor_height,
            "clip_start": camera.clip_start,
            "clip_end": camera.clip_end,
            "dof": {
                "use_dof": camera.dof.use_dof,
                "focus_object": (
                    camera.dof.focus_object.name
                    if camera.dof.focus_object
                    else None
                ),
                "focus_distance": camera.dof.focus_distance,
                "aperture_fstop": camera.dof.aperture_fstop,
                "aperture_blades": camera.dof.aperture_blades,
                "aperture_rotation": camera.dof.aperture_rotation,
                "aperture_ratio": camera.dof.aperture_ratio,
            },
        }
    return result


def _image_manifest(image: Any) -> dict[str, Any]:
    packed_size = 0
    if image.packed_file is not None:
        packed_size = image.packed_file.size
    return {
        "name": image.name,
        "source": image.source,
        "filepath": image.filepath,
        "filepath_raw": image.filepath_raw,
        "colorspace": image.colorspace_settings.name,
        "alpha_mode": image.alpha_mode,
        "size": list(image.size),
        "packed": image.packed_file is not None,
        "packed_size": packed_size,
        "tiles": [
            {
                "number": tile.number,
                "label": tile.label,
            }
            for tile in image.tiles
        ],
    }


def _cycles_settings(scene: Any) -> dict[str, Any]:
    cycles = scene.cycles
    names = [
        "samples",
        "use_adaptive_sampling",
        "adaptive_threshold",
        "adaptive_min_samples",
        "max_bounces",
        "min_light_bounces",
        "diffuse_bounces",
        "glossy_bounces",
        "transmission_bounces",
        "volume_bounces",
        "min_transparent_bounces",
        "transparent_max_bounces",
        "sample_clamp_direct",
        "sample_clamp_indirect",
        "use_fast_gi",
        "fast_gi_method",
        "ao_bounces",
        "ao_bounces_render",
        "blur_glossy",
        "caustics_reflective",
        "caustics_refractive",
        "direct_light_sampling_type",
        "use_light_tree",
        "light_sampling_threshold",
        "film_exposure",
        "use_denoising",
    ]
    result = {
        name: _json_value(getattr(cycles, name))
        for name in names
        if hasattr(cycles, name)
    }
    base_seed = int(cycles.seed)
    use_animated_seed = bool(
        getattr(cycles, "use_animated_seed", False)
    )
    frame = int(scene.frame_current)
    subframe = float(getattr(scene, "frame_subframe", 0.0))
    result.update(
        {
            # Preserve the authored value and expose the exact seed passed to
            # Cycles' Integrator. The runtime consumes the latter; old scene
            # bundles without it remain supported by the importer.
            "seed": base_seed,
            "use_animated_seed": use_animated_seed,
            "effective_seed": cycles_hash.effective_scene_seed(
                base_seed,
                frame,
                subframe,
                use_animated_seed,
            ),
            "seed_frame": frame,
            "seed_subframe": subframe,
        }
    )
    world = scene.world
    lighting = getattr(world, "light_settings", None) if world else None
    result["ao_factor"] = float(
        getattr(lighting, "ao_factor", 1.0) if world else 0.0
    )
    result["ao_distance"] = float(
        getattr(lighting, "distance", 10.0)
    )
    return result


def _main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(argv) != 1:
        raise SystemExit(
            "expected output path after '--': blender scene.blend "
            "--background --python blender_scene_manifest.py -- out.json"
        )
    output = pathlib.Path(argv[0]).resolve()
    scene = bpy.context.scene
    node_counts: collections.Counter[str] = collections.Counter()
    for material in bpy.data.materials:
        if material.use_nodes and material.node_tree:
            node_counts.update(
                node.bl_idname for node in material.node_tree.nodes
            )
    if scene.world and scene.world.use_nodes and scene.world.node_tree:
        node_counts.update(
            node.bl_idname for node in scene.world.node_tree.nodes
        )
    for light in bpy.data.lights:
        if light.use_nodes and light.node_tree:
            node_counts.update(
                node.bl_idname for node in light.node_tree.nodes
            )
    for group in bpy.data.node_groups:
        node_counts.update(node.bl_idname for node in group.nodes)

    manifest = {
        "schema": "psycles.blender-manifest.v1",
        "source": bpy.data.filepath,
        "blender": {
            "version": bpy.app.version_string,
            "version_cycle": bpy.app.version_cycle,
            "build_hash": bpy.app.build_hash.decode("ascii"),
        },
        "scene": {
            "name": scene.name,
            "engine": scene.render.engine,
            "camera": scene.camera.name if scene.camera else None,
            "frame": scene.frame_current,
            "render": {
                "resolution_x": scene.render.resolution_x,
                "resolution_y": scene.render.resolution_y,
                "resolution_percentage": scene.render.resolution_percentage,
                "film_transparent": scene.render.film_transparent,
            },
            "cycles": _cycles_settings(scene),
            "view_settings": {
                "look": scene.view_settings.look,
                "view_transform": scene.view_settings.view_transform,
                "exposure": scene.view_settings.exposure,
                "gamma": scene.view_settings.gamma,
            },
        },
        "summary": {
            "objects": len(bpy.data.objects),
            "materials": len(bpy.data.materials),
            "meshes": len(bpy.data.meshes),
            "lights": len(bpy.data.lights),
            "cameras": len(bpy.data.cameras),
            "images": len(bpy.data.images),
            "node_types": dict(sorted(node_counts.items())),
        },
        "objects": [
            _object_manifest(obj)
            for obj in sorted(bpy.data.objects, key=lambda item: item.name)
        ],
        "materials": [
            _material_manifest(material)
            for material in sorted(
                bpy.data.materials, key=lambda item: item.name
            )
        ],
        "world": (
            {
                "name": scene.world.name,
                "use_nodes": scene.world.use_nodes,
                "color": list(scene.world.color),
                "node_tree": _node_tree_manifest(scene.world.node_tree),
            }
            if scene.world
            else None
        ),
        "images": [
            _image_manifest(image)
            for image in sorted(bpy.data.images, key=lambda item: item.name)
        ],
        "node_groups": [
            _node_tree_manifest(group)
            for group in sorted(
                bpy.data.node_groups, key=lambda item: item.name
            )
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "PSYCLES_MANIFEST "
        f"objects={manifest['summary']['objects']} "
        f"materials={manifest['summary']['materials']} "
        f"node_types={len(node_counts)} "
        f"output={output}"
    )


if __name__ == "__main__":
    _main()
