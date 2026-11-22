"""Zero-disk in-memory Blender depsgraph and viewport camera sync for Psycles."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import math
import os
import pathlib
import struct
from typing import Any
import zlib

import bpy
from mathutils import Matrix
import numpy as np

from . import bridge
from ._export import blender_build_identity
from ._export import blender_scene_manifest as manifest
from ._export.blender_cycles_object import (
    ParticleSourceRegistry as _ParticleSourceRegistry,
    light_group as _cycles_light_group,
    uses_light_linking as _cycles_object_uses_light_linking,
)
from ._export.geometry import (
    camera_focus_distance,
    column_major,
    cycles_object_is_geometry,
    extract_geometry_instance,
    extract_light,
    extract_mesh_geometry,
    extract_particle_hair_geometry,
    geometry_cache_key,
    normalized_transform,
    object_has_particle_hair,
)
from ._export.shader_nodes import (
    CYCLES_BACKGROUND_SHADER_INDEX,
    cycles_shader_color_space,
    cycles_shader_indices,
    id_key,
    original_id,
    serialize_shader_tree,
)

_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tga",
    ".bmp",
    ".hdr",
    ".exr",
    ".tif",
    ".tiff",
}

_FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "TARGA": ".tga",
    "TARGA_RAW": ".tga",
    "BMP": ".bmp",
    "HDR": ".hdr",
    "OPEN_EXR": ".exr",
    "OPEN_EXR_MULTILAYER": ".exr",
    "TIFF": ".tif",
}


def _image_extension(image: Any) -> str:
    if image.source == "GENERATED":
        return ".png"
    suffix = pathlib.Path(image.filepath).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return suffix
    return _FORMAT_EXTENSIONS.get(image.file_format, ".bin")


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _linear_to_srgb_np(rgb: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    clamped = np.clip(rgb, 0.0, 1.0)
    return np.where(
        clamped <= 0.0031308,
        clamped * 12.92,
        1.055 * np.power(clamped, 1.0 / 2.4) - 0.055,
    )


def _encode_generated_image_png(image: Any) -> bytes:
    width = int(image.size[0])
    height = int(image.size[1])
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    rgba = pixels.reshape((height, width, 4))
    # Blender stores bottom-to-top rows; PNG expects top-to-bottom scanlines.
    rgba = rgba[::-1, :, :]
    colorspace = str(image.colorspace_settings.name).lower()
    is_data = colorspace in {"non-color", "raw", "linear", "linear rec.709"}
    rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
    if not is_data:
        rgb = _linear_to_srgb_np(rgb)
    alpha = np.clip(rgba[:, :, 3:4], 0.0, 1.0)
    out = np.rint(np.concatenate([rgb, alpha], axis=2) * 255.0).astype(np.uint8)
    scanlines = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), out.reshape(height, width * 4)],
        axis=1,
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(scanlines.tobytes(), level=1)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def _export_images_memory(stream: io.BytesIO) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for image in sorted(bpy.data.images, key=lambda candidate: candidate.name):
        if image.type in {"RENDER_RESULT", "COMPOSITING"}:
            continue
        metadata = {
            "name": image.name,
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "source": image.source,
            "colorspace": image.colorspace_settings.name,
            "alpha_mode": image.alpha_mode,
        }
        if min(image.size) <= 0:
            result.append({**metadata, "load_failed": True})
            continue
        extension = _image_extension(image)
        if image.packed_file is not None:
            raw_bytes = bytes(image.packed_file.data)
            offset = stream.tell()
            stream.write(raw_bytes)
            result.append({
                **metadata,
                "source_format": extension,
                "binary": {"offset": offset, "bytes": len(raw_bytes)},
            })
        elif image.source == "GENERATED":
            png_bytes = _encode_generated_image_png(image)
            offset = stream.tell()
            stream.write(png_bytes)
            result.append({
                **metadata,
                "source_format": ".png",
                "binary": {"offset": offset, "bytes": len(png_bytes)},
            })
        else:
            source = pathlib.Path(
                bpy.path.abspath(image.filepath, library=image.library)
            )
            if not source.is_file():
                result.append({**metadata, "load_failed": True})
                continue
            result.append({
                **metadata,
                "path": str(source.resolve()),
            })
    return result


def extract_viewport_camera(
    region: Any,
    v3d: Any,
    rv3d: Any,
    scene: Any,
    depsgraph: Any,
) -> tuple[dict[str, Any], bridge.PsyclesCameraDesc, tuple[Any, ...]]:
    width = max(int(region.width), 1)
    height = max(int(region.height), 1)
    proj = rv3d.window_matrix
    cam_to_world = normalized_transform(rv3d.view_matrix.inverted())
    p00 = max(abs(float(proj[0][0])), 1.0e-6)
    p11 = max(abs(float(proj[1][1])), 1.0e-6)
    ratio = p11 / p00
    pixel_aspect_x = 1.0
    pixel_aspect_y = (float(width) * p00) / (float(height) * p11)

    is_persp = bool(rv3d.is_perspective)
    in_camera = rv3d.view_perspective == "CAMERA" and scene.camera is not None
    cam_eval = (
        scene.camera.evaluated_get(depsgraph) if in_camera else None
    )

    if is_persp:
        angle_x = 2.0 * math.atan(1.0 / p00)
        angle_y = 2.0 * math.atan(1.0 / p11)
        ortho_scale = 1.0
        shift_x = float(proj[0][2]) * 0.5
        shift_y = float(proj[1][2]) / (2.0 * ratio)
        if cam_eval is not None:
            near_clip = max(float(cam_eval.data.clip_start), 1.0e-5)
            far_clip = max(float(cam_eval.data.clip_end), near_clip + 1.0e-3)
        else:
            near_clip = max(float(v3d.clip_start), 1.0e-5)
            far_clip = max(float(v3d.clip_end), near_clip + 1.0e-3)
        cam_type = "PERSP"
        capi_proj = bridge.PSYCLES_PROJECTION_PERSPECTIVE
    else:
        angle_x = 0.7853981633974483
        angle_y = 0.7853981633974483
        ortho_scale = 2.0 / p00
        shift_x = -float(proj[0][3]) * 0.5
        shift_y = -float(proj[1][3]) / (2.0 * ratio)
        p22 = float(proj[2][2])
        p23 = float(proj[2][3])
        if abs(p22) > 1.0e-8:
            z_near = (-1.0 - p23) / p22
            z_far = (1.0 - p23) / p22
            if z_near > 0.0:
                cam_to_world = cam_to_world @ Matrix.Translation((0.0, 0.0, z_near))
                near_clip = 0.0
                far_clip = max(z_near - z_far, 1.0e-3)
            else:
                near_clip = max(-z_near, 0.0)
                far_clip = max(-z_far, near_clip + 1.0e-3)
        else:
            near_clip = 0.0
            far_clip = max(float(v3d.clip_end), 1000.0)
        cam_type = "ORTHO"
        capi_proj = bridge.PSYCLES_PROJECTION_ORTHOGRAPHIC

    dof_enabled = False
    focus_distance = 10.0
    fstop = 2.8
    aperture_blades = 0
    aperture_rotation = 0.0
    aperture_ratio = 1.0
    lens_mm = 50.0
    if cam_eval is not None:
        dof = cam_eval.data.dof
        dof_enabled = bool(dof.use_dof)
        lens_mm = max(float(getattr(cam_eval.data, "lens", 50.0)), 1.0e-4)
        focus_distance = camera_focus_distance(cam_eval, depsgraph)
        fstop = max(float(dof.aperture_fstop), 1.0e-5)
        aperture_blades = int(dof.aperture_blades)
        aperture_rotation = float(dof.aperture_rotation)
        aperture_ratio = max(float(dof.aperture_ratio), 1.0e-5)

    aperture_radius = (
        0.0
        if not dof_enabled
        else (
            1.0 / (2.0 * fstop)
            if cam_type == "ORTHO"
            else lens_mm * 1.0e-3 / (2.0 * fstop)
        )
    )
    xform = column_major(cam_to_world)

    # Omit sensor_width/height so load_blender_scene_from_document uses the exact
    # rv3d.window_matrix FOV rather than deriving one from physical sensor size.
    json_cam: dict[str, Any] = {
        "name": "__ViewportCamera__",
        "type": cam_type,
        "transform": xform,
        "angle": angle_y,
        "angle_x": angle_x,
        "angle_y": angle_y,
        "sensor_fit": "HORIZONTAL",
        "pixel_aspect_x": pixel_aspect_x,
        "pixel_aspect_y": pixel_aspect_y,
        "ortho_scale": ortho_scale,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "clip_start": near_clip,
        "clip_end": far_clip,
        "dof": {
            "enabled": dof_enabled,
            "focus_distance": focus_distance,
            "fstop": fstop,
            "blades": aperture_blades,
            "rotation": aperture_rotation,
            "ratio": aperture_ratio,
        },
    }
    if dof_enabled:
        json_cam["lens"] = lens_mm

    capi_cam = bridge.PsyclesCameraDesc(
        projection=capi_proj,
        sensor_fit=bridge.PSYCLES_SENSOR_FIT_HORIZONTAL,
        transform=(ctypes.c_float * 16)(*xform),
        field_of_view=angle_y,
        horizontal_field_of_view=angle_x,
        orthographic_scale=ortho_scale,
        lens_shift_x=shift_x,
        lens_shift_y=shift_y,
        near_clip=near_clip,
        far_clip=far_clip,
        aperture_radius=aperture_radius,
        focal_distance=focus_distance,
        aperture_blades=aperture_blades,
        aperture_rotation=aperture_rotation,
        aperture_ratio=aperture_ratio,
        has_sensor=0,
        reserved=(ctypes.c_uint8 * 3)(0, 0, 0),
        sensor_width_mm=0.0,
        sensor_height_mm=0.0,
        sensor_lens_mm=0.0,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
    )
    signature = (
        width,
        height,
        cam_type,
        tuple(round(v, 7) for v in xform),
        round(angle_x, 7),
        round(angle_y, 7),
        round(ortho_scale, 7),
        round(shift_x, 7),
        round(shift_y, 7),
        round(near_clip, 6),
        round(far_clip, 6),
        round(pixel_aspect_y, 7),
        dof_enabled,
        round(aperture_radius, 7) if dof_enabled else 0.0,
        round(focus_distance, 6) if dof_enabled else 0.0,
    )
    return json_cam, capi_cam, signature


def _camera_from_scene(
    scene: Any,
    depsgraph: Any,
    source_camera: Any | None,
) -> dict[str, Any] | None:
    if source_camera is None:
        return None
    camera = source_camera.evaluated_get(depsgraph)
    return {
        "name": camera.name,
        "type": camera.data.type,
        "transform": column_major(normalized_transform(camera.matrix_world)),
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
            "focus_distance": camera_focus_distance(camera, depsgraph),
            "fstop": float(camera.data.dof.aperture_fstop),
            "blades": int(camera.data.dof.aperture_blades),
            "rotation": float(camera.data.dof.aperture_rotation),
            "ratio": float(camera.data.dof.aperture_ratio),
        },
    }


def extract_scene_camera_capi(
    scene: Any,
    depsgraph: Any,
    source_camera: Any | None = None,
) -> bridge.PsyclesCameraDesc | None:
    cam_obj = source_camera if source_camera is not None else scene.camera
    if cam_obj is None:
        return None
    camera = cam_obj.evaluated_get(depsgraph)
    cam_data = camera.data
    cam_type = str(cam_data.type)
    if cam_type == "ORTHO":
        capi_proj = bridge.PSYCLES_PROJECTION_ORTHOGRAPHIC
    elif cam_type == "PANO":
        capi_proj = bridge.PSYCLES_PROJECTION_PANORAMA
    else:
        capi_proj = bridge.PSYCLES_PROJECTION_PERSPECTIVE

    fit_name = str(cam_data.sensor_fit)
    if fit_name == "HORIZONTAL":
        sensor_fit = bridge.PSYCLES_SENSOR_FIT_HORIZONTAL
    elif fit_name == "VERTICAL":
        sensor_fit = bridge.PSYCLES_SENSOR_FIT_VERTICAL
    else:
        sensor_fit = bridge.PSYCLES_SENSOR_FIT_AUTO

    xform = column_major(normalized_transform(camera.matrix_world))
    dof = cam_data.dof
    dof_enabled = bool(dof.use_dof)
    lens_mm = float(cam_data.lens)
    sensor_w = float(cam_data.sensor_width)
    sensor_h = float(cam_data.sensor_height)
    f_stop = max(float(dof.aperture_fstop), 1.0e-5)
    if not dof_enabled:
        aperture_radius = 0.0
    elif cam_type == "ORTHO":
        aperture_radius = 1.0 / (2.0 * f_stop)
    else:
        aperture_radius = lens_mm * 1.0e-3 / (2.0 * f_stop)

    has_sensor = 1 if (lens_mm > 0.0 and sensor_w > 0.0 and sensor_h > 0.0) else 0
    return bridge.PsyclesCameraDesc(
        projection=capi_proj,
        sensor_fit=sensor_fit,
        transform=(ctypes.c_float * 16)(*xform),
        field_of_view=float(cam_data.angle_y),
        horizontal_field_of_view=float(cam_data.angle_x),
        orthographic_scale=float(cam_data.ortho_scale),
        lens_shift_x=float(cam_data.shift_x),
        lens_shift_y=float(cam_data.shift_y),
        near_clip=max(float(cam_data.clip_start), 1.0e-4),
        far_clip=max(float(cam_data.clip_end), 1.0e-3),
        aperture_radius=aperture_radius,
        focal_distance=camera_focus_distance(camera, depsgraph),
        aperture_blades=int(dof.aperture_blades),
        aperture_rotation=float(dof.aperture_rotation),
        aperture_ratio=max(float(dof.aperture_ratio), 1.0e-5),
        has_sensor=has_sensor,
        reserved=(ctypes.c_uint8 * 3)(0, 0, 0),
        sensor_width_mm=sensor_w if has_sensor else 0.0,
        sensor_height_mm=sensor_h if has_sensor else 0.0,
        sensor_lens_mm=lens_mm if has_sensor else 0.0,
        pixel_aspect_x=max(float(scene.render.pixel_aspect_x), 1.0e-5),
        pixel_aspect_y=max(float(scene.render.pixel_aspect_y), 1.0e-5),
    )


def compute_scene_content_hash(
    payload: dict[str, Any],
    geom_bytes: bytes | bytearray | memoryview,
) -> bytes:
    """Hash scene contents excluding dynamic camera & RenderSettings-only fields."""
    render_src = payload.get("render") or {}
    cycles_src = render_src.get("cycles") or {}
    cycles_stripped = {
        k: v
        for k, v in cycles_src.items()
        if k
        not in {
            "samples",
            "preview_samples",
            "seed",
            "effective_seed",
            "seed_frame",
            "seed_subframe",
        }
    }
    cm_src = render_src.get("color_management") or {}
    canonical = {
        "schema": payload.get("schema"),
        "cycles_sync": payload.get("cycles_sync"),
        "source": payload.get("source"),
        "blender_build": payload.get("blender_build"),
        "render": {
            "color_management": {
                "shader_transforms": cm_src.get("shader_transforms"),
            },
            "cycles": cycles_stripped,
        },
        "geometries": payload.get("geometries"),
        "curve_geometries": payload.get("curve_geometries"),
        "instances": payload.get("instances"),
        "materials": payload.get("materials"),
        "lights": payload.get("lights"),
        "world": payload.get("world"),
        "world_environment": payload.get("world_environment"),
        "node_groups": payload.get("node_groups"),
        "images": payload.get("images"),
    }
    ext_stats: list[tuple[str, int | None, int | None]] = []
    for img in payload.get("images") or ():
        path_str = img.get("path")
        if path_str:
            try:
                st = os.stat(path_str)
                ext_stats.append((path_str, int(st.st_mtime_ns), int(st.st_size)))
            except OSError:
                ext_stats.append((path_str, None, None))

    h = hashlib.blake2b(digest_size=16)
    h.update(json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    if ext_stats:
        h.update(
            json.dumps(ext_stats, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
    h.update(memoryview(geom_bytes))
    return h.digest()


def export_depsgraph_memory(
    depsgraph: Any,
    scene: Any | None = None,
    view_layer: Any | None = None,
    source_camera: Any | None = None,
    override_camera: dict[str, Any] | None = None,
    render_width: int | None = None,
    render_height: int | None = None,
) -> tuple[bytes, bytearray, str]:
    json_bytes, geom_bytes, asset_base_dir, _hash = export_depsgraph_with_hash(
        depsgraph,
        scene=scene,
        view_layer=view_layer,
        source_camera=source_camera,
        override_camera=override_camera,
        render_width=render_width,
        render_height=render_height,
    )
    return json_bytes, geom_bytes, asset_base_dir


def export_depsgraph_with_hash(
    depsgraph: Any,
    scene: Any | None = None,
    view_layer: Any | None = None,
    source_camera: Any | None = None,
    override_camera: dict[str, Any] | None = None,
    render_width: int | None = None,
    render_height: int | None = None,
    mesh_cache: Any = None,
) -> tuple[bytes, bytearray, str, bytes]:
    scene = scene if scene is not None else depsgraph.scene
    view_layer = view_layer if view_layer is not None else depsgraph.view_layer
    if mesh_cache is not None:
        mesh_cache.prepare_for_scene(scene)
        mesh_cache.invalidate_from_depsgraph(depsgraph)
    mode = str(getattr(depsgraph, "mode", "RENDER"))
    is_viewport = mode == "VIEWPORT"
    if mode not in {"RENDER", "VIEWPORT"}:
        raise RuntimeError(f"Unsupported Blender depsgraph mode: {mode!r}")

    shader_color_space = cycles_shader_color_space()
    rec709_to_rgb = tuple(
        tuple(float(component) for component in row)
        for row in shader_color_space["rec709_to_rgb"]
    )
    pointiness_cache: dict[int, bool] = {}
    geometries: list[dict[str, Any]] = []
    geometry_by_source: dict[tuple[Any, ...], int] = {}
    curve_geometries: list[dict[str, Any]] = []
    curve_geometry_by_source: dict[tuple[Any, ...], int] = {}
    instances: list[dict[str, Any]] = []
    lights: list[dict[str, Any]] = []
    light_shader_indices, material_shader_indices = cycles_shader_indices(depsgraph)
    light_groups = {
        group.name: index for index, group in enumerate(view_layer.lightgroups)
    }
    cycles_object_index = 0
    cycles_primitive_offset = 0
    cycles_curve_offset = 0
    cycles_curve_segment_offset = 0
    curve_shape = str(scene.cycles_curves.shape)
    curve_subdivisions = int(scene.cycles_curves.subdivisions)
    particle_sources = _ParticleSourceRegistry(scene)
    cycles_uses_light_linking = False

    stream = io.BytesIO()
    stream.write(b"PSYGEO2\0")
    stream.write(struct.pack("<II", 2, 0))

    for object_instance in depsgraph.object_instances:
        obj = object_instance.object
        original = obj.original if obj.original is not None else obj
        hidden = original.hide_viewport if is_viewport else original.hide_render
        if hidden:
            continue

        source_object_index: int | None = None
        if cycles_object_is_geometry(object_instance):
            source_object_index = cycles_object_index
            cycles_object_index += 1
            cycles_uses_light_linking = (
                cycles_uses_light_linking
                or _cycles_object_uses_light_linking(object_instance)
            )
        hair_object_index: int | None = None
        if (
            bool(object_instance.show_particles)
            and object_has_particle_hair(original, depsgraph, is_viewport=is_viewport)
        ):
            hair_object_index = cycles_object_index
            cycles_object_index += 1
            cycles_uses_light_linking = (
                cycles_uses_light_linking
                or _cycles_object_uses_light_linking(object_instance)
            )

        if obj.type == "LIGHT":
            if not object_instance.show_self or source_object_index is None:
                continue
            light_data = original_id(obj.data)
            light_shader_index = light_shader_indices.get(id_key(light_data))
            if light_shader_index is None:
                light_shader_index = 5
            lights.append(
                extract_light(
                    obj,
                    view_layer,
                    object_instance=object_instance,
                    particle_source=particle_sources.capture(object_instance),
                    transform=object_instance.matrix_world,
                    cycles_sync={
                        "object_index": source_object_index,
                        "light_group": _cycles_light_group(
                            object_instance, light_groups
                        ),
                        "shader_index": light_shader_index,
                        "pass_id": 0,
                    },
                )
            )
            continue

        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue

        if object_instance.show_self and source_object_index is not None:
            key = geometry_cache_key(original, scene, mode)
            geometry_index = geometry_by_source.get(key)
            if geometry_index is None:
                geometry_data = extract_mesh_geometry(
                    original,
                    depsgraph,
                    stream,
                    rec709_to_rgb,
                    is_viewport=is_viewport,
                    pointiness_cache=pointiness_cache,
                    mesh_cache=mesh_cache,
                    scene=scene,
                )
                if geometry_data is not None:
                    geometry_data["cycles_sync"] = {
                        "primitive_offset": cycles_primitive_offset,
                    }
                    cycles_primitive_offset += int(geometry_data["triangle_count"])
                    geometry_index = len(geometries)
                    geometry_by_source[key] = geometry_index
                    geometries.append(geometry_data)
            if geometry_index is not None:
                instances.append(
                    extract_geometry_instance(
                        object_instance,
                        "MESH",
                        geometry_index,
                        source_object_index,
                        light_groups,
                        particle_sources.capture(object_instance),
                        view_layer,
                    )
                )

        if hair_object_index is not None:
            hair_key = (
                "particle_hair",
                original.as_pointer(),
                curve_shape,
                curve_subdivisions,
            )
            curve_geometry_index = curve_geometry_by_source.get(hair_key)
            if curve_geometry_index is None:
                curve_geometry = extract_particle_hair_geometry(
                    original,
                    depsgraph,
                    stream,
                    curve_shape,
                    curve_subdivisions,
                    is_viewport=is_viewport,
                )
                if curve_geometry is not None:
                    curve_geometry["cycles_sync"] = {
                        "curve_offset": cycles_curve_offset,
                        "segment_offset": cycles_curve_segment_offset,
                    }
                    cycles_curve_offset += int(curve_geometry["curve_count"])
                    cycles_curve_segment_offset += int(curve_geometry["segment_count"])
                    curve_geometry_index = len(curve_geometries)
                    curve_geometry_by_source[hair_key] = curve_geometry_index
                    curve_geometries.append(curve_geometry)
            if curve_geometry_index is not None:
                instances.append(
                    extract_geometry_instance(
                        object_instance,
                        "CURVE",
                        curve_geometry_index,
                        hair_object_index,
                        light_groups,
                        particle_sources.capture(object_instance),
                        view_layer,
                    )
                )

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
        shader_index = material_shader_indices.get(id_key(material))
        materials.append({
            "name": material.name,
            "use_transparent_shadow": bool(
                getattr(material, "use_transparent_shadow", True)
            ),
            "use_bump_map_correction": bool(
                getattr(cycles, "use_bump_map_correction", True)
            ),
            "emission_sampling": str(getattr(cycles, "emission_sampling", "AUTO")),
            "volume_sampling": str(
                getattr(cycles, "volume_sampling", "MULTIPLE_IMPORTANCE")
            ),
            "volume_interpolation": str(
                getattr(cycles, "volume_interpolation", "LINEAR")
            ),
            "surface_render_method": getattr(material, "surface_render_method", None),
            "displacement_method": str(
                getattr(
                    material,
                    "displacement_method",
                    getattr(cycles, "displacement_method", "BUMP"),
                )
            ),
            "cycles_sync": (
                {"shader_index": shader_index, "pass_id": int(material.pass_index)}
                if shader_index is not None
                else None
            ),
            "node_tree": (
                serialize_shader_tree(material.node_tree)
                if material.use_nodes and material.node_tree
                else None
            ),
        })

    if override_camera is not None:
        camera_data = override_camera
    else:
        cam_obj = source_camera if source_camera is not None else scene.camera
        camera_data = _camera_from_scene(scene, depsgraph, cam_obj)

    pixel_filter_type = str(scene.cycles.pixel_filter_type)
    filter_width = (
        1.0 if pixel_filter_type == "BOX" else float(scene.cycles.filter_width)
    )
    width = (
        int(render_width)
        if render_width is not None
        else int(scene.render.resolution_x)
    )
    height = (
        int(render_height)
        if render_height is not None
        else int(scene.render.resolution_y)
    )
    percentage = 100 if render_width is not None else int(scene.render.resolution_percentage)
    images = _export_images_memory(stream)

    payload = {
        "schema": "psycles.blender-scene.v2",
        "cycles_sync": {
            "object_count": cycles_object_index + (1 if scene.world is not None else 0),
            "uses_light_linking": cycles_uses_light_linking,
        },
        "source": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "blender_build": blender_build_identity.current(bpy.app),
        "frame": scene.frame_current,
        "camera": camera_data,
        "render": {
            "width": width,
            "height": height,
            "percentage": percentage,
            "transparent": scene.render.film_transparent,
            "pixel_filter_type": pixel_filter_type,
            "filter_width": filter_width,
            "pass_alpha_threshold": float(view_layer.pass_alpha_threshold),
            "color_management": {
                "display_device": scene.display_settings.display_device,
                "view_transform": scene.view_settings.view_transform,
                "look": scene.view_settings.look,
                "exposure": float(scene.view_settings.exposure),
                "gamma": float(scene.view_settings.gamma),
                "use_curve_mapping": bool(scene.view_settings.use_curve_mapping),
                "sequencer_color_space": scene.sequencer_colorspace_settings.name,
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
                    getattr(getattr(scene.world, "cycles", None), "sampling_method", "AUTOMATIC")
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
                "max_bounces": int(getattr(scene.world.cycles, "max_bounces", 1024)),
                "use_shadows": bool(getattr(scene.world.cycles, "use_shadows", True)),
                "visibility": {
                    "camera": bool(getattr(scene.world.cycles_visibility, "camera", True)),
                    "diffuse": bool(getattr(scene.world.cycles_visibility, "diffuse", True)),
                    "glossy": bool(getattr(scene.world.cycles_visibility, "glossy", True)),
                    "transmission": bool(
                        getattr(scene.world.cycles_visibility, "transmission", True)
                    ),
                    "shadow": bool(getattr(scene.world.cycles_visibility, "shadow", True)),
                    "volume_scatter": bool(
                        getattr(scene.world.cycles_visibility, "scatter", True)
                    ),
                },
                "cycles_sync": {
                    "shader_index": CYCLES_BACKGROUND_SHADER_INDEX,
                    "pass_id": 0,
                    "object_index": cycles_object_index,
                    "light_group": light_groups.get(
                        str(getattr(scene.world, "lightgroup", "")), -1
                    ),
                },
                "node_tree": (
                    serialize_shader_tree(scene.world.node_tree, world=True)
                    if scene.world.use_nodes
                    else None
                ),
            }
            if scene.world
            else None
        ),
        "world_environment": None,
        "node_groups": [
            serialize_shader_tree(group)
            for group in sorted(bpy.data.node_groups, key=lambda item: item.name)
            if group.bl_idname == "ShaderNodeTree"
        ],
        "images": images,
    }
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    geom_bytes = bytearray(stream.getvalue())
    content_hash = compute_scene_content_hash(payload, geom_bytes)
    asset_base_dir = (
        str(pathlib.Path(bpy.data.filepath).resolve().parent)
        if bpy.data.filepath
        else str(pathlib.Path.cwd())
    )
    return json_bytes, geom_bytes, asset_base_dir, content_hash
