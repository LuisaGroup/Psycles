"""Probe the active Blender world through official Cycles camera rays.

Usage:

    blender scene.blend --background --python probe_cycles_world.py -- out.json

This is a differential diagnostic, not a Psycles execution path. It records
linear Cycles radiance for the sky-node sun direction and a few fixed world
directions so the Luisa implementation can be checked without relying on a
second renderer.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile
from typing import Any

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix, Vector


def _render_direction(scene: Any, camera: Any, direction: Vector) -> list[float]:
    forward = direction.normalized()
    reference_up = (
        Vector((0.0, 1.0, 0.0))
        if abs(forward.z) > 0.999
        else Vector((0.0, 0.0, 1.0))
    )
    right = forward.cross(reference_up).normalized()
    up = right.cross(forward).normalized()
    # Assign the camera basis directly. Quaternion construction around the
    # zenith introduces tiny X/Y components; atan2 then maps those to an
    # arbitrary azimuth in Cycles' finite top LUT row.
    camera.matrix_world = Matrix(
        (
            (right.x, up.x, -forward.x, 0.0),
            (right.y, up.y, -forward.y, 0.0),
            (right.z, up.z, -forward.z, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    bpy.context.view_layer.update()
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
            raise RuntimeError("could not open the temporary Cycles EXR")
        try:
            pixels = source.read_image(format=oiio.FLOAT)
            if pixels is None:
                raise RuntimeError("could not read the temporary Cycles EXR")
            array = np.asarray(pixels, dtype=np.float32)
            if tuple(array.shape[:2]) != (1, 1):
                raise RuntimeError(
                    "world probe must contain exactly one camera ray, "
                    f"got {array.shape[1]}x{array.shape[0]}"
                )
            return [
                float(value)
                for value in np.mean(array[:, :, :3], axis=(0, 1))
            ]
        finally:
            source.close()
    finally:
        temporary_path.unlink(missing_ok=True)


def _configure_raw_cycles_probe(scene: Any) -> None:
    """Configure a deterministic raw render-layer world probe."""
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    # A single camera ray is the probe. Background evaluation is
    # deterministic and does not need Monte Carlo accumulation; using an
    # odd, one-pixel image is important because an even-sized image has no
    # center ray and averages different sky azimuths near the zenith.
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    # Blender clamps the base resolution to at least 4 pixels. A 4-pixel
    # base at 25% is the supported way to request one effective output pixel;
    # assigning 1 directly silently produces a 4x4 render instead.
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.render.resolution_percentage = 25
    scene.render.film_transparent = False
    # Some headless Blender builds expose only the multilayer OpenEXR writer.
    # Both variants preserve the same linear Combined values consumed below;
    # the assignable enum is context-dependent and can be narrower than the
    # static RNA item list, so capability is established by assignment.
    try:
        scene.render.image_settings.file_format = "OPEN_EXR"
    except TypeError:
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    # Match render_cycles_golden.py: probe the raw Cycles render layer, never
    # the scene's compositor or sequencer output. Lone Monk's compositor
    # deliberately mixes the denoised and noisy images with factor 0.3; when
    # denoising is disabled for this one-ray probe, that graph scales the
    # solar disc by 0.7 and injects a 0.3 lower-hemisphere value. That is a
    # post-process result, not Nishita radiance.
    scene.render.use_compositing = False
    scene.render.use_sequencer = False


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected one output JSON path after '--'")
    output = pathlib.Path(args[0]).resolve()
    scene = bpy.context.scene
    _configure_raw_cycles_probe(scene)

    for obj in scene.objects:
        obj.hide_render = True
    camera_data = bpy.data.cameras.new("__psycles_world_probe_camera_data__")
    # Orthographic camera rays remain exactly parallel under Cycles' pixel
    # jitter. A perspective one-pixel probe still changes azimuth near the
    # zenith, which is observable because the finite Nishita LUT's top row is
    # not azimuthally constant.
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new(
        "__psycles_world_probe_camera__", camera_data
    )
    scene.collection.objects.link(camera)
    camera.hide_render = False
    scene.camera = camera

    directions: dict[str, Vector] = {
        # Exact +Z has undefined azimuth (including signed-zero differences)
        # while the finite Nishita top LUT row is not azimuthally constant.
        # Use a fixed near-zenith direction with a well-defined azimuth.
        "near_zenith": Vector((1.0e-3, 2.0e-3, 1.0)),
        "north_horizon": Vector((0.0, 1.0, 1.0e-4)),
        "east_horizon": Vector((1.0, 0.0, 1.0e-4)),
        "ground": Vector((0.0, 0.0, -1.0)),
    }
    sky_nodes: list[dict[str, Any]] = []
    if scene.world is not None and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.bl_idname != "ShaderNodeTexSky":
                continue
            elevation = float(node.sun_elevation)
            rotation = float(node.sun_rotation)
            cycles_rotation = (-rotation) % (2.0 * math.pi)
            sun_direction = Vector(
                (
                    -math.cos(elevation) * math.sin(cycles_rotation),
                    math.cos(elevation) * math.cos(cycles_rotation),
                    math.sin(elevation),
                )
            )
            key = f"sun_{len(sky_nodes)}"
            directions[key] = sun_direction
            sky_nodes.append(
                {
                    "name": node.name,
                    "sky_type": node.sky_type,
                    "sun_elevation": elevation,
                    "sun_rotation": rotation,
                    "cycles_sun_rotation": cycles_rotation,
                    "sun_size": float(node.sun_size),
                    "sun_intensity": float(node.sun_intensity),
                    "direction": list(sun_direction),
                    "probe": key,
                }
            )

    probes = {
        name: {
            "direction": list(direction.normalized()),
            "radiance": _render_direction(scene, camera, direction),
        }
        for name, direction in directions.items()
    }
    payload = {
        "schema": "psycles.cycles-world-probe.v1",
        "source": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "samples": scene.cycles.samples,
        "sky_nodes": sky_nodes,
        "probes": probes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
