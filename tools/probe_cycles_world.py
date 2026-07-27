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
from mathutils import Vector


def _render_direction(scene: Any, camera: Any, direction: Vector) -> list[float]:
    camera.matrix_world = direction.normalized().to_track_quat(
        "-Z", "Y"
    ).to_matrix().to_4x4()
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
            return [
                float(value)
                for value in np.mean(array[:, :, :3], axis=(0, 1))
            ]
        finally:
            source.close()
    finally:
        temporary_path.unlink(missing_ok=True)


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected one output JSON path after '--'")
    output = pathlib.Path(args[0]).resolve()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.use_nodes = False

    for obj in scene.objects:
        obj.hide_render = True
    camera_data = bpy.data.cameras.new("__psycles_world_probe_camera_data__")
    camera_data.type = "PERSP"
    camera_data.angle = 1.0e-4
    camera = bpy.data.objects.new(
        "__psycles_world_probe_camera__", camera_data
    )
    scene.collection.objects.link(camera)
    camera.hide_render = False
    scene.camera = camera

    directions: dict[str, Vector] = {
        "zenith": Vector((0.0, 0.0, 1.0)),
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
