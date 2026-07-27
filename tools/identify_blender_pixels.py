"""Identify Blender scene geometry under top-left-origin render pixels.

Usage:

    blender scene.blend --background --python identify_blender_pixels.py -- \
      WIDTH HEIGHT X,Y [X,Y ...]

The ray uses the active camera's exact projection matrix and the pixel center.
Depth-of-field lens jitter is intentionally omitted; this tool is for locating
spatially clustered differential outliers, not for rendering a reference.
"""

from __future__ import annotations

import sys

import bpy
from mathutils import Vector


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) < 3:
        raise SystemExit("expected WIDTH HEIGHT X,Y [X,Y ...]")
    width = int(args[0])
    height = int(args[1])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("scene has no active camera")
    depsgraph = bpy.context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(
        depsgraph,
        x=width,
        y=height,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    inverse_projection = projection.inverted()
    camera_to_world = camera.matrix_world
    origin = camera_to_world.translation

    for coordinate in args[2:]:
        x_text, y_text = coordinate.split(",", 1)
        x = int(x_text)
        y = int(y_text)
        ndc_x = 2.0 * (float(x) + 0.5) / float(width) - 1.0
        ndc_y = 1.0 - 2.0 * (float(y) + 0.5) / float(height)
        camera_point = inverse_projection @ Vector(
            (ndc_x, ndc_y, -1.0, 1.0)
        )
        camera_point /= camera_point.w
        world_point = camera_to_world @ camera_point
        direction = (world_point.xyz - origin).normalized()
        hit, location, normal, face, obj, matrix = scene.ray_cast(
            depsgraph, origin, direction
        )
        if not hit:
            print(f"{x},{y}: miss")
            continue
        materials = [
            slot.material.name if slot.material else "<none>"
            for slot in obj.material_slots
        ]
        print(
            f"{x},{y}: object={obj.name!r} face={face} "
            f"location={tuple(round(v, 6) for v in location)} "
            f"normal={tuple(round(v, 6) for v in normal)} "
            f"materials={materials}"
        )


if __name__ == "__main__":
    _main()
