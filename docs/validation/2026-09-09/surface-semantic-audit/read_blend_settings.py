"""Read original scene feature switches absent from the current export contract.

Run in background Blender; do not render, save, mutate settings, or create an
expected shader. Source bytes are hashed before/after every open.
"""
import hashlib
import json
from pathlib import Path
import sys
import bpy


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


baseline = Path(sys.argv[sys.argv.index("--") + 1])
for name in ("barbershop", "monk", "monster", "classroom"):
    manifest = json.loads((baseline / name / "run-1/benchmark.json").read_text())
    path = Path(manifest["scene"]["blend"])
    before = digest(path)
    assert before == manifest["scene"]["sha256"]
    bpy.ops.wm.open_mainfile(filepath=str(path))
    scene = bpy.context.scene
    values = {key: getattr(scene.cycles, key) for key in (
        "film_transparent_glass", "film_transparent_roughness", "use_fast_gi",
        "use_guiding", "use_denoising", "use_light_tree")
        if hasattr(scene.cycles, key)}
    values["film_transparent"] = scene.render.film_transparent
    values["camera_motion_blur"] = scene.render.use_motion_blur
    values["light_caustics"] = [light.name for light in bpy.data.lights
                                if getattr(light.cycles, "is_caustics_light", False)]
    values["pointcloud_objects"] = [obj.name for obj in scene.objects if obj.type == "POINTCLOUD"]
    values["volume_objects"] = [obj.name for obj in scene.objects if obj.type == "VOLUME"]
    values["udim_images"] = [image.name for image in bpy.data.images if image.source == "TILED"]
    values["view_layer_use_pass_ambient_occlusion"] = bpy.context.view_layer.use_pass_ambient_occlusion
    assert before == digest(path)
    print("SURFACE_AUDIT_SETTINGS " + json.dumps({"scene": name, "source": str(path),
          "sha256": before, "settings": values}), flush=True)
