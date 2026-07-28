"""Render the Cycles semantics of an unconnected Material Output surface.

Run through Blender:

    blender --background --factory-startup \
      --python tools/probe_cycles_empty_surface.py -- output.exr
"""

from __future__ import annotations

import pathlib
import sys

import bpy


def _arguments() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def main() -> None:
    arguments = _arguments()
    if len(arguments) != 1:
        raise SystemExit("expected output EXR path")
    output = pathlib.Path(arguments[0]).resolve()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    scene.render.resolution_x = 16
    scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    image_settings = scene.render.image_settings
    if hasattr(image_settings, "media_type"):
        image_settings.media_type = "MULTI_LAYER_IMAGE"
    image_settings.file_format = "OPEN_EXR_MULTILAYER"
    image_settings.color_mode = "RGBA"
    image_settings.color_depth = "32"
    scene.render.film_transparent = False
    scene.render.filepath = str(output)
    scene.view_layers[0].use_pass_normal = True
    scene.view_layers[0].use_pass_diffuse_color = True

    world = bpy.data.worlds.new("White World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 1.0
    scene.world = world

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = (0.0, 0.0, 3.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    bpy.ops.mesh.primitive_plane_add(size=10.0)
    plane = bpy.context.object
    material = bpy.data.materials.new("Empty Surface")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    plane.data.materials.append(material)

    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
