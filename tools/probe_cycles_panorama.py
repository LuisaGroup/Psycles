"""Probe Blender/Cycles panorama-camera direction conventions.

This is a diagnostic for the Blender scene adapter. It renders an official
Cycles world whose RGB value encodes the Geometry/Incoming vector and prints
selected raw EXR pixels. No Psycles rendering code participates.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import bpy
import numpy as np
import OpenImageIO as oiio


def _main() -> None:
    scene = bpy.data.scenes.new("__psycles_panorama_probe__")
    world = bpy.data.worlds.new("__psycles_panorama_probe_world__")
    camera_data = bpy.data.cameras.new(
        "__psycles_panorama_probe_camera_data__"
    )
    camera = bpy.data.objects.new(
        "__psycles_panorama_probe_camera__", camera_data
    )
    try:
        scene.world = world
        scene.collection.objects.link(camera)
        scene.camera = camera
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 1
        scene.render.resolution_x = 512
        scene.render.resolution_y = 256
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.film_transparent = False
        scene.use_nodes = False

        camera_data.type = "PANO"
        camera_data.panorama_type = "EQUIRECTANGULAR"
        camera.matrix_world.identity()

        world.use_nodes = True
        tree = world.node_tree
        tree.nodes.clear()
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        scale = tree.nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs["Scale"].default_value = 0.5
        add = tree.nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (0.5, 0.5, 0.5)
        background = tree.nodes.new("ShaderNodeBackground")
        output = tree.nodes.new("ShaderNodeOutputWorld")
        tree.links.new(geometry.outputs["Incoming"], scale.inputs[0])
        tree.links.new(scale.outputs[0], add.inputs[0])
        tree.links.new(add.outputs[0], background.inputs["Color"])
        tree.links.new(background.outputs[0], output.inputs["Surface"])

        bpy.context.window.scene = scene
        bpy.ops.render.render()
        with tempfile.NamedTemporaryFile(
            suffix=".exr", delete=False
        ) as file:
            temporary_path = pathlib.Path(file.name)
        try:
            bpy.data.images["Render Result"].save_render(
                str(temporary_path), scene=scene
            )
            source = oiio.ImageInput.open(str(temporary_path))
            if source is None:
                raise RuntimeError("could not open panorama probe EXR")
            try:
                pixels = np.asarray(
                    source.read_image(format=oiio.FLOAT),
                    dtype=np.float32,
                )[:, :, :3]
            finally:
                source.close()
        finally:
            temporary_path.unlink(missing_ok=True)

        for name, x, y in (
            ("top", 256, 0),
            ("bottom", 256, 255),
            ("center", 256, 128),
            ("left", 0, 128),
            ("right", 511, 128),
            ("quarter", 128, 128),
            ("three_quarter", 384, 128),
        ):
            incoming = pixels[y, x] * 2.0 - 1.0
            print(
                f"PANORAMA_PROBE {name} pixel=({x},{y}) "
                f"incoming={incoming.tolist()}"
            )
    finally:
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.cameras.remove(camera_data, do_unlink=True)
        bpy.data.worlds.remove(world, do_unlink=True)
        bpy.data.scenes.remove(scene, do_unlink=True)


if __name__ == "__main__":
    _main()
