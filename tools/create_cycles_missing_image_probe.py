#!/usr/bin/env python3
"""Create the assigned-but-unloadable image probe with original Blender.

Run with Blender 5.2.1 --background --python this_file -- /new/probe.blend.
The subsequent original Cycles HIP render supplies both the SVM dump and EXR;
this script neither synthesizes expected bytecode nor evaluates a texture.
"""

from pathlib import Path
import sys

import bpy


def main() -> None:
    destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_light_tree = False
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("Black World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

    images = []
    for index, colorspace in enumerate(("sRGB", "Non-Color")):
        missing = destination.parent / f"missing-image-{index}.png"
        if missing.exists():
            raise FileExistsError(missing)
        image = bpy.data.images.new(f"Missing {colorspace}", width=8, height=8)
        image.source = "FILE"
        image.filepath = str(missing)
        image.colorspace_settings.name = colorspace
        images.append(image)

    # Repeated source with identical sampler, distinct missing source, and
    # identical source with a different sampler test ImageManager identity.
    for index, (source, interpolation) in enumerate(((0, "Linear"), (0, "Linear"),
                                                     (1, "Linear"), (0, "Closest"))):
        material = bpy.data.materials.new(f"Missing image {index}")
        material.use_nodes = True
        graph = material.node_tree
        graph.nodes.clear()
        output = graph.nodes.new("ShaderNodeOutputMaterial")
        texture = graph.nodes.new("ShaderNodeTexImage")
        texture.image = images[source]
        texture.interpolation = interpolation
        texture.extension = "CLIP"
        # Outside the image. A 1x1 magenta CLIP texture incorrectly returns
        # black here; Cycles' failed-image state returns before wrapping.
        coordinates = graph.nodes.new("ShaderNodeCombineXYZ")
        coordinates.inputs["X"].default_value = -2.0 - index
        coordinates.inputs["Y"].default_value = 3.0
        graph.links.new(coordinates.outputs[0], texture.inputs["Vector"])
        emission = graph.nodes.new("ShaderNodeEmission")
        graph.links.new(texture.outputs["Color"], emission.inputs["Color"])
        graph.links.new(texture.outputs["Alpha"], emission.inputs["Strength"])
        graph.links.new(emission.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(index % 2 - 0.5,
                                                           index // 2 - 0.5, 0.0))
        bpy.context.object.name = f"Missing image plane {index}"
        bpy.context.object.data.materials.append(material)

    camera = bpy.data.cameras.new("Probe camera")
    camera.type = "ORTHO"
    camera.ortho_scale = 2.0
    obj = bpy.data.objects.new("Probe camera", camera)
    scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, 3.0)
    scene.camera = obj
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    print("Missing-image probe:", destination)
    for image in images:
        print(image.name, tuple(image.size), image.colorspace_settings.name)


if __name__ == "__main__":
    main()
