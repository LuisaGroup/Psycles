"""Make the Cycles zero-BSDF termination witness, without a reference renderer.

Run in Blender 5.2.1: blender -b --python this.py -- /path/to/probe.blend
Three orthographic panels show zero velvet, finite velvet, and red diffuse.
"""

from pathlib import Path
import sys

import bpy


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 12
    scene.render.resolution_y = 4
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 16
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    scene.cycles.auto_scrambling_distance = False
    scene.cycles.scrambling_distance = 1.0
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 4
    scene.world = bpy.data.worlds.new("Constant world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    camera = bpy.data.cameras.new("Orthographic")
    camera.type = "ORTHO"
    camera.ortho_scale = 6
    camera_object = bpy.data.objects.new("Camera", camera)
    scene.collection.objects.link(camera_object)
    camera_object.location = (0, 0, 3)
    scene.camera = camera_object
    for index, roughness in enumerate((0.01, 0.5, None)):
        material = bpy.data.materials.new(("Zero velvet", "Finite velvet", "Red diffuse")[index])
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        if roughness is None:
            bsdf = tree.nodes.new("ShaderNodeBsdfDiffuse")
            bsdf.inputs["Color"].default_value = (0.5, 0, 0, 1)
        else:
            bsdf = tree.nodes.new("ShaderNodeBsdfSheen")
            bsdf.distribution = "ASHIKHMIN"
            bsdf.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
            bsdf.inputs["Roughness"].default_value = roughness
        tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=2, location=((index - 1) * 2, 0, 0))
        bpy.context.object.name = material.name
        bpy.context.object.data.materials.append(material)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))


if __name__ == "__main__":
    main()
