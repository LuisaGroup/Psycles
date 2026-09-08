"""Create original Blender graphs sharing Mapping and texture conversions.

Only input graphs are created. Original Cycles supplies the word-image and
GPU-render oracle; this script never computes expected shader results.
"""
from pathlib import Path
import sys

import bpy


def main():
    path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if path.exists():
        raise FileExistsError(path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    index = 0
    for socket in ("Vector", "Location", "Rotation", "Scale"):
        for source_output in ("Color", "Fac"):
            for shared in (False, True):
                name = f"mapping-{socket}-{source_output}-{'shared' if shared else 'separate'}"
                material = bpy.data.materials.new(name)
                material.use_nodes = True
                tree = material.node_tree
                tree.nodes.clear()
                output = tree.nodes.new("ShaderNodeOutputMaterial")
                diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
                uv = tree.nodes.new("ShaderNodeUVMap")
                source = tree.nodes.new("ShaderNodeTexChecker")
                source.inputs["Color1"].default_value = (0.125, 0.25, 0.5, 1)
                source.inputs["Color2"].default_value = (0.75, 0.5, 0.25, 1)
                tree.links.new(uv.outputs[0], source.inputs["Vector"])
                mapping = tree.nodes.new("ShaderNodeMapping")
                mapping.inputs["Scale"].default_value = (2, 1, 0.5)
                mapping.inputs["Location"].default_value = (0.25, 0, 0)
                tree.links.new(uv.outputs[0], mapping.inputs["Vector"])
                tree.links.new(source.outputs[source_output], mapping.inputs[socket])
                texture = tree.nodes.new("ShaderNodeTexGradient")
                tree.links.new(source.outputs[source_output] if shared else uv.outputs[0],
                               texture.inputs["Vector"])
                mix = tree.nodes.new("ShaderNodeMix")
                mix.data_type = "RGBA"
                mix.inputs[0].default_value = 0.375
                tree.links.new(mapping.outputs[0], mix.inputs[6])
                tree.links.new(texture.outputs["Color"], mix.inputs[7])
                tree.links.new(mix.outputs[2], diffuse.inputs["Color"])
                tree.links.new(texture.outputs["Fac"], diffuse.inputs["Roughness"])
                tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
                bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
                bpy.context.object.data.materials.append(material)
                index += 1
    for shared in (False, True):
        material = bpy.data.materials.new(f"minimal-mapping-{'shared' if shared else 'separate'}")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        light = tree.nodes.new("ShaderNodeLightPath")
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (2, 1, 0.5)
        gradient = tree.nodes.new("ShaderNodeTexGradient")
        tree.links.new(light.outputs["Ray Depth"], mapping.inputs["Vector"])
        tree.links.new(light.outputs["Ray Depth" if shared else "Is Diffuse Ray"], gradient.inputs["Vector"])
        tree.links.new(mapping.outputs[0], diffuse.inputs["Color"])
        tree.links.new(gradient.outputs["Fac"], diffuse.inputs["Roughness"])
        tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
        bpy.context.object.data.materials.append(material)
        index += 1
    bpy.ops.object.camera_add(location=(1.5, 2, 8))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 6
    scene.render.resolution_x = 16
    scene.render.resolution_y = 20
    scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
