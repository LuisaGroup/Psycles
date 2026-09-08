"""Create original-Blender Light Path / bump scheduling reductions.

This constructs input graphs only. Original Cycles supplies the rendered
image and SVM stream; no expected shader evaluation is implemented here.
"""
from pathlib import Path
import sys

import bpy


def height(tree, color, minimal=False, clamp=True):
    light = tree.nodes.new("ShaderNodeLightPath")
    add = tree.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.use_clamp = clamp
    if minimal:
        add.inputs[1].default_value = 0.125
        tree.links.new(light.outputs["Ray Depth"], add.inputs[0])
    else:
        less = tree.nodes.new("ShaderNodeMath")
        less.operation = "LESS_THAN"
        less.use_clamp = True
        less.inputs[1].default_value = 2.0
        tree.links.new(light.outputs["Ray Depth"], less.inputs[0])
        tree.links.new(less.outputs[0], add.inputs[0])
        tree.links.new(light.outputs["Is Diffuse Ray"], add.inputs[1])
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 0.125
    tree.links.new(color, scale.inputs[0])
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.0, 0.0, 0.0, 1.0)
    tree.links.new(add.outputs[0], mix.inputs[0])
    tree.links.new(scale.outputs[0], mix.inputs[7])
    return mix.outputs[2]


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
    for grouped in (False, True):
        for shared in (False, True):
            for mode in ("SURFACE", "NORMAL", "BUMP", "BOTH"):
                name = f"{'group' if grouped else 'direct'}-{'shared' if shared else 'single'}-{mode}"
                material = bpy.data.materials.new(name)
                material.use_nodes = True
                material.displacement_method = "BOTH" if mode == "BOTH" else "BUMP"
                tree = material.node_tree
                tree.nodes.clear()
                output = tree.nodes.new("ShaderNodeOutputMaterial")
                diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
                uv = tree.nodes.new("ShaderNodeUVMap")
                checker = tree.nodes.new("ShaderNodeTexChecker")
                tree.links.new(uv.outputs[0], checker.inputs["Vector"])
                color = checker.outputs["Color"]
                if grouped:
                    group = bpy.data.node_groups.new(name, "ShaderNodeTree")
                    group.interface.new_socket(name="Color", in_out="INPUT", socket_type="NodeSocketFloat")
                    group.interface.new_socket(name="Height", in_out="OUTPUT", socket_type="NodeSocketFloat")
                    incoming = group.nodes.new("NodeGroupInput")
                    outgoing = group.nodes.new("NodeGroupOutput")
                    group.links.new(height(group, incoming.outputs[0]), outgoing.inputs[0])
                    instance = tree.nodes.new("ShaderNodeGroup")
                    instance.node_tree = group
                    tree.links.new(color, instance.inputs[0])
                    result = instance.outputs[0]
                else:
                    result = height(tree, color)
                if shared:
                    tree.links.new(color, diffuse.inputs["Color"])
                if mode == "SURFACE":
                    tree.links.new(result, diffuse.inputs["Roughness"])
                elif mode == "NORMAL":
                    bump = tree.nodes.new("ShaderNodeBump")
                    tree.links.new(result, bump.inputs["Height"])
                    tree.links.new(bump.outputs[0], diffuse.inputs["Normal"])
                else:
                    displacement = tree.nodes.new("ShaderNodeDisplacement")
                    tree.links.new(result, displacement.inputs["Height"])
                    tree.links.new(displacement.outputs[0], output.inputs["Displacement"])
                tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
                bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
                bpy.context.object.data.materials.append(material)
                index += 1
    for clamp in (False, True):
        material = bpy.data.materials.new(f"minimal-clamp-{int(clamp)}")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        uv = tree.nodes.new("ShaderNodeUVMap")
        tree.links.new(height(tree, uv.outputs[0], minimal=True, clamp=clamp), diffuse.inputs["Roughness"])
        tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
        bpy.context.object.data.materials.append(material)
        index += 1
    bpy.ops.object.camera_add(location=(1.5, 2.0, 8))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 6
    scene.render.resolution_x = 16
    scene.render.resolution_y = 20
    scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
