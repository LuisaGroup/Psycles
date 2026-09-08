"""Create original Blender lazy Group Input and parent-context controls."""
from pathlib import Path
import sys

import bpy


def passthrough(name):
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    tree.interface.new_socket(name="Ignored", in_out="INPUT", socket_type="NodeSocketColor")
    tree.interface.new_socket(name="Used", in_out="INPUT", socket_type="NodeSocketFloat")
    tree.interface.new_socket(name="Value", in_out="OUTPUT", socket_type="NodeSocketFloat")
    incoming = tree.nodes.new("NodeGroupInput")
    outgoing = tree.nodes.new("NodeGroupOutput")
    tree.links.new(incoming.outputs["Used"], outgoing.inputs["Value"])
    return tree


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
    for mode in ("direct", "group", "nested", "chain"):
        for bump in (False, True):
            for ignored in (False, True):
                name = f"{mode}-bump{int(bump)}-ignored{int(ignored)}"
                material = bpy.data.materials.new(name)
                material.use_nodes = True
                tree = material.node_tree
                tree.nodes.clear()
                output = tree.nodes.new("ShaderNodeOutputMaterial")
                diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
                uv = tree.nodes.new("ShaderNodeUVMap")
                mapping = tree.nodes.new("ShaderNodeMapping")
                mapping.inputs["Scale"].default_value = (2, 1, 0.5)
                mapped = tree.nodes.new("ShaderNodeTexChecker")
                tree.links.new(uv.outputs[0], mapping.inputs["Vector"])
                tree.links.new(mapping.outputs[0], mapped.inputs["Vector"])
                color = tree.nodes.new("ShaderNodeMixRGB")
                color.inputs[1].default_value = (0.125, 0.25, 0.5, 1)
                color.inputs[2].default_value = (0.75, 0.5, 0.25, 1)
                tree.links.new(mapped.outputs["Color"], color.inputs[0])
                coords = tree.nodes.new("ShaderNodeTexCoord")
                other = tree.nodes.new("ShaderNodeTexChecker")
                other.inputs["Scale"].default_value = 3
                tree.links.new(coords.outputs["UV"], other.inputs["Vector"])
                mix = tree.nodes.new("ShaderNodeMix")
                mix.data_type = "RGBA"
                mix.inputs[0].default_value = 0.375
                tree.links.new(other.outputs["Color"], mix.inputs[6])
                tree.links.new(color.outputs[0], mix.inputs[7])
                value = mix.outputs[2]
                if mode != "direct":
                    definition = passthrough(name)
                    if mode == "nested":
                        wrapper = passthrough(name + "-wrapper")
                        nested = wrapper.nodes.new("ShaderNodeGroup")
                        nested.node_tree = definition
                        incoming = next(n for n in wrapper.nodes if n.type == "GROUP_INPUT")
                        outgoing = next(n for n in wrapper.nodes if n.type == "GROUP_OUTPUT")
                        wrapper.links.new(incoming.outputs["Used"], nested.inputs["Used"])
                        wrapper.links.new(incoming.outputs["Ignored"], nested.inputs["Ignored"])
                        wrapper.links.new(nested.outputs[0], outgoing.inputs[0])
                        definition = wrapper
                    for _ in range(2 if mode == "chain" else 1):
                        group = tree.nodes.new("ShaderNodeGroup")
                        group.node_tree = definition
                        tree.links.new(value, group.inputs["Used"])
                        if ignored:
                            tree.links.new(other.outputs["Color"], group.inputs["Ignored"])
                        value = group.outputs[0]
                if bump:
                    normal = tree.nodes.new("ShaderNodeBump")
                    tree.links.new(value, normal.inputs["Height"])
                    tree.links.new(normal.outputs[0], diffuse.inputs["Normal"])
                else:
                    tree.links.new(value, diffuse.inputs["Roughness"])
                tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
                bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
                bpy.context.object.data.materials.append(material)
                index += 1
    bpy.ops.object.camera_add(location=(1.5, 1.5, 8))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 5
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
