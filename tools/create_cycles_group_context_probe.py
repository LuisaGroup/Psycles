"""Create original-Blender minimal chain and persistent instance-cache controls."""
from pathlib import Path
import sys

import bpy


def group(name, checker=False, nested=False):
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    tree.interface.new_socket(name="Value", in_out="INPUT", socket_type="NodeSocketFloat")
    outputs = (("Color", "NodeSocketColor"), ("Fac", "NodeSocketFloat")) if checker else (
        ("Value", "NodeSocketFloat"),)
    for label, kind in outputs:
        tree.interface.new_socket(name=label, in_out="OUTPUT", socket_type=kind)
    incoming = tree.nodes.new("NodeGroupInput")
    outgoing = tree.nodes.new("NodeGroupOutput")
    if nested:
        child = tree.nodes.new("ShaderNodeGroup")
        child.node_tree = group(name + "-inner", checker=checker)
        tree.links.new(incoming.outputs["Value"], child.inputs["Value"])
        for label, _ in outputs:
            tree.links.new(child.outputs[label], outgoing.inputs[label])
    elif checker:
        texture = make_checker(tree)
        tree.links.new(incoming.outputs["Value"], texture.inputs["Scale"])
        for label, _ in outputs:
            tree.links.new(texture.outputs[label], outgoing.inputs[label])
    else:
        tree.links.new(incoming.outputs["Value"], outgoing.inputs["Value"])
    return tree


def make_checker(tree):
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    texture = tree.nodes.new("ShaderNodeTexChecker")
    texture.inputs["Color1"].default_value = (0.125, 0.25, 0.5, 1)
    texture.inputs["Color2"].default_value = (0.75, 0.5, 0.25, 1)
    tree.links.new(geometry.outputs["Position"], texture.inputs["Vector"])
    return texture


def main():
    path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if path.exists():
        raise FileExistsError(path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    cases = [("minimal", mode, False) for mode in ("direct", "single", "chain", "nested")]
    cases += [("outputs", mode, bump) for mode in ("direct", "single", "pair", "chain", "nested")
              for bump in (False, True)]
    assert len(cases) == 14
    for index, (kind, mode, bump) in enumerate(cases):
        name = f"{kind}-{mode}-bump{int(bump)}"
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        if kind == "minimal":
            light_path = tree.nodes.new("ShaderNodeLightPath")
            value = light_path.outputs["Ray Depth"]
            if mode != "direct":
                definition = group(name, nested=mode == "nested")
                for _ in range(1 if mode == "single" else 2):
                    instance = tree.nodes.new("ShaderNodeGroup")
                    instance.node_tree = definition
                    tree.links.new(value, instance.inputs["Value"])
                    value = instance.outputs["Value"]
            tree.links.new(value, diffuse.inputs["Roughness"])
        else:
            if mode == "direct":
                texture = make_checker(tree)
                texture.inputs["Scale"].default_value = 2
                color, value = texture.outputs["Color"], texture.outputs["Fac"]
            else:
                definition = group(name, checker=True, nested=mode == "nested")
                first = tree.nodes.new("ShaderNodeGroup")
                first.node_tree = definition
                first.inputs["Value"].default_value = 2
                color, value = first.outputs["Color"], first.outputs["Fac"]
                if mode in ("pair", "chain"):
                    second = tree.nodes.new("ShaderNodeGroup")
                    second.node_tree = definition
                    second.inputs["Value"].default_value = 3
                    if mode == "chain":
                        tree.links.new(first.outputs["Fac"], second.inputs["Value"])
                        color = second.outputs["Color"]
                    value = second.outputs["Fac"]
            tree.links.new(color, diffuse.inputs["Color"])
            if bump:
                normal = tree.nodes.new("ShaderNodeBump")
                tree.links.new(value, normal.inputs["Height"])
                tree.links.new(normal.outputs[0], diffuse.inputs["Normal"])
            else:
                tree.links.new(value, diffuse.inputs["Roughness"])
        tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
        bpy.context.object.data.materials.append(material)
    bpy.ops.object.camera_add(location=(1.5, 1.5, 8))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 5
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
