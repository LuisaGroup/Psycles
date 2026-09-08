"""Create original-Blender linked/primitive group forwarding counterexamples.

Creates graphs only. Render/dump with original Cycles HIP and export those
same inputs; this script does not evaluate shading or generate expected words.
"""
from pathlib import Path
import sys

import bpy


def source(tree, mode):
    if mode == "linked":
        texture = tree.nodes.new("ShaderNodeTexChecker")
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        tree.links.new(geometry.outputs["Position"], texture.inputs["Vector"])
        return texture.outputs["Color"]
    if mode in ("rgb-node", "gamma-rgb"):
        rgb = tree.nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (
            (-0.25, 0.0, 0.5, 1.0) if mode == "gamma-rgb" else (0.125, 0.5, 0.875, 1.0))
        return rgb.outputs[0]
    if mode == "gamma-primitive":
        xyz = tree.nodes.new("ShaderNodeCombineXYZ")
        for socket, value in zip(xyz.inputs, (-0.25, 0.0, 0.5)):
            socket.default_value = value
        return xyz.outputs[0]
    if mode == "computed":
        xyz = tree.nodes.new("ShaderNodeCombineXYZ")
        for socket, value in zip(xyz.inputs, (0.125, 0.5, 0.875)):
            socket.default_value = value
        noise = tree.nodes.new("ShaderNodeTexWhiteNoise")
        tree.links.new(xyz.outputs[0], noise.inputs["Vector"])
        return noise.outputs["Color"]
    assert mode == "default"
    return None


def gamma(tree, producer, muted=False, exponent=2.0):
    node = tree.nodes.new("ShaderNodeGamma")
    node.inputs["Gamma"].default_value = exponent
    node.mute = muted
    tree.links.new(producer, node.inputs["Color"])
    return node.outputs[0]


def group(name, route):
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    color_input = route == "output"
    tree.interface.new_socket(name="Input", in_out="INPUT",
        socket_type="NodeSocketColor" if color_input else "NodeSocketFloat")
    tree.interface.new_socket(name="Output", in_out="OUTPUT",
        socket_type="NodeSocketColor" if route == "input" else "NodeSocketFloat")
    incoming = tree.nodes.new("NodeGroupInput")
    outgoing = tree.nodes.new("NodeGroupOutput")
    producer = incoming.outputs["Input"]
    if route == "input":
        producer = gamma(tree, producer)
    elif route == "nested":
        instance = tree.nodes.new("ShaderNodeGroup")
        instance.node_tree = group(name + "/inner", "input")
        tree.links.new(producer, instance.inputs["Input"])
        producer = instance.outputs["Output"]
    elif route == "reroute":
        reroute = tree.nodes.new("NodeReroute")
        tree.links.new(producer, reroute.inputs[0])
        producer = reroute.outputs[0]
    elif route == "muted":
        producer = gamma(tree, producer, muted=True)
    elif route == "actual-float":
        math = tree.nodes.new("ShaderNodeMath")
        math.operation = "MULTIPLY"
        math.inputs[1].default_value = 2.0
        tree.links.new(producer, math.inputs[0])
        producer = math.outputs[0]
    else:
        assert route in ("output", "both")
    tree.links.new(producer, outgoing.inputs["Output"])
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
    routes = ("direct", "input", "output", "both", "nested", "reroute", "muted", "actual-float")
    cases = [(mode, route, 2.0) for mode in ("linked", "rgb-node", "computed") for route in routes]
    cases += [("default", route, 2.0) for route in ("input", "output", "both", "actual-float")]
    cases += [(mode, "direct", exponent) for mode in ("gamma-rgb", "gamma-primitive")
              for exponent in (-1.0, 0.0, 2.0)]
    assert len(cases) == 34
    for index, (mode, route, exponent) in enumerate(cases):
        name = f"{mode}-{route}" + (f"-{exponent:g}" if mode.startswith("gamma-") else "")
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        producer = source(tree, mode)
        if route != "direct":
            instance = tree.nodes.new("ShaderNodeGroup")
            instance.node_tree = group(material.name + "/group", route)
            if producer is not None:
                tree.links.new(producer, instance.inputs["Input"])
            else:
                instance.inputs["Input"].default_value = (
                    (0.125, 0.5, 0.875, 1.0) if route == "output" else 0.375)
            producer = instance.outputs["Output"]
        producer = gamma(tree, producer, exponent=exponent)
        tree.links.new(producer, diffuse.inputs["Color"])
        tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 7, index // 7, 0))
        bpy.context.object.data.materials.append(material)
    bpy.ops.object.camera_add(location=(3, 2, 8))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 9
    scene.render.resolution_x = 28
    scene.render.resolution_y = 20
    scene.render.resolution_percentage = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
