"""Build the Blender 5.2.1 hidden-input provenance regression scene.

Run in Blender with -- /new/path/probe.blend. Render/dump with original
Cycles, then export with export_psycles_scene.py. No shader is evaluated here.
"""
from pathlib import Path
import sys

import bpy


def interface(tree, name, direction, kind, *, hidden=False):
    socket = tree.interface.new_socket(
        name=name, in_out=direction, socket_type=kind)
    if direction == "INPUT":
        socket.hide_value = hidden
    return socket


def group(name, *, hidden, kind="NodeSocketVector", mode="direct"):
    tree = bpy.data.node_groups.new(name, "ShaderNodeTree")
    interface(tree, "Normal", "INPUT", kind, hidden=hidden)
    interface(tree, "Shader", "OUTPUT", "NodeSocketShader")
    inp = tree.nodes.new("NodeGroupInput")
    out = tree.nodes.new("NodeGroupOutput")
    value = inp.outputs["Normal"]
    if mode == "nested":
        child = tree.nodes.new("ShaderNodeGroup")
        child.node_tree = group(name + " child", hidden=True)
        tree.links.new(value, child.inputs["Normal"])
        tree.links.new(child.outputs["Shader"], out.inputs["Shader"])
        return tree
    if mode == "reroute":
        reroute = tree.nodes.new("NodeReroute")
        tree.links.new(value, reroute.inputs[0])
        value = reroute.outputs[0]
    if mode in {"computed", "muted"}:
        math = tree.nodes.new("ShaderNodeVectorMath")
        math.operation = "ADD"
        math.inputs[1].default_value = (0.25, 0.5, 0.75)
        math.mute = mode == "muted"
        tree.links.new(value, math.inputs[0])
        value = math.outputs["Vector"]
    if mode == "bump":
        bump = tree.nodes.new("ShaderNodeBump")
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(geometry.outputs["Position"], separate.inputs[0])
        tree.links.new(separate.outputs["X"], bump.inputs["Height"])
        tree.links.new(value, bump.inputs["Normal"])
        value = bump.outputs["Normal"]
    bsdf = tree.nodes.new("ShaderNodeBsdfAnisotropic")
    bsdf.distribution = "GGX"
    bsdf.inputs["Roughness"].default_value = 0.25
    tree.links.new(value, bsdf.inputs["Normal"])
    tree.links.new(bsdf.outputs[0], out.inputs["Shader"])
    return tree


def main():
    path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if path.exists():
        raise FileExistsError(path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cases = [
        ("hidden-zero", True, "direct", "default", False),
        ("hidden-nonzero", True, "direct", "nonzero", False),
        ("visible-zero", False, "direct", "default", False),
        ("visible-nonzero", False, "direct", "nonzero", False),
        ("linked-zero", True, "direct", "linked-zero", False),
        ("linked-nonzero", True, "direct", "linked-nonzero", False),
        ("linked-geometry", True, "direct", "geometry", False),
        ("hidden-nested", True, "nested", "default", False),
        ("visible-nested", False, "nested", "nonzero", False),
        ("hidden-reroute", True, "reroute", "default", False),
        ("hidden-muted", True, "muted", "default", False),
        ("hidden-computed", True, "computed", "default", False),
        ("hidden-float-conversion", True, "direct", "nonzero", True),
        ("hidden-bump", True, "bump", "default", False),
        ("linked-bump", True, "bump", "linked-nonzero", False),
    ]
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    for index, (name, hidden, mode, source, scalar) in enumerate(cases):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        out = tree.nodes.new("ShaderNodeOutputMaterial")
        instance = tree.nodes.new("ShaderNodeGroup")
        instance.node_tree = group(
            name, hidden=hidden, mode=mode,
            kind="NodeSocketFloat" if scalar else "NodeSocketVector")
        if source == "nonzero":
            instance.inputs["Normal"].default_value = 0.5 if scalar else (0.25, 0.5, 0.75)
        elif source.startswith("linked-"):
            vector = tree.nodes.new("ShaderNodeCombineXYZ")
            if source == "linked-nonzero":
                for socket, value in zip(vector.inputs, (0.25, 0.5, 0.75)):
                    socket.default_value = value
            tree.links.new(vector.outputs[0], instance.inputs["Normal"])
        elif source == "geometry":
            geometry = tree.nodes.new("ShaderNodeNewGeometry")
            tree.links.new(geometry.outputs["Normal"], instance.inputs["Normal"])
        tree.links.new(instance.outputs["Shader"], out.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 5, index // 5, 0))
        bpy.context.object.data.materials.append(material)
    bpy.ops.object.camera_add(location=(2, 1, 5))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 6
    scene.render.resolution_x = 20
    scene.render.resolution_y = 12
    scene.render.resolution_percentage = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
