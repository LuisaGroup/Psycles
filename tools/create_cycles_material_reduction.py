"""Derive non-destructive original-Cycles shader reductions from a material.

Input-graph transformations only: retain full material, normal root or one
shared normal/roughness root; optionally replace images and ramps with small
procedural nodes. Each candidate must be checked against original Cycles.
"""
import argparse
from pathlib import Path
import sys

import bpy


def replace_node(tree, node, replacement, input_pairs, output_pairs):
    for before, after in input_pairs:
        incoming = tuple(node.inputs[before].links)
        if incoming:
            tree.links.new(incoming[0].from_socket, replacement.inputs[after])
        else:
            replacement.inputs[after].default_value = node.inputs[before].default_value
    for before, after in output_pairs:
        for link in tuple(node.outputs[before].links):
            tree.links.new(replacement.outputs[after], link.to_socket)
    tree.nodes.remove(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--material", default="heater_mat")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(args.source.resolve()), link=False) as (available, requested):
        assert args.material in available.materials
        requested.materials = [args.material]
    original = requested.materials[0]
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    index = 0
    for images in (True, False):
        for ramps in (True, False):
            for root in ("original", "normal", "shared"):
                material = original.copy()
                material.name = f"{args.material}-images{int(images)}-ramps{int(ramps)}-{root}"
                tree = material.node_tree
                if root != "original":
                    for output in [node for node in tree.nodes if node.type == "OUTPUT_MATERIAL"]:
                        tree.nodes.remove(output)
                    output = tree.nodes.new("ShaderNodeOutputMaterial")
                    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
                    tree.links.new(tree.nodes["Bump"].outputs["Normal"], diffuse.inputs["Normal"])
                    if root == "shared":
                        tree.links.new(tree.nodes["ColorRamp.002"].outputs["Color"], diffuse.inputs["Roughness"])
                    tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
                if not images:
                    image_keys = {}
                    for node in [n for n in tree.nodes if n.type == "TEX_IMAGE"]:
                        key = (node.image.as_pointer() if node.image else 0,
                               node.interpolation, node.extension, node.projection)
                        number = image_keys.setdefault(key, len(image_keys))
                        replacement = tree.nodes.new("ShaderNodeTexChecker")
                        replacement.inputs["Scale"].default_value = number + 2
                        replace_node(tree, node, replacement, [("Vector", "Vector")],
                                     [("Color", "Color"), ("Alpha", "Fac")])
                if not ramps:
                    for node in [n for n in tree.nodes if n.type == "VALTORGB"]:
                        assert not node.outputs["Alpha"].links
                        replacement = tree.nodes.new("ShaderNodeMixRGB")
                        replacement.inputs[1].default_value = (0.125, 0.25, 0.5, 1)
                        replacement.inputs[2].default_value = (0.75, 0.5, 0.25, 1)
                        replace_node(tree, node, replacement, [("Fac", "Fac")], [("Color", "Color")])
                bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
                bpy.context.object.data.materials.append(material)
                index += 1
    bpy.ops.object.camera_add(location=(1.5, 1, 8))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 5
    scene.render.resolution_x = 16
    scene.render.resolution_y = 12
    scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))


if __name__ == "__main__":
    main()
