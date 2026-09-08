"""Create original-Blender candidates deleting one authored material link."""
import argparse
import json
from pathlib import Path
import sys

import bpy


def trim(tree):
    keep = {node for node in tree.nodes if node.type in ("OUTPUT_MATERIAL", "GROUP_OUTPUT") and node.is_active_output}
    pending = list(keep)
    while pending:
        node = pending.pop()
        for socket in node.inputs:
            for link in socket.links:
                if link.from_node not in keep:
                    keep.add(link.from_node)
                    pending.append(link.from_node)
    for node in list(tree.nodes):
        if node not in keep:
            tree.nodes.remove(node)


def clone_groups(tree):
    for node in tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            node.node_tree = node.node_tree.copy()
            clone_groups(node.node_tree)


def trees(tree, path=()):
    yield path, tree
    for node in tree.nodes:
        if node.type == "GROUP" and node.node_tree:
            yield from trees(node.node_tree, (*path, node.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("material")
    parser.add_argument("output", type=Path)
    parser.add_argument("--nested", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(args.source.resolve()), link=False) as (available, requested):
        assert args.material in available.materials
        requested.materials = [args.material]
    original = requested.materials[0]
    # Do not collide with a candidate name when continuing from a prior batch.
    original.name = "__reduction_source__"
    trim(original.node_tree)
    links = [(*path, (link.from_node.name, link.from_socket.identifier,
                     link.to_node.name, link.to_socket.identifier))
             for path, tree in trees(original.node_tree) if args.nested or not path
             for link in tree.links]
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    records = []
    for index, cut in enumerate([None, *links]):
        material = original.copy()
        material.name = f"candidate-{index:03d}"
        tree = material.node_tree
        if args.nested:
            clone_groups(tree)
        if cut is not None:
            target = tree
            for group_name in cut[:-1]:
                target = target.nodes[group_name].node_tree
            link = next(link for link in target.links if
                (link.from_node.name, link.from_socket.identifier,
                 link.to_node.name, link.to_socket.identifier) == cut[-1])
            target.links.remove(link)
        trim(tree)
        if args.nested:
            for _, nested in list(trees(tree))[::-1]:
                trim(nested)
        contents = [nested for _, nested in trees(tree)] if args.nested else [tree]
        records.append({"name": material.name, "cut": cut,
                        "nodes": sum(len(n.nodes) for n in contents),
                        "links": sum(len(n.links) for n in contents)})
        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 8, index // 8, 0))
        bpy.context.object.data.materials.append(material)
    bpy.ops.object.camera_add(location=(3.5, 2, 10))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 12
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    args.output.with_suffix(".json").write_text(json.dumps({
        "source": str(args.source), "material": args.material, "candidates": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
