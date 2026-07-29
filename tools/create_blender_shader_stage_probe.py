"""Create a non-destructive shader-stage diagnostic scene.

Usage:

    blender source.blend --background --python \
        create_blender_shader_stage_probe.py -- \
        output.blend MATERIAL NODE SOCKET

The selected value output is connected to an Emission closure while all other
surface materials and the world emit black. The source file is never saved.
This is intended only to locate the first divergent node evaluation between
Cycles and Psycles; production scene export continues to preserve the original
closure graphs.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import bpy


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Create a Blender shader-stage diagnostic scene"
    )
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("material")
    parser.add_argument("node")
    parser.add_argument("socket")
    return parser.parse_args(argv)


def _socket_by_name_or_identifier(
    sockets: Any, requested: str
) -> Any:
    for socket in sockets:
        if socket.name == requested or socket.identifier == requested:
            return socket
    available = ", ".join(
        f"{socket.name!r}/{socket.identifier!r}" for socket in sockets
    )
    raise KeyError(
        f"socket {requested!r} was not found; available: {available}"
    )


def _active_output(tree: Any, output_type: str) -> Any:
    candidates = [
        node
        for node in tree.nodes
        if node.type == output_type
    ]
    if not candidates:
        node_id = (
            "ShaderNodeOutputWorld"
            if output_type == "OUTPUT_WORLD"
            else "ShaderNodeOutputMaterial"
        )
        return tree.nodes.new(node_id)
    return next(
        (
            node
            for node in candidates
            if getattr(node, "is_active_output", False)
        ),
        candidates[0],
    )


def _replace_surface_with_emission(
    material: Any, source_socket: Any | None
) -> None:
    material.use_nodes = True
    tree = material.node_tree
    output = _active_output(tree, "OUTPUT_MATERIAL")
    surface = _socket_by_name_or_identifier(output.inputs, "Surface")
    for link in tuple(surface.links):
        tree.links.remove(link)

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "__Psycles Shader Stage Probe"
    emission.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    if source_socket is not None:
        tree.links.new(source_socket, emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], surface)


def _black_world() -> None:
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new(
            "__Psycles Shader Stage Probe World"
        )
    world = bpy.context.scene.world
    world.use_nodes = True
    tree = world.node_tree
    output = _active_output(tree, "OUTPUT_WORLD")
    surface = _socket_by_name_or_identifier(output.inputs, "Surface")
    for link in tuple(surface.links):
        tree.links.remove(link)
    background = tree.nodes.new("ShaderNodeBackground")
    background.name = "__Psycles Shader Stage Probe"
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    tree.links.new(background.outputs["Background"], surface)


def _main() -> None:
    arguments = _arguments()
    material = bpy.data.materials.get(arguments.material)
    if material is None or not material.use_nodes:
        raise KeyError(
            f"node material {arguments.material!r} was not found"
        )
    node = material.node_tree.nodes.get(arguments.node)
    if node is None:
        raise KeyError(
            f"node {arguments.node!r} was not found in "
            f"material {arguments.material!r}"
        )
    source = _socket_by_name_or_identifier(
        node.outputs, arguments.socket
    )

    for candidate in bpy.data.materials:
        _replace_surface_with_emission(
            candidate, source if candidate == material else None
        )
    _black_world()

    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(
        "Created shader-stage probe "
        f"{arguments.material!r}/{arguments.node!r}/"
        f"{arguments.socket!r}: {output}"
    )


if __name__ == "__main__":
    _main()
