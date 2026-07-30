"""Create a non-destructive shader-stage diagnostic scene.

Usage:

    blender source.blend --background --python \
        create_blender_shader_stage_probe.py -- \
        output.blend MATERIAL NODE SOCKET

By default, the selected value output is connected to an Emission closure
while all other surface materials and the world emit black. ``--closure
DIFFUSE --world PRESERVE`` instead keeps the original world and feeds the
selected value into a Lambertian closure, which isolates BSDF and shadow
transport after a value-stage probe has aligned. ``--closure SOURCE`` connects
a selected raw closure socket directly to the material output. ``--closure
ORIGINAL`` keeps the selected material's complete raw closure graph and
replaces only the other materials with black emission, isolating one
production material without baking it. Repeated ``--preserve-material``
options keep additional raw closures, which is useful for adding one potential
light-transport dependency at a time. The source file is never saved. This is
intended only to locate the first divergent evaluation stage between Cycles
and Psycles; production scene export continues to preserve the original
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
    parser.add_argument(
        "--closure",
        choices=("EMISSION", "DIFFUSE", "SOURCE", "ORIGINAL"),
        default="EMISSION",
    )
    parser.add_argument(
        "--world",
        choices=("BLACK", "PRESERVE"),
        default="BLACK",
    )
    parser.add_argument(
        "--preserve-material",
        action="append",
        default=[],
        help=(
            "additional raw material closure to retain; repeatable and "
            "valid only with --closure ORIGINAL"
        ),
    )
    parser.add_argument(
        "--disconnect-input",
        action="append",
        nargs=2,
        default=[],
        metavar=("NODE", "SOCKET"),
        help=(
            "disconnect one selected-material input while retaining its "
            "authored default; repeatable"
        ),
    )
    parser.add_argument(
        "--disable-selected-shadow",
        action="store_true",
        help=(
            "keep selected-material surfaces camera-visible but remove "
            "their shadow-ray visibility"
        ),
    )
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


def _replace_surface(
    material: Any,
    source_socket: Any | None,
    closure: str,
) -> None:
    material.use_nodes = True
    # A stage probe observes camera-visible shader values; it must not turn
    # every probed triangle into an importance-sampled mesh light. In
    # particular, final-render particle systems can contain tens of thousands
    # of instances, so leaving Cycles' default AUTO mode here changes the
    # diagnostic workload by orders of magnitude without changing the value
    # being inspected.
    cycles = getattr(material, "cycles", None)
    if cycles is not None and hasattr(cycles, "emission_sampling"):
        cycles.emission_sampling = "NONE"
    tree = material.node_tree
    output = _active_output(tree, "OUTPUT_MATERIAL")
    surface = _socket_by_name_or_identifier(output.inputs, "Surface")
    for link in tuple(surface.links):
        tree.links.remove(link)

    if closure == "SOURCE":
        if source_socket is None:
            raise ValueError("SOURCE closure requires a source socket")
        tree.links.new(source_socket, surface)
        return
    if closure == "EMISSION":
        diagnostic = tree.nodes.new("ShaderNodeEmission")
        color_input = diagnostic.inputs["Color"]
        diagnostic.inputs["Strength"].default_value = 1.0
        closure_output = diagnostic.outputs["Emission"]
    else:
        diagnostic = tree.nodes.new("ShaderNodeBsdfDiffuse")
        color_input = diagnostic.inputs["Color"]
        diagnostic.inputs["Roughness"].default_value = 0.0
        closure_output = diagnostic.outputs["BSDF"]
    diagnostic.name = "__Psycles Shader Stage Probe"
    color_input.default_value = (0.0, 0.0, 0.0, 1.0)
    if source_socket is not None:
        tree.links.new(source_socket, color_input)
    tree.links.new(closure_output, surface)


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


def _isolate_surface(
    selected_material: Any,
    source_socket: Any,
    closure: str,
    preserved_materials: tuple[Any, ...] = (),
) -> None:
    for candidate in bpy.data.materials:
        preserve_original = closure == "ORIGINAL" and (
            candidate == selected_material
            or any(
                candidate == preserved
                for preserved in preserved_materials
            )
        )
        if preserve_original:
            continue
        _replace_surface(
            candidate,
            source_socket if candidate == selected_material else None,
            (
                "EMISSION"
                if closure in {"ORIGINAL", "SOURCE"}
                and candidate != selected_material
                else closure
            ),
        )


def _disconnect_inputs(
    material: Any,
    inputs: list[list[str]],
) -> None:
    tree = material.node_tree
    for node_name, socket_name in inputs:
        node = tree.nodes.get(node_name)
        if node is None:
            raise KeyError(
                f"node {node_name!r} was not found in "
                f"material {material.name!r}"
            )
        socket = _socket_by_name_or_identifier(
            node.inputs, socket_name
        )
        for link in tuple(socket.links):
            tree.links.remove(link)


def _disable_material_shadows(material: Any) -> None:
    for obj in bpy.data.objects:
        if any(
            slot.material == material
            for slot in obj.material_slots
        ):
            obj.visible_shadow = False


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
    _disconnect_inputs(material, arguments.disconnect_input)
    if arguments.disable_selected_shadow:
        _disable_material_shadows(material)

    if arguments.preserve_material and arguments.closure != "ORIGINAL":
        raise ValueError(
            "--preserve-material requires --closure ORIGINAL"
        )
    preserved_materials = tuple(
        bpy.data.materials[name]
        for name in arguments.preserve_material
    )
    _isolate_surface(
        material,
        source,
        arguments.closure,
        preserved_materials,
    )
    if arguments.world == "BLACK":
        _black_world()

    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(
        "Created shader-stage probe "
        f"{arguments.material!r}/{arguments.node!r}/"
        f"{arguments.socket!r} through {arguments.closure}, "
        f"preserved={[item.name for item in preserved_materials]}, "
        f"selected_shadow={not arguments.disable_selected_shadow}, "
        f"world={arguments.world}: {output}"
    )


if __name__ == "__main__":
    _main()
