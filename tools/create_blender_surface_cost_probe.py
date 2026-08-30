"""Create a non-destructive surface-cost probe.

Usage:

    blender source.blend --background --python \
        create_blender_surface_cost_probe.py -- output.blend \
            --mode constant-diffuse

The derived file preserves geometry, material-slot identities, lights, world,
camera, and render settings. ``constant-closure-inputs`` preserves every
shader-to-shader edge in the original material graph but disconnects all
non-shader inputs, including those inside nested node groups. The authored
socket defaults therefore replace the dynamic value DAG without baking it.
The ``constant-diffuse``, ``constant-glossy``, and ``constant-glass`` modes
replace the Surface graph with one raw closure of the named family.
``procedural-diffuse`` replaces every material by the same topology
Generated -> Noise Texture -> Diffuse BSDF, while retaining material-specific
runtime parameters. It is an image-free control for separating SVM/procedural
cost from image-sampler cost. All modes disconnect Volume and Displacement.
Together the probes separate value-program cost, closure-population cost, and
the rest of the integrator; they are not production-scene conversion tools.
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
        description="Create a constant-closure surface cost probe"
    )
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--mode",
        choices=(
            "constant-closure-inputs",
            "constant-diffuse",
            "constant-glossy",
            "constant-glass",
            "procedural-diffuse",
        ),
        default="constant-diffuse",
        help=(
            "preserve the closure graph while stripping value links, or "
            "replace it with one Diffuse closure (default: constant-diffuse)"
        ),
    )
    parser.add_argument(
        "--color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(0.8, 0.8, 0.8),
        help="linear Diffuse color shared by every material (default: 0.8)",
    )
    parser.add_argument(
        "--roughness",
        type=float,
        default=0.0,
        help="Diffuse roughness shared by every material (default: 0)",
    )
    return parser.parse_args(argv)


def _active_material_output(tree: Any) -> Any:
    outputs = [node for node in tree.nodes if node.type == "OUTPUT_MATERIAL"]
    if not outputs:
        return tree.nodes.new("ShaderNodeOutputMaterial")
    return next(
        (
            node
            for node in outputs
            if getattr(node, "is_active_output", False)
        ),
        outputs[0],
    )


def _disconnect_input(tree: Any, socket: Any) -> None:
    for link in tuple(socket.links):
        tree.links.remove(link)


def _replace_material_surface(
    material: Any,
    color: tuple[float, float, float],
    roughness: float,
    mode: str,
) -> None:
    material.use_nodes = True
    tree = material.node_tree
    # Single-closure probes intentionally have no dormant value graph. Keeping
    # disconnected image nodes would not affect shader reachability, but would
    # make the exported control scene retain irrelevant image assets and defeat
    # the stronger invariant that every material is constant-only.
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.is_active_output = True
    surface = output.inputs["Surface"]

    node_type = {
        "constant-diffuse": "ShaderNodeBsdfDiffuse",
        "constant-glossy": "ShaderNodeBsdfAnisotropic",
        "constant-glass": "ShaderNodeBsdfGlass",
    }.get(mode)
    if node_type is None:
        raise ValueError(f"unsupported single-closure probe mode: {mode}")
    closure = tree.nodes.new(node_type)
    closure.name = "__Psycles Constant Surface Cost Probe"
    closure.inputs["Color"].default_value = (*color, 1.0)
    closure.inputs["Roughness"].default_value = roughness
    tree.links.new(closure.outputs["BSDF"], surface)

    cycles = getattr(material, "cycles", None)
    if cycles is not None and hasattr(cycles, "emission_sampling"):
        cycles.emission_sampling = "NONE"


def _replace_material_surface_with_procedural_diffuse(
    material: Any,
    material_index: int,
) -> None:
    """Install one image-free, topology-identical procedural material.

    Only authored defaults vary with ``material_index``. The exporter therefore
    emits one graph topology with distinct parameter blocks, which is the same
    quotient used by the production SVM and avoids measuring artificial shader
    specialization.
    """

    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.is_active_output = True
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")

    # Keep the response bounded and broadly similar across materials while
    # forcing every visible hit through a non-trivial procedural value DAG.
    noise.inputs["Scale"].default_value = 3.0 + float(material_index % 7)
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.55
    noise.inputs["Lacunarity"].default_value = 2.0
    noise.inputs["Distortion"].default_value = 0.15
    diffuse.inputs["Roughness"].default_value = 0.2

    tree.links.new(coordinates.outputs["Generated"], noise.inputs["Vector"])
    tree.links.new(noise.outputs["Color"], diffuse.inputs["Color"])
    tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

    cycles = getattr(material, "cycles", None)
    if cycles is not None and hasattr(cycles, "emission_sampling"):
        cycles.emission_sampling = "NONE"


def _strip_non_shader_links(tree: Any, visited: set[int]) -> None:
    """Remove value-DAG edges while preserving the shader subgraph exactly."""

    identity = tree.as_pointer()
    if identity in visited:
        return
    visited.add(identity)
    for link in tuple(tree.links):
        if link.to_socket.type != "SHADER":
            tree.links.remove(link)
    for node in tree.nodes:
        nested = getattr(node, "node_tree", None)
        if nested is not None:
            _strip_non_shader_links(nested, visited)


def _strip_material_value_graph(material: Any, visited: set[int]) -> None:
    material.use_nodes = True
    tree = material.node_tree
    output = _active_material_output(tree)
    _disconnect_input(tree, output.inputs["Volume"])
    _disconnect_input(tree, output.inputs["Displacement"])
    _strip_non_shader_links(tree, visited)

    cycles = getattr(material, "cycles", None)
    if cycles is not None and hasattr(cycles, "emission_sampling"):
        cycles.emission_sampling = "NONE"


def _replace_all_material_surfaces(
    color: tuple[float, float, float],
    roughness: float,
    mode: str = "constant-diffuse",
) -> int:
    if any(value < 0.0 for value in color):
        raise ValueError("--color components must be non-negative")
    if not 0.0 <= roughness <= 1.0:
        raise ValueError("--roughness must be in [0, 1]")
    materials = tuple(bpy.data.materials)
    if mode in {"constant-diffuse", "constant-glossy", "constant-glass"}:
        for material in materials:
            _replace_material_surface(material, color, roughness, mode)
    elif mode == "procedural-diffuse":
        for material_index, material in enumerate(materials):
            _replace_material_surface_with_procedural_diffuse(
                material, material_index
            )
    elif mode == "constant-closure-inputs":
        visited: set[int] = set()
        for material in materials:
            _strip_material_value_graph(material, visited)
    else:
        raise ValueError(f"unsupported surface-cost probe mode: {mode}")
    return len(materials)


def _main() -> None:
    arguments = _arguments()
    color = tuple(arguments.color)
    material_count = _replace_all_material_surfaces(
        color,
        arguments.roughness,
        arguments.mode,
    )
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(
        f"Created {arguments.mode} surface cost probe with "
        f"{material_count} materials, color={color}, "
        f"roughness={arguments.roughness}: {output}"
    )


if __name__ == "__main__":
    _main()
