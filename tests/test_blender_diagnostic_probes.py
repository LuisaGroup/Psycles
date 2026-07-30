"""Blender-side regressions for non-production diagnostic probe tools."""

from __future__ import annotations

import pathlib
import runpy
import sys
import tempfile
from typing import Any

import bpy


def _load(path: pathlib.Path) -> dict[str, Any]:
    return runpy.run_path(
        str(path),
        run_name=f"psycles_test_{path.stem}",
    )


def _world_signature(world: Any) -> tuple[Any, ...]:
    tree = world.node_tree
    return (
        world,
        tuple(tree.nodes),
        tuple(
            (
                link.from_node,
                link.from_socket,
                link.to_node,
                link.to_socket,
            )
            for link in tree.links
        ),
    )


def _surface_source(tree: Any) -> Any:
    output = next(
        node
        for node in tree.nodes
        if node.type == "OUTPUT_MATERIAL"
        and node.is_active_output
    )
    links = output.inputs["Surface"].links
    if len(links) != 1:
        raise AssertionError("diagnostic surface has no unique closure")
    return links[0].from_node


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit(
            "expected shader-stage and world-probe script paths after '--'"
        )
    stage_path, world_path = map(
        pathlib.Path, args
    )
    stage = _load(stage_path.resolve())
    world_probe = _load(world_path.resolve())

    material = bpy.data.materials.new("Diagnostic Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    source = tree.nodes.new("ShaderNodeRGB")
    source.name = "Diagnostic Source"
    source.outputs["Color"].default_value = (0.2, 0.4, 0.7, 1.0)
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

    world = bpy.data.worlds.new("Preserved Diagnostic World")
    world.use_nodes = True
    bpy.context.scene.world = world
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise AssertionError("factory world has no Background node")
    background.inputs["Color"].default_value = (0.13, 0.27, 0.51, 1.0)
    background.inputs["Strength"].default_value = 0.73
    original_world = _world_signature(world)

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-diagnostic-probe-"
    ) as temporary:
        output_path = pathlib.Path(temporary) / "preserved.blend"
        old_argv = sys.argv
        try:
            sys.argv = [
                str(stage_path),
                "--",
                str(output_path),
                material.name,
                source.name,
                "Color",
                "--closure",
                "DIFFUSE",
                "--world",
                "PRESERVE",
            ]
            stage["_main"]()
        finally:
            sys.argv = old_argv
        if not output_path.is_file():
            raise AssertionError("shader-stage probe did not save its output")

    diagnostic = _surface_source(tree)
    if diagnostic.type != "BSDF_DIFFUSE":
        raise AssertionError(
            "DIFFUSE diagnostic did not preserve a raw Lambert closure"
        )
    if (
        diagnostic.inputs["Color"].links[0].from_node.name
        != source.name
    ):
        raise AssertionError(
            "DIFFUSE diagnostic did not connect the selected raw socket"
        )
    if material.cycles.emission_sampling != "NONE":
        raise AssertionError(
            "diagnostic material remained an importance-sampled mesh light"
        )
    if _world_signature(world) != original_world:
        raise AssertionError("--world PRESERVE modified the source world")

    stage["_replace_surface"](
        material,
        source.outputs["Color"],
        "EMISSION",
    )
    diagnostic = _surface_source(tree)
    if diagnostic.type != "EMISSION":
        raise AssertionError("EMISSION diagnostic closure was not created")
    if material.cycles.emission_sampling != "NONE":
        raise AssertionError(
            "EMISSION diagnostic became an importance-sampled mesh light"
        )

    source_closure_material = bpy.data.materials.new(
        "Source Closure Material"
    )
    source_closure_material.use_nodes = True
    source_tree = source_closure_material.node_tree
    source_output = next(
        node
        for node in source_tree.nodes
        if node.type == "OUTPUT_MATERIAL"
    )
    source_principled = next(
        node
        for node in source_tree.nodes
        if node.type == "BSDF_PRINCIPLED"
    )
    source_other = bpy.data.materials.new("Source Closure Other")
    source_other.use_nodes = True
    stage["_isolate_surface"](
        source_closure_material,
        source_principled.outputs["BSDF"],
        "SOURCE",
    )
    source_link = source_output.inputs["Surface"].links[0]
    if (
        source_link.from_node != source_principled
        or source_link.from_socket != source_principled.outputs["BSDF"]
    ):
        raise AssertionError(
            "SOURCE diagnostic did not connect the selected raw closure"
        )
    if _surface_source(source_other.node_tree).type != "EMISSION":
        raise AssertionError(
            "SOURCE diagnostic did not black out other materials"
        )

    disconnected_material = bpy.data.materials.new(
        "Disconnected Input Material"
    )
    disconnected_material.use_nodes = True
    disconnected_principled = next(
        node
        for node in disconnected_material.node_tree.nodes
        if node.type == "BSDF_PRINCIPLED"
    )
    disconnected_color = disconnected_material.node_tree.nodes.new(
        "ShaderNodeRGB"
    )
    disconnected_material.node_tree.links.new(
        disconnected_color.outputs["Color"],
        disconnected_principled.inputs["Base Color"],
    )
    stage["_disconnect_inputs"](
        disconnected_material,
        [["Principled BSDF", "Base Color"]],
    )
    if disconnected_principled.inputs["Base Color"].is_linked:
        raise AssertionError(
            "diagnostic input disconnection retained an incoming link"
        )

    shadow_mesh = bpy.data.meshes.new("Shadow Diagnostic Mesh")
    shadow_object = bpy.data.objects.new(
        "Shadow Diagnostic Object", shadow_mesh
    )
    shadow_mesh.materials.append(disconnected_material)
    shadow_object.visible_shadow = True
    stage["_disable_material_shadows"](disconnected_material)
    if shadow_object.visible_shadow:
        raise AssertionError(
            "selected-material shadow visibility was not disabled"
        )

    original_material = bpy.data.materials.new(
        "Original Closure Material"
    )
    original_material.use_nodes = True
    original_tree = original_material.node_tree
    original_output = next(
        node
        for node in original_tree.nodes
        if node.type == "OUTPUT_MATERIAL"
    )
    original_principled = next(
        node
        for node in original_tree.nodes
        if node.type == "BSDF_PRINCIPLED"
    )
    original_link = original_output.inputs["Surface"].links[0]

    preserved_dependency = bpy.data.materials.new(
        "Preserved Original Dependency"
    )
    preserved_dependency.use_nodes = True
    preserved_dependency_tree = preserved_dependency.node_tree
    preserved_dependency_link = next(
        node
        for node in preserved_dependency_tree.nodes
        if node.type == "OUTPUT_MATERIAL"
    ).inputs["Surface"].links[0]

    isolated_other = bpy.data.materials.new("Isolated Other Material")
    isolated_other.use_nodes = True
    stage["_isolate_surface"](
        original_material,
        original_principled.outputs["BSDF"],
        "ORIGINAL",
        (preserved_dependency,),
    )
    preserved_link = original_output.inputs["Surface"].links[0]
    if (
        preserved_link.from_node != original_link.from_node
        or preserved_link.from_socket != original_link.from_socket
    ):
        raise AssertionError(
            "ORIGINAL diagnostic replaced the selected raw closure"
        )
    current_dependency_link = next(
        node
        for node in preserved_dependency_tree.nodes
        if node.type == "OUTPUT_MATERIAL"
    ).inputs["Surface"].links[0]
    if (
        current_dependency_link.from_node
        != preserved_dependency_link.from_node
        or current_dependency_link.from_socket
        != preserved_dependency_link.from_socket
    ):
        raise AssertionError(
            "ORIGINAL diagnostic replaced a preserved dependency closure"
        )
    if _surface_source(isolated_other.node_tree).type != "EMISSION":
        raise AssertionError(
            "ORIGINAL diagnostic did not black out other materials"
        )

    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.render.use_sequencer = True
    world_probe["_configure_raw_cycles_probe"](scene)
    if scene.render.use_compositing or scene.render.use_sequencer:
        raise AssertionError(
            "world probe retained compositor/sequencer post-processing"
        )
    if (
        scene.render.engine != "CYCLES"
        or scene.cycles.samples != 1
        or scene.cycles.use_denoising
        or scene.render.resolution_x != 4
        or scene.render.resolution_y != 4
        or scene.render.resolution_percentage != 25
    ):
        raise AssertionError(
            "world probe is not a deterministic one-ray Cycles render: "
            f"engine={scene.render.engine}, "
            f"samples={scene.cycles.samples}, "
            f"denoising={scene.cycles.use_denoising}, "
            f"resolution={scene.render.resolution_x}x"
            f"{scene.render.resolution_y}@"
            f"{scene.render.resolution_percentage}%"
        )
    radiance = world_probe["_render_direction"](
        scene,
        scene.camera,
        world_probe["Vector"]((0.0, 0.0, 1.0)),
    )
    if len(radiance) != 3 or not all(
        isinstance(value, float) for value in radiance
    ):
        raise AssertionError(
            "world probe did not return one RGB camera-ray sample: "
            f"{radiance}"
        )

    print("Psycles Blender diagnostic-probe regressions passed")


if __name__ == "__main__":
    _main()
