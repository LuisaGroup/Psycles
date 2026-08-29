"""Regression for Cycles-target shader output selection during export."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


def _socket(sockets: object, identifier: str) -> object:
    return next(
        socket
        for socket in sockets
        if socket.identifier == identifier
    )


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.lights,
        bpy.data.worlds,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _add_competing_outputs(
    tree: object,
    output_type: str,
    cycles_source: object,
    fallback_source: object,
) -> object:
    fallback = tree.nodes.new(output_type)
    fallback.name = "Active ALL Output"
    fallback.target = "ALL"
    cycles = tree.nodes.new(output_type)
    cycles.name = "Inactive Cycles Output"
    cycles.target = "CYCLES"
    tree.links.new(
        _socket(fallback_source.outputs, fallback_source.outputs[0].identifier),
        _socket(fallback.inputs, "Surface"),
    )
    tree.links.new(
        _socket(cycles_source.outputs, cycles_source.outputs[0].identifier),
        _socket(cycles.inputs, "Surface"),
    )
    fallback.is_active_output = True
    if cycles.is_active_output:
        raise AssertionError("Cycles-specific output unexpectedly became active")
    selected = tree.get_output_node("CYCLES")
    if selected != cycles:
        raise AssertionError(
            "Blender did not prefer the exact CYCLES target over active ALL"
        )
    return cycles


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    material = bpy.data.materials.new("Cycles Output Material")
    material.use_nodes = True
    material_tree = material.node_tree
    assert material_tree is not None
    material_tree.nodes.clear()
    fallback_surface = material_tree.nodes.new("ShaderNodeEmission")
    fallback_surface.name = "Fallback Material Surface"
    cycles_surface = material_tree.nodes.new("ShaderNodeBsdfDiffuse")
    cycles_surface.name = "Cycles Material Surface"
    cycles_output = _add_competing_outputs(
        material_tree,
        "ShaderNodeOutputMaterial",
        cycles_surface,
        fallback_surface,
    )
    cycles_volume = material_tree.nodes.new("ShaderNodeVolumePrincipled")
    cycles_volume.name = "Cycles Material Volume"
    cycles_displacement = material_tree.nodes.new("ShaderNodeDisplacement")
    cycles_displacement.name = "Cycles Material Displacement"
    material_tree.links.new(
        _socket(cycles_volume.outputs, "Volume"),
        _socket(cycles_output.inputs, "Volume"),
    )
    material_tree.links.new(
        _socket(cycles_displacement.outputs, "Displacement"),
        _socket(cycles_output.inputs, "Displacement"),
    )

    mesh = bpy.data.meshes.new("Cycles Output Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Cycles Output Surface", mesh)
    scene.collection.objects.link(surface)

    world = bpy.data.worlds.new("Cycles Output World")
    world.use_nodes = True
    world_tree = world.node_tree
    assert world_tree is not None
    world_tree.nodes.clear()
    fallback_world = world_tree.nodes.new("ShaderNodeBackground")
    fallback_world.name = "Fallback World Surface"
    cycles_world = world_tree.nodes.new("ShaderNodeBackground")
    cycles_world.name = "Cycles World Surface"
    _add_competing_outputs(
        world_tree,
        "ShaderNodeOutputWorld",
        cycles_world,
        fallback_world,
    )
    scene.world = world

    light_data = bpy.data.lights.new("Cycles Output Light Data", "POINT")
    light_data.use_nodes = True
    light_tree = light_data.node_tree
    assert light_tree is not None
    light_tree.nodes.clear()
    fallback_light = light_tree.nodes.new("ShaderNodeEmission")
    fallback_light.name = "Fallback Light Surface"
    cycles_light = light_tree.nodes.new("ShaderNodeEmission")
    cycles_light.name = "Cycles Light Surface"
    _add_competing_outputs(
        light_tree,
        "ShaderNodeOutputLight",
        cycles_light,
        fallback_light,
    )
    light = bpy.data.objects.new("Cycles Output Light", light_data)
    scene.collection.objects.link(light)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-cycles-output-export-"
    ) as temporary:
        destination = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(destination)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv
        payload = json.loads(
            (destination / "scene.json").read_text(encoding="utf-8")
        )

    exported_material = next(
        item
        for item in payload["materials"]
        if item["name"] == material.name
    )["node_tree"]
    expected_material = {
        "surface_root": {
            "node": cycles_surface.name,
            "socket": "BSDF",
        },
        "volume_root": {
            "node": cycles_volume.name,
            "socket": "Volume",
        },
        "displacement_root": {
            "node": cycles_displacement.name,
            "socket": "Displacement",
        },
    }
    for key, expected in expected_material.items():
        if exported_material[key] != expected:
            raise AssertionError(
                f"material {key} ignored the CYCLES target: "
                f"{exported_material[key]!r} != {expected!r}"
            )

    exported_world = payload["world"]["node_tree"]
    if exported_world["surface_root"] != {
        "node": cycles_world.name,
        "socket": "Background",
    }:
        raise AssertionError(
            "world surface root ignored the CYCLES output target"
        )

    exported_light = next(
        item
        for item in payload["lights"]
        if item["name"] == light.name
    )["node_tree"]
    if exported_light["surface_root"] != {
        "node": cycles_light.name,
        "socket": "Emission",
    }:
        raise AssertionError(
            "light surface root ignored the CYCLES output target"
        )

    print("Psycles Cycles-output export regression passed")


if __name__ == "__main__":
    _main()
