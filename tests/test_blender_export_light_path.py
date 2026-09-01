"""Regression for Blender 5.2 Light Path raw export."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


_OUTPUTS = (
    "Is Camera Ray",
    "Is Shadow Ray",
    "Is Diffuse Ray",
    "Is Glossy Ray",
    "Is Singular Ray",
    "Is Reflection Ray",
    "Is Transmission Ray",
    "Is Volume Scatter Ray",
    "Ray Length",
    "Ray Depth",
    "Diffuse Depth",
    "Glossy Depth",
    "Transparent Depth",
    "Transmission Depth",
    "Portal Depth",
)


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    material = bpy.data.materials.new("Light Path Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.name = "Light Path Export"
    accumulator = light_path.outputs[_OUTPUTS[0]]
    for index, output_name in enumerate(_OUTPUTS[1:], 1):
        add = tree.nodes.new("ShaderNodeMath")
        add.name = f"Keep Light Path Output {index}"
        add.operation = "ADD"
        tree.links.new(accumulator, add.inputs[0])
        tree.links.new(light_path.outputs[output_name], add.inputs[1])
        accumulator = add.outputs[0]
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(accumulator, emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    mesh = bpy.data.meshes.new("Light Path Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Light Path Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(prefix="psycles-light-path-export-") as temporary:
        directory = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(directory)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            (directory / "scene.json").read_text(encoding="utf-8")
        )

    exported_material = next(
        item for item in manifest["materials"] if item["name"] == material.name
    )
    node = next(
        item
        for item in exported_material["node_tree"]["nodes"]
        if item["name"] == light_path.name
    )
    socket_contract = tuple(
        (socket["identifier"], socket["name"], socket["type"], socket["linked"])
        for socket in node["outputs"]
    )
    expected = tuple(
        (name, name, "NodeSocketFloat", True) for name in _OUTPUTS
    )
    if (
        node["type"] != "LIGHT_PATH"
        or node["bl_idname"] != "ShaderNodeLightPath"
        or node["inputs"]
        or socket_contract != expected
    ):
        raise AssertionError(f"Light Path export contract changed: {node}")
    print("Psycles Light Path export regression passed")


if __name__ == "__main__":
    _main()
