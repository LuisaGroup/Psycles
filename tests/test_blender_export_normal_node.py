"""Regression for the Blender 5.2 Normal-node export contract."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


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
    material = bpy.data.materials.new("Normal Node Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    normal = tree.nodes.new("ShaderNodeNormal")
    normal.name = "Normal"
    normal.inputs["Normal"].default_value = (-4.0, 5.0, 1.0)
    normal.outputs["Normal"].default_value = (1.0, -2.0, 3.0)

    normal_emission = tree.nodes.new("ShaderNodeEmission")
    normal_emission.name = "Normal Emission"
    dot_emission = tree.nodes.new("ShaderNodeEmission")
    dot_emission.name = "Dot Emission"
    add = tree.nodes.new("ShaderNodeAddShader")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(normal.outputs["Normal"], normal_emission.inputs["Color"])
    tree.links.new(normal.outputs["Dot"], dot_emission.inputs["Strength"])
    tree.links.new(normal_emission.outputs["Emission"], add.inputs[0])
    tree.links.new(dot_emission.outputs["Emission"], add.inputs[1])
    tree.links.new(add.outputs[0], output.inputs["Surface"])

    mesh = bpy.data.meshes.new("Normal Node Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Normal Node Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-normal-node-export-"
    ) as temporary:
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
    raw = next(
        node
        for node in exported_material["node_tree"]["nodes"]
        if node["name"] == "Normal"
    )
    inputs = tuple(
        (
            socket["identifier"],
            socket["name"],
            socket["type"],
            socket["linked"],
            tuple(socket["default"]),
        )
        for socket in raw["inputs"]
    )
    outputs = tuple(
        (
            socket["identifier"],
            socket["name"],
            socket["type"],
            socket["linked"],
            tuple(socket["default"]) if isinstance(socket["default"], list)
            else socket["default"],
        )
        for socket in raw["outputs"]
    )
    expected_inputs = (
        ("Normal", "Normal", "NodeSocketVectorDirection", False,
         (-4.0, 5.0, 1.0)),
    )
    expected_outputs = (
        ("Normal", "Normal", "NodeSocketVectorDirection", True,
         (1.0, -2.0, 3.0)),
        ("Dot", "Dot", "NodeSocketFloat", True, 0.0),
    )
    if (
        raw["type"] != "NORMAL"
        or raw["bl_idname"] != "ShaderNodeNormal"
        or inputs != expected_inputs
        or outputs != expected_outputs
    ):
        raise AssertionError(f"Normal-node export contract changed: {raw}")
    print("Psycles Blender Normal-node export regression passed")


if __name__ == "__main__":
    _main()
