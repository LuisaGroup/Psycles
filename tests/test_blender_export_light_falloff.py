"""Regression for the Blender 5.2 Light Falloff export contract."""

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
    material = bpy.data.materials.new("Light Falloff Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    falloff = tree.nodes.new("ShaderNodeLightFalloff")
    falloff.name = "Light Falloff"
    falloff.inputs["Strength"].default_value = 6.0
    falloff.inputs["Smooth"].default_value = 1.5
    closures = []
    for index, name in enumerate(("Quadratic", "Linear", "Constant")):
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Emission {index}"
        tree.links.new(falloff.outputs[name], emission.inputs["Strength"])
        closures.append(emission.outputs["Emission"])
    while len(closures) > 1:
        left = closures.pop(0)
        right = closures.pop(0)
        add = tree.nodes.new("ShaderNodeAddShader")
        tree.links.new(left, add.inputs[0])
        tree.links.new(right, add.inputs[1])
        closures.append(add.outputs[0])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(closures[0], output.inputs["Surface"])

    mesh = bpy.data.meshes.new("Light Falloff Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Light Falloff Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-light-falloff-export-"
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
        if node["name"] == "Light Falloff"
    )
    inputs = tuple(
        (
            socket["identifier"],
            socket["name"],
            socket["type"],
            socket["linked"],
            socket["default"],
        )
        for socket in raw["inputs"]
    )
    outputs = tuple(
        (
            socket["identifier"],
            socket["name"],
            socket["type"],
            socket["linked"],
        )
        for socket in raw["outputs"]
    )
    expected_inputs = (
        ("Strength", "Strength", "NodeSocketFloat", False, 6.0),
        ("Smooth", "Smooth", "NodeSocketFloat", False, 1.5),
    )
    expected_outputs = tuple(
        (name, name, "NodeSocketFloat", True)
        for name in ("Quadratic", "Linear", "Constant")
    )
    if (
        raw["type"] != "LIGHT_FALLOFF"
        or raw["bl_idname"] != "ShaderNodeLightFalloff"
        or inputs != expected_inputs
        or outputs != expected_outputs
    ):
        raise AssertionError(f"Light Falloff export contract changed: {raw}")
    print("Psycles Blender Light Falloff export regression passed")


if __name__ == "__main__":
    _main()
