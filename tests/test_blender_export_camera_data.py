"""Regression for Blender 5.2 Camera Data raw-node export."""

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
    material = bpy.data.materials.new("Camera Data Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    camera = tree.nodes.new("ShaderNodeCameraData")
    camera.name = "Camera Data Export"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(camera.outputs["View Vector"], separate.inputs["Vector"])
    tree.links.new(separate.outputs["X"], combine.inputs["X"])
    tree.links.new(camera.outputs["View Z Depth"], combine.inputs["Y"])
    tree.links.new(camera.outputs["View Distance"], combine.inputs["Z"])
    tree.links.new(combine.outputs["Vector"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    mesh = bpy.data.meshes.new("Camera Data Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Camera Data Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-camera-data-export-"
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
    exported = next(
        node
        for node in exported_material["node_tree"]["nodes"]
        if node["name"] == camera.name
    )
    expected_outputs = (
        ("View Vector", "View Vector", "NodeSocketVector", True),
        ("View Z Depth", "View Z Depth", "NodeSocketFloat", True),
        ("View Distance", "View Distance", "NodeSocketFloat", True),
    )
    actual_outputs = tuple(
        (
            socket["identifier"],
            socket["name"],
            socket["type"],
            socket["linked"],
        )
        for socket in exported["outputs"]
    )
    if (
        exported["type"] != "CAMERA"
        or exported["bl_idname"] != "ShaderNodeCameraData"
        or exported["inputs"]
        or actual_outputs != expected_outputs
    ):
        raise AssertionError(f"Camera Data export contract changed: {exported}")
    print("Psycles Camera Data export regression passed")


if __name__ == "__main__":
    _main()
