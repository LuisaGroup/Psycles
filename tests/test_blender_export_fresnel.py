"""Regression for Blender 5.2 Fresnel and Layer Weight raw export."""

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


def _socket_contract(sockets: list[dict[str, object]]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            socket["identifier"],
            socket["name"],
            socket["type"],
            socket["linked"],
        )
        for socket in sockets
    )


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    material = bpy.data.materials.new("Fresnel Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    fresnel = tree.nodes.new("ShaderNodeFresnel")
    fresnel.name = "Fresnel Export"
    layer = tree.nodes.new("ShaderNodeLayerWeight")
    layer.name = "Layer Weight Export"
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(layer.outputs["Fresnel"], combine.inputs["X"])
    tree.links.new(layer.outputs["Facing"], combine.inputs["Y"])
    tree.links.new(fresnel.outputs["Fac"], combine.inputs["Z"])
    tree.links.new(combine.outputs["Vector"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    mesh = bpy.data.meshes.new("Fresnel Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Fresnel Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(prefix="psycles-fresnel-export-") as temporary:
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
    nodes = exported_material["node_tree"]["nodes"]
    exported_fresnel = next(node for node in nodes if node["name"] == fresnel.name)
    exported_layer = next(node for node in nodes if node["name"] == layer.name)

    if (
        exported_fresnel["type"] != "FRESNEL"
        or exported_fresnel["bl_idname"] != "ShaderNodeFresnel"
        or _socket_contract(exported_fresnel["inputs"])
        != (
            ("IOR", "IOR", "NodeSocketFloat", False),
            ("Normal", "Normal", "NodeSocketVector", False),
        )
        or _socket_contract(exported_fresnel["outputs"])
        != (("Fac", "Factor", "NodeSocketFloat", True),)
    ):
        raise AssertionError(f"Fresnel export contract changed: {exported_fresnel}")

    if (
        exported_layer["type"] != "LAYER_WEIGHT"
        or exported_layer["bl_idname"] != "ShaderNodeLayerWeight"
        or _socket_contract(exported_layer["inputs"])
        != (
            ("Blend", "Blend", "NodeSocketFloat", False),
            ("Normal", "Normal", "NodeSocketVector", False),
        )
        or _socket_contract(exported_layer["outputs"])
        != (
            ("Fresnel", "Fresnel", "NodeSocketFloat", True),
            ("Facing", "Facing", "NodeSocketFloat", True),
        )
    ):
        raise AssertionError(
            f"Layer Weight export contract changed: {exported_layer}"
        )
    print("Psycles Fresnel-family export regression passed")


if __name__ == "__main__":
    _main()
