"""Regression for Blender/Cycles muted-node graph bypass semantics."""

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

    material = bpy.data.materials.new("Muted Node Contract")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    source = tree.nodes.new("ShaderNodeRGB")
    source.name = "Live Source"
    mix = tree.nodes.new("ShaderNodeMix")
    mix.name = "Muted Mix"
    mix.data_type = "RGBA"
    mix.mute = True
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse"
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    tree.links.new(
        _socket(source.outputs, "Color"),
        _socket(mix.inputs, "A_Color"),
    )
    tree.links.new(
        _socket(mix.outputs, "Result_Color"),
        _socket(diffuse.inputs, "Color"),
    )
    tree.links.new(
        _socket(diffuse.outputs, "BSDF"),
        _socket(output.inputs, "Surface"),
    )

    expected_internal = [
        {
            "from_socket": link.from_socket.identifier,
            "to_socket": link.to_socket.identifier,
        }
        for link in mix.internal_links
    ]
    expected_internal.sort(
        key=lambda item: (item["to_socket"], item["from_socket"])
    )
    if expected_internal != [
        {
            "from_socket": "A_Color",
            "to_socket": "Result_Color",
        }
    ]:
        raise AssertionError(
            "Blender did not expose the expected muted Mix bypass: "
            f"{expected_internal!r}"
        )

    mesh = bpy.data.meshes.new("Muted Node Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Muted Node Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-muted-node-export-"
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
    )
    nodes = {
        node["name"]: node
        for node in exported_material["node_tree"]["nodes"]
    }
    exported_mix = nodes[mix.name]
    if exported_mix["mute"] is not True:
        raise AssertionError("muted node state was not exported")
    if exported_mix["internal_links"] != expected_internal:
        raise AssertionError(
            "muted node bypass changed during export: "
            f"{exported_mix['internal_links']!r} != "
            f"{expected_internal!r}"
        )
    if nodes[source.name]["mute"] is not False:
        raise AssertionError("unmuted node state was not exported")

    print("Psycles muted-node export regression passed")


if __name__ == "__main__":
    _main()
