"""Regression for Cycles' legacy per-texture-node TexMapping state."""

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

    material = bpy.data.materials.new("Texture Mapping State")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    mapped = tree.nodes.new("ShaderNodeTexImage")
    mapped.name = "Legacy Mapped Image"
    mapping = mapped.texture_mapping
    mapping.vector_type = "TEXTURE"
    mapping.translation = (0.25, -0.5, 1.25)
    mapping.rotation = (0.125, -0.25, 0.375)
    mapping.scale = (0.0, -2.0, 0.5)
    mapping.mapping_x = "Z"
    mapping.mapping_y = "NONE"
    mapping.mapping_z = "X"

    identity = tree.nodes.new("ShaderNodeTexNoise")
    identity.name = "Identity Noise Mapping"

    mesh = bpy.data.meshes.new("Texture Mapping Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Texture Mapping Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-texture-node-mapping-"
    ) as temporary:
        output = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(output)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv
        payload = json.loads(
            (output / "scene.json").read_text(encoding="utf-8")
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
    actual = nodes[mapped.name]["special"].get("texture_mapping")
    expected = {
        "vector_type": "TEXTURE",
        "translation": [0.25, -0.5, 1.25],
        "rotation": [0.125, -0.25, 0.375],
        "scale": [0.0, -2.0, 0.5],
        "mapping_x": "Z",
        "mapping_y": "NONE",
        "mapping_z": "X",
    }
    if actual != expected:
        raise AssertionError(
            f"texture-node mapping changed: {actual!r} != {expected!r}"
        )
    if "texture_mapping" in nodes[identity.name]["special"]:
        raise AssertionError("identity TextureNode mapping was not elided")

    print("Psycles texture-node mapping export regression passed")


if __name__ == "__main__":
    _main()
