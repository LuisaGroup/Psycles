"""Regression for generated Blender image datablocks."""

from __future__ import annotations

import hashlib
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
        bpy.data.images,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected the exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    image = bpy.data.images.new(
        "Generated Checker",
        width=2,
        height=2,
        alpha=True,
        float_buffer=False,
    )
    pixels = (
        0.0,
        0.25,
        0.5,
        1.0,
        1.0,
        0.5,
        0.25,
        0.75,
        0.1,
        0.2,
        0.3,
        0.5,
        0.9,
        0.8,
        0.7,
        0.25,
    )
    image.pixels[:] = pixels

    material = bpy.data.materials.new("Generated Image Material")
    material.use_nodes = True
    node_tree = material.node_tree
    assert node_tree is not None
    image_node = node_tree.nodes.new("ShaderNodeTexImage")
    image_node.image = image

    mesh = bpy.data.meshes.new("Generated Image Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    obj = bpy.data.objects.new("Generated Image Object", mesh)
    bpy.context.scene.collection.objects.link(obj)

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-generated-image-"
    ) as temporary:
        output = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(output)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv

        scene = json.loads(
            (output / "scene.json").read_text(encoding="utf-8")
        )
        exported = [
            item
            for item in scene["images"]
            if item["name"] == "Generated Checker"
        ]
        if len(exported) != 1:
            raise AssertionError(
                f"generated image was not exported exactly once: {exported}"
            )
        encoded_path = output / exported[0]["path"]
        encoded = encoded_path.read_bytes()
        if not encoded.startswith(b"\x89PNG\r\n\x1a\n"):
            raise AssertionError("generated image is not a PNG payload")
        if exported[0]["sha256"] != hashlib.sha256(encoded).hexdigest():
            raise AssertionError("generated image digest does not match payload")

        decoded = bpy.data.images.load(
            str(encoded_path),
            check_existing=False,
        )
        try:
            actual = tuple(decoded.pixels)
        finally:
            bpy.data.images.remove(decoded)
        if len(actual) != len(pixels):
            raise AssertionError(
                f"generated image dimensions changed: {len(actual)} values"
            )
        for index, (expected, value) in enumerate(zip(pixels, actual)):
            if abs(expected - value) > 0.012:
                raise AssertionError(
                    "generated image pixel changed after encode/decode: "
                    f"channel {index}, {value} != {expected}"
                )

    print("Psycles generated-image regression passed")


if __name__ == "__main__":
    _main()
