"""Regression for image paths owned by linked Blender libraries."""

from __future__ import annotations

import base64
import hashlib
import json
import pathlib
import runpy
import sys
import tempfile

import bpy


_TEST_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgAgAAAAAcoT2JAAAABGdBTUEA"
    "AYagMeiWXwAAAB9JREFUeJxjYAhd9R+M8TCIUMIAU4aPATMJH2OQuQcA"
    "vUl/gYsJiakAAAAASUVORK5CYII="
)


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.images,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _create_library(
    library_path: pathlib.Path,
    texture_path: pathlib.Path,
) -> None:
    _clear_scene()
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path.write_bytes(_TEST_PNG)

    image = bpy.data.images.load(str(texture_path))
    image.name = "Library Relative Image"
    image.filepath = "//../../textures/base.png"

    material = bpy.data.materials.new("Linked Material")
    material.use_nodes = True
    node_tree = material.node_tree
    assert node_tree is not None
    image_node = node_tree.nodes.new("ShaderNodeTexImage")
    image_node.image = image

    mesh = bpy.data.meshes.new("Linked Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    obj = bpy.data.objects.new("Linked Object", mesh)
    bpy.context.scene.collection.objects.link(obj)

    library_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(library_path))


def _link_library_into_main(
    library_path: pathlib.Path,
    main_path: pathlib.Path,
) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(library_path), link=True) as (
        available,
        requested,
    ):
        if "Linked Object" not in available.objects:
            raise AssertionError("test library did not retain its object")
        requested.objects = ["Linked Object"]
    linked_object = requested.objects[0]
    if linked_object is None:
        raise AssertionError("failed to link the test object")
    bpy.context.scene.collection.objects.link(linked_object)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(main_path))


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected the exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-linked-image-"
    ) as temporary:
        root = pathlib.Path(temporary)
        library_path = root / "assets" / "linked" / "library.blend"
        texture_path = root / "textures" / "base.png"
        main_path = root / "main" / "scene.blend"
        output = root / "export"

        _create_library(library_path, texture_path)
        _link_library_into_main(library_path, main_path)

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
            image
            for image in scene["images"]
            if image["name"] == "Library Relative Image"
        ]
        if len(exported) != 1:
            raise AssertionError(
                f"linked image was not exported exactly once: {exported}"
            )
        copied = output / exported[0]["path"]
        expected_digest = hashlib.sha256(_TEST_PNG).hexdigest()
        if copied.read_bytes() != _TEST_PNG:
            raise AssertionError("linked image payload changed during export")
        if exported[0]["sha256"] != expected_digest:
            raise AssertionError("linked image digest does not match payload")

    print("Psycles linked-library image regression passed")


if __name__ == "__main__":
    _main()
