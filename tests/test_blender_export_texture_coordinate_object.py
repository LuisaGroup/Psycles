"""Regression for explicit Object coordinates on Texture Coordinate nodes."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy
from mathutils import Matrix


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _column_major(matrix: Matrix) -> list[float]:
    return [
        float(matrix[row][column])
        for column in range(4)
        for row in range(4)
    ]


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    helper = bpy.data.objects.new("Coordinate Projector", None)
    helper.matrix_world = Matrix(
        (
            (0.75, -0.20, 0.00, 1.25),
            (0.15, 1.10, 0.10, -0.50),
            (0.00, 0.25, 1.40, 0.75),
            (0.00, 0.00, 0.00, 1.00),
        )
    )
    scene.collection.objects.link(helper)

    material = bpy.data.materials.new("Explicit Object Coordinates")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    explicit = tree.nodes.new("ShaderNodeTexCoord")
    explicit.name = "Explicit Projector Coordinates"
    explicit.object = helper
    implicit = tree.nodes.new("ShaderNodeTexCoord")
    implicit.name = "Implicit Shading Object Coordinates"

    mesh = bpy.data.meshes.new("Coordinate Surface Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Coordinate Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-texture-coordinate-object-"
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
    special = nodes[explicit.name]["special"].get(
        "object_coordinates"
    )
    if special is None:
        raise AssertionError(
            "explicit Texture Coordinate object transform was discarded"
        )
    if special["object"] != helper.name:
        raise AssertionError(
            f"coordinate object identity changed: {special['object']!r}"
        )
    expected = _column_major(helper.matrix_world.inverted_safe())
    actual = special["world_to_object"]
    if len(actual) != 16 or any(
        abs(value - reference) > 1.0e-6
        for value, reference in zip(actual, expected)
    ):
        raise AssertionError(
            f"world-to-object transform changed: {actual} != {expected}"
        )
    if "object_coordinates" in nodes[implicit.name]["special"]:
        raise AssertionError(
            "implicit Object coordinates acquired an explicit transform"
        )

    print("Psycles Texture Coordinate object export regression passed")


if __name__ == "__main__":
    _main()
