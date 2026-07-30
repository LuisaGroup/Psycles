"""Blender-side regression for evaluated-geometry cache identity."""

from __future__ import annotations

import json
import math
import pathlib
import runpy
import struct
import sys
import tempfile

import bpy


def _mesh(name: str) -> object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        (
            (-0.75, -0.5, 0.0),
            (0.25, -0.5, 0.0),
            (0.25, 0.5, 0.0),
            (-0.75, 0.5, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.update()
    return mesh


def _object(name: str, mesh: object) -> object:
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _clear_scene() -> None:
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected the exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()

    shared_mesh = _mesh("Shared unmodified mesh")
    for polygon in shared_mesh.polygons:
        polygon.use_smooth = True
    shared_a_object = _object("shared-a", shared_mesh)
    shared_b_object = _object("shared-b", shared_mesh)
    shared_a_object.cycles.shadow_terminator_geometry_offset = 0.125
    shared_b_object.cycles.shadow_terminator_geometry_offset = 0.375

    modified_mesh = _mesh("Shared modified mesh")
    modified_two = _object("modified-two", modified_mesh)
    modified_four = _object("modified-four", modified_mesh)
    render_only_array = modified_two.modifiers.new("Array", "ARRAY")
    render_only_array.count = 2
    render_only_array.show_viewport = False
    render_only_array.show_render = True
    modified_four.modifiers.new("Array", "ARRAY").count = 4
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-geometry-cache-"
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
        geometry_by_instance = {
            instance["name"]: int(instance["geometry"])
            for instance in scene["instances"]
        }
        instance_by_name = {
            instance["name"]: instance
            for instance in scene["instances"]
        }

        shared_a = geometry_by_instance["shared-a"]
        shared_b = geometry_by_instance["shared-b"]
        modified_two_index = geometry_by_instance["modified-two"]
        modified_four_index = geometry_by_instance["modified-four"]

        if shared_a != shared_b:
            raise AssertionError(
                "unmodified objects sharing one Mesh datablock were not "
                "deduplicated"
            )
        for name, expected in (
            ("shared-a", 0.125),
            ("shared-b", 0.375),
        ):
            actual = float(
                instance_by_name[name][
                    "shadow_terminator_geometry_offset"
                ]
            )
            if not math.isclose(actual, expected, abs_tol=1.0e-7):
                raise AssertionError(
                    "instance-specific Cycles shadow-terminator offset "
                    f"did not round-trip: {name}={actual}, "
                    f"expected {expected}"
                )
        if modified_two_index == modified_four_index:
            raise AssertionError(
                "object-specific modifier results shared one geometry"
            )

        modified_two_triangles = int(
            scene["geometries"][modified_two_index]["triangle_count"]
        )
        modified_four_triangles = int(
            scene["geometries"][modified_four_index]["triangle_count"]
        )
        if (modified_two_triangles, modified_four_triangles) != (4, 8):
            raise AssertionError(
                "unexpected evaluated Array topology: "
                f"{modified_two_triangles}, {modified_four_triangles}"
            )
        if len(scene["geometries"]) != 3 or len(scene["instances"]) != 4:
            raise AssertionError(
                "unexpected cache cardinality: "
                f"{len(scene['geometries'])} geometries, "
                f"{len(scene['instances'])} instances"
            )
        smooth_section = scene["geometries"][shared_a][
            "triangle_smooth"
        ]
        with (output / "geometry.bin").open("rb") as stream:
            stream.seek(int(smooth_section["offset"]))
            smooth_values = struct.unpack(
                "<2I",
                stream.read(int(smooth_section["bytes"])),
            )
        if smooth_values != (1, 1):
            raise AssertionError(
                "smooth-polygon flags did not survive evaluated "
                f"triangulation: {smooth_values}"
            )

    print("Psycles Blender geometry-cache regression passed")


if __name__ == "__main__":
    _main()
