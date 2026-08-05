"""Regression for the exact Cycles/Blender corner-triangle order."""

from __future__ import annotations

import json
import pathlib
import runpy
import struct
import sys
import tempfile

import bpy


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.cameras,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _read_section(
    path: pathlib.Path,
    section: dict[str, int],
    format_code: str,
) -> tuple[object, ...]:
    with path.open("rb") as stream:
        stream.seek(int(section["offset"]))
        payload = stream.read(int(section["bytes"]))
    return struct.unpack(
        f"<{len(payload) // struct.calcsize(format_code)}{format_code}",
        payload,
    )


def _material(name: str) -> object:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(diffuse.outputs[0], output.inputs[0])
    return material


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected the exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.use_light_tree = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    camera_data = bpy.data.cameras.new("Triangle Order Camera")
    camera = bpy.data.objects.new("Triangle Order Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    mesh = bpy.data.meshes.new("Triangle Order Mesh")
    # The first face is an asymmetric concave n-gon. BMesh BEAUTY
    # triangulation produces a different support/order from the evaluated
    # Mesh corner-triangle cache copied by Cycles create_mesh().
    positions = (
        (-2.0, -1.0, 0.0),
        (0.0, -1.3, 0.0),
        (2.0, -0.7, 0.0),
        (0.6, 0.1, 0.0),
        (1.8, 1.5, 0.0),
        (-0.2, 0.8, 0.0),
        (-1.7, 1.4, 0.0),
        (3.0, 0.0, 0.0),
    )
    mesh.from_pydata(
        positions,
        (),
        ((0, 1, 2, 3, 4, 5, 6), (2, 7, 4)),
    )
    mesh.materials.append(_material("Ngon Material"))
    mesh.materials.append(_material("Triangle Material"))
    mesh.polygons[0].material_index = 0
    mesh.polygons[1].material_index = 1
    uv = mesh.uv_layers.new(name="Triangle Identity UV")
    for loop in mesh.loops:
        uv.data[loop.index].uv = (
            float(loop.index + 1) / 32.0,
            float(loop.index + 2) / 64.0,
        )
    mesh.update()
    mesh.calc_loop_triangles()
    expected_indices = tuple(
        int(mesh.loops[loop].vertex_index)
        for triangle in mesh.loop_triangles
        for loop in triangle.loops
    )
    expected_materials = tuple(
        int(triangle.material_index)
        for triangle in mesh.loop_triangles
    )
    expected_uv = tuple(
        component
        for triangle in mesh.loop_triangles
        for loop in triangle.loops
        for component in uv.data[loop].uv
    )

    obj = bpy.data.objects.new("Triangle Order Object", mesh)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-triangle-order-"
    ) as temporary:
        output = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(output)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv

        exported = json.loads(
            (output / "scene.json").read_text(encoding="utf-8")
        )
        instance = next(
            item
            for item in exported["instances"]
            if item["name"] == obj.name
        )
        geometry = exported["geometries"][int(instance["geometry"])]
        binary = output / "geometry.bin"
        actual_indices = _read_section(binary, geometry["indices"], "I")
        actual_materials = _read_section(
            binary, geometry["triangle_material_slots"], "I"
        )
        actual_uv = _read_section(binary, geometry["uv"], "f")

        if actual_indices != expected_indices:
            raise AssertionError(
                "export topology diverged from Cycles corner-triangle "
                f"order: got {actual_indices}, expected {expected_indices}"
            )
        if actual_materials != expected_materials:
            raise AssertionError(
                "triangle material identity diverged from corner-triangle "
                f"order: got {actual_materials}, expected "
                f"{expected_materials}"
            )
        if len(actual_uv) != len(expected_uv) or any(
            abs(actual - expected) > 1.0e-7
            for actual, expected in zip(actual_uv, expected_uv)
        ):
            raise AssertionError(
                "corner UVs were not flattened in Cycles triangle order"
            )

    print("Psycles Blender triangle-order regression passed")


if __name__ == "__main__":
    _main()
