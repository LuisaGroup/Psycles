"""Regression for the raw topology required by Cycles Pointiness."""

from __future__ import annotations

import json
import pathlib
import runpy
import struct
import subprocess
import sys
import tempfile
from typing import Any

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


def _material(name: str, pointiness: bool) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    if pointiness:
        group = bpy.data.node_groups.new(
            f"{name} Nested Pointiness", "ShaderNodeTree"
        )
        group.interface.new_socket(
            name="Pointiness",
            in_out="OUTPUT",
            socket_type="NodeSocketFloat",
        )
        geometry = group.nodes.new("ShaderNodeNewGeometry")
        group_output = group.nodes.new("NodeGroupOutput")
        group.links.new(
            geometry.outputs["Pointiness"],
            group_output.inputs["Pointiness"],
        )
        instance = tree.nodes.new("ShaderNodeGroup")
        instance.node_tree = group
        tree.links.new(
            instance.outputs["Pointiness"], emission.inputs["Strength"]
        )
    return material


def _object(
    scene: Any,
    name: str,
    material: Any,
    x_offset: float,
) -> Any:
    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(
        (
            (x_offset - 1.0, -1.0, 0.0),
            (x_offset + 1.0, -1.0, 0.0),
            (x_offset + 1.0, 1.0, 0.0),
            (x_offset - 1.0, 1.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


def _read_section(
    path: pathlib.Path,
    section: dict[str, int],
) -> bytes:
    with path.open("rb") as stream:
        stream.seek(int(section["offset"]))
        return stream.read(int(section["bytes"]))


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit("expected exporter and inspector paths after '--'")
    exporter = pathlib.Path(args[0]).resolve()
    inspector = pathlib.Path(args[1]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.use_light_tree = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    camera_data = bpy.data.cameras.new("Pointiness Camera")
    camera = bpy.data.objects.new("Pointiness Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    pointiness_material = _material("Pointiness Material", True)
    plain_material = _material("Plain Material", False)
    _object(scene, "Pointiness Object", pointiness_material, -1.5)
    _object(scene, "Plain Object", plain_material, 1.5)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-pointiness-source-"
    ) as temporary:
        output = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(output)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv

        manifest = json.loads(
            (output / "scene.json").read_text(encoding="utf-8")
        )
        geometry_by_name = {
            geometry["name"]: geometry
            for geometry in manifest["geometries"]
        }
        pointiness_geometry = geometry_by_name["Pointiness Object"]
        plain_geometry = geometry_by_name["Plain Object"]
        source = pointiness_geometry.get("pointiness_source")
        if source is None:
            raise AssertionError("linked Pointiness source was not exported")
        if plain_geometry.get("pointiness_source") is not None:
            raise AssertionError("unused Pointiness source was exported")
        if int(source["edge_count"]) != 4:
            raise AssertionError(
                f"expected four pre-tessellation edges, got {source}"
            )
        if int(source["point_normals"]["bytes"]) != 4 * 3 * 4:
            raise AssertionError("Pointiness point-normal extent changed")
        if int(source["edges"]["bytes"]) != 4 * 2 * 4:
            raise AssertionError("Pointiness edge extent changed")

        geometry_path = output / "geometry.bin"
        normal_values = struct.unpack(
            "<12f",
            _read_section(geometry_path, source["point_normals"]),
        )
        expected_normals = (0.0, 0.0, 1.0) * 4
        if any(
            abs(actual - expected) > 1.0e-7
            for actual, expected in zip(normal_values, expected_normals)
        ):
            raise AssertionError(
                f"evaluated point normals changed: {normal_values}"
            )
        edge_values = struct.unpack(
            "<8I", _read_section(geometry_path, source["edges"])
        )
        edges = {
            tuple(sorted(edge_values[index : index + 2]))
            for index in range(0, len(edge_values), 2)
        }
        if edges != {(0, 1), (1, 2), (2, 3), (0, 3)}:
            raise AssertionError(
                f"original edges were replaced by tessellation: {edges}"
            )

        inspected = subprocess.run(
            [
                str(inspector),
                str(output),
                "Pointiness Material",
                "--require-pointiness-source",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if inspected.returncode != 0:
            raise AssertionError(
                "C++ Pointiness import/lowering failed:\n"
                f"{inspected.stdout}\n{inspected.stderr}"
            )
        if "Pointiness" in inspected.stderr:
            raise AssertionError(
                "Pointiness import emitted a diagnostic:\n"
                f"{inspected.stderr}"
            )

    print("Psycles Blender Pointiness-source regression passed")


if __name__ == "__main__":
    _main()
