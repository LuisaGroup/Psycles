"""Regression for Cycles-style point/corner/face geometry domains."""

from __future__ import annotations

import json
import pathlib
import runpy
import struct
import subprocess
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
        raise SystemExit(
            "expected exporter and inspector paths after '--'"
        )
    exporter = pathlib.Path(args[0]).resolve()
    inspector = pathlib.Path(args[1]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.use_light_tree = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    camera_data = bpy.data.cameras.new("Domain Camera")
    camera = bpy.data.objects.new("Domain Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    mesh = bpy.data.meshes.new("Domain Mesh")
    positions = (
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )
    mesh.from_pydata(positions, (), ((0, 1, 2, 3),))
    mesh.polygons[0].use_smooth = True

    uv = mesh.uv_layers.new(name="Domain UV")
    for index, value in enumerate(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    ):
        uv.data[index].uv = value

    point_color = mesh.color_attributes.new(
        name="Point Color",
        type="FLOAT_COLOR",
        domain="POINT",
    )
    for index, value in enumerate(
        (
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        )
    ):
        point_color.data[index].color = value

    corner_color = mesh.color_attributes.new(
        name="Corner Color",
        type="FLOAT_COLOR",
        domain="CORNER",
    )
    for index, value in enumerate(
        (
            (0.1, 0.2, 0.3, 1.0),
            (0.4, 0.5, 0.6, 1.0),
            (0.7, 0.8, 0.9, 1.0),
            (0.2, 0.4, 0.6, 1.0),
        )
    ):
        corner_color.data[index].color = value

    material = bpy.data.materials.new("Domain Material")
    mesh.materials.append(material)
    obj = bpy.data.objects.new("Domain Object", mesh)
    scene.collection.objects.link(obj)
    mesh.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-attribute-domains-"
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
        if scene["schema"] != "psycles.blender-scene.v2":
            raise AssertionError(
                f"unexpected scene schema: {scene['schema']}"
            )
        geometry_path = output / "geometry.bin"
        with geometry_path.open("rb") as stream:
            if stream.read(8) != b"PSYGEO2\0":
                raise AssertionError("geometry v2 magic is missing")

        geometry = scene["geometries"][0]
        expected_counts = {
            "point_count": 4,
            "corner_count": 6,
            "triangle_count": 2,
        }
        for key, expected in expected_counts.items():
            if int(geometry[key]) != expected:
                raise AssertionError(
                    f"{key}={geometry[key]}, expected {expected}"
                )
        expected_sections = {
            "positions": 4 * 3 * 4,
            "normals": 4 * 3 * 4,
            "uv": 6 * 2 * 4,
            "uv_tangents": 6 * 4 * 4,
            "generated": 4 * 3 * 4,
            "indices": 6 * 4,
        }
        for name, expected in expected_sections.items():
            actual = int(geometry[name]["bytes"])
            if actual != expected:
                raise AssertionError(
                    f"{name} has {actual} bytes, expected {expected}"
                )
        if (
            geometry["normal_domain"] != "POINT"
            or geometry["uv_domain"] != "CORNER"
            or geometry["uv_tangent_domain"] != "CORNER"
            or geometry["generated_domain"] != "POINT"
        ):
            raise AssertionError(
                "primary attribute domains do not match Cycles layout"
            )

        indices = struct.unpack(
            "<6I",
            _read_section(geometry_path, geometry["indices"]),
        )
        if max(indices) >= 4 or set(indices) != {0, 1, 2, 3}:
            raise AssertionError(
                f"triangles do not reference shared points: {indices}"
            )

        uv_layers = {
            layer["name"]: layer
            for layer in geometry["uv_layers"]
        }
        if uv_layers["Domain UV"]["domain"] != "CORNER":
            raise AssertionError("UV layer lost its corner domain")
        colors = {
            attribute["name"]: attribute
            for attribute in geometry["color_attributes"]
        }
        if (
            colors["Point Color"]["domain"] != "POINT"
            or colors["Point Color"]["values"]["bytes"] != 4 * 4 * 4
        ):
            raise AssertionError("point color domain was flattened")
        if (
            colors["Corner Color"]["domain"] != "CORNER"
            or colors["Corner Color"]["values"]["bytes"] != 6 * 4 * 4
        ):
            raise AssertionError("corner color domain has invalid extent")
        inspected = subprocess.run(
            [
                str(inspector),
                str(output),
                "Domain Material",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if inspected.returncode != 0:
            raise AssertionError(
                "C++ compact-geometry import failed:\n"
                f"{inspected.stdout}\n{inspected.stderr}"
            )

    print("Psycles Blender attribute-domain regression passed")


if __name__ == "__main__":
    _main()
