"""Regression for Cycles' effective per-triangle smooth-normal flag."""

from __future__ import annotations

import json
import pathlib
import runpy
import struct
import sys
import tempfile

import bpy


def _clear_scene() -> None:
    for datablocks in (bpy.data.objects, bpy.data.meshes):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _object(
    name: str,
    positions: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, ...], ...],
) -> object:
    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(positions, (), faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _smooth_values(
    geometry_path: pathlib.Path,
    geometry: dict[str, object],
) -> tuple[int, ...]:
    section = geometry["triangle_smooth"]
    assert isinstance(section, dict)
    count = int(geometry["triangle_count"])
    with geometry_path.open("rb") as stream:
        stream.seek(int(section["offset"]))
        payload = stream.read(int(section["bytes"]))
    return struct.unpack(f"<{count}I", payload)


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected the exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()

    point = _object(
        "point-domain",
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)),
    )
    for polygon in point.data.polygons:
        polygon.use_smooth = True
    point.data.update()

    # Barbershop's dresser back wall has exactly this representation: no
    # sharp_face attribute, use_smooth reads true, but sharp edges reduce the
    # effective Blender normal domain to FACE. Cycles consequently stores a
    # flat shader flag for every triangle.
    face = _object(
        "face-domain-from-sharp-edges",
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
        ),
        ((0, 1, 2, 3),),
    )
    face.data.polygons[0].use_smooth = True
    sharp_face = face.data.attributes.get("sharp_face")
    if sharp_face is None:
        raise AssertionError(
            "setting use_smooth did not create sharp_face storage"
        )
    face.data.attributes.remove(sharp_face)
    for edge in face.data.edges:
        edge.use_edge_sharp = True
    face.data.update()

    # With mixed flat/smooth polygons Blender supplies corner normals. Cycles
    # then sets SHADER_SMOOTH_NORMAL on every triangle and lets the corner
    # values encode flatness; ignoring those values would change the surface.
    corner = _object(
        "corner-domain",
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        ((0, 1, 2), (0, 3, 1)),
    )
    corner.data.polygons[0].use_smooth = False
    corner.data.polygons[1].use_smooth = True
    corner.data.update()

    expectations = {
        "point-domain": ("POINT", (1, 1, 1, 1)),
        "face-domain-from-sharp-edges": ("POINT", (0, 0)),
        "corner-domain": ("CORNER", (1, 1)),
    }
    for name, obj in (
        ("point-domain", point),
        ("face-domain-from-sharp-edges", face),
        ("corner-domain", corner),
    ):
        actual_domain = str(obj.data.normals_domain)
        required_domain = (
            "FACE"
            if name.startswith("face-domain")
            else expectations[name][0]
        )
        if actual_domain != required_domain:
            raise AssertionError(
                f"{name} setup has normal domain {actual_domain}, "
                f"expected {required_domain}"
            )

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-smooth-normals-"
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
        geometry_by_name = {
            instance["name"]: scene["geometries"][int(instance["geometry"])]
            for instance in scene["instances"]
        }
        geometry_path = output / "geometry.bin"
        for name, (expected_domain, expected_smooth) in expectations.items():
            geometry = geometry_by_name[name]
            actual_domain = str(geometry["normal_domain"])
            actual_smooth = _smooth_values(geometry_path, geometry)
            if actual_domain != expected_domain:
                raise AssertionError(
                    f"{name} exported {actual_domain} normals, "
                    f"expected {expected_domain}"
                )
            if actual_smooth != expected_smooth:
                raise AssertionError(
                    f"{name} exported smooth flags {actual_smooth}, "
                    f"expected {expected_smooth}"
                )

    print("Psycles Blender smooth-normal regression passed")


if __name__ == "__main__":
    _main()
