"""Regression for BlenderSync ordering and source-scene identities."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


def _reset_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)


def _export(
    exporter: pathlib.Path,
    output: pathlib.Path,
) -> dict[str, object]:
    old_argv = sys.argv
    try:
        sys.argv = [str(exporter), "--", str(output)]
        runpy.run_path(str(exporter), run_name="__main__")
    finally:
        sys.argv = old_argv
    return json.loads(
        (output / "scene.json").read_text(encoding="utf-8")
    )


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _reset_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    group_a = bpy.context.view_layer.lightgroups.add()
    group_a.name = "Group A"
    group_b = bpy.context.view_layer.lightgroups.add()
    group_b.name = "Group B"

    material = bpy.data.materials.new("Middle Material")
    material.use_nodes = True
    mesh = bpy.data.meshes.new("Surface Mesh")
    mesh.from_pydata(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Middle Surface", mesh)
    scene.collection.objects.link(surface)

    def add_light(name: str, group: str) -> None:
        data = bpy.data.lights.new(f"{name} Data", type="POINT")
        data.shadow_soft_size = 0.0
        obj = bpy.data.objects.new(name, data)
        obj.visible_camera = False
        obj.is_shadow_catcher = True
        obj.lightgroup = group
        scene.collection.objects.link(obj)

    # Deliberately insert the reverse of lexical order. Cycles consumes the
    # dependency-graph object iterator; sorting these names changes which
    # emitter a fixed random number selects.
    add_light("Zulu Light", "Group B")
    add_light("Alpha Light", "Group A")

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-cycles-identity-"
    ) as temporary:
        payload = _export(exporter, pathlib.Path(temporary))

    materials = {
        item["name"]: item
        for item in payload["materials"]
    }
    material_sync = materials["Middle Material"]["cycles_sync"]
    if material_sync != {"shader_index": 7}:
        raise AssertionError(
            "material shader identity did not follow five defaults and "
            f"two dependency-graph light shaders: {material_sync}"
        )

    instances = payload["instances"]
    if len(instances) != 1:
        raise AssertionError(f"unexpected instances: {instances}")
    if instances[0]["cycles_sync"] != {
        "object_index": 0,
        "light_group": -1,
    }:
        raise AssertionError(
            "surface object identity changed: "
            f"{instances[0]['cycles_sync']}"
        )

    lights = payload["lights"]
    names = [item["name"] for item in lights]
    if names != ["Zulu Light", "Alpha Light"]:
        raise AssertionError(
            "analytic lights were not exported in Cycles object order: "
            f"{names}"
        )
    expected = {
        "Zulu Light": {
            "object_index": 1,
            "light_group": 1,
            "shader_index": 5,
        },
        "Alpha Light": {
            "object_index": 2,
            "light_group": 0,
            "shader_index": 6,
        },
    }
    for light in lights:
        if light["cycles_sync"] != expected[light["name"]]:
            raise AssertionError(
                f"{light['name']} identity changed: "
                f"{light['cycles_sync']}"
            )
        if (
            light["visibility"]["camera"]
            or not light["is_shadow_catcher"]
            or not light["cast_shadow"]
        ):
            raise AssertionError(
                f"{light['name']} shader flags were not preserved"
            )

    if payload["world"] is not None and payload["world"]["cycles_sync"] != {
        "shader_index": 3
    }:
        raise AssertionError(
            "world did not retain Cycles' default-background shader"
        )

    print("Psycles Blender Cycles-identity regression passed")


if __name__ == "__main__":
    _main()
