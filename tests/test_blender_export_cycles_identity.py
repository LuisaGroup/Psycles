"""Regression for BlenderSync ordering and source-scene identities."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


def _near_sequence(actual: object, expected: list[float]) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            abs(float(lhs) - rhs) <= 1.0e-6
            for lhs, rhs in zip(actual, expected, strict=True)
        )
    )


def _reset_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for curve in list(bpy.data.curves):
        bpy.data.curves.remove(curve)
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

    scene.world.cycles_visibility.camera = False
    scene.world.cycles_visibility.diffuse = True
    scene.world.cycles_visibility.glossy = False
    scene.world.cycles_visibility.transmission = True
    scene.world.cycles_visibility.shadow = False
    scene.world.cycles_visibility.scatter = False
    scene.world.cycles.use_shadows = False
    scene.world.cycles.max_bounces = 17

    material = bpy.data.materials.new("Middle Material")
    material.use_nodes = True
    material.pass_index = 23
    material.displacement_method = "BUMP"
    material.cycles.volume_sampling = "EQUIANGULAR"
    material.cycles.use_bump_map_correction = False
    # This datablock exists in the .blend but is outside the dependency-graph
    # surface image. It must not perturb the exported SVM input domain.
    bpy.data.materials.new("Unused Material")
    mesh = bpy.data.meshes.new("Surface Mesh")
    mesh.from_pydata(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        [],
        [(0, 1, 2)],
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Middle Surface", mesh)
    surface.color = (0.125, 0.25, 0.5, 0.75)
    surface.pass_index = 19
    surface.shadow_terminator_shading_offset = 0.375
    surface.shadow_terminator_geometry_offset = 0.625
    surface.is_shadow_catcher = True
    surface.cycles.is_caustics_caster = True
    surface.cycles.is_caustics_receiver = True
    asset_root = bpy.data.objects.new("Top Asset", None)
    holdout_collection = bpy.data.collections.new("Holdout Collection")
    scene.collection.children.link(holdout_collection)
    holdout_collection.objects.link(asset_root)
    holdout_collection.objects.link(surface)
    surface.parent = asset_root
    bpy.context.view_layer.layer_collection.children[
        "Holdout Collection"
    ].holdout = True
    receiver_collection = bpy.data.collections.new(
        "Light-link receivers"
    )
    scene.collection.children.link(receiver_collection)
    surface.light_linking.receiver_collection = receiver_collection
    bpy.context.view_layer.update()

    # Legacy Curve data is the important negative case: Object.to_mesh()
    # returns a mesh payload, but BlenderSync::object_is_geometry rejects the
    # dependency-graph entry while its object-data ID remains a Curve.
    # Counting it before discovering that it has no triangles shifts every
    # later Cycles object index.
    curve_data = bpy.data.curves.new("Convertible Legacy Curve", "CURVE")
    curve_data.dimensions = "3D"
    spline = curve_data.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (-0.5, 0.0, 0.0, 1.0)
    spline.points[1].co = (0.5, 0.0, 0.0, 1.0)
    legacy_curve = bpy.data.objects.new("Legacy Curve", curve_data)
    scene.collection.objects.link(legacy_curve)
    bpy.context.view_layer.update()
    evaluated_curve = legacy_curve.evaluated_get(
        bpy.context.evaluated_depsgraph_get()
    )
    converted_curve = evaluated_curve.to_mesh()
    try:
        if len(converted_curve.vertices) == 0:
            raise AssertionError(
                "legacy Curve regression fixture was not convertible"
            )
    finally:
        evaluated_curve.to_mesh_clear()

    def add_light(
        name: str, group: str, max_bounces: int, *, portal: bool = False
    ) -> None:
        data = bpy.data.lights.new(
            f"{name} Data", type="AREA" if portal else "POINT"
        )
        if portal:
            data.cycles.is_portal = True
        else:
            data.shadow_soft_size = 0.0
        data.cycles.max_bounces = max_bounces
        obj = bpy.data.objects.new(name, data)
        obj.visible_camera = False
        obj.is_shadow_catcher = True
        obj.lightgroup = group
        obj.color = (0.2, 0.3, 0.4, 0.5)
        obj.pass_index = max_bounces + 10
        obj.shadow_terminator_shading_offset = 0.25
        obj.shadow_terminator_geometry_offset = 0.75
        obj.cycles.is_caustics_caster = name == "Zulu Light"
        obj.cycles.is_caustics_receiver = name == "Alpha Light"
        scene.collection.objects.link(obj)

    # Deliberately insert the reverse of lexical order. Cycles consumes the
    # dependency-graph object iterator; sorting these names changes which
    # emitter a fixed random number selects.
    add_light("Zulu Light", "Group B", 3, portal=True)
    add_light("Alpha Light", "Group A", 7)

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-cycles-identity-"
    ) as temporary:
        payload = _export(exporter, pathlib.Path(temporary))

    identity_module = sys.modules.get("exporter_identity")
    if identity_module is None:
        raise AssertionError("exporter did not load its identity contract")
    expected_exporter = identity_module.current(exporter)
    if payload.get("exporter") != expected_exporter:
        raise AssertionError(
            "scene bundle did not record the complete exporter closure: "
            f"{payload.get('exporter')} != {expected_exporter}"
        )

    materials = {
        item["name"]: item
        for item in payload["materials"]
    }
    if "Unused Material" in materials:
        raise AssertionError(
            "unreferenced Blender material entered the exported graph domain"
        )
    material_sync = materials["Middle Material"]["cycles_sync"]
    if material_sync != {"shader_index": 7, "pass_id": 23}:
        raise AssertionError(
            "material shader identity did not follow five defaults and "
            f"two dependency-graph light shaders: {material_sync}"
        )
    if materials["Middle Material"]["volume_sampling"] != "EQUIANGULAR":
        raise AssertionError(
            "material volume-sampling policy did not round-trip through "
            "the Blender exporter"
        )
    if materials["Middle Material"]["displacement_method"] != "BUMP":
        raise AssertionError(
            "material displacement policy did not round-trip through "
            "the Blender exporter"
        )
    if materials["Middle Material"]["use_bump_map_correction"] is not False:
        raise AssertionError(
            "material bump-map correction policy did not round-trip "
            "through the Blender exporter"
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
    if not instances[0]["is_shadow_catcher"]:
        raise AssertionError(
            "mesh shadow-catcher membership was not preserved"
        )
    if (
        instances[0]["asset_name"] != "Top Asset"
        or not instances[0]["use_holdout"]
        or not instances[0]["is_caustics_caster"]
        or not instances[0]["is_caustics_receiver"]
        or not isinstance(instances[0]["random_id"], int)
    ):
        raise AssertionError(
            "surface raw Cycles object state changed: "
            f"{instances[0]}"
        )
    expected_surface_object = {
        "object_color": [0.125, 0.25, 0.5],
        "dupli_generated": [0.0, 0.0, 0.0],
        "dupli_uv": [0.0, 0.0],
    }
    for field, expected_value in expected_surface_object.items():
        if not _near_sequence(instances[0][field], expected_value):
            raise AssertionError(
                f"surface {field} changed: {instances[0][field]}"
            )
    if (
        abs(float(instances[0]["object_alpha"]) - 0.75) > 1.0e-6
        or instances[0]["object_pass_id"] != 19
        or abs(
            float(instances[0]["shadow_terminator_shading_offset"])
            - 0.375
        )
        > 1.0e-6
        or abs(
            float(instances[0]["shadow_terminator_geometry_offset"])
            - 0.625
        )
        > 1.0e-6
    ):
        raise AssertionError("surface scalar KernelObject fields changed")

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
            "pass_id": 0,
        },
        "Alpha Light": {
            "object_index": 2,
            "light_group": 0,
            "shader_index": 6,
            "pass_id": 0,
        },
    }
    expected_max_bounces = {
        "Zulu Light": 3,
        "Alpha Light": 7,
    }
    for light in lights:
        if light["cycles_sync"] != expected[light["name"]]:
            raise AssertionError(
                f"{light['name']} identity changed: "
                f"{light['cycles_sync']}"
            )
        if light["max_bounces"] != expected_max_bounces[light["name"]]:
            raise AssertionError(
                f"{light['name']} max-bounces policy changed: "
                f"{light['max_bounces']}"
            )
        if light["is_portal"] != (light["name"] == "Zulu Light"):
            raise AssertionError(
                f"{light['name']} portal policy changed: "
                f"{light['is_portal']}"
            )
        if (
            light["visibility"]["camera"]
            or not light["is_shadow_catcher"]
            or not light["cast_shadow"]
        ):
            raise AssertionError(
                f"{light['name']} shader flags were not preserved"
            )
        if (
            light["asset_name"] != light["name"]
            or light["use_holdout"]
            or light["is_caustics_caster"]
            != (light["name"] == "Zulu Light")
            or light["is_caustics_receiver"]
            != (light["name"] == "Alpha Light")
            or not isinstance(light["random_id"], int)
        ):
            raise AssertionError(
                f"{light['name']} raw Cycles object state changed"
            )
        if (
            not _near_sequence(light["object_color"], [0.2, 0.3, 0.4])
            or abs(float(light["object_alpha"]) - 0.5) > 1.0e-6
            or light["object_pass_id"]
            != expected_max_bounces[light["name"]] + 10
            or not _near_sequence(
                light["dupli_generated"], [0.0, 0.0, 0.0]
            )
            or not _near_sequence(light["dupli_uv"], [0.0, 0.0])
            or abs(
                float(light["shadow_terminator_shading_offset"]) - 0.25
            )
            > 1.0e-6
            or abs(
                float(light["shadow_terminator_geometry_offset"]) - 0.75
            )
            > 1.0e-6
        ):
            raise AssertionError(
                f"{light['name']} KernelObject fields changed"
            )

    if payload.get("cycles_sync") != {
        "object_count": 4,
        "uses_light_linking": True,
    }:
        raise AssertionError(
            "dense object extent or light-linking capability changed: "
            f"{payload.get('cycles_sync')}"
        )

    if payload["world"] is not None:
        if payload["world"]["name"] != scene.world.name:
            raise AssertionError("world asset identity changed")
        if payload["world"]["cycles_sync"] != {
            "shader_index": 3,
            "pass_id": 0,
            "object_index": 3,
            "light_group": -1,
        }:
            raise AssertionError("world Cycles identity changed")
        if payload["world"]["use_shadows"]:
            raise AssertionError("world shadow policy changed")
        if payload["world"]["max_bounces"] != 17:
            raise AssertionError("world max-bounces policy changed")
        visibility = payload["world"]["visibility"]
        if (
            visibility["camera"]
            or not visibility["diffuse"]
            or visibility["glossy"]
            or not visibility["transmission"]
            or visibility["shadow"]
            or visibility["volume_scatter"]
        ):
            raise AssertionError(
                f"world visibility changed: {visibility}"
            )

    print("Psycles Blender Cycles-identity regression passed")


if __name__ == "__main__":
    _main()
