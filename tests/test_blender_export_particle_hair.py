"""Blender-side regression for native Cycles particle-hair export."""

from __future__ import annotations

import json
import math
import pathlib
import runpy
import struct
import sys
import tempfile
from typing import Any

import bpy


_PARENT_COUNT = 4
_CHILDREN_PER_PARENT = 2
_RENDER_STEP = 2
_KEYS_PER_CURVE = (1 << _RENDER_STEP) + 1


def _clear_scene() -> None:
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in tuple(bpy.data.materials):
        bpy.data.materials.remove(material)


def _hair_material() -> Any:
    material = bpy.data.materials.new("Native Hair Intercept")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    hair_info = tree.nodes.new("ShaderNodeHairInfo")
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(hair_info.outputs["Intercept"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _fixture() -> Any:
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=2,
        y_subdivisions=2,
        size=1.0,
        enter_editmode=False,
        location=(1.25, -0.75, 0.5),
        rotation=(0.2, -0.35, 0.4),
        scale=(1.5, 0.75, 1.25),
    )
    emitter = bpy.context.object
    emitter.name = "Cycles Native Particle Hair"
    emitter.data.materials.append(_hair_material())
    bpy.context.view_layer.objects.active = emitter
    emitter.select_set(True)
    bpy.ops.object.particle_system_add()
    settings = emitter.particle_systems[-1].settings
    settings.type = "HAIR"
    settings.render_type = "PATH"
    settings.count = _PARENT_COUNT
    settings.hair_length = 0.65
    settings.emit_from = "VERT"
    settings.use_modifier_stack = True
    settings.child_type = "SIMPLE"
    settings.child_percent = _CHILDREN_PER_PARENT
    settings.rendered_child_count = _CHILDREN_PER_PARENT
    settings.render_step = _RENDER_STEP
    settings.radius_scale = 0.08
    settings.root_radius = 0.75
    settings.tip_radius = 0.25
    settings.shape = -0.25
    settings.use_close_tip = True
    settings.material = 1
    bpy.context.scene.cycles_curves.shape = "RIBBONS"
    bpy.context.scene.cycles_curves.subdivisions = 2
    bpy.context.view_layer.update()
    return emitter


def _read_section(
    path: pathlib.Path,
    section: dict[str, int],
    format_character: str,
) -> tuple[Any, ...]:
    item_size = struct.calcsize("<" + format_character)
    byte_count = int(section["bytes"])
    if byte_count % item_size != 0:
        raise AssertionError(f"misaligned binary section: {section}")
    with path.open("rb") as stream:
        stream.seek(int(section["offset"]))
        payload = stream.read(byte_count)
    if len(payload) != byte_count:
        raise AssertionError(f"truncated binary section: {section}")
    return struct.unpack(
        "<" + format_character * (byte_count // item_size), payload
    )


def _expected_first_child(emitter: Any) -> tuple[list[Any], float]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = emitter.evaluated_get(depsgraph)
    modifier = next(
        modifier
        for modifier in evaluated.modifiers
        if modifier.type == "PARTICLE_SYSTEM"
    )
    particle_system = modifier.particle_system
    expected_children = _PARENT_COUNT * _CHILDREN_PER_PARENT
    if len(particle_system.child_particles) != expected_children:
        raise AssertionError(
            "fixture did not create the expected child-strand cache: "
            f"{len(particle_system.child_particles)} != {expected_children}"
        )
    inverse = evaluated.matrix_world.inverted()
    positions = [
        inverse
        @ particle_system.co_hair(
            object=evaluated,
            particle_no=_PARENT_COUNT,
            step=step,
        )
        for step in range(_KEYS_PER_CURVE)
    ]
    length = sum(
        (positions[index] - positions[index - 1]).length
        for index in range(1, len(positions))
    )
    return positions, length


def _assert_near(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2.0e-5, abs_tol=2.0e-6):
        raise AssertionError(f"{message}: {actual} != {expected}")


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected the exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    emitter = _fixture()
    expected_positions, expected_length = _expected_first_child(emitter)
    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-particle-hair-"
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
        if len(scene["geometries"]) != 1:
            raise AssertionError("emitter surface mesh was not exported once")
        if len(scene["curve_geometries"]) != 1:
            raise AssertionError("particle hair was not a native curve geometry")
        curves = scene["curve_geometries"][0]
        expected_curve_count = _PARENT_COUNT * _CHILDREN_PER_PARENT
        if int(curves["curve_count"]) != expected_curve_count:
            raise AssertionError(
                "Cycles child-only strand selection changed: "
                f"{curves['curve_count']} != {expected_curve_count}"
            )
        expected_key_count = expected_curve_count * _KEYS_PER_CURVE
        if int(curves["key_count"]) != expected_key_count:
            raise AssertionError("render_step did not define the key count")
        if int(curves["segment_count"]) != (
            expected_curve_count * (_KEYS_PER_CURVE - 1)
        ):
            raise AssertionError("curve segment count is inconsistent")
        if curves["shape"] != "RIBBON" or curves["subdivisions"] != 2:
            raise AssertionError("Cycles scene curve settings were not retained")

        mesh_instances = [
            instance
            for instance in scene["instances"]
            if instance["geometry_type"] == "MESH"
        ]
        curve_instances = [
            instance
            for instance in scene["instances"]
            if instance["geometry_type"] == "CURVE"
        ]
        if len(mesh_instances) != 1 or len(curve_instances) != 1:
            raise AssertionError("surface/hair object split is not one-to-one")
        if mesh_instances[0]["cycles_sync"]["object_index"] != 0:
            raise AssertionError("surface object lost Cycles sync order")
        if curve_instances[0]["cycles_sync"]["object_index"] != 1:
            raise AssertionError("hair object was not synced after its emitter")
        if mesh_instances[0]["transform"] != curve_instances[0]["transform"]:
            raise AssertionError("hair object did not retain emitter transform")

        binary = output / "geometry.bin"
        keys = _read_section(binary, curves["keys"], "f")
        first_keys = _read_section(
            binary, curves["curve_first_key"], "I"
        )
        material_slots = _read_section(
            binary, curves["curve_material_slots"], "I"
        )
        intercept = _read_section(binary, curves["intercept"], "f")
        lengths = _read_section(binary, curves["length"], "f")
        random = _read_section(binary, curves["random"], "f")
        expected_first_keys = tuple(
            index * _KEYS_PER_CURVE
            for index in range(expected_curve_count)
        )
        if first_keys != expected_first_keys:
            raise AssertionError("curve key ranges are not contiguous")
        if material_slots != (0,) * expected_curve_count:
            raise AssertionError("particle material slot was not retained")
        for key, expected in enumerate(expected_positions):
            for component in range(3):
                _assert_near(
                    float(keys[key * 4 + component]),
                    float(expected[component]),
                    f"first child key {key} component {component}",
                )
        _assert_near(float(intercept[0]), 0.0, "root Intercept")
        _assert_near(
            float(intercept[_KEYS_PER_CURVE - 1]), 1.0, "tip Intercept"
        )
        _assert_near(float(lengths[0]), expected_length, "curve Length")
        _assert_near(float(keys[3]), 0.03, "root radius")
        _assert_near(
            float(keys[(_KEYS_PER_CURVE - 1) * 4 + 3]),
            0.0,
            "closed tip radius",
        )
        # Cycles hash_uint2_to_float(0, 0), pinned independently of Blender's
        # dependency-graph random ids.
        _assert_near(float(random[0]), 0.8603127584467439, "curve Random")

    print("Psycles native particle-hair export regression passed")


if __name__ == "__main__":
    _main()
