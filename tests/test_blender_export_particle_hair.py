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
_RENDER_STEP = 3
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
    uv_map = tree.nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "RootUV"
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "RootColor"
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    next(
        socket for socket in mix.inputs
        if socket.identifier == "Factor_Float"
    ).default_value = 0.25
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(
        uv_map.outputs["UV"],
        next(
            socket for socket in mix.inputs
            if socket.identifier == "A_Color"
        ),
    )
    tree.links.new(
        attribute.outputs["Color"],
        next(
            socket for socket in mix.inputs
            if socket.identifier == "B_Color"
        ),
    )
    tree.links.new(
        next(
            socket for socket in mix.outputs
            if socket.identifier == "Result_Color"
        ),
        emission.inputs["Color"],
    )
    tree.links.new(
        hair_info.outputs["Intercept"], emission.inputs["Strength"]
    )
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
    root_uv = emitter.data.uv_layers.active
    if root_uv is None:
        root_uv = emitter.data.uv_layers.new(name="RootUV")
    else:
        root_uv.name = "RootUV"
    detail_uv = emitter.data.uv_layers.new(name="DetailUV", do_init=False)
    root_uv.active_render = True
    for index, value in enumerate(root_uv.data):
        value.uv = (0.1 + 0.07 * index, 0.2 + 0.03 * index)
    for index, value in enumerate(detail_uv.data):
        value.uv = (0.8 - 0.04 * index, 0.6 - 0.02 * index)
    root_color = emitter.data.color_attributes.new(
        name="RootColor", type="BYTE_COLOR", domain="CORNER"
    )
    unused_color = emitter.data.color_attributes.new(
        name="UnusedColor", type="BYTE_COLOR", domain="CORNER"
    )
    for index, value in enumerate(root_color.data):
        value.color_srgb = (
            0.15 + 0.05 * (index % 4),
            0.25 + 0.04 * (index % 3),
            0.35 + 0.03 * (index % 2),
            1.0,
        )
    for value in unused_color.data:
        value.color_srgb = (0.9, 0.8, 0.7, 1.0)
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
    settings.child_length = 0.4
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


def _cycles_expected_positions(
    particle_system: Any,
    evaluated: Any,
    particle_no: int,
) -> tuple[list[Any], float]:
    inverse = evaluated.matrix_world.inverted()
    world_positions = [
        particle_system.co_hair(
            object=evaluated,
            particle_no=particle_no,
            step=step,
        )
        for step in range(_KEYS_PER_CURVE)
    ]
    # Child-length clipping makes BKE_particle_co_hair leave a suffix of the
    # output slots untouched. RNA exposes those slots as zero, whereas Cycles
    # seeds every call with the preceding coordinate. Require the fixture to
    # exercise that distinction and construct the exact Cycles expectation.
    suffix_begin = len(world_positions)
    while suffix_begin > 0 and world_positions[suffix_begin - 1].length == 0.0:
        suffix_begin -= 1
    if not 0 < suffix_begin < len(world_positions):
        raise AssertionError(
            "fixture did not produce a shortened child-particle path"
        )
    world_positions[suffix_begin:] = [
        world_positions[suffix_begin - 1].copy()
        for _ in range(len(world_positions) - suffix_begin)
    ]
    positions = [inverse @ position for position in world_positions]
    length = sum(
        (positions[index] - positions[index - 1]).length
        for index in range(1, len(positions))
    )
    return positions, length


def _cycles_expected_uv_layers(
    emitter: Any,
) -> tuple[str, dict[str, list[Any]]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = emitter.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    try:
        names = [layer.name for layer in mesh.uv_layers]
        default = mesh.uv_layers.active_render.name
    finally:
        evaluated.to_mesh_clear()
    expected = {name: [] for name in names}
    for modifier in evaluated.modifiers:
        if modifier.type != "PARTICLE_SYSTEM" or not modifier.show_render:
            continue
        system = modifier.particle_system
        settings = system.settings
        if settings.type != "HAIR" or settings.render_type != "PATH":
            continue
        parents = system.particles
        first = (
            len(parents)
            if settings.child_type != "NONE"
            and len(system.child_particles) != 0
            else 0
        )
        for particle_no in range(
            first, len(parents) + len(system.child_particles)
        ):
            particle = (
                parents[particle_no]
                if particle_no < len(parents)
                else parents[0]
            )
            for uv_no, name in enumerate(names):
                expected[name].append(
                    system.uv_on_emitter(
                        modifier=modifier,
                        particle=particle,
                        particle_no=particle_no,
                        uv_no=uv_no,
                    ).copy()
                )
    return default, expected


def _cycles_expected_color_attributes(
    emitter: Any,
) -> dict[str, list[Any]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = emitter.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    try:
        names = [
            attribute.name
            for attribute in mesh.attributes
            if attribute.data_type == "BYTE_COLOR"
            and attribute.domain == "CORNER"
        ]
    finally:
        evaluated.to_mesh_clear()
    expected = {"RootColor": []}
    color_no = names.index("RootColor")
    for modifier in evaluated.modifiers:
        if modifier.type != "PARTICLE_SYSTEM" or not modifier.show_render:
            continue
        system = modifier.particle_system
        settings = system.settings
        if settings.type != "HAIR" or settings.render_type != "PATH":
            continue
        parents = system.particles
        first = (
            len(parents)
            if settings.child_type != "NONE"
            and len(system.child_particles) != 0
            else 0
        )
        for particle_no in range(
            first, len(parents) + len(system.child_particles)
        ):
            particle = (
                parents[particle_no]
                if particle_no < len(parents)
                else parents[0]
            )
            expected["RootColor"].append(
                tuple(
                    system.mcol_on_emitter(
                        modifier=modifier,
                        particle=particle,
                        particle_no=particle_no,
                        vcol_no=color_no,
                    )
                )
            )
    return expected


def _srgb_to_linear(value: float) -> float:
    if value < 0.04045:
        return 0.0 if value < 0.0 else value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


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
    expected_default_uv, expected_uv_layers = (
        _cycles_expected_uv_layers(emitter)
    )
    expected_color_attributes = _cycles_expected_color_attributes(emitter)
    exporter_module = runpy.run_path(
        str(exporter), run_name="psycles_particle_hair_exporter_test"
    )
    original_positions = exporter_module[
        "_cycles_particle_hair_positions"
    ]
    captured_expected: list[tuple[list[Any], float]] = []

    def capture_positions(
        particle_system: Any,
        evaluated: Any,
        particle_no: int,
        key_count: int,
    ) -> Any:
        positions = original_positions(
            particle_system,
            evaluated,
            particle_no,
            key_count,
        )
        if particle_no == _PARENT_COUNT and not captured_expected:
            captured_expected.append(
                _cycles_expected_positions(
                    particle_system, evaluated, particle_no
                )
            )
        return positions

    exporter_module["_particle_hair_geometry"].__globals__[
        "_cycles_particle_hair_positions"
    ] = capture_positions
    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-particle-hair-"
    ) as temporary:
        output = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(output)]
            exporter_module["_main"]()
        finally:
            sys.argv = old_argv

        if len(captured_expected) != 1:
            raise AssertionError(
                "final-render dependency graph did not expose the first child"
            )
        expected_positions, expected_length = captured_expected[0]

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
        if curves["default_uv_layer"] != expected_default_uv:
            raise AssertionError("active-render particle UV layer changed")
        exported_uv_layers = {
            layer["name"]: _read_section(binary, layer["values"], "f")
            for layer in curves["uv_layers"]
        }
        if set(exported_uv_layers) != set(expected_uv_layers):
            raise AssertionError("particle UV layer names changed")
        for name, expected_values in expected_uv_layers.items():
            actual = exported_uv_layers[name]
            if len(actual) != 2 * len(expected_values):
                raise AssertionError("particle UV curve domain changed")
            for curve, expected in enumerate(expected_values):
                _assert_near(
                    float(actual[curve * 2]),
                    float(expected[0]),
                    f"{name} curve {curve} U",
                )
                _assert_near(
                    float(actual[curve * 2 + 1]),
                    float(expected[1]),
                    f"{name} curve {curve} V",
                )
        if exported_uv_layers["RootUV"] == exported_uv_layers["DetailUV"]:
            raise AssertionError("named particle UV layers were aliased")
        exported_color_attributes = {
            attribute["name"]: _read_section(
                binary, attribute["values"], "f"
            )
            for attribute in curves["color_attributes"]
        }
        if set(exported_color_attributes) != {"RootColor"}:
            raise AssertionError(
                "particle color demand did not match Cycles"
            )
        for name, expected_values in expected_color_attributes.items():
            actual = exported_color_attributes[name]
            if len(actual) != 4 * len(expected_values):
                raise AssertionError("particle color curve domain changed")
            for curve, expected_srgb in enumerate(expected_values):
                expected_linear = [
                    _srgb_to_linear(float(component))
                    for component in expected_srgb
                ]
                expected = expected_linear + [1.0]
                for channel, component in enumerate(expected):
                    _assert_near(
                        float(actual[curve * 4 + channel]),
                        component,
                        f"{name} curve {curve} channel {channel}",
                    )
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
