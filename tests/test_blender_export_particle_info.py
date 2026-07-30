"""Blender-side regression for Cycles Particle Info instance indexing."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile
from typing import Any

import bpy


_PARENT_COUNT = 8
_VIEWPORT_CHILDREN = 1
_RENDER_CHILDREN = 3


def _clear_scene() -> None:
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in tuple(bpy.data.materials):
        bpy.data.materials.remove(material)


def _particle_material() -> Any:
    material = bpy.data.materials.new("Particle Info Random")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    particle = tree.nodes.new("ShaderNodeParticleInfo")
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(
        particle.outputs["Random"],
        emission.inputs["Color"],
    )
    tree.links.new(
        emission.outputs["Emission"],
        output.inputs["Surface"],
    )
    return material


def _instance_object(name: str, material: Any, x: float) -> Any:
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=1,
        radius=0.08,
        enter_editmode=False,
        location=(x, 0.0, 20.0),
    )
    instance = bpy.context.object
    instance.name = name
    instance.data.materials.append(material)
    return instance


def _emitter(
    name: str,
    instance: Any,
    x: float,
    *,
    children: bool,
) -> None:
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=4,
        y_subdivisions=4,
        size=1.0,
        enter_editmode=False,
        location=(x, 0.0, 0.0),
    )
    emitter = bpy.context.object
    emitter.name = name
    bpy.context.view_layer.objects.active = emitter
    emitter.select_set(True)
    bpy.ops.object.particle_system_add()
    settings = emitter.particle_systems[-1].settings
    settings.type = "HAIR"
    settings.count = _PARENT_COUNT
    settings.hair_length = 0.1
    settings.render_type = "OBJECT"
    settings.instance_object = instance
    settings.particle_size = 1.0
    settings.size_random = 0.0
    settings.emit_from = "VERT"
    settings.use_modifier_stack = True
    # Cycles keeps evaluating the particle instances but rejects the
    # emitter's own mesh through OB_VISIBLE_SELF in the render graph.
    emitter.show_instancer_for_render = False
    if children:
        settings.child_type = "INTERPOLATED"
        settings.child_percent = _VIEWPORT_CHILDREN
        settings.rendered_child_count = _RENDER_CHILDREN


def _assert_parent_indices(instances: list[dict[str, Any]]) -> None:
    if not instances:
        raise AssertionError("parent-particle fixture exported no instances")
    nonzero = 0
    for instance in instances:
        persistent_index = int(instance["persistent_id"][0])
        exported_index = int(instance["particle_index"])
        if persistent_index < 0 or persistent_index >= _PARENT_COUNT:
            raise AssertionError(
                "parent fixture produced an out-of-domain persistent ID: "
                f"{persistent_index}"
            )
        if exported_index != persistent_index:
            raise AssertionError(
                "parent particle index did not preserve the Cycles-visible "
                f"source index: {persistent_index} -> {exported_index}"
            )
        nonzero += exported_index != 0
    if nonzero == 0:
        raise AssertionError(
            "parent fixture did not cover a nonzero Particle Info index"
        )


def _assert_child_sentinel(
    instances: list[dict[str, Any]],
    viewport_instance_count: int,
) -> None:
    if not instances:
        raise AssertionError("child-particle fixture exported no instances")
    expected_render_count = _PARENT_COUNT * _RENDER_CHILDREN
    if len(instances) != expected_render_count:
        raise AssertionError(
            "export did not use the final-render particle child count: "
            f"expected {expected_render_count}, got {len(instances)}"
        )
    if len(instances) <= viewport_instance_count:
        raise AssertionError(
            "export retained the viewport dependency graph: "
            f"{len(instances)} render instances versus "
            f"{viewport_instance_count} viewport instances"
        )
    child_count = 0
    for instance in instances:
        persistent_index = int(instance["persistent_id"][0])
        exported_index = int(instance["particle_index"])
        if persistent_index >= _PARENT_COUNT:
            child_count += 1
            if exported_index != 0:
                raise AssertionError(
                    "Cycles rejects child particles outside the parent "
                    "table, but the exporter retained dependency-graph ID "
                    f"{persistent_index} as Particle Info index "
                    f"{exported_index}"
                )
        elif exported_index != persistent_index:
            raise AssertionError(
                "a parent emitted by the child fixture lost its source "
                f"index: {persistent_index} -> {exported_index}"
            )
    if child_count == 0:
        raise AssertionError(
            "child fixture did not cover a persistent ID beyond totpart"
        )


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected the exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    material = _particle_material()
    parent_instance = _instance_object(
        "Parent Particle Instance",
        material,
        -1.0,
    )
    child_instance = _instance_object(
        "Child Particle Instance",
        material,
        1.0,
    )
    _emitter(
        "Parent Particle Emitter",
        parent_instance,
        -1.0,
        children=False,
    )
    _emitter(
        "Child Particle Emitter",
        child_instance,
        1.0,
        children=True,
    )
    bpy.context.view_layer.update()
    original_engine = bpy.context.scene.render.engine
    viewport_depsgraph = bpy.context.evaluated_depsgraph_get()
    viewport_child_count = sum(
        bool(instance.is_instance)
        and instance.object.name == "Child Particle Instance"
        for instance in viewport_depsgraph.object_instances
    )
    expected_viewport_count = _PARENT_COUNT * _VIEWPORT_CHILDREN
    if viewport_child_count != expected_viewport_count:
        raise AssertionError(
            "particle fixture did not establish distinct viewport "
            "semantics: expected "
            f"{expected_viewport_count}, got {viewport_child_count}"
        )

    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-particle-info-"
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
        exported_names = {
            instance["name"] for instance in scene["instances"]
        }
        for emitter_name in (
            "Parent Particle Emitter",
            "Child Particle Emitter",
        ):
            if emitter_name in exported_names:
                raise AssertionError(
                    "render-disabled particle emitter self geometry "
                    f"was exported: {emitter_name}"
                )
        if scene["camera"] is not None:
            raise AssertionError(
                "the temporary render-graph camera leaked into the export"
            )
        if bpy.context.scene.camera is not None:
            raise AssertionError(
                "the exporter did not remove its temporary camera"
            )
        if bpy.context.scene.render.engine != original_engine:
            raise AssertionError(
                "the exporter did not restore the source render engine"
            )
        parent_instances = [
            instance
            for instance in scene["instances"]
            if instance["is_instance"]
            and instance["name"] == "Parent Particle Instance"
        ]
        child_instances = [
            instance
            for instance in scene["instances"]
            if instance["is_instance"]
            and instance["name"] == "Child Particle Instance"
        ]
        _assert_parent_indices(parent_instances)
        _assert_child_sentinel(
            child_instances,
            viewport_child_count,
        )

    print("Psycles Blender Particle Info regression passed")


if __name__ == "__main__":
    _main()
