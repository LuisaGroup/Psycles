"""Regression for Blender 5.2 Object/Particle/Hair/Point Info export."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


_CONTRACTS = (
    (
        "ShaderNodeObjectInfo",
        "OBJECT_INFO",
        (
            ("Location", "NodeSocketVector"),
            ("Color", "NodeSocketColor"),
            ("Alpha", "NodeSocketFloat"),
            ("Object Index", "NodeSocketFloat"),
            ("Material Index", "NodeSocketFloat"),
            ("Random", "NodeSocketFloat"),
        ),
    ),
    (
        "ShaderNodeParticleInfo",
        "PARTICLE_INFO",
        (
            ("Index", "NodeSocketFloat"),
            ("Random", "NodeSocketFloat"),
            ("Age", "NodeSocketFloat"),
            ("Lifetime", "NodeSocketFloat"),
            ("Location", "NodeSocketVector"),
            ("Size", "NodeSocketFloat"),
            ("Velocity", "NodeSocketVector"),
            ("Angular Velocity", "NodeSocketVector"),
        ),
    ),
    (
        "ShaderNodeHairInfo",
        "HAIR_INFO",
        (
            ("Is Strand", "NodeSocketFloat"),
            ("Intercept", "NodeSocketFloat"),
            ("Length", "NodeSocketFloat"),
            ("Thickness", "NodeSocketFloat"),
            ("Tangent Normal", "NodeSocketVector"),
            ("Random", "NodeSocketFloat"),
        ),
    ),
    (
        "ShaderNodePointInfo",
        "POINT_INFO",
        (
            ("Position", "NodeSocketVector"),
            ("Radius", "NodeSocketFloat"),
            ("Random", "NodeSocketFloat"),
        ),
    ),
)


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    material = bpy.data.materials.new("Info Nodes Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    closures = []
    created = []
    for node_index, (bl_idname, _, sockets) in enumerate(_CONTRACTS):
        info = tree.nodes.new(bl_idname)
        info.name = f"Info {node_index}"
        created.append(info)
        if tuple(output.name for output in info.outputs) != tuple(
            name for name, _ in sockets
        ):
            raise AssertionError(
                f"Blender 5.2 socket order changed for {bl_idname}"
            )
        for output in info.outputs:
            emission = tree.nodes.new("ShaderNodeEmission")
            target = (
                emission.inputs["Strength"]
                if output.bl_idname == "NodeSocketFloat"
                else emission.inputs["Color"]
            )
            tree.links.new(output, target)
            closures.append(emission.outputs["Emission"])

    while len(closures) > 1:
        next_closures = []
        for index in range(0, len(closures), 2):
            if index + 1 == len(closures):
                next_closures.append(closures[index])
                continue
            add = tree.nodes.new("ShaderNodeAddShader")
            tree.links.new(closures[index], add.inputs[0])
            tree.links.new(closures[index + 1], add.inputs[1])
            next_closures.append(add.outputs[0])
        closures = next_closures
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(closures[0], output.inputs["Surface"])

    mesh = bpy.data.meshes.new("Info Nodes Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Info Nodes Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-info-nodes-export-"
    ) as temporary:
        directory = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(directory)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            (directory / "scene.json").read_text(encoding="utf-8")
        )

    exported_material = next(
        item for item in manifest["materials"] if item["name"] == material.name
    )
    raw_nodes = {
        node["name"]: node for node in exported_material["node_tree"]["nodes"]
    }
    for node_index, (bl_idname, raw_type, sockets) in enumerate(_CONTRACTS):
        node = raw_nodes[f"Info {node_index}"]
        actual = tuple(
            (
                socket["identifier"],
                socket["name"],
                socket["type"],
                socket["linked"],
            )
            for socket in node["outputs"]
        )
        expected = tuple(
            (name, name, socket_type, True)
            for name, socket_type in sockets
        )
        if (
            node["type"] != raw_type
            or node["bl_idname"] != bl_idname
            or node["inputs"]
            or actual != expected
        ):
            raise AssertionError(
                f"{bl_idname} export contract changed: {node}"
            )
    print("Psycles Blender Info-node export regression passed")


if __name__ == "__main__":
    _main()
