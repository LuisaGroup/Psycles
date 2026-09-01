"""Regression for the Blender 5.2 IES source-byte export contract."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile

import bpy


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
        bpy.data.texts,
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
    material = bpy.data.materials.new("IES Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None
    tree.nodes.clear()

    internal_source = (
        "IESNA:LM-63-2002\n[TEST] INTERNAL UTF-8 \u03bc\nTILT=NONE\n"
    )
    internal_text = bpy.data.texts.new("Internal Profile Trailing.ies")
    internal_text.write(internal_source)
    internal = tree.nodes.new("ShaderNodeTexIES")
    internal.name = "IES Internal Trailing"
    internal.mode = "INTERNAL"
    internal.ies = internal_text

    internal_no_final_text = bpy.data.texts.new(
        "Internal Profile No Final LF.ies"
    )
    internal_no_final_source = (
        "IESNA:LM-63-2002\n[TEST] NO FINAL LF\nTILT=NONE"
    )
    internal_no_final_text.write(internal_no_final_source)
    internal_no_final = tree.nodes.new("ShaderNodeTexIES")
    internal_no_final.name = "IES Internal No Final LF"
    internal_no_final.mode = "INTERNAL"
    internal_no_final.ies = internal_no_final_text

    with tempfile.TemporaryDirectory(
        prefix="psycles-ies-export-source-"
    ) as source_temporary:
        source_directory = pathlib.Path(source_temporary)
        external_path = source_directory / "External Profile.ies"
        external_bytes = b"IESNA:LM-63-2002\r\nTILT=NONE\r\n\x00\xff"
        external_path.write_bytes(external_bytes)
        external = tree.nodes.new("ShaderNodeTexIES")
        external.name = "IES External"
        external.mode = "EXTERNAL"
        external.filepath = str(external_path)

        missing_path = source_directory / "Missing Profile.ies"
        missing = tree.nodes.new("ShaderNodeTexIES")
        missing.name = "IES Missing"
        missing.mode = "EXTERNAL"
        missing.filepath = str(missing_path)

        output = tree.nodes.new("ShaderNodeOutputMaterial")
        emission = tree.nodes.new("ShaderNodeEmission")
        tree.links.new(
            internal.outputs["Factor"], emission.inputs["Strength"]
        )
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

        mesh = bpy.data.meshes.new("IES Mesh")
        mesh.from_pydata(
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
            (),
            ((0, 1, 2),),
        )
        mesh.materials.append(material)
        surface = bpy.data.objects.new("IES Surface", mesh)
        scene.collection.objects.link(surface)
        bpy.context.view_layer.update()

        with tempfile.TemporaryDirectory(
            prefix="psycles-ies-export-bundle-"
        ) as bundle_temporary:
            bundle = pathlib.Path(bundle_temporary)
            old_argv = sys.argv
            try:
                sys.argv = [str(exporter), "--", str(bundle)]
                runpy.run_path(str(exporter), run_name="__main__")
            finally:
                sys.argv = old_argv
            manifest = json.loads(
                (bundle / "scene.json").read_text(encoding="utf-8")
            )

    exported_material = next(
        item for item in manifest["materials"] if item["name"] == material.name
    )
    nodes = {
        node["name"]: node
        for node in exported_material["node_tree"]["nodes"]
    }

    expected = {
        "IES Internal Trailing": {
            "mode": "INTERNAL",
            "source": internal_text.name,
            "available": True,
            # Cycles appends one LF per Blender TextLine, including the final
            # empty line created by a source that already ends in LF.
            "content_bytes": list((internal_source + "\n").encode("utf-8")),
        },
        "IES Internal No Final LF": {
            "mode": "INTERNAL",
            "source": internal_no_final_text.name,
            "available": True,
            "content_bytes": list(
                (internal_no_final_source + "\n").encode("utf-8")
            ),
        },
        "IES External": {
            "mode": "EXTERNAL",
            "source": str(external_path),
            "available": True,
            "content_bytes": list(external_bytes),
        },
        "IES Missing": {
            "mode": "EXTERNAL",
            "source": str(missing_path),
            "available": False,
            "content_bytes": [],
        },
    }
    for name, expected_ies in expected.items():
        raw = nodes[name]
        actual = raw.get("special", {}).get("ies")
        if raw.get("type") != "TEX_IES" or actual != expected_ies:
            raise AssertionError(f"IES export contract changed for {name}: {raw}")
        if set(actual) != {"mode", "source", "available", "content_bytes"}:
            raise AssertionError(f"IES exporter baked derived data for {name}: {actual}")

    print("Psycles Blender IES raw-source export regression passed")


if __name__ == "__main__":
    _main()
