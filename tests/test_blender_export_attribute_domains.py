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
    edit_uv = mesh.uv_layers.new(name="Editing UV")
    for index, value in enumerate(
        ((8.0, 3.0), (9.0, 3.0), (9.0, 4.0), (8.0, 4.0))
    ):
        edit_uv.data[index].uv = value
    # Cycles' default Texture Coordinate UV follows the render/default map,
    # not the UV layer selected for editing in Blender's UI.
    mesh.uv_layers.active = edit_uv
    uv.active_render = True
    if (
        mesh.uv_layers.active != edit_uv
        or mesh.uv_layers.active_render != uv
    ):
        raise AssertionError(
            "multi-UV regression setup did not separate editing and "
            "render/default UV maps"
        )

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

    corner_byte_color = mesh.color_attributes.new(
        name="Corner Byte Color",
        type="BYTE_COLOR",
        domain="CORNER",
    )
    for item in corner_byte_color.data:
        item.color = (0.15, 0.45, 0.90, 0.35)
    byte_rec709_linear = tuple(
        float(component)
        for component in corner_byte_color.data[0].color
    )

    material = bpy.data.materials.new("Domain Material")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.name = "Cycles Attribute Contract"
    attribute.attribute_type = "GEOMETRY"
    attribute.attribute_name = "Corner Color"
    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.name = "Color Vector Factor Consumer"
    tree.links.new(attribute.outputs[0], mix.inputs[1])
    tree.links.new(attribute.outputs[1], mix.inputs[2])
    tree.links.new(attribute.outputs[2], mix.inputs[0])
    alpha = tree.nodes.new("ShaderNodeMath")
    alpha.name = "Alpha Consumer"
    alpha.operation = "ADD"
    alpha.inputs[1].default_value = 1.0
    tree.links.new(attribute.outputs[3], alpha.inputs[0])
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(mix.outputs[0], emission.inputs[0])
    tree.links.new(alpha.outputs[0], emission.inputs[1])
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs[0], output.inputs[0])
    mesh.materials.append(material)
    obj = bpy.data.objects.new("Domain Object", mesh)
    scene.collection.objects.link(obj)
    mesh.update()
    texspace_location = tuple(float(value) for value in mesh.texspace_location)
    texspace_size = tuple(float(value) for value in mesh.texspace_size)
    generated_scale = tuple(
        0.5 / value if value != 0.0 else 0.0
        for value in texspace_size
    )
    generated_offset = tuple(
        0.5 - texspace_location[axis] * generated_scale[axis]
        for axis in range(3)
    )
    expected_generated_transform = (
        generated_scale[0],
        0.0,
        0.0,
        0.0,
        0.0,
        generated_scale[1],
        0.0,
        0.0,
        0.0,
        0.0,
        generated_scale[2],
        0.0,
        generated_offset[0],
        generated_offset[1],
        generated_offset[2],
        1.0,
    )

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
        if geometry.get("default_uv_available") is not True:
            raise AssertionError(
                "real render UV layer was marked as placeholder storage"
            )
        actual_generated_transform = tuple(
            float(value)
            for value in geometry["generated_transform"]
        )
        if len(actual_generated_transform) != 16 or any(
            abs(actual - expected) > 1.0e-7
            for actual, expected in zip(
                actual_generated_transform,
                expected_generated_transform,
            )
        ):
            raise AssertionError(
                "volume Generated transform does not match Blender "
                f"texspace: {actual_generated_transform}, expected "
                f"{expected_generated_transform}"
            )
        generated_values = struct.unpack(
            "<12f",
            _read_section(geometry_path, geometry["generated"]),
        )
        for point_index, position in enumerate(positions):
            expected_generated = tuple(
                position[axis] * generated_scale[axis]
                + generated_offset[axis]
                for axis in range(3)
            )
            actual_generated = generated_values[
                point_index * 3 : point_index * 3 + 3
            ]
            if any(
                abs(actual - expected) > 1.0e-6
                for actual, expected in zip(
                    actual_generated,
                    expected_generated,
                )
            ):
                raise AssertionError(
                    "surface Generated values and volume transform "
                    f"diverged at point {point_index}: "
                    f"{actual_generated}, expected {expected_generated}"
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
        primary_uv = _read_section(geometry_path, geometry["uv"])
        render_uv = _read_section(
            geometry_path,
            uv_layers["Domain UV"]["values"],
        )
        editing_uv = _read_section(
            geometry_path,
            uv_layers["Editing UV"]["values"],
        )
        if primary_uv != render_uv:
            raise AssertionError(
                "default Texture Coordinate UV did not use Blender's "
                "render/default UV map"
            )
        if primary_uv == editing_uv:
            raise AssertionError(
                "default Texture Coordinate UV incorrectly followed the "
                "UI/editing-active UV map"
            )
        primary_tangents = _read_section(
            geometry_path,
            geometry["uv_tangents"],
        )
        render_tangents = _read_section(
            geometry_path,
            uv_layers["Domain UV"]["tangents"],
        )
        if primary_tangents != render_tangents:
            raise AssertionError(
                "default tangent frame did not follow the render/default "
                "UV map"
            )
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
        if (
            colors["Corner Byte Color"]["domain"] != "CORNER"
            or colors["Corner Byte Color"]["data_type"] != "BYTE_COLOR"
            or colors["Corner Byte Color"]["values"]["bytes"]
            != 6 * 4 * 4
        ):
            raise AssertionError("corner byte-color contract is invalid")
        rec709_to_rgb = scene["render"]["color_management"][
            "shader_transforms"
        ]["rec709_to_rgb"]
        expected_byte_scene_linear = tuple(
            sum(
                float(row[channel]) * byte_rec709_linear[channel]
                for channel in range(3)
            )
            for row in rec709_to_rgb
        ) + (byte_rec709_linear[3],)
        byte_values = struct.unpack(
            "<24f",
            _read_section(
                geometry_path,
                colors["Corner Byte Color"]["values"],
            ),
        )
        for corner in range(6):
            actual = byte_values[corner * 4 : corner * 4 + 4]
            if any(
                abs(component - expected) > 5.0e-7
                for component, expected in zip(
                    actual,
                    expected_byte_scene_linear,
                )
            ):
                raise AssertionError(
                    "Cycles CORNER/BYTE_COLOR Rec.709 conversion was lost: "
                    f"{actual}, expected {expected_byte_scene_linear}"
                )
        inspected = subprocess.run(
            [
                str(inspector),
                str(output),
                "Domain Material",
                "--require-generated-transform",
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
        if "node type 'ATTRIBUTE'" in inspected.stderr:
            raise AssertionError(
                "Cycles Attribute outputs fell back during typed lowering:\n"
                f"{inspected.stderr}"
            )

    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])
    mesh.update()
    with tempfile.TemporaryDirectory(
        prefix="psycles-blender-no-default-uv-"
    ) as temporary:
        output = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(output)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv
        no_uv_scene = json.loads(
            (output / "scene.json").read_text(encoding="utf-8")
        )
        no_uv_geometry = no_uv_scene["geometries"][0]
        if no_uv_geometry.get("default_uv_available") is not False:
            raise AssertionError(
                "absent default UV was confused with its zero placeholder"
            )
        if int(no_uv_geometry["uv"]["bytes"]) != 6 * 2 * 4:
            raise AssertionError(
                "absent UV did not retain the total coordinate buffer ABI"
            )

    print("Psycles Blender attribute-domain regression passed")


if __name__ == "__main__":
    _main()
