"""Regression for Cycles 5.2 CurveMapping SVM table export."""

from __future__ import annotations

import json
import pathlib
import runpy
import sys
import tempfile
from typing import Any

import bpy


def _clear_scene() -> None:
    for datablocks in (
        bpy.data.objects,
        bpy.data.materials,
        bpy.data.meshes,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock, do_unlink=True)


def _shape_curve(
    curve: Any,
    domain: tuple[float, float],
    midpoint: tuple[float, float],
) -> None:
    domain_min, domain_max = domain
    curve.points[0].location = (domain_min, 0.13)
    curve.points[-1].location = (domain_max, 0.87)
    point = curve.points.new(
        domain_min + midpoint[0] * (domain_max - domain_min),
        midpoint[1],
    )
    point.handle_type = "AUTO"


def _expected_domain(mapping: Any, curve_count: int) -> tuple[float, float]:
    curves = list(mapping.curves)[:curve_count]
    return (
        min(float(curve.points[0].location[0]) for curve in curves),
        max(float(curve.points[-1].location[0]) for curve in curves),
    )


def _require_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1.0e-7 * max(1.0, abs(expected)):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _validate_vector_samples(payload: dict[str, Any], mapping: Any) -> None:
    curve_mapping = payload["special"]["curve_mapping"]
    samples = curve_mapping["samples"]
    if len(samples) != 257 or any(len(sample) != 3 for sample in samples):
        raise AssertionError("Vector Curves did not export 257 float3 samples")
    minimum, maximum = _expected_domain(mapping, 3)
    if (
        float(curve_mapping["min_x"]) != minimum
        or float(curve_mapping["max_x"]) != maximum
        or not curve_mapping["extrapolate"]
    ):
        raise AssertionError(
            f"Vector Curves domain/extend changed: {curve_mapping}"
        )
    curves = list(mapping.curves)
    for index in (0, 1, 64, 128, 255, 256):
        x = minimum + index / 256.0 * (maximum - minimum)
        for channel in range(3):
            _require_close(
                float(samples[index][channel]),
                float(mapping.evaluate(curves[channel], x)),
                f"Vector Curves sample {index} channel {channel}",
            )


def _validate_float_samples(payload: dict[str, Any], mapping: Any) -> None:
    curve_mapping = payload["special"]["curve_mapping"]
    samples = curve_mapping["samples"]
    if len(samples) != 257 or any(
        not isinstance(sample, (int, float)) for sample in samples
    ):
        raise AssertionError("Float Curve did not export 257 scalar samples")
    minimum, maximum = _expected_domain(mapping, 1)
    if (
        float(curve_mapping["min_x"]) != minimum
        or float(curve_mapping["max_x"]) != maximum
        or curve_mapping["extrapolate"]
    ):
        raise AssertionError(
            f"Float Curve domain/extend changed: {curve_mapping}"
        )
    curve = mapping.curves[0]
    for index in (0, 1, 64, 128, 255, 256):
        x = minimum + index / 256.0 * (maximum - minimum)
        _require_close(
            float(samples[index]),
            float(mapping.evaluate(curve, x)),
            f"Float Curve sample {index}",
        )


def _validate_rgb_samples(payload: dict[str, Any], mapping: Any) -> None:
    curve_mapping = payload["special"]["curve_mapping"]
    samples = curve_mapping["samples"]
    if len(samples) != 257 or any(len(sample) != 3 for sample in samples):
        raise AssertionError("RGB Curves did not export 257 float3 samples")
    minimum, maximum = _expected_domain(mapping, 4)
    curves = list(mapping.curves)
    for index in (0, 1, 64, 128, 255, 256):
        x = minimum + index / 256.0 * (maximum - minimum)
        common = mapping.evaluate(curves[3], x)
        for channel in range(3):
            _require_close(
                float(samples[index][channel]),
                float(mapping.evaluate(curves[channel], common)),
                f"RGB Curves sample {index} channel {channel}",
            )


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("expected exporter path after '--'")
    exporter = pathlib.Path(args[0]).resolve()

    _clear_scene()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    material = bpy.data.materials.new("Curve Mapping Export")
    material.use_nodes = True
    tree = material.node_tree
    assert tree is not None

    vector = tree.nodes.new("ShaderNodeVectorCurve")
    vector.name = "Vector Curves Export"
    vector.mapping.extend = "EXTRAPOLATED"
    vector_domains = ((-0.3, 1.1), (-0.1, 1.3), (-0.2, 1.2))
    for index, (curve, domain) in enumerate(
        zip(vector.mapping.curves, vector_domains, strict=True)
    ):
        _shape_curve(curve, domain, (0.27 + index * 0.13, 0.81 - index * 0.2))
    vector.mapping.update()

    scalar = tree.nodes.new("ShaderNodeFloatCurve")
    scalar.name = "Float Curve Export"
    scalar.mapping.extend = "HORIZONTAL"
    _shape_curve(scalar.mapping.curves[0], (-0.4, 1.4), (0.43, 0.22))
    scalar.mapping.update()

    rgb = tree.nodes.new("ShaderNodeRGBCurve")
    rgb.name = "RGB Curves Export"
    rgb.mapping.extend = "EXTRAPOLATED"
    rgb_domains = ((-0.2, 1.0), (0.0, 1.2), (-0.1, 1.1), (-0.3, 1.3))
    for index, (curve, domain) in enumerate(
        zip(rgb.mapping.curves, rgb_domains, strict=True)
    ):
        _shape_curve(curve, domain, (0.21 + index * 0.14, 0.76 - index * 0.11))
    rgb.mapping.update()

    mesh = bpy.data.meshes.new("Curve Mapping Mesh")
    mesh.from_pydata(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        (),
        ((0, 1, 2),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new("Curve Mapping Surface", mesh)
    scene.collection.objects.link(surface)
    bpy.context.view_layer.update()

    with tempfile.TemporaryDirectory(
        prefix="psycles-curve-mapping-export-"
    ) as temporary:
        output = pathlib.Path(temporary)
        old_argv = sys.argv
        try:
            sys.argv = [str(exporter), "--", str(output)]
            runpy.run_path(str(exporter), run_name="__main__")
        finally:
            sys.argv = old_argv
        manifest = json.loads(
            (output / "scene.json").read_text(encoding="utf-8")
        )

    exported_material = next(
        item
        for item in manifest["materials"]
        if item["name"] == material.name
    )
    nodes = {
        node["name"]: node
        for node in exported_material["node_tree"]["nodes"]
    }
    _validate_vector_samples(nodes[vector.name], vector.mapping)
    _validate_float_samples(nodes[scalar.name], scalar.mapping)
    _validate_rgb_samples(nodes[rgb.name], rgb.mapping)
    print("Psycles CurveMapping export regression passed")


if __name__ == "__main__":
    _main()
