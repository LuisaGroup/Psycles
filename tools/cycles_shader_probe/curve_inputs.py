"""Vector and scalar CurveMapping probe scenes."""

from __future__ import annotations

from typing import Any

from .support import _input, _material, _material_matrix, _output


def _shape_curve(
    curve: Any,
    domain: tuple[float, float],
    shape: tuple[tuple[float, float], ...],
) -> None:
    domain_min, domain_max = domain
    curve.points[0].location = (domain_min, shape[0][1])
    curve.points[-1].location = (domain_max, shape[-1][1])
    domain_range = domain_max - domain_min
    for relative_x, value in shape[1:-1]:
        point = curve.points.new(domain_min + relative_x * domain_range, value)
        point.handle_type = "AUTO"


def _vector_curve_material(
    name: str,
    *,
    factor: float,
    extend: str,
    wide_domain: bool,
) -> Any:
    material, tree, output = _material(name)
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = f"{name} Coordinates"
    curves = tree.nodes.new("ShaderNodeVectorCurve")
    curves.name = name
    mapping = curves.mapping
    mapping.extend = extend
    mapping.use_clip = False
    domains = (
        ((-0.35, 1.25), (-0.15, 1.10), (-0.25, 1.35))
        if wide_domain
        else ((0.0, 1.0),) * 3
    )
    shapes = (
        ((0.0, 0.12), (0.31, 0.84), (0.73, 0.23), (1.0, 0.91)),
        ((0.0, 0.88), (0.27, 0.18), (0.69, 0.79), (1.0, 0.16)),
        ((0.0, 0.24), (0.39, 0.93), (0.76, 0.11), (1.0, 0.72)),
    )
    for curve, domain, shape in zip(
        mapping.curves, domains, shapes, strict=True
    ):
        _shape_curve(curve, domain, shape)
    mapping.update()
    _input(curves, "Factor").default_value = factor
    tree.links.new(_output(coordinates, "Generated"), _input(curves, "Vector"))
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(_output(curves, "Vector"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _vector_curve_matrix(scene: Any) -> None:
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (1.0, "EXTRAPOLATED", False),
        (0.37, "EXTRAPOLATED", False),
        (1.0, "HORIZONTAL", True),
        (1.0, "EXTRAPOLATED", True),
    )
    materials = [
        _vector_curve_material(
            f"Vector Curve {index:02d} {extend}",
            factor=factor,
            extend=extend,
            wide_domain=wide_domain,
        )
        for index, (factor, extend, wide_domain) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=2,
        rows=2,
        name="Vector Curve Matrix",
    )


def _float_curve_material(
    name: str,
    *,
    factor: float,
    extend: str,
    wide_domain: bool,
) -> Any:
    material, tree, output = _material(name)
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = f"{name} Coordinates"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = f"{name} X"
    tree.links.new(_output(coordinates, "Generated"), _input(separate, "Vector"))
    curve = tree.nodes.new("ShaderNodeFloatCurve")
    curve.name = name
    mapping = curve.mapping
    mapping.extend = extend
    mapping.use_clip = False
    domain = (-0.35, 1.25) if wide_domain else (0.0, 1.0)
    shape = (
        (0.0, 0.08),
        (0.23, 0.89),
        (0.61, 0.17),
        (0.82, 0.74),
        (1.0, 0.31),
    )
    _shape_curve(mapping.curves[0], domain, shape)
    mapping.update()
    _input(curve, "Factor").default_value = factor
    tree.links.new(_output(separate, "X"), _input(curve, "Value"))
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(_output(curve, "Value"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _float_curve_matrix(scene: Any) -> None:
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (1.0, "EXTRAPOLATED", False),
        (0.37, "EXTRAPOLATED", False),
        (1.0, "HORIZONTAL", True),
        (1.0, "EXTRAPOLATED", True),
    )
    materials = [
        _float_curve_material(
            f"Float Curve {index:02d} {extend}",
            factor=factor,
            extend=extend,
            wide_domain=wide_domain,
        )
        for index, (factor, extend, wide_domain) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=2,
        rows=2,
        name="Float Curve Matrix",
    )
