"""Geometry attribute probes against raw Blender/Cycles node graphs."""

from __future__ import annotations

import math
from typing import Any

import bpy

from .support import (
    _input,
    _material,
    _material_matrix,
    _output,
    _output_identifier,
)


def _geometry_attribute_outputs(scene: Any) -> None:
    """Pin Cycles RGBA projection and missing-attribute semantics."""

    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0
    outputs = (
        ("Color", "Barbershop Dirt"),
        ("Vector", "Barbershop Dirt"),
        ("Fac", "Barbershop Dirt"),
        ("Alpha", "Barbershop Dirt"),
        ("Color", "Missing Barbershop Attribute"),
        ("Vector", "Missing Barbershop Attribute"),
        ("Fac", "Missing Barbershop Attribute"),
        ("Alpha", "Missing Barbershop Attribute"),
    )
    materials = []
    for index, (socket, name) in enumerate(outputs):
        material, tree, output = _material(
            f"Geometry Attribute {index:02d} {socket}"
        )
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.name = f"Raw Cycles Attribute {index:02d}"
        attribute.attribute_type = "GEOMETRY"
        attribute.attribute_name = name
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Attribute {socket} Emission"
        source = (
            _output_identifier(attribute, "Fac")
            if socket == "Fac"
            else _output(attribute, socket)
        )
        tree.links.new(source, _input(emission, "Color"))
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    surface = _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="Geometry Attribute Outputs Matrix",
        frame_bleed=0.02,
    )
    # Barbershop stores the Dirt/Col masks as corner-domain BYTE_COLOR.
    # Keeping that exact source type makes both Cycles and the exporter apply
    # Blender's quantized sRGB-to-scene-linear conversion before shading.
    color = surface.data.color_attributes.new(
        name="Barbershop Dirt",
        type="BYTE_COLOR",
        domain="CORNER",
    )
    values = (
        (0.15, 0.45, 0.90, 0.35),
        (0.90, 0.15, 0.45, 0.80),
        (0.12, 0.42, 0.72, 0.33),
        (0.64, 0.24, 0.11, 0.27),
        (0.31, 0.52, 0.73, 0.94),
        (0.31, 0.52, 0.73, 0.94),
        (0.31, 0.52, 0.73, 0.94),
        (0.31, 0.52, 0.73, 0.94),
    )
    for polygon, value in zip(surface.data.polygons, values):
        for loop_index in polygon.loop_indices:
            color.data[loop_index].color = (*value[:3], value[3])


def _height(x: float, y: float) -> float:
    convex = 0.42 * math.exp(-7.0 * ((x + 0.48) ** 2 + y**2))
    concave = 0.36 * math.exp(-8.0 * ((x - 0.46) ** 2 + y**2))
    ripple = 0.045 * math.sin(5.0 * x - 1.7 * y)
    return convex - concave + ripple


def _geometry_pointiness(scene: Any) -> None:
    """Exercise curvature, boundary blur, duplicate welding, and edge dedup."""

    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    material, tree, output = _material("Geometry Pointiness")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Raw Geometry Pointiness"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Pointiness Emission"
    tree.links.new(
        _output(geometry, "Pointiness"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )

    # Build two disconnected patches that meet at x=0. The coincident seam
    # vertices and duplicated seam edges are deliberate: Cycles must quotient
    # them before welding normals and constructing the blurred attribute.
    resolution = 24
    half = resolution // 2
    coordinates = [
        -1.15 + 2.30 * index / resolution
        for index in range(resolution + 1)
    ]
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for first_x, last_x in ((0, half), (half, resolution)):
        patch_width = last_x - first_x + 1
        base = len(vertices)
        for y_index in range(resolution + 1):
            y = coordinates[y_index]
            for x_index in range(first_x, last_x + 1):
                x = coordinates[x_index]
                vertices.append((x, y, _height(x, y)))
        for y_index in range(resolution):
            for local_x in range(patch_width - 1):
                lower = base + y_index * patch_width + local_x
                upper = lower + patch_width
                faces.append(
                    (lower, lower + 1, upper + 1, upper)
                )

    mesh = bpy.data.meshes.new("Geometry Pointiness Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.update()
    surface = bpy.data.objects.new("Probe Surface", mesh)
    scene.collection.objects.link(surface)
