"""Geometry attribute probes against raw Blender/Cycles node graphs."""

from __future__ import annotations

import math
from typing import Any

import bpy

from .support import _input, _material, _output


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
