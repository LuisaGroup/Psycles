"""Geometry attribute probes against raw Blender/Cycles node graphs."""

from __future__ import annotations

import math
from typing import Any

import bpy
from mathutils import Matrix, Vector

from .support import (
    _input,
    _input_identifier,
    _material,
    _material_matrix,
    _output,
    _output_identifier,
)


def _geometry_position_color_conversion(scene: Any) -> None:
    """Exercise Cycles' component-preserving point-to-color conversion."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    material, tree, output = _material("Geometry Position to Color")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Barbershop Geometry Position"
    mix = tree.nodes.new("ShaderNodeMix")
    mix.name = "Barbershop Position Color Add"
    mix.data_type = "RGBA"
    mix.blend_type = "ADD"
    mix.clamp_factor = True
    mix.clamp_result = False
    _input_identifier(mix, "Factor_Float").default_value = 0.2
    _input_identifier(mix, "A_Color").default_value = (
        0.5,
        0.5,
        0.5,
        1.0,
    )
    tree.links.new(
        _output(geometry, "Position"),
        _input_identifier(mix, "B_Color"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Position Color Emission"
    tree.links.new(
        _output_identifier(mix, "Result_Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _material_matrix(
        scene,
        [material],
        columns=1,
        rows=1,
        name="Geometry Position Color Surface",
        frame_bleed=0.02,
    )


def _texture_coordinate_object_transform(scene: Any) -> None:
    """Pin Cycles' explicit-projector versus shading-object coordinates."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    projector = bpy.data.objects.new("Coordinate Projector", None)
    projector.matrix_world = Matrix(
        (
            (0.75, -0.20, 0.00, 0.35),
            (0.15, 1.10, 0.00, -0.25),
            (0.00, 0.00, 1.00, 0.10),
            (0.00, 0.00, 0.00, 1.00),
        )
    )
    scene.collection.objects.link(projector)

    materials = []
    for explicit in (False, True):
        label = "Explicit Projector" if explicit else "Shading Object"
        material, tree, output = _material(
            f"Texture Coordinate Object - {label}"
        )
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        coordinates.name = f"{label} Coordinates"
        if explicit:
            coordinates.object = projector
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.name = f"{label} Display Mapping"
        mapping.vector_type = "POINT"
        _input(mapping, "Location").default_value = (0.5, 0.5, 0.5)
        _input(mapping, "Scale").default_value = (0.25, 0.25, 0.25)
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"{label} Emission"
        tree.links.new(
            _output(coordinates, "Object"),
            _input(mapping, "Vector"),
        )
        tree.links.new(
            _output(mapping, "Vector"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=2,
        rows=1,
        name="Texture Coordinate Object Matrix",
        frame_bleed=0.02,
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


def _displacement_material(
    name: str,
    method: str,
    color: tuple[float, float, float, float],
) -> Any:
    material, tree, output = _material(name)
    # Blender 4.1 moved the Cycles displacement mode from the custom Cycles
    # settings onto Material itself.
    material.displacement_method = method
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = f"{name} Diffuse"
    _input(diffuse, "Color").default_value = color
    _input(diffuse, "Roughness").default_value = 0.35
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )

    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = f"{name} Coordinates"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = f"{name} Separate UV"
    tree.links.new(
        _output(coordinates, "UV"),
        _input(separate, "Vector"),
    )
    frequency = tree.nodes.new("ShaderNodeMath")
    frequency.name = f"{name} Frequency"
    frequency.operation = "MULTIPLY"
    _input(frequency, "Value").default_value = 11.0
    tree.links.new(
        _output(separate, "X"),
        frequency.inputs[0],
    )
    sine = tree.nodes.new("ShaderNodeMath")
    sine.name = f"{name} Sine Height"
    sine.operation = "SINE"
    tree.links.new(
        _output(frequency, "Value"),
        sine.inputs[0],
    )
    displacement = tree.nodes.new("ShaderNodeDisplacement")
    displacement.name = f"{name} Displacement"
    _input(displacement, "Midlevel").default_value = 0.0
    _input(displacement, "Scale").default_value = 0.22
    tree.links.new(
        _output(sine, "Value"),
        _input(displacement, "Height"),
    )
    tree.links.new(
        _output(displacement, "Displacement"),
        _input(output, "Displacement"),
    )
    return material


def _displacement_grid(
    scene: Any,
    name: str,
    x_center: float,
    materials: list[Any],
    *,
    mixed: bool = False,
) -> None:
    resolution = 12
    vertices = []
    faces = []
    for y_index in range(resolution + 1):
        y = -0.62 + 1.24 * y_index / resolution
        for x_index in range(resolution + 1):
            x = -0.52 + 1.04 * x_index / resolution
            vertices.append((x, y, 0.0))
    for y_index in range(resolution):
        for x_index in range(resolution):
            lower = y_index * (resolution + 1) + x_index
            faces.append(
                (
                    lower,
                    lower + 1,
                    lower + resolution + 2,
                    lower + resolution + 1,
                )
            )
    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(vertices, (), faces)
    for material in materials:
        mesh.materials.append(material)
    for index, polygon in enumerate(mesh.polygons):
        polygon.use_smooth = True
        if mixed:
            polygon.material_index = (index // resolution + index) & 1
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        vertex = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (
            (vertex.x + 0.52) / 1.04,
            (vertex.y + 0.62) / 1.24,
        )
    mesh.update()
    surface = bpy.data.objects.new(name, mesh)
    surface.location.x = x_center
    scene.collection.objects.link(surface)


def _geometry_displacement_methods(scene: Any) -> None:
    """True/BOTH geometry, shared-vertex ownership, and bump-only control."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.light_sampling_threshold = 0.0

    camera = scene.camera
    camera.data.ortho_scale = 4.9
    camera.location = (3.4, -4.6, 3.2)
    camera.rotation_euler = (
        Vector((0.0, 0.0, 0.0)) - camera.location
    ).to_track_quat("-Z", "Y").to_euler()

    light_data = bpy.data.lights.new("Displacement Sun", type="SUN")
    light_data.energy = 2.5
    light_data.angle = 0.0
    light = bpy.data.objects.new(light_data.name, light_data)
    light.rotation_euler = (0.55, -0.35, -0.4)
    scene.collection.objects.link(light)

    bump = _displacement_material(
        "Bump Control", "BUMP", (0.55, 0.16, 0.08, 1.0)
    )
    true = _displacement_material(
        "True Displacement", "DISPLACEMENT", (0.08, 0.48, 0.16, 1.0)
    )
    both = _displacement_material(
        "Both Displacement", "BOTH", (0.08, 0.2, 0.62, 1.0)
    )
    _displacement_grid(scene, "Bump Control", -1.65, [bump])
    _displacement_grid(scene, "True Displacement", -0.55, [true])
    _displacement_grid(scene, "Both Displacement", 0.55, [both])
    _displacement_grid(
        scene,
        "Mixed Shared Vertices",
        1.65,
        [bump, true],
        mixed=True,
    )
