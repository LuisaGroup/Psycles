"""Cycles 5.2 SVM geometry-dependent node probes."""

from __future__ import annotations

import math
from typing import Any

import bpy
from mathutils import Euler, Matrix, Vector

from .support import _input, _material, _output


def _wireframe_material(
    name: str,
    *,
    use_pixel_size: bool,
    size: float,
    linked_size: bool,
) -> Any:
    """Build an unbaked Wireframe-to-Emission Cycles graph."""
    material, tree, output = _material(name)
    wireframe = tree.nodes.new("ShaderNodeWireframe")
    wireframe.name = f"{name} Wireframe"
    wireframe.use_pixel_size = use_pixel_size
    if linked_size:
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        geometry.name = f"{name} Geometry"
        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        separate.name = f"{name} Separate Normal"
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Scale Size"
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = size
        tree.links.new(
            _output(geometry, "Normal"),
            _input(separate, "Vector"),
        )
        tree.links.new(_output(separate, "Z"), scale.inputs[0])
        tree.links.new(_output(scale, "Value"), _input(wireframe, "Size"))
    else:
        _input(wireframe, "Size").default_value = size

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    tree.links.new(_output(wireframe, "Fac"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _wireframe_row_mesh(
    name: str,
    materials: list[Any],
    y0: float,
    y1: float,
    object_to_world: Matrix,
) -> Any:
    """Create final-world triangles in the object's untransformed mesh space."""
    columns = len(materials)
    subdivisions = 4
    extent = 1.1
    world_to_object = object_to_world.inverted()
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_materials: list[int] = []
    for material_index in range(columns):
        x0 = -extent + 2.0 * extent * material_index / columns
        x1 = -extent + 2.0 * extent * (material_index + 1) / columns
        first = len(vertices)
        stride = subdivisions + 1
        for y_index in range(stride):
            v = y_index / subdivisions
            for x_index in range(stride):
                u = x_index / subdivisions
                world = Vector(
                    (
                        x0 + (x1 - x0) * u,
                        y0 + (y1 - y0) * v,
                        0.0,
                    )
                )
                local = world_to_object @ world
                vertices.append(tuple(local))
        for y_index in range(subdivisions):
            for x_index in range(subdivisions):
                lower = first + y_index * stride + x_index
                upper = lower + stride
                if (x_index + y_index) & 1:
                    faces.extend(
                        (
                            (lower, lower + 1, upper),
                            (lower + 1, upper + 1, upper),
                        )
                    )
                else:
                    faces.extend(
                        (
                            (lower, lower + 1, upper + 1),
                            (lower, upper + 1, upper),
                        )
                    )
                face_materials.extend((material_index, material_index))

    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(vertices, (), faces)
    for material in materials:
        mesh.materials.append(material)
    for polygon, material_index in zip(
        mesh.polygons, face_materials, strict=True
    ):
        polygon.material_index = material_index
    mesh.update()

    surface = bpy.data.objects.new(name, mesh)
    surface.matrix_world = object_to_world
    bpy.context.scene.collection.objects.link(surface)
    return surface


def _vertex_color_material(
    name: str,
    *,
    layer_name: str,
    output_name: str,
) -> Any:
    """Build an unbaked Vertex Color-to-Emission Cycles graph."""
    material, tree, output = _material(name)
    vertex_color = tree.nodes.new("ShaderNodeVertexColor")
    vertex_color.name = f"{name} Vertex Color"
    vertex_color.layer_name = layer_name

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    if output_name == "Color":
        tree.links.new(
            _output(vertex_color, "Color"),
            _input(emission, "Color"),
        )
    elif output_name == "Alpha":
        _input(emission, "Color").default_value = (0.31, 0.57, 0.83, 1.0)
        tree.links.new(
            _output(vertex_color, "Alpha"),
            _input(emission, "Strength"),
        )
    else:
        raise ValueError(f"unsupported Vertex Color output {output_name!r}")
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _attribute_material(
    name: str,
    *,
    attribute_name: str,
    output_name: str,
) -> Any:
    """Build an unbaked Attribute-to-Emission Cycles graph."""
    material, tree, output = _material(name)
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.name = f"{name} Attribute"
    attribute.attribute_name = attribute_name

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    if output_name in {"Color", "Vector"}:
        tree.links.new(
            _output(attribute, output_name),
            _input(emission, "Color"),
        )
    elif output_name in {"Fac", "Alpha"}:
        _input(emission, "Color").default_value = (0.31, 0.57, 0.83, 1.0)
        tree.links.new(
            _output(attribute, output_name),
            _input(emission, "Strength"),
        )
    else:
        raise ValueError(f"unsupported Attribute output {output_name!r}")
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _attribute_bump_material(
    name: str,
    *,
    use_vertex_color: bool,
) -> Any:
    """Build the exact Attribute/Vertex Color subgraph cloned by Cycles Bump."""
    material, tree, output = _material(name)
    if use_vertex_color:
        attribute = tree.nodes.new("ShaderNodeVertexColor")
        attribute.name = f"{name} Vertex Color"
        attribute.layer_name = "ProbeColor"
    else:
        attribute = tree.nodes.new("ShaderNodeAttribute")
        attribute.name = f"{name} Attribute"
        attribute.attribute_name = "ProbeColor"

    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = f"{name} Bump"
    bump.invert = False
    _input(bump, "Strength").default_value = 0.73
    _input(bump, "Distance").default_value = 0.41
    _input(bump, "Filter Width").default_value = 0.29
    tree.links.new(_output(attribute, "Alpha"), _input(bump, "Height"))

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    tree.links.new(_output(bump, "Normal"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _volume_attribute_material(name: str, *, nonlinear: bool) -> Any:
    """Build the Attribute volume path audited by optimize_volume_output."""
    material, tree, output = _material(name)
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.name = f"{name} Attribute"
    attribute.attribute_name = "density"
    density = _output(attribute, "Fac")
    if nonlinear:
        math_node = tree.nodes.new("ShaderNodeMath")
        math_node.name = f"{name} Sine"
        math_node.operation = "SINE"
        tree.links.new(density, math_node.inputs[0])
        density = _output(math_node, "Value")

    volume = tree.nodes.new("ShaderNodeVolumeAbsorption")
    volume.name = f"{name} Volume Absorption"
    tree.links.new(density, _input(volume, "Density"))
    tree.links.new(_output(volume, "Volume"), _input(output, "Volume"))
    return material


def _svm_vertex_color(scene: Any) -> None:
    """Freeze Cycles Vertex Color named/default IDs and output payloads."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    materials = [
        _vertex_color_material(
            "SVM Vertex Color Named Color",
            layer_name="ProbeColor",
            output_name="Color",
        ),
        _vertex_color_material(
            "SVM Vertex Color Named Alpha",
            layer_name="ProbeColor",
            output_name="Alpha",
        ),
        _vertex_color_material(
            "SVM Vertex Color Named Second",
            layer_name="ProbeColorSecond",
            output_name="Color",
        ),
        _vertex_color_material(
            "SVM Vertex Color Default",
            layer_name="",
            output_name="Color",
        ),
        _attribute_material(
            "SVM Attribute Named Color",
            attribute_name="ProbeColor",
            output_name="Color",
        ),
        _attribute_material(
            "SVM Attribute Named Alpha",
            attribute_name="ProbeColorSecond",
            output_name="Alpha",
        ),
        _attribute_material(
            "SVM Attribute Standard Fac",
            attribute_name="pointiness",
            output_name="Fac",
        ),
        _attribute_material(
            "SVM Attribute Empty Color",
            attribute_name="",
            output_name="Color",
        ),
        _volume_attribute_material(
            "SVM Attribute Volume Linear",
            nonlinear=False,
        ),
        _volume_attribute_material(
            "SVM Attribute Volume Nonlinear",
            nonlinear=True,
        ),
        _attribute_bump_material(
            "SVM Vertex Color Bump",
            use_vertex_color=True,
        ),
        _attribute_bump_material(
            "SVM Attribute Bump",
            use_vertex_color=False,
        ),
    ]
    surface = _wireframe_row_mesh(
        "SVM Vertex Color Surface",
        materials,
        -1.1,
        1.1,
        Matrix.Identity(4),
    )

    mesh = surface.data
    probe_color = mesh.color_attributes.new(
        name="ProbeColor",
        type="FLOAT_COLOR",
        domain="CORNER",
    )
    probe_color_second = mesh.color_attributes.new(
        name="ProbeColorSecond",
        type="BYTE_COLOR",
        domain="CORNER",
    )
    for index, item in enumerate(probe_color.data):
        u = (index % 11) / 10.0
        item.color = (0.13 + 0.61 * u, 0.79 - 0.47 * u, 0.24, 0.37 + 0.53 * u)
    for index, item in enumerate(probe_color_second.data):
        u = (index % 7) / 6.0
        item.color = (0.82 - 0.55 * u, 0.16 + 0.63 * u, 0.68, 0.91 - 0.41 * u)
    mesh.color_attributes.active_color = probe_color


def _svm_wireframe_matrix(scene: Any) -> None:
    """Exercise both size modes, input ABIs, and transform-applied branches."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    cases = (
        (False, 0.09, False, "World Immediate"),
        (False, 0.09, True, "World Stack"),
        (True, 2.5, False, "Pixel Immediate"),
        (True, 2.5, True, "Pixel Stack"),
    )
    materials = [
        _wireframe_material(
            f"SVM Wireframe {label}",
            use_pixel_size=use_pixel_size,
            size=size,
            linked_size=linked_size,
        )
        for use_pixel_size, size, linked_size, label in cases
    ]

    object_to_world = (
        Matrix.Translation((0.17, -0.13, 0.21))
        @ Euler((0.29, -0.21, 0.17), "XYZ").to_matrix().to_4x4()
        @ Matrix.Diagonal((1.31, 0.74, 1.16, 1.0))
    )

    # A single-user mesh takes Cycles' SD_OBJECT_TRANSFORM_APPLIED path.
    _wireframe_row_mesh(
        "SVM Wireframe Transform Applied",
        materials,
        -1.1,
        0.0,
        object_to_world,
    )

    # A second user prevents Cycles from baking the object transform. The
    # visible object therefore exercises object_position_transform(), while
    # its geometry-sharing twin stays outside the camera frame.
    shared = _wireframe_row_mesh(
        "SVM Wireframe Runtime Transform",
        materials,
        0.0,
        1.1,
        object_to_world,
    )
    twin = bpy.data.objects.new(
        "SVM Wireframe Runtime Transform Offscreen Twin",
        shared.data,
    )
    twin.matrix_world = Matrix.Translation((8.0, 0.0, 0.0)) @ object_to_world
    scene.collection.objects.link(twin)


def _svm_wireframe_bump(scene: Any) -> None:
    """Freeze Cycles' CENTER/DX/DY refinement of a Wireframe height graph."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    material, tree, output = _material("SVM Wireframe Bump")
    wireframe = tree.nodes.new("ShaderNodeWireframe")
    wireframe.name = "Wireframe Height"
    wireframe.use_pixel_size = False
    _input(wireframe, "Size").default_value = 0.13

    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = "Wireframe Bump"
    bump.invert = True
    _input(bump, "Strength").default_value = 0.8
    _input(bump, "Distance").default_value = 0.2
    _input(bump, "Filter Width").default_value = 0.37
    tree.links.new(_output(wireframe, "Fac"), _input(bump, "Height"))

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Bump Normal Emission"
    tree.links.new(_output(bump, "Normal"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    _wireframe_row_mesh(
        "SVM Wireframe Bump Surface",
        [material],
        -1.1,
        1.1,
        Matrix.Identity(4),
    )


def _svm_bump_constant_fold(scene: Any) -> None:
    """Freeze Cycles' unlinked-Height BumpNode constant-fold contract."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    material, tree, output = _material("SVM Bump Constant Fold")
    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = "Unlinked Height Bump"
    bump.invert = True
    _input(bump, "Strength").default_value = 0.37
    _input(bump, "Distance").default_value = 0.19
    _input(bump, "Filter Width").default_value = 0.23

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Folded Normal Emission"
    tree.links.new(_output(bump, "Normal"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    _wireframe_row_mesh(
        "SVM Bump Constant Fold Surface",
        [material],
        -1.1,
        1.1,
        Matrix.Identity(4),
    )


def _geometry_bump_material(name: str, output_name: str) -> Any:
    """Build the exact Geometry-output subgraph cloned by Cycles Bump."""
    material, tree, output = _material(name)
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = f"{name} Geometry"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = f"{name} Separate"
    tree.links.new(_output(geometry, output_name), _input(separate, "Vector"))

    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = f"{name} Bump"
    bump.invert = False
    _input(bump, "Strength").default_value = 0.73
    _input(bump, "Distance").default_value = 0.41
    _input(bump, "Filter Width").default_value = 0.29
    tree.links.new(_output(separate, "X"), _input(bump, "Height"))

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    tree.links.new(_output(bump, "Normal"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _svm_geometry_bump_offsets(scene: Any) -> None:
    """Freeze Cycles Geometry dual evaluation and CENTER/DX/DY emission."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    materials = [
        _geometry_bump_material("SVM Geometry Position Bump", "Position"),
        _geometry_bump_material("SVM Geometry Parametric Bump", "Parametric"),
    ]
    _wireframe_row_mesh(
        "SVM Geometry Bump Offset Surface",
        materials,
        -1.1,
        1.1,
        Matrix.Identity(4),
    )


def _geometry_attribute_material(
    name: str,
    output_name: str,
    *,
    use_bump: bool,
) -> Any:
    """Build an unbaked Geometry standard-attribute SVM graph."""
    material, tree, output = _material(name)
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = f"{name} Geometry"

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    if use_bump:
        bump = tree.nodes.new("ShaderNodeBump")
        bump.name = f"{name} Bump"
        bump.invert = False
        _input(bump, "Strength").default_value = 0.73
        _input(bump, "Distance").default_value = 0.41
        _input(bump, "Filter Width").default_value = 0.29
        tree.links.new(
            _output(geometry, output_name),
            _input(bump, "Height"),
        )
        tree.links.new(
            _output(bump, "Normal"),
            _input(emission, "Color"),
        )
    else:
        tree.links.new(
            _output(geometry, output_name),
            _input(emission, "Color"),
        )
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    return material


def _svm_geometry_attributes(scene: Any) -> None:
    """Freeze Cycles Pointiness/Random NODE_ATTR surface and dual paths."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0

    cases = (
        ("Pointiness", False, "Pointiness Surface"),
        ("Random Per Island", False, "Random Surface"),
        ("Pointiness", True, "Pointiness Bump"),
        ("Random Per Island", True, "Random Bump"),
    )
    materials = [
        _geometry_attribute_material(
            f"SVM Geometry Attribute {label}",
            output_name,
            use_bump=use_bump,
        )
        for output_name, use_bump, label in cases
    ]

    # Four topologically disconnected curved patches make Random Per Island
    # non-degenerate while retaining a spatially varying Pointiness field.
    columns = 2
    rows = 2
    resolution = 12
    extent = 1.1
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    material_indices: list[int] = []
    for material_index in range(len(materials)):
        column = material_index % columns
        row = material_index // columns
        x0 = -extent + 2.0 * extent * column / columns
        x1 = -extent + 2.0 * extent * (column + 1) / columns
        y0 = -extent + 2.0 * extent * row / rows
        y1 = -extent + 2.0 * extent * (row + 1) / rows
        first = len(vertices)
        stride = resolution + 1
        for y_index in range(stride):
            v = y_index / resolution
            y = y0 + (y1 - y0) * v
            for x_index in range(stride):
                u = x_index / resolution
                x = x0 + (x1 - x0) * u
                z = (
                    0.13 * math.sin(4.1 * x + 0.7 * y)
                    + 0.09 * math.cos(2.3 * x - 3.7 * y)
                    + 0.06 * math.sin(5.2 * (x * x + y * y))
                )
                vertices.append((x, y, z))
        for y_index in range(resolution):
            for x_index in range(resolution):
                lower = first + y_index * stride + x_index
                upper = lower + stride
                faces.append(
                    (lower, lower + 1, upper + 1, upper)
                )
                material_indices.append(material_index)

    mesh = bpy.data.meshes.new("SVM Geometry Attribute Mesh")
    mesh.from_pydata(vertices, (), faces)
    for material in materials:
        mesh.materials.append(material)
    for polygon, material_index in zip(
        mesh.polygons, material_indices, strict=True
    ):
        polygon.material_index = material_index
        polygon.use_smooth = True
    mesh.update()
    surface = bpy.data.objects.new("SVM Geometry Attribute Surface", mesh)
    scene.collection.objects.link(surface)
