"""Shared Blender scene, socket, material, and graph construction helpers."""

from __future__ import annotations

from typing import Any

import bpy


def _clear() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.meshes,
        bpy.data.node_groups,
        bpy.data.worlds,
    ):
        for datablock in list(collection):
            collection.remove(datablock)


def _input(node: Any, name: str) -> Any:
    socket = node.inputs.get(name)
    if socket is None:
        raise RuntimeError(
            f"{node.bl_idname} has no input {name!r}; "
            f"available: {[item.name for item in node.inputs]}"
        )
    return socket


def _output(node: Any, name: str) -> Any:
    socket = node.outputs.get(name)
    if socket is None:
        raise RuntimeError(
            f"{node.bl_idname} has no output {name!r}; "
            f"available: {[item.name for item in node.outputs]}"
        )
    return socket


def _input_identifier(node: Any, identifier: str) -> Any:
    for socket in node.inputs:
        if socket.identifier == identifier:
            return socket
    raise RuntimeError(
        f"{node.bl_idname} has no input identifier {identifier!r}; "
        f"available: {[item.identifier for item in node.inputs]}"
    )


def _output_identifier(node: Any, identifier: str) -> Any:
    for socket in node.outputs:
        if socket.identifier == identifier:
            return socket
    raise RuntimeError(
        f"{node.bl_idname} has no output identifier {identifier!r}; "
        f"available: {[item.identifier for item in node.outputs]}"
    )


def _camera(scene: Any) -> None:
    data = bpy.data.cameras.new("Probe Camera")
    data.type = "ORTHO"
    data.ortho_scale = 2.2
    data.lens = 50.0
    camera = bpy.data.objects.new("Probe Camera", data)
    camera.location = (0.0, 0.0, 3.0)
    scene.collection.objects.link(camera)
    scene.camera = camera


def _bsdf_matrix_sun(
    scene: Any,
    *,
    transmission: bool,
) -> None:
    """Add a zero-angle Sun for variance-free BSDF evaluation."""
    data = bpy.data.lights.new("BSDF Matrix Sun", type="SUN")
    data.color = (0.36, 0.72, 1.0)
    data.energy = 1.7
    data.normalize = True
    data.angle = 0.0
    data.use_shadow = True
    light = bpy.data.objects.new(data.name, data)
    if transmission:
        light.rotation_euler = (3.141592653589793, 0.0, 0.0)
    scene.collection.objects.link(light)
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 1
    # Tiny closure-allocation threshold cases must not be randomized by
    # Cycles' direct-light sample roulette.
    scene.cycles.light_sampling_threshold = 0.0


def _plane(material: Any) -> None:
    bpy.ops.mesh.primitive_plane_add(
        # Fill the complete orthographic frame so value/closure probes have
        # no silhouette pixels whose reconstruction-filter coverage would
        # obscure the node formula being tested.
        size=8.0,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, 0.0),
    )
    plane = bpy.context.object
    plane.name = "Probe Surface"
    plane.data.materials.append(material)


def _material_matrix(
    scene: Any,
    materials: list[Any],
    columns: int,
    rows: int,
    name: str,
    *,
    backfacing: set[int] | None = None,
    frame_bleed: float = 0.0,
) -> Any:
    """Fill the orthographic frame with one material per contiguous cell."""
    if len(materials) != columns * rows:
        raise ValueError(
            f"{name}: expected {columns * rows} materials, "
            f"got {len(materials)}"
        )
    if frame_bleed < 0.0:
        raise ValueError(f"{name}: frame_bleed must be non-negative")
    # The orthographic probe camera spans exactly [-1.1, 1.1]. Cell
    # boundaries therefore land on integer pixels whenever the render
    # dimensions are divisible by the matrix dimensions.
    extent = 1.1
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for index in range(len(materials)):
        column = index % columns
        row = index // columns
        x0 = -extent + 2.0 * extent * column / columns
        x1 = -extent + 2.0 * extent * (column + 1) / columns
        y0 = -extent + 2.0 * extent * row / rows
        y1 = -extent + 2.0 * extent * (row + 1) / rows
        # Keep internal material boundaries at their exact pixel-aligned
        # coordinates while enclosing the entire camera filter footprint at
        # the four outer edges. This prevents a single edge sample from
        # turning a shader differential into a raster-coverage differential.
        if column == 0:
            x0 -= frame_bleed
        if column + 1 == columns:
            x1 += frame_bleed
        if row == 0:
            y0 -= frame_bleed
        if row + 1 == rows:
            y1 += frame_bleed
        first = len(vertices)
        vertices.extend(
            (
                (x0, y0, 0.0),
                (x1, y0, 0.0),
                (x1, y1, 0.0),
                (x0, y1, 0.0),
            )
        )
        face = (first, first + 1, first + 2, first + 3)
        if backfacing is not None and index in backfacing:
            face = tuple(reversed(face))
        faces.append(face)

    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(vertices, [], faces)
    for material in materials:
        mesh.materials.append(material)
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = index
    surface = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(surface)
    return surface


def _sphere(material: Any) -> Any:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64,
        ring_count=32,
        radius=0.82,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, 0.0),
    )
    sphere = bpy.context.object
    sphere.name = "Probe Surface"
    sphere.data.materials.append(material)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    return sphere


def _material(name: str) -> tuple[Any, Any, Any]:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.name = "Material Output"
    return material, tree, output


def _world(
    scene: Any,
    color: tuple[float, float, float, float],
    strength: float,
) -> tuple[Any, Any, Any]:
    world = bpy.data.worlds.new("Probe World")
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    background = tree.nodes.new("ShaderNodeBackground")
    background.name = "Background"
    _input(background, "Color").default_value = color
    _input(background, "Strength").default_value = strength
    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.name = "World Output"
    tree.links.new(
        _output(background, "Background"),
        _input(output, "Surface"),
    )
    scene.world = world
    return world, tree, background


def _linked_vector(
    tree: Any,
    name: str,
    value: tuple[float, float, float],
) -> Any:
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    node.name = name
    _input(node, "X").default_value = value[0]
    _input(node, "Y").default_value = value[1]
    _input(node, "Z").default_value = value[2]
    return _output(node, "Vector")
