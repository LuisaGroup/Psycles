"""Camera, analytic-light, emission, and direct-light probe scenes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import _input, _material, _output, _plane


def _analytic_light_probe(
    scene: Any,
    light_type: str,
) -> Any:
    material, tree, output = _material(
        f"{light_type.title()} Light Probe Surface"
    )
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse"
    _input(diffuse, "Color").default_value = (
        0.62,
        0.41,
        0.23,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _plane(material)

    data = bpy.data.lights.new(
        f"{light_type.title()} Probe Light",
        type=light_type,
    )
    data.color = (0.36, 0.72, 1.0)
    data.use_shadow = True
    data.normalize = True
    light = bpy.data.objects.new(data.name, data)
    light.location = (0.37, -0.21, 1.4)
    scene.collection.objects.link(light)

    if light_type == "POINT":
        data.energy = 37.0
        data.shadow_soft_size = 0.0
    elif light_type == "SPOT":
        data.energy = 37.0
        data.shadow_soft_size = 0.0
        data.spot_size = 0.92
        data.spot_blend = 0.37
    elif light_type == "AREA":
        data.energy = 37.0
        data.shape = "RECTANGLE"
        data.size = 0.91
        data.size_y = 0.47
        data.spread = 3.141592653589793
    elif light_type == "SUN":
        data.energy = 1.7
        data.angle = 0.0
    else:
        raise ValueError(f"unsupported analytic light probe: {light_type}")

    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    return data


def _area_light(scene: Any) -> None:
    _analytic_light_probe(scene, "AREA")


def _area_light_ellipse(scene: Any) -> None:
    data = _analytic_light_probe(scene, "AREA")
    data.shape = "ELLIPSE"


def _area_light_spread(scene: Any) -> None:
    data = _analytic_light_probe(scene, "AREA")
    data.spread = 0.73


def _flat_light_distribution(scene: Any) -> None:
    """Exercise Cycles' mixed triangle/lamp flat CDF on one receiver."""
    world = scene.world
    if world is not None:
        world.cycles.sampling_method = "NONE"

    receiver, tree, output = _material(
        "Flat Distribution Receiver"
    )
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse"
    _input(diffuse, "Color").default_value = (
        0.52,
        0.31,
        0.17,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _plane(receiver)

    for index, (x, size, color, strength) in enumerate(
        (
            (-2.2, 0.5, (1.0, 0.18, 0.06, 1.0), 8.0),
            (2.2, 1.0, (0.08, 0.22, 1.0, 1.0), 3.0),
        )
    ):
        material, emitter_tree, emitter_output = _material(
            f"Flat Distribution Emitter {index}"
        )
        material.cycles.emission_sampling = "FRONT"
        emission = emitter_tree.nodes.new("ShaderNodeEmission")
        emission.name = "Emission"
        _input(emission, "Color").default_value = color
        _input(emission, "Strength").default_value = strength
        emitter_tree.links.new(
            _output(emission, "Emission"),
            _input(emitter_output, "Surface"),
        )
        bpy.ops.mesh.primitive_plane_add(
            size=size,
            enter_editmode=False,
            align="WORLD",
            location=(x, 0.0, 3.0),
            rotation=(3.141592653589793, 0.0, 0.0),
        )
        emitter = bpy.context.object
        emitter.name = f"Flat Distribution Emitter {index}"
        emitter.data.materials.append(material)

    data = bpy.data.lights.new(
        "Flat Distribution Point", type="POINT"
    )
    data.color = (0.19, 1.0, 0.27)
    data.energy = 24.0
    data.shadow_soft_size = 0.0
    light = bpy.data.objects.new(data.name, data)
    light.location = (0.0, 1.7, 2.4)
    scene.collection.objects.link(light)

    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0


def _triangle_light_solid_angle(scene: Any) -> None:
    """Exercise Cycles' near-triangle solid-angle sampling branch."""
    world = scene.world
    if world is not None:
        world.cycles.sampling_method = "NONE"

    receiver, tree, output = _material(
        "Solid Angle Triangle Receiver"
    )
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse"
    _input(diffuse, "Color").default_value = (
        0.37,
        0.58,
        0.21,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _plane(receiver)

    emitter_material, emitter_tree, emitter_output = _material(
        "Solid Angle Triangle Emitter"
    )
    emitter_material.cycles.emission_sampling = "FRONT"
    emission = emitter_tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (
        0.83,
        0.29,
        0.11,
        1.0,
    )
    _input(emission, "Strength").default_value = 7.0
    emitter_tree.links.new(
        _output(emission, "Emission"),
        _input(emitter_output, "Surface"),
    )
    # This vertical triangle remains outside the orthographic camera frame,
    # but it subtends a large solid angle at every visible receiver point.
    # The winding gives it a -Y front face, toward the receiver.
    mesh = bpy.data.meshes.new("Solid Angle Triangle Mesh")
    mesh.from_pydata(
        (
            (-3.0, 1.4, 0.2),
            (3.0, 1.4, 0.2),
            (0.0, 1.4, 3.0),
        ),
        (),
        ((0, 1, 2),),
    )
    emitter = bpy.data.objects.new(
        "Solid Angle Triangle Emitter", mesh
    )
    emitter.data.materials.append(emitter_material)
    scene.collection.objects.link(emitter)

    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0


def _camera_emission_silhouettes(
    scene: Any,
    name: str,
    row_depths: tuple[tuple[float, float], ...],
) -> None:
    material, tree, output = _material(f"{name} Emitter")
    material.cycles.emission_sampling = "NONE"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (
        0.71,
        0.29,
        0.13,
        1.0,
    )
    _input(emission, "Strength").default_value = 1.0
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for row, (y, z) in enumerate(row_depths):
        for column in range(15):
            center_x = -0.84 + 0.12 * column
            half_width = 0.010 + 0.002 * ((row + column) % 3)
            half_height = 0.07 + 0.01 * (column % 2)
            first = len(vertices)
            vertices.extend(
                (
                    (center_x - half_width, y - half_height, z),
                    (center_x + half_width, y - half_height, z),
                    (center_x + half_width, y + half_height, z),
                    (center_x - half_width, y + half_height, z),
                )
            )
            faces.append(
                (first, first + 1, first + 2, first + 3)
            )
    mesh = bpy.data.meshes.new(f"{name} Silhouettes Mesh")
    mesh.from_pydata(vertices, (), faces)
    silhouettes = bpy.data.objects.new(
        f"{name} Silhouettes", mesh
    )
    silhouettes.data.materials.append(material)
    scene.collection.objects.link(silhouettes)

    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0


def _camera_dof_disk(scene: Any) -> None:
    """Exercise the Sobol-preserving concentric disk-aperture map."""
    # Camera coverage is a sample-correspondence test, not a convergence
    # comparison. Cycles 5.3 resolves AUTOMATIC to BLUE_NOISE_PURE for final
    # renders, while Lone Monk explicitly uses TABULATED_SOBOL. Pin the probe
    # to the latter so Cycles and Psycles consume the same (time, lens_x,
    # lens_y) triples.
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    camera = scene.camera
    camera.data.type = "PERSP"
    camera.data.lens = 50.0
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = 3.0
    camera.data.dof.aperture_fstop = 0.7
    camera.data.dof.aperture_blades = 0
    camera.data.dof.aperture_ratio = 1.0
    # Cycles' BOX filter has a fixed effective width of one pixel; the UI
    # filter_width property is dormant for this filter type.
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    _camera_emission_silhouettes(
        scene,
        "Disk DOF",
        ((0.36, 1.0), (0.0, 0.0), (-0.36, -2.0)),
    )


def _camera_blackman_harris_filter(scene: Any) -> None:
    """Exercise the Cycles Blackman-Harris inverse-CDF film table."""
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    camera = scene.camera
    camera.data.type = "PERSP"
    camera.data.lens = 50.0
    camera.data.dof.use_dof = False
    scene.cycles.pixel_filter_type = "BLACKMAN_HARRIS"
    scene.cycles.filter_width = 1.5
    _camera_emission_silhouettes(
        scene,
        "Blackman-Harris Film",
        ((0.36, 0.0), (0.0, 0.0), (-0.36, 0.0)),
    )


def _point_light(scene: Any) -> None:
    _analytic_light_probe(scene, "POINT")


def _point_light_nodes(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.use_nodes = True
    emission = data.node_tree.nodes.get("Emission")
    if emission is None:
        raise RuntimeError("point light has no default Emission node")
    _input(emission, "Color").default_value = (
        0.77,
        0.31,
        0.58,
        1.0,
    )
    _input(emission, "Strength").default_value = 0.63


def _point_light_light_path(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.shadow_soft_size = 0.19
    data.use_soft_falloff = False
    data.use_nodes = True
    tree = data.node_tree
    emission = tree.nodes.get("Emission")
    if emission is None:
        raise RuntimeError("point light has no default Emission node")

    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.name = "Light Path Contract"

    weighted_outputs = (
        ("Is Camera Ray", 8.0),
        ("Is Diffuse Ray", 4.0),
        ("Is Shadow Ray", 2.0),
    )
    accumulated = None
    for socket_name, weight in weighted_outputs:
        multiply = tree.nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        multiply.inputs[1].default_value = weight
        tree.links.new(
            _output(light_path, socket_name),
            multiply.inputs[0],
        )
        if accumulated is None:
            accumulated = _output(multiply, "Value")
            continue
        add = tree.nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        tree.links.new(accumulated, add.inputs[0])
        tree.links.new(
            _output(multiply, "Value"),
            add.inputs[1],
        )
        accumulated = _output(add, "Value")

    add_depth = tree.nodes.new("ShaderNodeMath")
    add_depth.operation = "ADD"
    tree.links.new(accumulated, add_depth.inputs[0])
    tree.links.new(
        _output(light_path, "Ray Depth"),
        add_depth.inputs[1],
    )
    add_bias = tree.nodes.new("ShaderNodeMath")
    add_bias.operation = "ADD"
    add_bias.inputs[1].default_value = 1.0
    tree.links.new(
        _output(add_depth, "Value"),
        add_bias.inputs[0],
    )
    tree.links.new(
        _output(add_bias, "Value"),
        _input(emission, "Strength"),
    )


def _transmission_light_path_visibility(scene: Any) -> None:
    """Keep shader path visibility distinct from traversal visibility.

    Cycles classifies a singular transmission as both TRANSMIT and GLOSSY.
    Its BVH projection removes GLOSSY, but the shader behind the refractive
    plane must still observe Is Glossy Ray.  A full-frame, normal-incidence
    refraction makes this a deterministic one-path state transition rather
    than a noisy transport comparison.
    """
    refractive, tree, output = _material(
        "Transmission Visibility Boundary"
    )
    refraction = tree.nodes.new("ShaderNodeBsdfRefraction")
    refraction.name = "Singular Refraction"
    _input(refraction, "Color").default_value = (1.0, 1.0, 1.0, 1.0)
    _input(refraction, "Roughness").default_value = 0.0
    _input(refraction, "IOR").default_value = 1.45
    tree.links.new(
        _output(refraction, "BSDF"),
        _input(output, "Surface"),
    )
    bpy.ops.mesh.primitive_plane_add(
        size=8.0,
        enter_editmode=False,
        align="WORLD",
        location=(0.0, 0.0, 1.0),
    )
    boundary = bpy.context.object
    boundary.name = "Transmission Visibility Boundary"
    boundary.data.materials.append(refractive)

    emitter, tree, output = _material(
        "Transmission Visibility Observer"
    )
    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.name = "Uncontracted Path Visibility"
    scale = tree.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 0.75
    tree.links.new(
        _output(light_path, "Is Glossy Ray"),
        scale.inputs[0],
    )
    bias = tree.nodes.new("ShaderNodeMath")
    bias.operation = "ADD"
    bias.inputs[1].default_value = 0.25
    tree.links.new(_output(scale, "Value"), bias.inputs[0])
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Visibility-coded Emission"
    _input(emission, "Color").default_value = (0.71, 0.37, 0.19, 1.0)
    tree.links.new(
        _output(bias, "Value"),
        _input(emission, "Strength"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(emitter)

    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2


def _shadow_depth_transparency_material(
    name: str,
    base: float,
    slope: float,
) -> Any:
    material, tree, output = _material(name)
    light_path = tree.nodes.new("ShaderNodeLightPath")
    light_path.name = "Ordered Shadow Depth"
    multiply = tree.nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = slope
    tree.links.new(
        _output(light_path, "Transparent Depth"),
        multiply.inputs[0],
    )
    add = tree.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    add.inputs[1].default_value = base
    tree.links.new(
        _output(multiply, "Value"),
        add.inputs[0],
    )
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.name = "Depth-dependent Transparent BSDF"
    tree.links.new(
        _output(add, "Value"),
        _input(transparent, "Color"),
    )
    tree.links.new(
        _output(transparent, "BSDF"),
        _input(output, "Surface"),
    )
    return material


def _vertical_shadow_layer(
    scene: Any,
    name: str,
    x: float,
    material: Any,
) -> None:
    # Vertical YZ quads are invisible to the orthographic -Z camera but
    # intersect receiver-to-light shadow rays. This isolates transparent
    # shadow ordering from camera-path transparent bounces.
    mesh = bpy.data.meshes.new(f"{name} Mesh")
    mesh.from_pydata(
        (
            (x, -4.0, 0.01),
            (x, 4.0, 0.01),
            (x, 4.0, 1.39),
            (x, -4.0, 1.39),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    layer = bpy.data.objects.new(name, mesh)
    layer.data.materials.append(material)
    scene.collection.objects.link(layer)


def _point_light_shadow_light_path(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.shadow_soft_size = 0.0
    data.use_soft_falloff = False

    # Create the far layer first so dependency-graph/acceleration insertion
    # order is the opposite of required ray-distance shading order.
    far = _shadow_depth_transparency_material(
        "Far Ordered Shadow Layer",
        0.85,
        -0.25,
    )
    near = _shadow_depth_transparency_material(
        "Near Ordered Shadow Layer",
        0.20,
        0.35,
    )
    _vertical_shadow_layer(
        scene,
        "Far Ordered Shadow Layer",
        0.24,
        far,
    )
    _vertical_shadow_layer(
        scene,
        "Near Ordered Shadow Layer",
        0.08,
        near,
    )
    scene.cycles.transparent_max_bounces = 8


def _point_light_shadow_limit(scene: Any) -> None:
    _point_light_shadow_light_path(scene)
    # The nearer layer consumes the only allowed transparent intersection;
    # Cycles therefore treats the farther layer as opaque.
    scene.cycles.transparent_max_bounces = 1


def _point_light_soft_disk(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.shadow_soft_size = 0.19
    data.use_soft_falloff = True


def _point_light_soft_sphere(scene: Any) -> None:
    data = _analytic_light_probe(scene, "POINT")
    data.shadow_soft_size = 0.19
    data.use_soft_falloff = False


def _spot_light(scene: Any) -> None:
    _analytic_light_probe(scene, "SPOT")


def _spot_light_soft(scene: Any) -> None:
    data = _analytic_light_probe(scene, "SPOT")
    data.shadow_soft_size = 0.13
    data.use_soft_falloff = False


def _sun_light(scene: Any) -> None:
    _analytic_light_probe(scene, "SUN")


def _sun_light_disk(scene: Any) -> None:
    data = _analytic_light_probe(scene, "SUN")
    data.angle = 0.17


def _rgb_emission(scene: Any) -> None:
    material, tree, output = _material("RGB Probe")
    rgb = tree.nodes.new("ShaderNodeRGB")
    rgb.name = "RGB"
    _output(rgb, "Color").default_value = (0.12, 0.37, 0.83, 1.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Strength").default_value = 1.7
    tree.links.new(_output(rgb, "Color"), _input(emission, "Color"))
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _value_emission(scene: Any) -> None:
    material, tree, output = _material("Value Probe")
    value = tree.nodes.new("ShaderNodeValue")
    value.name = "Value"
    _output(value, "Value").default_value = 2.25
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (0.18, 0.61, 0.29, 1.0)
    tree.links.new(
        _output(value, "Value"), _input(emission, "Strength")
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _emission_surface(scene: Any) -> None:
    material, tree, output = _material("Emission Probe")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (0.73, 0.21, 0.08, 1.0)
    _input(emission, "Strength").default_value = 3.5
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _integrator_clamp_direct(scene: Any) -> None:
    """Exercise Blender UI clamp to Cycles device-clamp conversion."""
    material, tree, output = _material("Direct Clamp Probe")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (10.0, 10.0, 10.0, 1.0)
    _input(emission, "Strength").default_value = 1.0
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    scene.cycles.sample_clamp_direct = 2.0
    scene.cycles.sample_clamp_indirect = 0.0
    _plane(material)
