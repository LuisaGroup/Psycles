"""Brick, white-noise, and particle-input procedural probe scenes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import _input, _material, _output, _plane


def _brick_texture(scene: Any) -> None:
    """Exercise both Brick Color and Fac with non-default row controls."""
    material, tree, output = _material("Brick Texture Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    brick = tree.nodes.new("ShaderNodeTexBrick")
    brick.name = "Brick Texture"
    brick.offset = 0.37
    brick.offset_frequency = 3
    brick.squash = 0.72
    brick.squash_frequency = 2
    _input(brick, "Color1").default_value = (0.78, 0.08, 0.03, 1.0)
    _input(brick, "Color2").default_value = (0.12, 0.43, 0.91, 1.0)
    _input(brick, "Mortar").default_value = (0.07, 0.21, 0.13, 1.0)
    _input(brick, "Scale").default_value = 5.3
    _input(brick, "Mortar Size").default_value = 0.036
    _input(brick, "Mortar Smooth").default_value = 0.017
    _input(brick, "Bias").default_value = -0.14
    _input(brick, "Brick Width").default_value = 0.53
    _input(brick, "Row Height").default_value = 0.19
    tree.links.new(
        _output(coordinates, "Generated"),
        _input(brick, "Vector"),
    )
    channels = tree.nodes.new("ShaderNodeSeparateColor")
    channels.mode = "RGB"
    tree.links.new(
        _output(brick, "Color"),
        _input(channels, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.mode = "RGB"
    tree.links.new(
        _output(channels, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(channels, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(brick, "Fac"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _brick_texture_constants(scene: Any) -> None:
    """Compare Brick numerics without reconstruction-filter edge noise."""
    material, tree, output = _material("Brick Texture Constant Probe")

    def brick_at(
        name: str, coordinate: tuple[float, float, float]
    ) -> Any:
        coordinate_node = tree.nodes.new("ShaderNodeRGB")
        coordinate_node.name = f"{name} Coordinates"
        _output(coordinate_node, "Color").default_value = (
            *coordinate,
            1.0,
        )
        brick = tree.nodes.new("ShaderNodeTexBrick")
        brick.name = name
        brick.offset = 0.37
        brick.offset_frequency = 3
        brick.squash = 0.72
        brick.squash_frequency = 2
        tree.links.new(
            _output(coordinate_node, "Color"),
            _input(brick, "Vector"),
        )
        _input(brick, "Color1").default_value = (
            0.78,
            0.08,
            0.03,
            1.0,
        )
        _input(brick, "Color2").default_value = (
            0.12,
            0.43,
            0.91,
            1.0,
        )
        _input(brick, "Mortar").default_value = (
            0.07,
            0.21,
            0.13,
            1.0,
        )
        _input(brick, "Scale").default_value = 5.3
        _input(brick, "Mortar Size").default_value = 0.036
        _input(brick, "Mortar Smooth").default_value = 0.017
        _input(brick, "Bias").default_value = -0.14
        _input(brick, "Brick Width").default_value = 0.53
        _input(brick, "Row Height").default_value = 0.19
        return brick

    brick_red = brick_at("Brick Red Sample", (0.113, 0.257, 0.0))
    brick_green = brick_at(
        "Brick Green Sample", (0.471, 0.379, 0.0)
    )
    brick_factor = brick_at(
        "Brick Mortar Sample", (0.811, 0.541, 0.0)
    )
    red_channels = tree.nodes.new("ShaderNodeSeparateColor")
    red_channels.name = "Separate Red Sample"
    red_channels.mode = "RGB"
    tree.links.new(
        _output(brick_red, "Color"),
        _input(red_channels, "Color"),
    )
    green_channels = tree.nodes.new("ShaderNodeSeparateColor")
    green_channels.name = "Separate Green Sample"
    green_channels.mode = "RGB"
    tree.links.new(
        _output(brick_green, "Color"),
        _input(green_channels, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Brick Samples"
    combine.mode = "RGB"
    tree.links.new(
        _output(red_channels, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(green_channels, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(brick_factor, "Fac"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _white_noise_dimensions(scene: Any) -> None:
    material, tree, output = _material(
        "White Noise Dimensions Probe"
    )
    values: list[Any] = []
    colors: list[dict[str, Any]] = []
    constants = (
        ((0.17, -2.3, 5.1), 0.37),
        ((-0.25, 1.75, 3.5), -4.25),
        ((2.125, -0.75, 0.03125), 8.5),
        ((-6.0, 0.625, 4.75), 1.125),
    )
    for dimensions, (vector, w) in enumerate(
        constants, start=1
    ):
        white = tree.nodes.new("ShaderNodeTexWhiteNoise")
        white.name = f"White Noise {dimensions}D"
        white.noise_dimensions = f"{dimensions}D"
        if dimensions != 1:
            _input(white, "Vector").default_value = vector
        if dimensions in (1, 4):
            _input(white, "W").default_value = w
        values.append(_output(white, "Value"))
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate {dimensions}D Color"
        separate.mode = "RGB"
        tree.links.new(
            _output(white, "Color"),
            _input(separate, "Color"),
        )
        colors.append(
            {
                channel: _output(separate, channel)
                for channel in ("Red", "Green", "Blue")
            }
        )

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Add {index}"
            add.operation = "ADD"
            tree.links.new(current, _input(add, "Value"))
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, _input(scale, "Value"))
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    red = average(
        "Red",
        [values[0], values[3], colors[1]["Red"]],
    )
    green = average(
        "Green",
        [values[1], colors[2]["Green"], colors[0]["Green"]],
    )
    blue = average(
        "Blue",
        [values[2], colors[3]["Blue"], colors[0]["Blue"]],
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Combine White Noise Coverage"
    combine.mode = "RGB"
    tree.links.new(red, _input(combine, "Red"))
    tree.links.new(green, _input(combine, "Green"))
    tree.links.new(blue, _input(combine, "Blue"))
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _particle_random_nonparticle(scene: Any) -> None:
    material, tree, output = _material("Particle Random Probe")
    particle = tree.nodes.new("ShaderNodeParticleInfo")
    particle.name = "Particle Info"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(particle, "Random"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _particle_random_instances(scene: Any) -> None:
    material, tree, output = _material("Particle Instance Random Probe")
    particle = tree.nodes.new("ShaderNodeParticleInfo")
    particle.name = "Particle Info"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(particle, "Random"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=2,
        radius=0.11,
        enter_editmode=False,
        location=(10.0, 10.0, 0.0),
    )
    instance = bpy.context.object
    instance.name = "Particle Instance"
    instance.data.materials.append(material)

    emitter_material, emitter_tree, emitter_output = _material(
        "Particle Emitter"
    )
    emitter_diffuse = emitter_tree.nodes.new("ShaderNodeBsdfDiffuse")
    _input(emitter_diffuse, "Color").default_value = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    emitter_tree.links.new(
        _output(emitter_diffuse, "BSDF"),
        _input(emitter_output, "Surface"),
    )
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=5,
        y_subdivisions=5,
        size=1.8,
        enter_editmode=False,
        location=(0.0, 0.0, -0.12),
    )
    emitter = bpy.context.object
    emitter.name = "Particle Emitter"
    emitter.data.materials.append(emitter_material)
    bpy.context.view_layer.objects.active = emitter
    emitter.select_set(True)
    bpy.ops.object.particle_system_add()
    system = emitter.particle_systems[-1]
    settings = system.settings
    settings.type = "HAIR"
    settings.count = 25
    settings.hair_length = 0.12
    settings.render_type = "OBJECT"
    settings.instance_object = instance
    settings.particle_size = 1.0
    settings.size_random = 0.0
    settings.emit_from = "VERT"
    settings.use_modifier_stack = True
