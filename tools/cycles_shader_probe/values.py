"""Color, scalar, vector, mix, group, and channel-operation probes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import (
    _input,
    _input_identifier,
    _linked_vector,
    _material,
    _material_matrix,
    _output,
    _output_identifier,
    _plane,
    _world,
)


def _background_world(scene: Any) -> None:
    _world(scene, (0.16, 0.48, 0.77, 1.0), 2.3)


def _ambient_occlusion_matrix(scene: Any) -> None:
    configurations = (
        ("Explicit Distance", 0.5, False, False, None, None),
        (
            "Only Local Color",
            0.5,
            False,
            True,
            None,
            (0.23, 0.51, 0.79, 1.0),
        ),
        ("Global Radius", 0.0, False, False, None, None),
        ("Inside", 0.5, True, False, None, None),
        ("Linked Side Normal", 0.5, False, False, (1.0, 0.0, 0.0), None),
        (
            "Explicit Distance Color",
            0.5,
            False,
            False,
            None,
            (0.81, 0.37, 0.19, 1.0),
        ),
    )
    materials = []
    for name, distance, inside, only_local, normal, color in configurations:
        material, tree, output = _material(f"AO {name}")
        ambient_occlusion = tree.nodes.new("ShaderNodeAmbientOcclusion")
        ambient_occlusion.name = name
        ambient_occlusion.samples = 16
        ambient_occlusion.inside = inside
        ambient_occlusion.only_local = only_local
        _input(ambient_occlusion, "Distance").default_value = distance
        if color is not None:
            _input(ambient_occlusion, "Color").default_value = color
        if normal is not None:
            tree.links.new(
                _linked_vector(tree, f"{name} Normal", normal),
                _input(ambient_occlusion, "Normal"),
            )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"{name} Emission"
        if color is None:
            _input(emission, "Color").default_value = (1.0, 1.0, 1.0, 1.0)
            tree.links.new(
                _output(ambient_occlusion, "AO"),
                _input(emission, "Strength"),
            )
        else:
            _input(emission, "Strength").default_value = 1.0
            tree.links.new(
                _output(ambient_occlusion, "Color"),
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
        columns=3,
        rows=2,
        name="Ambient Occlusion Matrix",
        frame_bleed=0.1,
    )

    # One camera-invisible plane covers the complete matrix at z=0.2. Global
    # AO sees it through the ordinary shadow-visible TLAS; Only Local must
    # ignore it because it is a distinct object. Global Radius uses the World
    # distance below (0.1), which is deliberately too short to reach it.
    blocker_mesh = bpy.data.meshes.new("AO Blocker Mesh")
    blocker_mesh.from_pydata(
        [(-4.0, -4.0, 0.2),
         (4.0, -4.0, 0.2),
         (4.0, 4.0, 0.2),
         (-4.0, 4.0, 0.2)],
        [],
        [(0, 1, 2, 3)],
    )
    blocker = bpy.data.objects.new("AO Blocker", blocker_mesh)
    scene.collection.objects.link(blocker)
    blocker.visible_camera = False
    blocker.visible_shadow = True
    if scene.world is None:
        raise RuntimeError("Ambient Occlusion probe requires a World")
    scene.world.light_settings.distance = 0.1


def _node_group_color(scene: Any) -> None:
    material, tree, output = _material("Node Group Probe")

    group = bpy.data.node_groups.new(
        "Generic Color Transform", "ShaderNodeTree"
    )
    group.interface.new_socket(
        name="Color",
        in_out="INPUT",
        socket_type="NodeSocketColor",
    )
    group.interface.new_socket(
        name="Color",
        in_out="OUTPUT",
        socket_type="NodeSocketColor",
    )
    group_input = group.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"
    group_output = group.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    invert = group.nodes.new("ShaderNodeInvert")
    invert.name = "Invert"
    _input(invert, "Fac").default_value = 0.25
    group.links.new(
        _output(group_input, "Color"),
        _input(invert, "Color"),
    )
    group.links.new(
        _output(invert, "Color"),
        _input(group_output, "Color"),
    )

    outer = bpy.data.node_groups.new(
        "Nested Color Wrapper", "ShaderNodeTree"
    )
    outer.interface.new_socket(
        name="Color",
        in_out="INPUT",
        socket_type="NodeSocketColor",
    )
    outer.interface.new_socket(
        name="Color",
        in_out="OUTPUT",
        socket_type="NodeSocketColor",
    )
    outer_input = outer.nodes.new("NodeGroupInput")
    outer_input.name = "Group Input"
    outer_output = outer.nodes.new("NodeGroupOutput")
    outer_output.name = "Group Output"
    nested = outer.nodes.new("ShaderNodeGroup")
    nested.name = "Nested Arbitrary Instance"
    nested.node_tree = group
    outer.links.new(
        _output(outer_input, "Color"),
        _input(nested, "Color"),
    )
    outer.links.new(
        _output(nested, "Color"),
        _input(outer_output, "Color"),
    )

    source = tree.nodes.new("ShaderNodeRGB")
    source.name = "Group Source"
    _output(source, "Color").default_value = (
        0.12,
        0.47,
        0.81,
        1.0,
    )
    instance = tree.nodes.new("ShaderNodeGroup")
    instance.name = "Arbitrarily Named Group Instance"
    instance.node_tree = outer
    tree.links.new(
        _output(source, "Color"),
        _input(instance, "Color"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Strength").default_value = 2.0
    tree.links.new(
        _output(instance, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _rgb_to_bw(scene: Any) -> None:
    material, tree, output = _material("RGB to BW")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Luminance Weights"
    combine.mode = "RGB"
    for color, target in (
        ((1.0, 0.0, 0.0, 1.0), "Red"),
        ((0.0, 1.0, 0.0, 1.0), "Green"),
        ((0.0, 0.0, 1.0, 1.0), "Blue"),
    ):
        convert = tree.nodes.new("ShaderNodeRGBToBW")
        convert.name = "RGB to BW"
        _input(convert, "Color").default_value = color
        tree.links.new(
            _output(convert, "Val"),
            _input(combine, target),
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


def _gamma_color(scene: Any) -> None:
    material, tree, output = _material("Gamma")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Gamma Branches"
    combine.mode = "RGB"
    for index, (color, gamma, source, target) in enumerate(
        (
            ((0.2, 0.5, 0.9, 1.0), 0.0, "Red", "Red"),
            ((0.18, 0.5, 0.87, 1.0), 2.2, "Green", "Green"),
            ((0.0, 0.25, 0.25, 1.0), -0.5, "Blue", "Blue"),
        )
    ):
        node = tree.nodes.new("ShaderNodeGamma")
        node.name = f"Gamma Branch {index}"
        _input(node, "Color").default_value = color
        _input(node, "Gamma").default_value = gamma
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select Gamma Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(node, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, source),
            _input(combine, target),
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


def _brightness_contrast(scene: Any) -> None:
    material, tree, output = _material("Brightness Contrast")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Brightness/Contrast Branches"
    combine.mode = "RGB"
    for index, (color, bright, contrast, source, target) in enumerate(
        (
            ((0.14, 0.52, 0.88, 1.0), 0.0, 0.0, "Red", "Red"),
            (
                (0.09, 0.44, 0.81, 1.0),
                0.17,
                -0.35,
                "Green",
                "Green",
            ),
            (
                (0.08, 0.37, 0.08, 1.0),
                -0.22,
                0.48,
                "Blue",
                "Blue",
            ),
        )
    ):
        node = tree.nodes.new("ShaderNodeBrightContrast")
        node.name = f"Brightness/Contrast Branch {index}"
        _input(node, "Color").default_value = color
        _input(node, "Bright").default_value = bright
        _input(node, "Contrast").default_value = contrast
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select Brightness Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(node, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, source),
            _input(combine, target),
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


def _hue_saturation_value(scene: Any) -> None:
    """Cover Cycles HSV adjustment, hue wrapping, and factor blending."""
    material, tree, output = _material("Hue Saturation Value")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack HSV Branches"
    combine.mode = "RGB"
    branches = (
        (
            (0.12, 0.65, 0.9, 1.0),
            0.3,
            1.4,
            0.75,
            1.0,
            "Red",
        ),
        (
            (0.83, 0.2, 0.06, 1.0),
            0.82,
            0.25,
            1.7,
            0.37,
            "Green",
        ),
        (
            (0.1, 0.4, 0.7, 1.0),
            0.05,
            2.2,
            0.4,
            0.85,
            "Blue",
        ),
    )
    for index, (
        color,
        hue,
        saturation,
        value,
        factor,
        channel,
    ) in enumerate(branches):
        node = tree.nodes.new("ShaderNodeHueSaturation")
        node.name = f"HSV Branch {index}"
        _input(node, "Color").default_value = color
        _input(node, "Hue").default_value = hue
        _input(node, "Saturation").default_value = saturation
        _input(node, "Value").default_value = value
        _input(node, "Fac").default_value = factor
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select HSV Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(node, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, channel),
            _input(combine, channel),
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


def _clamp(scene: Any) -> None:
    material, tree, output = _material("Clamp")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Clamp Branches"
    combine.mode = "RGB"
    for index, (mode, value, minimum, maximum, target) in enumerate(
        (
            ("MINMAX", 0.5, 0.8, 0.2, "Red"),
            ("RANGE", 0.5, 0.8, 0.2, "Green"),
            ("MINMAX", 2.0, -0.2, 0.7, "Blue"),
        )
    ):
        node = tree.nodes.new("ShaderNodeClamp")
        node.name = f"Clamp Branch {index}"
        node.clamp_type = mode
        _input(node, "Value").default_value = value
        _input(node, "Min").default_value = minimum
        _input(node, "Max").default_value = maximum
        tree.links.new(
            _output(node, "Result"),
            _input(combine, target),
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


def _math_operations(scene: Any) -> None:
    """Exercise every Blender 4.5.10 ShaderNodeMath operation."""
    material, tree, output = _material("Math Operations Probe")
    cases = (
        ("ADD", 0.12, 0.23, 0.5),
        ("SUBTRACT", 0.72, 0.19, 0.5),
        ("MULTIPLY", 0.45, 0.55, 0.5),
        ("DIVIDE", 0.24, 0.8, 0.5),
        ("MULTIPLY_ADD", 0.3, 0.5, 0.2),
        ("POWER", 0.64, 0.5, 0.5),
        ("LOGARITHM", 0.5, 0.25, 0.5),
        ("SQRT", 0.36, 0.5, 0.5),
        ("INVERSE_SQRT", 4.0, 0.5, 0.5),
        ("ABSOLUTE", -0.47, 0.5, 0.5),
        ("EXPONENT", -0.7, 0.5, 0.5),
        ("MINIMUM", 0.33, 0.71, 0.5),
        ("MAXIMUM", 0.33, 0.71, 0.5),
        ("LESS_THAN", 0.2, 0.3, 0.5),
        ("GREATER_THAN", 0.8, 0.3, 0.5),
        ("SIGN", 0.4, 0.5, 0.5),
        ("COMPARE", 0.4, 0.42, 0.03),
        ("SMOOTH_MIN", 0.4, 0.6, 0.3),
        ("SMOOTH_MAX", 0.4, 0.6, 0.3),
        ("ROUND", 0.51, 0.5, 0.5),
        ("FLOOR", 1.8, 0.5, 0.5),
        ("CEIL", 0.2, 0.5, 0.5),
        ("TRUNC", 1.8, 0.5, 0.5),
        ("FRACT", 1.37, 0.5, 0.5),
        ("MODULO", 1.3, 0.8, 0.5),
        ("FLOORED_MODULO", -0.3, 0.8, 0.5),
        ("WRAP", 1.4, 1.0, 0.2),
        ("SNAP", 0.74, 0.2, 0.5),
        ("PINGPONG", 0.73, 0.6, 0.5),
        ("SINE", 0.5, 0.5, 0.5),
        ("COSINE", 0.7, 0.5, 0.5),
        ("TANGENT", 0.3, 0.5, 0.5),
        ("ARCSINE", 0.5, 0.5, 0.5),
        ("ARCCOSINE", 0.8, 0.5, 0.5),
        ("ARCTANGENT", 0.6, 0.5, 0.5),
        ("ARCTAN2", 0.4, 0.8, 0.5),
        ("SINH", 0.5, 0.5, 0.5),
        ("COSH", 0.0, 0.5, 0.5),
        ("TANH", 0.7, 0.5, 0.5),
        ("RADIANS", 45.0, 0.5, 0.5),
        ("DEGREES", 0.01, 0.5, 0.5),
    )
    channels: list[list[Any]] = [[], [], []]
    for index, (operation, a, b, c) in enumerate(cases):
        math = tree.nodes.new("ShaderNodeMath")
        math.name = f"Math {index:02d} {operation}"
        math.operation = operation
        math.use_clamp = False
        math.inputs[0].default_value = a
        math.inputs[1].default_value = b
        math.inputs[2].default_value = c
        channels[index % 3].append(_output(math, "Value"))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Math Operations"
    combine.mode = "RGB"
    for name, sockets in zip(
        ("Red", "Green", "Blue"),
        channels,
        strict=True,
    ):
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
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


def _math_edge_cases(scene: Any) -> None:
    """Cover Cycles guards, signed behavior, and Math output clamping."""
    material, tree, output = _material("Math Edge Cases Probe")
    cases = (
        ("DIVIDE", 0.7, 0.0, 0.5, False),
        ("POWER", -0.5, 0.5, 0.5, False),
        ("POWER", -0.5, 3.0, 0.5, False),
        ("POWER", 0.0, 0.0, 0.5, False),
        ("LOGARITHM", -2.0, 10.0, 0.5, False),
        ("LOGARITHM", 0.5, 1.0, 0.5, False),
        ("SQRT", -4.0, 0.5, 0.5, False),
        ("INVERSE_SQRT", 0.0, 0.5, 0.5, False),
        ("ARCSINE", 2.0, 0.5, 0.5, False),
        ("ARCCOSINE", -2.0, 0.5, 0.5, False),
        ("ARCTAN2", 0.0, 0.0, 0.5, False),
        ("SIGN", 0.0, 0.5, 0.5, False),
        ("MODULO", -1.3, 0.8, 0.5, False),
        ("FLOORED_MODULO", -1.3, 0.8, 0.5, False),
        ("MODULO", 0.7, 0.0, 0.5, False),
        ("WRAP", 0.7, 0.4, 0.4, False),
        ("WRAP", 0.7, 0.2, 0.8, False),
        ("SNAP", 0.7, 0.0, 0.5, False),
        ("PINGPONG", 0.7, 0.0, 0.5, False),
        ("PINGPONG", -0.7, 0.6, 0.5, False),
        ("SMOOTH_MIN", 0.4, 0.6, 0.0, False),
        ("SMOOTH_MAX", 0.4, 0.6, 0.0, False),
        ("COMPARE", 1.0, 1.0000001192092896, -1.0, False),
        ("COMPARE", 1.0, 1.000000238418579, 0.0, False),
        ("ROUND", -1.5, 0.5, 0.5, False),
        ("TRUNC", -1.8, 0.5, 0.5, False),
        ("FRACT", -1.3, 0.5, 0.5, False),
        ("ADD", 0.8, 0.7, 0.5, True),
        ("SUBTRACT", 0.2, 0.7, 0.5, True),
    )
    channels: list[list[Any]] = [[], [], []]
    for index, (operation, a, b, c, use_clamp) in enumerate(cases):
        math = tree.nodes.new("ShaderNodeMath")
        math.name = f"Math Edge {index:02d} {operation}"
        math.operation = operation
        math.use_clamp = use_clamp
        math.inputs[0].default_value = a
        math.inputs[1].default_value = b
        math.inputs[2].default_value = c
        channels[index % 3].append(_output(math, "Value"))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Edge Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Edge Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Math Edge Cases"
    combine.mode = "RGB"
    for name, sockets in zip(
        ("Red", "Green", "Blue"),
        channels,
        strict=True,
    ):
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
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


def _mix_color_modes(scene: Any) -> None:
    """Cover every Cycles color blend mode of the modern Mix node."""
    material, tree, output = _material("Mix Color Modes Probe")
    modes = (
        "MIX",
        "DARKEN",
        "MULTIPLY",
        "BURN",
        "LIGHTEN",
        "SCREEN",
        "DODGE",
        "ADD",
        "OVERLAY",
        "SOFT_LIGHT",
        "LINEAR_LIGHT",
        "DIFFERENCE",
        "EXCLUSION",
        "SUBTRACT",
        "DIVIDE",
        "HUE",
        "SATURATION",
        "COLOR",
        "VALUE",
    )
    channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, mode in enumerate(modes):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Color {index:02d} {mode}"
        mix.data_type = "RGBA"
        mix.blend_type = mode
        mix.clamp_factor = True
        mix.clamp_result = False
        _input_identifier(
            mix, "Factor_Float"
        ).default_value = 0.37
        _input_identifier(mix, "A_Color").default_value = (
            0.17,
            0.63,
            0.89,
            1.0,
        )
        _input_identifier(mix, "B_Color").default_value = (
            0.82,
            0.24,
            0.51,
            1.0,
        )
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate {mode}"
        separate.mode = "RGB"
        tree.links.new(
            _output_identifier(mix, "Result_Color"),
            _input(separate, "Color"),
        )
        for channel in channels:
            channels[channel].append(_output(separate, channel))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Mix Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Mix Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Mix Color Modes"
    combine.mode = "RGB"
    for name, sockets in channels.items():
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
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


def _mix_data_types(scene: Any) -> None:
    """Cover float, uniform-vector, and non-uniform-vector Mix paths."""
    material, tree, output = _material("Mix Data Types Probe")

    float_outputs: list[Any] = []
    for index, (factor, clamp_factor, a, b) in enumerate(
        (
            (1.4, True, 0.2, 0.8),
            (1.4, False, 0.2, 0.8),
            (-0.3, True, 0.2, 0.8),
        )
    ):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Float {index}"
        mix.data_type = "FLOAT"
        mix.clamp_factor = clamp_factor
        _input_identifier(mix, "Factor_Float").default_value = factor
        _input_identifier(mix, "A_Float").default_value = a
        _input_identifier(mix, "B_Float").default_value = b
        float_outputs.append(
            _output_identifier(mix, "Result_Float")
        )

    vector_outputs: list[Any] = []
    vector_cases = (
        ("UNIFORM", 1.3, (0.5, 0.5, 0.5), True),
        ("UNIFORM", -0.25, (0.5, 0.5, 0.5), False),
        ("NON_UNIFORM", 0.5, (-0.2, 0.5, 1.4), True),
        ("NON_UNIFORM", 0.5, (-0.2, 0.5, 1.4), False),
    )
    for index, (factor_mode, scalar_factor, vector_factor, clamp) in enumerate(
        vector_cases
    ):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Vector {index} {factor_mode}"
        mix.data_type = "VECTOR"
        mix.factor_mode = factor_mode
        mix.clamp_factor = clamp
        _input_identifier(
            mix, "Factor_Float"
        ).default_value = scalar_factor
        _input_identifier(
            mix, "Factor_Vector"
        ).default_value = vector_factor
        _input_identifier(mix, "A_Vector").default_value = (
            0.1,
            0.7,
            -0.2,
        )
        _input_identifier(mix, "B_Vector").default_value = (
            0.9,
            -0.1,
            0.6,
        )
        vector_outputs.append(
            _output_identifier(mix, "Result_Vector")
        )

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Data Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Data Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    vector_channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, vector_output in enumerate(vector_outputs):
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Mix Vector {index}"
        separate.mode = "RGB"
        tree.links.new(vector_output, _input(separate, "Color"))
        for channel, component in zip(
            vector_channels,
            ("Red", "Green", "Blue"),
            strict=True,
        ):
            vector_channels[channel].append(
                _output(separate, component)
            )

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Mix Data Types"
    combine.mode = "RGB"
    tree.links.new(
        average("Float", float_outputs),
        _input(combine, "Red"),
    )
    tree.links.new(
        average("Vector Y", vector_channels["Green"]),
        _input(combine, "Green"),
    )
    tree.links.new(
        average("Vector Z", vector_channels["Blue"]),
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


def _mix_color_edge_cases(scene: Any) -> None:
    """Cover guarded branches and factor/result clamping of color Mix."""
    material, tree, output = _material("Mix Color Edge Cases Probe")
    cases = (
        (
            "DIVIDE",
            0.7,
            (0.2, 0.7, 0.9, 1.0),
            (0.0, 0.2, 0.0, 1.0),
            True,
            False,
        ),
        (
            "DODGE",
            0.8,
            (0.0, 0.5, 0.9, 1.0),
            (0.3, 3.0, 0.9, 1.0),
            True,
            False,
        ),
        (
            "BURN",
            1.0,
            (0.5, 0.1, 0.9, 1.0),
            (0.0, 0.2, 2.0, 1.0),
            True,
            False,
        ),
        (
            "HUE",
            0.6,
            (0.2, 0.7, 0.4, 1.0),
            (0.5, 0.5, 0.5, 1.0),
            True,
            False,
        ),
        (
            "SATURATION",
            0.6,
            (0.5, 0.5, 0.5, 1.0),
            (0.1, 0.8, 0.3, 1.0),
            True,
            False,
        ),
        (
            "COLOR",
            0.6,
            (0.2, 0.7, 0.4, 1.0),
            (0.5, 0.5, 0.5, 1.0),
            True,
            False,
        ),
        (
            "MIX",
            1.4,
            (0.2, 0.3, 0.4, 1.0),
            (0.8, 0.7, 0.6, 1.0),
            False,
            False,
        ),
        (
            "ADD",
            1.4,
            (0.7, 0.8, 0.9, 1.0),
            (0.8, 0.7, 0.6, 1.0),
            True,
            False,
        ),
        (
            "LINEAR_LIGHT",
            0.9,
            (0.1, 0.9, 0.5, 1.0),
            (0.0, 1.0, 0.8, 1.0),
            True,
            True,
        ),
        (
            "SUBTRACT",
            1.0,
            (0.1, 0.7, 0.3, 1.0),
            (0.8, 0.2, 0.9, 1.0),
            True,
            True,
        ),
    )
    channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, (
        mode,
        factor,
        color_a,
        color_b,
        clamp_factor,
        clamp_result,
    ) in enumerate(cases):
        mix = tree.nodes.new("ShaderNodeMix")
        mix.name = f"Mix Edge {index:02d} {mode}"
        mix.data_type = "RGBA"
        mix.blend_type = mode
        mix.clamp_factor = clamp_factor
        mix.clamp_result = clamp_result
        _input_identifier(
            mix, "Factor_Float"
        ).default_value = factor
        _input_identifier(mix, "A_Color").default_value = color_a
        _input_identifier(mix, "B_Color").default_value = color_b
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Mix Edge {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output_identifier(mix, "Result_Color"),
            _input(separate, "Color"),
        )
        for channel in channels:
            channels[channel].append(_output(separate, channel))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Edge Mix Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Edge Mix Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Mix Color Edge Cases"
    combine.mode = "RGB"
    for name, sockets in channels.items():
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
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


def _gradient_matrix(scene: Any) -> None:
    """Cover every Cycles Gradient Texture mode and saturation branch."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("LINEAR", (-2.0, 0.0, 0.0)),
        ("LINEAR", (0.37, 2.0, -1.0)),
        ("LINEAR", (2.0, 0.0, 0.0)),
        ("QUADRATIC", (-1.0, 0.0, 0.0)),
        ("QUADRATIC", (0.5, 0.0, 0.0)),
        ("EASING", (-0.5, 0.0, 0.0)),
        ("EASING", (0.3, 0.0, 0.0)),
        ("EASING", (1.5, 0.0, 0.0)),
        ("DIAGONAL", (-1.0, 0.2, 0.0)),
        ("DIAGONAL", (1.4, 0.8, 0.0)),
        ("RADIAL", (1.0, 0.0, 0.0)),
        ("RADIAL", (0.0, -1.0, 0.0)),
        ("SPHERICAL", (0.0, 0.0, 0.0)),
        ("SPHERICAL", (1.0, 0.0, 0.0)),
        ("QUADRATIC_SPHERE", (0.0, 0.0, 0.0)),
        ("QUADRATIC_SPHERE", (0.5, 0.5, 0.5)),
    )
    materials = []
    for index, (gradient_type, vector) in enumerate(cases):
        material, tree, output = _material(
            f"Gradient Matrix {index:02d} {gradient_type}"
        )
        gradient = tree.nodes.new("ShaderNodeTexGradient")
        gradient.name = f"Gradient {index:02d} {gradient_type}"
        gradient.gradient_type = gradient_type
        tree.links.new(
            _linked_vector(
                tree,
                f"Gradient Vector {index:02d}",
                vector,
            ),
            _input(gradient, "Vector"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Gradient Emission {index:02d}"
        tree.links.new(
            _output(gradient, "Fac"),
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
        columns=4,
        rows=4,
        name="Gradient Matrix",
    )


def _mix_rgb_legacy_modes(scene: Any) -> None:
    """Cover every Cycles blend mode of the legacy MixRGB node."""
    material, tree, output = _material("Legacy MixRGB Modes Probe")
    modes = (
        "MIX",
        "DARKEN",
        "MULTIPLY",
        "BURN",
        "LIGHTEN",
        "SCREEN",
        "DODGE",
        "ADD",
        "OVERLAY",
        "SOFT_LIGHT",
        "LINEAR_LIGHT",
        "DIFFERENCE",
        "EXCLUSION",
        "SUBTRACT",
        "DIVIDE",
        "HUE",
        "SATURATION",
        "COLOR",
        "VALUE",
    )
    channels: dict[str, list[Any]] = {
        "Red": [],
        "Green": [],
        "Blue": [],
    }
    for index, mode in enumerate(modes):
        mix = tree.nodes.new("ShaderNodeMixRGB")
        mix.name = f"Legacy MixRGB {index:02d} {mode}"
        mix.blend_type = mode
        mix.use_alpha = index % 2 == 0
        mix.use_clamp = False
        _input(mix, "Fac").default_value = 0.37
        _input(mix, "Color1").default_value = (
            0.17,
            0.63,
            0.89,
            0.2,
        )
        _input(mix, "Color2").default_value = (
            0.82,
            0.24,
            0.51,
            0.9,
        )
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Legacy {mode}"
        separate.mode = "RGB"
        tree.links.new(
            _output(mix, "Color"),
            _input(separate, "Color"),
        )
        for channel in channels:
            channels[channel].append(_output(separate, channel))

    def average(name: str, sockets: list[Any]) -> Any:
        current = sockets[0]
        for index, socket in enumerate(sockets[1:], start=1):
            add = tree.nodes.new("ShaderNodeMath")
            add.name = f"{name} Legacy Sum {index}"
            add.operation = "ADD"
            tree.links.new(current, add.inputs[0])
            tree.links.new(socket, add.inputs[1])
            current = _output(add, "Value")
        scale = tree.nodes.new("ShaderNodeMath")
        scale.name = f"{name} Legacy Average"
        scale.operation = "MULTIPLY"
        tree.links.new(current, scale.inputs[0])
        scale.inputs[1].default_value = 1.0 / len(sockets)
        return _output(scale, "Value")

    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Legacy MixRGB Modes"
    combine.mode = "RGB"
    for name, sockets in channels.items():
        tree.links.new(
            average(name, sockets),
            _input(combine, name),
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


def _separate_color_modes(scene: Any) -> None:
    material, tree, output = _material("Separate Color Modes")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Separate Modes"
    combine.mode = "RGB"
    for index, (mode, source, target) in enumerate(
        (
            ("RGB", "Red", "Red"),
            ("HSV", "Green", "Green"),
            ("HSL", "Red", "Blue"),
        )
    ):
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Separate Color {mode} {index}"
        separate.mode = mode
        _input(separate, "Color").default_value = (
            0.13,
            0.47,
            0.82,
            1.0,
        )
        tree.links.new(
            _output(separate, source),
            _input(combine, target),
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


def _combine_color_modes(scene: Any) -> None:
    material, tree, output = _material("Combine Color Modes")
    packed = tree.nodes.new("ShaderNodeCombineColor")
    packed.name = "Pack Combine Modes"
    packed.mode = "RGB"
    for index, (mode, channels, source, target) in enumerate(
        (
            ("RGB", (0.2, 0.6, 0.9), "Red", "Red"),
            ("HSV", (0.73, 0.61, 0.84), "Green", "Green"),
            ("HSL", (0.13, 0.55, 0.36), "Blue", "Blue"),
        )
    ):
        combine = tree.nodes.new("ShaderNodeCombineColor")
        combine.name = f"Combine Color {mode} {index}"
        combine.mode = mode
        for name, value in zip(
            ("Red", "Green", "Blue"),
            channels,
            strict=True,
        ):
            _input(combine, name).default_value = value
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Select Combine Channel {index}"
        separate.mode = "RGB"
        tree.links.new(
            _output(combine, "Color"),
            _input(separate, "Color"),
        )
        tree.links.new(
            _output(separate, source),
            _input(packed, target),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(packed, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _legacy_separate_combine_matrix(scene: Any) -> None:
    """Cover all sockets of legacy RGB, HSV, and XYZ split/pack nodes."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    value_sets = (
        (
            (0.13, 0.47, 0.91),
            (0.13, 0.70, 0.80),
            (-0.70, 0.25, 1.30),
        ),
        (
            (1.20, -0.30, 0.50),
            (0.83, 0.20, 1.30),
            (2.00, -3.00, 4.00),
        ),
        (
            (0.00, 1.00, 2.00),
            (1.20, 0.90, 0.40),
            (0.00, 0.00, 0.00),
        ),
        (
            (-1.50, 0.25, 3.00),
            (0.00, 0.00, 2.00),
            (0.0001, 0.999, -1.25),
        ),
    )
    materials = []
    for set_index, (rgb, hsv, xyz) in enumerate(value_sets):
        components = (0, 1, 2, set_index % 3)
        for local_index, component in enumerate(components):
            index = set_index * 4 + local_index
            material, tree, output = _material(
                f"Legacy Split Pack {index:02d}"
            )

            combine_rgb = tree.nodes.new("ShaderNodeCombineRGB")
            combine_rgb.name = f"Combine RGB {index:02d}"
            for socket, value in zip(
                ("R", "G", "B"), rgb, strict=True
            ):
                _input(combine_rgb, socket).default_value = value
            separate_rgb = tree.nodes.new("ShaderNodeSeparateRGB")
            separate_rgb.name = f"Separate RGB {index:02d}"
            tree.links.new(
                _output(combine_rgb, "Image"),
                _input(separate_rgb, "Image"),
            )

            combine_hsv = tree.nodes.new("ShaderNodeCombineHSV")
            combine_hsv.name = f"Combine HSV {index:02d}"
            for socket, value in zip(
                ("H", "S", "V"), hsv, strict=True
            ):
                _input(combine_hsv, socket).default_value = value
            separate_hsv = tree.nodes.new("ShaderNodeSeparateHSV")
            separate_hsv.name = f"Separate HSV {index:02d}"
            tree.links.new(
                _output(combine_hsv, "Color"),
                _input(separate_hsv, "Color"),
            )

            combine_xyz = tree.nodes.new("ShaderNodeCombineXYZ")
            combine_xyz.name = f"Combine XYZ {index:02d}"
            for socket, value in zip(
                ("X", "Y", "Z"), xyz, strict=True
            ):
                _input(combine_xyz, socket).default_value = value
            separate_xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
            separate_xyz.name = f"Separate XYZ {index:02d}"
            tree.links.new(
                _output(combine_xyz, "Vector"),
                _input(separate_xyz, "Vector"),
            )

            packed = tree.nodes.new("ShaderNodeCombineColor")
            packed.name = f"Pack Legacy Results {index:02d}"
            packed.mode = "RGB"
            tree.links.new(
                _output(
                    separate_rgb,
                    ("R", "G", "B")[component],
                ),
                _input(packed, "Red"),
            )
            tree.links.new(
                _output(
                    separate_hsv,
                    ("H", "S", "V")[component],
                ),
                _input(packed, "Green"),
            )
            tree.links.new(
                _output(
                    separate_xyz,
                    ("X", "Y", "Z")[component],
                ),
                _input(packed, "Blue"),
            )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"Legacy Emission {index:02d}"
            tree.links.new(
                _output(packed, "Color"),
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
        columns=4,
        rows=4,
        name="Legacy Separate Combine Matrix",
    )
