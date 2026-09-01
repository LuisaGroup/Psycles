"""Cycles 5.2 camera and path-state node probes."""

from __future__ import annotations

from typing import Any

from mathutils import Vector

from .support import (
    _input,
    _input_identifier,
    _material,
    _material_matrix,
    _output,
    _plane,
)


_LIGHT_PATH_OUTPUTS = (
    "Is Camera Ray",
    "Is Shadow Ray",
    "Is Diffuse Ray",
    "Is Glossy Ray",
    "Is Singular Ray",
    "Is Reflection Ray",
    "Is Transmission Ray",
    "Is Volume Scatter Ray",
    "Ray Length",
    "Ray Depth",
    "Diffuse Depth",
    "Glossy Depth",
    "Transparent Depth",
    "Transmission Depth",
    "Portal Depth",
)

_INFO_OUTPUTS = (
    (
        "Object",
        "ShaderNodeObjectInfo",
        (
            "Location",
            "Color",
            "Alpha",
            "Object Index",
            "Material Index",
            "Random",
        ),
    ),
    (
        "Particle",
        "ShaderNodeParticleInfo",
        (
            "Index",
            "Random",
            "Age",
            "Lifetime",
            "Location",
            "Size",
            "Velocity",
            "Angular Velocity",
        ),
    ),
    (
        "Hair",
        "ShaderNodeHairInfo",
        (
            "Is Strand",
            "Intercept",
            "Length",
            "Thickness",
            "Tangent Normal",
            "Random",
        ),
    ),
    (
        "Point",
        "ShaderNodePointInfo",
        ("Position", "Radius", "Random"),
    ),
)


def _camera_data(scene: Any) -> None:
    """Keep all three Camera Data outputs live under a nontrivial transform."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01

    camera = scene.camera
    camera.location = (1.1, -0.8, 3.2)
    direction = Vector((0.0, 0.0, 0.0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    material, tree, output = _material("Camera Data")
    data = tree.nodes.new("ShaderNodeCameraData")
    data.name = "Camera Data"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = "Separate View Vector"
    tree.links.new(_output(data, "View Vector"), _input(separate, "Vector"))

    view_x = tree.nodes.new("ShaderNodeMath")
    view_x.name = "Map View X"
    view_x.operation = "MULTIPLY_ADD"
    _input_identifier(view_x, "Value_001").default_value = 0.5
    _input_identifier(view_x, "Value_002").default_value = 0.5
    tree.links.new(
        _output(separate, "X"), _input_identifier(view_x, "Value")
    )

    z_depth = tree.nodes.new("ShaderNodeMath")
    z_depth.name = "Scale Z Depth"
    z_depth.operation = "MULTIPLY"
    _input_identifier(z_depth, "Value_001").default_value = -0.2
    tree.links.new(
        _output(data, "View Z Depth"),
        _input_identifier(z_depth, "Value"),
    )

    distance = tree.nodes.new("ShaderNodeMath")
    distance.name = "Scale View Distance"
    distance.operation = "MULTIPLY"
    _input_identifier(distance, "Value_001").default_value = 0.2
    tree.links.new(
        _output(data, "View Distance"),
        _input_identifier(distance, "Value"),
    )

    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = "Camera Data RGB"
    tree.links.new(_output(view_x, "Value"), _input(combine, "X"))
    tree.links.new(_output(z_depth, "Value"), _input(combine, "Y"))
    tree.links.new(_output(distance, "Value"), _input(combine, "Z"))

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(_output(combine, "Vector"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    _plane(material)


def _light_path_matrix(scene: Any) -> None:
    """Expose every Cycles 5.2 Light Path output on primary camera rays."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = []
    for index, output_name in enumerate(_LIGHT_PATH_OUTPUTS):
        material, tree, output = _material(
            f"Light Path {index:02d} {output_name}"
        )
        light_path = tree.nodes.new("ShaderNodeLightPath")
        light_path.name = "Light Path"
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "Emission"
        tree.links.new(
            _output(light_path, output_name),
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
        columns=5,
        rows=3,
        name="Light Path Matrix",
        frame_bleed=0.01,
    )


def _light_falloff_matrix(scene: Any) -> None:
    """Keep every Cycles Light Falloff output and input encoding live."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01

    cases = (
        ("Quadratic", 8.0, 0.0, False, False),
        ("Linear", 3.5, 2.0, False, False),
        ("Constant", 0.25, 4.0, False, False),
        ("ALL", 6.0, 1.5, False, False),
        ("ALL", 0.0, 0.0, True, True),
        ("Constant", 2.0, -1.0, False, False),
    )
    materials = []
    for index, (
        mode,
        strength,
        smooth,
        linked_strength,
        linked_smooth,
    ) in enumerate(cases):
        material, tree, output = _material(
            f"Light Falloff {index:02d} {mode}"
        )
        falloff = tree.nodes.new("ShaderNodeLightFalloff")
        falloff.name = f"Light Falloff {index:02d}"
        _input(falloff, "Strength").default_value = strength
        _input(falloff, "Smooth").default_value = smooth

        if linked_strength or linked_smooth:
            light_path = tree.nodes.new("ShaderNodeLightPath")
            light_path.name = f"Light Path {index:02d}"
            if linked_strength:
                tree.links.new(
                    _output(light_path, "Ray Length"),
                    _input(falloff, "Strength"),
                )
            if linked_smooth:
                tree.links.new(
                    _output(light_path, "Ray Depth"),
                    _input(falloff, "Smooth"),
                )

        output_names = (
            ("Quadratic", "Linear", "Constant")
            if mode == "ALL"
            else (mode,)
        )
        closures = []
        for output_index, output_name in enumerate(output_names):
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = (
                f"Light Falloff Emission {index:02d} {output_index:02d}"
            )
            tree.links.new(
                _output(falloff, output_name),
                _input(emission, "Color"),
            )
            closures.append(_output(emission, "Emission"))
        while len(closures) > 1:
            left = closures.pop(0)
            right = closures.pop(0)
            add = tree.nodes.new("ShaderNodeAddShader")
            add.name = f"Keep Light Falloff Outputs {index:02d}"
            tree.links.new(left, _input_identifier(add, "Shader"))
            tree.links.new(right, _input_identifier(add, "Shader_001"))
            closures.append(_output(add, "Shader"))
        tree.links.new(closures[0], _input(output, "Surface"))
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=3,
        rows=2,
        name="Light Falloff Matrix",
        frame_bleed=0.01,
    )


def _all_info_outputs_material(
    family: str,
    bl_idname: str,
    output_names: tuple[str, ...],
) -> Any:
    """Keep one original Info node and all of its outputs live."""
    material, tree, output = _material(f"{family} Info All Outputs")
    info = tree.nodes.new(bl_idname)
    info.name = f"{family} Info"
    closures = []
    for index, output_name in enumerate(output_names):
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"{family} Emission {index:02d}"
        socket = _output(info, output_name)
        target = (
            _input(emission, "Strength")
            if socket.bl_idname == "NodeSocketFloat"
            else _input(emission, "Color")
        )
        tree.links.new(socket, target)
        closures.append(_output(emission, "Emission"))
    while len(closures) > 1:
        next_closures = []
        for index in range(0, len(closures), 2):
            if index + 1 == len(closures):
                next_closures.append(closures[index])
                continue
            add = tree.nodes.new("ShaderNodeAddShader")
            add.name = f"{family} Keep Outputs {index:02d}"
            tree.links.new(closures[index], _input_identifier(add, "Shader"))
            tree.links.new(
                closures[index + 1],
                _input_identifier(add, "Shader_001"),
            )
            next_closures.append(_output(add, "Shader"))
        closures = next_closures
    tree.links.new(closures[0], _input(output, "Surface"))
    return material


def _offscreen_material_plane(material: Any, index: int) -> None:
    """Reference a material without evaluating it on a camera ray."""
    import bpy

    bpy.ops.mesh.primitive_plane_add(
        size=0.5,
        enter_editmode=False,
        align="WORLD",
        location=(5.0 + index, 0.0, 0.0),
    )
    plane = bpy.context.object
    plane.name = f"{material.name} Offscreen"
    plane.data.materials.append(material)


def _info_nodes_matrix(scene: Any) -> None:
    """Object Info visual oracle plus all-family SVM stream oracle."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01

    object_materials = []
    for index, output_name in enumerate(_INFO_OUTPUTS[0][2]):
        material, tree, output = _material(
            f"Object Info {index:02d} {output_name}"
        )
        material.pass_index = index + 3
        info = tree.nodes.new("ShaderNodeObjectInfo")
        info.name = "Object Info"
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "Emission"
        socket = _output(info, output_name)
        tree.links.new(
            socket,
            _input(emission, "Strength")
            if socket.bl_idname == "NodeSocketFloat"
            else _input(emission, "Color"),
        )
        tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
        object_materials.append(material)
    surface = _material_matrix(
        scene,
        object_materials,
        columns=3,
        rows=2,
        name="Object Info Matrix",
        frame_bleed=0.01,
    )
    surface.pass_index = 11
    surface.color = (0.2, 0.4, 0.7, 0.6)

    for index, (family, bl_idname, output_names) in enumerate(_INFO_OUTPUTS):
        material = _all_info_outputs_material(
            family,
            bl_idname,
            output_names,
        )
        _offscreen_material_plane(material, index)
