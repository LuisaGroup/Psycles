"""Create canonical Blender scenes for focused Cycles shader-node probes.

Usage:

    blender --background --python create_cycles_shader_probe.py -- \
        output.blend probe-name

The generated .blend is an input to both ``render_cycles_golden.py`` and
``export_psycles_scene.py``. Probe scenes contain no baked shader result.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable
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


def _camera(scene: Any) -> None:
    data = bpy.data.cameras.new("Probe Camera")
    data.type = "ORTHO"
    data.ortho_scale = 2.2
    data.lens = 50.0
    camera = bpy.data.objects.new("Probe Camera", data)
    camera.location = (0.0, 0.0, 3.0)
    scene.collection.objects.link(camera)
    scene.camera = camera


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


def _sphere(material: Any) -> None:
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


def _add_shader_emission(scene: Any) -> None:
    material, tree, output = _material("Add Shader Probe")
    first = tree.nodes.new("ShaderNodeEmission")
    first.name = "Red Emission"
    _input(first, "Color").default_value = (0.8, 0.1, 0.03, 1.0)
    _input(first, "Strength").default_value = 0.75
    second = tree.nodes.new("ShaderNodeEmission")
    second.name = "Blue Emission"
    _input(second, "Color").default_value = (0.02, 0.2, 0.9, 1.0)
    _input(second, "Strength").default_value = 1.25
    add = tree.nodes.new("ShaderNodeAddShader")
    add.name = "Add Shader"
    tree.links.new(_output(first, "Emission"), add.inputs[0])
    tree.links.new(_output(second, "Emission"), add.inputs[1])
    tree.links.new(
        _output(add, "Shader"), _input(output, "Surface")
    )
    _plane(material)


def _mix_shader_emission(scene: Any) -> None:
    material, tree, output = _material("Mix Shader Probe")
    first = tree.nodes.new("ShaderNodeEmission")
    first.name = "First Emission"
    _input(first, "Color").default_value = (0.9, 0.04, 0.12, 1.0)
    _input(first, "Strength").default_value = 1.4
    second = tree.nodes.new("ShaderNodeEmission")
    second.name = "Second Emission"
    _input(second, "Color").default_value = (0.03, 0.72, 0.2, 1.0)
    _input(second, "Strength").default_value = 0.65
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "Mix Shader"
    _input(mix, "Fac").default_value = 0.37
    tree.links.new(_output(first, "Emission"), mix.inputs[1])
    tree.links.new(_output(second, "Emission"), mix.inputs[2])
    tree.links.new(
        _output(mix, "Shader"), _input(output, "Surface")
    )
    _plane(material)


def _transparent_mix(scene: Any) -> None:
    _world(scene, (0.04, 0.22, 0.7, 1.0), 1.8)
    material, tree, output = _material("Transparent Probe")
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    transparent.name = "Transparent BSDF"
    _input(transparent, "Color").default_value = (
        0.75,
        0.9,
        0.6,
        1.0,
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    _input(emission, "Color").default_value = (0.85, 0.08, 0.03, 1.0)
    _input(emission, "Strength").default_value = 1.2
    mix = tree.nodes.new("ShaderNodeMixShader")
    mix.name = "Mix Shader"
    _input(mix, "Fac").default_value = 0.62
    tree.links.new(_output(transparent, "BSDF"), mix.inputs[1])
    tree.links.new(_output(emission, "Emission"), mix.inputs[2])
    tree.links.new(
        _output(mix, "Shader"), _input(output, "Surface")
    )
    _plane(material)


def _diffuse_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Diffuse Probe")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (0.68, 0.24, 0.09, 1.0)
    _input(diffuse, "Roughness").default_value = 0.43
    tree.links.new(
        _output(diffuse, "BSDF"), _input(output, "Surface")
    )
    _sphere(material)


def _principled_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Principled Probe")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Principled BSDF"
    principled.distribution = "GGX"
    _input(principled, "Base Color").default_value = (
        0.32,
        0.12,
        0.06,
        1.0,
    )
    _input(principled, "Metallic").default_value = 0.35
    _input(principled, "Roughness").default_value = 0.28
    _input(principled, "Diffuse Roughness").default_value = 0.0
    _input(principled, "IOR").default_value = 1.45
    _input(principled, "Specular IOR Level").default_value = 0.5
    _input(principled, "Specular Tint").default_value = (
        0.7,
        0.9,
        1.0,
        1.0,
    )
    tree.links.new(
        _output(principled, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _bump_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Bump Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    gradient = tree.nodes.new("ShaderNodeTexGradient")
    gradient.name = "Gradient Texture"
    gradient.gradient_type = "LINEAR"
    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = "Bump"
    _input(bump, "Strength").default_value = 1.0
    _input(bump, "Distance").default_value = 0.2
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.5,
        0.22,
        0.08,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(coordinates, "Generated"),
        _input(gradient, "Vector"),
    )
    tree.links.new(
        _output(gradient, "Fac"),
        _input(bump, "Height"),
    )
    tree.links.new(
        _output(bump, "Normal"),
        _input(diffuse, "Normal"),
    )
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _normal_map_surface(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Normal Map Probe")
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.name = "Normal Map"
    normal_map.space = "TANGENT"
    _input(normal_map, "Strength").default_value = 0.7
    _input(normal_map, "Color").default_value = (
        0.65,
        0.35,
        0.95,
        1.0,
    )
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.5,
        0.22,
        0.08,
        1.0,
    )
    _input(diffuse, "Roughness").default_value = 0.0
    tree.links.new(
        _output(normal_map, "Normal"),
        _input(diffuse, "Normal"),
    )
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _sphere(material)


def _noise_color_3d(scene: Any) -> None:
    material, tree, output = _material("Noise Color 3D Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 3D"
    noise.noise_dimensions = "3D"
    noise.noise_type = "FBM"
    noise.normalize = True
    _input(noise, "Scale").default_value = 1.7
    _input(noise, "Detail").default_value = 2.35
    _input(noise, "Roughness").default_value = 0.61
    _input(noise, "Lacunarity").default_value = 2.2
    _input(noise, "Distortion").default_value = 0.37
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _noise_factor_2d(scene: Any) -> None:
    material, tree, output = _material("Noise Factor 2D Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 2D"
    noise.noise_dimensions = "2D"
    noise.noise_type = "FBM"
    noise.normalize = False
    _input(noise, "Scale").default_value = 2.3
    _input(noise, "Detail").default_value = 1.75
    _input(noise, "Roughness").default_value = 0.43
    _input(noise, "Lacunarity").default_value = 1.8
    _input(noise, "Distortion").default_value = 0.0
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Fac"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _noise_bump_object(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Noise Bump Object Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 3D"
    noise.noise_dimensions = "3D"
    noise.noise_type = "FBM"
    noise.normalize = True
    _input(noise, "Scale").default_value = 20.0
    _input(noise, "Detail").default_value = 2.0
    _input(noise, "Roughness").default_value = 0.5
    _input(noise, "Lacunarity").default_value = 2.0
    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = "Bump"
    _input(bump, "Strength").default_value = 0.2
    _input(bump, "Distance").default_value = 0.005
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.5,
        0.22,
        0.08,
        1.0,
    )
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Fac"),
        _input(bump, "Height"),
    )
    tree.links.new(
        _output(bump, "Normal"),
        _input(diffuse, "Normal"),
    )
    tree.links.new(
        _output(diffuse, "BSDF"),
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


def _background_world(scene: Any) -> None:
    _world(scene, (0.16, 0.48, 0.77, 1.0), 2.3)


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


_PROBES: dict[str, Callable[[Any], None]] = {
    "add_shader_emission": _add_shader_emission,
    "background_world": _background_world,
    "bump_surface": _bump_surface,
    "brightness_contrast": _brightness_contrast,
    "clamp": _clamp,
    "combine_color_modes": _combine_color_modes,
    "diffuse_surface": _diffuse_surface,
    "emission_surface": _emission_surface,
    "gamma_color": _gamma_color,
    "integrator_clamp_direct": _integrator_clamp_direct,
    "mix_shader_emission": _mix_shader_emission,
    "node_group_color": _node_group_color,
    "noise_bump_object": _noise_bump_object,
    "noise_color_3d": _noise_color_3d,
    "noise_factor_2d": _noise_factor_2d,
    "normal_map_surface": _normal_map_surface,
    "particle_random_instances": _particle_random_instances,
    "particle_random_nonparticle": _particle_random_nonparticle,
    "principled_surface": _principled_surface,
    "rgb_emission": _rgb_emission,
    "rgb_to_bw": _rgb_to_bw,
    "separate_color_modes": _separate_color_modes,
    "transparent_mix": _transparent_mix,
    "value_emission": _value_emission,
}


def _main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit(
            "expected: output.blend probe-name; probes: "
            + ", ".join(sorted(_PROBES))
        )
    output = pathlib.Path(args[0]).resolve()
    probe_name = args[1]
    try:
        create = _PROBES[probe_name]
    except KeyError as error:
        raise SystemExit(
            f"unknown probe {probe_name!r}; probes: "
            + ", ".join(sorted(_PROBES))
        ) from error

    _clear()
    scene = bpy.context.scene
    scene.name = f"Psycles Probe - {probe_name}"
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_depth = "32"
    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 0x51A7
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_light_tree = False
    scene["psycles_probe"] = probe_name
    _camera(scene)
    if probe_name != "background_world":
        _world(scene, (0.0, 0.0, 0.0, 1.0), 0.0)
    create(scene)

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(f"Created Psycles Cycles probe {probe_name}: {output}")


if __name__ == "__main__":
    _main()
