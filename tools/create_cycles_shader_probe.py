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


_PROBES: dict[str, Callable[[Any], None]] = {
    "add_shader_emission": _add_shader_emission,
    "background_world": _background_world,
    "diffuse_surface": _diffuse_surface,
    "emission_surface": _emission_surface,
    "mix_shader_emission": _mix_shader_emission,
    "node_group_color": _node_group_color,
    "rgb_emission": _rgb_emission,
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
