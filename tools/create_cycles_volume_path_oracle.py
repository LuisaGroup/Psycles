#!/usr/bin/env python3
"""Render the finite homogeneous-volume path regression with official Cycles.

Run through Blender:
  blender --background --factory-startup --python \
    tools/create_cycles_volume_path_oracle.py -- /tmp/volume-path.exr
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy


def argument_options() -> argparse.Namespace:
    arguments = sys.argv
    if "--" not in arguments:
        raise RuntimeError("expected arguments after --")
    trailing = arguments[arguments.index("--") + 1 :]
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Cycles AA sample count (default: 1)",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="enable Cycles transparent film",
    )
    options = parser.parse_args(trailing)
    if options.samples <= 0:
        parser.error("--samples must be positive")
    options.output = options.output.resolve()
    return options


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for material in tuple(bpy.data.materials):
        bpy.data.materials.remove(material)
    for world in tuple(bpy.data.worlds):
        bpy.data.worlds.remove(world)


def make_volume_material() -> bpy.types.Material:
    material = bpy.data.materials.new("Cycles homogeneous absorption boundary")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    transparent.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    absorption.inputs["Color"].default_value = (0.2, 0.5, 0.8, 1.0)
    absorption.inputs["Density"].default_value = 0.5
    material.node_tree.links.new(
        transparent.outputs["BSDF"], output.inputs["Surface"]
    )
    material.node_tree.links.new(
        absorption.outputs["Volume"], output.inputs["Volume"]
    )
    return material


def make_world() -> bpy.types.World:
    world = bpy.data.worlds.new("Cycles white world")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 1.0
    world.node_tree.links.new(
        background.outputs["Background"], output.inputs["Surface"]
    )
    world.cycles.sampling_method = "NONE"
    return world


def configure_scene(
    output: Path,
    samples: int,
    transparent: bool,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.seed = 11939
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    scene.cycles.max_bounces = 1
    scene.cycles.min_light_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 1
    scene.cycles.volume_bounces = 0
    scene.cycles.min_transparent_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.blur_glossy = 0.0
    scene.cycles.use_light_tree = False
    scene.cycles.direct_light_sampling_type = "FORWARD_PATH_TRACING"
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = transparent
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.filepath = str(output)
    scene.world = make_world()

    layer = scene.view_layers[0]
    layer.use_pass_environment = True
    layer.cycles.use_pass_volume_direct = True
    layer.cycles.use_pass_volume_indirect = True

    bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0.0, 0.0, 0.0))
    volume = bpy.context.object
    volume.name = "Volume box"
    volume.data.materials.append(make_volume_material())

    camera_data = bpy.data.cameras.new("Inside-volume orthographic camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera_data.clip_start = 0.1
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera


def main() -> None:
    options = argument_options()
    output = options.output
    output.parent.mkdir(parents=True, exist_ok=True)
    clear_scene()
    configure_scene(
        output,
        options.samples,
        options.transparent,
    )
    bpy.ops.render.render(write_still=True)
    print(
        "Cycles volume-path oracle: "
        f"{output} ({options.samples} spp, "
        f"transparent={options.transparent})"
    )


if __name__ == "__main__":
    main()
