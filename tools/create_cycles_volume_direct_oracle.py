#!/usr/bin/env python3
"""Render the homogeneous distant-light volume regression with Cycles.

Run through Blender:
  blender --background --factory-startup --python \
    tools/create_cycles_volume_direct_oracle.py -- /tmp/volume-direct.exr
"""

from __future__ import annotations

import argparse
import math
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
        "--light",
        choices=(
            "distant",
            "point",
            "spot",
            "area",
            "mesh",
            "environment",
            "environment_constant",
        ),
        default="distant",
        help="emitter family (default: distant)",
    )
    parser.add_argument(
        "--volume-sampling",
        choices=(
            "DISTANCE",
            "EQUIANGULAR",
            "MULTIPLE_IMPORTANCE",
        ),
        default="DISTANCE",
        help="Cycles material volume-sampling policy",
    )
    parser.add_argument(
        "--direct-light-sampling",
        choices=(
            "MULTIPLE_IMPORTANCE_SAMPLING",
            "NEXT_EVENT_ESTIMATION",
            "FORWARD_PATH_TRACING",
        ),
        default="MULTIPLE_IMPORTANCE_SAMPLING",
        help="Cycles direct-light strategy",
    )
    parser.add_argument(
        "--area-shape",
        choices=("RECTANGLE", "ELLIPSE"),
        default="RECTANGLE",
        help="area-light primitive (default: RECTANGLE)",
    )
    parser.add_argument(
        "--area-spread",
        type=float,
        default=math.pi,
        help="area-light spread in radians (default: pi)",
    )
    parser.add_argument(
        "--hide-world-camera",
        action="store_true",
        help="disable the World Camera visibility bit",
    )
    parser.add_argument(
        "--hide-world-scatter",
        action="store_true",
        help="disable the World Volume Scatter visibility bit",
    )
    parser.add_argument(
        "--world-sampling",
        choices=("MANUAL", "AUTOMATIC", "NONE"),
        default=None,
        help="override the environment World sampling policy",
    )
    options = parser.parse_args(trailing)
    if options.samples <= 0:
        parser.error("--samples must be positive")
    if (
        not math.isfinite(options.area_spread)
        or not 0.0 <= options.area_spread <= math.pi
    ):
        parser.error("--area-spread must be finite and within [0, pi]")
    options.output = options.output.resolve()
    return options


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for material in tuple(bpy.data.materials):
        bpy.data.materials.remove(material)
    for world in tuple(bpy.data.worlds):
        bpy.data.worlds.remove(world)


def make_volume_material(
    volume_sampling: str,
) -> bpy.types.Material:
    material = bpy.data.materials.new(
        "Cycles homogeneous isotropic volume boundary"
    )
    if material.node_tree is None:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    transparent.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    scatter.inputs["Density"].default_value = 0.5
    scatter.inputs["Anisotropy"].default_value = 0.0
    material.node_tree.links.new(
        transparent.outputs["BSDF"], output.inputs["Surface"]
    )
    material.node_tree.links.new(
        scatter.outputs["Volume"], output.inputs["Volume"]
    )
    material.cycles.volume_sampling = volume_sampling
    return material


def make_world(
    light_type: str,
    sampling_method: str | None,
) -> bpy.types.World:
    environment = light_type in {
        "environment",
        "environment_constant",
    }
    spatial_environment = (
        light_type == "environment"
    )
    world = bpy.data.worlds.new(
        "Cycles raw environment closure"
        if environment
        else "Cycles black world"
    )
    if world.node_tree is None:
        world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    if spatial_environment:
        coordinates = nodes.new(
            "ShaderNodeTexCoord"
        )
        gradient = nodes.new(
            "ShaderNodeTexGradient"
        )
        gradient.gradient_type = "LINEAR"
        remap = nodes.new(
            "ShaderNodeMath"
        )
        remap.operation = "MULTIPLY_ADD"
        remap.inputs[1].default_value = 0.25
        remap.inputs[2].default_value = 0.5
        world.node_tree.links.new(
            coordinates.outputs["Generated"],
            gradient.inputs["Vector"],
        )
        world.node_tree.links.new(
            gradient.outputs["Fac"],
            remap.inputs[0],
        )
        world.node_tree.links.new(
            remap.outputs["Value"],
            background.inputs["Color"],
        )
    else:
        background.inputs[
            "Color"
        ].default_value = (
            (0.8, 0.4, 0.2, 1.0)
            if environment
            else (0.0, 0.0, 0.0, 1.0)
        )
    background.inputs["Strength"].default_value = (
        1.0 if environment else 0.0
    )
    world.node_tree.links.new(
        background.outputs["Background"], output.inputs["Surface"]
    )
    world.cycles.sampling_method = (
        sampling_method
        if sampling_method is not None
        else (
            "MANUAL"
            if environment
            else "NONE"
        )
    )
    if (
        environment
        and world.cycles.sampling_method
        == "MANUAL"
    ):
        world.cycles.sample_map_resolution = 4
    return world


def make_light(
    scene: bpy.types.Scene,
    light_type: str,
    area_shape: str,
    area_spread: float,
) -> None:
    if light_type in {
        "environment",
        "environment_constant",
    }:
        return
    if light_type == "mesh":
        material = bpy.data.materials.new(
            "Cycles raw mesh emission closure"
        )
        if material.node_tree is None:
            material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (
            1.0,
            1.0,
            1.0,
            1.0,
        )
        emission.inputs["Strength"].default_value = 10.0
        material.node_tree.links.new(
            emission.outputs["Emission"],
            output.inputs["Surface"],
        )
        material.cycles.emission_sampling = "FRONT"

        mesh = bpy.data.meshes.new(
            "Cycles finite-volume triangle mesh"
        )
        mesh.from_pydata(
            (
                (0.6, -0.6, -0.8),
                (1.4, -0.6, -0.8),
                (1.0, 0.2, -0.8),
            ),
            (),
            ((0, 1, 2),),
        )
        mesh.materials.append(material)
        light = bpy.data.objects.new(mesh.name, mesh)
        # The authored local triangle has +Z orientation. Reflection changes
        # the world-space cross product, so this exercises Cycles'
        # SD_OBJECT_NEGATIVE_SCALE correction while keeping the emitting
        # front oriented toward the camera-side half-space.
        light.scale.x = -1.0
        scene.collection.objects.link(light)
        return
    if light_type == "area":
        light_data = bpy.data.lights.new(
            "Cycles finite-volume area light",
            type="AREA",
        )
        light_data.color = (1.0, 1.0, 1.0)
        light_data.energy = 10.0
        light_data.shape = area_shape
        light_data.size = 1.0
        light_data.size_y = 0.6
        light_data.spread = area_spread
        light_data.normalize = True
        light = bpy.data.objects.new(
            light_data.name, light_data
        )
        light.location = (0.2, -0.1, -0.4)
        scene.collection.objects.link(light)
        return
    if light_type == "point":
        light_data = bpy.data.lights.new(
            "Cycles finite-volume point light",
            type="POINT",
        )
        light_data.color = (1.0, 1.0, 1.0)
        light_data.energy = 10.0
        light_data.shadow_soft_size = 0.0
        light_data.normalize = True
        light = bpy.data.objects.new(
            light_data.name, light_data
        )
        light.location = (0.6, -0.25, -0.8)
        scene.collection.objects.link(light)
        return
    if light_type == "spot":
        light_data = bpy.data.lights.new(
            "Cycles finite-volume spot light",
            type="SPOT",
        )
        light_data.color = (1.0, 1.0, 1.0)
        light_data.energy = 10.0
        light_data.shadow_soft_size = 0.0
        light_data.spot_size = 0.9
        light_data.spot_blend = 0.35
        light_data.use_soft_falloff = False
        light_data.normalize = True
        light = bpy.data.objects.new(
            light_data.name, light_data
        )
        light.location = (0.2, -0.1, -0.4)
        scene.collection.objects.link(light)
        return

    light_data = bpy.data.lights.new(
        "Cycles unit distant light", type="SUN"
    )
    light_data.color = (1.0, 1.0, 1.0)
    light_data.energy = 1.0
    light_data.angle = 0.0
    light_data.normalize = True
    light = bpy.data.objects.new(light_data.name, light_data)
    scene.collection.objects.link(light)


def configure_scene(
    output: Path,
    samples: int,
    light_type: str,
    volume_sampling: str,
    direct_light_sampling: str,
    area_shape: str,
    area_spread: float,
    hide_world_camera: bool,
    hide_world_scatter: bool,
    world_sampling: str | None,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.seed = 11939
    # Psycles currently implements Cycles' explicit Tabulated Sobol mode.
    # Keep the oracle on that authored mode instead of Blender 5.2's
    # Automatic -> Blue-Noise render default.
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
    scene.cycles.direct_light_sampling_type = (
        direct_light_sampling
    )
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.render.resolution_percentage = 100
    image_settings = scene.render.image_settings
    if hasattr(image_settings, "media_type"):
        image_settings.media_type = "MULTI_LAYER_IMAGE"
    image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.filepath = str(output)
    scene.world = make_world(
        light_type,
        world_sampling,
    )
    scene.world.cycles_visibility.camera = (
        not hide_world_camera
    )
    scene.world.cycles_visibility.scatter = (
        not hide_world_scatter
    )
    print(
        "Cycles world sampling: "
        f"{scene.world.cycles.sampling_method}, "
        "camera="
        f"{scene.world.cycles_visibility.camera}, "
        "scatter="
        f"{scene.world.cycles_visibility.scatter}"
    )

    layer = scene.view_layers[0]
    layer.use_pass_environment = True
    layer.cycles.use_pass_volume_direct = True
    layer.cycles.use_pass_volume_indirect = True

    bpy.ops.mesh.primitive_cube_add(
        size=4.0, location=(0.0, 0.0, 0.0)
    )
    volume = bpy.context.object
    volume.name = "Volume box"
    volume.data.materials.append(
        make_volume_material(volume_sampling)
    )

    make_light(
        scene,
        light_type,
        area_shape,
        area_spread,
    )

    camera_data = bpy.data.cameras.new(
        "Inside-volume orthographic camera"
    )
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
        options.light,
        options.volume_sampling,
        options.direct_light_sampling,
        options.area_shape,
        options.area_spread,
        options.hide_world_camera,
        options.hide_world_scatter,
        options.world_sampling,
    )
    bpy.ops.render.render(write_still=True)
    print(
        "Cycles volume-direct oracle: "
        f"{output} ({options.samples} spp, "
        f"light={options.light}, "
        f"volume_sampling={options.volume_sampling})"
    )


if __name__ == "__main__":
    main()
