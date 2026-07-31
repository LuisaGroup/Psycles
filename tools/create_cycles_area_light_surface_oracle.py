"""Create the authoritative Cycles area-light surface fixture.

Run through Blender:

    blender --background --factory-startup \
      --python tools/create_cycles_area_light_surface_oracle.py -- \
      output.exr --area-shape ELLIPSE --area-spread 1.2

The scene is the raw-closure counterpart of
``tests/test_luisa_area_light_forward.cpp``.
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
        "--samples",
        type=int,
        default=1,
        help="Cycles AA sample count (default: 1)",
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


def make_diffuse_material() -> bpy.types.Material:
    material = bpy.data.materials.new(
        "Cycles area-light oracle diffuse"
    )
    # Current Blender creates shader node trees eagerly. Retain compatibility
    # with releases that created them only after opting in.
    if material.node_tree is None:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (
        0.6200000047683716,
        0.4099999964237213,
        0.23000000417232513,
        1.0,
    )
    diffuse.inputs["Roughness"].default_value = 0.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(
        diffuse.outputs["BSDF"],
        output.inputs["Surface"],
    )
    return material


def make_surface(
    scene: bpy.types.Scene,
    material: bpy.types.Material,
) -> None:
    mesh = bpy.data.meshes.new(
        "Cycles area-light oracle plane"
    )
    mesh.from_pydata(
        (
            (-4.0, -4.0, 0.0),
            (4.0, -4.0, 0.0),
            (4.0, 4.0, 0.0),
            (-4.0, 4.0, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    mesh.materials.append(material)
    surface = bpy.data.objects.new(
        mesh.name, mesh
    )
    scene.collection.objects.link(surface)


def make_area_light(
    scene: bpy.types.Scene,
    shape: str,
    spread: float,
) -> None:
    data = bpy.data.lights.new(
        "Cycles area-light oracle light",
        type="AREA",
    )
    data.color = (
        0.36000001430511475,
        0.7200000286102295,
        1.0,
    )
    data.energy = 37.0
    data.shape = shape
    data.size = 0.800000011920929
    data.size_y = 0.5
    data.spread = spread
    data.normalize = True
    light = bpy.data.objects.new(
        data.name, data
    )
    light.location = (
        0.3700000047683716,
        -0.20999999344348907,
        1.399999976158142,
    )
    light.visible_camera = False
    scene.collection.objects.link(light)


def make_camera(
    scene: bpy.types.Scene,
) -> None:
    data = bpy.data.cameras.new(
        "Cycles area-light oracle camera"
    )
    data.type = "ORTHO"
    data.ortho_scale = 2.200000047683716
    data.clip_start = 0.10000000149011612
    data.clip_end = 1000.0
    camera = bpy.data.objects.new(
        data.name, data
    )
    camera.location = (0.0, 0.0, 3.0)
    scene.collection.objects.link(camera)
    scene.camera = camera


def make_world(
    scene: bpy.types.Scene,
) -> None:
    world = bpy.data.worlds.new(
        "Cycles area-light oracle world"
    )
    if world.node_tree is None:
        world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    background = nodes.new(
        "ShaderNodeBackground"
    )
    background.inputs["Color"].default_value = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    background.inputs["Strength"].default_value = 0.0
    output = nodes.new(
        "ShaderNodeOutputWorld"
    )
    world.node_tree.links.new(
        background.outputs["Background"],
        output.inputs["Surface"],
    )
    scene.world = world


def set_if_present(
    owner: object,
    name: str,
    value: object,
) -> None:
    if hasattr(owner, name):
        setattr(owner, name, value)


def configure_scene(
    output: Path,
    shape: str,
    spread: float,
    samples: int,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.seed = 20903
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    set_if_present(
        scene.cycles,
        "sampling_pattern",
        "TABULATED_SOBOL",
    )
    set_if_present(
        scene.cycles,
        "scrambling_distance",
        1.0,
    )
    set_if_present(
        scene.cycles,
        "use_auto_scrambling_distance",
        False,
    )
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 10.0
    scene.cycles.blur_glossy = 1.0
    scene.cycles.use_light_tree = False
    scene.cycles.pixel_filter_type = (
        "BLACKMAN_HARRIS"
    )
    scene.cycles.filter_width = 1.5

    scene.render.resolution_x = 32
    scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.render.use_border = False
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.filepath = str(output)
    image = scene.render.image_settings
    if hasattr(image, "media_type"):
        image.media_type = "MULTI_LAYER_IMAGE"
    image.file_format = "OPEN_EXR_MULTILAYER"
    image.color_mode = "RGBA"
    image.color_depth = "32"
    image.exr_codec = "ZIP"
    for view_layer in scene.view_layers:
        view_layer.use = True
        view_layer.use_pass_combined = True

    make_surface(
        scene,
        make_diffuse_material(),
    )
    make_area_light(
        scene,
        shape,
        spread,
    )
    make_camera(scene)
    make_world(scene)


def main() -> None:
    options = argument_options()
    options.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    clear_scene()
    configure_scene(
        options.output,
        options.area_shape,
        options.area_spread,
        options.samples,
    )
    bpy.ops.render.render(write_still=True)
    print(
        "Cycles area-light surface oracle rendered: "
        f"{options.output}"
    )


if __name__ == "__main__":
    main()
