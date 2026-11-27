"""Generate a Blender 5.3 MNEE Shadow Caustics differential test scene.

Usage:
    blender --background --factory-startup --python \
        tests/scenes/generate_mnee_caustics_scene.py -- \
        <out_dir> [--size 512] [--samples 128]

Writes `<out_dir>/scene.blend` and renders the Cycles 5.3 reference EXR to
`<out_dir>/cycles_ref.exr`. Export the Psycles bundle afterward with:
    blender --background <out_dir>/scene.blend --python \
        tools/export_psycles_scene.py -- <out_dir>/bundle
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import bpy
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=128)
    return parser.parse_args(argv)


def _aim_object(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _make_diffuse_material(name: str, color: tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _make_glass_material(
    name: str, color: tuple[float, float, float], ior: float, roughness: float = 0.0
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (*color, 1.0)
    glass.inputs["Roughness"].default_value = roughness
    glass.inputs["IOR"].default_value = ior
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _set_smooth(obj: bpy.types.Object) -> None:
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True


def _configure_cycles_gpu(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("HIP", "OPTIX", "CUDA", "NONE"):
        if dev_type == "NONE":
            scene.cycles.device = "CPU"
            return
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            enabled = False
            for d in prefs.devices:
                if d.type == dev_type:
                    d.use = True
                    enabled = True
                else:
                    d.use = False
            if enabled:
                scene.cycles.device = "GPU"
                return
        except Exception:
            continue
    scene.cycles.device = "CPU"


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # World: subtle ambient fill so silhouettes and shadow contrast read clearly.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.75, 0.85, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 0.04

    # 1. Diffuse receiver floor at z=0.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=16, y_subdivisions=16, size=30.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "CausticFloor"
    floor.data.materials.append(_make_diffuse_material("FloorDiffuse", (0.72, 0.72, 0.75)))
    floor.cycles.is_caustics_receiver = True

    # 2. Primary smooth glass sphere caster.
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.85, location=(0.65, 0.10, 0.90))
    sphere = bpy.context.active_object
    sphere.name = "GlassSphere"
    _set_smooth(sphere)
    sphere.data.materials.append(_make_glass_material("GlassClear", (0.98, 0.99, 1.0), 1.52, 0.0))
    sphere.cycles.is_caustics_caster = True

    # 3. Curved aqua glass lens caster (scaled smooth ellipsoid).
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.80, location=(-1.45, -0.15, 0.95))
    lens = bpy.context.active_object
    lens.name = "GlassLens"
    lens.scale = (1.0, 0.95, 0.55)
    bpy.ops.object.transform_apply(scale=True)
    _set_smooth(lens)
    lens.data.materials.append(_make_glass_material("GlassAqua", (0.92, 0.99, 0.96), 1.55, 0.0))
    lens.cycles.is_caustics_caster = True

    # 4. Small sharp key area light with caustics enabled.
    light_data = bpy.data.lights.new(name="CausticKeyLight", type="AREA")
    light_data.shape = "SQUARE"
    light_data.size = 0.35
    light_data.energy = 320.0
    light_data.spread = math.pi
    light_data.color = (1.0, 0.98, 0.95)
    light_data.cycles.use_multiple_importance_sampling = True
    light_data.cycles.is_caustics_light = True
    light_obj = bpy.data.objects.new(name="CausticKeyLight", object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (2.2, -1.3, 4.4)
    _aim_object(light_obj, (-0.35, 0.0, 0.35))

    # 5. Camera framing both glass objects and their projected caustics.
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 45.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.15, -5.6, 3.15)
    _aim_object(cam_obj, (-0.35, 0.05, 0.45))
    scene.camera = cam_obj

    # 6. Cycles MNEE shadow caustics configuration.
    _configure_cycles_gpu(scene)
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 8
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = True
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0

    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"

    blend_path = out_dir / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    scene.render.filepath = str(out_dir / "cycles_ref.exr")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
