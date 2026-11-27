"""Generate a Blender 5.3 Phase 3 Spectral Volume + Random-Walk SSS showcase scene.

Usage:
    blender --background --factory-startup --python \
        tests/scenes/generate_spectral_volume_sss_scene.py -- <out_dir>

Writes `<out_dir>/scene.blend`. Export to Psycles bundle with:
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
    return parser.parse_args(argv)


def _aim_object(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _make_diffuse_material(
    name: str, color: tuple[float, float, float]
) -> bpy.types.Material:
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


def _make_heterogeneous_blackbody_volume_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    vec_len = nt.nodes.new("ShaderNodeVectorMath")
    vec_len.operation = "LENGTH"
    nt.links.new(tex_coord.outputs["Object"], vec_len.inputs[0])

    density_map = nt.nodes.new("ShaderNodeMapRange")
    density_map.clamp = True
    density_map.interpolation_type = "SMOOTHSTEP"
    density_map.inputs["From Min"].default_value = 0.0
    density_map.inputs["From Max"].default_value = 0.95
    density_map.inputs["To Min"].default_value = 3.2
    density_map.inputs["To Max"].default_value = 0.0
    nt.links.new(vec_len.outputs["Value"], density_map.inputs["Value"])

    blackbody_map = nt.nodes.new("ShaderNodeMapRange")
    blackbody_map.clamp = True
    blackbody_map.interpolation_type = "SMOOTHSTEP"
    blackbody_map.inputs["From Min"].default_value = 0.0
    blackbody_map.inputs["From Max"].default_value = 0.45
    blackbody_map.inputs["To Min"].default_value = 2.5
    blackbody_map.inputs["To Max"].default_value = 0.0
    nt.links.new(vec_len.outputs["Value"], blackbody_map.inputs["Value"])

    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (0.28, 0.78, 0.95, 1.0)
    pv.inputs["Absorption Color"].default_value = (0.90, 0.35, 0.18, 1.0)
    pv.inputs["Anisotropy"].default_value = 0.25
    pv.inputs["Emission Strength"].default_value = 0.0
    pv.inputs["Temperature"].default_value = 2500.0
    pv.inputs["Blackbody Tint"].default_value = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(density_map.outputs["Result"], pv.inputs["Density"])
    nt.links.new(blackbody_map.outputs["Result"], pv.inputs["Blackbody Intensity"])

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
    return mat


def _make_random_walk_sss_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.subsurface_method = "RANDOM_WALK"
    bsdf.inputs["Base Color"].default_value = (0.85, 0.72, 0.62, 1.0)
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Scale"].default_value = 0.35
    bsdf.inputs["Subsurface Radius"].default_value = (0.80, 0.15, 0.04)
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["IOR"].default_value = 1.40

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.6, 0.7, 0.9, 1.0)
        bg.inputs["Strength"].default_value = 0.02

    # 1. Matte gray diffuse floor at z = 0.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=16, y_subdivisions=16, size=24.0, location=(0.0, 0.0, 0.0)
    )
    floor = bpy.context.active_object
    floor.name = "StudioFloor"
    floor.data.materials.append(_make_diffuse_material("FloorDiffuse", (0.45, 0.45, 0.45)))

    # 2. Left: heterogeneous chromatic Principled Volume with 2500K Blackbody core.
    bpy.ops.mesh.primitive_cube_add(size=1.4, location=(-1.15, 0.0, 0.75))
    vol_cube = bpy.context.active_object
    vol_cube.name = "SpectralVolumeCube"
    vol_cube.data.materials.append(
        _make_heterogeneous_blackbody_volume_material("HeterogeneousBlackbodyVolume")
    )

    # 3. Right: smooth subdivided Suzanne with Random-Walk chromatic SSS.
    bpy.ops.mesh.primitive_monkey_add(size=1.35, location=(1.15, 0.0, 0.72))
    suzanne = bpy.context.active_object
    suzanne.name = "RandomWalkSuzanne"
    suzanne.rotation_euler = (0.0, 0.0, math.radians(-18.0))
    subsurf = suzanne.modifiers.new(name="Subsurf", type="SUBSURF")
    subsurf.levels = 2
    subsurf.render_levels = 2
    bpy.context.view_layer.objects.active = suzanne
    bpy.ops.object.modifier_apply(modifier=subsurf.name)
    for poly in suzanne.data.polygons:
        poly.use_smooth = True
    suzanne.data.materials.append(_make_random_walk_sss_material("ChromaticRandomWalkSSS"))

    # 4. Backlight and front-left soft key/fill area lights.
    back_data = bpy.data.lights.new(name="Backlight", type="AREA")
    back_data.shape = "SQUARE"
    back_data.size = 1.1
    back_data.energy = 260.0
    back_data.color = (1.0, 0.98, 0.95)
    back_data.cycles.use_multiple_importance_sampling = True
    back_obj = bpy.data.objects.new(name="Backlight", object_data=back_data)
    scene.collection.objects.link(back_obj)
    back_obj.location = (0.4, 2.5, 1.35)
    _aim_object(back_obj, (0.2, 0.0, 0.65))

    key_data = bpy.data.lights.new(name="SoftKeyFill", type="AREA")
    key_data.shape = "SQUARE"
    key_data.size = 2.0
    key_data.energy = 90.0
    key_data.color = (0.95, 0.98, 1.0)
    key_data.cycles.use_multiple_importance_sampling = True
    key_obj = bpy.data.objects.new(name="SoftKeyFill", object_data=key_data)
    scene.collection.objects.link(key_obj)
    key_obj.location = (-2.6, -2.8, 2.4)
    _aim_object(key_obj, (0.0, 0.0, 0.65))

    # 5. Camera framing both the volume cube and backlit Suzanne.
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 42.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -4.4, 1.85)
    _aim_object(cam_obj, (0.0, 0.0, 0.65))
    scene.camera = cam_obj

    scene.render.engine = "CYCLES"
    scene.cycles.max_bounces = 8
    scene.cycles.volume_bounces = 4
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.volume_step_rate = 0.5
    scene.cycles.volume_max_steps = 256
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0

    blend_path = out_dir / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


if __name__ == "__main__":
    main()
