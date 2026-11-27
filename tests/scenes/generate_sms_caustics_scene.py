"""Generate a Blender 5.3 Phase 2B SMS (Glass Sphere + Concave Metal Mirror) scene.

Usage:
    blender --background --factory-startup --python \
        tests/scenes/generate_sms_caustics_scene.py -- \
        <out_dir> [--roughness 0.0]

Writes `<out_dir>/scene.blend`. Export to Psycles bundle with:
    blender --background <out_dir>/scene.blend --python \
        tools/export_psycles_scene.py -- <out_dir>/bundle
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import bmesh
import bpy
from mathutils import Vector


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=pathlib.Path)
    parser.add_argument("--roughness", type=float, default=0.0)
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


def _make_glass_material(
    name: str, color: tuple[float, float, float], ior: float, roughness: float
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


def _make_glossy_material(
    name: str, color: tuple[float, float, float], roughness: float
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    glossy = nt.nodes.new("ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (*color, 1.0)
    glossy.inputs["Roughness"].default_value = roughness
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(glossy.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _set_smooth(obj: bpy.types.Object) -> None:
    for poly in obj.data.polygons:
        poly.use_smooth = True


def _create_concave_cylinder_arc(
    name: str,
    center_xy: tuple[float, float],
    radius: float,
    half_angle_rad: float,
    z0: float,
    z1: float,
    segs: int,
    mat: bpy.types.Material,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name + "Mesh")
    bm = bmesh.new()
    cx, cy = center_xy
    bot_verts = []
    top_verts = []
    for i in range(segs + 1):
        t = float(i) / float(segs)
        theta = -half_angle_rad + 2.0 * half_angle_rad * t
        px = cx + radius * math.sin(theta)
        py = cy + radius * math.cos(theta)
        bot_verts.append(bm.verts.new((px, py, z0)))
        top_verts.append(bm.verts.new((px, py, z1)))
    bm.verts.ensure_lookup_table()
    for i in range(segs):
        # CCW winding from viewer side (-Y) so face normal points inward.
        bm.faces.new((bot_verts[i], bot_verts[i + 1], top_verts[i + 1], top_verts[i]))
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    _set_smooth(obj)
    return obj


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    r = float(args.roughness)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Subtle ambient fill.
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.75, 0.85, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 0.03

    # 1. Diffuse receiver floor at z = 0.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=16, y_subdivisions=16, size=30.0, location=(0.0, 0.0, 0.0)
    )
    floor = bpy.context.active_object
    floor.name = "CausticFloor"
    floor.data.materials.append(
        _make_diffuse_material("FloorDiffuse", (0.58, 0.58, 0.60))
    )
    floor.cycles.is_caustics_receiver = True

    # 2. Smooth/glossy glass sphere caster.
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96, ring_count=48, radius=0.62, location=(-0.90, -0.25, 0.65)
    )
    sphere = bpy.context.active_object
    sphere.name = "GlassSphere"
    _set_smooth(sphere)
    sphere.data.materials.append(
        _make_glass_material("GlassClear", (0.98, 0.99, 1.0), 1.52, r)
    )
    sphere.cycles.is_caustics_caster = True

    # 3. Concave golden metallic mirror arc casting reflective cardioid caustics.
    gold_mat = _make_glossy_material("GoldMirror", (0.96, 0.78, 0.42), r)
    mirror = _create_concave_cylinder_arc(
        "GoldMirrorArc",
        center_xy=(0.90, -0.10),
        radius=0.95,
        half_angle_rad=1.55,
        z0=0.0,
        z1=0.85,
        segs=96,
        mat=gold_mat,
    )
    mirror.cycles.is_caustics_caster = True

    # 4. Small sharp key area light hitting both glass sphere and concave mirror.
    light_data = bpy.data.lights.new(name="CausticKeyLight", type="AREA")
    light_data.shape = "SQUARE"
    light_data.size = 0.28
    light_data.energy = 85.0
    light_data.spread = math.pi
    light_data.color = (1.0, 0.98, 0.95)
    light_data.cycles.use_multiple_importance_sampling = True
    light_data.cycles.is_caustics_light = True
    light_obj = bpy.data.objects.new(name="CausticKeyLight", object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (0.10, -2.20, 2.20)
    _aim_object(light_obj, (0.0, 0.20, 0.30))

    # 5. Camera framing both casters and projected refractive + reflective caustics.
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 38.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -4.30, 2.65)
    _aim_object(cam_obj, (0.0, 0.15, 0.22))
    scene.camera = cam_obj

    scene.render.engine = "CYCLES"
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 8
    # Keep PT caustics flags true and blur_glossy disabled in exported scene.json
    # so --caustics=none serves as clean brute-force path tracing ground truth,
    # while --caustics=sms suppresses double-counting on SMS caster chains.
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True
    scene.cycles.blur_glossy = 0.0
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0

    blend_path = out_dir / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


if __name__ == "__main__":
    main()
