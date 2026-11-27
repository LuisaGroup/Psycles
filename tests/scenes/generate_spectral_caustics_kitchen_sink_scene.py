"""Generate a Blender 5.3 Phase 4 Unified Kitchen-Sink showcase scene.

Combines in one frame:
- Matte caustics-receiver floor
- Smooth dispersive glass sphere (caustics caster; verification script patches dispersion_abbe=15)
- Concave gold metallic mirror arc (reflective cardioid caustics caster)
- Iridescent Airy thin-film dielectric sphere
- Heterogeneous chromatic Principled Volume cube with 2500K Blackbody core
- Subdivided Suzanne with chromatic Random-Walk SSS

Usage:
    blender --background --factory-startup --python \
        tests/scenes/generate_spectral_caustics_kitchen_sink_scene.py -- <out_dir>
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
    return parser.parse_args(argv)


def _aim_object(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _set_smooth(obj: bpy.types.Object) -> None:
    for poly in obj.data.polygons:
        poly.use_smooth = True


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


def _make_thin_film_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.12, 0.14, 0.18, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.06
    bsdf.inputs["IOR"].default_value = 1.50
    bsdf.inputs["Thin Film Thickness"].default_value = 480.0
    bsdf.inputs["Thin Film IOR"].default_value = 1.38
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
    density_map.inputs["From Max"].default_value = 0.85
    density_map.inputs["To Min"].default_value = 3.0
    density_map.inputs["To Max"].default_value = 0.0
    nt.links.new(vec_len.outputs["Value"], density_map.inputs["Value"])

    blackbody_map = nt.nodes.new("ShaderNodeMapRange")
    blackbody_map.clamp = True
    blackbody_map.interpolation_type = "SMOOTHSTEP"
    blackbody_map.inputs["From Min"].default_value = 0.0
    blackbody_map.inputs["From Max"].default_value = 0.42
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

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.70, 0.80, 1.0, 1.0)
        bg.inputs["Strength"].default_value = 0.025

    # 1. Matte caustics receiver floor.
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=16, y_subdivisions=16, size=32.0, location=(0.0, 0.0, 0.0)
    )
    floor = bpy.context.active_object
    floor.name = "KitchenSinkFloor"
    floor.data.materials.append(_make_diffuse_material("FloorDiffuse", (0.52, 0.52, 0.54)))
    floor.cycles.is_caustics_receiver = True

    # 2. Front-left: smooth dispersive glass sphere caustics caster.
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=96, ring_count=48, radius=0.54, location=(-0.95, -0.35, 0.56)
    )
    glass_sphere = bpy.context.active_object
    glass_sphere.name = "GlassSphere"
    _set_smooth(glass_sphere)
    glass_sphere.data.materials.append(
        _make_glass_material("GlassDispersive", (0.98, 0.99, 1.0), 1.52, 0.0)
    )
    glass_sphere.cycles.is_caustics_caster = True

    # 3. Front-right: concave gold metallic mirror arc casting reflective caustics.
    gold_mat = _make_glossy_material("GoldMirror", (0.96, 0.78, 0.42), 0.04)
    mirror = _create_concave_cylinder_arc(
        "GoldMirrorArc",
        center_xy=(0.95, -0.25),
        radius=0.85,
        half_angle_rad=1.45,
        z0=0.0,
        z1=0.72,
        segs=96,
        mat=gold_mat,
    )
    mirror.cycles.is_caustics_caster = True

    # 4. Center-front: Airy thin-film iridescent dielectric sphere.
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=0.36, location=(0.0, 0.25, 0.36)
    )
    thin_film_sphere = bpy.context.active_object
    thin_film_sphere.name = "ThinFilmSphere"
    _set_smooth(thin_film_sphere)
    thin_film_sphere.data.materials.append(_make_thin_film_material("ThinFilmDielectric"))

    # 5. Back-left: heterogeneous chromatic Principled Volume cube with 2500K Blackbody core.
    bpy.ops.mesh.primitive_cube_add(size=1.25, location=(-1.35, 1.35, 0.72))
    vol_cube = bpy.context.active_object
    vol_cube.name = "SpectralVolumeCube"
    vol_cube.data.materials.append(
        _make_heterogeneous_blackbody_volume_material("HeterogeneousBlackbodyVolume")
    )

    # 6. Back-right: subdivided smooth Suzanne with chromatic Random-Walk SSS.
    bpy.ops.mesh.primitive_monkey_add(size=1.20, location=(1.35, 1.30, 0.68))
    suzanne = bpy.context.active_object
    suzanne.name = "RandomWalkSuzanne"
    suzanne.rotation_euler = (0.0, 0.0, math.radians(-18.0))
    subsurf = suzanne.modifiers.new(name="Subsurf", type="SUBSURF")
    subsurf.levels = 2
    subsurf.render_levels = 2
    bpy.context.view_layer.objects.active = suzanne
    bpy.ops.object.modifier_apply(modifier=subsurf.name)
    _set_smooth(suzanne)
    suzanne.data.materials.append(_make_random_walk_sss_material("ChromaticRandomWalkSSS"))

    # 7. Sharp caustics key area light + warm backlight for SSS & volume.
    key_data = bpy.data.lights.new(name="CausticKeyLight", type="AREA")
    key_data.shape = "SQUARE"
    key_data.size = 0.28
    key_data.energy = 110.0
    key_data.spread = math.pi
    key_data.color = (1.0, 0.98, 0.95)
    key_data.cycles.use_multiple_importance_sampling = True
    key_data.cycles.is_caustics_light = True
    key_obj = bpy.data.objects.new(name="CausticKeyLight", object_data=key_data)
    scene.collection.objects.link(key_obj)
    key_obj.location = (0.05, -2.35, 2.35)
    _aim_object(key_obj, (0.0, 0.05, 0.30))

    back_data = bpy.data.lights.new(name="WarmBacklight", type="AREA")
    back_data.shape = "SQUARE"
    back_data.size = 1.2
    back_data.energy = 240.0
    back_data.color = (1.0, 0.95, 0.90)
    back_data.cycles.use_multiple_importance_sampling = True
    back_data.cycles.is_caustics_light = False
    back_obj = bpy.data.objects.new(name="WarmBacklight", object_data=back_data)
    scene.collection.objects.link(back_obj)
    back_obj.location = (0.50, 3.20, 1.55)
    _aim_object(back_obj, (0.40, 1.20, 0.65))

    # 8. Camera framing the full kitchen-sink tabletop.
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 36.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -4.85, 2.55)
    _aim_object(cam_obj, (0.0, 0.30, 0.42))
    scene.camera = cam_obj

    scene.render.engine = "CYCLES"
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 8
    scene.cycles.volume_bounces = 4
    scene.cycles.volume_step_rate = 0.5
    scene.cycles.volume_max_steps = 256
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True
    scene.cycles.blur_glossy = 0.0
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0

    blend_path = out_dir / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))


if __name__ == "__main__":
    main()
