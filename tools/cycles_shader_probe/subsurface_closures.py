"""Closed-geometry Cycles BSSRDF transport probes."""

from __future__ import annotations

from typing import Any

import bpy

from .support import _input, _material, _output, _world


def _subsurface_material(
    name: str,
    method: str,
    color: tuple[float, float, float],
    radius: tuple[float, float, float],
    scale: float,
    roughness: float,
    ior: float,
    anisotropy: float,
) -> Any:
    material, tree, output = _material(name)
    bssrdf = tree.nodes.new("ShaderNodeSubsurfaceScattering")
    bssrdf.name = f"{name} BSSRDF"
    bssrdf.falloff = method
    _input(bssrdf, "Color").default_value = (*color, 1.0)
    _input(bssrdf, "Radius").default_value = radius
    _input(bssrdf, "Scale").default_value = scale
    if method != "RANDOM_WALK_SKIN":
        _input(bssrdf, "Roughness").default_value = roughness
    _input(bssrdf, "IOR").default_value = ior
    _input(bssrdf, "Anisotropy").default_value = anisotropy
    tree.links.new(
        _output(bssrdf, "BSSRDF"),
        _input(output, "Surface"),
    )
    return material


def _random_walk_transport(scene: Any) -> None:
    """Exercise all current Random Walk mappings in closed geometry."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 4
    scene.cycles.light_sampling_threshold = 0.0
    scene.cycles.use_light_tree = False
    _world(scene, (0.025, 0.045, 0.09, 1.0), 0.3)

    cases = (
        (
            "Random Walk",
            "RANDOM_WALK",
            (0.72, 0.24, 0.08),
            (1.0, 0.45, 0.2),
            0.16,
            0.35,
            1.4,
            0.45,
        ),
        (
            "Random Walk Negative G",
            "RANDOM_WALK",
            (0.12, 0.52, 0.72),
            (0.35, 0.8, 1.0),
            0.18,
            0.62,
            1.33,
            -0.55,
        ),
        (
            "Random Walk Legacy",
            "RANDOM_WALK_LEGACY",
            (0.18, 0.7, 0.2),
            (0.6, 1.0, 0.32),
            1.4,
            0.48,
            1.5,
            0.55,
        ),
        (
            "Random Walk Skin",
            "RANDOM_WALK_SKIN",
            (0.68, 0.22, 0.1),
            (1.0, 0.5, 0.25),
            0.18,
            0.08,
            1.4,
            0.2,
        ),
    )
    positions = (
        (-0.55, 0.55, 0.0),
        (0.55, 0.55, 0.0),
        (-0.55, -0.55, 0.0),
        (0.55, -0.55, 0.0),
    )
    for case, position in zip(cases, positions, strict=True):
        material = _subsurface_material(*case)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64,
            ring_count=32,
            radius=0.42,
            enter_editmode=False,
            align="WORLD",
            location=position,
        )
        sphere = bpy.context.object
        sphere.name = case[0]
        sphere.data.materials.append(material)
        for polygon in sphere.data.polygons:
            polygon.use_smooth = True

    key_data = bpy.data.lights.new("Random Walk Key", type="AREA")
    key_data.color = (0.38, 0.63, 1.0)
    key_data.energy = 180.0
    key_data.shape = "DISK"
    key_data.size = 1.3
    key_data.normalize = True
    key = bpy.data.objects.new(key_data.name, key_data)
    key.location = (-0.4, -0.25, 1.8)
    scene.collection.objects.link(key)

    back_data = bpy.data.lights.new("Random Walk Backlight", type="AREA")
    back_data.color = (1.0, 0.24, 0.06)
    back_data.energy = 90.0
    back_data.shape = "DISK"
    back_data.size = 0.8
    back_data.normalize = True
    back = bpy.data.objects.new(back_data.name, back_data)
    back.location = (0.45, 0.2, -1.3)
    back.rotation_euler = (3.141592653589793, 0.0, 0.0)
    scene.collection.objects.link(back)
