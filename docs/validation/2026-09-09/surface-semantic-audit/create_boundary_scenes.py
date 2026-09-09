"""Create fresh original-renderer inputs for the path-bound audit.

This creates scenes only; expected pixels always come from original Cycles.
One absorbing slab is the positive control; two slabs require four volume
crossings plus the terminal background, without an ordinary/transparent bounce.
"""
from pathlib import Path
import sys
import bpy

destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
destination.mkdir(parents=True, exist_ok=True)
for count, transparent_budget, name in (
        (1, 0, "boundary-1"), (2, 0, "boundary-2"),
        (2, 2, "boundary-2-budget-control")):
    target = destination / f"{name}.blend"
    assert not target.exists(), target
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 32
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 1
    scene.cycles.seed = 0
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    scene.cycles.use_light_tree = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    for field in ("max_bounces", "min_light_bounces", "diffuse_bounces",
                  "glossy_bounces", "transmission_bounces", "volume_bounces",
                  "min_transparent_bounces", "transparent_max_bounces"):
        setattr(scene.cycles, field, 0)
    scene.cycles.transparent_max_bounces = transparent_budget
    scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.blur_glossy = 0.0
    world = bpy.data.worlds.new("White background")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    world.cycles.sampling_method = "NONE"
    scene.world = world
    material = bpy.data.materials.new("Volume only; no surface terminal")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    absorption.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    absorption.inputs["Density"].default_value = 0.125
    material.node_tree.links.new(absorption.outputs["Volume"], output.inputs["Volume"])
    for index in range(count):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 3 - 2 * index))
        obj = bpy.context.object
        obj.name = f"Absorbing slab {index}"
        obj.scale = (4.0, 4.0, 1.0)
        obj.data.materials.append(material)
    camera_data = bpy.data.cameras.new("Orthographic camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, 0, 5)
    scene.camera = camera
    bpy.ops.wm.save_as_mainfile(filepath=str(target), check_existing=True)
    print(f"AUDIT_SCENE {target}", flush=True)
