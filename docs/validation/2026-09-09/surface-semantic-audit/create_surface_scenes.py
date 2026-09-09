"""Generate original renderer inputs, not expected shader or pixel values.

Ordinary emission vs object holdout isolates the missing object-mask consumer;
transparent vs portal isolates the production bounce dispatch beyond eval_nodes.
"""
from pathlib import Path
import sys
import bpy

destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
destination.mkdir(parents=True, exist_ok=True)
for case in ("ordinary-emission", "object-holdout", "transparent-control", "ray-portal"):
    target = destination / f"{case}.blend"
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
    scene.cycles.max_bounces = 0
    scene.cycles.min_light_bounces = 0
    scene.cycles.min_transparent_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.volume_bounces = 0
    scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.blur_glossy = 0.0
    scene.render.film_transparent = case in ("ordinary-emission", "object-holdout")
    world = bpy.data.worlds.new("White background")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    world.cycles.sampling_method = "NONE"
    scene.world = world
    material = bpy.data.materials.new(case)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    if case in ("ordinary-emission", "object-holdout"):
        terminal = nodes.new("ShaderNodeEmission")
        terminal.inputs["Color"].default_value = (1, 1, 1, 1)
        terminal.inputs["Strength"].default_value = 1.0
    elif case == "transparent-control":
        terminal = nodes.new("ShaderNodeBsdfTransparent")
    else:
        terminal = nodes.new("ShaderNodeBsdfRayPortal")
        terminal.inputs["Color"].default_value = (1, 1, 1, 1)
        terminal.inputs["Position"].default_value = (0, 0, 0)
        terminal.inputs["Direction"].default_value = (1, 0, 0)
    material.node_tree.links.new(terminal.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 3))
    obj = bpy.context.object
    obj.name = case
    obj.is_holdout = case == "object-holdout"
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
