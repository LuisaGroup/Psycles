"""Author three native Holdout domain probes; no words or transport are synthesized."""
from pathlib import Path
import sys
import bpy

target = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
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
scene.cycles.use_denoising = scene.cycles.use_adaptive_sampling = False
scene.cycles.use_light_tree = False
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 1.0
scene.cycles.max_bounces = scene.cycles.volume_bounces = 0
scene.cycles.transparent_max_bounces = 8
scene.cycles.min_light_bounces = scene.cycles.min_transparent_bounces = 0

names = ("holdout-only-volume", "holdout-emission-volume", "holdout-shared-domains")
for index, name in enumerate(names):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.cycles.emission_sampling = "NONE"
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    holdout = tree.nodes.new("ShaderNodeHoldout")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.25, 0.5, 1.0, 1.0)
    if name == "holdout-only-volume":
        tree.links.new(holdout.outputs[0], output.inputs["Volume"])
    else:
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 0.25
        tree.links.new(holdout.outputs[0], mix.inputs[1])
        tree.links.new(emission.outputs[0], mix.inputs[2])
        tree.links.new(mix.outputs[0], output.inputs["Volume"])
        if name == "holdout-shared-domains":
            tree.links.new(holdout.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(index - 1, 0, 0))
    bpy.context.object.name = name
    bpy.context.object.data.materials.append(material)

camera_data = bpy.data.cameras.new("Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.113, -0.057, 5.0)
scene.camera = camera
bpy.ops.wm.save_as_mainfile(filepath=str(target), check_existing=True)
