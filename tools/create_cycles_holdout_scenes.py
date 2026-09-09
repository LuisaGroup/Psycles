"""Author original Blender inputs; no expected transport or pixels are computed."""
from pathlib import Path
import sys
import bpy

destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
destination.mkdir(parents=True, exist_ok=True)
CASES = (
    "ordinary-emission", "object-emission", "object-transparent",
    "object-mixed-emission", "object-mixed-diffuse", "object-colored-transparent",
    "object-mixed-opaque", "object-secondary-emission", "node-holdout",
    "node-mixed-emission", "node-mixed-transparent", "node-dynamic-holdout",
)
for name in CASES:
    target = destination / (name + ".blend")
    assert not target.exists(), target
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 32
    scene.render.resolution_x = scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = name != "object-mixed-opaque"
    scene.cycles.samples = 1
    scene.cycles.seed = 0
    scene.cycles.use_denoising = scene.cycles.use_adaptive_sampling = False
    scene.cycles.sampling_pattern = "TABULATED_SOBOL"
    scene.cycles.use_light_tree = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.cycles.max_bounces = 1 if name == "object-secondary-emission" else 0
    scene.cycles.min_light_bounces = scene.cycles.min_transparent_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.volume_bounces = 0
    scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.blur_glossy = 0.0
    world = bpy.data.worlds.new("Background")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.4, 0.6, 0.8, 1)
    world.cycles.sampling_method = "NONE"
    scene.world = world

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.cycles.emission_sampling = "NONE"
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1, 0.5, 0.25, 1)
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    holdout = tree.nodes.new("ShaderNodeHoldout")
    if name == "object-colored-transparent":
        transparent.inputs["Color"].default_value = (0.25, 0.5, 0.75, 1)
    if name in ("object-transparent", "object-colored-transparent"):
        terminal = transparent
    elif name == "node-holdout":
        terminal = holdout
    elif "mixed" in name or name == "node-dynamic-holdout":
        terminal = tree.nodes.new("ShaderNodeMixShader")
        terminal.inputs[0].default_value = 0.25
        first = holdout if name.startswith("node-") else transparent
        second = (transparent if name == "node-mixed-transparent" else
                  diffuse if name == "object-mixed-diffuse" else emission)
        tree.links.new(first.outputs[0], terminal.inputs[1])
        tree.links.new(second.outputs[0], terminal.inputs[2])
        if name == "node-dynamic-holdout":
            geometry = tree.nodes.new("ShaderNodeNewGeometry")
            separate = tree.nodes.new("ShaderNodeSeparateXYZ")
            shift = tree.nodes.new("ShaderNodeMath")
            shift.operation = "ADD"
            shift.inputs[1].default_value = 0.5
            tree.links.new(geometry.outputs["Position"], separate.inputs[0])
            tree.links.new(separate.outputs["X"], shift.inputs[0])
            tree.links.new(shift.outputs[0], terminal.inputs[0])
    else:
        terminal = emission
    tree.links.new(terminal.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=100.0 if name == "object-secondary-emission" else 4.0,
                                   location=(0, 0, 6 if name == "object-secondary-emission" else 3))
    plane = bpy.context.object
    plane.name = name
    plane.is_holdout = name.startswith("object-")
    plane.data.materials.append(material)
    if name == "object-secondary-emission":
        receiver = bpy.data.materials.new("Diffuse receiver")
        receiver.use_nodes = True
        receiver.node_tree.nodes.clear()
        shader = receiver.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        receiver_output = receiver.node_tree.nodes.new("ShaderNodeOutputMaterial")
        receiver.node_tree.links.new(shader.outputs[0], receiver_output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 3))
        bpy.context.object.data.materials.append(receiver)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.0
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0.113, -0.057, 5)
    scene.camera = camera
    bpy.ops.wm.save_as_mainfile(filepath=str(target), check_existing=True)
    print("HOLDOUT_SCENE", target, flush=True)
