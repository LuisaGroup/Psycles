"""Author portal integration inputs; expected films come only from Cycles HIP.

Run with Blender --background --python this_file -- destination.
"""
from pathlib import Path
import sys
import bpy

destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
destination.mkdir(parents=True, exist_ok=True)
cases = ("portal-depth-world", "portal-depth-surface", "portal-depth-chain",
         "portal-signed-weight", "portal-transparent-mix", "portal-limit-zero",
         "portal-depth-nee", "portal-depth-shadow", "portal-default-position")
selected = sys.argv[sys.argv.index("--") + 2:] or cases
assert all(case in cases for case in selected), selected


def depth_color(tree, socket, scale=1.0):
    path = tree.nodes.new("ShaderNodeLightPath")
    output = path.outputs["Portal Depth"]
    if scale != 1.0:
        multiply = tree.nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        multiply.inputs[1].default_value = scale
        tree.links.new(output, multiply.inputs[0])
        output = multiply.outputs[0]
    tree.links.new(output, socket)


def plane(name, z, destination_z=None, signed=False, mixed=False, scale=1.0, relocate=True):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    if destination_z is None:
        terminal = tree.nodes.new("ShaderNodeEmission")
        depth_color(tree, terminal.inputs["Color"], scale)
    else:
        terminal = tree.nodes.new("ShaderNodeBsdfRayPortal")
        terminal.inputs["Position"].default_value = (0, 0, destination_z)
        if relocate:
            # Position is LINK_POSITION: an unlinked numeric default is
            # deliberately ignored by Cycles. A real wire is required to
            # test relocation rather than a direction-only portal.
            position = tree.nodes.new("ShaderNodeCombineXYZ")
            for socket, value in zip(position.inputs, (0.125, -0.0625, destination_z)):
                socket.default_value = value
            tree.links.new(position.outputs[0], terminal.inputs["Position"])
        terminal.inputs["Direction"].default_value = (0, 0, -1)
        if signed:
            color = tree.nodes.new("ShaderNodeCombineXYZ")
            for socket, value in zip(color.inputs, (-0.25, 0.5, 0.75)):
                socket.default_value = value
            tree.links.new(color.outputs[0], terminal.inputs["Color"])
        if mixed:
            transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
            mix = tree.nodes.new("ShaderNodeMixShader")
            mix.inputs[0].default_value = 0.35
            tree.links.new(transparent.outputs[0], mix.inputs[1])
            tree.links.new(terminal.outputs[0], mix.inputs[2])
            terminal = mix
    tree.links.new(terminal.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, z))
    bpy.context.object.name = name
    bpy.context.object.data.materials.append(material)


for case in selected:
    target = destination / (case + ".blend")
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
    if case in ("portal-depth-nee", "portal-depth-shadow"):
        scene.cycles.max_bounces = 2
    scene.cycles.min_light_bounces = scene.cycles.min_transparent_bounces = 0
    scene.cycles.transparent_max_bounces = 0 if case == "portal-limit-zero" else 8
    scene.cycles.volume_bounces = 0
    scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
    scene.cycles.blur_glossy = 0
    world = bpy.data.worlds.new("Portal world")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1, 1, 1, 1)
    background.inputs["Strength"].default_value = 1
    if case in ("portal-depth-nee", "portal-depth-shadow"):
        background.inputs["Strength"].default_value = 0
    elif case not in ("portal-signed-weight", "portal-limit-zero"):
        depth_color(world.node_tree, background.inputs["Color"])
    world.cycles.sampling_method = "NONE"
    scene.world = world
    plane("Entry", 3, 2 if case == "portal-depth-chain" else 1,
          signed=case == "portal-signed-weight", mixed=case == "portal-transparent-mix",
          relocate=case != "portal-default-position")
    if case == "portal-depth-surface":
        plane("Depth emission", 0)
    if case == "portal-depth-chain":
        plane("Second portal", 1, 0)
    if case == "portal-transparent-mix":
        plane("Transparent-only endpoint", 2, scale=0.25)
    if case in ("portal-depth-nee", "portal-depth-shadow"):
        receiver = bpy.data.materials.new("Diffuse receiver")
        receiver.use_nodes = True
        tree = receiver.node_tree
        tree.nodes.clear()
        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
        bpy.context.object.data.materials.append(receiver)
        light_data = bpy.data.lights.new("Portal depth emitter", "POINT")
        light_data.use_nodes = True
        light_data.shadow_soft_size = 0
        emission = light_data.node_tree.nodes.get("Emission")
        assert emission is not None
        emission.inputs["Strength"].default_value = 10
        depth_color(light_data.node_tree, emission.inputs["Color"])
        light = bpy.data.objects.new(light_data.name, light_data)
        scene.collection.objects.link(light)
        light.location = (0, 0, 2)
        if case == "portal-depth-shadow":
            shadow = bpy.data.materials.new("Portal depth transparency")
            shadow.use_nodes = True
            tree = shadow.node_tree
            tree.nodes.clear()
            transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            depth_color(tree, transparent.inputs["Color"], 0.5)
            tree.links.new(transparent.outputs[0], output.inputs["Surface"])
            bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 1.5))
            bpy.context.object.data.materials.append(shadow)
    camera_data = bpy.data.cameras.new("Orthographic camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, 0, 5)
    if case == "portal-default-position":
        # Exercise the unlinked-position contract without deliberately
        # putting a whole pixel diagonal on an equal-distance triangle tie.
        # The original tied-edge captures and GPU boundary inputs are retained
        # separately; no renderer intersection rule or tolerance is changed.
        camera.location = (0.113, -0.057, 5)
    scene.camera = camera
    bpy.ops.wm.save_as_mainfile(filepath=str(target), check_existing=True)
    print("PORTAL_INPUT", target, flush=True)
