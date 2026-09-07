"""Build an unmodified Cycles graph for the read-only GDB budget observer.

blender --background --factory-startup --python probe.py -- CASE
No Psycles renderer, shading bake, or replacement closure evaluator is used.
"""

import sys

import bpy


case = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add()
mesh = bpy.context.object
material = bpy.data.materials.new("Closure budget: " + case)
material.use_nodes = True
mesh.data.materials.append(material)
nodes = material.node_tree.nodes
links = material.node_tree.links
nodes.clear()
output = nodes.new("ShaderNodeOutputMaterial")


def node(kind):
    return nodes.new("ShaderNode" + kind)


socket = "Surface"
if case in {"diffuse", "diffuse-emission", "folded-principled", "shared-diffuse"}:
    diffuse = node("BsdfDiffuse")
    root = diffuse.outputs[0]
    if case in {"diffuse-emission", "folded-principled"}:
        emission = node("Emission")
        add = node("AddShader")
        links.new(root, add.inputs[0])
        links.new(emission.outputs[0], add.inputs[1])
        root = add.outputs[0]
    if case == "folded-principled":
        principled = node("BsdfPrincipled")
        mix = node("MixShader")
        mix.inputs[0].default_value = 0.0
        links.new(root, mix.inputs[1])
        links.new(principled.outputs[0], mix.inputs[2])
        root = mix.outputs[0]
    if case == "shared-diffuse":
        add = node("AddShader")
        links.new(root, add.inputs[0])
        links.new(root, add.inputs[1])
        root = add.outputs[0]
elif case in {"emission", "background", "principled", "bssrdf", "conductor", "hair"}:
    root = node({"emission": "Emission", "principled": "BsdfPrincipled",
                 "background": "Background",
                 "bssrdf": "SubsurfaceScattering", "conductor": "BsdfMetallic",
                 "hair": "BsdfHair"}[case]).outputs[0]
elif case.startswith(("glossy-", "glass-")):
    family, distribution = case.split("-", 1)
    bsdf = node("BsdfAnisotropic" if family == "glossy" else "BsdfGlass")
    bsdf.distribution = distribution
    root = bsdf.outputs[0]
elif case in {"absorption", "scatter", "volume-cap"}:
    socket = "Volume"
    root = node("VolumeScatter" if case == "scatter" else "VolumeAbsorption").outputs[0]
    if case == "volume-cap":
        for index in range(1, 3):
            scatter = node("VolumeScatter")
            scatter.inputs["Color"].default_value = (index * 0.25, 0.5, 0.75, 1.0)
            add = node("AddShader")
            links.new(root, add.inputs[0])
            links.new(scatter.outputs[0], add.inputs[1])
            root = add.outputs[0]
else:
    raise ValueError(case)
links.new(root, output.inputs[socket])

bpy.ops.object.camera_add(location=(0.0, 0.0, 2.0))
scene.camera = bpy.context.object
scene.render.engine = "CYCLES"
scene.render.resolution_x = 8
scene.render.resolution_y = 8
scene.render.resolution_percentage = 100
scene.cycles.samples = 1
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.device = "GPU"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "HIP"
preferences.get_devices()
selected = []
for device in preferences.devices:
    device.use = device.type == "HIP" and "9070 XT" in device.name
    if device.use:
        selected.append(device.name)
if len(selected) != 1:
    raise RuntimeError(f"expected one RX 9070 XT HIP device, got {selected}")
print(f"CLOSURE_BUDGET_CASE {case} device={selected[0]}", flush=True)
bpy.ops.render.render()
