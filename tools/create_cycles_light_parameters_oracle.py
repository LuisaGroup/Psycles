"""Run with the observer Blender; dump native scene parameters at boundary inputs.

blender --background --factory-startup --python-exit-code 1 --python THIS -- OUTPUT.blend
PSYCLES_CYCLES_LIGHT_PARAMETERS_DUMP selects the native dump, not this script.
"""
from pathlib import Path
import math
import struct
import sys

import bpy


def adjacent(value, delta):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + delta))[0]


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "HIP"
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == "HIP"
if not any(d.use for d in prefs.devices):
    raise RuntimeError("No original Cycles HIP device")


def light(name, kind, scale):
    data = bpy.data.lights.new(name, kind)
    data.energy = 20
    data.normalize = True
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = (0, 0, 2)
    obj.scale = scale
    return data


# Include full spread, both sides of the exact pi and Taylor predicates,
# zero, and ordinary angles. No expected values are calculated here.
spreads = [0.0, 0.001, adjacent(0.1, -1), 0.1, adjacent(0.1, 1),
           1.2, adjacent(math.pi, -1), math.pi]
for i, spread in enumerate(spreads):
    for ellipse in (False, True):
        data = light(f"area_{i}_{int(ellipse)}", "AREA", (2, 0.5, 3))
        data.shape = "ELLIPSE" if ellipse else "RECTANGLE"
        data.size, data.size_y, data.spread = 0.8, 0.5, spread

# Blender's spot_size RNA clamps zero to its documented minimum; the dump
# preserves the effective Cycles input rather than pretending it was zero.
spots = [(0.0, 0.0, 0.0, (1, 1, 1)),
         (0.02, 0.0, 0.25, (1, 1, 1)),
         (0.6, 0.0, 0.5, (1, 1, 1)),
         (0.6, 0.5, 0.5, (1, 1, 1)),
         (0.6, 1.0, 0.5, (2, 0.5, 3)),
         (1.0471975803, 1.0, 0.25, (1, 1, 1)),
         (2.5464353561, 1.0, 0.1, (1, 1, 1)),
         (math.pi, 0.15, 0.5, (0.5, 2, 1)),
         (0.6, 0.5, 0.0, (-2, 0.5, 3))]
for i, (angle, smooth, radius, scale) in enumerate(spots):
    data = light(f"spot_{i}", "SPOT", scale)
    data.spot_size, data.spot_blend = angle, smooth
    data.shadow_soft_size = radius
camera = bpy.data.cameras.new("camera")
obj = bpy.data.objects.new("camera", camera)
obj.location = (0, 0, 5)
scene.collection.objects.link(obj)
scene.camera = obj
bpy.ops.mesh.primitive_plane_add(size=4)
bpy.context.view_layer.update()
output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
bpy.ops.wm.save_as_mainfile(filepath=str(output))
bpy.ops.render.render()
