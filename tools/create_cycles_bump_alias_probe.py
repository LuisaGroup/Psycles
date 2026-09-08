"""Create original-Cycles regressions for automatic bump graph edge ownership.

Run with Blender 5.2.1 --background --python this_file -- /new/probe.blend.
The surface reads Geometry.Normal through the same float3 aliases as the
automatic displacement dot products. No shader is evaluated by this script.
"""
from pathlib import Path
import sys

import bpy


def main():
    path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if path.exists():
        raise FileExistsError(path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    index = 0
    for mode in ("normal-only", "color", "vector-math", "fresnel"):
        for displacement in ("NONE", "BUMP", "BOTH"):
            material = bpy.data.materials.new(f"{mode}-{displacement}")
            material.use_nodes = True
            material.displacement_method = "BUMP" if displacement == "NONE" else displacement
            tree = material.node_tree
            tree.nodes.clear()
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
            diffuse.inputs["Roughness"].default_value = 0.25
            geometry = tree.nodes.new("ShaderNodeNewGeometry")
            normal = geometry.outputs["Normal"]
            if mode == "normal-only":
                tree.links.new(normal, diffuse.inputs["Normal"])
            elif mode == "color":
                tree.links.new(normal, diffuse.inputs["Color"])
            elif mode == "vector-math":
                math = tree.nodes.new("ShaderNodeVectorMath")
                math.operation = "SCALE"
                math.inputs["Scale"].default_value = 0.75
                tree.links.new(normal, math.inputs[0])
                tree.links.new(math.outputs["Vector"], diffuse.inputs["Color"])
            else:
                mix = tree.nodes.new("ShaderNodeMix")
                mix.data_type = "RGBA"
                mix.inputs[0].default_value = 0.25
                tree.links.new(normal, mix.inputs[6])
                tree.links.new(geometry.outputs["Incoming"], mix.inputs[7])
                fresnel = tree.nodes.new("ShaderNodeFresnel")
                tree.links.new(mix.outputs[2], fresnel.inputs["Normal"])
                tree.links.new(fresnel.outputs[0], diffuse.inputs["Color"])
            tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
            if displacement != "NONE":
                disp = tree.nodes.new("ShaderNodeDisplacement")
                disp.space = "OBJECT"
                disp.inputs["Height"].default_value = 0.25
                disp.inputs["Midlevel"].default_value = 0.0
                disp.inputs["Scale"].default_value = 0.125
                tree.links.new(disp.outputs[0], output.inputs["Displacement"])
            bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 4, index // 4, 0))
            bpy.context.object.data.materials.append(material)
            index += 1
    bpy.ops.object.camera_add(location=(1.5, 1, 5))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 5
    scene.render.resolution_x = 16
    scene.render.resolution_y = 12
    scene.render.resolution_percentage = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
