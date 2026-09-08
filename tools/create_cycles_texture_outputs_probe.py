"""Create original-Cycles multi-output procedural texture regressions.

Run in Blender 5.2.1 with -- /new/probe.blend, render/dump with the original
Cycles HIP backend, then export the same scene. This script evaluates no nodes.
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
    for kind in ("Noise", "WhiteNoise", "Magic", "Wave"):
        for mode in ("color-only", "dual-surface", "dual-bump"):
            material = bpy.data.materials.new(f"{kind}-{mode}")
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
            geometry = tree.nodes.new("ShaderNodeNewGeometry")
            texture = tree.nodes.new("ShaderNodeTex" + kind)
            tree.links.new(geometry.outputs["Position"], texture.inputs["Vector"])
            tree.links.new(texture.outputs["Color"], diffuse.inputs["Color"])
            if mode != "color-only":
                factor = texture.outputs["Value" if kind == "WhiteNoise" else "Fac"]
                if mode == "dual-surface":
                    tree.links.new(factor, diffuse.inputs["Roughness"])
                else:
                    bump = tree.nodes.new("ShaderNodeBump")
                    tree.links.new(factor, bump.inputs["Height"])
                    tree.links.new(bump.outputs["Normal"], diffuse.inputs["Normal"])
            tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
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
