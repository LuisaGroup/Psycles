"""Create original-Cycles typed texture-input and shared-conversion probes.

Run with Blender 5.2.1 -- /new/probe.blend, then use original Cycles HIP to
render/dump and export_psycles_scene.py to retain the unchanged input graphs.
This script creates graphs only; it evaluates no shading or sampling code.
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
    for kind in ("Noise", "WhiteNoise", "Gradient", "Gabor", "Voronoi",
                 "Wave", "Magic", "Checker", "Brick"):
        for mode in ("scalar-single", "scalar-shared", "scalar-bump"):
            material = bpy.data.materials.new(f"{kind}-{mode}")
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
            info = tree.nodes.new("ShaderNodeObjectInfo")
            texture = tree.nodes.new("ShaderNodeTex" + kind)
            tree.links.new(info.outputs["Random"], texture.inputs["Vector"])
            scalar = next(s for s in texture.outputs if s.type == "VALUE" and s.enabled)
            if mode == "scalar-single":
                tree.links.new(scalar, diffuse.inputs["Color"])
            elif mode == "scalar-shared":
                tree.links.new(info.outputs["Random"], diffuse.inputs["Color"])
                tree.links.new(scalar, diffuse.inputs["Roughness"])
            else:
                mix = tree.nodes.new("ShaderNodeMixRGB")
                mix.blend_type = "MULTIPLY"
                mix.inputs[0].default_value = 0.5
                tree.links.new(info.outputs["Random"], mix.inputs[1])
                tree.links.new(scalar, mix.inputs[2])
                bump = tree.nodes.new("ShaderNodeBump")
                tree.links.new(mix.outputs[0], bump.inputs["Height"])
                tree.links.new(bump.outputs["Normal"], diffuse.inputs["Normal"])
            tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
            bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 9, index // 9, 0))
            bpy.context.object.data.materials.append(material)
            index += 1
    bpy.ops.object.camera_add(location=(4, 1, 7))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 10
    scene.render.resolution_x = 36
    scene.render.resolution_y = 12
    scene.render.resolution_percentage = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
