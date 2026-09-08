"""Create original Blender default-coordinate phase and sharing controls."""
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
    for kind in ("Checker", "Gradient"):
        for explicit in (False, True):
            for bump in (False, True):
                for shared in (False, True):
                    for reverse in (False, True):
                        name = f"{kind}-explicit{int(explicit)}-bump{int(bump)}-shared{int(shared)}-reverse{int(reverse)}"
                        material = bpy.data.materials.new(name)
                        material.use_nodes = True
                        tree = material.node_tree
                        tree.nodes.clear()
                        output = tree.nodes.new("ShaderNodeOutputMaterial")
                        diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
                        uv = tree.nodes.new("ShaderNodeUVMap")
                        mapping = tree.nodes.new("ShaderNodeMapping")
                        mapping.inputs["Scale"].default_value = (2, 1, 0.5)
                        mapped = tree.nodes.new("ShaderNodeTexChecker")
                        tree.links.new(uv.outputs[0], mapping.inputs["Vector"])
                        tree.links.new(mapping.outputs[0], mapped.inputs["Vector"])
                        default = tree.nodes.new("ShaderNodeTex" + kind)
                        if kind == "Checker":
                            default.inputs["Scale"].default_value = 3
                        if explicit:
                            coords = tree.nodes.new("ShaderNodeTexCoord")
                            tree.links.new(coords.outputs["Generated"], default.inputs["Vector"])
                        mix = tree.nodes.new("ShaderNodeMix")
                        mix.data_type = "RGBA"
                        mix.inputs[0].default_value = 0.375
                        tree.links.new(default.outputs["Color"], mix.inputs[7 if reverse else 6])
                        tree.links.new(mapped.outputs["Color"], mix.inputs[6 if reverse else 7])
                        if shared:
                            tree.links.new(default.outputs["Color"], diffuse.inputs["Color"])
                        if bump:
                            normal = tree.nodes.new("ShaderNodeBump")
                            tree.links.new(mix.outputs[2], normal.inputs["Height"])
                            tree.links.new(normal.outputs[0], diffuse.inputs["Normal"])
                        else:
                            tree.links.new(mix.outputs[2], diffuse.inputs["Roughness"])
                        tree.links.new(diffuse.outputs[0], output.inputs["Surface"])
                        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 8, index // 8, 0))
                        bpy.context.object.data.materials.append(material)
                        index += 1
    bpy.ops.object.camera_add(location=(3.5, 1.5, 10))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 9
    scene.render.resolution_x = 32
    scene.render.resolution_y = 16
    scene.render.resolution_percentage = 100
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
