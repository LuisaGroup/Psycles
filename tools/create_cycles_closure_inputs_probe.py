"""Create original-Cycles closure declaration-order regression graphs."""
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
    kinds = ("BsdfDiffuse", "BsdfAnisotropic", "BsdfGlass", "BsdfRefraction",
             "BsdfToon", "BsdfSheen", "BsdfMetallic", "BsdfTranslucent",
             "BsdfTransparent", "BsdfRayPortal", "BsdfHair", "BsdfPrincipled",
             "SubsurfaceScattering")
    index = 0
    for kind in kinds:
        for mode in ("defaults", "linked"):
            material = bpy.data.materials.new(f"{kind}-{mode}")
            material.use_nodes = True
            tree = material.node_tree
            tree.nodes.clear()
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            closure = tree.nodes.new("ShaderNode" + kind)
            if mode == "linked":
                info = tree.nodes.new("ShaderNodeObjectInfo")
                geometry = tree.nodes.new("ShaderNodeNewGeometry")
                for i, socket in enumerate(closure.inputs):
                    if not socket.enabled or socket.type == "SHADER":
                        continue
                    if socket.type in {"VALUE", "BOOLEAN", "INT"}:
                        math = tree.nodes.new("ShaderNodeMath")
                        math.operation = "ADD"
                        math.inputs[1].default_value = (i + 1) / 64
                        tree.links.new(info.outputs["Random"], math.inputs[0])
                        tree.links.new(math.outputs[0], socket)
                    elif socket.type == "RGBA":
                        mix = tree.nodes.new("ShaderNodeMixRGB")
                        mix.inputs[0].default_value = (i + 1) / 64
                        mix.inputs[2].default_value = (0.25, 0.5, 0.75, 1)
                        tree.links.new(info.outputs["Random"], mix.inputs[1])
                        tree.links.new(mix.outputs[0], socket)
                    elif socket.type == "VECTOR":
                        tree.links.new(geometry.outputs["Normal"], socket)
            tree.links.new(closure.outputs[0], output.inputs["Surface"])
            bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 13, index // 13, 0))
            bpy.context.object.data.materials.append(material)
            index += 1
    bpy.ops.object.camera_add(location=(6, 0.5, 9))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 14
    scene.render.resolution_x = 52
    scene.render.resolution_y = 8
    scene.render.resolution_percentage = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
