"""Create original-Cycles shared-closure Add/Mix compiler probes.

Use Blender 5.2.1 --background --python this_file -- /new/probe.blend.
No shader is evaluated here and no expected SVM stream is synthesized.
"""

from pathlib import Path
import sys

import bpy


def main() -> None:
    destination = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.use_light_tree = False
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True

    for index, (volume, nested) in enumerate(((False, False), (False, True),
                                               (True, False), (True, True))):
        name = ("Volume" if volume else "Surface") + ("Mix" if nested else "Add")
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        graph = material.node_tree
        graph.nodes.clear()

        def create(kind: str, label: str):
            node = graph.nodes.new(kind)
            node.name = label
            return node

        if nested:
            other = create("ShaderNodeVolumeScatter" if volume else
                           "ShaderNodeBsdfDiffuse", "00 Other")
            other.inputs["Color"].default_value = (0.5, 0.25, 0.125, 1.0)
            if volume:
                other.inputs["Density"].default_value = 0.75
        shared = create("ShaderNodeVolumeAbsorption" if volume else
                        "ShaderNodeBsdfTransparent", "01 Shared")
        shared.inputs["Color"].default_value = (0.25, 0.5, 0.75, 1.0)
        if volume:
            shared.inputs["Density"].default_value = 2.0
        if nested:
            light_path = create("ShaderNodeLightPath", "02 Light Path")
            inner = create("ShaderNodeMixShader", "03 Inner")
            root = create("ShaderNodeMixShader", "04 Root")
            for mix in (inner, root):
                graph.links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
            graph.links.new(other.outputs[0], inner.inputs[1])
            graph.links.new(shared.outputs[0], inner.inputs[2])
            graph.links.new(shared.outputs[0], root.inputs[1])
            graph.links.new(inner.outputs[0], root.inputs[2])
        else:
            root = create("ShaderNodeAddShader", "04 Root")
            graph.links.new(shared.outputs[0], root.inputs[0])
            graph.links.new(shared.outputs[0], root.inputs[1])
        output = create("ShaderNodeOutputMaterial", "05 Output")
        graph.links.new(root.outputs[0], output.inputs["Volume" if volume else "Surface"])
        bpy.ops.mesh.primitive_cube_add(size=0.75, location=(index % 2 - 0.5,
                                                          index // 2 - 0.5, 0.0))
        bpy.context.object.data.materials.append(material)

    camera = bpy.data.cameras.new("Camera")
    camera.type = "ORTHO"
    camera.ortho_scale = 2.0
    obj = bpy.data.objects.new("Camera", camera)
    scene.collection.objects.link(obj)
    obj.location = (0.0, 0.0, 3.0)
    scene.camera = obj
    bpy.ops.wm.save_as_mainfile(filepath=str(destination))
    print("Shared-closure probe:", destination)


if __name__ == "__main__":
    main()
