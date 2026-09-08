"""Create original-Cycles constant/identity Vector Math compiler witnesses."""
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
    operations = [item.identifier for item in
        bpy.types.ShaderNodeVectorMath.bl_rna.properties["operation"].enum_items]
    assert len(operations) == 30
    cases = [(f"constant-{operation}", operation, None, None) for operation in operations]
    for operation in ("NORMALIZE", "REFLECT", "REFRACT"):
        for magnitude in ("tiny", "regular"):
            cases.append((f"constant-{operation}-{magnitude}", operation, None, None))
    cases += [("constant-CROSS_PRODUCT-cancellation", "CROSS_PRODUCT", None, None),
              ("constant-PROJECT-underflow", "PROJECT", None, None)]
    for operation in ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "DOT_PRODUCT",
                      "CROSS_PRODUCT", "SCALE"):
        for operand in (0, 1):
            for identity in (0.0, 1.0):
                cases.append((f"identity-{operation}-{operand}-{int(identity)}",
                              operation, operand, identity))
    for index, (name, operation, operand, identity) in enumerate(cases):
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        math = tree.nodes.new("ShaderNodeVectorMath")
        math.operation = operation
        for socket, value in zip(math.inputs[:3],
                                 ((-1.25, 0.5, 2.0), (2.0, 0.0, -0.5), (0.25, -1.0, 1.0))):
            socket.default_value = value
        scale = next(socket for socket in math.inputs if socket.identifier == "Scale")
        scale.default_value = 0.5
        if name.endswith(("-tiny", "-regular")):
            normal = math.inputs[0 if operation == "NORMALIZE" else 1]
            normal.default_value = (1e-20 if name.endswith("-tiny") else 1e-16, 0, 0)
        if name.endswith("-cancellation"):
            math.inputs[0].default_value = (1, 1 + 2**-23, 0)
            math.inputs[1].default_value = (1 + 2**-23, 1 + 2**-22, 0)
        if name.endswith("-underflow"):
            math.inputs[1].default_value = (1e-25, 0, 0)
        if operand is not None:
            geometry = tree.nodes.new("ShaderNodeNewGeometry")
            if operation == "SCALE":
                if operand == 0:
                    tree.links.new(geometry.outputs["Position"], math.inputs[0])
                    scale.default_value = identity
                else:
                    tree.links.new(geometry.outputs["Position"], scale)
                    math.inputs[0].default_value = (identity,) * 3
            else:
                tree.links.new(geometry.outputs["Position"], math.inputs[operand])
                math.inputs[1 - operand].default_value = (identity,) * 3
        emission = tree.nodes.new("ShaderNodeEmission")
        result = math.outputs["Value" if operation in
                              {"DOT_PRODUCT", "LENGTH", "DISTANCE"} else "Vector"]
        tree.links.new(result, emission.inputs["Color"])
        tree.links.new(emission.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=1, location=(index % 10, index // 10, 0))
        bpy.context.object.data.materials.append(material)
    bpy.ops.object.camera_add(location=(4.5, 2.5, 10))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 12
    scene.render.resolution_x = 20
    scene.render.resolution_y = 12
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


if __name__ == "__main__":
    main()
