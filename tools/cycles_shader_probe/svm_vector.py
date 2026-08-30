"""Cycles 5.2 SVM vector-node probes."""

from __future__ import annotations

from typing import Any

from mathutils import Euler, Matrix, Vector

from .support import _input, _material, _material_matrix, _output, _world


def _svm_sepcomb_vector_pipeline(scene: Any) -> None:
    """Keep all three Cycles vector split/pack transitions dynamic."""
    material, tree, output = _material("SVM Separate Combine Vector")

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = "Geometry"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = "Separate XYZ"
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = "Combine XYZ"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"

    tree.links.new(_output(geometry, "Normal"), _input(separate, "Vector"))
    tree.links.new(_output(separate, "Z"), _input(combine, "X"))
    tree.links.new(_output(separate, "X"), _input(combine, "Y"))
    tree.links.new(_output(separate, "Y"), _input(combine, "Z"))
    tree.links.new(_output(combine, "Vector"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))

    surface = _material_matrix(
        scene,
        [material, material],
        columns=2,
        rows=1,
        name="SVM Separate Combine Vector",
        backfacing={1},
        frame_bleed=0.1,
    )
    surface.rotation_euler = (0.31, -0.27, 0.19)


def _svm_sepcomb_vector_constant_fold(scene: Any) -> None:
    """Force Cycles' SeparateXYZNode and CombineXYZNode constant folders."""
    material, tree, output = _material(
        "SVM Separate Combine Vector Constant"
    )

    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = "Combine XYZ Constant"
    for socket, value in zip(
        ("X", "Y", "Z"), (-0.7, 0.25, 1.3), strict=True
    ):
        scalar = tree.nodes.new("ShaderNodeValue")
        scalar.name = f"{socket} Constant"
        _output(scalar, "Value").default_value = value
        tree.links.new(_output(scalar, "Value"), _input(combine, socket))

    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = "Separate XYZ Constant"
    packed = tree.nodes.new("ShaderNodeCombineColor")
    packed.name = "Pack Constant Result"
    packed.mode = "RGB"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"

    tree.links.new(_output(combine, "Vector"), _input(separate, "Vector"))
    tree.links.new(_output(separate, "Z"), _input(packed, "Red"))
    tree.links.new(_output(separate, "X"), _input(packed, "Green"))
    tree.links.new(_output(separate, "Y"), _input(packed, "Blue"))
    tree.links.new(_output(packed, "Color"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))

    _material_matrix(
        scene,
        [material],
        columns=1,
        rows=1,
        name="SVM Separate Combine Vector Constant",
        frame_bleed=0.1,
    )


def _svm_vector_rotate_matrix(scene: Any) -> None:
    """Exercise every Cycles Vector Rotate mode and its zero-axis branch."""
    cases = [
        ("AXIS_ANGLE", False, (0.29, 0.73, -0.41), "Forward"),
        ("AXIS_ANGLE", True, (0.29, 0.73, -0.41), "Inverse"),
        ("X_AXIS", False, (0.29, 0.73, -0.41), "Forward"),
        ("X_AXIS", True, (0.29, 0.73, -0.41), "Inverse"),
        ("Y_AXIS", False, (0.29, 0.73, -0.41), "Forward"),
        ("Y_AXIS", True, (0.29, 0.73, -0.41), "Inverse"),
        ("Z_AXIS", False, (0.29, 0.73, -0.41), "Forward"),
        ("Z_AXIS", True, (0.29, 0.73, -0.41), "Inverse"),
        ("EULER_XYZ", False, (0.29, 0.73, -0.41), "Forward"),
        ("EULER_XYZ", True, (0.29, 0.73, -0.41), "Inverse"),
        ("AXIS_ANGLE", False, (0.0, 0.0, 0.0), "Zero Axis"),
        ("AXIS_ANGLE", True, (0.0, 0.0, 0.0), "Zero Axis Inverse"),
    ]
    materials = []
    for index, (rotate_type, invert, axis, label) in enumerate(cases):
        material, tree, output = _material(
            f"SVM Vector Rotate {index:02d} {rotate_type} {label}"
        )
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        geometry.name = "Geometry"
        rotate = tree.nodes.new("ShaderNodeVectorRotate")
        rotate.name = "Vector Rotate"
        rotate.rotation_type = rotate_type
        rotate.invert = invert
        _input(rotate, "Center").default_value = (0.17, -0.23, 0.31)
        if rotate_type == "EULER_XYZ":
            _input(rotate, "Rotation").default_value = (0.31, -0.52, 0.27)
        else:
            _input(rotate, "Angle").default_value = 0.71
            if rotate_type == "AXIS_ANGLE":
                _input(rotate, "Axis").default_value = axis
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "Emission"

        tree.links.new(_output(geometry, "Normal"), _input(rotate, "Vector"))
        tree.links.new(_output(rotate, "Vector"), _input(emission, "Color"))
        tree.links.new(
            _output(emission, "Emission"), _input(output, "Surface")
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=6,
        rows=2,
        name="SVM Vector Rotate Matrix",
        frame_bleed=0.1,
    )


def _set_vector_transform_probe_space(scene: Any, surface: Any) -> None:
    """Give a probe Cycles-visible camera and non-rigid object transforms."""
    camera_to_world = Matrix.Translation((0.42, -0.31, 3.17)) @ Euler(
        (0.23, -0.17, 0.31), "XYZ"
    ).to_matrix().to_4x4()
    object_to_world = (
        Matrix.Translation((-0.28, 0.41, -0.19))
        @ Euler((-0.37, 0.26, -0.18), "XYZ").to_matrix().to_4x4()
        @ Matrix.Diagonal((1.35, 0.72, 1.18, 1.0))
    )

    # Keep the matrix camera-aligned while retaining a non-trivial object
    # transform in ShaderData. Each original vertex is a camera-plane (x, y)
    # coordinate; solve its object-space position before assigning the object
    # matrix so the final world-space surface remains unchanged.
    world_to_object = object_to_world.inverted()
    for vertex in surface.data.vertices:
        camera_position = Vector((vertex.co.x, vertex.co.y, -3.0))
        vertex.co = world_to_object @ (camera_to_world @ camera_position)
    surface.matrix_world = object_to_world
    surface.data.update()
    scene.camera.matrix_world = camera_to_world


def _svm_vector_transform_matrix(scene: Any) -> None:
    """Exercise every Cycles Vector Transform type and space pair."""
    vector_types = ("VECTOR", "POINT", "NORMAL")
    spaces = ("WORLD", "OBJECT", "CAMERA")
    materials = []
    for vector_type in vector_types:
        for convert_from in spaces:
            for convert_to in spaces:
                material, tree, output = _material(
                    "SVM Vector Transform "
                    f"{vector_type} {convert_from} to {convert_to}"
                )
                transform = tree.nodes.new("ShaderNodeVectorTransform")
                transform.name = "Vector Transform"
                transform.vector_type = vector_type
                transform.convert_from = convert_from
                transform.convert_to = convert_to
                _input(transform, "Vector").default_value = (
                    0.37,
                    -0.21,
                    0.63,
                )
                emission = tree.nodes.new("ShaderNodeEmission")
                emission.name = "Emission"
                tree.links.new(
                    _output(transform, "Vector"),
                    _input(emission, "Color"),
                )
                tree.links.new(
                    _output(emission, "Emission"),
                    _input(output, "Surface"),
                )
                materials.append(material)

    surface = _material_matrix(
        scene,
        materials,
        columns=9,
        rows=3,
        name="SVM Vector Transform Matrix",
        frame_bleed=0.1,
    )
    _set_vector_transform_probe_space(scene, surface)


def _svm_vector_transform_zero_normal(scene: Any) -> None:
    """Freeze Cycles' safe-normalize zero branch for world-to-object."""
    material, tree, output = _material("SVM Vector Transform Zero Normal")
    transform = tree.nodes.new("ShaderNodeVectorTransform")
    transform.name = "Vector Transform"
    transform.vector_type = "NORMAL"
    transform.convert_from = "WORLD"
    transform.convert_to = "OBJECT"
    _input(transform, "Vector").default_value = (0.0, 0.0, 0.0)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(_output(transform, "Vector"), _input(emission, "Color"))
    tree.links.new(_output(emission, "Emission"), _input(output, "Surface"))
    surface = _material_matrix(
        scene,
        [material],
        columns=1,
        rows=1,
        name="SVM Vector Transform Zero Normal",
        frame_bleed=0.1,
    )
    _set_vector_transform_probe_space(scene, surface)


def _svm_vector_transform_no_object(
    scene: Any, convert_from: str, convert_to: str
) -> None:
    """Exercise Cycles' OBJECT_NONE transform branches in a world shader."""
    _, tree, background = _world(scene, (0.0, 0.0, 0.0, 1.0), 1.0)
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = "Pack No Object Results"

    vector_types = ("VECTOR", "POINT", "NORMAL")
    components = ("X", "Y", "Z")
    for vector_type, component in zip(
        vector_types, components, strict=True
    ):
        transform = tree.nodes.new("ShaderNodeVectorTransform")
        transform.name = f"{vector_type} {convert_from} to {convert_to}"
        transform.vector_type = vector_type
        transform.convert_from = convert_from
        transform.convert_to = convert_to
        _input(transform, "Vector").default_value = (
            0.37,
            -0.21,
            0.63,
        )
        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        separate.name = f"Separate {vector_type}"
        tree.links.new(
            _output(transform, "Vector"), _input(separate, "Vector")
        )
        tree.links.new(
            _output(separate, component), _input(combine, component)
        )

    tree.links.new(_output(combine, "Vector"), _input(background, "Color"))
    scene.camera.matrix_world = Matrix.Translation((0.42, -0.31, 3.17)) @ Euler(
        (0.23, -0.17, 0.31), "XYZ"
    ).to_matrix().to_4x4()


def _svm_vector_transform_no_object_to_camera(scene: Any) -> None:
    _svm_vector_transform_no_object(scene, "OBJECT", "CAMERA")


def _svm_vector_transform_no_object_from_camera(scene: Any) -> None:
    _svm_vector_transform_no_object(scene, "CAMERA", "OBJECT")
