"""Magic Texture probes against raw Blender/Cycles node graphs."""

from __future__ import annotations

from typing import Any

from .support import (
    _input,
    _linked_vector,
    _material,
    _material_matrix,
    _output,
    _output_identifier,
)


def _linked_value(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return node


def _linked_runtime_vector(
    tree: Any,
    name: str,
    value: tuple[float, float, float],
) -> Any:
    """Keep a nominally constant vector on Cycles' SVM execution path.

    Blender's shader-node inliner evaluates Magic Texture on the host whenever
    all inputs are constant. That host implementation intentionally does not
    use the range reduction performed by Cycles' SVM kernel, so it cannot be
    used as a kernel oracle for large-coordinate cases. Adding Position.x to
    1e20 keeps the graph dynamic while rounding back to exactly 1e20 in
    float32 for every point on this probe mesh.
    """

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    geometry.name = f"{name} Geometry"
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    separate.name = f"{name} Separate Position"
    tree.links.new(
        _output(geometry, "Position"),
        _input(separate, "Vector"),
    )

    add = tree.nodes.new("ShaderNodeMath")
    add.name = f"{name} Dynamic X"
    add.operation = "ADD"
    add.inputs[0].default_value = value[0]
    tree.links.new(_output(separate, "X"), add.inputs[1])

    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    combine.name = name
    tree.links.new(_output(add, "Value"), _input(combine, "X"))
    _input(combine, "Y").default_value = value[1]
    _input(combine, "Z").default_value = value[2]
    return _output(combine, "Vector")


def _magic_material(
    name: str,
    *,
    depth: int,
    vector: tuple[float, float, float] | None,
    scale: float,
    distortion: float,
    output_name: str,
    force_runtime_vector: bool = False,
) -> Any:
    material, tree, output = _material(name)
    magic = tree.nodes.new("ShaderNodeTexMagic")
    magic.name = name
    magic.turbulence_depth = depth
    if vector is not None:
        vector_output = (
            _linked_runtime_vector(
                tree,
                f"{name} Coordinates",
                vector,
            )
            if force_runtime_vector
            else _linked_vector(
                tree,
                f"{name} Coordinates",
                vector,
            )
        )
        tree.links.new(
            vector_output,
            _input(magic, "Vector"),
        )
    for input_name, value in (
        ("Scale", scale),
        ("Distortion", distortion),
    ):
        source = _linked_value(
            tree,
            f"{name} {input_name}",
            value,
        )
        tree.links.new(
            _output(source, "Value"),
            _input(magic, input_name),
        )

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    source = (
        _output_identifier(magic, "Fac")
        if output_name == "Fac"
        else _output(magic, "Color")
    )
    tree.links.new(source, _input(emission, "Color"))
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _magic_texture_matrix(scene: Any) -> None:
    """Cover every static depth plus signed and range-reduction cases."""

    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases: list[
        tuple[
            int,
            tuple[float, float, float] | None,
            float,
            float,
            str,
        ]
    ] = []
    for depth in range(11):
        cases.append(
            (
                depth,
                (
                    0.173 + 0.137 * depth,
                    -0.625 + 0.071 * depth,
                    1.375 - 0.097 * depth,
                ),
                -2.35 if depth in {3, 8} else 6.6,
                -0.73 if depth in {5, 9} else 1.0,
                "Color" if depth % 2 == 0 else "Fac",
            )
        )
    cases.extend(
        (
            # The exact fmod reduction is visible only when one coordinate
            # dwarfs the others by enough float32 exponents.
            (2, (1.0e20, -0.375, 0.8125), 0.001, 1.0, "Color"),
            # Distortion zero bypasses both the recurrence scale and final
            # normalization branch without producing NaNs.
            (10, (-1.25, 2.5, -4.0), 3.75, 0.0, "Fac"),
            (10, (0.3125, -0.9375, 2.125), -7.0, -2.25, "Color"),
            (2, (8.0, -16.0, 32.0), 0.0, 1.0, "Fac"),
            # Leave Vector unlinked once to cover Cycles' implicit Generated
            # coordinate route on the matrix mesh.
            (6, None, 4.25, 0.37, "Color"),
        )
    )
    materials = [
        _magic_material(
            f"Magic {index:02d} Depth {depth} {output_name}",
            depth=depth,
            vector=vector,
            scale=scale,
            distortion=distortion,
            output_name=output_name,
            force_runtime_vector=index == 11,
        )
        for index, (
            depth,
            vector,
            scale,
            distortion,
            output_name,
        ) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Magic Texture Matrix",
    )
