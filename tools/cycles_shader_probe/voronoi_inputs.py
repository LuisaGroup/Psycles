"""Voronoi Texture probes against raw Blender/Cycles node graphs."""

from __future__ import annotations

from typing import Any

from .support import (
    _input,
    _input_identifier,
    _linked_vector,
    _material,
    _material_matrix,
    _output,
)


def _linked_value(tree: Any, name: str, value: float) -> Any:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    _output(node, "Value").default_value = value
    return _output(node, "Value")


def _voronoi_material(
    name: str,
    *,
    dimensions: str,
    feature: str,
    metric: str,
    normalize: bool,
    output_name: str,
    coordinates: tuple[float, float, float] | None,
    w: float,
    scale: float,
    detail: float,
    roughness: float,
    lacunarity: float,
    smoothness: float,
    exponent: float,
    randomness: float,
    link_scalars: bool = False,
    runtime_coordinates: bool = False,
) -> Any:
    material, tree, output = _material(name)
    voronoi = tree.nodes.new("ShaderNodeTexVoronoi")
    voronoi.name = name
    voronoi.voronoi_dimensions = dimensions
    voronoi.feature = feature
    voronoi.distance = metric
    voronoi.normalize = normalize

    vector_input = voronoi.inputs.get("Vector")
    runtime_w = None
    if runtime_coordinates:
        if coordinates is None:
            raise ValueError("runtime Voronoi coordinates require a vector")
        geometry = tree.nodes.new("ShaderNodeNewGeometry")
        geometry.name = f"{name} Geometry"
        if vector_input is not None:
            vector_add = tree.nodes.new("ShaderNodeVectorMath")
            vector_add.name = f"{name} Runtime Vector"
            vector_add.operation = "ADD"
            _input_identifier(vector_add, "Vector_001").default_value = coordinates
            tree.links.new(
                _output(geometry, "Normal"),
                _input_identifier(vector_add, "Vector"),
            )
            tree.links.new(_output(vector_add, "Vector"), vector_input)

        separate = tree.nodes.new("ShaderNodeSeparateXYZ")
        separate.name = f"{name} Runtime W Source"
        w_scale = tree.nodes.new("ShaderNodeMath")
        w_scale.name = f"{name} Runtime W"
        w_scale.operation = "MULTIPLY"
        _input_identifier(w_scale, "Value_001").default_value = w
        tree.links.new(_output(geometry, "Normal"), _input(separate, "Vector"))
        tree.links.new(
            _output(separate, "Z"),
            _input_identifier(w_scale, "Value"),
        )
        runtime_w = _output(w_scale, "Value")
    elif coordinates is not None and vector_input is not None:
        vector = _linked_vector(
            tree,
            f"{name} Coordinates",
            coordinates,
        )
        tree.links.new(vector, vector_input)

    scalar_inputs = (
        ("W", w),
        ("Scale", scale),
        ("Detail", detail),
        ("Roughness", roughness),
        ("Lacunarity", lacunarity),
        ("Smoothness", smoothness),
        ("Exponent", exponent),
        ("Randomness", randomness),
    )
    for input_name, value in scalar_inputs:
        input_socket = voronoi.inputs.get(input_name)
        if input_socket is None:
            continue
        if runtime_coordinates and input_name == "W":
            tree.links.new(runtime_w, input_socket)
            continue
        if link_scalars:
            tree.links.new(
                _linked_value(
                    tree,
                    f"{name} {input_name}",
                    value,
                ),
                input_socket,
            )
        else:
            input_socket.default_value = value

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = f"{name} Emission"
    tree.links.new(
        _output(voronoi, output_name),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _svm_voronoi_matrix(scene: Any) -> None:
    """Force Cycles' Voronoi SVM handler with known runtime coordinates."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("1D", "F1", "EUCLIDEAN", False, "Distance"),
        ("1D", "F2", "CHEBYCHEV", True, "W"),
        ("1D", "SMOOTH_F1", "MINKOWSKI", False, "Color"),
        ("1D", "DISTANCE_TO_EDGE", "EUCLIDEAN", True, "Distance"),
        ("1D", "N_SPHERE_RADIUS", "EUCLIDEAN", False, "Radius"),
        ("2D", "F1", "MANHATTAN", False, "Color"),
        ("2D", "F2", "MINKOWSKI", False, "Position"),
        ("2D", "SMOOTH_F1", "CHEBYCHEV", True, "Distance"),
        ("2D", "DISTANCE_TO_EDGE", "EUCLIDEAN", True, "Distance"),
        ("2D", "N_SPHERE_RADIUS", "EUCLIDEAN", False, "Radius"),
        ("3D", "F1", "EUCLIDEAN", False, "Position"),
        ("3D", "F2", "MANHATTAN", True, "Distance"),
        ("3D", "SMOOTH_F1", "MINKOWSKI", False, "Color"),
        ("3D", "DISTANCE_TO_EDGE", "EUCLIDEAN", False, "Distance"),
        ("3D", "N_SPHERE_RADIUS", "EUCLIDEAN", True, "Radius"),
        ("4D", "F1", "CHEBYCHEV", False, "W"),
        ("4D", "F2", "EUCLIDEAN", True, "Color"),
        ("4D", "SMOOTH_F1", "MANHATTAN", True, "Position"),
        ("4D", "DISTANCE_TO_EDGE", "EUCLIDEAN", False, "Distance"),
        ("4D", "N_SPHERE_RADIUS", "EUCLIDEAN", True, "Radius"),
    )
    materials = []
    for index, (dimension, feature, metric, normalize, output_name) in enumerate(
        cases
    ):
        name = f"SVM Voronoi {index:02d} {dimension} {feature}"
        materials.append(
            _voronoi_material(
                name,
                dimensions=dimension,
                feature=feature,
                metric=metric,
                normalize=normalize,
                output_name=output_name,
                # Geometry Normal keeps the path shader-varying. The flat
                # matrix has N=(0, 0, 1), so the exact node input remains
                # known: Vector=coordinates+N and W=w.
                coordinates=(
                    -0.73 + 0.097 * index,
                    0.41 - 0.061 * index,
                    -1.17 + 0.083 * index,
                ),
                w=-0.83 + 0.071 * index,
                scale=(-2.3, 0.75, 1.9, 4.1)[index % 4],
                detail=(0.0, 0.375, 2.625, 5.25)[index % 4],
                roughness=(0.0, 0.37, 0.68, 1.0)[index % 4],
                lacunarity=(2.0, -1.25, 0.73, 2.4)[index % 4],
                smoothness=(0.0, 0.2, 0.83, 3.0)[index % 4],
                exponent=(0.5, 1.3, 2.0, 3.7)[index % 4],
                randomness=(0.0, 0.29, 0.71, 1.0)[index % 4],
                runtime_coordinates=True,
            )
        )

    _material_matrix(
        scene,
        materials,
        columns=5,
        rows=4,
        name="SVM Voronoi Matrix",
    )


def _voronoi_texture_distance(scene: Any) -> None:
    """Cover F1/F2, 1D--4D, every metric, and every X-FX output."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = []
    metrics = ("EUCLIDEAN", "MANHATTAN", "CHEBYCHEV", "MINKOWSKI")
    dimensions = ("1D", "2D", "3D", "4D")
    outputs = {
        "1D": ("Distance", "Color", "W"),
        "2D": ("Distance", "Color", "Position"),
        "3D": ("Distance", "Color", "Position"),
        "4D": ("Distance", "Color", "Position", "W"),
    }
    index = 0
    for dimension_index, dimension in enumerate(dimensions):
        for feature in ("F1", "F2"):
            for metric_index, metric in enumerate(metrics):
                output_name = outputs[dimension][index % len(outputs[dimension])]
                coordinates = (
                    None
                    if index % 3 == 0 and dimension != "1D"
                    else (
                        -1.875 + 0.173 * index,
                        2.125 - 0.119 * metric_index,
                        -0.625 + 0.137 * dimension_index,
                    )
                )
                name = (
                    f"Voronoi Distance {index:02d} {dimension} "
                    f"{feature} {metric} {output_name}"
                )
                materials.append(
                    _voronoi_material(
                        name,
                        dimensions=dimension,
                        feature=feature,
                        metric=metric,
                        normalize=False,
                        output_name=output_name,
                        coordinates=coordinates,
                        w=-2.375 + 0.211 * index,
                        scale=(0.0, 0.75, -2.25, 11.5)[index % 4],
                        detail=0.0,
                        roughness=0.5,
                        lacunarity=2.0,
                        smoothness=1.0,
                        exponent=(0.5, 1.0, 1.75, 4.0)[metric_index],
                        randomness=(0.0, 0.37, 0.83, 1.0)[index % 4],
                    )
                )
                index += 1

    _material_matrix(
        scene,
        materials,
        columns=8,
        rows=4,
        name="Voronoi Distance Matrix",
    )


def _voronoi_texture_fractal(scene: Any) -> None:
    """Cover smooth F1 and Cycles' dynamic fractal/Normalize recurrence."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    # High-detail cases keep the largest octave inside the signed 32-bit cell
    # domain used by Cycles' integer hashes. Float-to-int conversion outside
    # that domain is backend-dependent and is not a defined Voronoi result.
    cases = (
        ("1D", "F1", "EUCLIDEAN", False, "Distance", 0.0, 0.5, 2.0, 1.0),
        ("1D", "SMOOTH_F1", "MINKOWSKI", True, "Color", 2.375, 0.63, 1.7, 0.0),
        ("1D", "F2", "CHEBYCHEV", True, "W", 15.0, 0.0, 2.0, 0.7),
        ("1D", "SMOOTH_F1", "MANHATTAN", False, "Distance", 0.25, 1.0, -1.0, 3.0),
        ("2D", "F1", "EUCLIDEAN", True, "Position", 1.25, 0.35, 2.3, 0.4),
        ("2D", "F2", "MANHATTAN", False, "Color", 4.75, 1.0, 0.5, 0.8),
        ("2D", "SMOOTH_F1", "CHEBYCHEV", True, "Distance", 7.5, 0.72, -1.5, 0.01),
        ("2D", "SMOOTH_F1", "MINKOWSKI", False, "Position", 16.0, 1.5, 3.0, 2.0),
        ("3D", "F1", "MANHATTAN", False, "Color", 0.001, 0.4, 1.999, 0.5),
        ("3D", "F2", "CHEBYCHEV", True, "Distance", 3.5, 0.8, 2.5, 0.25),
        ("3D", "SMOOTH_F1", "EUCLIDEAN", False, "Position", 9.25, 0.55, 0.75, 1.0),
        ("3D", "SMOOTH_F1", "MINKOWSKI", True, "Color", 14.5, 0.42, 2.0, -0.3),
        ("4D", "F1", "CHEBYCHEV", True, "W", 2.75, 0.7, 2.0, 0.65),
        ("4D", "F2", "EUCLIDEAN", False, "Position", 5.125, 0.2, -0.75, 0.3),
        ("4D", "SMOOTH_F1", "MANHATTAN", True, "Color", 12.875, 0.9, 1.1, 0.9),
        ("4D", "SMOOTH_F1", "MINKOWSKI", False, "Distance", 0.0, 0.0, 8.0, 0.0),
    )
    materials = []
    for index, (
        dimension,
        feature,
        metric,
        normalize,
        output_name,
        detail,
        roughness,
        lacunarity,
        smoothness,
    ) in enumerate(cases):
        name = f"Voronoi Fractal {index:02d} {dimension} {feature}"
        materials.append(
            _voronoi_material(
                name,
                dimensions=dimension,
                feature=feature,
                metric=metric,
                normalize=normalize,
                output_name=output_name,
                # Keep multidimensional coordinates shader-varying. Besides
                # covering many coordinate cells per configuration, this
                # prevents Blender's host constant folder from replacing the
                # Cycles kernel evaluation. The two paths differ for linked
                # Detail/Roughness values outside their declared socket range;
                # this probe is specifically an oracle for the Cycles kernel.
                coordinates=(
                    (
                        -0.875 + 0.161 * index,
                        1.625 - 0.093 * index,
                        -2.125 + 0.127 * index,
                    )
                    if dimension == "1D"
                    else None
                ),
                w=0.375 - 0.149 * index,
                scale=(-3.5, 0.0, 0.875, 7.25)[index % 4],
                detail=detail,
                roughness=roughness,
                lacunarity=lacunarity,
                smoothness=smoothness,
                exponent=(0.5, 1.3, 2.0, 4.5)[index % 4],
                randomness=(-0.5, 0.0, 0.61, 1.5)[index % 4],
                link_scalars=True,
            )
        )

    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Voronoi Fractal Matrix",
    )


def _voronoi_texture_edges(scene: Any) -> None:
    """Cover Distance to Edge and N-Sphere Radius in every dimension."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = []
    index = 0
    for dimension in ("1D", "2D", "3D", "4D"):
        for feature, output_name in (
            ("DISTANCE_TO_EDGE", "Distance"),
            ("N_SPHERE_RADIUS", "Radius"),
        ):
            for variant in range(2):
                name = f"Voronoi Edge {index:02d} {dimension} {feature}"
                materials.append(
                    _voronoi_material(
                        name,
                        dimensions=dimension,
                        feature=feature,
                        metric="MINKOWSKI",
                        normalize=variant == 1,
                        output_name=output_name,
                        coordinates=(
                            -1.25 + 0.217 * index,
                            0.875 - 0.131 * index,
                            -2.5 + 0.179 * index,
                        ),
                        w=1.75 - 0.233 * index,
                        scale=(-2.0, 5.5)[variant],
                        detail=(0.0, 3.625)[variant],
                        roughness=(0.0, 0.73)[variant],
                        lacunarity=(2.0, -1.25)[variant],
                        smoothness=0.5,
                        exponent=1.7,
                        randomness=(0.0, 1.0)[variant],
                        link_scalars=True,
                    )
                )
                index += 1

    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Voronoi Edge Matrix",
    )
