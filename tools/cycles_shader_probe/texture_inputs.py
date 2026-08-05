"""Noise, image, coordinate, ramp, curve, and numeric texture probes."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Any

import bpy
import numpy as np
import OpenImageIO as oiio

from .support import (
    _input,
    _input_identifier,
    _material,
    _material_matrix,
    _output,
    _output_identifier,
    _plane,
    _world,
)


def _noise_color_3d(scene: Any) -> None:
    material, tree, output = _material("Noise Color 3D Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 3D"
    noise.noise_dimensions = "3D"
    noise.noise_type = "FBM"
    noise.normalize = True
    _input(noise, "Scale").default_value = 1.7
    _input(noise, "Detail").default_value = 2.35
    _input(noise, "Roughness").default_value = 0.61
    _input(noise, "Lacunarity").default_value = 2.2
    _input(noise, "Distortion").default_value = 0.37
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _noise_factor_2d(scene: Any) -> None:
    material, tree, output = _material("Noise Factor 2D Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 2D"
    noise.noise_dimensions = "2D"
    noise.noise_type = "FBM"
    noise.normalize = False
    _input(noise, "Scale").default_value = 2.3
    _input(noise, "Detail").default_value = 1.75
    _input(noise, "Roughness").default_value = 0.43
    _input(noise, "Lacunarity").default_value = 1.8
    _input(noise, "Distortion").default_value = 0.0
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Fac"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _noise_bump_object(scene: Any) -> None:
    _world(scene, (0.42, 0.52, 0.65, 1.0), 0.8)
    material, tree, output = _material("Noise Bump Object Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.name = "Cycles Noise 3D"
    noise.noise_dimensions = "3D"
    noise.noise_type = "FBM"
    noise.normalize = True
    _input(noise, "Scale").default_value = 20.0
    _input(noise, "Detail").default_value = 2.0
    _input(noise, "Roughness").default_value = 0.5
    _input(noise, "Lacunarity").default_value = 2.0
    bump = tree.nodes.new("ShaderNodeBump")
    bump.name = "Bump"
    _input(bump, "Strength").default_value = 0.2
    _input(bump, "Distance").default_value = 0.005
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.name = "Diffuse BSDF"
    _input(diffuse, "Color").default_value = (
        0.5,
        0.22,
        0.08,
        1.0,
    )
    tree.links.new(
        _output(coordinates, "Object"),
        _input(noise, "Vector"),
    )
    tree.links.new(
        _output(noise, "Fac"),
        _input(bump, "Height"),
    )
    tree.links.new(
        _output(bump, "Normal"),
        _input(diffuse, "Normal"),
    )
    tree.links.new(
        _output(diffuse, "BSDF"),
        _input(output, "Surface"),
    )
    _plane(material)


def _noise_type_matrix(scene: Any, noise_type: str) -> None:
    """Tile every dimension/normalize/output combination for one noise type."""
    # The matrix intentionally contains discontinuities at exact pixel
    # boundaries. A narrow box filter keeps this a shader-formula comparison
    # instead of measuring Cycles' reconstruction filter against Psycles'
    # current film filter.
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials: list[Any] = []
    for normalize in (False, True):
        for dimensions in range(1, 5):
            for output_name in ("Fac", "Color"):
                material, tree, output = _material(
                    "Noise "
                    f"{noise_type} {dimensions}D "
                    f"{'Normalized' if normalize else 'Raw'} "
                    f"{output_name}"
                )
                noise = tree.nodes.new("ShaderNodeTexNoise")
                noise.name = (
                    f"Noise {noise_type} {dimensions}D "
                    f"{output_name}"
                )
                noise.noise_dimensions = f"{dimensions}D"
                noise.noise_type = noise_type
                noise.normalize = normalize
                if dimensions != 1:
                    vector = tree.nodes.new("ShaderNodeRGB")
                    vector.name = "Constant Noise Coordinates"
                    _output(vector, "Color").default_value = (
                        0.173,
                        -0.625,
                        1.375,
                        1.0,
                    )
                    tree.links.new(
                        _output(vector, "Color"),
                        _input_identifier(noise, "Vector"),
                    )
                _input_identifier(noise, "W").default_value = -0.437
                _input_identifier(noise, "Scale").default_value = 2.35
                _input_identifier(noise, "Detail").default_value = 2.375
                _input_identifier(
                    noise, "Roughness"
                ).default_value = 0.63
                _input_identifier(
                    noise, "Lacunarity"
                ).default_value = 2.17
                _input_identifier(noise, "Offset").default_value = 0.37
                _input_identifier(noise, "Gain").default_value = 1.11
                _input_identifier(
                    noise, "Distortion"
                ).default_value = 0.42
                emission = tree.nodes.new("ShaderNodeEmission")
                emission.name = "Emission"
                tree.links.new(
                    _output(noise, output_name),
                    _input(emission, "Color"),
                )
                tree.links.new(
                    _output(emission, "Emission"),
                    _input(output, "Surface"),
                )
                materials.append(material)

    # A single contiguous mesh eliminates reconstruction-filter background
    # edges. Each face has a separate material so a wrong mode remains
    # spatially visible instead of disappearing inside an aggregate.
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name=f"Noise {noise_type} Matrix",
    )


def _noise_fbm_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "FBM")


def _noise_multifractal_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "MULTIFRACTAL")


def _noise_ridged_multifractal_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "RIDGED_MULTIFRACTAL")


def _noise_hybrid_multifractal_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "HYBRID_MULTIFRACTAL")


def _noise_hetero_terrain_matrix(scene: Any) -> None:
    _noise_type_matrix(scene, "HETERO_TERRAIN")


def _gradient_spherical(scene: Any) -> None:
    material, tree, output = _material("Spherical Gradient Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.name = "Point Mapping"
    mapping.vector_type = "POINT"
    _input(mapping, "Scale").default_value = (0.6, 0.6, 0.6)
    gradient = tree.nodes.new("ShaderNodeTexGradient")
    gradient.name = "Spherical Gradient"
    gradient.gradient_type = "SPHERICAL"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(coordinates, "Object"),
        _input(mapping, "Vector"),
    )
    tree.links.new(
        _output(mapping, "Vector"),
        _input(gradient, "Vector"),
    )
    tree.links.new(
        _output(gradient, "Fac"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _packed_rgba_image(
    name: str,
    pixels: np.ndarray,
    *,
    colorspace: str,
    alpha_mode: str,
) -> Any:
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False
    ) as temporary:
        texture_path = pathlib.Path(temporary.name)
    output_image = oiio.ImageOutput.create(str(texture_path))
    if output_image is None:
        raise RuntimeError("could not create probe PNG")
    try:
        height, width, channels = pixels.shape
        if channels != 4:
            raise ValueError("packed probe image must have four channels")
        if not output_image.open(
            str(texture_path),
            oiio.ImageSpec(width, height, channels, oiio.UINT8),
        ):
            raise RuntimeError("could not open probe PNG")
        if not output_image.write_image(pixels):
            raise RuntimeError("could not write probe PNG")
    finally:
        output_image.close()

    try:
        image = bpy.data.images.load(
            str(texture_path), check_existing=False
        )
        image.name = name
        image.colorspace_settings.name = colorspace
        image.alpha_mode = alpha_mode
        image.pack()
    finally:
        texture_path.unlink(missing_ok=True)
    return image


def _image_texture_srgb(scene: Any) -> None:
    """Exercise sRGB-before-filtering and straight-alpha sampling."""
    image = _packed_rgba_image(
        "Probe sRGB Texture",
        np.asarray(
            [
                [
                    (16, 64, 240, 32),
                    (240, 32, 16, 224),
                ],
                [
                    (32, 220, 64, 96),
                    (180, 128, 200, 160),
                ],
            ],
            dtype=np.uint8,
        ),
        colorspace="sRGB",
        alpha_mode="STRAIGHT",
    )

    material, tree, output = _material("sRGB Image Texture")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.name = "Image Texture"
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "EXTEND"
    separate = tree.nodes.new("ShaderNodeSeparateColor")
    separate.name = "Separate Image Color"
    separate.mode = "RGB"
    tree.links.new(
        _output(texture, "Color"),
        _input(separate, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = "Pack Color and Alpha"
    combine.mode = "RGB"
    tree.links.new(
        _output(separate, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(separate, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(texture, "Alpha"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _image_texture_node_mapping(scene: Any) -> None:
    """Cover Cycles' hidden TextureNode TexMapping before image lookup."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scene.cycles.max_bounces = 0
    pixels = np.empty((8, 8, 4), dtype=np.uint8)
    for y in range(8):
        for x in range(8):
            pixels[y, x] = (
                (29 * x + 47 * y + 11) % 256,
                (83 * x + 17 * y + 37) % 256,
                (13 * x + 101 * y + 71) % 256,
                255,
            )
    image = _packed_rgba_image(
        "TextureNode Mapping Probe",
        pixels,
        colorspace="Non-Color",
        alpha_mode="STRAIGHT",
    )

    cases = (
        (
            "POINT",
            (0.13, -0.21, 0.37),
            (0.17, -0.11, 0.23),
            (0.63, 1.17, -0.81),
            ("X", "Y", "Z"),
        ),
        (
            "TEXTURE",
            (-0.19, 0.31, -0.07),
            (-0.14, 0.27, 0.09),
            (0.83, -1.31, 0.57),
            ("Z", "X", "Y"),
        ),
        (
            "VECTOR",
            (0.0, 0.0, 0.0),
            (0.21, 0.08, -0.19),
            (-0.71, 1.23, 0.49),
            ("Y", "NONE", "X"),
        ),
        (
            "NORMAL",
            (0.0, 0.0, 0.0),
            (-0.09, 0.18, 0.31),
            (0.77, -0.59, 1.41),
            ("X", "Z", "Y"),
        ),
    )
    materials = []
    for vector_type, translation, rotation, scale, axes in cases:
        material, tree, output = _material(
            f"TextureNode Mapping {vector_type}"
        )
        coordinates = tree.nodes.new("ShaderNodeTexCoord")
        coordinates.name = f"{vector_type} Object Coordinates"
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.name = f"{vector_type} Legacy Texture Mapping"
        texture.image = image
        texture.interpolation = "Closest"
        texture.extension = "REPEAT"
        mapping = texture.texture_mapping
        mapping.vector_type = vector_type
        mapping.translation = translation
        mapping.rotation = rotation
        mapping.scale = scale
        mapping.mapping_x, mapping.mapping_y, mapping.mapping_z = axes
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"{vector_type} Mapping Emission"
        tree.links.new(
            _output(coordinates, "Object"),
            _input(texture, "Vector"),
        )
        tree.links.new(
            _output(texture, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=1,
        name="TextureNode Mapping Matrix",
        frame_bleed=0.02,
    )


def _image_texture_sampling_modes(scene: Any) -> None:
    """Exercise Cycles' 2D interpolation and extension cross-product."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    image = _packed_rgba_image(
        "Probe Raw Sampling Texture",
        np.asarray(
            [
                [
                    (7, 19, 233, 31),
                    (41, 211, 67, 83),
                    (173, 29, 109, 149),
                    (251, 137, 11, 223),
                    (97, 59, 197, 47),
                ],
                [
                    (227, 73, 17, 199),
                    (61, 151, 239, 101),
                    (131, 239, 53, 181),
                    (19, 97, 157, 59),
                    (199, 31, 79, 241),
                ],
                [
                    (83, 229, 127, 113),
                    (149, 43, 193, 167),
                    (239, 181, 23, 71),
                    (53, 7, 211, 137),
                    (167, 113, 61, 211),
                ],
                [
                    (31, 109, 179, 251),
                    (211, 67, 97, 43),
                    (107, 197, 37, 191),
                    (71, 241, 139, 89),
                    (233, 17, 151, 157),
                ],
            ],
            dtype=np.uint8,
        ),
        colorspace="Non-Color",
        alpha_mode="STRAIGHT",
    )
    coordinates = {
        "REPEAT": (-0.173, 1.217, 0.37),
        "EXTEND": (1.127, -0.083, 0.37),
        "CLIP": (-0.037, 0.617, 0.37),
        "MIRROR": (-0.283, 1.193, 0.37),
    }
    materials = []
    for interpolation in ("Closest", "Linear", "Cubic", "Smart"):
        for extension in ("REPEAT", "EXTEND", "CLIP", "MIRROR"):
            name = f"Image {interpolation} {extension}"
            material, tree, output = _material(name)
            mapping = tree.nodes.new("ShaderNodeMapping")
            mapping.name = f"{name} Coordinate"
            mapping.vector_type = "POINT"
            _input(mapping, "Vector").default_value = coordinates[extension]
            texture = tree.nodes.new("ShaderNodeTexImage")
            texture.name = name
            texture.image = image
            texture.interpolation = interpolation
            texture.extension = extension
            texture.projection = "FLAT"
            tree.links.new(
                _output(mapping, "Vector"),
                _input(texture, "Vector"),
            )
            channels = tree.nodes.new("ShaderNodeSeparateColor")
            channels.name = f"{name} Channels"
            channels.mode = "RGB"
            tree.links.new(
                _output(texture, "Color"),
                _input(channels, "Color"),
            )
            packed = tree.nodes.new("ShaderNodeCombineColor")
            packed.name = f"{name} RGB Alpha"
            packed.mode = "RGB"
            tree.links.new(
                _output(channels, "Red"),
                _input(packed, "Red"),
            )
            tree.links.new(
                _output(channels, "Green"),
                _input(packed, "Green"),
            )
            tree.links.new(
                _output(texture, "Alpha"),
                _input(packed, "Blue"),
            )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"{name} Emission"
            tree.links.new(
                _output(packed, "Color"),
                _input(emission, "Color"),
            )
            tree.links.new(
                _output(emission, "Emission"),
                _input(output, "Surface"),
            )
            materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Image Sampling Modes Matrix",
    )


def _image_texture_projection_modes(scene: Any) -> None:
    """Exercise flat, spherical, tube, and blended box projection."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    image = _packed_rgba_image(
        "Probe Projection Texture",
        np.asarray(
            [
                [
                    (11, 47, 229, 37),
                    (83, 197, 29, 113),
                    (179, 17, 131, 193),
                    (241, 109, 61, 73),
                ],
                [
                    (59, 223, 151, 167),
                    (211, 71, 101, 43),
                    (127, 239, 19, 227),
                    (31, 137, 199, 97),
                ],
                [
                    (197, 31, 79, 211),
                    (103, 181, 233, 59),
                    (53, 97, 173, 149),
                    (229, 251, 41, 181),
                ],
                [
                    (139, 89, 13, 251),
                    (17, 157, 217, 83),
                    (251, 61, 107, 137),
                    (73, 209, 163, 199),
                ],
            ],
            dtype=np.uint8,
        ),
        colorspace="Non-Color",
        alpha_mode="STRAIGHT",
    )
    cases = (
        ("FLAT", 0.0, (0.173, 0.617, 0.83), (0.0, 0.0, 1.0), False),
        ("FLAT", 0.0, (-0.213, 1.137, 0.31), (0.0, 0.0, 1.0), False),
        ("FLAT", 0.0, (0.917, 0.081, 0.57), (0.0, 0.0, 1.0), False),
        ("FLAT", 0.0, (1.271, -0.191, 0.49), (0.0, 0.0, 1.0), False),
        ("SPHERE", 0.0, (0.83, 0.71, 0.26), (0.0, 0.0, 1.0), False),
        ("SPHERE", 0.0, (0.19, 0.87, 0.63), (0.0, 0.0, 1.0), False),
        ("SPHERE", 0.0, (0.51, 0.49, 0.93), (0.0, 0.0, 1.0), False),
        ("SPHERE", 0.0, (0.5, 0.5, 0.5), (0.0, 0.0, 1.0), False),
        ("TUBE", 0.0, (0.83, 0.71, 0.26), (0.0, 0.0, 1.0), False),
        ("TUBE", 0.0, (0.19, 0.87, 0.63), (0.0, 0.0, 1.0), False),
        ("TUBE", 0.0, (0.51, 0.49, 0.93), (0.0, 0.0, 1.0), False),
        ("TUBE", 0.0, (0.5, 0.5, 0.5), (0.0, 0.0, 1.0), False),
        ("BOX", 0.0, (0.21, 0.73, 0.42), (0.93, 0.21, 0.30), False),
        ("BOX", 0.2, (0.67, 0.18, 0.91), (-0.31, 0.89, 0.34), False),
        ("BOX", 0.55, (0.37, 0.82, 0.14), (0.41, -0.52, 0.75), False),
        # This face is deliberately wound away from the camera. Cycles flips
        # sd->N during shader setup before BOX projection, which changes all
        # three cube-face orientation tests, not merely the blend weights.
        ("BOX", 1.0, (0.76, 0.29, 0.58), (0.58, 0.49, -0.65), True),
    )
    materials = []
    normals = []
    back_facing = []
    for index, (
        projection,
        blend,
        coordinate,
        normal,
        is_back_facing,
    ) in enumerate(cases):
        name = f"Image Projection {index:02d} {projection}"
        material, tree, output = _material(name)
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.name = f"{name} Coordinate"
        mapping.vector_type = "POINT"
        _input(mapping, "Vector").default_value = coordinate
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.name = name
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "REPEAT"
        texture.projection = projection
        texture.projection_blend = blend
        tree.links.new(
            _output(mapping, "Vector"),
            _input(texture, "Vector"),
        )
        channels = tree.nodes.new("ShaderNodeSeparateColor")
        channels.name = f"{name} Channels"
        channels.mode = "RGB"
        tree.links.new(
            _output(texture, "Color"),
            _input(channels, "Color"),
        )
        packed = tree.nodes.new("ShaderNodeCombineColor")
        packed.name = f"{name} RGB Alpha"
        packed.mode = "RGB"
        tree.links.new(
            _output(channels, "Red"),
            _input(packed, "Red"),
        )
        tree.links.new(
            _output(channels, "Green"),
            _input(packed, "Green"),
        )
        tree.links.new(
            _output(texture, "Alpha"),
            _input(packed, "Blue"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"{name} Emission"
        tree.links.new(
            _output(packed, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
        back_facing.append(is_back_facing)
        length = sum(component * component for component in normal) ** 0.5
        normals.append(tuple(component / length for component in normal))

    extent = 1.1
    vertices = []
    faces = []
    for index in range(len(materials)):
        column = index % 4
        row = index // 4
        x0 = -extent + 2.0 * extent * column / 4
        x1 = -extent + 2.0 * extent * (column + 1) / 4
        y0 = -extent + 2.0 * extent * row / 4
        y1 = -extent + 2.0 * extent * (row + 1) / 4
        first = len(vertices)
        vertices.extend(
            (
                (x0, y0, 0.0),
                (x1, y0, 0.0),
                (x1, y1, 0.0),
                (x0, y1, 0.0),
            )
        )
        faces.append(
            (first, first + 3, first + 2, first + 1)
            if back_facing[index]
            else (first, first + 1, first + 2, first + 3)
        )
    mesh = bpy.data.meshes.new("Image Projection Modes Matrix Mesh")
    mesh.from_pydata(vertices, [], faces)
    for material in materials:
        mesh.materials.append(material)
    custom_normals = []
    for index, polygon in enumerate(mesh.polygons):
        polygon.material_index = index
        polygon.use_smooth = True
        custom_normals.extend([normals[index]] * polygon.loop_total)
    mesh.normals_split_custom_set(custom_normals)
    surface = bpy.data.objects.new(
        "Image Projection Modes Matrix", mesh
    )
    scene.collection.objects.link(surface)


def _color_ramp_rgb(scene: Any) -> None:
    """Exercise scene-reachable linear and constant RGB ramps."""
    material, tree, output = _material("RGB Color Ramp Probe")
    coordinates = tree.nodes.new("ShaderNodeTexCoord")
    coordinates.name = "Texture Coordinate"
    gradient = tree.nodes.new("ShaderNodeTexGradient")
    gradient.name = "Linear Gradient"
    gradient.gradient_type = "LINEAR"
    tree.links.new(
        _output(coordinates, "Generated"),
        _input(gradient, "Vector"),
    )

    linear = tree.nodes.new("ShaderNodeValToRGB")
    linear.name = "Linear RGB Ramp"
    linear.color_ramp.color_mode = "RGB"
    linear.color_ramp.interpolation = "LINEAR"
    linear.color_ramp.elements[0].position = 0.13
    linear.color_ramp.elements[0].color = (0.04, 0.17, 0.73, 0.21)
    middle = linear.color_ramp.elements.new(0.47)
    middle.color = (0.82, 0.09, 0.24, 0.68)
    linear.color_ramp.elements[1].position = 0.82
    linear.color_ramp.elements[1].color = (0.15, 0.91, 0.36, 0.94)
    tree.links.new(
        _output(gradient, "Fac"),
        _input(linear, "Fac"),
    )

    constant = tree.nodes.new("ShaderNodeValToRGB")
    constant.name = "Constant RGB Ramp"
    constant.color_ramp.color_mode = "RGB"
    constant.color_ramp.interpolation = "CONSTANT"
    constant.color_ramp.elements[0].position = 0.19
    constant.color_ramp.elements[0].color = (0.91, 0.11, 0.07, 0.32)
    middle = constant.color_ramp.elements.new(0.58)
    middle.color = (0.13, 0.77, 0.31, 0.61)
    constant.color_ramp.elements[1].position = 0.88
    constant.color_ramp.elements[1].color = (0.22, 0.36, 0.95, 0.86)
    tree.links.new(
        _output(gradient, "Fac"),
        _input(constant, "Fac"),
    )

    linear_channels = tree.nodes.new("ShaderNodeSeparateColor")
    linear_channels.mode = "RGB"
    constant_channels = tree.nodes.new("ShaderNodeSeparateColor")
    constant_channels.mode = "RGB"
    tree.links.new(
        _output(linear, "Color"),
        _input(linear_channels, "Color"),
    )
    tree.links.new(
        _output(constant, "Color"),
        _input(constant_channels, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.mode = "RGB"
    tree.links.new(
        _output(linear_channels, "Red"),
        _input(combine, "Red"),
    )
    tree.links.new(
        _output(constant_channels, "Green"),
        _input(combine, "Green"),
    )
    tree.links.new(
        _output(linear, "Alpha"),
        _input(combine, "Blue"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    _plane(material)


def _color_ramp_material(
    name: str,
    factor: float,
    *,
    color_mode: str,
    interpolation: str,
    hue_interpolation: str = "NEAR",
    output_name: str = "Color",
) -> Any:
    material, tree, output = _material(name)
    value = tree.nodes.new("ShaderNodeValue")
    value.name = f"{name} Factor"
    _output(value, "Value").default_value = factor

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.name = name
    ramp.color_ramp.color_mode = color_mode
    ramp.color_ramp.interpolation = interpolation
    ramp.color_ramp.hue_interpolation = hue_interpolation
    elements = ramp.color_ramp.elements
    elements[0].position = 0.11
    elements[0].color = (0.97, 0.025, 0.18, 0.13)
    middle_a = elements.new(0.39)
    middle_a.color = (0.03, 0.88, 0.74, 0.88)
    middle_b = elements.new(0.71)
    middle_b.color = (0.12, 0.055, 0.96, 0.31)
    elements[1].position = 0.91
    elements[1].color = (0.91, 0.72, 0.035, 0.76)
    tree.links.new(
        _output(value, "Value"),
        _input(ramp, "Fac"),
    )

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(ramp, output_name),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _color_ramp_modes(scene: Any) -> None:
    """Exercise every Blender Color Ramp color/interpolation mode."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = [
        ("RGB", interpolation, "NEAR", factor)
        for interpolation, factor in zip(
            ("LINEAR", "CONSTANT", "EASE", "CARDINAL", "B_SPLINE"),
            (0.173, 0.287, 0.443, 0.619, 0.823),
            strict=True,
        )
    ]
    cases.extend(
        ("HSV", "LINEAR", hue, factor)
        for hue, factor in zip(
            ("NEAR", "FAR", "CW", "CCW"),
            (0.241, 0.367, 0.557, 0.779),
            strict=True,
        )
    )
    cases.extend(
        ("HSL", "LINEAR", hue, factor)
        for hue, factor in zip(
            ("NEAR", "FAR", "CW", "CCW"),
            (0.197, 0.331, 0.593, 0.857),
            strict=True,
        )
    )
    # The final three cells verify Cycles' clamping at both ends and a
    # factor exactly on a 1/256 normalized lookup-table sample.
    cases.extend(
        (
            ("RGB", "LINEAR", "NEAR", -0.25),
            ("RGB", "EASE", "NEAR", 1.25),
            ("HSV", "LINEAR", "FAR", 127.0 / 256.0),
        )
    )
    materials = [
        _color_ramp_material(
            f"Color Ramp {index:02d} "
            f"{color_mode} {interpolation} {hue}",
            factor,
            color_mode=color_mode,
            interpolation=interpolation,
            hue_interpolation=hue,
        )
        for index, (
            color_mode,
            interpolation,
            hue,
            factor,
        ) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Color Ramp Modes Matrix",
    )


def _color_ramp_alpha_modes(scene: Any) -> None:
    """Exercise sampled alpha interpolation, clamp, and table boundaries."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ("LINEAR", 0.173),
        ("CONSTANT", 0.287),
        ("EASE", 0.443),
        ("CARDINAL", 0.619),
        ("B_SPLINE", 0.823),
        ("LINEAR", -0.25),
        ("EASE", 1.25),
        ("CARDINAL", 129.0 / 256.0),
    )
    materials = [
        _color_ramp_material(
            f"Color Ramp Alpha {index:02d} {interpolation}",
            factor,
            color_mode="RGB",
            interpolation=interpolation,
            output_name="Alpha",
        )
        for index, (interpolation, factor) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="Color Ramp Alpha Matrix",
    )


def _rgb_curve_material(
    name: str,
    *,
    factor: float,
    color: tuple[float, float, float, float],
    extend: str,
    wide_domain: bool,
) -> Any:
    material, tree, output = _material(name)
    factor_node = tree.nodes.new("ShaderNodeValue")
    factor_node.name = f"{name} Factor"
    _output(factor_node, "Value").default_value = factor
    color_node = tree.nodes.new("ShaderNodeRGB")
    color_node.name = f"{name} Color"
    _output(color_node, "Color").default_value = color

    curves = tree.nodes.new("ShaderNodeRGBCurve")
    curves.name = name
    mapping = curves.mapping
    mapping.extend = extend
    mapping.use_clip = False
    if wide_domain:
        endpoints = (
            (-0.25, 1.15),
            (-0.05, 1.05),
            (-0.10, 1.30),
            (-0.20, 1.20),
        )
    else:
        endpoints = ((0.0, 1.0),) * 4
    shapes = (
        ((0.08, 0.87), (0.31, 0.76), (0.74, 0.21)),
        ((0.17, 0.94), (0.29, 0.16), (0.68, 0.83)),
        ((0.04, 0.72), (0.43, 0.91), (0.79, 0.12)),
        ((0.11, 0.89), (0.37, 0.24), (0.63, 0.78)),
    )
    for curve, (domain_min, domain_max), shape in zip(
        mapping.curves,
        endpoints,
        shapes,
        strict=True,
    ):
        curve.points[0].location = (domain_min, shape[0][1])
        curve.points[-1].location = (domain_max, shape[-1][1])
        domain_range = domain_max - domain_min
        for relative_x, y in shape[1:-1]:
            point = curve.points.new(
                domain_min + relative_x * domain_range,
                y,
            )
            point.handle_type = "AUTO"
    mapping.update()

    tree.links.new(
        _output(factor_node, "Value"),
        _input(curves, "Fac"),
    )
    tree.links.new(
        _output(color_node, "Color"),
        _input(curves, "Color"),
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(curves, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _rgb_curve_matrix(scene: Any) -> None:
    """Cover normalized tables, domains, extrapolation, and Fac mixing."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (1.0, (0.17, 0.43, 0.81, 1.0), "EXTRAPOLATED", False),
        (0.37, (0.71, 0.22, 0.49, 1.0), "EXTRAPOLATED", False),
        (0.0, (0.29, 0.63, 0.11, 1.0), "EXTRAPOLATED", False),
        (1.4, (0.57, 0.08, 0.92, 1.0), "EXTRAPOLATED", False),
        (-0.4, (0.33, 0.77, 0.26, 1.0), "HORIZONTAL", False),
        (1.0, (-0.61, 1.72, 0.46, 1.0), "HORIZONTAL", True),
        (1.0, (-0.61, 1.72, 0.46, 1.0), "EXTRAPOLATED", True),
        (0.58, (1.41, -0.32, 0.73, 1.0), "EXTRAPOLATED", True),
    )
    materials = [
        _rgb_curve_material(
            f"RGB Curve {index:02d} {extend}",
            factor=factor,
            color=color,
            extend=extend,
            wide_domain=wide_domain,
        )
        for index, (
            factor,
            color,
            extend,
            wide_domain,
        ) in enumerate(cases)
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="RGB Curve Matrix",
    )


def _mapping_material(
    name: str,
    vector_type: str,
    *,
    edge_scale: bool,
) -> Any:
    material, tree, output = _material(name)
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.name = name
    mapping.vector_type = vector_type
    _input_identifier(mapping, "Vector").default_value = (
        0.17,
        0.31,
        0.47,
    )
    _input_identifier(mapping, "Location").default_value = (
        0.05,
        0.04,
        0.03,
    )
    _input_identifier(mapping, "Rotation").default_value = (
        0.27,
        -0.19,
        0.33,
    )
    _input_identifier(mapping, "Scale").default_value = (
        (0.0, -0.7, 1.3)
        if edge_scale
        else (1.1, 0.8, 1.3)
    )

    separate = tree.nodes.new("ShaderNodeSeparateColor")
    separate.name = f"{name} Separate"
    separate.mode = "RGB"
    tree.links.new(
        _output(mapping, "Vector"),
        _input(separate, "Color"),
    )
    combine = tree.nodes.new("ShaderNodeCombineColor")
    combine.name = f"{name} Encode"
    combine.mode = "RGB"
    for channel in ("Red", "Green", "Blue"):
        encode = tree.nodes.new("ShaderNodeMath")
        encode.name = f"{name} Encode {channel}"
        encode.operation = "MULTIPLY_ADD"
        tree.links.new(
            _output(separate, channel),
            _input(encode, "Value"),
        )
        encode.inputs[1].default_value = 0.25
        encode.inputs[2].default_value = 0.5
        tree.links.new(
            _output(encode, "Value"),
            _input(combine, channel),
        )
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    tree.links.new(
        _output(combine, "Color"),
        _input(emission, "Color"),
    )
    tree.links.new(
        _output(emission, "Emission"),
        _input(output, "Surface"),
    )
    return material


def _mapping_modes(scene: Any) -> None:
    """Cover all Mapping vector types and zero/negative scale semantics."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    materials = [
        _mapping_material(
            f"Mapping {vector_type} "
            f"{'Edge' if edge_scale else 'Regular'}",
            vector_type,
            edge_scale=edge_scale,
        )
        for edge_scale in (False, True)
        for vector_type in ("POINT", "TEXTURE", "VECTOR", "NORMAL")
    ]
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=2,
        name="Mapping Modes Matrix",
    )


def _checker_texture_matrix(scene: Any) -> None:
    """Cover Checker precision correction, signs, scale, Color, and Fac."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        ((0.0, 0.0, 0.0), 1.0),
        ((1.0, 0.0, 0.0), 1.0),
        ((-1.0, 0.0, 0.0), 1.0),
        ((0.0, 0.0, 1.0), 1.0),
        ((0.5, 0.5, 0.5), 2.0),
        ((-0.5, 0.5, 0.5), 2.0),
        ((1.25, -2.5, 3.75), 0.8),
        ((-1.25, 2.5, -3.75), 0.8),
        ((0.999999, 1.000001, -0.999999), 1.0),
        ((17.0, 18.0, 19.0), 0.25),
        ((-17.0, -18.0, -19.0), 0.25),
        ((0.13, 0.37, 0.91), 7.3),
        ((0.13, 0.37, 0.91), -7.3),
        ((41.0, -29.0, 13.0), 0.0),
        ((1024.25, -2048.5, 4096.75), 0.125),
        ((-0.000001, 0.000001, -0.000001), 1000000.0),
    )
    materials = []
    for index, (coordinate, scale) in enumerate(cases):
        material, tree, output = _material(
            f"Checker {index:02d}"
        )
        coordinate_node = tree.nodes.new("ShaderNodeRGB")
        coordinate_node.name = f"Checker Coordinates {index:02d}"
        _output(coordinate_node, "Color").default_value = (
            *coordinate,
            1.0,
        )
        checker = tree.nodes.new("ShaderNodeTexChecker")
        checker.name = f"Checker {index:02d}"
        tree.links.new(
            _output(coordinate_node, "Color"),
            _input(checker, "Vector"),
        )
        _input(checker, "Color1").default_value = (
            0.13,
            0.37,
            0.79,
            1.0,
        )
        _input(checker, "Color2").default_value = (
            0.83,
            0.61,
            0.17,
            1.0,
        )
        _input(checker, "Scale").default_value = scale
        separate = tree.nodes.new("ShaderNodeSeparateColor")
        separate.name = f"Checker Separate {index:02d}"
        separate.mode = "RGB"
        tree.links.new(
            _output(checker, "Color"),
            _input(separate, "Color"),
        )
        combine = tree.nodes.new("ShaderNodeCombineColor")
        combine.name = f"Checker Encode {index:02d}"
        combine.mode = "RGB"
        tree.links.new(
            _output(separate, "Red"),
            _input(combine, "Red"),
        )
        tree.links.new(
            _output(separate, "Green"),
            _input(combine, "Green"),
        )
        tree.links.new(
            _output(checker, "Fac"),
            _input(combine, "Blue"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Checker Emission {index:02d}"
        tree.links.new(
            _output(combine, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Checker Texture Matrix",
    )


def _fresnel_matrix(scene: Any) -> None:
    """Cover Fresnel socket vectors, IOR clamp, and backfacing eta."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (1.5, None),
        (1.0, (0.0, 0.0, 1.0)),
        (0.0, (0.0, 0.0, 1.0)),
        (0.5, (0.0, 0.0, 1.0)),
        (1.45, (0.07807466, 0.10947394, 0.8824053)),
        (1.33, (0.0, 0.0, 0.0)),
        (2.5, (0.8660254, 0.0, 0.5)),
        (10.0, (1.0, 0.0, 0.0)),
        (1.5, None),
        (0.5, (0.0, 0.0, 1.0)),
        (1.33, (0.21650635, 0.0, 0.125)),
        (2.5, (0.8660254, 0.0, 0.5)),
        (1.0, (1.0, 0.0, 0.0)),
        (0.0, (0.0, 0.0, 0.0)),
        (-2.0, (0.9682458, 0.0, 0.25)),
        (10.0, (0.9999995, 0.0, 0.001)),
    )
    materials = []
    for index, (ior, normal) in enumerate(cases):
        material, tree, output = _material(
            f"Fresnel {index:02d}"
        )
        fresnel = tree.nodes.new("ShaderNodeFresnel")
        fresnel.name = f"Fresnel {index:02d}"
        _input(fresnel, "IOR").default_value = ior
        if normal is not None:
            normal_node = tree.nodes.new("ShaderNodeRGB")
            normal_node.name = f"Fresnel Normal {index:02d}"
            _output(normal_node, "Color").default_value = (
                *normal,
                1.0,
            )
            tree.links.new(
                _output(normal_node, "Color"),
                _input(fresnel, "Normal"),
            )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Fresnel Emission {index:02d}"
        tree.links.new(
            _output(fresnel, "Fac"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Fresnel Matrix",
        backfacing=set(range(8, 16)),
    )


def _layer_weight_matrix(scene: Any) -> None:
    """Cover both outputs, blend branches, normals, and backfacing hits."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (-2.0, None, False),
        (-0.5, (0.0, 0.0, 1.0), False),
        (0.0, (0.0, 0.0, 1.0), False),
        (0.25, (0.4358899, 0.0, 0.9), False),
        (0.499, (0.21650635, 0.0, 0.125), False),
        (0.5, (0.9682458, 0.0, 0.25), False),
        (0.75, (0.9999995, 0.0, 0.001), False),
        (0.99999, (1.0, 0.0, 0.0), False),
        (1.0, (0.0, 0.0, 1.0), False),
        (2.0, (0.4358899, 0.0, 0.9), False),
        (0.25, (0.0, 0.0, 0.0), False),
        (0.75, (-0.9682458, 0.0, -0.25), False),
        (0.0, None, True),
        (0.25, (0.4358899, 0.0, 0.9), True),
        (0.75, (0.9682458, 0.0, 0.25), True),
        (1.0, (0.0, 0.0, 1.0), True),
    )
    materials = []
    backfacing: set[int] = set()
    for case_index, (blend, normal, is_backfacing) in enumerate(cases):
        for output_name in ("Fresnel", "Facing"):
            index = len(materials)
            material, tree, output = _material(
                f"Layer Weight {case_index:02d} {output_name}"
            )
            blend_node = tree.nodes.new("ShaderNodeValue")
            blend_node.name = f"Layer Blend {index:02d}"
            _output(blend_node, "Value").default_value = blend
            layer = tree.nodes.new("ShaderNodeLayerWeight")
            layer.name = f"Layer Weight {index:02d}"
            tree.links.new(
                _output(blend_node, "Value"),
                _input(layer, "Blend"),
            )
            if normal is not None:
                normal_node = tree.nodes.new("ShaderNodeCombineXYZ")
                normal_node.name = f"Layer Normal {index:02d}"
                _input(normal_node, "X").default_value = normal[0]
                _input(normal_node, "Y").default_value = normal[1]
                _input(normal_node, "Z").default_value = normal[2]
                tree.links.new(
                    _output(normal_node, "Vector"),
                    _input(layer, "Normal"),
                )
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.name = f"Layer Emission {index:02d}"
            tree.links.new(
                _output(layer, output_name),
                _input(emission, "Color"),
            )
            tree.links.new(
                _output(emission, "Emission"),
                _input(output, "Surface"),
            )
            if is_backfacing:
                backfacing.add(index)
            materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=8,
        rows=4,
        name="Layer Weight Matrix",
        backfacing=backfacing,
    )


def _map_range_matrix(scene: Any) -> None:
    """Cover scalar/vector Map Range modes, guards, steps, and clamp."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    scalar_cases = (
        ("LINEAR", False, 0.25, 0.0, 1.0, 0.1, 0.9, 4.0),
        ("LINEAR", True, 1.5, 0.0, 1.0, 0.8, 0.2, 4.0),
        ("LINEAR", True, 5.0, 2.0, 2.0, 0.3, 0.7, 4.0),
        ("STEPPED", False, 0.37, 0.0, 1.0, 0.1, 0.9, 4.0),
        ("STEPPED", False, 0.7, 0.0, 1.0, 0.2, 0.8, 0.0),
        ("STEPPED", True, -0.2, 0.0, 1.0, 0.2, 0.8, -3.0),
        ("SMOOTHSTEP", False, 1.5, 0.0, 1.0, 0.1, 0.9, 4.0),
        ("SMOOTHERSTEP", True, 0.25, 1.0, -1.0, 0.9, 0.1, 5.0),
    )
    vector_cases = (
        (
            "LINEAR",
            False,
            (0.25, 0.5, 0.75),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 2.0),
            (0.1, 0.2, 0.3),
            (0.9, 0.8, 0.7),
            (4.0, 4.0, 4.0),
        ),
        (
            "LINEAR",
            True,
            (2.0, -1.0, 0.5),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.8, 0.1, 0.9),
            (0.2, 0.7, 0.3),
            (4.0, 4.0, 4.0),
        ),
        (
            "STEPPED",
            False,
            (0.37, 0.7, 0.7),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.1, 0.2, 0.3),
            (0.9, 0.8, 0.7),
            (4.0, 0.0, -2.0),
        ),
        (
            "STEPPED",
            True,
            (-0.2, 1.4, 0.6),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.2, 0.8, 0.1),
            (0.8, 0.2, 0.9),
            (3.0, 2.0, 5.0),
        ),
        (
            "SMOOTHSTEP",
            False,
            (-1.0, 0.5, 2.0),
            (0.0, 1.0, 2.0),
            (1.0, -1.0, 2.0),
            (0.1, 0.2, 0.3),
            (0.9, 0.8, 0.7),
            (4.0, 4.0, 4.0),
        ),
        (
            "SMOOTHERSTEP",
            True,
            (0.2, 0.5, 0.8),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.9, 0.1, 0.8),
            (0.1, 0.7, 0.2),
            (4.0, 4.0, 4.0),
        ),
        (
            "LINEAR",
            False,
            (0.25, -0.5, 2.0),
            (1.0, -1.0, 3.0),
            (-1.0, 1.0, 1.0),
            (0.2, 0.3, 0.4),
            (0.8, 0.7, 0.6),
            (4.0, 4.0, 4.0),
        ),
        (
            "SMOOTHERSTEP",
            False,
            (-2.0, 0.35, 3.0),
            (0.0, 0.0, 2.0),
            (1.0, 1.0, -2.0),
            (0.8, 0.1, 0.9),
            (0.2, 0.7, 0.3),
            (4.0, 4.0, 4.0),
        ),
    )

    materials = []
    for index, (
        interpolation,
        use_clamp,
        value,
        from_min,
        from_max,
        to_min,
        to_max,
        steps,
    ) in enumerate(scalar_cases):
        material, tree, output = _material(
            f"Map Range Scalar {index:02d}"
        )
        map_range = tree.nodes.new("ShaderNodeMapRange")
        map_range.name = f"Map Range Scalar {index:02d}"
        map_range.data_type = "FLOAT"
        map_range.interpolation_type = interpolation
        map_range.clamp = use_clamp
        for identifier, default in (
            ("Value", value),
            ("From Min", from_min),
            ("From Max", from_max),
            ("To Min", to_min),
            ("To Max", to_max),
            ("Steps", steps),
        ):
            _input_identifier(
                map_range, identifier
            ).default_value = default
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Map Range Scalar Emission {index:02d}"
        tree.links.new(
            _output_identifier(map_range, "Result"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    for local_index, (
        interpolation,
        use_clamp,
        value,
        from_min,
        from_max,
        to_min,
        to_max,
        steps,
    ) in enumerate(vector_cases):
        index = len(scalar_cases) + local_index
        material, tree, output = _material(
            f"Map Range Vector {local_index:02d}"
        )
        map_range = tree.nodes.new("ShaderNodeMapRange")
        map_range.name = f"Map Range Vector {local_index:02d}"
        map_range.data_type = "FLOAT_VECTOR"
        map_range.interpolation_type = interpolation
        map_range.clamp = use_clamp
        for identifier, default in (
            ("Vector", value),
            ("From_Min_FLOAT3", from_min),
            ("From_Max_FLOAT3", from_max),
            ("To_Min_FLOAT3", to_min),
            ("To_Max_FLOAT3", to_max),
            ("Steps_FLOAT3", steps),
        ):
            _input_identifier(
                map_range, identifier
            ).default_value = default
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Map Range Vector Emission {local_index:02d}"
        tree.links.new(
            _output_identifier(map_range, "Vector"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Map Range Matrix",
    )


def _vector_math_matrix(scene: Any) -> None:
    """Cover every Cycles Vector Math operation and guarded edge paths."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    operations = (
        "ADD",
        "SUBTRACT",
        "MULTIPLY",
        "DIVIDE",
        "MULTIPLY_ADD",
        "CROSS_PRODUCT",
        "PROJECT",
        "REFLECT",
        "REFRACT",
        "FACEFORWARD",
        "DOT_PRODUCT",
        "DISTANCE",
        "LENGTH",
        "SCALE",
        "NORMALIZE",
        "ABSOLUTE",
        "POWER",
        "SIGN",
        "MINIMUM",
        "MAXIMUM",
        "FLOOR",
        "CEIL",
        "FRACTION",
        "MODULO",
        "WRAP",
        "SNAP",
        "SINE",
        "COSINE",
        "TANGENT",
    )
    base_a = (0.17, -0.63, 1.2)
    base_b = (0.7, -0.2, 0.5)
    base_c = (-0.4, 0.8, 0.1)
    base_scale = 0.73
    operation_inputs = {
        "POWER": (
            (-2.0, -4.0, 8.0),
            (3.0, 2.0, 0.5),
            base_c,
            base_scale,
        ),
        "SIGN": (
            (-2.0, 0.0, 3.0),
            base_b,
            base_c,
            base_scale,
        ),
        "MODULO": (
            (-0.7, 0.7, 1.5),
            (0.2, -0.2, 0.5),
            base_c,
            base_scale,
        ),
        "WRAP": (
            (-0.2, 2.4, 0.75),
            (1.0, 1.0, 1.0),
            (0.0, -1.0, 0.25),
            base_scale,
        ),
        "SNAP": (
            (0.73, -0.73, 1.4),
            (0.25, -0.2, 0.5),
            base_c,
            base_scale,
        ),
        "REFRACT": (
            (0.1, 0.0, -0.9949874),
            (0.0, 0.0, 2.0),
            base_c,
            0.67,
        ),
    }
    cases = [
        (
            operation,
            operation,
            *operation_inputs.get(
                operation,
                (base_a, base_b, base_c, base_scale),
            ),
        )
        for operation in operations
    ]
    cases.extend(
        (
            (
                "DIVIDE_ZERO_COMPONENTS",
                "DIVIDE",
                (1.0, -2.0, 4.0),
                (0.0, 2.0, 0.0),
                base_c,
                base_scale,
            ),
            (
                "PROJECT_ZERO_AXIS",
                "PROJECT",
                base_a,
                (0.0, 0.0, 0.0),
                base_c,
                base_scale,
            ),
            (
                "REFLECT_ZERO_NORMAL",
                "REFLECT",
                base_a,
                (0.0, 0.0, 0.0),
                base_c,
                base_scale,
            ),
            (
                "REFRACT_TOTAL_INTERNAL_REFLECTION",
                "REFRACT",
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                base_c,
                1.5,
            ),
            (
                "FACEFORWARD_OPPOSITE",
                "FACEFORWARD",
                base_a,
                (0.5, 0.25, 0.75),
                (0.5, 0.25, 0.75),
                base_scale,
            ),
            (
                "NORMALIZE_ZERO",
                "NORMALIZE",
                (0.0, 0.0, 0.0),
                base_b,
                base_c,
                base_scale,
            ),
            (
                "POWER_INVALID_NEGATIVE",
                "POWER",
                (-2.0, -4.0, -8.0),
                (0.5, -0.5, 0.25),
                base_c,
                base_scale,
            ),
            (
                "MODULO_ZERO_DIVISORS",
                "MODULO",
                base_a,
                (0.0, -0.2, 0.0),
                base_c,
                base_scale,
            ),
            (
                "WRAP_ZERO_RANGE",
                "WRAP",
                base_a,
                (1.0, 0.5, -0.25),
                (1.0, -0.5, -0.25),
                base_scale,
            ),
            (
                "SNAP_ZERO_INCREMENT",
                "SNAP",
                base_a,
                (0.0, -0.2, 0.0),
                base_c,
                base_scale,
            ),
            (
                "CROSS_PARALLEL",
                "CROSS_PRODUCT",
                (0.5, -1.0, 2.0),
                (1.0, -2.0, 4.0),
                base_c,
                base_scale,
            ),
        )
    )
    if len(cases) != 40:
        raise RuntimeError(
            f"Vector Math matrix must contain 40 cells, got {len(cases)}"
        )

    scalar_operations = {"DOT_PRODUCT", "DISTANCE", "LENGTH"}
    materials = []
    for index, (label, operation, a, b, c, scale) in enumerate(cases):
        material, tree, output = _material(
            f"Vector Math Matrix {index:02d} {label}"
        )
        vector_math = tree.nodes.new("ShaderNodeVectorMath")
        vector_math.name = f"Vector Math {index:02d} {label}"
        vector_math.operation = operation
        _input_identifier(vector_math, "Vector").default_value = a
        _input_identifier(
            vector_math, "Vector_001"
        ).default_value = b
        _input_identifier(
            vector_math, "Vector_002"
        ).default_value = c
        _input_identifier(vector_math, "Scale").default_value = scale
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Vector Math Emission {index:02d}"
        result_identifier = (
            "Value" if operation in scalar_operations else "Vector"
        )
        tree.links.new(
            _output_identifier(vector_math, result_identifier),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=8,
        rows=5,
        name="Vector Math Matrix",
    )


def _blackbody_matrix(scene: Any) -> None:
    """Cover every Cycles blackbody polynomial interval and clamp."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    temperatures = (
        -100.0,
        0.0,
        799.0,
        800.0,
        964.0,
        965.0,
        1166.0,
        1167.0,
        1449.0,
        1902.0,
        3315.0,
        6365.0,
        6500.0,
        11999.0,
        12000.0,
        20000.0,
    )
    materials = []
    for index, temperature in enumerate(temperatures):
        material, tree, output = _material(
            f"Blackbody Matrix {index:02d} {temperature:g}K"
        )
        value = tree.nodes.new("ShaderNodeValue")
        value.name = f"Temperature {index:02d}"
        _output(value, "Value").default_value = temperature
        blackbody = tree.nodes.new("ShaderNodeBlackbody")
        blackbody.name = f"Blackbody {index:02d}"
        tree.links.new(
            _output(value, "Value"),
            _input(blackbody, "Temperature"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Blackbody Emission {index:02d}"
        tree.links.new(
            _output(blackbody, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Blackbody Matrix",
    )


def _wavelength_matrix(scene: Any) -> None:
    """Cover Cycles CIE interpolation, truncation, and range guards."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    wavelengths = (
        -100.0,
        374.9,
        375.0,
        379.0,
        380.0,
        382.5,
        385.0,
        400.0,
        445.0,
        500.0,
        520.1,
        555.0,
        600.0,
        650.0,
        700.0,
        775.0,
        779.999,
        780.0,
        781.0,
        1000.0,
    )
    materials = []
    for index, wavelength in enumerate(wavelengths):
        material, tree, output = _material(
            f"Wavelength Matrix {index:02d} {wavelength:g}nm"
        )
        value = tree.nodes.new("ShaderNodeValue")
        value.name = f"Wavelength Value {index:02d}"
        _output(value, "Value").default_value = wavelength
        wavelength_node = tree.nodes.new("ShaderNodeWavelength")
        wavelength_node.name = f"Wavelength {index:02d}"
        tree.links.new(
            _output(value, "Value"),
            _input(wavelength_node, "Wavelength"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Wavelength Emission {index:02d}"
        tree.links.new(
            _output(wavelength_node, "Color"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=5,
        rows=4,
        name="Wavelength Matrix",
    )


def _invert_color_matrix(scene: Any) -> None:
    """Cover linked unbounded factors and signed HDR color inputs."""
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    cases = (
        (-2.0, (0.2, 0.4, 0.8, 1.0)),
        (-1.0, (0.2, 0.4, 0.8, 1.0)),
        (0.0, (0.2, 0.4, 0.8, 1.0)),
        (0.25, (0.2, 0.4, 0.8, 1.0)),
        (0.5, (0.2, 0.4, 0.8, 1.0)),
        (1.0, (0.2, 0.4, 0.8, 1.0)),
        (2.0, (0.2, 0.4, 0.8, 1.0)),
        (3.0, (0.2, 0.4, 0.8, 1.0)),
        (-0.5, (-0.3, 1.4, 2.1, 0.2)),
        (0.0, (-0.3, 1.4, 2.1, 0.2)),
        (0.25, (-0.3, 1.4, 2.1, 0.2)),
        (0.75, (-0.3, 1.4, 2.1, 0.2)),
        (1.0, (-0.3, 1.4, 2.1, 0.2)),
        (1.5, (-0.3, 1.4, 2.1, 0.2)),
        (-3.0, (0.2, 0.4, 0.8, 1.0)),
        (4.0, (0.2, 0.4, 0.8, 1.0)),
    )
    materials = []
    for index, (factor, color) in enumerate(cases):
        material, tree, output = _material(
            f"Invert Color Matrix {index:02d}"
        )
        factor_node = tree.nodes.new("ShaderNodeValue")
        factor_node.name = f"Invert Factor {index:02d}"
        _output(factor_node, "Value").default_value = factor
        invert = tree.nodes.new("ShaderNodeInvert")
        invert.name = f"Invert Color {index:02d}"
        _input(invert, "Color").default_value = color
        tree.links.new(
            _output(factor_node, "Value"),
            _input(invert, "Fac"),
        )
        bias = tree.nodes.new("ShaderNodeVectorMath")
        bias.name = f"Invert Positive Bias {index:02d}"
        bias.operation = "ADD"
        _input_identifier(
            bias, "Vector_001"
        ).default_value = (10.0, 10.0, 10.0)
        tree.links.new(
            _output(invert, "Color"),
            _input_identifier(bias, "Vector"),
        )
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = f"Invert Emission {index:02d}"
        tree.links.new(
            _output(bias, "Vector"),
            _input(emission, "Color"),
        )
        tree.links.new(
            _output(emission, "Emission"),
            _input(output, "Surface"),
        )
        materials.append(material)
    _material_matrix(
        scene,
        materials,
        columns=4,
        rows=4,
        name="Invert Color Matrix",
    )
