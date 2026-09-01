"""Principled closure probes kept separate from the near-2k core module."""

from __future__ import annotations

from typing import Any

from .support import _input, _material, _material_matrix, _output


def _set_color(socket: Any, value: tuple[float, float, float]) -> None:
    socket.default_value = (*value, 1.0)


def _principled_svm_oracle(scene: Any) -> None:
    """Finite all-field product for Cycles 5.2 Principled word images.

    Inputs stay literal so the dump exposes Cycles' automatic default normal
    and tangent topology together with the complete typed 176-byte payload.
    The default case freezes Cycles' node-type defaults. Two all-field cases
    then cover both distribution enums, two subsurface enums, both Thin Wall
    values, and deliberately non-default values in every reachable field
    across the product.
    """
    cases = (
        {
            "distribution": "MULTI_GGX",
            "subsurface_method": "RANDOM_WALK_SKIN",
            "base_color": (0.23, 0.41, 0.67),
            "metallic": 0.37,
            "roughness": 0.29,
            "ior": 1.47,
            "alpha": 0.73,
            "thin_wall": False,
            "diffuse_roughness": 0.18,
            "subsurface_weight": 0.26,
            "subsurface_radius": (1.20, 0.34, 0.08),
            "subsurface_scale": 0.013,
            "subsurface_ior": 1.38,
            "subsurface_anisotropy": 0.22,
            "specular_ior_level": 0.61,
            "specular_tint": (0.81, 0.93, 0.57),
            "anisotropic": 0.44,
            "anisotropic_rotation": 0.17,
            "transmission_weight": 0.31,
            "coat_weight": 0.28,
            "coat_roughness": 0.12,
            "coat_ior": 1.39,
            "coat_tint": (0.95, 0.72, 0.51),
            "sheen_weight": 0.19,
            "sheen_roughness": 0.63,
            "sheen_tint": (0.36, 0.84, 0.58),
            "emission_color": (0.11, 0.32, 0.77),
            "emission_strength": 1.70,
            "thin_film_thickness": 370.0,
            "thin_film_ior": 1.58,
        },
        {
            "distribution": "GGX",
            "subsurface_method": "BURLEY",
            "base_color": (0.71, 0.19, 0.055),
            "metallic": 0.08,
            "roughness": 0.54,
            "ior": 1.22,
            "alpha": 0.46,
            "thin_wall": True,
            "diffuse_roughness": 0.43,
            "subsurface_weight": 0.57,
            "subsurface_radius": (0.27, 0.83, 1.41),
            "subsurface_scale": 0.027,
            "subsurface_anisotropy": -0.31,
            "specular_ior_level": 0.34,
            "specular_tint": (0.42, 0.68, 0.97),
            "anisotropic": 0.23,
            "anisotropic_rotation": 0.39,
            "transmission_weight": 0.64,
            "coat_weight": 0.41,
            "coat_roughness": 0.27,
            "coat_ior": 1.61,
            "coat_tint": (0.62, 0.88, 0.73),
            "sheen_weight": 0.52,
            "sheen_roughness": 0.36,
            "sheen_tint": (0.91, 0.24, 0.47),
            "emission_color": (0.63, 0.14, 0.29),
            "emission_strength": 0.82,
            "thin_film_thickness": 515.0,
            "thin_film_ior": 1.31,
        },
    )
    default_material, default_tree, default_output = _material(
        "Principled SVM Oracle 00 Defaults"
    )
    default_principled = default_tree.nodes.new("ShaderNodeBsdfPrincipled")
    default_principled.name = "Principled SVM Oracle 00 Defaults"
    default_tree.links.new(
        _output(default_principled, "BSDF"),
        _input(default_output, "Surface"),
    )
    materials = [default_material]

    for index, case in enumerate(cases, start=1):
        material, tree, output = _material(
            f"Principled SVM Oracle {index:02d}"
        )
        principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
        principled.name = f"Principled SVM Oracle {index:02d}"
        principled.distribution = case["distribution"]
        principled.subsurface_method = case["subsurface_method"]

        _set_color(_input(principled, "Base Color"), case["base_color"])
        _input(principled, "Metallic").default_value = case["metallic"]
        _input(principled, "Roughness").default_value = case["roughness"]
        _input(principled, "IOR").default_value = case["ior"]
        _input(principled, "Alpha").default_value = case["alpha"]
        _input(principled, "Thin Wall").default_value = case["thin_wall"]
        _input(principled, "Diffuse Roughness").default_value = case[
            "diffuse_roughness"
        ]
        _input(principled, "Subsurface Weight").default_value = case[
            "subsurface_weight"
        ]
        _input(principled, "Subsurface Radius").default_value = case[
            "subsurface_radius"
        ]
        _input(principled, "Subsurface Scale").default_value = case[
            "subsurface_scale"
        ]
        # Cycles exposes this socket only for RANDOM_WALK_SKIN. The other
        # methods still encode the node-type default in their typed payload.
        if case["subsurface_method"] == "RANDOM_WALK_SKIN":
            _input(principled, "Subsurface IOR").default_value = case[
                "subsurface_ior"
            ]
        _input(principled, "Subsurface Anisotropy").default_value = case[
            "subsurface_anisotropy"
        ]
        _input(principled, "Specular IOR Level").default_value = case[
            "specular_ior_level"
        ]
        _set_color(_input(principled, "Specular Tint"), case["specular_tint"])
        _input(principled, "Anisotropic").default_value = case["anisotropic"]
        _input(principled, "Anisotropic Rotation").default_value = case[
            "anisotropic_rotation"
        ]
        _input(principled, "Transmission Weight").default_value = case[
            "transmission_weight"
        ]
        _input(principled, "Coat Weight").default_value = case["coat_weight"]
        _input(principled, "Coat Roughness").default_value = case[
            "coat_roughness"
        ]
        _input(principled, "Coat IOR").default_value = case["coat_ior"]
        _set_color(_input(principled, "Coat Tint"), case["coat_tint"])
        _input(principled, "Sheen Weight").default_value = case["sheen_weight"]
        _input(principled, "Sheen Roughness").default_value = case[
            "sheen_roughness"
        ]
        _set_color(_input(principled, "Sheen Tint"), case["sheen_tint"])
        _set_color(
            _input(principled, "Emission Color"), case["emission_color"]
        )
        _input(principled, "Emission Strength").default_value = case[
            "emission_strength"
        ]
        _input(principled, "Thin Film Thickness").default_value = case[
            "thin_film_thickness"
        ]
        _input(principled, "Thin Film IOR").default_value = case[
            "thin_film_ior"
        ]
        tree.links.new(
            _output(principled, "BSDF"), _input(output, "Surface")
        )
        materials.append(material)

    _material_matrix(
        scene,
        materials,
        columns=3,
        rows=1,
        name="Principled SVM Oracle",
    )


def _principled_sheen_svm_oracle(scene: Any) -> None:
    """Isolate Cycles' ordered Principled sheen transition.

    IOR 1 removes the dielectric closure and every other optional layer is
    zero.  Transparency, emission, and diffuse remain live so the closure
    trace observes both sheen allocation and its attenuation of all lower
    layers without relying on a pre-evaluated material result.
    """
    material, tree, output = _material("Principled Sheen SVM Oracle")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Principled Sheen SVM Oracle"
    _set_color(_input(principled, "Base Color"), (0.31, 0.57, 0.83))
    _input(principled, "IOR").default_value = 1.0
    _input(principled, "Roughness").default_value = 0.42
    _input(principled, "Alpha").default_value = 0.8
    _input(principled, "Sheen Weight").default_value = 0.6
    _input(principled, "Sheen Roughness").default_value = 0.37
    _set_color(_input(principled, "Sheen Tint"), (0.25, 0.70, 0.45))
    _set_color(_input(principled, "Emission Color"), (0.20, 0.10, 0.60))
    _input(principled, "Emission Strength").default_value = 1.3
    tree.links.new(
        _output(principled, "BSDF"),
        _input(output, "Surface"),
    )
    _material_matrix(
        scene,
        [material],
        columns=1,
        rows=1,
        name="Principled Sheen SVM Oracle",
    )


def _principled_coat_svm_oracle(scene: Any) -> None:
    """Isolate Cycles' ordered Principled coat transition.

    IOR 1 removes the lower dielectric closure and every other optional layer
    is zero. Transparency, colored coat attenuation, emission, and diffuse
    remain live so the trace observes both the dielectric GGX closure and the
    Beer-law transmission applied to every lower layer.
    """
    material, tree, output = _material("Principled Coat SVM Oracle")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Principled Coat SVM Oracle"
    _set_color(_input(principled, "Base Color"), (0.31, 0.57, 0.83))
    _input(principled, "IOR").default_value = 1.0
    _input(principled, "Roughness").default_value = 0.42
    _input(principled, "Alpha").default_value = 0.8
    _input(principled, "Coat Weight").default_value = 0.65
    _input(principled, "Coat Roughness").default_value = 0.28
    _input(principled, "Coat IOR").default_value = 1.45
    _set_color(_input(principled, "Coat Tint"), (0.70, 0.85, 0.55))
    _set_color(_input(principled, "Emission Color"), (0.20, 0.10, 0.60))
    _input(principled, "Emission Strength").default_value = 1.3
    tree.links.new(
        _output(principled, "BSDF"),
        _input(output, "Surface"),
    )
    _material_matrix(
        scene,
        [material],
        columns=1,
        rows=1,
        name="Principled Coat SVM Oracle",
    )


def _principled_metallic_svm_oracle(scene: Any) -> None:
    """Isolate Cycles' ordered Principled metallic transition.

    Sheen, coat, transmission, and subsurface are zero. The remaining
    transparency, emission, dielectric, and diffuse transitions make the
    trace expose that Metallic allocates an F82 GGX closure and attenuates
    only the lower BSDF components, independent of closure allocation.
    Multi-GGX, anisotropy, rotation, and thin film keep every Metallic setup
    branch live without introducing a second material representation.
    """
    material, tree, output = _material("Principled Metallic SVM Oracle")
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.name = "Principled Metallic SVM Oracle"
    principled.distribution = "MULTI_GGX"
    _set_color(_input(principled, "Base Color"), (0.18, 0.55, 0.82))
    _input(principled, "Metallic").default_value = 0.64
    _input(principled, "Roughness").default_value = 0.47
    _input(principled, "IOR").default_value = 1.37
    _input(principled, "Alpha").default_value = 0.78
    _set_color(_input(principled, "Specular Tint"), (0.72, 0.35, 0.91))
    _input(principled, "Anisotropic").default_value = 0.38
    _input(principled, "Anisotropic Rotation").default_value = 0.21
    _set_color(_input(principled, "Emission Color"), (0.22, 0.07, 0.48))
    _input(principled, "Emission Strength").default_value = 1.25
    _input(principled, "Thin Film Thickness").default_value = 360.0
    _input(principled, "Thin Film IOR").default_value = 1.52
    tree.links.new(
        _output(principled, "BSDF"),
        _input(output, "Surface"),
    )
    _material_matrix(
        scene,
        [material],
        columns=1,
        rows=1,
        name="Principled Metallic SVM Oracle",
    )
