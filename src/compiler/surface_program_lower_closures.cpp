#include "surface_program_builder.h"

#include <utility>

namespace psycles::compiler::detail {
namespace {

[[nodiscard]] bool enum_token_is(
    std::string_view value,
    std::string_view canonical_lowercase) noexcept {
    if (value.size() != canonical_lowercase.size()) {
        return false;
    }
    for (auto i = std::size_t{0u}; i < value.size(); ++i) {
        auto character = value[i];
        if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
        if (character != canonical_lowercase[i]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] BssrdfMethod bssrdf_method(
    const contract::ShaderNode &node,
    std::string_view property) {
    const auto method = property_string(
        node, property, "RANDOM_WALK");
    if (enum_token_is(method, "burley")) {
        return BssrdfMethod::burley;
    }
    if (enum_token_is(method, "random_walk_legacy")) {
        return BssrdfMethod::random_walk_legacy;
    }
    if (enum_token_is(method, "random_walk_skin")) {
        return BssrdfMethod::random_walk_skin;
    }
    return BssrdfMethod::random_walk;
}

[[nodiscard]] bool normal_uses_bump(
    const ShaderProgram &shader,
    const contract::ShaderNode &node) noexcept {
    const auto normal = node.inputs.find("Normal");
    if (normal == node.inputs.end() ||
        !normal->second.source.has_value()) {
        return false;
    }
    const auto *source = shader.graph().find(
        normal->second.source->node);
    // A missing source is impossible after graph validation. Preserve the
    // conservative scheduling result if a malformed program reaches here.
    return source == nullptr || source->type != node_type::geometry;
}

}// namespace

// Lowers surface and volume closure nodes. A true result means the node
// family was recognized, even when input diagnostics prevented an
// instruction from being emitted.
[[nodiscard]] bool SurfaceProgramBuilder::lower_closure_node(
    const contract::ShaderNode &node) {
    using contract::SocketType;

    if (node.type == node_type::diffuse_bsdf ||
        node.type == node_type::principled_bsdf ||
        node.type == node_type::subsurface_scattering ||
        node.type == node_type::glossy_bsdf ||
        node.type == node_type::metallic_bsdf ||
        node.type == node_type::sheen_bsdf ||
        node.type == node_type::hair_bsdf ||
        node.type == node_type::glass_bsdf ||
        node.type == node_type::refraction_bsdf) {
        const auto standalone_metallic =
            node.type == node_type::metallic_bsdf;
        const auto standalone_sheen =
            node.type == node_type::sheen_bsdf;
        const auto standalone_hair =
            node.type == node_type::hair_bsdf;
        const auto distribution = property_string(
            node,
            "Distribution",
            standalone_sheen ? "MICROFIBER" : "GGX");
        const auto fresnel_type = property_string(
            node, "FresnelType", "F82");
        const auto hair_component = property_string(
            node, "Component", "REFLECTION");
        const auto physical_conductor =
            standalone_metallic &&
            enum_token_is(fresnel_type, "physical_conductor");
        std::optional<ValueExpressionId> color;
        std::optional<ValueExpressionId> metallic_base_ior;
        std::optional<ValueExpressionId> metallic_edge_tint_k;
        if (standalone_metallic) {
            metallic_base_ior = lower_value_input(
                node, physical_conductor ? "IOR" : "BaseColor");
            metallic_edge_tint_k = lower_value_input(
                node, physical_conductor ? "Extinction" : "EdgeTint");
        } else {
            color = lower_value_input(
                node,
                node.type == node_type::principled_bsdf
                    ? "BaseColor"
                    : "Color");
        }
        auto roughness = lower_value_input(
            node, standalone_hair ? "RoughnessU" : "Roughness");
        auto normal = standalone_hair
                          ? std::optional<ValueExpressionId>{}
                          : lower_value_input(node, "Normal");
        std::optional<ValueExpressionId> metallic;
        std::optional<ValueExpressionId> diffuse_roughness;
        std::optional<ValueExpressionId> subsurface_weight;
        std::optional<ValueExpressionId> subsurface_radius;
        std::optional<ValueExpressionId> subsurface_scale;
        std::optional<ValueExpressionId> subsurface_ior;
        std::optional<ValueExpressionId> subsurface_anisotropy;
        std::optional<ValueExpressionId> transmission_weight;
        std::optional<ValueExpressionId> ior;
        std::optional<ValueExpressionId> specular_ior_level;
        std::optional<ValueExpressionId> specular_tint;
        std::optional<ValueExpressionId> microfacet_anisotropy;
        std::optional<ValueExpressionId> microfacet_rotation;
        std::optional<ValueExpressionId> tangent;
        std::optional<ValueExpressionId> hair_offset;
        bool hair_tangent_linked = false;
        std::optional<ValueExpressionId> alpha;
        std::optional<ValueExpressionId> thin_wall;
        std::optional<ValueExpressionId> sheen_weight;
        std::optional<ValueExpressionId> sheen_roughness;
        std::optional<ValueExpressionId> sheen_tint;
        std::optional<ValueExpressionId> coat_weight;
        std::optional<ValueExpressionId> coat_roughness;
        std::optional<ValueExpressionId> coat_ior;
        std::optional<ValueExpressionId> coat_tint;
        std::optional<ValueExpressionId> coat_normal;
        bool coat_normal_linked = false;
        std::optional<ValueExpressionId> emission_color;
        std::optional<ValueExpressionId> emission_strength;
        std::optional<ValueExpressionId> thin_film_thickness;
        std::optional<ValueExpressionId> thin_film_ior;
        if (node.type == node_type::principled_bsdf ||
            node.type == node_type::glass_bsdf ||
            node.type == node_type::refraction_bsdf) {
            ior = lower_value_input(node, "IOR");
        }
        if (node.type == node_type::principled_bsdf) {
            metallic = lower_value_input(node, "Metallic");
            diffuse_roughness =
                lower_value_input(node, "DiffuseRoughness");
            subsurface_weight =
                lower_value_input(node, "SubsurfaceWeight");
            subsurface_radius =
                lower_value_input(node, "SubsurfaceRadius");
            subsurface_scale =
                lower_value_input(node, "SubsurfaceScale");
            subsurface_ior =
                lower_value_input(node, "SubsurfaceIOR");
            subsurface_anisotropy =
                lower_value_input(node, "SubsurfaceAnisotropy");
            transmission_weight =
                lower_value_input(node, "TransmissionWeight");
            specular_ior_level =
                lower_value_input(node, "SpecularIORLevel");
            specular_tint =
                lower_value_input(node, "SpecularTint");
            microfacet_anisotropy =
                lower_value_input(node, "Anisotropic");
            microfacet_rotation =
                lower_value_input(node, "AnisotropicRotation");
            tangent = lower_value_input(node, "Tangent");
            alpha = lower_value_input(node, "Alpha");
            thin_wall = lower_value_input(node, "ThinWall");
            sheen_weight = lower_value_input(node, "SheenWeight");
            sheen_roughness = lower_value_input(node, "SheenRoughness");
            sheen_tint = lower_value_input(node, "SheenTint");
            coat_weight = lower_value_input(node, "CoatWeight");
            coat_roughness = lower_value_input(node, "CoatRoughness");
            coat_ior = lower_value_input(node, "CoatIOR");
            coat_tint = lower_value_input(node, "CoatTint");
            const auto coat_normal_binding =
                node.inputs.find("CoatNormal");
            coat_normal_linked =
                coat_normal_binding != node.inputs.end() &&
                coat_normal_binding->second.source.has_value();
            coat_normal = lower_value_input(node, "CoatNormal");
            emission_color =
                lower_value_input(node, "EmissionColor");
            emission_strength =
                lower_value_input(node, "EmissionStrength");
            thin_film_thickness =
                lower_value_input(node, "ThinFilmThickness");
            thin_film_ior =
                lower_value_input(node, "ThinFilmIOR");
        } else if (node.type == node_type::glass_bsdf) {
            thin_film_thickness =
                lower_value_input(node, "ThinFilmThickness");
            thin_film_ior =
                lower_value_input(node, "ThinFilmIOR");
        } else if (node.type == node_type::subsurface_scattering) {
            subsurface_radius = lower_value_input(node, "Radius");
            subsurface_scale = lower_value_input(node, "Scale");
            subsurface_ior = lower_value_input(node, "IOR");
            subsurface_anisotropy =
                lower_value_input(node, "Anisotropy");
        } else if (node.type == node_type::glossy_bsdf ||
                   node.type == node_type::metallic_bsdf) {
            microfacet_anisotropy =
                lower_value_input(node, "Anisotropy");
            microfacet_rotation =
                lower_value_input(node, "Rotation");
            tangent = lower_value_input(node, "Tangent");
            if (node.type == node_type::metallic_bsdf) {
                thin_film_thickness =
                    lower_value_input(node, "ThinFilmThickness");
                thin_film_ior = lower_value_input(node, "ThinFilmIOR");
            }
        } else if (standalone_hair) {
            diffuse_roughness = lower_value_input(node, "RoughnessV");
            hair_offset = lower_value_input(node, "Offset");
            const auto tangent_binding = node.inputs.find("Tangent");
            hair_tangent_linked =
                tangent_binding != node.inputs.end() &&
                tangent_binding->second.source.has_value();
            tangent = lower_value_input(node, "Tangent");
        }
        if ((standalone_metallic
                 ? metallic_base_ior && metallic_edge_tint_k
                 : color.has_value()) &&
            roughness && (standalone_hair || normal.has_value()) &&
            (node.type != node_type::principled_bsdf ||
             (metallic && diffuse_roughness && subsurface_weight &&
              subsurface_radius && subsurface_scale &&
              subsurface_ior && subsurface_anisotropy &&
              transmission_weight && ior &&
              specular_ior_level && specular_tint && alpha && thin_wall &&
              microfacet_anisotropy && microfacet_rotation && tangent &&
              sheen_weight && sheen_roughness && sheen_tint &&
              coat_weight && coat_roughness && coat_ior && coat_tint &&
              coat_normal &&
              emission_color && emission_strength &&
              thin_film_thickness && thin_film_ior)) &&
            (node.type != node_type::subsurface_scattering ||
                (subsurface_radius && subsurface_scale &&
                 subsurface_ior && subsurface_anisotropy)) &&
            (node.type != node_type::glossy_bsdf ||
             (microfacet_anisotropy && microfacet_rotation && tangent)) &&
            (node.type != node_type::metallic_bsdf ||
             (microfacet_anisotropy && microfacet_rotation && tangent &&
              thin_film_thickness && thin_film_ior)) &&
            (!standalone_hair ||
             (diffuse_roughness && hair_offset && tangent)) &&
            ((node.type != node_type::glass_bsdf &&
                 node.type != node_type::refraction_bsdf) ||
                ior) &&
            (node.type != node_type::glass_bsdf ||
             (thin_film_thickness && thin_film_ior))) {
            auto operation = ClosureOperation::diffuse;
            if (node.type == node_type::principled_bsdf) {
                operation = ClosureOperation::principled;
            } else if (node.type == node_type::subsurface_scattering) {
                operation = ClosureOperation::subsurface;
            } else if (node.type == node_type::glass_bsdf) {
                operation = ClosureOperation::glass;
            } else if (node.type == node_type::refraction_bsdf) {
                operation = ClosureOperation::refraction;
            } else if (node.type == node_type::glossy_bsdf) {
                operation = ClosureOperation::glossy;
            } else if (node.type == node_type::metallic_bsdf) {
                operation = physical_conductor
                                ? ClosureOperation::metallic_conductor
                                : ClosureOperation::metallic_f82;
            } else if (node.type == node_type::sheen_bsdf) {
                operation = enum_token_is(distribution, "ashikhmin")
                                ? ClosureOperation::sheen_ashikhmin
                                : ClosureOperation::sheen_microfiber;
            } else if (standalone_hair) {
                operation = enum_token_is(hair_component, "transmission")
                                ? ClosureOperation::hair_transmission
                                : ClosureOperation::hair_reflection;
            }
            publish(
                node.id,
                "Closure",
                append(ClosureInstruction{
                    .operation = operation,
                    .source_node = node.id,
                    .color = color.value_or(ValueExpressionId{}),
                    .normal = normal.value_or(ValueExpressionId{}),
                    .normal_uses_bump =
                        !standalone_hair && normal_uses_bump(_shader, node),
                    .roughness = *roughness,
                    .diffuse_roughness =
                        diffuse_roughness.value_or(
                            ValueExpressionId{}),
                    .subsurface_weight =
                        subsurface_weight.value_or(
                            ValueExpressionId{}),
                    .subsurface_radius =
                        subsurface_radius.value_or(
                            ValueExpressionId{}),
                    .subsurface_scale =
                        subsurface_scale.value_or(
                            ValueExpressionId{}),
                    .subsurface_ior =
                        subsurface_ior.value_or(
                            ValueExpressionId{}),
                    .subsurface_anisotropy =
                        subsurface_anisotropy.value_or(
                            ValueExpressionId{}),
                    .subsurface_method = bssrdf_method(
                        node,
                        node.type == node_type::principled_bsdf
                            ? "SubsurfaceMethod"
                            : "Method"),
                    .transmission_weight =
                        transmission_weight.value_or(
                            ValueExpressionId{}),
                    .metallic =
                        metallic.value_or(ValueExpressionId{}),
                    .ior = ior.value_or(ValueExpressionId{}),
                    .specular_ior_level =
                        specular_ior_level.value_or(
                            ValueExpressionId{}),
                    .specular_tint =
                        specular_tint.value_or(
                            ValueExpressionId{}),
                    .metallic_base_ior =
                        metallic_base_ior.value_or(
                            ValueExpressionId{}),
                    .metallic_edge_tint_k =
                        metallic_edge_tint_k.value_or(
                            ValueExpressionId{}),
                    .microfacet_anisotropy =
                        microfacet_anisotropy.value_or(
                            ValueExpressionId{}),
                    .microfacet_rotation =
                        microfacet_rotation.value_or(
                            ValueExpressionId{}),
                    .tangent = tangent.value_or(ValueExpressionId{}),
                    .hair_offset =
                        hair_offset.value_or(ValueExpressionId{}),
                    .hair_tangent_linked = hair_tangent_linked,
                    .alpha = alpha.value_or(ValueExpressionId{}),
                    .thin_wall =
                        thin_wall.value_or(ValueExpressionId{}),
                    .sheen_weight =
                        sheen_weight.value_or(ValueExpressionId{}),
                    .sheen_roughness =
                        sheen_roughness.value_or(ValueExpressionId{}),
                    .sheen_tint =
                        sheen_tint.value_or(ValueExpressionId{}),
                    .coat_weight =
                        coat_weight.value_or(ValueExpressionId{}),
                    .coat_roughness =
                        coat_roughness.value_or(ValueExpressionId{}),
                    .coat_ior =
                        coat_ior.value_or(ValueExpressionId{}),
                    .coat_tint =
                        coat_tint.value_or(ValueExpressionId{}),
                    .coat_normal =
                        coat_normal.value_or(ValueExpressionId{}),
                    .coat_normal_linked = coat_normal_linked,
                    .emission_color =
                        emission_color.value_or(
                            ValueExpressionId{}),
                    .emission_strength =
                        emission_strength.value_or(
                            ValueExpressionId{}),
                    .thin_film_thickness =
                        thin_film_thickness.value_or(
                            ValueExpressionId{}),
                    .thin_film_ior =
                        thin_film_ior.value_or(
                            ValueExpressionId{}),
                    .preserve_ggx_energy =
                        enum_token_is(distribution, "multi_ggx"),
                    .beckmann =
                        enum_token_is(distribution, "beckmann")}));
        }
        return true;
    }
    if (node.type == node_type::translucent_bsdf) {
        auto color = lower_value_input(node, "Color");
        auto normal = lower_value_input(node, "Normal");
        if (color && normal) {
            publish(
                node.id,
                "Closure",
                append(ClosureInstruction{
                    .operation = ClosureOperation::translucent,
                    .source_node = node.id,
                    .color = *color,
                    .normal = *normal}));
        }
        return true;
    }
    if (node.type == node_type::emission ||
        node.type == node_type::background) {
        // Cycles' background_setup() and emission_setup() are identical for
        // surface evaluation. Keep the legacy surface-only execution path
        // compatible while the SVM path retains the distinct opcodes (and
        // therefore the volume-density distinction) used by Cycles.
        auto color = lower_value_input(node, "Color");
        auto strength = lower_value_input(node, "Strength");
        if (color && strength) {
            publish(
                node.id,
                "Closure",
                append(ClosureInstruction{
                    .operation = ClosureOperation::emission,
                    .source_node = node.id,
                    .color = *color,
                    .strength = *strength}));
        }
        return true;
    }
    if (node.type == node_type::transparent_bsdf) {
        if (auto color = lower_value_input(node, "Color")) {
            publish(
                node.id,
                "Closure",
                append(ClosureInstruction{
                    .operation = ClosureOperation::transparent,
                    .source_node = node.id,
                    .color = *color}));
        }
        return true;
    }
    if (node.type == node_type::null_closure) {
        publish(
            node.id,
            "Closure",
            append(ClosureInstruction{
                .operation = ClosureOperation::null_closure,
                .source_node = node.id}));
        return true;
    }
    if (node.type == node_type::add_closure ||
        node.type == node_type::mix_closure) {
        auto a = lower_closure_input(node, "A");
        auto b = lower_closure_input(node, "B");
        std::optional<ValueExpressionId> factor;
        if (node.type == node_type::mix_closure) {
            factor = lower_value_input(node, "Factor");
        }
        if (a && b &&
            (node.type == node_type::add_closure || factor)) {
            publish(
                node.id,
                "Closure",
                append(ClosureInstruction{
                    .operation =
                        node.type == node_type::add_closure
                            ? ClosureOperation::add
                            : ClosureOperation::mix,
                    .source_node = node.id,
                    .factor =
                        factor.value_or(ValueExpressionId{}),
                    .a = *a,
                    .b = *b}));
        }
        return true;
    }
    if (node.type == node_type::volume_absorption) {
        auto color = lower_value_input(node, "Color");
        auto density = lower_value_input(node, "Density");
        if (color && density) {
            publish(
                node.id,
                "Volume",
                append(VolumeInstruction{
                    .operation =
                        VolumeOperation::absorption,
                    .source_node = node.id,
                    .color = *color,
                    .density = *density}));
        }
        return true;
    }
    if (node.type == node_type::volume_scatter) {
        auto color = lower_value_input(node, "Color");
        auto density = lower_value_input(node, "Density");
        auto anisotropy =
            lower_value_input(node, "Anisotropy");
        auto ior = lower_value_input(node, "IOR");
        auto backscatter =
            lower_value_input(node, "Backscatter");
        auto alpha = lower_value_input(node, "Alpha");
        auto diameter =
            lower_value_input(node, "Diameter");
        if (color && density && anisotropy && ior &&
            backscatter && alpha && diameter) {
            publish(
                node.id,
                "Volume",
                append(VolumeInstruction{
                    .operation = VolumeOperation::scatter,
                    .source_node = node.id,
                    .color = *color,
                    .density = *density,
                    .anisotropy = *anisotropy,
                    .ior = *ior,
                    .backscatter = *backscatter,
                    .alpha = *alpha,
                    .diameter = *diameter,
                    .phase = volume_phase(node)}));
        }
        return true;
    }
    if (node.type == node_type::volume_coefficients) {
        auto scatter = lower_value_input(
            node, "ScatterCoefficients");
        auto absorption = lower_value_input(
            node, "AbsorptionCoefficients");
        auto anisotropy =
            lower_value_input(node, "Anisotropy");
        auto ior = lower_value_input(node, "IOR");
        auto backscatter =
            lower_value_input(node, "Backscatter");
        auto alpha = lower_value_input(node, "Alpha");
        auto diameter =
            lower_value_input(node, "Diameter");
        auto emission = lower_value_input(
            node, "EmissionCoefficients");
        if (scatter && absorption && anisotropy && ior &&
            backscatter && alpha && diameter && emission) {
            publish(
                node.id,
                "Volume",
                append(VolumeInstruction{
                    .operation =
                        VolumeOperation::coefficients,
                    .source_node = node.id,
                    .anisotropy = *anisotropy,
                    .ior = *ior,
                    .backscatter = *backscatter,
                    .alpha = *alpha,
                    .diameter = *diameter,
                    .scatter_coefficients = *scatter,
                    .absorption_coefficients =
                        *absorption,
                    .emission_coefficients = *emission,
                    .phase = volume_phase(node)}));
        }
        return true;
    }
    if (node.type == node_type::volume_emission) {
        auto color = lower_value_input(node, "Color");
        auto strength = lower_value_input(node, "Strength");
        if (color && strength) {
            publish(
                node.id,
                "Volume",
                append(VolumeInstruction{
                    .operation = VolumeOperation::emission,
                    .source_node = node.id,
                    .color = *color,
                    .emission_strength = *strength}));
        }
        return true;
    }
    if (node.type == node_type::principled_volume) {
        auto color = lower_value_input(node, "Color");
        auto density = lower_value_input(node, "Density");
        auto anisotropy =
            lower_value_input(node, "Anisotropy");
        auto absorption =
            lower_value_input(node, "AbsorptionColor");
        auto emission_strength =
            lower_value_input(node, "EmissionStrength");
        auto emission_color =
            lower_value_input(node, "EmissionColor");
        auto blackbody_intensity =
            lower_value_input(node, "BlackbodyIntensity");
        auto blackbody_tint =
            lower_value_input(node, "BlackbodyTint");
        auto temperature =
            lower_value_input(node, "Temperature");
        if (color && density && anisotropy && absorption &&
            emission_strength && emission_color &&
            blackbody_intensity && blackbody_tint &&
            temperature) {
            publish(
                node.id,
                "Volume",
                append(VolumeInstruction{
                    .operation =
                        VolumeOperation::principled,
                    .source_node = node.id,
                    .color = *color,
                    .density = *density,
                    .anisotropy = *anisotropy,
                    .absorption_color = *absorption,
                    .emission_strength =
                        *emission_strength,
                    .emission_color = *emission_color,
                    .blackbody_intensity =
                        *blackbody_intensity,
                    .blackbody_tint = *blackbody_tint,
                    .temperature = *temperature,
                    .phase =
                        VolumePhase::henyey_greenstein}));
        }
        return true;
    }
    if (node.type == node_type::null_volume) {
        publish(
            node.id,
            "Volume",
            append(VolumeInstruction{
                .operation =
                    VolumeOperation::null_volume,
                .source_node = node.id}));
        return true;
    }
    if (node.type == node_type::add_volume ||
        node.type == node_type::mix_volume) {
        auto a = lower_volume_input(node, "A");
        auto b = lower_volume_input(node, "B");
        std::optional<ValueExpressionId> factor;
        if (node.type == node_type::mix_volume) {
            factor = lower_value_input(node, "Factor");
        }
        if (a && b &&
            (node.type == node_type::add_volume || factor)) {
            publish(
                node.id,
                "Volume",
                append(VolumeInstruction{
                    .operation =
                        node.type == node_type::add_volume
                            ? VolumeOperation::add
                            : VolumeOperation::mix,
                    .source_node = node.id,
                    .factor =
                        factor.value_or(ValueExpressionId{}),
                    .a = *a,
                    .b = *b}));
        }
        return true;
    }
    return false;
}

}// namespace psycles::compiler::detail
