#include "surface_program_builder.h"

#include <utility>

namespace psycles::compiler::detail {

// Lowers surface and volume closure nodes. A true result means the node
// family was recognized, even when input diagnostics prevented an
// instruction from being emitted.
[[nodiscard]] bool SurfaceProgramBuilder::lower_closure_node(
    const contract::ShaderNode &node) {
    using contract::SocketType;

    if (node.type == node_type::diffuse_bsdf ||
        node.type == node_type::principled_bsdf ||
        node.type == node_type::glossy_bsdf ||
        node.type == node_type::glass_bsdf) {
        const auto color_name =
            node.type == node_type::principled_bsdf
                ? "BaseColor"
                : "Color";
        auto color = lower_value_input(node, color_name);
        auto roughness =
            lower_value_input(node, "Roughness");
        auto normal = lower_value_input(node, "Normal");
        std::optional<ValueExpressionId> metallic;
        std::optional<ValueExpressionId> diffuse_roughness;
        std::optional<ValueExpressionId> subsurface_weight;
        std::optional<ValueExpressionId> subsurface_radius;
        std::optional<ValueExpressionId> subsurface_scale;
        std::optional<ValueExpressionId> ior;
        std::optional<ValueExpressionId> specular_ior_level;
        std::optional<ValueExpressionId> specular_tint;
        std::optional<ValueExpressionId> alpha;
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
        if (node.type == node_type::principled_bsdf ||
            node.type == node_type::glass_bsdf) {
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
            specular_ior_level =
                lower_value_input(node, "SpecularIORLevel");
            specular_tint =
                lower_value_input(node, "SpecularTint");
            alpha = lower_value_input(node, "Alpha");
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
        }
        if (color && roughness && normal &&
            (node.type != node_type::principled_bsdf ||
             (metallic && diffuse_roughness && subsurface_weight &&
              subsurface_radius && subsurface_scale && ior &&
              specular_ior_level && specular_tint && alpha &&
              sheen_weight && sheen_roughness && sheen_tint &&
              coat_weight && coat_roughness && coat_ior && coat_tint &&
              coat_normal &&
              emission_color && emission_strength)) &&
            (node.type != node_type::glass_bsdf || ior)) {
            publish(
                node.id,
                "Closure",
                append(ClosureInstruction{
                    .operation =
                        node.type == node_type::principled_bsdf
                            ? ClosureOperation::principled
                            : node.type == node_type::glass_bsdf
                                  ? ClosureOperation::glass
                            : node.type ==
                                      node_type::glossy_bsdf
                                  ? ClosureOperation::glossy
                                  : ClosureOperation::diffuse,
                    .source_node = node.id,
                    .color = *color,
                    .normal = *normal,
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
                    .metallic =
                        metallic.value_or(ValueExpressionId{}),
                    .ior = ior.value_or(ValueExpressionId{}),
                    .specular_ior_level =
                        specular_ior_level.value_or(
                            ValueExpressionId{}),
                    .specular_tint =
                        specular_tint.value_or(
                            ValueExpressionId{}),
                    .alpha = alpha.value_or(ValueExpressionId{}),
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
                    .preserve_ggx_energy =
                        property_string(
                            node, "Distribution", "GGX") ==
                        "MULTI_GGX",
                    .beckmann =
                        property_string(
                            node, "Distribution", "GGX") ==
                        "BECKMANN"}));
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
    if (node.type == node_type::emission) {
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
