#include "blender_graph_lowering_component.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace psycles::adapter::detail {
namespace {

class ClosureNodeLoweringComponent final
    : public BlenderNodeLoweringComponent {

public:
    [[nodiscard]] std::optional<TypedOutput> lower(
        BlenderNodeLoweringContext &context,
        const BlenderNodeLoweringRequest &request) const override {
        using contract::SocketType;
        [[maybe_unused]] const auto &node_name = request.node_name;
        [[maybe_unused]] const auto &socket = request.socket;
        [[maybe_unused]] const auto requested = request.requested;
        [[maybe_unused]] auto *node = request.node;
        [[maybe_unused]] const auto &type = request.type;
        [[maybe_unused]] const auto &finish = request.finish;
        if (type == "BSDF_PRINCIPLED") {
            const auto id = context.graph().add_node(
                compiler::node_type::principled_bsdf,
                node_name);
            static_cast<void>(context.graph().set_property(
                id,
                "Distribution",
                SocketValue::string(context.node_property_text(
                    node, "distribution", "GGX"))));
            static_cast<void>(context.bind(
                id,
                "BaseColor",
                node,
                "Base Color",
                SocketType::color));
            static_cast<void>(context.bind(
                id,
                "Metallic",
                node,
                "Metallic",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "Roughness",
                node,
                "Roughness",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "DiffuseRoughness",
                node,
                "Diffuse Roughness",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "SubsurfaceWeight",
                node,
                "Subsurface Weight",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "SubsurfaceRadius",
                node,
                "Subsurface Radius",
                SocketType::vector));
            static_cast<void>(context.bind(
                id,
                "SubsurfaceScale",
                node,
                "Subsurface Scale",
                SocketType::floating));
            auto *subsurface_weight =
                context.raw_input(node, "Subsurface Weight");
            const auto uses_subsurface =
                subsurface_weight != nullptr &&
                (boolean(member(subsurface_weight, "linked")) ||
                    number(member(subsurface_weight, "default")) >
                        1.0e-5f);
            if (uses_subsurface) {
                context.warn_once(
                    "principled-subsurface-approximation:" + node_name,
                    "Principled subsurface transport uses a "
                    "radius-weighted diffuse approximation; "
                    "spatial BSSRDF random walks are not yet implemented");
            }
            static_cast<void>(context.bind(
                id, "IOR", node, "IOR", SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "SpecularIORLevel",
                node,
                "Specular IOR Level",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "SpecularTint",
                node,
                "Specular Tint",
                SocketType::color));
            static_cast<void>(context.bind(
                id,
                "EmissionColor",
                node,
                "Emission Color",
                SocketType::color));
            static_cast<void>(context.bind(
                id,
                "EmissionStrength",
                node,
                "Emission Strength",
                SocketType::floating));
            static_cast<void>(context.bind(
                id, "Normal", node, "Normal", SocketType::normal));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "BSDF_DIFFUSE") {
            const auto id = context.graph().add_node(
                compiler::node_type::diffuse_bsdf,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.bind(
                id,
                "Roughness",
                node,
                "Roughness",
                SocketType::floating));
            static_cast<void>(context.bind(
                id, "Normal", node, "Normal", SocketType::normal));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "BSDF_TRANSLUCENT") {
            const auto id = context.graph().add_node(
                compiler::node_type::translucent_bsdf,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.bind(
                id, "Normal", node, "Normal", SocketType::normal));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "BSDF_GLOSSY") {
            const auto id = context.graph().add_node(
                compiler::node_type::glossy_bsdf,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.bind(
                id,
                "Roughness",
                node,
                "Roughness",
                SocketType::floating));
            if (context.raw_input(node, "Normal") != nullptr) {
                static_cast<void>(context.bind(
                    id,
                    "Normal",
                    node,
                    "Normal",
                    SocketType::normal));
            } else {
                static_cast<void>(context.graph().connect(
                    context.geometry_output(
                        "Normal", SocketType::normal)
                        .ref,
                    id,
                    "Normal"));
            }
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "BSDF_GLASS") {
            const auto id = context.graph().add_node(
                compiler::node_type::glass_bsdf,
                node_name);
            static_cast<void>(context.graph().set_property(
                id,
                "Distribution",
                SocketValue::string(context.node_property_text(
                    node, "distribution", "GGX"))));
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.bind(
                id,
                "Roughness",
                node,
                "Roughness",
                SocketType::floating));
            static_cast<void>(context.bind(
                id, "IOR", node, "IOR", SocketType::floating));
            if (context.raw_input(node, "Normal") != nullptr) {
                static_cast<void>(context.bind(
                    id,
                    "Normal",
                    node,
                    "Normal",
                    SocketType::normal));
            } else {
                static_cast<void>(context.graph().connect(
                    context.geometry_output(
                        "Normal", SocketType::normal)
                        .ref,
                    id,
                    "Normal"));
            }
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "EMISSION" || type == "BACKGROUND") {
            const auto id = context.graph().add_node(
                compiler::node_type::emission,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.bind(
                id,
                "Strength",
                node,
                "Strength",
                SocketType::floating));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "BSDF_TRANSPARENT") {
            const auto id = context.graph().add_node(
                compiler::node_type::transparent_bsdf,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "VOLUME_ABSORPTION") {
            const auto id = context.graph().add_node(
                compiler::node_type::volume_absorption,
                node_name);
            static_cast<void>(context.bind(
                id,
                "Color",
                node,
                "Color",
                SocketType::color));
            static_cast<void>(context.bind(
                id,
                "Density",
                node,
                "Density",
                SocketType::floating));
            return finish({
                .ref = {.node = id, .socket = "Volume"},
                .type = SocketType::volume_closure});
        }
        if (type == "VOLUME_SCATTER") {
            const auto id = context.graph().add_node(
                compiler::node_type::volume_scatter,
                node_name);
            static_cast<void>(context.graph().set_property(
                id,
                "Phase",
                SocketValue::string(context.node_property_text(
                    node,
                    "phase",
                    "HENYEY_GREENSTEIN"))));
            static_cast<void>(context.bind(
                id,
                "Color",
                node,
                "Color",
                SocketType::color));
            static_cast<void>(context.bind(
                id,
                "Density",
                node,
                "Density",
                SocketType::floating));
            for (const auto &[target, source] : {
                     std::pair{"Anisotropy", "Anisotropy"},
                     std::pair{"IOR", "IOR"},
                     std::pair{"Backscatter", "Backscatter"},
                     std::pair{"Alpha", "Alpha"},
                     std::pair{"Diameter", "Diameter"}}) {
                static_cast<void>(context.bind(
                    id,
                    target,
                    node,
                    source,
                    SocketType::floating));
            }
            return finish({
                .ref = {.node = id, .socket = "Volume"},
                .type = SocketType::volume_closure});
        }
        if (type == "VOLUME_COEFFICIENTS") {
            const auto id = context.graph().add_node(
                compiler::node_type::volume_coefficients,
                node_name);
            static_cast<void>(context.graph().set_property(
                id,
                "Phase",
                SocketValue::string(context.node_property_text(
                    node,
                    "phase",
                    "HENYEY_GREENSTEIN"))));
            for (const auto &[target, source] : {
                     std::pair{
                         "ScatterCoefficients",
                         "Scatter Coefficients"},
                     std::pair{
                         "AbsorptionCoefficients",
                         "Absorption Coefficients"},
                     std::pair{
                         "EmissionCoefficients",
                         "Emission Coefficients"}}) {
                static_cast<void>(context.bind(
                    id,
                    target,
                    node,
                    source,
                    SocketType::vector));
            }
            for (const auto &[target, source] : {
                     std::pair{"Anisotropy", "Anisotropy"},
                     std::pair{"IOR", "IOR"},
                     std::pair{"Backscatter", "Backscatter"},
                     std::pair{"Alpha", "Alpha"},
                     std::pair{"Diameter", "Diameter"}}) {
                static_cast<void>(context.bind(
                    id,
                    target,
                    node,
                    source,
                    SocketType::floating));
            }
            return finish({
                .ref = {.node = id, .socket = "Volume"},
                .type = SocketType::volume_closure});
        }
        if (type == "PRINCIPLED_VOLUME") {
            const auto id = context.graph().add_node(
                compiler::node_type::principled_volume,
                node_name);
            for (const auto &[target, source] : {
                     std::pair{"Color", "Color"},
                     std::pair{
                         "AbsorptionColor",
                         "Absorption Color"},
                     std::pair{
                         "EmissionColor",
                         "Emission Color"},
                     std::pair{
                         "BlackbodyTint",
                         "Blackbody Tint"}}) {
                static_cast<void>(context.bind(
                    id,
                    target,
                    node,
                    source,
                    SocketType::color));
            }
            for (const auto &[target, source] : {
                     std::pair{"Density", "Density"},
                     std::pair{"Anisotropy", "Anisotropy"},
                     std::pair{
                         "EmissionStrength",
                         "Emission Strength"},
                     std::pair{
                         "BlackbodyIntensity",
                         "Blackbody Intensity"},
                     std::pair{"Temperature", "Temperature"}}) {
                static_cast<void>(context.bind(
                    id,
                    target,
                    node,
                    source,
                    SocketType::floating));
            }
            return finish({
                .ref = {.node = id, .socket = "Volume"},
                .type = SocketType::volume_closure});
        }
        if (type == "MIX_SHADER" ||
            type == "ADD_SHADER") {
            const auto volume =
                requested == SocketType::volume_closure;
            const auto id = context.graph().add_node(
                type == "MIX_SHADER"
                    ? volume
                          ? compiler::node_type::mix_volume
                          : compiler::node_type::mix_closure
                    : volume
                          ? compiler::node_type::add_volume
                          : compiler::node_type::add_closure,
                node_name);
            if (type == "MIX_SHADER") {
                static_cast<void>(context.bind(
                    id,
                    "Factor",
                    node,
                    "Fac",
                    SocketType::floating));
            }
            static_cast<void>(context.bind(
                id,
                "A",
                node,
                "Shader",
                volume
                    ? SocketType::volume_closure
                    : SocketType::closure));
            static_cast<void>(context.bind(
                id,
                "B",
                node,
                "Shader_001",
                volume
                    ? SocketType::volume_closure
                    : SocketType::closure));
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        volume ? "Volume" : "Closure"},
                .type =
                    volume
                        ? SocketType::volume_closure
                        : SocketType::closure});
        }
        return std::nullopt;
    }
};

}// namespace

std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_closure_lowering_component() {
    return std::make_unique<ClosureNodeLoweringComponent>();
}

}// namespace psycles::adapter::detail
