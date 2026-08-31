#include "blender_graph_lowering_component.h"

#include <cmath>
#include <string>
#include <string_view>
#include <tuple>

namespace psycles::adapter::detail {
namespace {

[[nodiscard]] float cycles_texture_scale(float value) noexcept {
    return std::abs(value) < 1.0e-5f
               ? (value < 0.0f ? -1.0e-5f : 1.0e-5f)
               : value;
}

void connect_source(
    BlenderNodeLoweringContext &context,
    contract::NodeId destination,
    yyjson_val *raw_node,
    TypedOutput implicit_coordinates) {
    if (context.input_source(raw_node, "Vector")) {
        static_cast<void>(context.bind(
            destination,
            "Vector",
            raw_node,
            "Vector",
            contract::SocketType::vector));
    } else {
        static_cast<void>(context.graph().connect(
            implicit_coordinates.ref,
            destination,
            "Vector"));
    }
}

}// namespace

void bind_blender_texture_vector(
    BlenderNodeLoweringContext &context,
    contract::NodeId destination,
    yyjson_val *raw_node,
    TypedOutput implicit_coordinates) {
    auto *mapping = member(
        member(raw_node, "special"), "texture_mapping");
    if (mapping == nullptr) {
        connect_source(
            context, destination, raw_node, implicit_coordinates);
        return;
    }

    const auto mapping_node = context.graph().add_node(
        compiler::node_type::mapping,
        text(member(raw_node, "name")) + " / Texture Mapping");
    static_cast<void>(context.graph().set_property(
        mapping_node,
        "LegacyTextureMapping",
        contract::SocketValue::boolean(true)));
    connect_source(
        context, mapping_node, raw_node, implicit_coordinates);

    const auto vector_type = text(
        member(mapping, "vector_type"), "POINT");
    auto scale = float3(
        member(mapping, "scale"), {1.0f, 1.0f, 1.0f});
    if (vector_type == "TEXTURE" || vector_type == "NORMAL") {
        // TextureMapping::compute_transform keeps the static transform
        // invertible. This differs intentionally from a user-authored
        // ShaderNodeMapping, whose zero-scale behavior is evaluated by SVM.
        scale.x = cycles_texture_scale(scale.x);
        scale.y = cycles_texture_scale(scale.y);
        scale.z = cycles_texture_scale(scale.z);
    }
    static_cast<void>(context.graph().set_input(
        mapping_node,
        "Location",
        contract::SocketValue::vector(float3(
            member(mapping, "translation")))));
    static_cast<void>(context.graph().set_input(
        mapping_node,
        "Rotation",
        contract::SocketValue::vector(float3(
            member(mapping, "rotation")))));
    static_cast<void>(context.graph().set_input(
        mapping_node,
        "Scale",
        contract::SocketValue::vector(scale)));
    for (const auto &[property, source, fallback] : {
             std::tuple{"VectorType", "vector_type", "POINT"},
             std::tuple{"XMapping", "mapping_x", "X"},
             std::tuple{"YMapping", "mapping_y", "Y"},
             std::tuple{"ZMapping", "mapping_z", "Z"}}) {
        static_cast<void>(context.graph().set_property(
            mapping_node,
            property,
            contract::SocketValue::string(
                property == std::string_view{"VectorType"}
                    ? vector_type
                    : text(member(mapping, source), fallback))));
    }
    static_cast<void>(context.graph().connect(
        {.node = mapping_node, .socket = "Vector"},
        destination,
        "Vector"));
}

}// namespace psycles::adapter::detail
