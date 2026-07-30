#include "blender_graph_lowering_component.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace psycles::adapter::detail {
namespace {

class ProceduralNodeLoweringComponent final
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
        if (type == "TEX_NOISE") {
            const auto id = context.graph().add_node(
                compiler::node_type::noise_texture,
                node_name);
            for (const auto &[target, source] : {
                     std::pair{"Vector", "Vector"},
                     std::pair{"W", "W"},
                     std::pair{"Scale", "Scale"},
                     std::pair{"Detail", "Detail"},
                     std::pair{"Roughness", "Roughness"},
                     std::pair{"Lacunarity", "Lacunarity"},
                     std::pair{"Offset", "Offset"},
                     std::pair{"Gain", "Gain"},
                     std::pair{"Distortion", "Distortion"}}) {
                const auto target_type =
                    std::string_view{target} == "Vector"
                        ? SocketType::vector
                        : SocketType::floating;
                if (target_type == SocketType::vector &&
                    !context.input_source(node, source)) {
                    static_cast<void>(context.graph().connect(
                        context.default_generated_coordinates().ref,
                        id,
                        target));
                } else {
                    static_cast<void>(context.bind(
                        id,
                        target,
                        node,
                        source,
                        target_type));
                }
            }
            const auto dimensions =
                context.node_property_text(
                    node, "noise_dimensions", "3D");
            static_cast<void>(context.graph().set_property(
                id,
                "Dimensions",
                SocketValue::unsigned_integer(
                    dimensions == "1D"
                        ? 1u
                        : dimensions == "2D"
                              ? 2u
                              : dimensions == "4D" ? 4u : 3u)));
            static_cast<void>(context.graph().set_property(
                id,
                "Normalize",
                SocketValue::boolean(context.node_property_bool(
                    node, "normalize"))));
            static_cast<void>(context.graph().set_property(
                id,
                "NoiseType",
                SocketValue::string(context.node_property_text(
                    node, "noise_type", "FBM"))));
            static_cast<void>(context.graph().set_property(
                id,
                "NeedsColor",
                SocketValue::boolean(socket == "Color")));
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        socket == "Color" ? "Color" : "Factor"},
                .type =
                    socket == "Color"
                        ? SocketType::color
                        : SocketType::floating});
        }
        if (type == "TEX_WHITE_NOISE") {
            const auto id = context.graph().add_node(
                compiler::node_type::white_noise_texture,
                node_name);
            static_cast<void>(context.bind(
                id,
                "Vector",
                node,
                "Vector",
                SocketType::vector));
            static_cast<void>(context.bind(
                id,
                "W",
                node,
                "W",
                SocketType::floating));
            const auto dimensions =
                context.node_property_text(
                    node, "noise_dimensions", "3D");
            static_cast<void>(context.graph().set_property(
                id,
                "Dimensions",
                SocketValue::unsigned_integer(
                    dimensions == "1D"
                        ? 1u
                        : dimensions == "2D"
                              ? 2u
                              : dimensions == "4D" ? 4u : 3u)));
            static_cast<void>(context.graph().set_property(
                id,
                "NeedsColor",
                SocketValue::boolean(socket == "Color")));
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        socket == "Color" ? "Color" : "Value"},
                .type =
                    socket == "Color"
                        ? SocketType::color
                        : SocketType::floating});
        }
        if (type == "TEX_CHECKER") {
            const auto id = context.graph().add_node(
                compiler::node_type::checker_texture,
                node_name);
            for (const auto &[target, source] : {
                     std::pair{"Vector", "Vector"},
                     std::pair{"Color1", "Color1"},
                     std::pair{"Color2", "Color2"},
                     std::pair{"Scale", "Scale"}}) {
                const auto target_view =
                    std::string_view{target};
                const auto target_type =
                    target_view == "Vector"
                        ? SocketType::vector
                        : target_view == "Color1" ||
                                  target_view == "Color2"
                              ? SocketType::color
                              : SocketType::floating;
                if (target_type == SocketType::vector &&
                    !context.input_source(node, source)) {
                    static_cast<void>(context.graph().connect(
                        context.default_generated_coordinates().ref,
                        id,
                        target));
                } else {
                    static_cast<void>(context.bind(
                        id,
                        target,
                        node,
                        source,
                        target_type));
                }
            }
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        socket == "Fac" ? "Factor" : "Color"},
                .type =
                    socket == "Fac"
                        ? SocketType::floating
                        : SocketType::color});
        }
        if (type == "TEX_BRICK") {
            const auto id = context.graph().add_node(
                compiler::node_type::brick_texture,
                node_name);
            for (const auto &[target, source] : {
                     std::pair{"Vector", "Vector"},
                     std::pair{"Color1", "Color1"},
                     std::pair{"Color2", "Color2"},
                     std::pair{"Mortar", "Mortar"},
                     std::pair{"Scale", "Scale"},
                     std::pair{"MortarSize", "Mortar Size"},
                     std::pair{"MortarSmooth", "Mortar Smooth"},
                     std::pair{"Bias", "Bias"},
                     std::pair{"BrickWidth", "Brick Width"},
                     std::pair{"RowHeight", "Row Height"}}) {
                const auto target_view =
                    std::string_view{target};
                const auto target_type =
                    target_view == "Vector"
                        ? SocketType::vector
                        : target_view == "Color1" ||
                                  target_view == "Color2" ||
                                  target_view == "Mortar"
                              ? SocketType::color
                              : SocketType::floating;
                if (target_type == SocketType::vector &&
                    !context.input_source(node, source)) {
                    static_cast<void>(context.graph().connect(
                        context.default_generated_coordinates().ref,
                        id,
                        target));
                } else {
                    static_cast<void>(context.bind(
                        id,
                        target,
                        node,
                        source,
                        target_type));
                }
            }
            static_cast<void>(context.graph().set_property(
                id,
                "OffsetAmount",
                SocketValue::floating(context.node_property_number(
                    node, "offset", 0.5f))));
            static_cast<void>(context.graph().set_property(
                id,
                "OffsetFrequency",
                SocketValue::unsigned_integer(
                    static_cast<std::uint64_t>(std::max(
                        context.node_property_number(
                            node, "offset_frequency", 2.0f),
                        0.0f)))));
            static_cast<void>(context.graph().set_property(
                id,
                "SquashAmount",
                SocketValue::floating(context.node_property_number(
                    node, "squash", 1.0f))));
            static_cast<void>(context.graph().set_property(
                id,
                "SquashFrequency",
                SocketValue::unsigned_integer(
                    static_cast<std::uint64_t>(std::max(
                        context.node_property_number(
                            node, "squash_frequency", 2.0f),
                        0.0f)))));
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        socket == "Fac" ? "Factor" : "Color"},
                .type =
                    socket == "Fac"
                        ? SocketType::floating
                        : SocketType::color});
        }
        if (type == "TEX_GRADIENT") {
            const auto id = context.graph().add_node(
                compiler::node_type::gradient_texture,
                node_name);
            if (context.input_source(node, "Vector")) {
                static_cast<void>(context.bind(
                    id,
                    "Vector",
                    node,
                    "Vector",
                    SocketType::vector));
            } else {
                static_cast<void>(context.graph().connect(
                    context.default_generated_coordinates().ref,
                    id,
                    "Vector"));
            }
            static_cast<void>(context.graph().set_property(
                id,
                "GradientType",
                SocketValue::string(context.node_property_text(
                    node, "gradient_type", "LINEAR"))));
            return finish({
                .ref = {.node = id, .socket = "Factor"},
                .type = SocketType::floating});
        }
        if (type == "VERTEX_COLOR") {
            const auto id = context.graph().add_node(
                compiler::node_type::vertex_color,
                node_name);
            static_cast<void>(context.graph().set_property(
                id,
                "AttributeId",
                SocketValue::unsigned_integer(
                    contract::attribute_id(
                    context.node_property_text(
                        node, "layer_name")))));
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        socket == "Alpha" ? "Alpha" : "Color"},
                .type =
                    socket == "Alpha"
                        ? SocketType::floating
                        : SocketType::color});
        }
        if (type == "SEPARATE_COLOR") {
            const auto id = context.graph().add_node(
                compiler::node_type::separate_color,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.graph().set_property(
                id,
                "Mode",
                SocketValue::string(context.node_property_text(
                    node, "mode", "RGB"))));
            const auto output =
                socket == "Red"
                    ? "R"
                    : socket == "Green" ? "G" : "B";
            return finish({
                .ref = {.node = id, .socket = output},
                .type = SocketType::floating});
        }
        if (type == "SEPRGB" ||
            type == "SEPHSV" ||
            type == "SEPXYZ") {
            const auto id = context.graph().add_node(
                compiler::node_type::separate_color,
                node_name);
            const auto input_name =
                type == "SEPRGB"
                    ? "Image"
                    : type == "SEPHSV"
                          ? "Color"
                          : "Vector";
            static_cast<void>(context.bind(
                id,
                "Color",
                node,
                input_name,
                SocketType::color));
            static_cast<void>(context.graph().set_property(
                id,
                "Mode",
                SocketValue::string(
                    type == "SEPHSV" ? "HSV" : "RGB")));
            const auto output =
                socket == "R" || socket == "H" ||
                        socket == "X"
                    ? "R"
                    : socket == "G" || socket == "S" ||
                              socket == "Y"
                          ? "G"
                          : "B";
            return finish({
                .ref = {.node = id, .socket = output},
                .type = SocketType::floating});
        }
        if (type == "COMBINE_COLOR") {
            const auto mode =
                context.node_property_text(node, "mode", "RGB");
            const auto id = context.graph().add_node(
                compiler::node_type::combine_color,
                node_name);
            static_cast<void>(context.bind(
                id, "R", node, "Red", SocketType::floating));
            static_cast<void>(context.bind(
                id, "G", node, "Green", SocketType::floating));
            static_cast<void>(context.bind(
                id, "B", node, "Blue", SocketType::floating));
            static_cast<void>(context.graph().set_property(
                id,
                "Mode",
                SocketValue::string(mode)));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "COMBRGB" ||
            type == "COMBHSV" ||
            type == "COMBXYZ") {
            const auto id = context.graph().add_node(
                compiler::node_type::combine_color,
                node_name);
            const auto first =
                type == "COMBRGB"
                    ? "R"
                    : type == "COMBHSV" ? "H" : "X";
            const auto second =
                type == "COMBRGB"
                    ? "G"
                    : type == "COMBHSV" ? "S" : "Y";
            const auto third =
                type == "COMBRGB"
                    ? "B"
                    : type == "COMBHSV" ? "V" : "Z";
            static_cast<void>(context.bind(
                id,
                "R",
                node,
                first,
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "G",
                node,
                second,
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "B",
                node,
                third,
                SocketType::floating));
            static_cast<void>(context.graph().set_property(
                id,
                "Mode",
                SocketValue::string(
                    type == "COMBHSV" ? "HSV" : "RGB")));
            TypedOutput result{
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color};
            return finish(
                type == "COMBXYZ"
                    ? context.conversion(
                          result, SocketType::vector)
                    : result);
        }
        if (type == "TEX_SKY") {
            const auto id = context.graph().add_node(
                compiler::node_type::nishita_sky,
                node_name);
            if (context.input_source(node, "Vector")) {
                static_cast<void>(context.bind(
                    id,
                    "Vector",
                    node,
                    "Vector",
                    SocketType::vector));
            } else {
                static_cast<void>(context.graph().connect(
                    context.default_generated_coordinates().ref,
                    id,
                    "Vector"));
            }
            constexpr auto pi = 3.14159265358979323846f;
            constexpr auto two_pi = 2.0f * pi;
            auto elevation = std::fmod(
                context.node_property_number(
                    node, "sun_elevation", 0.7853982f),
                two_pi);
            auto rotation = context.node_property_number(
                node, "sun_rotation", 0.0f);
            if (std::abs(elevation) >= pi) {
                elevation -= std::copysign(two_pi, elevation);
            }
            if (elevation >= pi * 0.5f ||
                elevation <= -pi * 0.5f) {
                elevation =
                    std::copysign(pi, elevation) - elevation;
                rotation += pi;
            }
            rotation = std::fmod(rotation, two_pi);
            if (rotation < 0.0f) {
                rotation += two_pi;
            }
            rotation = two_pi - rotation;
            static_cast<void>(context.graph().set_input(
                id,
                "SunElevation",
                SocketValue::floating(elevation)));
            static_cast<void>(context.graph().set_input(
                id,
                "SunRotation",
                SocketValue::floating(rotation)));
            static_cast<void>(context.graph().set_input(
                id,
                "SunSize",
                SocketValue::floating(
                    context.node_property_bool(
                        node, "sun_disc", true)
                        ? context.node_property_number(
                              node,
                              "sun_size",
                              0.00918043f)
                        : -1.0f)));
            for (const auto &[target, property_name, fallback] : {
                     std::tuple{
                         "SunIntensity",
                         "sun_intensity",
                         1.0f},
                     std::tuple{
                         "Altitude", "altitude", 0.0f},
                     std::tuple{
                         "AirDensity", "air_density", 1.0f},
                     std::tuple{
                         "DustDensity",
                         "aerosol_density",
                         context.node_property_number(
                             node,
                             "dust_density",
                             1.0f)},
                     std::tuple{
                         "OzoneDensity",
                         "ozone_density",
                         1.0f}}) {
                static_cast<void>(context.graph().set_input(
                    id,
                    target,
                    SocketValue::floating(
                        context.node_property_number(
                            node,
                            property_name,
                            fallback))));
            }
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        return std::nullopt;
    }
};

}// namespace

std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_procedural_lowering_component() {
    return std::make_unique<ProceduralNodeLoweringComponent>();
}

}// namespace psycles::adapter::detail
