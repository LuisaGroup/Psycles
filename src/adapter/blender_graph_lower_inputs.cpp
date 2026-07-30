#include "blender_graph_lowering_component.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace psycles::adapter::detail {
namespace {

class InputNodeLoweringComponent final
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
        if (type == "RGB") {
            return finish(context.constant_from_output(
                node, socket, SocketType::color));
        }
        if (type == "VALUE") {
            return finish(context.constant_from_output(
                node, socket, SocketType::floating));
        }
        if (type == "TEX_COORD" || type == "UVMAP") {
            const auto id = context.graph().add_node(
                compiler::node_type::texture_coordinate,
                node_name);
            if (type == "UVMAP") {
                const auto uv_map =
                    context.node_property_text(node, "uv_map");
                static_cast<void>(context.graph().set_property(
                    id,
                    "UvMapNamed",
                    SocketValue::boolean(!uv_map.empty())));
                static_cast<void>(context.graph().set_property(
                    id,
                    "UvMapId",
                    SocketValue::unsigned_integer(
                        uv_map.empty()
                            ? 0u
                            : contract::uv_attribute_id(
                                  uv_map))));
            }
            const auto output_socket =
                type == "UVMAP" ? std::string{"UV"} : socket;
            const auto output_type =
                output_socket == "Normal"
                    ? SocketType::vector
                    : SocketType::vector;
            return finish({
                .ref = {
                    .node = id,
                    .socket = output_socket},
                .type = output_type});
        }
        if (type == "OBJECT_INFO") {
            const auto id = context.graph().add_node(
                compiler::node_type::object_info,
                node_name);
            const auto output_socket =
                socket == "Location" ? "Location" : "Random";
            return finish({
                .ref = {.node = id, .socket = output_socket},
                .type =
                    output_socket == std::string_view{"Location"}
                        ? SocketType::vector
                        : SocketType::floating});
        }
        if (type == "PARTICLE_INFO") {
            const auto id = context.graph().add_node(
                compiler::node_type::particle_info,
                node_name);
            const auto output_socket =
                socket == "Index" ? "Index" : "Random";
            return finish({
                .ref = {.node = id, .socket = output_socket},
                .type = SocketType::floating});
        }
        if (type == "LIGHT_PATH") {
            const auto id = context.graph().add_node(
                compiler::node_type::light_path,
                node_name);
            const auto output_socket =
                socket == "Is Camera Ray"
                    ? "IsCameraRay"
                    : socket == "Is Shadow Ray"
                          ? "IsShadowRay"
                          : socket == "Is Diffuse Ray"
                                ? "IsDiffuseRay"
                                : socket == "Is Glossy Ray"
                                      ? "IsGlossyRay"
                                      : socket == "Is Singular Ray"
                                            ? "IsSingularRay"
                                            : socket ==
                                                      "Is Reflection Ray"
                                                  ? "IsReflectionRay"
                                                  : socket ==
                                                            "Is Transmission Ray"
                                                        ? "IsTransmissionRay"
                                                        : socket ==
                                                                  "Is Volume Scatter Ray"
                                                              ? "IsVolumeScatterRay"
                                                              : socket ==
                                                                        "Ray Length"
                                                                    ? "RayLength"
                                                                    : socket ==
                                                                              "Ray Depth"
                                                                          ? "RayDepth"
                                                                          : socket ==
                                                                                    "Diffuse Depth"
                                                                                ? "DiffuseDepth"
                                                                                : socket ==
                                                                                          "Glossy Depth"
                                                                                      ? "GlossyDepth"
                                                                                      : socket ==
                                                                                                "Transparent Depth"
                                                                                            ? "TransparentDepth"
                                                                                            : "TransmissionDepth";
            return finish({
                .ref = {.node = id, .socket = output_socket},
                .type = SocketType::floating});
        }
        if (type == "LAYER_WEIGHT") {
            const auto id = context.graph().add_node(
                compiler::node_type::layer_weight,
                node_name);
            static_cast<void>(context.graph().set_property(
                id,
                "NormalLinked",
                SocketValue::boolean(
                    context.input_source(node, "Normal").has_value())));
            static_cast<void>(context.bind(
                id,
                "Blend",
                node,
                "Blend",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "Normal",
                node,
                "Normal",
                SocketType::normal));
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        socket == "Facing"
                            ? "Facing"
                            : "Fresnel"},
                .type = SocketType::floating});
        }
        if (type == "FRESNEL") {
            const auto id = context.graph().add_node(
                compiler::node_type::fresnel,
                node_name);
            static_cast<void>(context.bind(
                id,
                "IOR",
                node,
                "IOR",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "Normal",
                node,
                "Normal",
                SocketType::normal));
            return finish({
                .ref = {.node = id, .socket = "Factor"},
                .type = SocketType::floating});
        }
        if (type == "NEW_GEOMETRY") {
            if (socket == "Normal" ||
                socket == "True Normal") {
                return finish(context.geometry_output(
                    socket == "Normal"
                        ? "Normal"
                        : "GeometricNormal",
                    SocketType::normal));
            }
            if (socket == "Position") {
                return finish(context.geometry_output(
                    "Position", SocketType::point));
            }
            if (socket == "Incoming") {
                return finish(context.geometry_output(
                    "Incoming", SocketType::vector));
            }
            if (socket == "Backfacing") {
                return finish(context.geometry_output(
                    "Backfacing", SocketType::floating));
            }
            if (socket == "Random Per Island") {
                return finish(context.geometry_output(
                    "RandomPerIsland", SocketType::floating));
            }
            context.warn_once(
                "geometry:" + socket,
                "Geometry output '" + socket +
                    "' is not yet represented; using zero");
            return finish(context.constant_from_output(
                node, socket, SocketType::floating));
        }
        if (type == "MAPPING") {
            const auto id = context.graph().add_node(
                compiler::node_type::mapping, node_name);
            static_cast<void>(context.bind(
                id, "Vector", node, "Vector", SocketType::vector));
            static_cast<void>(context.bind(
                id, "Location", node, "Location", SocketType::vector));
            static_cast<void>(context.bind(
                id, "Rotation", node, "Rotation", SocketType::vector));
            static_cast<void>(context.bind(
                id, "Scale", node, "Scale", SocketType::vector));
            static_cast<void>(context.graph().set_property(
                id,
                "VectorType",
                SocketValue::string(context.node_property_text(
                    node, "vector_type", "POINT"))));
            return finish({
                .ref = {.node = id, .socket = "Vector"},
                .type = SocketType::vector});
        }
        if (type == "TEX_IMAGE") {
            const auto id = context.graph().add_node(
                compiler::node_type::image_texture,
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
                    context.default_image_coordinates().ref,
                    id,
                    "Vector"));
            }
            const auto image_name =
                text(member(node, "image"));
            const auto image_iter =
                context.image_ids().find(image_name);
            if (image_iter == context.image_ids().end()) {
                context.warn_once(
                    "image:" + image_name,
                    "image '" + image_name +
                        "' is unavailable");
            }
            const auto image_id =
                image_iter == context.image_ids().end()
                    ? 0u
                    : image_iter->second.value;
            const auto color_iter =
                context.image_color_spaces().find(image_name);
            const auto image_color_space =
                color_iter == context.image_color_spaces().end()
                    ? ImageColorSpace::data
                    : color_iter->second;
            const auto color_space =
                image_color_space == ImageColorSpace::srgb
                    ? "sRGB"
                    : "Non-Color";
            const auto alpha_iter =
                context.image_alpha_types().find(image_name);
            const auto alpha_type =
                alpha_iter == context.image_alpha_types().end()
                    ? ImageAlphaType::straight
                    : alpha_iter->second;
            const auto unassociate_alpha =
                context.output_is_linked(node, "Alpha") &&
                image_color_space != ImageColorSpace::data &&
                alpha_type != ImageAlphaType::channel_packed &&
                alpha_type != ImageAlphaType::ignore;
            static_cast<void>(context.graph().set_property(
                id,
                "Image",
                SocketValue::unsigned_integer(image_id)));
            static_cast<void>(context.graph().set_property(
                id,
                "Extension",
                SocketValue::string(context.node_property_text(
                    node, "extension", "REPEAT"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Interpolation",
                SocketValue::string(context.node_property_text(
                    node, "interpolation", "Linear"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Projection",
                SocketValue::string(context.node_property_text(
                    node, "projection", "FLAT"))));
            static_cast<void>(context.graph().set_property(
                id,
                "ProjectionBlend",
                SocketValue::floating(context.node_property_number(
                    node, "projection_blend", 0.0f))));
            static_cast<void>(context.graph().set_property(
                id,
                "ColorSpace",
                SocketValue::string(color_space)));
            static_cast<void>(context.graph().set_property(
                id,
                "UnassociateAlpha",
                SocketValue::boolean(unassociate_alpha)));
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
        return std::nullopt;
    }
};

}// namespace

std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_input_lowering_component() {
    return std::make_unique<InputNodeLoweringComponent>();
}

}// namespace psycles::adapter::detail
