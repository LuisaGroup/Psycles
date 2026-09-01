#include "blender_graph_lowering_component.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <utility>

namespace psycles::adapter::detail {
namespace {

struct BlenderImageBinding {
  std::uint64_t id{};
  std::string color_space{"Non-Color"};
  bool unassociate_alpha{};
};

[[nodiscard]] BlenderImageBinding
resolve_image_binding(BlenderNodeLoweringContext &context, yyjson_val *node,
                      bool alpha_output_controls_unassociation) {
  const auto image_name = text(member(node, "image"));
  const auto image_iter = context.image_ids().find(image_name);
  if (image_iter == context.image_ids().end()) {
    context.warn_once("image:" + image_name,
                      "image '" + image_name + "' is unavailable");
  }
  const auto color_iter = context.image_color_spaces().find(image_name);
  const auto image_color_space =
      color_iter == context.image_color_spaces().end() ? ImageColorSpace::data
                                                       : color_iter->second;
  const auto alpha_iter = context.image_alpha_types().find(image_name);
  const auto alpha_type = alpha_iter == context.image_alpha_types().end()
                              ? ImageAlphaType::straight
                              : alpha_iter->second;
  return {.id = image_iter == context.image_ids().end()
                    ? 0u
                    : image_iter->second.value,
          .color_space =
              image_color_space == ImageColorSpace::srgb ? "sRGB" : "Non-Color",
          .unassociate_alpha = alpha_output_controls_unassociation &&
                               context.output_is_linked(node, "Alpha") &&
                               image_color_space != ImageColorSpace::data &&
                               alpha_type != ImageAlphaType::channel_packed &&
                               alpha_type != ImageAlphaType::ignore};
}

void set_image_resource_properties(BlenderNodeLoweringContext &context,
                                   contract::NodeId node,
                                   const BlenderImageBinding &binding) {
  static_cast<void>(context.graph().set_property(
      node, "Image", SocketValue::unsigned_integer(binding.id)));
  static_cast<void>(context.graph().set_property(
      node, "ColorSpace", SocketValue::string(binding.color_space)));
}

class InputNodeLoweringComponent final : public BlenderNodeLoweringComponent {

public:
  [[nodiscard]] std::optional<TypedOutput>
  lower(BlenderNodeLoweringContext &context,
        const BlenderNodeLoweringRequest &request) const override {
    using contract::SocketType;
    [[maybe_unused]] const auto &node_name = request.node_name;
    [[maybe_unused]] const auto &socket = request.socket;
    [[maybe_unused]] const auto requested = request.requested;
    [[maybe_unused]] auto *node = request.node;
    [[maybe_unused]] const auto &type = request.type;
    [[maybe_unused]] const auto &finish = request.finish;
    if (type == "RGB") {
      return finish(
          context.constant_from_output(node, socket, SocketType::color));
    }
    if (type == "VALUE") {
      return finish(
          context.constant_from_output(node, socket, SocketType::floating));
    }
    if (type == "CAMERA") {
      static constexpr auto outputs = std::array{
          std::pair{std::string_view{"View Vector"}, SocketType::vector},
          std::pair{std::string_view{"View Z Depth"}, SocketType::floating},
          std::pair{std::string_view{"View Distance"}, SocketType::floating}};
      const auto selected = std::find_if(
          outputs.begin(), outputs.end(),
          [&](const auto &entry) noexcept { return entry.first == socket; });
      if (selected == outputs.end()) {
        return std::nullopt;
      }
      const auto semantic = "camera_data." + socket;
      auto output = context.shared_output(node_name, semantic);
      if (!output) {
        const auto id = context.graph().add_node(
            compiler::node_type::camera_data, node_name);
        for (const auto &[name, output_type] : outputs) {
          context.remember_shared_output(
              node_name, "camera_data." + std::string{name},
              TypedOutput{.ref = {.node = id, .socket = std::string{name}},
                          .type = output_type});
        }
        output = context.shared_output(node_name, semantic);
      }
      return finish(*output);
    }
    if (type == "TEX_COORD" || type == "UVMAP") {
      const auto id = context.graph().add_node(
          type == "TEX_COORD" ? compiler::node_type::texture_coordinate
                              : compiler::node_type::uv_map,
          node_name);
      const auto from_dupli =
          context.node_property_bool(node, "from_instancer", false);
      static_cast<void>(context.graph().set_property(
          id, "FromDupli", SocketValue::boolean(from_dupli)));
      if (type == "TEX_COORD") {
        auto *object_coordinates =
            member(member(node, "special"), "object_coordinates");
        auto *object_to_world = member(object_coordinates, "object_to_world");
        if (object_to_world != nullptr) {
          static_cast<void>(context.graph().set_property(
              id, "UseTransform", SocketValue::boolean(true)));
          static_cast<void>(context.graph().set_property(
              id, "ObjectTransform",
              SocketValue::transform(matrix(object_to_world))));
        }
      } else {
        const auto uv_map = context.node_property_text(node, "uv_map");
        static_cast<void>(context.graph().set_property(
            id, "Attribute", SocketValue::string(uv_map)));
        static_cast<void>(context.graph().set_property(
            id, "AttributeId",
            SocketValue::unsigned_integer(
                uv_map.empty() ? 0u : contract::uv_attribute_id(uv_map))));
      }
      const auto output_socket = type == "UVMAP" ? std::string{"UV"} : socket;
      const auto output_type =
          output_socket == "Normal" || output_socket == "Reflection"
              ? SocketType::normal
              : SocketType::point;
      return finish(
          {.ref = {.node = id, .socket = output_socket}, .type = output_type});
    }
    if (type == "OBJECT_INFO") {
      const auto id =
          context.graph().add_node(compiler::node_type::object_info, node_name);
      const auto output_socket = socket == "Location" ? "Location" : "Random";
      return finish({.ref = {.node = id, .socket = output_socket},
                     .type = output_socket == std::string_view{"Location"}
                                 ? SocketType::vector
                                 : SocketType::floating});
    }
    if (type == "PARTICLE_INFO") {
      const auto id = context.graph().add_node(
          compiler::node_type::particle_info, node_name);
      const auto output_socket = socket == "Index" ? "Index" : "Random";
      return finish({.ref = {.node = id, .socket = output_socket},
                     .type = SocketType::floating});
    }
    if (type == "HAIR_INFO") {
      const auto id =
          context.graph().add_node(compiler::node_type::hair_info, node_name);
      const auto output_socket = socket == "Is Strand"        ? "IsStrand"
                                 : socket == "Tangent Normal" ? "TangentNormal"
                                                              : socket;
      return finish({.ref = {.node = id, .socket = output_socket},
                     .type = socket == "Tangent Normal"
                                 ? SocketType::normal
                                 : SocketType::floating});
    }
    if (type == "LIGHT_FALLOFF") {
      // Cycles gives Light Falloff a semantic distant-light branch: an exact
      // FLT_MAX ShaderData::ray_length returns Strength without evaluating
      // any distance algebra. Preserve that operation in the graph;
      // decomposing it into multiply/divide nodes turns the distant branch
      // into inf/inf under IEEE-754.
      const auto light_path = context.graph().add_node(
          compiler::node_type::light_path, node_name + " / Light Path");
      const auto falloff = context.graph().add_node(
          compiler::node_type::light_falloff, node_name);
      static_cast<void>(context.bind(falloff, "Strength", node, "Strength",
                                     SocketType::floating));
      static_cast<void>(context.bind(falloff, "Smooth", node, "Smooth",
                                     SocketType::floating));
      static_cast<void>(context.graph().connect(
          {.node = light_path, .socket = "RayLength"}, falloff, "RayLength"));
      const auto output = socket == "Linear"   ? std::string{"Linear"}
                          : socket == "Constant" ? std::string{"Constant"}
                                                 : std::string{"Quadratic"};
      return finish({.ref = {.node = falloff, .socket = output},
                     .type = SocketType::floating});
    }
    if (type == "LIGHT_PATH") {
      static constexpr auto outputs = std::array{
          std::pair{std::string_view{"Is Camera Ray"},
                    std::string_view{"IsCameraRay"}},
          std::pair{std::string_view{"Is Shadow Ray"},
                    std::string_view{"IsShadowRay"}},
          std::pair{std::string_view{"Is Diffuse Ray"},
                    std::string_view{"IsDiffuseRay"}},
          std::pair{std::string_view{"Is Glossy Ray"},
                    std::string_view{"IsGlossyRay"}},
          std::pair{std::string_view{"Is Singular Ray"},
                    std::string_view{"IsSingularRay"}},
          std::pair{std::string_view{"Is Reflection Ray"},
                    std::string_view{"IsReflectionRay"}},
          std::pair{std::string_view{"Is Transmission Ray"},
                    std::string_view{"IsTransmissionRay"}},
          std::pair{std::string_view{"Is Volume Scatter Ray"},
                    std::string_view{"IsVolumeScatterRay"}},
          std::pair{std::string_view{"Ray Length"},
                    std::string_view{"RayLength"}},
          std::pair{std::string_view{"Ray Depth"},
                    std::string_view{"RayDepth"}},
          std::pair{std::string_view{"Diffuse Depth"},
                    std::string_view{"DiffuseDepth"}},
          std::pair{std::string_view{"Glossy Depth"},
                    std::string_view{"GlossyDepth"}},
          std::pair{std::string_view{"Transparent Depth"},
                    std::string_view{"TransparentDepth"}},
          std::pair{std::string_view{"Transmission Depth"},
                    std::string_view{"TransmissionDepth"}},
          std::pair{std::string_view{"Portal Depth"},
                    std::string_view{"PortalDepth"}}};
      const auto mapped = std::find_if(
          outputs.begin(), outputs.end(),
          [&](const auto &entry) noexcept { return entry.first == socket; });
      if (mapped == outputs.end()) {
        return std::nullopt;
      }
      const auto semantic = "light_path." + std::string{mapped->second};
      auto output = context.shared_output(node_name, semantic);
      if (!output) {
        const auto id = context.graph().add_node(
            compiler::node_type::light_path, node_name);
        for (const auto &[raw_name, projected_name] : outputs) {
          static_cast<void>(raw_name);
          context.remember_shared_output(
              node_name, "light_path." + std::string{projected_name},
              TypedOutput{
                  .ref = {.node = id,
                          .socket = std::string{projected_name}},
                  .type = SocketType::floating});
        }
        output = context.shared_output(node_name, semantic);
      }
      return finish(*output);
    }
    if (type == "LAYER_WEIGHT") {
      const auto output_name = socket == "Facing" ? std::string_view{"Facing"}
                                                   : std::string_view{"Fresnel"};
      const auto semantic = "layer_weight." + std::string{output_name};
      auto output = context.shared_output(node_name, semantic);
      if (!output) {
        const auto id = context.graph().add_node(
            compiler::node_type::layer_weight, node_name);
        static_cast<void>(context.graph().set_property(
            id, "NormalLinked",
            SocketValue::boolean(
                context.input_source(node, "Normal").has_value())));
        static_cast<void>(context.bind(id, "Blend", node, "Blend",
                                       SocketType::floating));
        static_cast<void>(context.bind(id, "Normal", node, "Normal",
                                       SocketType::normal));
        for (const auto name : {std::string_view{"Fresnel"},
                                std::string_view{"Facing"}}) {
          context.remember_shared_output(
              node_name, "layer_weight." + std::string{name},
              TypedOutput{.ref = {.node = id, .socket = std::string{name}},
                          .type = SocketType::floating});
        }
        output = context.shared_output(node_name, semantic);
      }
      return finish(*output);
    }
    if (type == "FRESNEL") {
      const auto id =
          context.graph().add_node(compiler::node_type::fresnel, node_name);
      static_cast<void>(context.graph().set_property(
          id, "NormalLinked",
          SocketValue::boolean(
              context.input_source(node, "Normal").has_value())));
      static_cast<void>(
          context.bind(id, "IOR", node, "IOR", SocketType::floating));
      static_cast<void>(
          context.bind(id, "Normal", node, "Normal", SocketType::normal));
      return finish({.ref = {.node = id, .socket = "Factor"},
                     .type = SocketType::floating});
    }
    if (type == "AMBIENT_OCCLUSION") {
      constexpr auto ao_semantic = "ambient_occlusion.factor";
      auto ao = context.shared_output(node_name, ao_semantic);
      if (!ao) {
        const auto id = context.graph().add_node(
            compiler::node_type::ambient_occlusion, node_name);
        const auto normal_linked =
            context.input_source(node, "Normal").has_value();
        const auto distance_linked =
            context.input_source(node, "Distance").has_value();
        auto *distance_socket = context.raw_input(node, "Distance");
        const auto global_radius =
            !distance_linked &&
            number(member(distance_socket, "default")) == 0.0f;
        static_cast<void>(context.bind(
            id, "Distance", node, "Distance", SocketType::floating));
        static_cast<void>(context.bind(
            id, "Normal", node, "Normal", SocketType::normal));
        static_cast<void>(context.graph().set_property(
            id, "Samples",
            SocketValue::unsigned_integer(static_cast<std::uint8_t>(
                static_cast<std::int64_t>(context.node_property_number(
                    node, "samples", 16.0f))))));
        static_cast<void>(context.graph().set_property(
            id, "NormalLinked", SocketValue::boolean(normal_linked)));
        static_cast<void>(context.graph().set_property(
            id, "Inside",
            SocketValue::boolean(
                context.node_property_bool(node, "inside"))));
        static_cast<void>(context.graph().set_property(
            id, "OnlyLocal",
            SocketValue::boolean(
                context.node_property_bool(node, "only_local"))));
        static_cast<void>(context.graph().set_property(
            id, "GlobalRadius", SocketValue::boolean(global_radius)));
        ao = TypedOutput{
            .ref = {.node = id, .socket = "AO"},
            .type = SocketType::floating};
        context.remember_shared_output(
            node_name, ao_semantic, *ao);
      }

      if (socket == "AO") {
        return finish(*ao);
      }
      if (socket == "Color") {
        constexpr auto color_semantic = "ambient_occlusion.color";
        if (auto color = context.shared_output(
                node_name, color_semantic)) {
          return finish(*color);
        }
        const auto multiply = context.graph().add_node(
            compiler::node_type::multiply_color,
            node_name + " / Color");
        static_cast<void>(context.bind(
            multiply, "A", node, "Color", SocketType::color));
        const auto ao_color = context.conversion(
            *ao, SocketType::color);
        static_cast<void>(context.graph().connect(
            ao_color.ref, multiply, "B"));
        static_cast<void>(context.graph().set_input(
            multiply, "Factor", SocketValue::floating(1.0f)));
        auto color = TypedOutput{
            .ref = {.node = multiply, .socket = "Color"},
            .type = SocketType::color};
        context.remember_shared_output(
            node_name, color_semantic, color);
        return finish(color);
      }
      return std::nullopt;
    }
    if (type == "NEW_GEOMETRY") {
      if (socket == "Normal" || socket == "True Normal") {
        return finish(context.geometry_output(
            socket == "Normal" ? "Normal" : "GeometricNormal",
            SocketType::normal));
      }
      if (socket == "Position") {
        return finish(context.geometry_output("Position", SocketType::point));
      }
      if (socket == "Incoming") {
        return finish(context.geometry_output("Incoming", SocketType::vector));
      }
      if (socket == "Tangent") {
        return finish(context.geometry_output("Tangent", SocketType::vector));
      }
      if (socket == "Parametric") {
        return finish(context.geometry_output("Parametric", SocketType::point));
      }
      if (socket == "Backfacing") {
        return finish(
            context.geometry_output("Backfacing", SocketType::floating));
      }
      if (socket == "Pointiness") {
        return finish(
            context.geometry_output("Pointiness", SocketType::floating));
      }
      if (socket == "Random Per Island") {
        return finish(
            context.geometry_output("RandomPerIsland", SocketType::floating));
      }
      context.warn_once("geometry:" + socket,
                        "Geometry output '" + socket +
                            "' is not yet represented; using zero");
      return finish(
          context.constant_from_output(node, socket, SocketType::floating));
    }
    if (type == "MAPPING") {
      const auto id =
          context.graph().add_node(compiler::node_type::mapping, node_name);
      static_cast<void>(
          context.bind(id, "Vector", node, "Vector", SocketType::vector));
      static_cast<void>(
          context.bind(id, "Location", node, "Location", SocketType::vector));
      static_cast<void>(
          context.bind(id, "Rotation", node, "Rotation", SocketType::vector));
      static_cast<void>(
          context.bind(id, "Scale", node, "Scale", SocketType::vector));
      static_cast<void>(context.graph().set_property(
          id, "VectorType",
          SocketValue::string(
              context.node_property_text(node, "vector_type", "POINT"))));
      return finish({.ref = {.node = id, .socket = "Vector"},
                     .type = SocketType::vector});
    }
    if (type == "TEX_IMAGE") {
      const auto id = context.graph().add_node(
          compiler::node_type::image_texture, node_name);
      bind_blender_texture_vector(context, id, node,
                                  context.default_image_coordinates());
      const auto image = resolve_image_binding(context, node, true);
      set_image_resource_properties(context, id, image);
      static_cast<void>(context.graph().set_property(
          id, "Extension",
          SocketValue::string(
              context.node_property_text(node, "extension", "REPEAT"))));
      static_cast<void>(context.graph().set_property(
          id, "Interpolation",
          SocketValue::string(
              context.node_property_text(node, "interpolation", "Linear"))));
      static_cast<void>(context.graph().set_property(
          id, "Projection",
          SocketValue::string(
              context.node_property_text(node, "projection", "FLAT"))));
      static_cast<void>(context.graph().set_property(
          id, "ProjectionBlend",
          SocketValue::floating(
              context.node_property_number(node, "projection_blend", 0.0f))));
      static_cast<void>(context.graph().set_property(
          id, "UnassociateAlpha",
          SocketValue::boolean(image.unassociate_alpha)));
      return finish(
          {.ref = {.node = id, .socket = socket == "Alpha" ? "Alpha" : "Color"},
           .type =
               socket == "Alpha" ? SocketType::floating : SocketType::color});
    }
    if (type == "TEX_ENVIRONMENT") {
      const auto id = context.graph().add_node(
          compiler::node_type::environment_texture, node_name);
      // Cycles marks Environment Texture's implicit input as LINK_POSITION.
      // shader_setup_from_background assigns the ray direction to that
      // position for world evaluation.
      bind_blender_texture_vector(
          context, id, node,
          context.conversion(
              context.geometry_output("Position", SocketType::point),
              SocketType::vector));
      const auto image = resolve_image_binding(context, node, false);
      set_image_resource_properties(context, id, image);
      static_cast<void>(context.graph().set_property(
          id, "Interpolation",
          SocketValue::string(
              context.node_property_text(node, "interpolation", "Linear"))));
      static_cast<void>(context.graph().set_property(
          id, "Projection",
          SocketValue::string(context.node_property_text(node, "projection",
                                                         "EQUIRECTANGULAR"))));
      return finish(
          {.ref = {.node = id, .socket = socket == "Alpha" ? "Alpha" : "Color"},
           .type =
               socket == "Alpha" ? SocketType::floating : SocketType::color});
    }
    return std::nullopt;
  }
};

} // namespace

std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_input_lowering_component() {
  return std::make_unique<InputNodeLoweringComponent>();
}

} // namespace psycles::adapter::detail
