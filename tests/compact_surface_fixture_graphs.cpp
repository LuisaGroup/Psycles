#include "compact_surface_fixture_graphs.h"
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_transform.h>

#include <optional>
#include <stdexcept>
#include <string>

namespace psycles::test_support {
using namespace compiler;
using namespace contract;

[[nodiscard]] ShaderGraph make_layered_principled_graph() {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry,
        "Linked coat normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Layered Principled");
    const auto configured =
        graph.set_input(
            principled,
            "BaseColor",
            SocketValue::color({0.44f, 0.13f, 0.06f})) &&
        graph.set_input(
            principled,
            "Roughness",
            SocketValue::floating(0.27f)) &&
        graph.set_input(
            principled,
            "DiffuseRoughness",
            SocketValue::floating(0.16f)) &&
        graph.set_input(
            principled,
            "Metallic",
            SocketValue::floating(0.21f)) &&
        graph.set_input(
            principled,
            "TransmissionWeight",
            SocketValue::floating(0.29f)) &&
        graph.set_input(
            principled,
            "SubsurfaceWeight",
            SocketValue::floating(0.17f)) &&
        graph.set_input(
            principled,
            "SubsurfaceRadius",
            SocketValue::vector({1.0f, 0.31f, 0.12f})) &&
        graph.set_input(
            principled,
            "SubsurfaceScale",
            SocketValue::floating(0.83f)) &&
        graph.set_input(
            principled,
            "SubsurfaceIOR",
            SocketValue::floating(1.41f)) &&
        graph.set_input(
            principled,
            "SubsurfaceAnisotropy",
            SocketValue::floating(0.23f)) &&
        graph.set_input(
            principled,
            "IOR",
            SocketValue::floating(1.52f)) &&
        graph.set_input(
            principled,
            "SpecularIORLevel",
            SocketValue::floating(0.48f)) &&
        graph.set_input(
            principled,
            "SpecularTint",
            SocketValue::color({0.82f, 0.94f, 1.0f})) &&
        graph.set_input(
            principled,
            "Alpha",
            SocketValue::floating(0.71f)) &&
        graph.set_input(
            principled,
            "ThinWall",
            SocketValue::boolean(true)) &&
        graph.set_input(
            principled,
            "SheenWeight",
            SocketValue::floating(0.24f)) &&
        graph.set_input(
            principled,
            "SheenRoughness",
            SocketValue::floating(0.39f)) &&
        graph.set_input(
            principled,
            "SheenTint",
            SocketValue::color({0.91f, 0.42f, 0.18f})) &&
        graph.set_input(
            principled,
            "CoatWeight",
            SocketValue::floating(0.19f)) &&
        graph.set_input(
            principled,
            "CoatRoughness",
            SocketValue::floating(0.12f)) &&
        graph.set_input(
            principled,
            "CoatIOR",
            SocketValue::floating(1.63f)) &&
        graph.set_input(
            principled,
            "CoatTint",
            SocketValue::color({0.31f, 0.72f, 0.95f})) &&
        graph.set_input(
            principled,
            "EmissionColor",
            SocketValue::color({0.14f, 0.37f, 0.81f})) &&
        graph.set_input(
            principled,
            "EmissionStrength",
            SocketValue::floating(1.8f)) &&
        graph.set_property(
            principled,
            "Distribution",
            SocketValue::string("MULTI_GGX")) &&
        graph.set_property(
            principled,
            "SubsurfaceMethod",
            SocketValue::string("RANDOM_WALK")) &&
        graph.connect(
            {.node = geometry, .socket = "Normal"},
            principled,
            "CoatNormal");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure layered Principled graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_automatic_normal_graph() {
    ShaderGraph graph;
    const auto normal = graph.add_node(
        node_type::normal_map,
        "Automatic surface normal");
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf,
        "Automatic-normal diffuse");
    const auto configured =
        graph.set_input(
            normal,
            "Strength",
            SocketValue::floating(0.83f)) &&
        graph.set_input(
            normal,
            "Color",
            SocketValue::color({0.78f, 0.31f, 0.91f})) &&
        graph.set_property(
            normal,
            "Space",
            SocketValue::string("TANGENT")) &&
        graph.set_property(
            normal,
            "Base",
            SocketValue::string("DISPLACED")) &&
        graph.set_property(
            normal,
            "Convention",
            SocketValue::string("OPENGL")) &&
        graph.set_input(
            diffuse,
            "Color",
            SocketValue::color({0.37f, 0.61f, 0.19f})) &&
        graph.set_input(
            diffuse,
            "Roughness",
            SocketValue::floating(0.23f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure automatic-normal graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    graph.set_root(
        ShaderDomain::surface_normal,
        OutputRef{.node = normal, .socket = "Normal"});
    return graph;
}

[[nodiscard]] ShaderGraph make_bssrdf_bump_graph(bool zero_weight) {
    ShaderGraph graph;
    const auto first_normal =
        graph.add_node(node_type::normal_map, "First BSSRDF bump normal");
    const auto second_normal =
        graph.add_node(node_type::normal_map, "Second BSSRDF bump normal");
    const auto shader_normal =
        graph.add_node(node_type::normal_map, "BSSRDF ShaderData normal");
    const auto first =
        graph.add_node(node_type::subsurface_scattering, "First BSSRDF closure");
    const auto second =
        graph.add_node(node_type::subsurface_scattering, "Second BSSRDF closure");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Ignored diffuse closure");
    const auto bssrdf_sum =
        graph.add_node(node_type::add_closure, "BSSRDF closure sum");
    const auto root =
        graph.add_node(node_type::add_closure, "BSSRDF plus diffuse");
    const auto first_color = zero_weight ? Vec3f{} : Vec3f{0.24f, 0.24f, 0.24f};
    const auto second_color = zero_weight ? Vec3f{} : Vec3f{0.71f, 0.71f, 0.71f};
    const auto configured =
        graph.set_input(first_normal, "Strength", SocketValue::floating(0.91f)) &&
        graph.set_input(first_normal, "Color",
                        SocketValue::color({0.78f, 0.29f, 0.91f})) &&
        graph.set_input(second_normal, "Strength",
                        SocketValue::floating(0.73f)) &&
        graph.set_input(second_normal, "Color",
                        SocketValue::color({0.23f, 0.81f, 0.67f})) &&
        graph.set_input(shader_normal, "Strength",
                        SocketValue::floating(0.62f)) &&
        graph.set_input(shader_normal, "Color",
                        SocketValue::color({0.62f, 0.34f, 0.88f})) &&
        graph.set_input(first, "Color", SocketValue::color(first_color)) &&
        graph.set_input(first, "Scale", SocketValue::floating(0.83f)) &&
        graph.set_input(second, "Color", SocketValue::color(second_color)) &&
        graph.set_input(second, "Scale", SocketValue::floating(1.17f)) &&
        graph.set_input(diffuse, "Color",
                        SocketValue::color({0.19f, 0.31f, 0.47f})) &&
        graph.connect({.node = first_normal, .socket = "Normal"}, first,
                      "Normal") &&
        graph.connect({.node = second_normal, .socket = "Normal"}, second,
                      "Normal") &&
        graph.connect({.node = first, .socket = "Closure"}, bssrdf_sum, "A") &&
        graph.connect({.node = second, .socket = "Closure"}, bssrdf_sum, "B") &&
        graph.connect({.node = bssrdf_sum, .socket = "Closure"}, root, "A") &&
        graph.connect({.node = diffuse, .socket = "Closure"}, root, "B");
    if (!configured) {
        throw std::runtime_error{"failed to configure BSSRDF bump graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = root, .socket = "Closure"});
    graph.set_root(ShaderDomain::surface_normal,
                   OutputRef{.node = shader_normal, .socket = "Normal"});
    return graph;
}

[[nodiscard]] ShaderGraph make_nested_bump_graph() {
  ShaderGraph graph;
  const auto inner = graph.add_node(node_type::bump, "Nested height Bump");
  const auto normal_to_vector = graph.add_node(node_type::normal_to_vector,
                                               "Nested Bump normal to vector");
  const auto vector_to_scalar = graph.add_node(node_type::vector_to_scalar,
                                               "Nested Bump vector to height");
  const auto outer = graph.add_node(node_type::bump, "Root Bump");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Nested Bump diffuse");
  const auto configured =
      graph.set_input(inner, "Height", SocketValue::floating(0.37f)) &&
      graph.set_input(inner, "Strength", SocketValue::floating(0.61f)) &&
      graph.set_input(inner, "Distance", SocketValue::floating(0.29f)) &&
      graph.set_property(inner, "Invert", SocketValue::boolean(false)) &&
      graph.set_property(inner, "NormalLinked", SocketValue::boolean(false)) &&
      graph.set_input(outer, "Strength", SocketValue::floating(0.73f)) &&
      graph.set_input(outer, "Distance", SocketValue::floating(0.41f)) &&
      graph.set_property(outer, "Invert", SocketValue::boolean(true)) &&
      graph.set_property(outer, "NormalLinked", SocketValue::boolean(false)) &&
      graph.set_input(diffuse, "Color",
                      SocketValue::color({0.27f, 0.53f, 0.81f})) &&
      graph.set_input(diffuse, "Roughness", SocketValue::floating(0.19f)) &&
      graph.connect({.node = inner, .socket = "Normal"}, normal_to_vector,
                    "Normal") &&
      graph.connect({.node = normal_to_vector, .socket = "Vector"},
                    vector_to_scalar, "Vector") &&
      graph.connect({.node = vector_to_scalar, .socket = "Value"}, outer,
                    "Height") &&
      graph.connect({.node = outer, .socket = "Normal"}, diffuse, "Normal");
  if (!configured) {
    throw std::runtime_error{"failed to configure nested Bump graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_capacity_transparency_graph(
    std::uint32_t prefix_count) {
    ShaderGraph graph;
    std::optional<NodeId> root;
    const auto append = [&](NodeId closure) {
        if (!root) {
            root = closure;
            return;
        }
        const auto add = graph.add_node(
            node_type::add_closure,
            "Capacity sequence");
        if (!graph.connect(
                {.node = *root, .socket = "Closure"}, add, "A") ||
            !graph.connect(
                {.node = closure, .socket = "Closure"}, add, "B")) {
            throw std::runtime_error{
                "failed to append capacity closure"};
        }
        root = add;
    };

    for (auto index = 0u; index < prefix_count; ++index) {
        const auto diffuse = graph.add_node(
            node_type::diffuse_bsdf,
            "Capacity prefix diffuse");
        const auto color = 0.025f + 0.006f * static_cast<float>(index);
        const auto x = -0.24f + 0.04f * static_cast<float>(index);
        if (!graph.set_input(
                diffuse,
                "Color",
                SocketValue::color({color, 0.7f * color, 0.4f * color})) ||
            !graph.set_input(
                diffuse,
                "Roughness",
                SocketValue::floating(0.03f * static_cast<float>(index))) ||
            !graph.set_input(
                diffuse,
                "Normal",
                SocketValue::normal({x, 0.0f, 0.97f}))) {
            throw std::runtime_error{
                "failed to configure capacity prefix"};
        }
        append(diffuse);
    }

    const auto rejected = graph.add_node(
        node_type::transparent_bsdf,
        "Sub-cutoff transparent");
    const auto allocated = graph.add_node(
        node_type::transparent_bsdf,
        "Allocated transparent");
    if (!graph.set_input(
            rejected,
            "Color",
            SocketValue::color({0.5e-5f, 0.5e-5f, 0.5e-5f})) ||
        !graph.set_input(
            allocated,
            "Color",
            SocketValue::color({0.23f, 0.19f, 0.17f}))) {
        throw std::runtime_error{
            "failed to configure capacity transparency"};
    }
    append(rejected);
    append(allocated);

    for (auto index = 0u; index < 2u; ++index) {
        const auto diffuse = graph.add_node(
            node_type::diffuse_bsdf,
            "Capacity suffix diffuse");
        if (!graph.set_input(
                diffuse,
                "Color",
                SocketValue::color(
                    {0.71f + 0.08f * static_cast<float>(index),
                     0.11f,
                     0.06f})) ||
            !graph.set_input(
                diffuse,
                "Roughness",
                SocketValue::floating(0.77f + 0.1f * index)) ||
            !graph.set_input(
                diffuse,
                "Normal",
                SocketValue::normal({0.0f, 0.6f, 0.8f}))) {
            throw std::runtime_error{
                "failed to configure capacity suffix"};
        }
        append(diffuse);
    }
    // Once the capacity is full, a later transparent setup must still merge
    // into the first transparent slot instead of requiring another slot.
    const auto merged = graph.add_node(
        node_type::transparent_bsdf,
        "Post-capacity transparent merge");
    if (!graph.set_input(
            merged,
            "Color",
            SocketValue::color({0.11f, 0.07f, 0.05f}))) {
        throw std::runtime_error{
            "failed to configure post-capacity transparency"};
    }
    append(merged);
    if (!root) {
        throw std::runtime_error{"empty capacity graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = *root, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_transformed_emission_graph(
    const std::array<float, 16u> &world_to_object, std::string_view label) {
    ShaderGraph graph;
    const auto coordinates = graph.add_node(node_type::texture_coordinate,
                                            std::string{label} + " coordinates");
    const auto point_to_vector = graph.add_node(
        node_type::point_to_vector, std::string{label} + " point to vector");
    Mat4f world_to_object_transform;
    world_to_object_transform.elements = world_to_object;
    const auto object_to_world = cycles_inverse_affine_transform(
        world_to_object_transform);
    const auto conversion = graph.add_node(node_type::vector_to_color,
                                           std::string{label} + " vector to color");
    const auto emission = graph.add_node(node_type::emission,
                                         std::string{label} + " emission");
    const auto configured =
        graph.set_property(coordinates, "UseTransform", SocketValue::boolean(true)) &&
        graph.set_property(coordinates, "ObjectTransform", SocketValue::transform(object_to_world)) &&
        graph.connect(
            {.node = coordinates, .socket = "Object"},
            point_to_vector,
            "Point") &&
        graph.connect(
            {.node = point_to_vector, .socket = "Vector"},
            conversion,
            "Vector") &&
        graph.connect(
            {.node = conversion, .socket = "Color"},
            emission,
            "Color") &&
        graph.set_input(
            emission,
            "Strength",
            SocketValue::floating(1.0f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure transformed emission graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_light_path_depth_emission_graph(bool portal) {
    ShaderGraph graph;
    const auto light_path = graph.add_node(
        node_type::light_path,
        portal ? "Portal depth" : "Transmission depth");
    const auto emission = graph.add_node(
        node_type::emission,
        portal ? "Portal emission" : "Transmission emission");
    if (!graph.connect(
            {.node = light_path,
             .socket = portal ? "PortalDepth" : "TransmissionDepth"},
            emission,
            "Strength") ||
        !graph.set_input(
            emission,
            "Color",
            SocketValue::color({1.0f, 1.0f, 1.0f}))) {
        throw std::runtime_error{
            "failed to configure Light Path depth emission graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

} // namespace psycles::test_support
