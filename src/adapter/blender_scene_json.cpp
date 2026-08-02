#include "blender_scene_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <cmath>
#include <limits>
#include <utility>

namespace psycles::adapter::detail {

using contract::NishitaSkyDesc;
using contract::ShaderDomain;
using contract::ShaderGraph;
using contract::SocketValue;

void DocumentDeleter::operator()(
    yyjson_doc *document) const noexcept {
    yyjson_doc_free(document);
}

[[nodiscard]] yyjson_val *member(
    yyjson_val *object,
    const char *name) noexcept {
    return object == nullptr ? nullptr : yyjson_obj_get(object, name);
}

[[nodiscard]] std::string text(
    yyjson_val *value,
    std::string fallback) {
    if (value == nullptr || !yyjson_is_str(value)) {
        return fallback;
    }
    return std::string{yyjson_get_str(value)};
}

[[nodiscard]] std::uint64_t unsigned_number(
    yyjson_val *value,
    std::uint64_t fallback) noexcept {
    return value != nullptr && yyjson_is_uint(value)
               ? yyjson_get_uint(value)
               : fallback;
}

[[nodiscard]] std::optional<std::uint32_t>
optional_unsigned_number(yyjson_val *value) noexcept {
    if (value == nullptr || !yyjson_is_uint(value)) {
        return std::nullopt;
    }
    const auto number = yyjson_get_uint(value);
    if (number > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(number);
}

[[nodiscard]] std::int64_t signed_number(
    yyjson_val *value,
    std::int64_t fallback) noexcept {
    if (value == nullptr) {
        return fallback;
    }
    if (yyjson_is_sint(value)) {
        return yyjson_get_sint(value);
    }
    if (yyjson_is_uint(value) &&
        yyjson_get_uint(value) <=
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max())) {
        return static_cast<std::int64_t>(
            yyjson_get_uint(value));
    }
    return fallback;
}

[[nodiscard]] float number(
    yyjson_val *value,
    float fallback) noexcept {
    return value != nullptr && yyjson_is_num(value)
               ? static_cast<float>(yyjson_get_num(value))
               : fallback;
}

[[nodiscard]] bool boolean(
    yyjson_val *value,
    bool fallback) noexcept {
    return value != nullptr && yyjson_is_bool(value)
               ? yyjson_get_bool(value)
               : fallback;
}

[[nodiscard]] std::uint32_t ray_visibility_mask(
    yyjson_val *visibility) noexcept {
    std::uint32_t mask = 0u;
    for (const auto &[name, bit] : {
             std::pair{
                 "camera",
                 contract::RayVisibility::camera},
             std::pair{
                 "diffuse",
                 contract::RayVisibility::diffuse},
             std::pair{
                 "glossy",
                 contract::RayVisibility::glossy},
             std::pair{
                 "transmission",
                 contract::RayVisibility::transmission},
             std::pair{
                 "shadow",
                 contract::RayVisibility::shadow},
             std::pair{
                 "volume_scatter",
                 contract::RayVisibility::volume_scatter}}) {
        if (boolean(
                member(visibility, name),
                visibility == nullptr)) {
            mask |= contract::visibility_bit(bit);
        }
    }
    return mask;
}

[[nodiscard]] Vec3f float3(
    yyjson_val *value,
    Vec3f fallback) noexcept {
    if (value == nullptr || !yyjson_is_arr(value) ||
        yyjson_arr_size(value) < 3u) {
        return fallback;
    }
    return {
        number(yyjson_arr_get(value, 0u), fallback.x),
        number(yyjson_arr_get(value, 1u), fallback.y),
        number(yyjson_arr_get(value, 2u), fallback.z)};
}

[[nodiscard]] Mat4f matrix(yyjson_val *value) noexcept {
    Mat4f result{};
    if (value == nullptr || !yyjson_is_arr(value) ||
        yyjson_arr_size(value) != result.elements.size()) {
        return result;
    }
    for (std::size_t i = 0u; i < result.elements.size(); ++i) {
        result.elements[i] =
            number(yyjson_arr_get(value, i), 0.0f);
    }
    return result;
}

[[nodiscard]] yyjson_val *find_socket(
    yyjson_val *node,
    const char *collection,
    std::string_view identifier) noexcept {
    auto *sockets = member(node, collection);
    if (sockets == nullptr || !yyjson_is_arr(sockets)) {
        return nullptr;
    }
    yyjson_arr_iter iterator = yyjson_arr_iter_with(sockets);
    while (auto *socket = yyjson_arr_iter_next(&iterator)) {
        if (text(member(socket, "identifier")) == identifier) {
            return socket;
        }
    }
    return nullptr;
}

[[nodiscard]] std::optional<NishitaSkyDesc>
find_simple_world_nishita(yyjson_val *world) {
    auto *tree = member(world, "node_tree");
    auto *root = member(tree, "surface_root");
    const auto background_name = text(member(root, "node"));
    if (background_name.empty()) {
        return std::nullopt;
    }

    yyjson_val *background = nullptr;
    yyjson_val *sky = nullptr;
    auto *nodes = member(tree, "nodes");
    if (nodes == nullptr || !yyjson_is_arr(nodes)) {
        return std::nullopt;
    }
    yyjson_arr_iter node_iterator =
        yyjson_arr_iter_with(nodes);
    while (auto *node =
               yyjson_arr_iter_next(&node_iterator)) {
        if (text(member(node, "name")) == background_name &&
            text(member(node, "type")) == "BACKGROUND") {
            background = node;
            break;
        }
    }
    if (background == nullptr) {
        return std::nullopt;
    }

    std::string sky_name;
    auto *links = member(tree, "links");
    if (links != nullptr && yyjson_is_arr(links)) {
        yyjson_arr_iter link_iterator =
            yyjson_arr_iter_with(links);
        while (auto *link =
                   yyjson_arr_iter_next(&link_iterator)) {
            if (text(member(link, "to_node")) ==
                    background_name &&
                text(member(link, "to_socket")) == "Color") {
                sky_name = text(member(link, "from_node"));
                break;
            }
        }
    }
    if (sky_name.empty()) {
        return std::nullopt;
    }
    node_iterator = yyjson_arr_iter_with(nodes);
    while (auto *node =
               yyjson_arr_iter_next(&node_iterator)) {
        const auto sky_type = text(
            member(member(node, "properties"), "sky_type"),
            "NISHITA");
        // Blender 5.2 split the procedural Nishita implementation into
        // explicit single- and multiple-scattering modes. Psycles currently
        // implements Cycles' single-scattering equations; NISHITA is the
        // legacy spelling for the same supported procedural path.
        const auto supported_single_scattering =
            sky_type == "NISHITA" ||
            sky_type == "SINGLE_SCATTERING";
        if (text(member(node, "name")) == sky_name &&
            text(member(node, "type")) == "TEX_SKY" &&
            supported_single_scattering) {
            sky = node;
            break;
        }
    }
    if (sky == nullptr) {
        return std::nullopt;
    }

    // A linked Background Strength requires evaluating the complete world
    // graph into the environment distribution. Keep that case on the normal
    // graph path instead of silently freezing it.
    if (links != nullptr && yyjson_is_arr(links)) {
        yyjson_arr_iter link_iterator =
            yyjson_arr_iter_with(links);
        while (auto *link =
                   yyjson_arr_iter_next(&link_iterator)) {
            if (text(member(link, "to_node")) ==
                    background_name &&
                text(member(link, "to_socket")) == "Strength") {
                return std::nullopt;
            }
        }
    }

    constexpr auto pi = 3.14159265358979323846f;
    constexpr auto two_pi = 2.0f * pi;
    auto *properties = member(sky, "properties");
    auto elevation = std::fmod(
        number(
            member(properties, "sun_elevation"),
            0.7853982f),
        two_pi);
    auto rotation = number(
        member(properties, "sun_rotation"), 0.0f);
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

    auto *strength_socket =
        find_socket(background, "inputs", "Strength");
    const auto sun_disc =
        boolean(member(properties, "sun_disc"), true);
    return NishitaSkyDesc{
        .sun_elevation = elevation,
        .sun_rotation = rotation,
        .angular_diameter =
            sun_disc
                ? number(
                      member(properties, "sun_size"),
                      0.00918043f)
                : -1.0f,
        .sun_intensity = number(
            member(properties, "sun_intensity"), 1.0f),
        .altitude =
            number(member(properties, "altitude"), 0.0f),
        .air_density =
            number(member(properties, "air_density"), 1.0f),
        .dust_density =
            number(
                member(properties, "aerosol_density"),
                number(
                    member(properties, "dust_density"),
                    1.0f)),
        .ozone_density =
            number(member(properties, "ozone_density"), 1.0f),
        .background_strength = number(
            member(strength_socket, "default"), 1.0f)};
}

[[nodiscard]] ShaderGraph diffuse_graph(
    Vec3f color,
    float roughness) {
    ShaderGraph graph;
    const auto closure =
        graph.add_node(compiler::node_type::diffuse_bsdf);
    static_cast<void>(graph.set_input(
        closure, "Color", SocketValue::color(color)));
    static_cast<void>(graph.set_input(
        closure,
        "Roughness",
        SocketValue::floating(roughness)));
    static_cast<void>(graph.set_input(
        closure,
        "Normal",
        SocketValue::normal({0.0f, 0.0f, 0.0f})));
    graph.set_root(
        ShaderDomain::surface,
        contract::OutputRef{closure, "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph cycles_default_surface_graph() {
    ShaderGraph graph;
    const auto closure =
        graph.add_node(
            compiler::node_type::principled_bsdf,
            "Cycles Default Surface");
    // ShaderManager::add_default() constructs a default Principled BSDF.
    // Keep these values explicit so this adapter contract cannot silently
    // drift with Psycles' own node-schema defaults.
    static_cast<void>(graph.set_input(
        closure,
        "BaseColor",
        SocketValue::color({0.8f, 0.8f, 0.8f})));
    static_cast<void>(graph.set_input(
        closure,
        "Metallic",
        SocketValue::floating(0.0f)));
    static_cast<void>(graph.set_input(
        closure,
        "Roughness",
        SocketValue::floating(0.5f)));
    static_cast<void>(graph.set_input(
        closure,
        "DiffuseRoughness",
        SocketValue::floating(0.0f)));
    static_cast<void>(graph.set_input(
        closure,
        "SubsurfaceWeight",
        SocketValue::floating(0.0f)));
    static_cast<void>(graph.set_input(
        closure,
        "SubsurfaceRadius",
        SocketValue::vector({1.0f, 0.2f, 0.1f})));
    static_cast<void>(graph.set_input(
        closure,
        "SubsurfaceScale",
        SocketValue::floating(0.005f)));
    static_cast<void>(graph.set_input(
        closure,
        "IOR",
        SocketValue::floating(1.5f)));
    static_cast<void>(graph.set_input(
        closure,
        "SpecularIORLevel",
        SocketValue::floating(0.5f)));
    static_cast<void>(graph.set_input(
        closure,
        "SpecularTint",
        SocketValue::color({1.0f, 1.0f, 1.0f})));
    static_cast<void>(graph.set_input(
        closure,
        "Normal",
        SocketValue::normal({0.0f, 0.0f, 0.0f})));
    static_cast<void>(graph.set_property(
        closure,
        "Distribution",
        SocketValue::string("MULTI_GGX")));
    graph.set_root(
        ShaderDomain::surface,
        contract::OutputRef{closure, "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph emission_graph(
    Vec3f color,
    float strength) {
    ShaderGraph graph;
    const auto closure =
        graph.add_node(compiler::node_type::emission);
    static_cast<void>(graph.set_input(
        closure, "Color", SocketValue::color(color)));
    static_cast<void>(graph.set_input(
        closure,
        "Strength",
        SocketValue::floating(strength)));
    graph.set_root(
        ShaderDomain::surface,
        contract::OutputRef{closure, "Closure"});
    return graph;
}

}// namespace psycles::adapter::detail
