#include <psycles/adapter/blender_scene.h>

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <yyjson.h>

namespace psycles::adapter {

namespace {

using contract::CameraDesc;
using contract::CameraId;
using contract::CameraProjection;
using contract::CameraSensorFit;
using contract::EnvironmentDesc;
using contract::EnvironmentSunDesc;
using contract::GeometryId;
using contract::ImageColorSpace;
using contract::ImageDesc;
using contract::ImageId;
using contract::InstanceId;
using contract::LightDesc;
using contract::LightId;
using contract::LightType;
using contract::MaterialDesc;
using contract::MaterialId;
using contract::SceneSnapshot;
using contract::ShaderDomain;
using contract::ShaderGraph;
using contract::SocketValue;
using contract::TriangleMeshDesc;

struct DocumentDeleter {
    void operator()(yyjson_doc *document) const noexcept {
        yyjson_doc_free(document);
    }
};

using Document = std::unique_ptr<yyjson_doc, DocumentDeleter>;

[[nodiscard]] yyjson_val *member(
    yyjson_val *object,
    const char *name) noexcept {
    return object == nullptr ? nullptr : yyjson_obj_get(object, name);
}

[[nodiscard]] std::string text(
    yyjson_val *value,
    std::string fallback = {}) {
    if (value == nullptr || !yyjson_is_str(value)) {
        return fallback;
    }
    return std::string{yyjson_get_str(value)};
}

[[nodiscard]] std::uint64_t unsigned_number(
    yyjson_val *value,
    std::uint64_t fallback = 0u) noexcept {
    return value != nullptr && yyjson_is_uint(value)
               ? yyjson_get_uint(value)
               : fallback;
}

[[nodiscard]] float number(
    yyjson_val *value,
    float fallback = 0.0f) noexcept {
    return value != nullptr && yyjson_is_num(value)
               ? static_cast<float>(yyjson_get_num(value))
               : fallback;
}

[[nodiscard]] bool boolean(
    yyjson_val *value,
    bool fallback = false) noexcept {
    return value != nullptr && yyjson_is_bool(value)
               ? yyjson_get_bool(value)
               : fallback;
}

[[nodiscard]] Vec3f float3(
    yyjson_val *value,
    Vec3f fallback = {}) noexcept {
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

[[nodiscard]] ShaderGraph diffuse_graph(
    Vec3f color,
    float roughness = 0.0f) {
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

struct TypedOutput {
    contract::OutputRef ref;
    contract::SocketType type{contract::SocketType::floating};
};

struct RawOutputKey {
    std::string node;
    std::string socket;

    auto operator<=>(const RawOutputKey &) const noexcept = default;
};

class BlenderGraphNormalizer {

private:
    using RawNodeMap =
        std::map<std::string, yyjson_val *, std::less<>>;
    using RawLinkMap =
        std::map<RawOutputKey, RawOutputKey>;
    using LoweredOutputMap =
        std::map<RawOutputKey, TypedOutput>;
    using GroupInputMap =
        std::map<std::string, TypedOutput, std::less<>>;
    using NodeGroupMap =
        std::map<std::string, yyjson_val *, std::less<>>;

    yyjson_val *_tree{};
    std::string _material_name;
    const std::map<std::string, ImageId, std::less<>> &_image_ids;
    const std::map<std::string, ImageColorSpace, std::less<>> &
        _image_color_spaces;
    const NodeGroupMap &_node_groups;
    std::vector<BlenderSceneDiagnostic> &_diagnostics;
    ShaderGraph _graph;
    RawNodeMap _raw_nodes;
    RawLinkMap _links;
    LoweredOutputMap _outputs;
    GroupInputMap _group_inputs;
    std::set<std::string, std::less<>> _building;
    std::set<std::string, std::less<>> _group_stack;
    std::set<std::string, std::less<>> _warned;
    std::optional<contract::NodeId> _default_image_coordinates;
    std::optional<contract::NodeId> _geometry;

private:
    void warn_once(std::string key, std::string message) {
        if (_warned.emplace(std::move(key)).second) {
            _diagnostics.emplace_back(BlenderSceneDiagnostic{
                .severity =
                    BlenderSceneDiagnosticSeverity::warning,
                .message =
                    "material '" + _material_name + "': " +
                    std::move(message)});
        }
    }

    [[nodiscard]] yyjson_val *raw_node(
        std::string_view name) const noexcept {
        auto iter = _raw_nodes.find(name);
        return iter == _raw_nodes.end() ? nullptr : iter->second;
    }

    [[nodiscard]] yyjson_val *raw_input(
        yyjson_val *node,
        std::string_view identifier) const noexcept {
        return find_socket(node, "inputs", identifier);
    }

    [[nodiscard]] yyjson_val *raw_output(
        yyjson_val *node,
        std::string_view identifier) const noexcept {
        return find_socket(node, "outputs", identifier);
    }

    [[nodiscard]] static contract::SocketType socket_type(
        yyjson_val *socket,
        contract::SocketType fallback =
            contract::SocketType::color) {
        using contract::SocketType;
        const auto type = text(member(socket, "type"));
        if (type.find("Shader") != std::string::npos) {
            return SocketType::closure;
        }
        if (type.find("Color") != std::string::npos) {
            return SocketType::color;
        }
        if (type.find("Vector") != std::string::npos) {
            return SocketType::vector;
        }
        if (type.find("Float") != std::string::npos) {
            return SocketType::floating;
        }
        if (type.find("Bool") != std::string::npos) {
            return SocketType::boolean;
        }
        if (type.find("Int") != std::string::npos) {
            return SocketType::integer;
        }
        return fallback;
    }

    void load_tree_context(yyjson_val *tree) {
        _tree = tree;
        _raw_nodes.clear();
        _links.clear();
        _outputs.clear();
        _building.clear();

        auto *nodes = member(_tree, "nodes");
        if (nodes != nullptr && yyjson_is_arr(nodes)) {
            yyjson_arr_iter iterator =
                yyjson_arr_iter_with(nodes);
            while (auto *node =
                       yyjson_arr_iter_next(&iterator)) {
                _raw_nodes.emplace(
                    text(member(node, "name")), node);
            }
        }
        auto *links = member(_tree, "links");
        if (links != nullptr && yyjson_is_arr(links)) {
            yyjson_arr_iter iterator =
                yyjson_arr_iter_with(links);
            while (auto *link =
                       yyjson_arr_iter_next(&iterator)) {
                _links.insert_or_assign(
                    RawOutputKey{
                        .node = text(
                            member(link, "to_node")),
                        .socket = text(
                            member(link, "to_socket"))},
                    RawOutputKey{
                        .node = text(
                            member(link, "from_node")),
                        .socket = text(
                            member(link, "from_socket"))});
            }
        }
    }

    [[nodiscard]] std::optional<RawOutputKey> input_source(
        yyjson_val *node,
        std::string_view identifier) const {
        const auto key = RawOutputKey{
            .node = text(member(node, "name")),
            .socket = std::string{identifier}};
        auto iter = _links.find(key);
        return iter == _links.end()
                   ? std::nullopt
                   : std::optional<RawOutputKey>{iter->second};
    }

    [[nodiscard]] std::string node_property_text(
        yyjson_val *node,
        const char *name,
        std::string fallback = {}) const {
        return text(
            member(member(node, "properties"), name),
            std::move(fallback));
    }

    [[nodiscard]] float node_property_number(
        yyjson_val *node,
        const char *name,
        float fallback = 0.0f) const noexcept {
        return number(
            member(member(node, "properties"), name),
            fallback);
    }

    [[nodiscard]] bool node_property_bool(
        yyjson_val *node,
        const char *name,
        bool fallback = false) const noexcept {
        auto *value = member(member(node, "properties"), name);
        return value != nullptr && yyjson_is_bool(value)
                   ? yyjson_get_bool(value)
                   : fallback;
    }

    [[nodiscard]] static SocketValue literal(
        yyjson_val *value,
        contract::SocketType type) {
        using contract::SocketType;
        switch (type) {
            case SocketType::floating:
                return SocketValue::floating(number(value));
            case SocketType::color:
                return SocketValue::color(float3(value));
            case SocketType::normal:
                return SocketValue::normal(float3(value));
            case SocketType::point:
                return SocketValue::point(float3(value));
            case SocketType::vector:
            case SocketType::float3:
                return SocketValue::vector(float3(value));
            case SocketType::boolean:
                return SocketValue::boolean(
                    value != nullptr && yyjson_is_bool(value)
                        ? yyjson_get_bool(value)
                        : false);
            case SocketType::unsigned_integer:
                return SocketValue::unsigned_integer(
                    unsigned_number(value));
            case SocketType::integer:
                return SocketValue::integer(
                    static_cast<std::int64_t>(number(value)));
            case SocketType::string:
                return SocketValue::string(text(value));
            default:
                return SocketValue::floating(0.0f);
        }
    }

    [[nodiscard]] TypedOutput default_image_coordinates() {
        if (!_default_image_coordinates) {
            _default_image_coordinates =
                _graph.add_node(
                    compiler::node_type::texture_coordinate,
                    "Default Image Coordinates");
        }
        return {
            .ref = {
                .node = *_default_image_coordinates,
                .socket = "UV"},
            .type = contract::SocketType::vector};
    }

    [[nodiscard]] TypedOutput geometry_output(
        std::string socket,
        contract::SocketType type) {
        if (!_geometry) {
            _geometry = _graph.add_node(
                compiler::node_type::geometry,
                "Geometry");
        }
        return {
            .ref = {
                .node = *_geometry,
                .socket = std::move(socket)},
            .type = type};
    }

    [[nodiscard]] TypedOutput conversion(
        TypedOutput source,
        contract::SocketType target) {
        using contract::SocketType;
        if (source.type == target) {
            return source;
        }
        auto vector_like = [](SocketType type) {
            return type == SocketType::vector ||
                   type == SocketType::point ||
                   type == SocketType::normal ||
                   type == SocketType::float3;
        };

        const char *node_type = nullptr;
        std::string input;
        std::string output;
        if (source.type == SocketType::floating &&
            target == SocketType::color) {
            node_type = compiler::node_type::scalar_to_color;
            input = "Value";
            output = "Color";
        } else if (
            source.type == SocketType::color &&
            target == SocketType::floating) {
            node_type = compiler::node_type::color_to_scalar;
            input = "Color";
            output = "Value";
        } else if (
            vector_like(source.type) &&
            target == SocketType::floating) {
            if (source.type == SocketType::normal) {
                source = conversion(source, SocketType::vector);
            }
            node_type = compiler::node_type::vector_to_scalar;
            input = "Vector";
            output = "Value";
        } else if (
            source.type == SocketType::vector &&
            target == SocketType::color) {
            node_type = compiler::node_type::vector_to_color;
            input = "Vector";
            output = "Color";
        } else if (
            source.type == SocketType::color &&
            target == SocketType::vector) {
            node_type = compiler::node_type::color_to_vector;
            input = "Color";
            output = "Vector";
        } else if (
            source.type == SocketType::vector &&
            target == SocketType::normal) {
            node_type = compiler::node_type::vector_to_normal;
            input = "Vector";
            output = "Normal";
        } else if (
            source.type == SocketType::normal &&
            target == SocketType::vector) {
            node_type = compiler::node_type::normal_to_vector;
            input = "Normal";
            output = "Vector";
        } else if (
            source.type == SocketType::color &&
            target == SocketType::normal) {
            return conversion(
                conversion(source, SocketType::vector),
                target);
        } else if (
            source.type == SocketType::normal &&
            target == SocketType::color) {
            return conversion(
                conversion(source, SocketType::vector),
                target);
        } else if (
            source.type == SocketType::floating &&
            target == SocketType::vector) {
            return conversion(
                conversion(source, SocketType::color),
                target);
        } else if (
            vector_like(source.type) &&
            target == SocketType::vector) {
            if (source.type == SocketType::normal) {
                return conversion(source, SocketType::vector);
            }
            return {
                .ref = source.ref,
                .type = SocketType::vector};
        }

        if (node_type == nullptr) {
            warn_once(
                "conversion:" +
                    std::to_string(
                        static_cast<std::uint32_t>(source.type)) +
                    ":" +
                    std::to_string(
                        static_cast<std::uint32_t>(target)),
                "unsupported implicit socket conversion; "
                "using a zero literal");
            const auto constant =
                target == SocketType::floating
                    ? _graph.add_node(
                          compiler::node_type::constant_float)
                    : _graph.add_node(
                          compiler::node_type::constant_color);
            if (target == SocketType::floating) {
                static_cast<void>(_graph.set_input(
                    constant,
                    "Value",
                    SocketValue::floating(0.0f)));
                return {
                    .ref = {.node = constant, .socket = "Value"},
                    .type = target};
            }
            static_cast<void>(_graph.set_input(
                constant,
                "Color",
                SocketValue::color({0.0f, 0.0f, 0.0f})));
            return {
                .ref = {.node = constant, .socket = "Color"},
                .type = SocketType::color};
        }

        const auto converted =
            _graph.add_node(node_type, "Implicit Conversion");
        static_cast<void>(_graph.connect(
            source.ref, converted, input));
        return {
            .ref = {
                .node = converted,
                .socket = std::move(output)},
            .type = target};
    }

    [[nodiscard]] bool bind(
        contract::NodeId destination,
        std::string target_socket,
        yyjson_val *raw_destination,
        std::string_view raw_input_name,
        contract::SocketType target_type) {
        if (auto source =
                input_source(
                    raw_destination, raw_input_name)) {
            auto output = lower_output(
                source->node,
                source->socket,
                target_type);
            return _graph.connect(
                output.ref,
                destination,
                std::move(target_socket));
        }
        auto *socket =
            raw_input(raw_destination, raw_input_name);
        return _graph.set_input(
            destination,
            std::move(target_socket),
            literal(member(socket, "default"), target_type));
    }

    [[nodiscard]] static float evaluate_curve(
        yyjson_val *curve,
        float x) noexcept {
        auto *points = member(curve, "points");
        if (points == nullptr || !yyjson_is_arr(points) ||
            yyjson_arr_size(points) == 0u) {
            return x;
        }
        auto *first = yyjson_arr_get(points, 0u);
        auto previous = float3(
            member(first, "location"),
            {0.0f, 0.0f, 0.0f});
        for (std::size_t i = 1u;
             i < yyjson_arr_size(points);
             ++i) {
            auto current = float3(
                member(
                    yyjson_arr_get(points, i),
                    "location"),
                {1.0f, 1.0f, 0.0f});
            if (x <= current.x) {
                const auto t =
                    std::clamp(
                        (x - previous.x) /
                            std::max(
                                current.x - previous.x,
                                1.0e-20f),
                        0.0f,
                        1.0f);
                return previous.y +
                       (current.y - previous.y) * t;
            }
            previous = current;
        }
        return previous.y;
    }

    [[nodiscard]] std::string color_ramp_table(
        yyjson_val *node) const {
        auto *ramp = member(
            member(node, "special"), "color_ramp");
        auto *elements = member(ramp, "elements");
        std::ostringstream stream;
        stream << std::setprecision(9);
        if (elements != nullptr && yyjson_is_arr(elements)) {
            yyjson_arr_iter iterator =
                yyjson_arr_iter_with(elements);
            bool first = true;
            while (auto *element =
                       yyjson_arr_iter_next(&iterator)) {
                auto color = member(element, "color");
                if (!first) {
                    stream << ';';
                }
                first = false;
                stream
                    << number(member(element, "position"))
                    << ',' << number(yyjson_arr_get(color, 0u))
                    << ',' << number(yyjson_arr_get(color, 1u))
                    << ',' << number(yyjson_arr_get(color, 2u))
                    << ',' << number(yyjson_arr_get(color, 3u), 1.0f);
            }
        }
        return stream.str();
    }

    [[nodiscard]] std::string rgb_curve_table(
        yyjson_val *node) const {
        auto *mapping = member(
            member(node, "special"), "curve_mapping");
        auto *curves = member(mapping, "curves");
        if (curves == nullptr || !yyjson_is_arr(curves) ||
            yyjson_arr_size(curves) < 4u) {
            return {};
        }
        std::ostringstream stream;
        stream << std::setprecision(9);
        for (std::uint32_t i = 0u; i <= 16u; ++i) {
            const auto x = static_cast<float>(i) / 16.0f;
            if (i != 0u) {
                stream << ';';
            }
            stream << x;
            for (std::size_t channel = 1u;
                 channel <= 3u;
                 ++channel) {
                auto value = evaluate_curve(
                    yyjson_arr_get(curves, channel), x);
                value = evaluate_curve(
                    yyjson_arr_get(curves, 0u), value);
                stream << ',' << value;
            }
        }
        return stream.str();
    }

    [[nodiscard]] TypedOutput constant_from_socket(
        yyjson_val *socket,
        std::string label,
        contract::SocketType type) {
        using contract::SocketType;
        if (type == SocketType::closure) {
            const auto id = _graph.add_node(
                compiler::node_type::diffuse_bsdf,
                std::move(label));
            static_cast<void>(_graph.set_input(
                id,
                "Color",
                SocketValue::color({0.0f, 0.0f, 0.0f})));
            static_cast<void>(_graph.set_input(
                id,
                "Roughness",
                SocketValue::floating(0.0f)));
            static_cast<void>(_graph.set_input(
                id,
                "Normal",
                SocketValue::normal({0.0f, 0.0f, 0.0f})));
            return {
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure};
        }
        if (type == SocketType::floating ||
            type == SocketType::boolean ||
            type == SocketType::integer ||
            type == SocketType::unsigned_integer) {
            const auto id = _graph.add_node(
                compiler::node_type::constant_float,
                std::move(label));
            static_cast<void>(_graph.set_input(
                id,
                "Value",
                SocketValue::floating(number(
                    member(socket, "default")))));
            return {
                .ref = {.node = id, .socket = "Value"},
                .type = SocketType::floating};
        }
        const auto id = _graph.add_node(
            compiler::node_type::constant_color,
            std::move(label));
        static_cast<void>(_graph.set_input(
            id,
            "Color",
            literal(
                member(socket, "default"),
                SocketType::color)));
        TypedOutput result{
            .ref = {.node = id, .socket = "Color"},
            .type = SocketType::color};
        return type == SocketType::color
                   ? result
                   : conversion(result, type);
    }

    [[nodiscard]] TypedOutput constant_from_output(
        yyjson_val *node,
        std::string_view socket,
        contract::SocketType type) {
        return constant_from_socket(
            raw_output(node, socket),
            text(member(node, "name")),
            type);
    }

    [[nodiscard]] TypedOutput lower_natural_output(
        const std::string &node_name,
        const std::string &socket) {
        using contract::SocketType;
        auto *node = raw_node(node_name);
        if (node == nullptr) {
            warn_once(
                "missing:" + node_name,
                "node '" + node_name + "' is missing");
            return constant_from_output(
                nullptr, {}, SocketType::color);
        }
        const auto type = text(member(node, "type"));

        if (type == "REROUTE") {
            auto source = input_source(node, "Input");
            if (source) {
                return lower_natural_output(
                    source->node, source->socket);
            }
            return constant_from_output(
                node, socket, SocketType::color);
        }
        if (type == "BUMP") {
            const auto id = _graph.add_node(
                compiler::node_type::bump,
                node_name);
            static_cast<void>(bind(
                id,
                "Height",
                node,
                "Height",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "Strength",
                node,
                "Strength",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "Distance",
                node,
                "Distance",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "FilterWidth",
                node,
                "Filter Width",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "Normal",
                node,
                "Normal",
                SocketType::normal));
            static_cast<void>(_graph.set_property(
                id,
                "Invert",
                SocketValue::boolean(node_property_bool(
                    node, "invert"))));
            return {
                .ref = {.node = id, .socket = "Normal"},
                .type = SocketType::normal};
        }
        if (type == "GROUP_INPUT") {
            if (auto iter = _group_inputs.find(socket);
                iter != _group_inputs.end()) {
                return iter->second;
            }
            auto *output = raw_output(node, socket);
            warn_once(
                "group-input:" + node_name + ":" + socket,
                "node group input '" + socket +
                    "' has no instance binding; using its default");
            return constant_from_socket(
                output,
                node_name + " / " + socket,
                socket_type(output));
        }
        if (type == "GROUP") {
            return lower_group_output(node, socket);
        }

        if (!_building.emplace(node_name).second) {
            warn_once(
                "cycle:" + node_name,
                "recursive node dependency detected at '" +
                    node_name + "'");
            return constant_from_output(
                node, socket, SocketType::color);
        }
        auto finish = [&](TypedOutput output) {
            _building.erase(node_name);
            return output;
        };

        if (type == "RGB") {
            return finish(constant_from_output(
                node, socket, SocketType::color));
        }
        if (type == "VALUE") {
            return finish(constant_from_output(
                node, socket, SocketType::floating));
        }
        if (type == "TEX_COORD" || type == "UVMAP") {
            const auto id = _graph.add_node(
                compiler::node_type::texture_coordinate,
                node_name);
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
            const auto id = _graph.add_node(
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
            const auto id = _graph.add_node(
                compiler::node_type::particle_info,
                node_name);
            const auto output_socket =
                socket == "Index" ? "Index" : "Random";
            return finish({
                .ref = {.node = id, .socket = output_socket},
                .type = SocketType::floating});
        }
        if (type == "LIGHT_PATH") {
            const auto id = _graph.add_node(
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
            const auto id = _graph.add_node(
                compiler::node_type::layer_weight,
                node_name);
            static_cast<void>(bind(
                id,
                "Blend",
                node,
                "Blend",
                SocketType::floating));
            static_cast<void>(bind(
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
        if (type == "NEW_GEOMETRY") {
            if (socket == "Normal" ||
                socket == "True Normal") {
                return finish(geometry_output(
                    socket == "Normal"
                        ? "Normal"
                        : "GeometricNormal",
                    SocketType::normal));
            }
            if (socket == "Position") {
                return finish(geometry_output(
                    "Position", SocketType::point));
            }
            if (socket == "Incoming") {
                return finish(geometry_output(
                    "Incoming", SocketType::vector));
            }
            if (socket == "Backfacing") {
                return finish(geometry_output(
                    "Backfacing", SocketType::floating));
            }
            if (socket == "Random Per Island") {
                return finish(geometry_output(
                    "RandomPerIsland", SocketType::floating));
            }
            warn_once(
                "geometry:" + socket,
                "Geometry output '" + socket +
                    "' is not yet represented; using zero");
            return finish(constant_from_output(
                node, socket, SocketType::floating));
        }
        if (type == "MAPPING") {
            const auto id = _graph.add_node(
                compiler::node_type::mapping, node_name);
            static_cast<void>(bind(
                id, "Vector", node, "Vector", SocketType::vector));
            static_cast<void>(bind(
                id, "Location", node, "Location", SocketType::vector));
            static_cast<void>(bind(
                id, "Rotation", node, "Rotation", SocketType::vector));
            static_cast<void>(bind(
                id, "Scale", node, "Scale", SocketType::vector));
            static_cast<void>(_graph.set_property(
                id,
                "VectorType",
                SocketValue::string(node_property_text(
                    node, "vector_type", "POINT"))));
            return finish({
                .ref = {.node = id, .socket = "Vector"},
                .type = SocketType::vector});
        }
        if (type == "TEX_IMAGE") {
            const auto id = _graph.add_node(
                compiler::node_type::image_texture,
                node_name);
            if (input_source(node, "Vector")) {
                static_cast<void>(bind(
                    id,
                    "Vector",
                    node,
                    "Vector",
                    SocketType::vector));
            } else {
                static_cast<void>(_graph.connect(
                    default_image_coordinates().ref,
                    id,
                    "Vector"));
            }
            const auto image_name =
                text(member(node, "image"));
            const auto image_iter =
                _image_ids.find(image_name);
            if (image_iter == _image_ids.end()) {
                warn_once(
                    "image:" + image_name,
                    "image '" + image_name +
                        "' is unavailable");
            }
            const auto image_id =
                image_iter == _image_ids.end()
                    ? 0u
                    : image_iter->second.value;
            const auto color_iter =
                _image_color_spaces.find(image_name);
            const auto color_space =
                color_iter != _image_color_spaces.end() &&
                        color_iter->second ==
                            ImageColorSpace::srgb
                    ? "sRGB"
                    : "Non-Color";
            static_cast<void>(_graph.set_property(
                id,
                "Image",
                SocketValue::unsigned_integer(image_id)));
            static_cast<void>(_graph.set_property(
                id,
                "Extension",
                SocketValue::string(node_property_text(
                    node, "extension", "REPEAT"))));
            static_cast<void>(_graph.set_property(
                id,
                "ColorSpace",
                SocketValue::string(color_space)));
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
        if (type == "HUE_SAT") {
            const auto id = _graph.add_node(
                compiler::node_type::hue_saturation,
                node_name);
            static_cast<void>(bind(
                id, "Hue", node, "Hue", SocketType::floating));
            static_cast<void>(bind(
                id,
                "Saturation",
                node,
                "Saturation",
                SocketType::floating));
            static_cast<void>(bind(
                id, "Value", node, "Value", SocketType::floating));
            static_cast<void>(bind(
                id, "Factor", node, "Fac", SocketType::floating));
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "INVERT") {
            const auto id = _graph.add_node(
                compiler::node_type::invert_color,
                node_name);
            static_cast<void>(bind(
                id, "Factor", node, "Fac", SocketType::floating));
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "RGBTOBW") {
            const auto id = _graph.add_node(
                compiler::node_type::color_to_scalar,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Value"},
                .type = SocketType::floating});
        }
        if (type == "GAMMA") {
            const auto id = _graph.add_node(
                compiler::node_type::gamma_color,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(bind(
                id, "Gamma", node, "Gamma", SocketType::floating));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "BRIGHTCONTRAST") {
            const auto id = _graph.add_node(
                compiler::node_type::brightness_contrast,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(bind(
                id, "Bright", node, "Bright", SocketType::floating));
            static_cast<void>(bind(
                id,
                "Contrast",
                node,
                "Contrast",
                SocketType::floating));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "MIX") {
            const auto blend_type =
                node_property_text(node, "blend_type", "MIX");
            const auto multiply = blend_type == "MULTIPLY";
            const auto id = _graph.add_node(
                multiply
                    ? compiler::node_type::multiply_color
                    : compiler::node_type::mix_color,
                node_name);
            static_cast<void>(bind(
                id,
                "Factor",
                node,
                "Factor_Float",
                SocketType::floating));
            static_cast<void>(bind(
                id, "A", node, "A_Color", SocketType::color));
            static_cast<void>(bind(
                id, "B", node, "B_Color", SocketType::color));
            if (!multiply) {
                static_cast<void>(_graph.set_property(
                    id,
                    "BlendMode",
                    SocketValue::string(blend_type)));
            }
            if (blend_type != "MIX" &&
                blend_type != "MULTIPLY" &&
                blend_type != "VALUE" &&
                blend_type != "COLOR") {
                warn_once(
                    "mix:" + blend_type,
                    "Mix blend mode '" + blend_type +
                        "' currently uses linear Mix");
            }
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "CLAMP") {
            const auto id = _graph.add_node(
                compiler::node_type::clamp_range,
                node_name);
            static_cast<void>(bind(
                id, "Value", node, "Value", SocketType::floating));
            static_cast<void>(bind(
                id, "Min", node, "Min", SocketType::floating));
            static_cast<void>(bind(
                id, "Max", node, "Max", SocketType::floating));
            static_cast<void>(_graph.set_property(
                id,
                "Mode",
                SocketValue::string(node_property_text(
                    node, "clamp_type", "MINMAX"))));
            return finish({
                .ref = {.node = id, .socket = "Result"},
                .type = SocketType::floating});
        }
        if (type == "MATH") {
            const auto operation =
                node_property_text(node, "operation", "ADD");
            const auto *psycles_type =
                operation == "MULTIPLY"
                    ? compiler::node_type::multiply_float
                    : operation == "SUBTRACT"
                          ? compiler::node_type::subtract_float
                          : operation == "DIVIDE"
                                ? compiler::node_type::divide_float
                                : operation == "MINIMUM"
                                      ? compiler::node_type::minimum_float
                                      : operation == "MAXIMUM"
                                            ? compiler::node_type::maximum_float
                                            : operation == "POWER"
                                                  ? compiler::node_type::power_float
                                                  : compiler::node_type::add_float;
            if (operation != "ADD" &&
                operation != "MULTIPLY" &&
                operation != "SUBTRACT" &&
                operation != "DIVIDE" &&
                operation != "MINIMUM" &&
                operation != "MAXIMUM" &&
                operation != "POWER") {
                warn_once(
                    "math:" + operation,
                    "Math operation '" + operation +
                        "' currently uses Add");
            }
            const auto id =
                _graph.add_node(psycles_type, node_name);
            static_cast<void>(bind(
                id, "A", node, "Value", SocketType::floating));
            static_cast<void>(bind(
                id,
                "B",
                node,
                "Value_001",
                SocketType::floating));
            TypedOutput result{
                .ref = {.node = id, .socket = "Value"},
                .type = SocketType::floating};
            if (node_property_bool(node, "use_clamp")) {
                const auto clamped = _graph.add_node(
                    compiler::node_type::clamp_float,
                    node_name + " Clamp");
                static_cast<void>(_graph.connect(
                    result.ref, clamped, "Value"));
                result.ref = {
                    .node = clamped, .socket = "Value"};
            }
            return finish(result);
        }
        if (type == "VALTORGB") {
            const auto id = _graph.add_node(
                compiler::node_type::color_ramp,
                node_name);
            static_cast<void>(bind(
                id, "Factor", node, "Fac", SocketType::floating));
            auto *ramp = member(
                member(node, "special"), "color_ramp");
            static_cast<void>(_graph.set_property(
                id,
                "Interpolation",
                SocketValue::string(text(
                    member(ramp, "interpolation"),
                    "LINEAR"))));
            static_cast<void>(_graph.set_property(
                id,
                "Table",
                SocketValue::string(color_ramp_table(node))));
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
        if (type == "CURVE_RGB") {
            const auto id = _graph.add_node(
                compiler::node_type::rgb_curve,
                node_name);
            static_cast<void>(bind(
                id, "Factor", node, "Fac", SocketType::floating));
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(_graph.set_property(
                id,
                "Table",
                SocketValue::string(rgb_curve_table(node))));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "NORMAL_MAP") {
            const auto id = _graph.add_node(
                compiler::node_type::normal_map,
                node_name);
            static_cast<void>(bind(
                id,
                "Strength",
                node,
                "Strength",
                SocketType::floating));
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(_graph.set_property(
                id,
                "Space",
                SocketValue::string(node_property_text(
                    node, "space", "TANGENT"))));
            return finish({
                .ref = {.node = id, .socket = "Normal"},
                .type = SocketType::normal});
        }
        if (type == "TEX_NOISE") {
            const auto id = _graph.add_node(
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
                static_cast<void>(bind(
                    id,
                    target,
                    node,
                    source,
                    target_type));
            }
            const auto dimensions =
                node_property_text(
                    node, "noise_dimensions", "3D");
            static_cast<void>(_graph.set_property(
                id,
                "Dimensions",
                SocketValue::unsigned_integer(
                    dimensions == "1D"
                        ? 1u
                        : dimensions == "2D"
                              ? 2u
                              : dimensions == "4D" ? 4u : 3u)));
            static_cast<void>(_graph.set_property(
                id,
                "Normalize",
                SocketValue::boolean(node_property_bool(
                    node, "normalize"))));
            static_cast<void>(_graph.set_property(
                id,
                "NoiseType",
                SocketValue::string(node_property_text(
                    node, "noise_type", "FBM"))));
            static_cast<void>(_graph.set_property(
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
            const auto id = _graph.add_node(
                compiler::node_type::white_noise_texture,
                node_name);
            static_cast<void>(bind(
                id,
                "Vector",
                node,
                "Vector",
                SocketType::vector));
            static_cast<void>(bind(
                id,
                "W",
                node,
                "W",
                SocketType::floating));
            const auto dimensions =
                node_property_text(
                    node, "noise_dimensions", "3D");
            static_cast<void>(_graph.set_property(
                id,
                "Dimensions",
                SocketValue::unsigned_integer(
                    dimensions == "1D"
                        ? 1u
                        : dimensions == "2D"
                              ? 2u
                              : dimensions == "4D" ? 4u : 3u)));
            static_cast<void>(_graph.set_property(
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
        if (type == "TEX_BRICK") {
            const auto id = _graph.add_node(
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
                static_cast<void>(bind(
                    id,
                    target,
                    node,
                    source,
                    target_type));
            }
            static_cast<void>(_graph.set_property(
                id,
                "OffsetAmount",
                SocketValue::floating(node_property_number(
                    node, "offset", 0.5f))));
            static_cast<void>(_graph.set_property(
                id,
                "OffsetFrequency",
                SocketValue::unsigned_integer(
                    static_cast<std::uint64_t>(std::max(
                        node_property_number(
                            node, "offset_frequency", 2.0f),
                        0.0f)))));
            static_cast<void>(_graph.set_property(
                id,
                "SquashAmount",
                SocketValue::floating(node_property_number(
                    node, "squash", 1.0f))));
            static_cast<void>(_graph.set_property(
                id,
                "SquashFrequency",
                SocketValue::unsigned_integer(
                    static_cast<std::uint64_t>(std::max(
                        node_property_number(
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
            const auto id = _graph.add_node(
                compiler::node_type::gradient_texture,
                node_name);
            static_cast<void>(bind(
                id, "Vector", node, "Vector", SocketType::vector));
            static_cast<void>(_graph.set_property(
                id,
                "GradientType",
                SocketValue::string(node_property_text(
                    node, "gradient_type", "LINEAR"))));
            return finish({
                .ref = {.node = id, .socket = "Factor"},
                .type = SocketType::floating});
        }
        if (type == "VERTEX_COLOR") {
            const auto id = _graph.add_node(
                compiler::node_type::vertex_color,
                node_name);
            static_cast<void>(_graph.set_property(
                id,
                "AttributeId",
                SocketValue::unsigned_integer(
                    contract::attribute_id(
                    node_property_text(
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
            const auto id = _graph.add_node(
                compiler::node_type::separate_color,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(_graph.set_property(
                id,
                "Mode",
                SocketValue::string(node_property_text(
                    node, "mode", "RGB"))));
            const auto output =
                socket == "Red"
                    ? "R"
                    : socket == "Green" ? "G" : "B";
            return finish({
                .ref = {.node = id, .socket = output},
                .type = SocketType::floating});
        }
        if (type == "COMBINE_COLOR") {
            const auto mode =
                node_property_text(node, "mode", "RGB");
            const auto id = _graph.add_node(
                compiler::node_type::combine_color,
                node_name);
            static_cast<void>(bind(
                id, "R", node, "Red", SocketType::floating));
            static_cast<void>(bind(
                id, "G", node, "Green", SocketType::floating));
            static_cast<void>(bind(
                id, "B", node, "Blue", SocketType::floating));
            static_cast<void>(_graph.set_property(
                id,
                "Mode",
                SocketValue::string(mode)));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "TEX_SKY") {
            const auto id = _graph.add_node(
                compiler::node_type::nishita_sky,
                node_name);
            constexpr auto pi = 3.14159265358979323846f;
            constexpr auto two_pi = 2.0f * pi;
            auto elevation = std::fmod(
                node_property_number(
                    node, "sun_elevation", 0.7853982f),
                two_pi);
            auto rotation = node_property_number(
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
            static_cast<void>(_graph.set_input(
                id,
                "SunElevation",
                SocketValue::floating(elevation)));
            static_cast<void>(_graph.set_input(
                id,
                "SunRotation",
                SocketValue::floating(rotation)));
            for (const auto &[target, property_name, fallback] : {
                     std::tuple{
                         "SunSize", "sun_size", 0.00918043f},
                     std::tuple{
                         "SunIntensity",
                         "sun_intensity",
                         1.0f},
                     std::tuple{
                         "Altitude", "altitude", 0.0f},
                     std::tuple{
                         "AirDensity", "air_density", 1.0f},
                     std::tuple{
                         "DustDensity", "dust_density", 1.0f},
                     std::tuple{
                         "OzoneDensity",
                         "ozone_density",
                         1.0f}}) {
                static_cast<void>(_graph.set_input(
                    id,
                    target,
                    SocketValue::floating(
                        node_property_number(
                            node,
                            property_name,
                            fallback))));
            }
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "BSDF_PRINCIPLED") {
            const auto id = _graph.add_node(
                compiler::node_type::principled_bsdf,
                node_name);
            static_cast<void>(_graph.set_property(
                id,
                "Distribution",
                SocketValue::string(node_property_text(
                    node, "distribution", "GGX"))));
            static_cast<void>(bind(
                id,
                "BaseColor",
                node,
                "Base Color",
                SocketType::color));
            static_cast<void>(bind(
                id,
                "Metallic",
                node,
                "Metallic",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "Roughness",
                node,
                "Roughness",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "DiffuseRoughness",
                node,
                "Diffuse Roughness",
                SocketType::floating));
            static_cast<void>(bind(
                id, "IOR", node, "IOR", SocketType::floating));
            static_cast<void>(bind(
                id,
                "SpecularIORLevel",
                node,
                "Specular IOR Level",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "SpecularTint",
                node,
                "Specular Tint",
                SocketType::color));
            static_cast<void>(bind(
                id, "Normal", node, "Normal", SocketType::normal));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "BSDF_DIFFUSE") {
            const auto id = _graph.add_node(
                compiler::node_type::diffuse_bsdf,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(bind(
                id,
                "Roughness",
                node,
                "Roughness",
                SocketType::floating));
            static_cast<void>(bind(
                id, "Normal", node, "Normal", SocketType::normal));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "BSDF_GLOSSY") {
            const auto id = _graph.add_node(
                compiler::node_type::glossy_bsdf,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(bind(
                id,
                "Roughness",
                node,
                "Roughness",
                SocketType::floating));
            if (raw_input(node, "Normal") != nullptr) {
                static_cast<void>(bind(
                    id,
                    "Normal",
                    node,
                    "Normal",
                    SocketType::normal));
            } else {
                static_cast<void>(_graph.connect(
                    geometry_output(
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
            const auto id = _graph.add_node(
                compiler::node_type::emission,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(bind(
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
            const auto id = _graph.add_node(
                compiler::node_type::transparent_bsdf,
                node_name);
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }
        if (type == "MIX_SHADER" ||
            type == "ADD_SHADER") {
            const auto id = _graph.add_node(
                type == "MIX_SHADER"
                    ? compiler::node_type::mix_closure
                    : compiler::node_type::add_closure,
                node_name);
            if (type == "MIX_SHADER") {
                static_cast<void>(bind(
                    id,
                    "Factor",
                    node,
                    "Fac",
                    SocketType::floating));
            }
            static_cast<void>(bind(
                id, "A", node, "Shader", SocketType::closure));
            static_cast<void>(bind(
                id,
                "B",
                node,
                "Shader_001",
                SocketType::closure));
            return finish({
                .ref = {.node = id, .socket = "Closure"},
                .type = SocketType::closure});
        }

        auto *output = raw_output(node, socket);
        const auto socket_type =
            text(member(output, "type"));
        auto fallback_type =
            socket_type.find("Float") != std::string::npos
                ? SocketType::floating
                : SocketType::color;
        warn_once(
            "unsupported:" + type,
            "node type '" + type +
                "' is not yet lowered; using its output default");
        return finish(constant_from_output(
            node, socket, fallback_type));
    }

    [[nodiscard]] TypedOutput lower_output(
        const std::string &node,
        const std::string &socket,
        contract::SocketType requested) {
        const auto key =
            RawOutputKey{.node = node, .socket = socket};
        auto iter = _outputs.find(key);
        if (iter == _outputs.end()) {
            iter = _outputs.emplace(
                key,
                lower_natural_output(node, socket))
                       .first;
        }
        return conversion(iter->second, requested);
    }

    [[nodiscard]] TypedOutput lower_group_output(
        yyjson_val *instance,
        const std::string &socket) {
        using contract::SocketType;
        const auto instance_name =
            text(member(instance, "name"));
        const auto group_name =
            text(member(instance, "node_tree"));
        auto *instance_output = raw_output(instance, socket);
        const auto result_type =
            socket_type(instance_output);

        auto group = _node_groups.find(group_name);
        if (group == _node_groups.end()) {
            warn_once(
                "group-missing:" + group_name,
                "node group '" + group_name +
                    "' was not exported; using the output default");
            return constant_from_socket(
                instance_output,
                instance_name + " / " + socket,
                result_type);
        }
        if (!_group_stack.emplace(group_name).second) {
            warn_once(
                "group-recursion:" + group_name,
                "recursive node group '" + group_name +
                    "' is unsupported");
            return constant_from_socket(
                instance_output,
                instance_name + " / " + socket,
                result_type);
        }

        GroupInputMap bindings;
        auto *inputs = member(instance, "inputs");
        if (inputs != nullptr && yyjson_is_arr(inputs)) {
            yyjson_arr_iter iterator =
                yyjson_arr_iter_with(inputs);
            while (auto *input =
                       yyjson_arr_iter_next(&iterator)) {
                const auto identifier =
                    text(member(input, "identifier"));
                if (identifier.empty() ||
                    identifier == "__extend__") {
                    continue;
                }
                const auto type = socket_type(input);
                if (auto source =
                        input_source(instance, identifier)) {
                    bindings.insert_or_assign(
                        identifier,
                        lower_output(
                            source->node,
                            source->socket,
                            type));
                } else {
                    bindings.insert_or_assign(
                        identifier,
                        constant_from_socket(
                            input,
                            instance_name + " / " +
                                identifier,
                            type));
                }
            }
        }

        auto *saved_tree = _tree;
        auto saved_raw_nodes = std::move(_raw_nodes);
        auto saved_links = std::move(_links);
        auto saved_outputs = std::move(_outputs);
        auto saved_group_inputs =
            std::move(_group_inputs);
        auto saved_building = std::move(_building);
        auto restore = [&] {
            _tree = saved_tree;
            _raw_nodes = std::move(saved_raw_nodes);
            _links = std::move(saved_links);
            _outputs = std::move(saved_outputs);
            _group_inputs =
                std::move(saved_group_inputs);
            _building = std::move(saved_building);
            _group_stack.erase(group_name);
        };

        try {
            load_tree_context(group->second);
            _group_inputs = std::move(bindings);

            yyjson_val *active_output = nullptr;
            yyjson_val *fallback_output = nullptr;
            for (const auto &[name, node] : _raw_nodes) {
                static_cast<void>(name);
                if (text(member(node, "type")) !=
                    "GROUP_OUTPUT") {
                    continue;
                }
                fallback_output = node;
                if (node_property_bool(
                        node, "is_active_output")) {
                    active_output = node;
                    break;
                }
            }
            if (active_output == nullptr) {
                active_output = fallback_output;
            }

            TypedOutput result;
            if (active_output == nullptr) {
                warn_once(
                    "group-output:" + group_name,
                    "node group '" + group_name +
                        "' has no Group Output node");
                result = constant_from_socket(
                    instance_output,
                    instance_name + " / " + socket,
                    result_type);
            } else if (
                auto source =
                    input_source(active_output, socket)) {
                result = lower_output(
                    source->node,
                    source->socket,
                    result_type);
            } else {
                auto *group_output =
                    raw_input(active_output, socket);
                warn_once(
                    "group-output-unlinked:" +
                        group_name + ":" + socket,
                    "node group '" + group_name +
                        "' output '" + socket +
                        "' is unlinked; using its default");
                result = constant_from_socket(
                    group_output,
                    group_name + " / " + socket,
                    result_type);
            }
            restore();
            return result;
        } catch (...) {
            restore();
            throw;
        }
    }

public:
    BlenderGraphNormalizer(
        yyjson_val *tree,
        std::string material_name,
        const std::map<std::string, ImageId, std::less<>> &
            image_ids,
        const std::map<
            std::string,
            ImageColorSpace,
            std::less<>> &image_color_spaces,
        const NodeGroupMap &node_groups,
        std::vector<BlenderSceneDiagnostic> &diagnostics)
        : _tree{tree},
          _material_name{std::move(material_name)},
          _image_ids{image_ids},
          _image_color_spaces{image_color_spaces},
          _node_groups{node_groups},
          _diagnostics{diagnostics} {
        load_tree_context(_tree);
    }

    [[nodiscard]] ShaderGraph build() {
        auto *root = member(_tree, "surface_root");
        const auto node = text(member(root, "node"));
        const auto socket = text(member(root, "socket"));
        if (node.empty() || socket.empty()) {
            // Cycles treats an unconnected material surface as an opaque
            // black path terminator while retaining the shading normal pass.
            return diffuse_graph({0.0f, 0.0f, 0.0f}, 0.0f);
        }
        auto output = lower_output(
            node, socket, contract::SocketType::closure);
        _graph.set_root(ShaderDomain::surface, output.ref);
        return std::move(_graph);
    }
};

[[nodiscard]] ShaderGraph normalized_material_graph(
    yyjson_val *material,
    const std::map<std::string, ImageId, std::less<>> &image_ids,
    const std::map<
        std::string,
        ImageColorSpace,
        std::less<>> &image_color_spaces,
    const std::map<
        std::string,
        yyjson_val *,
        std::less<>> &node_groups,
    std::vector<BlenderSceneDiagnostic> &diagnostics) {
    const auto material_name = text(member(material, "name"));
    auto *tree = member(material, "node_tree");
    if (tree == nullptr || yyjson_is_null(tree)) {
        return diffuse_graph({0.0f, 0.0f, 0.0f}, 0.0f);
    }
    const auto connected_root =
        [tree](const char *name) noexcept {
            auto *root = member(tree, name);
            return root != nullptr && !yyjson_is_null(root) &&
                   !text(member(root, "node")).empty() &&
                   !text(member(root, "socket")).empty();
        };
    for (const auto &[root, label] : {
             std::pair{"volume_root", "Volume"},
             std::pair{"displacement_root", "Displacement"}}) {
        if (connected_root(root)) {
            diagnostics.emplace_back(BlenderSceneDiagnostic{
                .severity =
                    BlenderSceneDiagnosticSeverity::error,
                .message =
                    "shader '" + material_name +
                    "' has a connected " + label +
                    " root, which the Luisa integrator does not yet "
                    "implement"});
        }
    }
    if (
        member(tree, "surface_root") == nullptr ||
        yyjson_is_null(member(tree, "surface_root"))) {
        return diffuse_graph({0.0f, 0.0f, 0.0f}, 0.0f);
    }
    return BlenderGraphNormalizer{
        tree,
        material_name,
        image_ids,
        image_color_spaces,
        node_groups,
        diagnostics}
        .build();
}

[[nodiscard]] std::uint64_t section_offset(
    yyjson_val *geometry,
    const char *name) noexcept {
    return unsigned_number(member(member(geometry, name), "offset"));
}

template<typename T>
[[nodiscard]] std::vector<T> read_values(
    std::ifstream &stream,
    std::uint64_t offset,
    std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("binary scene section is too large");
    }
    std::vector<T> result(count);
    stream.clear();
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(
        reinterpret_cast<char *>(result.data()),
        static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) {
        throw std::runtime_error("failed to read geometry.bin section");
    }
    return result;
}

[[nodiscard]] ImageColorSpace image_color_space(
    std::string_view name) noexcept {
    if (name == "sRGB") {
        return ImageColorSpace::srgb;
    }
    if (name == "Linear" || name == "Linear Rec.709" ||
        name == "Linear Rec.2020") {
        return ImageColorSpace::linear;
    }
    return ImageColorSpace::data;
}

[[nodiscard]] std::vector<std::uint8_t> read_file(
    const std::filesystem::path &path) {
    std::ifstream stream{path, std::ios::binary | std::ios::ate};
    if (!stream) {
        throw std::runtime_error(
            "failed to open file: " + path.string());
    }
    const auto end = stream.tellg();
    if (end < 0) {
        throw std::runtime_error(
            "failed to size file: " + path.string());
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(end));
    stream.seekg(0);
    stream.read(
        reinterpret_cast<char *>(result.data()),
        static_cast<std::streamsize>(result.size()));
    if (!stream) {
        throw std::runtime_error(
            "failed to read file: " + path.string());
    }
    return result;
}

[[nodiscard]] std::vector<Vec3f> read_float3_file(
    const std::filesystem::path &path,
    std::uint32_t width,
    std::uint32_t height) {
    const auto count =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);
    if (width == 0u || height == 0u ||
        count >
            std::numeric_limits<std::size_t>::max() /
                (3u * sizeof(float))) {
        throw std::runtime_error(
            "invalid float3 image dimensions: " + path.string());
    }
    const auto bytes = read_file(path);
    const auto expected = count * 3u * sizeof(float);
    if (bytes.size() != expected) {
        throw std::runtime_error(
            "float3 image byte count mismatch: " + path.string());
    }
    std::vector<Vec3f> result(count);
    for (std::size_t i = 0u; i < count; ++i) {
        std::array<float, 3u> value{};
        std::memcpy(
            value.data(),
            bytes.data() + i * 3u * sizeof(float),
            3u * sizeof(float));
        result[i] = {value[0u], value[1u], value[2u]};
    }
    return result;
}

}// namespace

BlenderSceneImport load_blender_scene_bundle(
    const std::filesystem::path &directory) {
    BlenderSceneImport result;
    auto error = [&](std::string message) {
        result.diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity = BlenderSceneDiagnosticSeverity::error,
            .message = std::move(message)});
    };
    auto warning = [&](std::string message) {
        result.diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity = BlenderSceneDiagnosticSeverity::warning,
            .message = std::move(message)});
    };

    const auto json_path = directory / "scene.json";
    yyjson_read_err read_error{};
    Document document{
        yyjson_read_file(
            json_path.string().c_str(),
            0u,
            nullptr,
            &read_error)};
    if (!document) {
        error(
            "failed to parse '" + json_path.string() +
            "': " +
            (read_error.msg == nullptr
                 ? std::string{"unknown JSON error"}
                 : std::string{read_error.msg}));
        return result;
    }
    auto *root = yyjson_doc_get_root(document.get());
    if (text(member(root, "schema")) !=
        "psycles.blender-scene.v1") {
        error("unsupported Blender scene bundle schema");
        return result;
    }

    try {
        SceneSnapshot scene;
        scene.revision = 1u;

        std::map<std::string, ImageId, std::less<>>
            image_ids;
        std::map<
            std::string,
            ImageColorSpace,
            std::less<>>
            image_color_spaces;
        auto *images = member(root, "images");
        yyjson_arr_iter image_metadata_iterator =
            yyjson_arr_iter_with(images);
        std::uint64_t image_metadata_index = 1u;
        while (auto *image = yyjson_arr_iter_next(
                   &image_metadata_iterator)) {
            const auto name = text(member(image, "name"));
            const auto id = ImageId{image_metadata_index++};
            image_ids.emplace(name, id);
            image_color_spaces.emplace(
                name,
                image_color_space(
                    text(member(image, "colorspace"))));
        }

        std::map<
            std::string,
            yyjson_val *,
            std::less<>>
            node_groups;
        auto *raw_node_groups = member(root, "node_groups");
        if (raw_node_groups != nullptr &&
            yyjson_is_arr(raw_node_groups)) {
            yyjson_arr_iter group_iterator =
                yyjson_arr_iter_with(raw_node_groups);
            while (auto *group =
                       yyjson_arr_iter_next(
                           &group_iterator)) {
                if (group == nullptr ||
                    yyjson_is_null(group)) {
                    continue;
                }
                node_groups.insert_or_assign(
                    text(member(group, "name")),
                    group);
            }
        }

        std::map<std::string, MaterialId, std::less<>>
            material_ids;
        const MaterialId default_material{1u};
        scene.materials.emplace(
            default_material,
            MaterialDesc{
                .name = "__psycles_missing_material__",
                .shader = diffuse_graph(
                    {1.0f, 0.0f, 1.0f}, 0.0f)});

        auto *materials = member(root, "materials");
        yyjson_arr_iter material_iterator =
            yyjson_arr_iter_with(materials);
        std::uint64_t material_index = 2u;
        while (auto *material =
                   yyjson_arr_iter_next(&material_iterator)) {
            const auto id = MaterialId{material_index++};
            const auto name = text(member(material, "name"));
            material_ids.emplace(name, id);
            scene.materials.emplace(
                id,
                MaterialDesc{
                    .name = name,
                    .shader = normalized_material_graph(
                        material,
                        image_ids,
                        image_color_spaces,
                        node_groups,
                        result.diagnostics)});
        }

        yyjson_arr_iter image_iterator =
            yyjson_arr_iter_with(images);
        std::uint64_t image_index = 1u;
        while (auto *image =
                   yyjson_arr_iter_next(&image_iterator)) {
            const auto relative_path =
                std::filesystem::path{
                    text(member(image, "path"))};
            const auto source_path =
                directory / relative_path;
            scene.images.emplace(
                ImageId{image_index++},
                ImageDesc{
                    .name = text(member(image, "name")),
                    .source_format =
                        source_path.extension().string(),
                    .color_space = image_color_space(
                        text(member(image, "colorspace"))),
                    .width = static_cast<std::uint32_t>(
                        unsigned_number(
                            member(image, "width"))),
                    .height = static_cast<std::uint32_t>(
                        unsigned_number(
                            member(image, "height"))),
                    .encoded_data = read_file(source_path)});
        }

        auto *render = member(root, "render");
        result.width = static_cast<std::uint32_t>(
            unsigned_number(member(render, "width"), 320u));
        result.height = static_cast<std::uint32_t>(
            unsigned_number(member(render, "height"), 240u));
        const auto percentage = static_cast<std::uint32_t>(
            unsigned_number(member(render, "percentage"), 100u));
        result.width =
            std::max(result.width * percentage / 100u, 1u);
        result.height =
            std::max(result.height * percentage / 100u, 1u);
        result.samples = static_cast<std::uint32_t>(
            unsigned_number(
                member(member(render, "cycles"), "samples"),
                64u));
        auto *cycles = member(render, "cycles");
        result.seed = static_cast<std::uint32_t>(
            unsigned_number(member(cycles, "seed")));
        result.adaptive_sampling = boolean(
            member(cycles, "use_adaptive_sampling"));
        result.denoising =
            boolean(member(cycles, "use_denoising"));
        result.integrator.max_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "max_bounces"),
                result.integrator.max_bounces));
        result.integrator.min_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "min_light_bounces"),
                result.integrator.min_bounces));
        result.integrator.diffuse_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "diffuse_bounces"),
                result.integrator.diffuse_bounces));
        result.integrator.glossy_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "glossy_bounces"),
                result.integrator.glossy_bounces));
        result.integrator.transmission_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "transmission_bounces"),
                result.integrator.transmission_bounces));
        result.integrator.volume_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "volume_bounces"),
                result.integrator.volume_bounces));
        result.integrator.transparent_min_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "min_transparent_bounces"),
                result.integrator.transparent_min_bounces));
        result.integrator.transparent_max_bounces =
            static_cast<std::uint32_t>(unsigned_number(
                member(cycles, "transparent_max_bounces"),
                result.integrator.transparent_max_bounces));
        result.integrator.sample_clamp_direct = std::max(
            number(
                member(cycles, "sample_clamp_direct"),
                result.integrator.sample_clamp_direct),
            0.0f);
        result.integrator.sample_clamp_indirect = std::max(
            number(
                member(cycles, "sample_clamp_indirect"),
                result.integrator.sample_clamp_indirect),
            0.0f);
        result.integrator.film_exposure = std::max(
            number(
                member(cycles, "film_exposure"),
                result.integrator.film_exposure),
            0.0f);
        result.integrator.light_sampling_threshold = std::max(
            number(
                member(cycles, "light_sampling_threshold"),
                result.integrator.light_sampling_threshold),
            0.0f);
        result.integrator.reflective_caustics = boolean(
            member(cycles, "caustics_reflective"),
            result.integrator.reflective_caustics);
        result.integrator.refractive_caustics = boolean(
            member(cycles, "caustics_refractive"),
            result.integrator.refractive_caustics);
        result.integrator.use_light_tree = boolean(
            member(cycles, "use_light_tree"),
            result.integrator.use_light_tree);
        const auto direct_light_sampling = text(
            member(cycles, "direct_light_sampling_type"),
            "MULTIPLE_IMPORTANCE_SAMPLING");
        if (direct_light_sampling ==
            "MULTIPLE_IMPORTANCE_SAMPLING") {
            result.integrator.direct_light_sampling =
                contract::DirectLightSampling::
                    multiple_importance_sampling;
        } else if (
            direct_light_sampling ==
            "FORWARD_PATH_TRACING") {
            result.integrator.direct_light_sampling =
                contract::DirectLightSampling::
                    forward_path_tracing;
        } else if (
            direct_light_sampling ==
            "NEXT_EVENT_ESTIMATION") {
            result.integrator.direct_light_sampling =
                contract::DirectLightSampling::
                    next_event_estimation;
        } else {
            throw std::runtime_error(
                "unsupported Cycles direct light sampling type: " +
                direct_light_sampling);
        }
        if (result.adaptive_sampling) {
            warning(
                "Cycles adaptive sampling is enabled in the source scene; "
                "Psycles currently renders a fixed sample count. Official "
                "differential goldens must disable adaptive sampling.");
        }
        if (result.denoising) {
            warning(
                "Cycles denoising is enabled in the source scene; Psycles "
                "outputs un-denoised linear passes. Official differential "
                "goldens must disable denoising.");
        }
        if (result.integrator.use_light_tree) {
            error(
                "Cycles light-tree sampling is enabled, but the Luisa "
                "integrator does not yet implement light-tree traversal.");
        }
        result.transparent_background =
            boolean(member(render, "transparent"));
        result.filter_width = std::max(
            number(member(render, "filter_width"), 1.0f),
            1.0e-5f);
        const auto pixel_filter_type =
            text(member(render, "pixel_filter_type"));
        if (pixel_filter_type.empty() ||
            pixel_filter_type == "BOX") {
            result.pixel_filter =
                contract::PixelFilter::box;
        } else if (pixel_filter_type == "GAUSSIAN") {
            result.pixel_filter =
                contract::PixelFilter::gaussian;
        } else if (
            pixel_filter_type == "BLACKMAN_HARRIS") {
            result.pixel_filter =
                contract::PixelFilter::blackman_harris;
        } else {
            throw std::runtime_error(
                "unsupported Cycles pixel filter: " +
                pixel_filter_type);
        }

        auto *camera = member(root, "camera");
        if (camera == nullptr || yyjson_is_null(camera)) {
            throw std::runtime_error(
                "bundle contains no active camera");
        }
        const CameraId camera_id{1u};
        const auto camera_type = text(member(camera, "type"));
        const auto sensor_fit_name =
            text(member(camera, "sensor_fit"));
        const auto sensor_fit =
            sensor_fit_name == "HORIZONTAL"
                ? CameraSensorFit::horizontal
                : sensor_fit_name == "VERTICAL"
                      ? CameraSensorFit::vertical
                      : CameraSensorFit::automatic;
        auto *depth_of_field = member(camera, "dof");
        const auto depth_of_field_enabled =
            boolean(member(depth_of_field, "enabled"));
        const auto f_stop = std::max(
            number(member(depth_of_field, "fstop"), 2.8f),
            1.0e-5f);
        const auto aperture_radius =
            depth_of_field_enabled
                ? camera_type == "ORTHO"
                      ? 1.0f / (2.0f * f_stop)
                      : number(member(camera, "lens"), 50.0f) *
                            1.0e-3f / (2.0f * f_stop)
                : 0.0f;
        scene.cameras.emplace(
            camera_id,
            CameraDesc{
                .name = text(member(camera, "name")),
                .projection =
                    camera_type == "ORTHO"
                        ? CameraProjection::orthographic
                        : camera_type == "PANO"
                              ? CameraProjection::panorama
                              : CameraProjection::perspective,
                .transform =
                    matrix(member(camera, "transform")),
                .field_of_view = number(
                    member(camera, "angle_y"),
                    number(member(camera, "angle"), 0.7853982f)),
                .horizontal_field_of_view = number(
                    member(camera, "angle_x"),
                    number(member(camera, "angle"), 0.7853982f)),
                .sensor_fit = sensor_fit,
                .orthographic_scale = number(
                    member(camera, "ortho_scale"), 1.0f),
                .lens_shift_x = number(
                    member(camera, "shift_x"), 0.0f),
                .lens_shift_y = number(
                    member(camera, "shift_y"), 0.0f),
                .near_clip = number(
                    member(camera, "clip_start"), 1.0e-4f),
                .far_clip = number(
                    member(camera, "clip_end"), 1.0e5f),
                .aperture_radius = aperture_radius,
                .focal_distance = number(
                    member(depth_of_field, "focus_distance")),
                .aperture_blades =
                    static_cast<std::uint32_t>(
                        unsigned_number(
                            member(depth_of_field, "blades"))),
                .aperture_rotation = number(
                    member(depth_of_field, "rotation")),
                .aperture_ratio = std::max(
                    number(
                        member(depth_of_field, "ratio"),
                        1.0f),
                    1.0e-5f)});
        scene.active_camera = camera_id;

        std::ifstream geometry_stream{
            directory / "geometry.bin", std::ios::binary};
        if (!geometry_stream) {
            throw std::runtime_error(
                "failed to open geometry.bin");
        }
        std::array<char, 8u> magic{};
        geometry_stream.read(
            magic.data(),
            static_cast<std::streamsize>(magic.size()));
        if (!geometry_stream ||
            std::string_view{magic.data(), magic.size()} !=
                std::string_view{"PSYGEO1\0", 8u}) {
            throw std::runtime_error(
                "geometry.bin has an invalid header");
        }

        auto *geometries = member(root, "geometries");
        yyjson_arr_iter geometry_iterator =
            yyjson_arr_iter_with(geometries);
        std::uint64_t geometry_index = 1u;
        while (auto *geometry =
                   yyjson_arr_iter_next(&geometry_iterator)) {
            const auto vertex_count = static_cast<std::size_t>(
                unsigned_number(
                    member(geometry, "vertex_count")));
            const auto triangle_count = static_cast<std::size_t>(
                unsigned_number(
                    member(geometry, "triangle_count")));
            auto position_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "positions"),
                vertex_count * 3u);
            auto normal_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "normals"),
                vertex_count * 3u);
            auto uv_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "uv"),
                vertex_count * 2u);
            std::vector<float> uv_tangent_values;
            if (member(geometry, "uv_tangents") != nullptr) {
                uv_tangent_values = read_values<float>(
                    geometry_stream,
                    section_offset(geometry, "uv_tangents"),
                    vertex_count * 4u);
            }
            auto generated_values = read_values<float>(
                geometry_stream,
                section_offset(geometry, "generated"),
                vertex_count * 3u);
            auto index_values = read_values<std::uint32_t>(
                geometry_stream,
                section_offset(geometry, "indices"),
                triangle_count * 3u);
            auto material_values =
                read_values<std::uint32_t>(
                    geometry_stream,
                    section_offset(
                        geometry,
                        "triangle_material_slots"),
                    triangle_count);
            auto random_per_island_values =
                read_values<float>(
                    geometry_stream,
                    section_offset(
                        geometry,
                        "triangle_random_per_island"),
                    triangle_count);

            TriangleMeshDesc mesh;
            mesh.name = text(member(geometry, "name"));
            mesh.positions.reserve(vertex_count);
            mesh.normals.reserve(vertex_count);
            mesh.uv.reserve(vertex_count);
            mesh.uv_tangents.reserve(vertex_count);
            mesh.generated.reserve(vertex_count);
            for (std::size_t i = 0u; i < vertex_count; ++i) {
                mesh.positions.emplace_back(Vec3f{
                    position_values[i * 3u],
                    position_values[i * 3u + 1u],
                    position_values[i * 3u + 2u]});
                mesh.normals.emplace_back(Vec3f{
                    normal_values[i * 3u],
                    normal_values[i * 3u + 1u],
                    normal_values[i * 3u + 2u]});
                mesh.uv.emplace_back(Vec2f{
                    uv_values[i * 2u],
                    uv_values[i * 2u + 1u]});
                mesh.uv_tangents.emplace_back(
                    uv_tangent_values.size() ==
                            vertex_count * 4u
                        ? Vec4f{
                              uv_tangent_values[i * 4u],
                              uv_tangent_values[i * 4u + 1u],
                              uv_tangent_values[i * 4u + 2u],
                              uv_tangent_values[i * 4u + 3u]}
                        : Vec4f{});
                mesh.generated.emplace_back(Vec3f{
                    generated_values[i * 3u],
                    generated_values[i * 3u + 1u],
                    generated_values[i * 3u + 2u]});
            }
            auto *color_attributes =
                member(geometry, "color_attributes");
            if (color_attributes != nullptr &&
                yyjson_is_arr(color_attributes)) {
                yyjson_arr_iter attribute_iterator =
                    yyjson_arr_iter_with(color_attributes);
                while (auto *attribute =
                           yyjson_arr_iter_next(
                               &attribute_iterator)) {
                    const auto name =
                        text(member(attribute, "name"));
                    auto values = read_values<float>(
                        geometry_stream,
                        section_offset(attribute, "values"),
                        vertex_count * 4u);
                    auto &destination =
                        mesh.color_attributes[name];
                    destination.reserve(vertex_count);
                    for (std::size_t i = 0u;
                         i < vertex_count;
                         ++i) {
                        destination.emplace_back(Vec4f{
                            values[i * 4u],
                            values[i * 4u + 1u],
                            values[i * 4u + 2u],
                            values[i * 4u + 3u]});
                    }
                }
            }
            mesh.triangles.reserve(triangle_count);
            for (std::size_t i = 0u;
                 i < triangle_count;
                 ++i) {
                mesh.triangles.emplace_back(
                    std::array<std::uint32_t, 3u>{
                        index_values[i * 3u],
                        index_values[i * 3u + 1u],
                        index_values[i * 3u + 2u]});
            }
            mesh.triangle_material_slots =
                std::move(material_values);
            mesh.triangle_random_per_island =
                std::move(random_per_island_values);

            auto *slots = member(geometry, "material_slots");
            yyjson_arr_iter slot_iterator =
                yyjson_arr_iter_with(slots);
            while (auto *slot =
                       yyjson_arr_iter_next(&slot_iterator)) {
                if (yyjson_is_null(slot)) {
                    mesh.material_slots.emplace_back(
                        default_material);
                    continue;
                }
                const auto iter =
                    material_ids.find(text(slot));
                mesh.material_slots.emplace_back(
                    iter == material_ids.end()
                        ? default_material
                        : iter->second);
            }
            if (mesh.material_slots.empty()) {
                mesh.material_slots.emplace_back(
                    default_material);
            }
            scene.geometries.emplace(
                GeometryId{geometry_index++},
                std::move(mesh));
        }

        auto *instances = member(root, "instances");
        yyjson_arr_iter instance_iterator =
            yyjson_arr_iter_with(instances);
        std::uint64_t instance_index = 1u;
        while (auto *instance =
                   yyjson_arr_iter_next(&instance_iterator)) {
            const auto geometry =
                unsigned_number(member(instance, "geometry"));
            auto *visibility = member(instance, "visibility");
            std::uint32_t visibility_mask = 0u;
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
                    visibility_mask |=
                        contract::visibility_bit(bit);
                }
            }
            scene.instances.emplace(
                InstanceId{instance_index++},
                contract::InstanceDesc{
                    .name = text(member(instance, "name")),
                    .geometry = GeometryId{geometry + 1u},
                    .transform =
                        matrix(member(instance, "transform")),
                    .motion = {},
                    .material_overrides = {},
                    .random = std::clamp(
                        number(member(instance, "random_id")) /
                            4294967295.0f,
                        0.0f,
                        1.0f),
                    .particle_index = static_cast<std::uint32_t>(
                        unsigned_number(
                            member(instance, "particle_index"))),
                    .visibility_mask = visibility_mask});
        }

        auto *lights = member(root, "lights");
        yyjson_arr_iter light_iterator =
            yyjson_arr_iter_with(lights);
        std::uint64_t light_index = 1u;
        while (auto *light =
                   yyjson_arr_iter_next(&light_iterator)) {
            const auto type = text(member(light, "type"));
            scene.lights.emplace(
                LightId{light_index++},
                LightDesc{
                    .name = text(member(light, "name")),
                    .type =
                        type == "AREA"
                            ? LightType::area
                            : type == "SUN"
                                  ? LightType::distant
                                  : type == "SPOT"
                                        ? LightType::spot
                                        : LightType::point,
                    .transform =
                        matrix(member(light, "transform")),
                    .color = float3(
                        member(light, "color"),
                        {1.0f, 1.0f, 1.0f}),
                    .power =
                        number(member(light, "energy"), 1.0f),
                    .size =
                        number(member(light, "size"), 0.0f),
                    .spread =
                        number(member(light, "spread"), 3.14159265f),
                    .shader = std::nullopt});
        }

        auto *world = member(root, "world");
        if (world != nullptr && !yyjson_is_null(world)) {
            Vec3f world_color =
                float3(member(world, "color"), {0.05f, 0.05f, 0.05f});
            auto *tree = member(world, "node_tree");
            const auto world_id =
                MaterialId{material_index++};
            auto world_graph =
                tree != nullptr &&
                        !yyjson_is_null(tree) &&
                        member(tree, "surface_root") != nullptr
                    ? normalized_material_graph(
                          world,
                          image_ids,
                          image_color_spaces,
                          node_groups,
                          result.diagnostics)
                    : emission_graph(world_color, 1.0f);
            scene.materials.emplace(
                world_id,
                MaterialDesc{
                    .name =
                        "__world__" + text(member(world, "name")),
                    .shader = std::move(world_graph)});
            scene.world_shader = world_id;
        }

        auto *environment = member(root, "world_environment");
        if (environment != nullptr &&
            !yyjson_is_null(environment)) {
            const auto width = static_cast<std::uint32_t>(
                unsigned_number(member(environment, "width")));
            const auto height = static_cast<std::uint32_t>(
                unsigned_number(member(environment, "height")));
            EnvironmentDesc imported_environment{
                .name = text(
                    member(environment, "name"),
                    "__world_environment__"),
                .width = width,
                .height = height,
                .pixels = read_float3_file(
                    directory /
                        std::filesystem::path{
                            text(member(environment, "path"))},
                    width,
                    height),
                .suns = {}};
            auto *suns = member(environment, "suns");
            yyjson_arr_iter sun_iterator =
                yyjson_arr_iter_with(suns);
            while (auto *sun =
                       yyjson_arr_iter_next(&sun_iterator)) {
                imported_environment.suns.emplace_back(
                    EnvironmentSunDesc{
                        .direction = float3(
                            member(sun, "direction"),
                            {0.0f, 0.0f, 1.0f}),
                        .radiance = float3(
                            member(sun, "radiance")),
                        .angular_radius = number(
                            member(sun, "angular_radius"))});
            }
            scene.environment =
                std::move(imported_environment);
        }

        result.scene = std::move(scene);
    } catch (const std::exception &exception) {
        error(exception.what());
    }
    return result;
}

}// namespace psycles::adapter
