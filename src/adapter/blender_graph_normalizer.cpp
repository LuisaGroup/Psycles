#include "blender_graph_lowering_component.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>

namespace psycles::adapter::detail {

using contract::ImageAlphaType;
using contract::ImageColorSpace;
using contract::ImageId;
using contract::ShaderDomain;
using contract::ShaderGraph;
using contract::SocketValue;

struct LoweredOutputKey {
    RawOutputKey raw;
    contract::SocketType requested{
        contract::SocketType::floating};

    auto operator<=>(const LoweredOutputKey &) const noexcept = default;
};

class BlenderGraphNormalizer final
    : public BlenderNodeLoweringContext {

private:
    using RawNodeMap =
        std::map<std::string, yyjson_val *, std::less<>>;
    using RawLinkMap =
        std::map<RawOutputKey, RawOutputKey>;
    using LoweredOutputMap =
        std::map<LoweredOutputKey, TypedOutput>;
    using GroupInputMap =
        std::map<std::string, TypedOutput, std::less<>>;
    using NodeGroupMap =
        std::map<std::string, yyjson_val *, std::less<>>;

    yyjson_val *_tree{};
    std::string _material_name;
    const std::map<std::string, ImageId, std::less<>> &_image_ids;
    const std::map<std::string, ImageColorSpace, std::less<>> &
        _image_color_spaces;
    const std::map<std::string, ImageAlphaType, std::less<>> &
        _image_alpha_types;
    const NodeGroupMap &_node_groups;
    std::vector<BlenderSceneDiagnostic> &_diagnostics;
    ShaderGraph _graph;
    std::vector<
        std::unique_ptr<BlenderNodeLoweringComponent>>
        _lowering_components;
    RawNodeMap _raw_nodes;
    RawLinkMap _links;
    LoweredOutputMap _outputs;
    GroupInputMap _group_inputs;
    std::set<std::string, std::less<>> _building;
    std::set<std::string, std::less<>> _group_stack;
    std::set<std::string, std::less<>> _warned;
    std::optional<contract::NodeId> _default_image_coordinates;
    std::optional<contract::NodeId> _default_generated_coordinates;
    std::optional<contract::NodeId> _geometry;
    bool _automatic_bump_from_displacement{};

private:
    [[nodiscard]] ShaderGraph &graph() noexcept override {
        return _graph;
    }

    [[nodiscard]] const ImageIdMap &image_ids()
        const noexcept override {
        return _image_ids;
    }

    [[nodiscard]] const ImageColorSpaceMap &image_color_spaces()
        const noexcept override {
        return _image_color_spaces;
    }

    [[nodiscard]] const ImageAlphaTypeMap &image_alpha_types()
        const noexcept override {
        return _image_alpha_types;
    }

    void warn_once(
        std::string key,
        std::string message) override {
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
        std::string_view identifier) const noexcept override {
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
        std::string_view identifier) const override {
        const auto key = RawOutputKey{
            .node = text(member(node, "name")),
            .socket = std::string{identifier}};
        auto iter = _links.find(key);
        return iter == _links.end()
                   ? std::nullopt
                   : std::optional<RawOutputKey>{iter->second};
    }

    [[nodiscard]] bool output_is_linked(
        yyjson_val *node,
        std::string_view identifier) const override {
        const auto key = RawOutputKey{
            .node = text(member(node, "name")),
            .socket = std::string{identifier}};
        return std::any_of(
            _links.begin(),
            _links.end(),
            [&key](const auto &link) {
                return link.second == key;
            });
    }

    [[nodiscard]] std::string node_property_text(
        yyjson_val *node,
        const char *name,
        std::string fallback = {}) const override {
        return text(
            member(member(node, "properties"), name),
            std::move(fallback));
    }

    [[nodiscard]] float node_property_number(
        yyjson_val *node,
        const char *name,
        float fallback = 0.0f) const noexcept override {
        return number(
            member(member(node, "properties"), name),
            fallback);
    }

    [[nodiscard]] bool node_property_bool(
        yyjson_val *node,
        const char *name,
        bool fallback = false) const noexcept override {
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

    [[nodiscard]] TypedOutput
    default_image_coordinates() override {
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

    [[nodiscard]] TypedOutput
    default_generated_coordinates() override {
        if (!_default_generated_coordinates) {
            _default_generated_coordinates =
                _graph.add_node(
                    compiler::node_type::texture_coordinate,
                    "Default Generated Coordinates");
        }
        return {
            .ref = {
                .node = *_default_generated_coordinates,
                .socket = "Generated"},
            .type = contract::SocketType::vector};
    }

    [[nodiscard]] TypedOutput geometry_output(
        std::string socket,
        contract::SocketType type) override {
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
        contract::SocketType target) override {
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
            if (target == SocketType::closure) {
                return null_closure("Invalid Surface Conversion");
            }
            if (target == SocketType::volume_closure) {
                return null_volume("Invalid Volume Conversion");
            }
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

    [[nodiscard]] TypedOutput null_closure(std::string label) {
        const auto id = _graph.add_node(
            compiler::node_type::null_closure,
            std::move(label));
        return {
            .ref = {.node = id, .socket = "Closure"},
            .type = contract::SocketType::closure};
    }

    [[nodiscard]] TypedOutput null_volume(std::string label) {
        const auto id = _graph.add_node(
            compiler::node_type::null_volume,
            std::move(label));
        return {
            .ref = {.node = id, .socket = "Volume"},
            .type = contract::SocketType::volume_closure};
    }

    [[nodiscard]] bool bind(
        contract::NodeId destination,
        std::string target_socket,
        yyjson_val *raw_destination,
        std::string_view raw_input_name,
        contract::SocketType target_type) override {
        if (auto source =
                input_source(
                    raw_destination, raw_input_name)) {
            auto *raw_source = raw_node(source->node);
            if (raw_source != nullptr &&
                text(member(raw_source, "type")) == "MIX" &&
                node_property_text(
                    raw_source,
                    "data_type",
                    "FLOAT") == "ROTATION") {
                // Blender 4.5.10's Cycles adapter creates a MixFloatNode
                // for ROTATION but does not map A_Rotation,
                // B_Rotation, or Result_Rotation. The link is therefore
                // absent from the Cycles graph and the destination keeps
                // its own socket default.
                warn_once(
                    "mix-data:ROTATION",
                    "Cycles 4.5.10 does not map Mix ROTATION sockets; "
                    "using the destination socket default");
                auto *socket =
                    raw_input(raw_destination, raw_input_name);
                return _graph.set_input(
                    destination,
                    std::move(target_socket),
                    literal(
                        member(socket, "default"),
                        target_type));
            }
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
        if (target_type == contract::SocketType::closure) {
            auto source = null_closure(
                text(member(raw_destination, "name")) +
                " Empty Closure");
            return _graph.connect(
                source.ref,
                destination,
                std::move(target_socket));
        }
        if (
            target_type ==
            contract::SocketType::volume_closure) {
            auto source = null_volume(
                text(member(raw_destination, "name")) +
                " Empty Volume");
            return _graph.connect(
                source.ref,
                destination,
                std::move(target_socket));
        }
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
        yyjson_val *node) const override {
        auto *ramp = member(
            member(node, "special"), "color_ramp");
        auto *samples = member(ramp, "samples");
        if (samples != nullptr && yyjson_is_arr(samples) &&
            yyjson_arr_size(samples) >= 2u) {
            std::ostringstream stream;
            stream << std::setprecision(9);
            const auto count = yyjson_arr_size(samples);
            for (std::size_t i = 0u; i < count; ++i) {
                auto *color = yyjson_arr_get(samples, i);
                if (i != 0u) {
                    stream << ';';
                }
                stream
                    << static_cast<double>(i) /
                           static_cast<double>(count - 1u)
                    << ',' << number(yyjson_arr_get(color, 0u))
                    << ',' << number(yyjson_arr_get(color, 1u))
                    << ',' << number(yyjson_arr_get(color, 2u))
                    << ',' << number(
                           yyjson_arr_get(color, 3u), 1.0f);
            }
            return stream.str();
        }
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
        yyjson_val *node) const override {
        auto *mapping = member(
            member(node, "special"), "curve_mapping");
        auto *samples = member(mapping, "samples");
        if (
            samples != nullptr &&
            yyjson_is_arr(samples) &&
            yyjson_arr_size(samples) >= 2u) {
            std::ostringstream stream;
            stream << std::setprecision(9);
            const auto count = yyjson_arr_size(samples);
            for (std::size_t i = 0u; i < count; ++i) {
                auto *sample = yyjson_arr_get(samples, i);
                if (i != 0u) {
                    stream << ';';
                }
                stream
                    << static_cast<double>(i) /
                           static_cast<double>(count - 1u)
                    << ',' << number(yyjson_arr_get(sample, 0u))
                    << ',' << number(yyjson_arr_get(sample, 1u))
                    << ',' << number(yyjson_arr_get(sample, 2u));
            }
            return stream.str();
        }
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
            return null_closure(std::move(label));
        }
        if (type == SocketType::volume_closure) {
            return null_volume(std::move(label));
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
        contract::SocketType type) override {
        return constant_from_socket(
            raw_output(node, socket),
            text(member(node, "name")),
            type);
    }

    [[nodiscard]] TypedOutput lower_natural_output(
        const std::string &node_name,
        const std::string &socket,
        contract::SocketType requested) {
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
                return lower_output(
                    source->node,
                    source->socket,
                    requested);
            }
            return constant_from_output(
                node, socket, requested);
        }
        if (type == "DISPLACEMENT") {
            // Cycles' scalar Displacement node produces
            // normal * (height - midlevel) * scale. The automatic bump
            // lowering below projects this vector back onto the geometric
            // normal before evaluating the height derivatives.
            const auto offset = _graph.add_node(
                compiler::node_type::subtract_float,
                node_name + " / Height Offset");
            static_cast<void>(bind(
                offset,
                "A",
                node,
                "Height",
                SocketType::floating));
            static_cast<void>(bind(
                offset,
                "B",
                node,
                "Midlevel",
                SocketType::floating));

            const auto scaled_height = _graph.add_node(
                compiler::node_type::multiply_float,
                node_name + " / Scaled Height");
            static_cast<void>(_graph.connect(
                {.node = offset, .socket = "Value"},
                scaled_height,
                "A"));
            static_cast<void>(bind(
                scaled_height,
                "B",
                node,
                "Scale",
                SocketType::floating));

            TypedOutput normal;
            if (auto source = input_source(node, "Normal")) {
                normal = lower_output(
                    source->node,
                    source->socket,
                    SocketType::normal);
            } else {
                normal = geometry_output(
                    "Normal", SocketType::normal);
            }
            const auto normal_vector = conversion(
                normal, SocketType::vector);
            const auto displacement = _graph.add_node(
                compiler::node_type::vector_math,
                node_name);
            static_cast<void>(_graph.connect(
                normal_vector.ref, displacement, "A"));
            static_cast<void>(_graph.connect(
                {.node = scaled_height, .socket = "Value"},
                displacement,
                "Scale"));
            static_cast<void>(_graph.set_property(
                displacement,
                "Operation",
                SocketValue::string("SCALE")));
            return {
                .ref = {
                    .node = displacement,
                    .socket = "Vector"},
                .type = SocketType::vector};
        }
        if (type == "BUMP") {
            const auto id = _graph.add_node(
                compiler::node_type::bump,
                node_name);
            static_cast<void>(_graph.set_property(
                id,
                "NormalLinked",
                SocketValue::boolean(
                    input_source(node, "Normal").has_value())));
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
            return lower_group_output(
                node, socket, requested);
        }

        if (!_building.emplace(node_name).second) {
            warn_once(
                "cycle:" + node_name,
                "recursive node dependency detected at '" +
                    node_name + "'");
            return constant_from_output(
                node, socket, SocketType::color);
        }
        const std::function<TypedOutput(TypedOutput)> finish =
            [&](TypedOutput output) {
                _building.erase(node_name);
                return output;
            };

        const auto request = BlenderNodeLoweringRequest{
            .node_name = node_name,
            .socket = socket,
            .requested = requested,
            .node = node,
            .type = type,
            .finish = finish};
        for (const auto &component :
             _lowering_components) {
            if (auto output =
                    component->lower(*this, request)) {
                return *std::move(output);
            }
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

    void lower_automatic_bump() {
        using contract::SocketType;
        auto *root = member(_tree, "displacement_root");
        const auto node = text(member(root, "node"));
        const auto socket = text(member(root, "socket"));
        if (!_automatic_bump_from_displacement ||
            node.empty() || socket.empty()) {
            return;
        }

        const auto displacement = lower_output(
            node, socket, SocketType::vector);
        const auto normal = geometry_output(
            "Normal", SocketType::normal);
        const auto normal_vector = conversion(
            normal, SocketType::vector);
        const auto dot = _graph.add_node(
            compiler::node_type::vector_math,
            "Automatic Bump Displacement Height");
        static_cast<void>(_graph.connect(
            displacement.ref, dot, "A"));
        static_cast<void>(_graph.connect(
            normal_vector.ref, dot, "B"));
        static_cast<void>(_graph.set_property(
            dot,
            "Operation",
            SocketValue::string("DOT_PRODUCT")));

        const auto bump = _graph.add_node(
            compiler::node_type::bump,
            "Automatic Bump from Displacement");
        static_cast<void>(_graph.connect(
            {.node = dot, .socket = "Value"},
            bump,
            "Height"));
        static_cast<void>(_graph.set_input(
            bump,
            "Strength",
            SocketValue::floating(1.0f)));
        static_cast<void>(_graph.set_input(
            bump,
            "Distance",
            SocketValue::floating(1.0f)));
        static_cast<void>(_graph.set_input(
            bump,
            "FilterWidth",
            SocketValue::floating(0.1f)));
        static_cast<void>(_graph.connect(
            normal.ref, bump, "Normal"));
        static_cast<void>(_graph.set_property(
            bump,
            "Invert",
            SocketValue::boolean(false)));
        static_cast<void>(_graph.set_property(
            bump,
            "NormalLinked",
            SocketValue::boolean(true)));

        const auto bump_vector = conversion(
            TypedOutput{
                .ref = {.node = bump, .socket = "Normal"},
                .type = SocketType::normal},
            SocketType::vector);
        _graph.set_root(
            ShaderDomain::displacement, bump_vector.ref);
    }

    [[nodiscard]] TypedOutput lower_output(
        const std::string &node,
        const std::string &socket,
        contract::SocketType requested) {
        const auto key = LoweredOutputKey{
            .raw = {.node = node, .socket = socket},
            .requested = requested};
        auto iter = _outputs.find(key);
        if (iter == _outputs.end()) {
            iter = _outputs.emplace(
                key,
                lower_natural_output(
                    node, socket, requested))
                       .first;
        }
        return conversion(iter->second, requested);
    }

    [[nodiscard]] TypedOutput lower_group_output(
        yyjson_val *instance,
        const std::string &socket,
        contract::SocketType requested) {
        using contract::SocketType;
        const auto instance_name =
            text(member(instance, "name"));
        const auto group_name =
            text(member(instance, "node_tree"));
        auto *instance_output = raw_output(instance, socket);
        const auto natural_result_type =
            socket_type(instance_output);
        const auto result_type =
            natural_result_type == SocketType::closure &&
                    requested == SocketType::volume_closure
                ? SocketType::volume_closure
                : natural_result_type;

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
                const auto natural_type = socket_type(input);
                const auto type =
                    natural_type == SocketType::closure &&
                            requested ==
                                SocketType::volume_closure
                        ? SocketType::volume_closure
                        : natural_type;
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
        const std::map<
            std::string,
            ImageAlphaType,
            std::less<>> &image_alpha_types,
        const NodeGroupMap &node_groups,
        std::vector<BlenderSceneDiagnostic> &diagnostics,
        bool automatic_bump_from_displacement)
        : _tree{tree},
          _material_name{std::move(material_name)},
          _image_ids{image_ids},
          _image_color_spaces{image_color_spaces},
          _image_alpha_types{image_alpha_types},
          _node_groups{node_groups},
          _diagnostics{diagnostics},
          _lowering_components{
              make_blender_node_lowering_components()},
          _automatic_bump_from_displacement{
              automatic_bump_from_displacement} {
        load_tree_context(_tree);
    }

    [[nodiscard]] ShaderGraph build() {
        // Cycles installs this SetNormal-like side effect before lowering the
        // surface graph. Keeping the bump as the displacement root preserves
        // it in Psycles' evaluation order without changing closure sockets.
        lower_automatic_bump();

        auto *root = member(_tree, "surface_root");
        const auto node = text(member(root, "node"));
        const auto socket = text(member(root, "socket"));
        if (node.empty() || socket.empty()) {
            // Cycles treats an unconnected material surface as an opaque
            // black path terminator while retaining the shading normal pass.
            const auto closure = _graph.add_node(
                compiler::node_type::diffuse_bsdf,
                "Unconnected Surface");
            static_cast<void>(_graph.set_input(
                closure,
                "Color",
                SocketValue::color({0.0f, 0.0f, 0.0f})));
            static_cast<void>(_graph.set_input(
                closure,
                "Roughness",
                SocketValue::floating(0.0f)));
            static_cast<void>(_graph.set_input(
                closure,
                "Normal",
                SocketValue::normal({0.0f, 0.0f, 0.0f})));
            _graph.set_root(
                ShaderDomain::surface,
                contract::OutputRef{
                    closure, "Closure"});
        } else {
            auto output = lower_output(
                node, socket, contract::SocketType::closure);
            _graph.set_root(
                ShaderDomain::surface, output.ref);
        }

        auto *raw_volume_root =
            member(_tree, "volume_root");
        const auto volume_node =
            text(member(raw_volume_root, "node"));
        const auto volume_socket =
            text(member(raw_volume_root, "socket"));
        if (!volume_node.empty() &&
            !volume_socket.empty()) {
            auto output = lower_output(
                volume_node,
                volume_socket,
                contract::SocketType::volume_closure);
            _graph.set_root(
                ShaderDomain::volume, output.ref);
        }
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
        ImageAlphaType,
        std::less<>> &image_alpha_types,
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
    const auto has_displacement =
        connected_root("displacement_root");
    const auto displacement_method = text(
        member(material, "displacement_method"), "BUMP");
    const auto automatic_bump =
        has_displacement && displacement_method == "BUMP";
    if (has_displacement && !automatic_bump) {
        diagnostics.emplace_back(BlenderSceneDiagnostic{
            .severity =
                BlenderSceneDiagnosticSeverity::error,
            .message =
                "shader '" + material_name +
                "' requests Blender displacement method '" +
                displacement_method +
                "'; true geometry displacement is not yet "
                "implemented"});
    }
    return BlenderGraphNormalizer{
        tree,
        material_name,
        image_ids,
        image_color_spaces,
        image_alpha_types,
        node_groups,
        diagnostics,
        automatic_bump}
        .build();
}

}// namespace psycles::adapter::detail
