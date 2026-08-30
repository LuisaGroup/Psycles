#include "blender_graph_lowering_component.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace psycles::adapter::detail {
namespace {

class ValueNodeLoweringComponent final
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
        if (type == "HUE_SAT") {
            const auto id = context.graph().add_node(
                compiler::node_type::hue_saturation,
                node_name);
            static_cast<void>(context.bind(
                id, "Hue", node, "Hue", SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "Saturation",
                node,
                "Saturation",
                SocketType::floating));
            static_cast<void>(context.bind(
                id, "Value", node, "Value", SocketType::floating));
            static_cast<void>(context.bind(
                id, "Factor", node, "Fac", SocketType::floating));
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "INVERT") {
            const auto id = context.graph().add_node(
                compiler::node_type::invert_color,
                node_name);
            static_cast<void>(context.bind(
                id, "Factor", node, "Fac", SocketType::floating));
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "RGBTOBW") {
            const auto id = context.graph().add_node(
                compiler::node_type::color_to_scalar,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            return finish({
                .ref = {.node = id, .socket = "Value"},
                .type = SocketType::floating});
        }
        if (type == "GAMMA") {
            const auto id = context.graph().add_node(
                compiler::node_type::gamma_color,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.bind(
                id, "Gamma", node, "Gamma", SocketType::floating));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "BRIGHTCONTRAST") {
            const auto id = context.graph().add_node(
                compiler::node_type::brightness_contrast,
                node_name);
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.bind(
                id, "Bright", node, "Bright", SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "Contrast",
                node,
                "Contrast",
                SocketType::floating));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "BLACKBODY" || type == "WAVELENGTH") {
            const auto is_blackbody = type == "BLACKBODY";
            const auto input_name =
                is_blackbody ? "Temperature" : "Wavelength";
            const auto id = context.graph().add_node(
                is_blackbody
                    ? compiler::node_type::blackbody
                    : compiler::node_type::wavelength,
                node_name);
            static_cast<void>(context.bind(
                id,
                input_name,
                node,
                input_name,
                SocketType::floating));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "MIX" || type == "MIX_RGB") {
            const auto blend_type =
                context.node_property_text(node, "blend_type", "MIX");
            constexpr std::array supported_blends{
                "MIX",
                "DARKEN",
                "MULTIPLY",
                "BURN",
                "LIGHTEN",
                "SCREEN",
                "DODGE",
                "ADD",
                "OVERLAY",
                "SOFT_LIGHT",
                "LINEAR_LIGHT",
                "DIFFERENCE",
                "EXCLUSION",
                "SUBTRACT",
                "DIVIDE",
                "HUE",
                "SATURATION",
                "COLOR",
                "VALUE"};
            if (std::find(
                    supported_blends.begin(),
                    supported_blends.end(),
                    blend_type) == supported_blends.end()) {
                context.warn_once(
                    "mix:" + blend_type,
                    "Mix blend mode '" + blend_type +
                        "' currently uses linear Mix");
            }
            if (type == "MIX_RGB") {
                const auto id = context.graph().add_node(
                    compiler::node_type::legacy_mix_color,
                    node_name);
                static_cast<void>(context.bind(
                    id,
                    "Factor",
                    node,
                    "Fac",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "A",
                    node,
                    "Color1",
                    SocketType::color));
                static_cast<void>(context.bind(
                    id,
                    "B",
                    node,
                    "Color2",
                    SocketType::color));
                static_cast<void>(context.graph().set_property(
                    id,
                    "BlendMode",
                    SocketValue::string(blend_type)));
                // Cycles' legacy MixNode always clamps the factor and has no
                // clamp-factor socket or property. Blender's use_alpha option
                // is not consumed by Cycles.
                static_cast<void>(context.graph().set_property(
                    id,
                    "ClampResult",
                    SocketValue::boolean(context.node_property_bool(
                        node, "use_clamp"))));
                return finish({
                    .ref = {.node = id, .socket = "Color"},
                    .type = SocketType::color});
            }

            const auto data_type =
                context.node_property_text(node, "data_type", "FLOAT");
            const auto clamp_factor =
                context.node_property_bool(node, "clamp_factor", true);
            if (data_type == "FLOAT") {
                const auto id = context.graph().add_node(
                    compiler::node_type::mix_float,
                    node_name);
                static_cast<void>(context.bind(
                    id,
                    "Factor",
                    node,
                    "Factor_Float",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "A",
                    node,
                    "A_Float",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "B",
                    node,
                    "B_Float",
                    SocketType::floating));
                static_cast<void>(context.graph().set_property(
                    id,
                    "ClampFactor",
                    SocketValue::boolean(clamp_factor)));
                return finish({
                    .ref = {.node = id, .socket = "Value"},
                    .type = SocketType::floating});
            }
            if (data_type == "VECTOR") {
                const auto nonuniform =
                    context.node_property_text(
                        node,
                        "factor_mode",
                        "UNIFORM") == "NON_UNIFORM";
                const auto id = context.graph().add_node(
                    nonuniform
                        ? compiler::node_type::
                              mix_vector_nonuniform
                        : compiler::node_type::mix_vector,
                    node_name);
                static_cast<void>(context.bind(
                    id,
                    "Factor",
                    node,
                    nonuniform
                        ? "Factor_Vector"
                        : "Factor_Float",
                    nonuniform
                        ? SocketType::vector
                        : SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "A",
                    node,
                    "A_Vector",
                    SocketType::vector));
                static_cast<void>(context.bind(
                    id,
                    "B",
                    node,
                    "B_Vector",
                    SocketType::vector));
                static_cast<void>(context.graph().set_property(
                    id,
                    "ClampFactor",
                    SocketValue::boolean(clamp_factor)));
                return finish({
                    .ref = {.node = id, .socket = "Vector"},
                    .type = SocketType::vector});
            }
            if (data_type == "RGBA") {
                const auto id = context.graph().add_node(
                    compiler::node_type::mix_color,
                    node_name);
                static_cast<void>(context.bind(
                    id,
                    "Factor",
                    node,
                    "Factor_Float",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "A",
                    node,
                    "A_Color",
                    SocketType::color));
                static_cast<void>(context.bind(
                    id,
                    "B",
                    node,
                    "B_Color",
                    SocketType::color));
                static_cast<void>(context.graph().set_property(
                    id,
                    "BlendMode",
                    SocketValue::string(blend_type)));
                static_cast<void>(context.graph().set_property(
                    id,
                    "ClampFactor",
                    SocketValue::boolean(clamp_factor)));
                static_cast<void>(context.graph().set_property(
                    id,
                    "ClampResult",
                    SocketValue::boolean(context.node_property_bool(
                        node, "clamp_result"))));
                return finish({
                    .ref = {.node = id, .socket = "Color"},
                    .type = SocketType::color});
            }
            context.warn_once(
                "mix-data:" + data_type,
                "Mix data type '" + data_type +
                    "' is not exposed by the Cycles device adapter");
            return finish(context.constant_from_output(
                node,
                socket,
                SocketType::vector));
        }
        if (type == "CLAMP") {
            const auto id = context.graph().add_node(
                compiler::node_type::clamp_range,
                node_name);
            static_cast<void>(context.bind(
                id, "Value", node, "Value", SocketType::floating));
            static_cast<void>(context.bind(
                id, "Min", node, "Min", SocketType::floating));
            static_cast<void>(context.bind(
                id, "Max", node, "Max", SocketType::floating));
            static_cast<void>(context.graph().set_property(
                id,
                "Mode",
                SocketValue::string(context.node_property_text(
                    node, "clamp_type", "MINMAX"))));
            return finish({
                .ref = {.node = id, .socket = "Result"},
                .type = SocketType::floating});
        }
        if (type == "MAP_RANGE") {
            const auto data_type =
                context.node_property_text(node, "data_type", "FLOAT");
            const auto vector_mode =
                data_type == "FLOAT_VECTOR";
            const auto id = context.graph().add_node(
                compiler::node_type::map_range,
                node_name);
            if (vector_mode) {
                static_cast<void>(context.bind(
                    id,
                    "Vector",
                    node,
                    "Vector",
                    SocketType::vector));
                static_cast<void>(context.bind(
                    id,
                    "FromMinVector",
                    node,
                    "From_Min_FLOAT3",
                    SocketType::vector));
                static_cast<void>(context.bind(
                    id,
                    "FromMaxVector",
                    node,
                    "From_Max_FLOAT3",
                    SocketType::vector));
                static_cast<void>(context.bind(
                    id,
                    "ToMinVector",
                    node,
                    "To_Min_FLOAT3",
                    SocketType::vector));
                static_cast<void>(context.bind(
                    id,
                    "ToMaxVector",
                    node,
                    "To_Max_FLOAT3",
                    SocketType::vector));
                static_cast<void>(context.bind(
                    id,
                    "StepsVector",
                    node,
                    "Steps_FLOAT3",
                    SocketType::vector));
            } else {
                static_cast<void>(context.bind(
                    id,
                    "Value",
                    node,
                    "Value",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "FromMin",
                    node,
                    "From Min",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "FromMax",
                    node,
                    "From Max",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "ToMin",
                    node,
                    "To Min",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "ToMax",
                    node,
                    "To Max",
                    SocketType::floating));
                static_cast<void>(context.bind(
                    id,
                    "Steps",
                    node,
                    "Steps",
                    SocketType::floating));
            }
            static_cast<void>(context.graph().set_property(
                id,
                "DataType",
                SocketValue::string(data_type)));
            static_cast<void>(context.graph().set_property(
                id,
                "Interpolation",
                SocketValue::string(context.node_property_text(
                    node,
                    "interpolation_type",
                    "LINEAR"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Clamp",
                SocketValue::boolean(context.node_property_bool(
                    node, "clamp", true))));
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        vector_mode ? "Vector" : "Result"},
                .type =
                    vector_mode
                        ? SocketType::vector
                        : SocketType::floating});
        }
        if (type == "VECT_MATH") {
            const auto id = context.graph().add_node(
                compiler::node_type::vector_math,
                node_name);
            static_cast<void>(context.bind(
                id,
                "A",
                node,
                "Vector",
                SocketType::vector));
            static_cast<void>(context.bind(
                id,
                "B",
                node,
                "Vector_001",
                SocketType::vector));
            static_cast<void>(context.bind(
                id,
                "C",
                node,
                "Vector_002",
                SocketType::vector));
            static_cast<void>(context.bind(
                id,
                "Scale",
                node,
                "Scale",
                SocketType::floating));
            static_cast<void>(context.graph().set_property(
                id,
                "Operation",
                SocketValue::string(context.node_property_text(
                    node, "operation", "ADD"))));
            const auto scalar_output = socket == "Value";
            return finish({
                .ref = {
                    .node = id,
                    .socket =
                        scalar_output ? "Value" : "Vector"},
                .type =
                    scalar_output
                        ? SocketType::floating
                        : SocketType::vector});
        }
        if (type == "VECTOR_ROTATE") {
            const auto id = context.graph().add_node(
                compiler::node_type::vector_rotate,
                node_name);
            static_cast<void>(context.bind(
                id, "Vector", node, "Vector", SocketType::vector));
            static_cast<void>(context.bind(
                id, "Rotation", node, "Rotation", SocketType::point));
            static_cast<void>(context.bind(
                id, "Center", node, "Center", SocketType::point));
            static_cast<void>(context.bind(
                id, "Axis", node, "Axis", SocketType::vector));
            static_cast<void>(context.bind(
                id, "Angle", node, "Angle", SocketType::floating));
            static_cast<void>(context.graph().set_property(
                id,
                "Type",
                SocketValue::string(context.node_property_text(
                    node, "rotation_type", "AXIS_ANGLE"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Invert",
                SocketValue::boolean(context.node_property_bool(
                    node, "invert"))));
            return finish({
                .ref = {.node = id, .socket = "Vector"},
                .type = SocketType::vector});
        }
        if (type == "VECTOR_TRANSFORM") {
            const auto id = context.graph().add_node(
                compiler::node_type::vector_transform,
                node_name);
            static_cast<void>(context.bind(
                id, "Vector", node, "Vector", SocketType::vector));
            static_cast<void>(context.graph().set_property(
                id,
                "Type",
                SocketValue::string(context.node_property_text(
                    node, "vector_type", "VECTOR"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Convert From",
                SocketValue::string(context.node_property_text(
                    node, "convert_from", "WORLD"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Convert To",
                SocketValue::string(context.node_property_text(
                    node, "convert_to", "OBJECT"))));
            return finish({
                .ref = {.node = id, .socket = "Vector"},
                .type = SocketType::vector});
        }
        if (type == "NORMAL") {
            // Cycles' NormalNode has one authored direction shared by both
            // outputs. It is stored on the Blender output socket rather than
            // on the input socket: D = normalize(direction), Normal = D, and
            // Dot = dot(D, normalize(input Normal)). Build that definition as
            // a typed DAG so output-driven lowering cannot duplicate D and so
            // ordinary value scheduling/compaction remains authoritative.
            auto direction = context.shared_output(
                node_name, "normalized_direction");
            if (!direction) {
                const auto authored_direction =
                    context.constant_from_output(
                        node, "Normal", SocketType::vector);
                const auto normalize_direction =
                    context.graph().add_node(
                        compiler::node_type::vector_math,
                        node_name + " / Normalize Direction");
                static_cast<void>(context.graph().connect(
                    authored_direction.ref,
                    normalize_direction,
                    "A"));
                static_cast<void>(context.graph().set_property(
                    normalize_direction,
                    "Operation",
                    SocketValue::string("CYCLES_NORMALIZE")));
                direction = TypedOutput{
                    .ref = {
                        .node = normalize_direction,
                        .socket = "Vector"},
                    .type = SocketType::vector};
                context.remember_shared_output(
                    node_name,
                    "normalized_direction",
                    *direction);
            }
            if (socket == "Normal") {
                return finish(context.conversion(
                    *direction, SocketType::normal));
            }
            if (socket == "Dot") {
                const auto normalize_input =
                    context.graph().add_node(
                        compiler::node_type::vector_math,
                        node_name + " / Normalize Input");
                static_cast<void>(context.bind(
                    normalize_input,
                    "A",
                    node,
                    "Normal",
                    SocketType::vector));
                static_cast<void>(context.graph().set_property(
                    normalize_input,
                    "Operation",
                    SocketValue::string("CYCLES_NORMALIZE")));

                const auto dot = context.graph().add_node(
                    compiler::node_type::vector_math,
                    node_name + " / Dot");
                static_cast<void>(context.graph().connect(
                    direction->ref, dot, "A"));
                static_cast<void>(context.graph().connect(
                    {.node = normalize_input,
                     .socket = "Vector"},
                    dot,
                    "B"));
                static_cast<void>(context.graph().set_property(
                    dot,
                    "Operation",
                    SocketValue::string("DOT_PRODUCT")));
                return finish({
                    .ref = {.node = dot, .socket = "Value"},
                    .type = SocketType::floating});
            }
            return std::nullopt;
        }
        if (type == "MATH") {
            const auto operation =
                context.node_property_text(node, "operation", "ADD");
            constexpr std::array supported{
                "ADD",
                "SUBTRACT",
                "MULTIPLY",
                "DIVIDE",
                "MULTIPLY_ADD",
                "POWER",
                "LOGARITHM",
                "SQRT",
                "INVERSE_SQRT",
                "ABSOLUTE",
                "EXPONENT",
                "MINIMUM",
                "MAXIMUM",
                "LESS_THAN",
                "GREATER_THAN",
                "SIGN",
                "COMPARE",
                "SMOOTH_MIN",
                "SMOOTH_MAX",
                "ROUND",
                "FLOOR",
                "CEIL",
                "TRUNC",
                "FRACT",
                "MODULO",
                "FLOORED_MODULO",
                "WRAP",
                "SNAP",
                "PINGPONG",
                "SINE",
                "COSINE",
                "TANGENT",
                "ARCSINE",
                "ARCCOSINE",
                "ARCTANGENT",
                "ARCTAN2",
                "SINH",
                "COSH",
                "TANH",
                "RADIANS",
                "DEGREES"};
            if (std::find(
                    supported.begin(),
                    supported.end(),
                    operation) == supported.end()) {
                context.warn_once(
                    "math:" + operation,
                    "Math operation '" + operation +
                        "' currently uses Add");
            }
            const auto id = context.graph().add_node(
                compiler::node_type::math,
                node_name);
            static_cast<void>(context.bind(
                id, "A", node, "Value", SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "B",
                node,
                "Value_001",
                SocketType::floating));
            static_cast<void>(context.bind(
                id,
                "C",
                node,
                "Value_002",
                SocketType::floating));
            static_cast<void>(context.graph().set_property(
                id,
                "Operation",
                SocketValue::string(operation)));
            TypedOutput result{
                .ref = {.node = id, .socket = "Value"},
                .type = SocketType::floating};
            if (context.node_property_bool(node, "use_clamp")) {
                const auto clamped = context.graph().add_node(
                    compiler::node_type::clamp_float,
                    node_name + " Clamp");
                static_cast<void>(context.graph().connect(
                    result.ref, clamped, "Value"));
                result.ref = {
                    .node = clamped, .socket = "Value"};
            }
            return finish(result);
        }
        if (type == "VALTORGB") {
            const auto id = context.graph().add_node(
                compiler::node_type::color_ramp,
                node_name);
            static_cast<void>(context.bind(
                id, "Factor", node, "Fac", SocketType::floating));
            auto *ramp = member(
                member(node, "special"), "color_ramp");
            auto *samples = member(ramp, "samples");
            static_cast<void>(context.graph().set_property(
                id,
                "Interpolation",
                SocketValue::string(text(
                    member(ramp, "interpolation"),
                    "LINEAR"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Sampled",
                SocketValue::boolean(
                    samples != nullptr &&
                    yyjson_is_arr(samples) &&
                    yyjson_arr_size(samples) >= 2u)));
            static_cast<void>(context.graph().set_property(
                id,
                "Table",
                SocketValue::string(context.color_ramp_table(node))));
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
            const auto id = context.graph().add_node(
                compiler::node_type::rgb_curve,
                node_name);
            auto *mapping = member(
                member(node, "special"), "curve_mapping");
            auto *samples = member(mapping, "samples");
            static_cast<void>(context.bind(
                id, "Factor", node, "Fac", SocketType::floating));
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.graph().set_property(
                id,
                "Sampled",
                SocketValue::boolean(
                    samples != nullptr &&
                    yyjson_is_arr(samples) &&
                    yyjson_arr_size(samples) >= 2u)));
            static_cast<void>(context.graph().set_property(
                id,
                "MinX",
                SocketValue::floating(
                    number(member(mapping, "min_x"), 0.0f))));
            static_cast<void>(context.graph().set_property(
                id,
                "MaxX",
                SocketValue::floating(
                    number(member(mapping, "max_x"), 1.0f))));
            static_cast<void>(context.graph().set_property(
                id,
                "Extrapolate",
                SocketValue::boolean(
                    boolean(
                        member(mapping, "extrapolate"),
                        true))));
            static_cast<void>(context.graph().set_property(
                id,
                "Table",
                SocketValue::string(context.rgb_curve_table(node))));
            return finish({
                .ref = {.node = id, .socket = "Color"},
                .type = SocketType::color});
        }
        if (type == "NORMAL_MAP") {
            const auto id = context.graph().add_node(
                compiler::node_type::normal_map,
                node_name);
            const auto uv_map =
                context.node_property_text(node, "uv_map");
            const auto base =
                context.node_property_text(
                    node, "base", "ORIGINAL");
            static_cast<void>(context.bind(
                id,
                "Strength",
                node,
                "Strength",
                SocketType::floating));
            static_cast<void>(context.bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(context.graph().set_property(
                id,
                "Space",
                SocketValue::string(context.node_property_text(
                    node, "space", "TANGENT"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Convention",
                SocketValue::string(context.node_property_text(
                    node, "convention", "OPENGL"))));
            static_cast<void>(context.graph().set_property(
                id,
                "Base",
                SocketValue::string(base)));
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
                        : base == "ORIGINAL"
                              ? contract::
                                    uv_undisplaced_tangent_attribute_id(
                                        uv_map)
                              : contract::uv_tangent_attribute_id(
                                    uv_map))));
            return finish({
                .ref = {.node = id, .socket = "Normal"},
                .type = SocketType::normal});
        }
        return std::nullopt;
    }
};

}// namespace

std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_value_lowering_component() {
    return std::make_unique<ValueNodeLoweringComponent>();
}

}// namespace psycles::adapter::detail
