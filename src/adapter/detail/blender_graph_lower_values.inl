// Color, mix, scalar/vector math, ramps, curves, and normal-map lowering.
// Included by blender_graph_normalizer.cpp inside lower_natural_output.

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
        if (type == "BLACKBODY" || type == "WAVELENGTH") {
            const auto is_blackbody = type == "BLACKBODY";
            const auto input_name =
                is_blackbody ? "Temperature" : "Wavelength";
            const auto id = _graph.add_node(
                is_blackbody
                    ? compiler::node_type::blackbody
                    : compiler::node_type::wavelength,
                node_name);
            static_cast<void>(bind(
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
                node_property_text(node, "blend_type", "MIX");
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
                warn_once(
                    "mix:" + blend_type,
                    "Mix blend mode '" + blend_type +
                        "' currently uses linear Mix");
            }
            if (type == "MIX_RGB") {
                const auto id = _graph.add_node(
                    compiler::node_type::mix_color,
                    node_name);
                static_cast<void>(bind(
                    id,
                    "Factor",
                    node,
                    "Fac",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "A",
                    node,
                    "Color1",
                    SocketType::color));
                static_cast<void>(bind(
                    id,
                    "B",
                    node,
                    "Color2",
                    SocketType::color));
                static_cast<void>(_graph.set_property(
                    id,
                    "BlendMode",
                    SocketValue::string(blend_type)));
                // Cycles always clamps the legacy MixRGB factor. The
                // Blender use_alpha option is not consumed by Cycles.
                static_cast<void>(_graph.set_property(
                    id,
                    "ClampFactor",
                    SocketValue::boolean(true)));
                static_cast<void>(_graph.set_property(
                    id,
                    "ClampResult",
                    SocketValue::boolean(node_property_bool(
                        node, "use_clamp"))));
                return finish({
                    .ref = {.node = id, .socket = "Color"},
                    .type = SocketType::color});
            }

            const auto data_type =
                node_property_text(node, "data_type", "FLOAT");
            const auto clamp_factor =
                node_property_bool(node, "clamp_factor", true);
            if (data_type == "FLOAT") {
                const auto id = _graph.add_node(
                    compiler::node_type::mix_float,
                    node_name);
                static_cast<void>(bind(
                    id,
                    "Factor",
                    node,
                    "Factor_Float",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "A",
                    node,
                    "A_Float",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "B",
                    node,
                    "B_Float",
                    SocketType::floating));
                static_cast<void>(_graph.set_property(
                    id,
                    "ClampFactor",
                    SocketValue::boolean(clamp_factor)));
                return finish({
                    .ref = {.node = id, .socket = "Value"},
                    .type = SocketType::floating});
            }
            if (data_type == "VECTOR") {
                const auto nonuniform =
                    node_property_text(
                        node,
                        "factor_mode",
                        "UNIFORM") == "NON_UNIFORM";
                const auto id = _graph.add_node(
                    nonuniform
                        ? compiler::node_type::
                              mix_vector_nonuniform
                        : compiler::node_type::mix_vector,
                    node_name);
                static_cast<void>(bind(
                    id,
                    "Factor",
                    node,
                    nonuniform
                        ? "Factor_Vector"
                        : "Factor_Float",
                    nonuniform
                        ? SocketType::vector
                        : SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "A",
                    node,
                    "A_Vector",
                    SocketType::vector));
                static_cast<void>(bind(
                    id,
                    "B",
                    node,
                    "B_Vector",
                    SocketType::vector));
                static_cast<void>(_graph.set_property(
                    id,
                    "ClampFactor",
                    SocketValue::boolean(clamp_factor)));
                return finish({
                    .ref = {.node = id, .socket = "Vector"},
                    .type = SocketType::vector});
            }
            if (data_type == "RGBA") {
                const auto id = _graph.add_node(
                    compiler::node_type::mix_color,
                    node_name);
                static_cast<void>(bind(
                    id,
                    "Factor",
                    node,
                    "Factor_Float",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "A",
                    node,
                    "A_Color",
                    SocketType::color));
                static_cast<void>(bind(
                    id,
                    "B",
                    node,
                    "B_Color",
                    SocketType::color));
                static_cast<void>(_graph.set_property(
                    id,
                    "BlendMode",
                    SocketValue::string(blend_type)));
                static_cast<void>(_graph.set_property(
                    id,
                    "ClampFactor",
                    SocketValue::boolean(clamp_factor)));
                static_cast<void>(_graph.set_property(
                    id,
                    "ClampResult",
                    SocketValue::boolean(node_property_bool(
                        node, "clamp_result"))));
                return finish({
                    .ref = {.node = id, .socket = "Color"},
                    .type = SocketType::color});
            }
            warn_once(
                "mix-data:" + data_type,
                "Mix data type '" + data_type +
                    "' is not exposed by the Cycles device adapter");
            return finish(constant_from_output(
                node,
                socket,
                SocketType::vector));
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
        if (type == "MAP_RANGE") {
            const auto data_type =
                node_property_text(node, "data_type", "FLOAT");
            const auto vector_mode =
                data_type == "FLOAT_VECTOR";
            const auto id = _graph.add_node(
                compiler::node_type::map_range,
                node_name);
            if (vector_mode) {
                static_cast<void>(bind(
                    id,
                    "Vector",
                    node,
                    "Vector",
                    SocketType::vector));
                static_cast<void>(bind(
                    id,
                    "FromMinVector",
                    node,
                    "From_Min_FLOAT3",
                    SocketType::vector));
                static_cast<void>(bind(
                    id,
                    "FromMaxVector",
                    node,
                    "From_Max_FLOAT3",
                    SocketType::vector));
                static_cast<void>(bind(
                    id,
                    "ToMinVector",
                    node,
                    "To_Min_FLOAT3",
                    SocketType::vector));
                static_cast<void>(bind(
                    id,
                    "ToMaxVector",
                    node,
                    "To_Max_FLOAT3",
                    SocketType::vector));
                static_cast<void>(bind(
                    id,
                    "StepsVector",
                    node,
                    "Steps_FLOAT3",
                    SocketType::vector));
            } else {
                static_cast<void>(bind(
                    id,
                    "Value",
                    node,
                    "Value",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "FromMin",
                    node,
                    "From Min",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "FromMax",
                    node,
                    "From Max",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "ToMin",
                    node,
                    "To Min",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "ToMax",
                    node,
                    "To Max",
                    SocketType::floating));
                static_cast<void>(bind(
                    id,
                    "Steps",
                    node,
                    "Steps",
                    SocketType::floating));
            }
            static_cast<void>(_graph.set_property(
                id,
                "DataType",
                SocketValue::string(data_type)));
            static_cast<void>(_graph.set_property(
                id,
                "Interpolation",
                SocketValue::string(node_property_text(
                    node,
                    "interpolation_type",
                    "LINEAR"))));
            static_cast<void>(_graph.set_property(
                id,
                "Clamp",
                SocketValue::boolean(node_property_bool(
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
            const auto id = _graph.add_node(
                compiler::node_type::vector_math,
                node_name);
            static_cast<void>(bind(
                id,
                "A",
                node,
                "Vector",
                SocketType::vector));
            static_cast<void>(bind(
                id,
                "B",
                node,
                "Vector_001",
                SocketType::vector));
            static_cast<void>(bind(
                id,
                "C",
                node,
                "Vector_002",
                SocketType::vector));
            static_cast<void>(bind(
                id,
                "Scale",
                node,
                "Scale",
                SocketType::floating));
            static_cast<void>(_graph.set_property(
                id,
                "Operation",
                SocketValue::string(node_property_text(
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
        if (type == "MATH") {
            const auto operation =
                node_property_text(node, "operation", "ADD");
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
                warn_once(
                    "math:" + operation,
                    "Math operation '" + operation +
                        "' currently uses Add");
            }
            const auto id = _graph.add_node(
                compiler::node_type::math,
                node_name);
            static_cast<void>(bind(
                id, "A", node, "Value", SocketType::floating));
            static_cast<void>(bind(
                id,
                "B",
                node,
                "Value_001",
                SocketType::floating));
            static_cast<void>(bind(
                id,
                "C",
                node,
                "Value_002",
                SocketType::floating));
            static_cast<void>(_graph.set_property(
                id,
                "Operation",
                SocketValue::string(operation)));
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
            auto *samples = member(ramp, "samples");
            static_cast<void>(_graph.set_property(
                id,
                "Interpolation",
                SocketValue::string(text(
                    member(ramp, "interpolation"),
                    "LINEAR"))));
            static_cast<void>(_graph.set_property(
                id,
                "Sampled",
                SocketValue::boolean(
                    samples != nullptr &&
                    yyjson_is_arr(samples) &&
                    yyjson_arr_size(samples) >= 2u)));
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
            auto *mapping = member(
                member(node, "special"), "curve_mapping");
            auto *samples = member(mapping, "samples");
            static_cast<void>(bind(
                id, "Factor", node, "Fac", SocketType::floating));
            static_cast<void>(bind(
                id, "Color", node, "Color", SocketType::color));
            static_cast<void>(_graph.set_property(
                id,
                "Sampled",
                SocketValue::boolean(
                    samples != nullptr &&
                    yyjson_is_arr(samples) &&
                    yyjson_arr_size(samples) >= 2u)));
            static_cast<void>(_graph.set_property(
                id,
                "MinX",
                SocketValue::floating(
                    number(member(mapping, "min_x"), 0.0f))));
            static_cast<void>(_graph.set_property(
                id,
                "MaxX",
                SocketValue::floating(
                    number(member(mapping, "max_x"), 1.0f))));
            static_cast<void>(_graph.set_property(
                id,
                "Extrapolate",
                SocketValue::boolean(
                    boolean(
                        member(mapping, "extrapolate"),
                        true))));
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
            const auto uv_map =
                node_property_text(node, "uv_map");
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
            static_cast<void>(_graph.set_property(
                id,
                "UvMapNamed",
                SocketValue::boolean(!uv_map.empty())));
            static_cast<void>(_graph.set_property(
                id,
                "UvMapId",
                SocketValue::unsigned_integer(
                    uv_map.empty()
                        ? 0u
                        : contract::uv_tangent_attribute_id(
                              uv_map))));
            return finish({
                .ref = {.node = id, .socket = "Normal"},
                .type = SocketType::normal});
        }
