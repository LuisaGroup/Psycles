// Constants, coordinate/context nodes, mapping, and image input lowering.
// Included by blender_graph_normalizer.cpp inside lower_natural_output.

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
            if (type == "UVMAP") {
                const auto uv_map =
                    node_property_text(node, "uv_map");
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
            static_cast<void>(_graph.set_property(
                id,
                "NormalLinked",
                SocketValue::boolean(
                    input_source(node, "Normal").has_value())));
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
        if (type == "FRESNEL") {
            const auto id = _graph.add_node(
                compiler::node_type::fresnel,
                node_name);
            static_cast<void>(bind(
                id,
                "IOR",
                node,
                "IOR",
                SocketType::floating));
            static_cast<void>(bind(
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
            const auto image_color_space =
                color_iter == _image_color_spaces.end()
                    ? ImageColorSpace::data
                    : color_iter->second;
            const auto color_space =
                image_color_space == ImageColorSpace::srgb
                    ? "sRGB"
                    : "Non-Color";
            const auto alpha_iter =
                _image_alpha_types.find(image_name);
            const auto alpha_type =
                alpha_iter == _image_alpha_types.end()
                    ? ImageAlphaType::straight
                    : alpha_iter->second;
            const auto unassociate_alpha =
                output_is_linked(node, "Alpha") &&
                image_color_space != ImageColorSpace::data &&
                alpha_type != ImageAlphaType::channel_packed &&
                alpha_type != ImageAlphaType::ignore;
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
                "Interpolation",
                SocketValue::string(node_property_text(
                    node, "interpolation", "Linear"))));
            static_cast<void>(_graph.set_property(
                id,
                "Projection",
                SocketValue::string(node_property_text(
                    node, "projection", "FLAT"))));
            static_cast<void>(_graph.set_property(
                id,
                "ProjectionBlend",
                SocketValue::floating(node_property_number(
                    node, "projection_blend", 0.0f))));
            static_cast<void>(_graph.set_property(
                id,
                "ColorSpace",
                SocketValue::string(color_space)));
            static_cast<void>(_graph.set_property(
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
