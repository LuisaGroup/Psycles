// Color transforms, geometry/path inputs, Fresnel, and mapping cases.
// Included by <psycles/luisa/graph_surface.h>; not a standalone header.

                case compiler::ValueOperation::multiply_color: {
                    auto t = clamp(
                        scalar(instruction.c, result),
                        0.0f,
                        1.0f);
                    auto a = vector(instruction.a, result);
                    value = make_float4(
                        lerp(
                            a,
                            a * vector(instruction.b, result),
                            t),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::hue_saturation: {
                    // Cycles' NODE_HSV contract: adjust in HSV space,
                    // wrap hue with fract(), clamp only saturation, blend
                    // with the unmodified input, and clamp the final RGB
                    // against negative oversaturation artifacts. Fac is
                    // intentionally not clamped.
                    auto color = vector(instruction.a, result);
                    auto adjusted = rgb_to_hsv(color);
                    adjusted.x = fract(
                        adjusted.x +
                        scalar(instruction.b, result) +
                        0.5f);
                    adjusted.y = clamp(
                        adjusted.y *
                            scalar(instruction.c, result),
                        0.0f,
                        1.0f);
                    adjusted.z *=
                        scalar(instruction.d, result);
                    adjusted = hsv_to_rgb(adjusted);
                    auto factor =
                        scalar(instruction.e, result);
                    value = make_float4(
                        max(
                            lerp(color, adjusted, factor),
                            make_float3(0.0f)),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::invert: {
                    auto color = vector(instruction.a, result);
                    auto factor =
                        scalar(instruction.b, result);
                    value = make_float4(
                        lerp(
                            color,
                            make_float3(1.0f) - color,
                            factor),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::gamma: {
                    auto color = vector(instruction.a, result);
                    auto exponent =
                        scalar(instruction.b, result);
                    auto adjusted = make_float3(
                        select(
                            color.x,
                            pow(max(color.x, 0.0f), exponent),
                            color.x > 0.0f),
                        select(
                            color.y,
                            pow(max(color.y, 0.0f), exponent),
                            color.y > 0.0f),
                        select(
                            color.z,
                            pow(max(color.z, 0.0f), exponent),
                            color.z > 0.0f));
                    adjusted = select(
                        adjusted,
                        make_float3(1.0f),
                        exponent == 0.0f);
                    value = make_float4(adjusted, 1.0f);
                    break;
                }
                case compiler::ValueOperation::brightness_contrast: {
                    auto color = vector(instruction.a, result);
                    auto brightness =
                        scalar(instruction.b, result);
                    auto contrast =
                        scalar(instruction.c, result);
                    auto a = 1.0f + contrast;
                    auto b = brightness -
                             contrast * 0.5f;
                    value = make_float4(
                        max(
                            a * color + make_float3(b),
                            make_float3(0.0f)),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::blackbody:
                    value = make_float4(
                        max(
                            services.rec709_to_rgb(
                                cycles_color_nodes::
                                    blackbody_rec709(
                                        scalar(
                                            instruction.a,
                                            result))),
                            make_float3(0.0f)),
                        1.0f);
                    break;
                case compiler::ValueOperation::wavelength:
                    value = make_float4(
                        max(
                            services.xyz_to_rgb(
                                cycles_color_nodes::
                                    wavelength_xyz(
                                        scalar(
                                            instruction.a,
                                            result))) *
                                (1.0f / 2.52f),
                            make_float3(0.0f)),
                        1.0f);
                    break;
                case compiler::ValueOperation::surface_position:
                    value = make_float4(point.position, 1.0f);
                    break;
                case compiler::ValueOperation::shading_normal:
                    value = make_float4(
                        point.shading_normal, 0.0f);
                    break;
                case compiler::ValueOperation::geometric_normal:
                    value = make_float4(
                        point.geometric_normal, 0.0f);
                    break;
                case compiler::ValueOperation::incoming:
                    value = make_float4(point.incoming, 0.0f);
                    break;
                case compiler::ValueOperation::tangent:
                    value = make_float4(point.dpdu, 0.0f);
                    break;
                case compiler::ValueOperation::uv:
                    if (instruction.static_u0 != 0u) {
                        value = services.attribute(
                            instruction.static_u1, point)
                                    .value;
                    } else {
                        value = make_float4(
                            point.uv.x,
                            point.uv.y,
                            0.0f,
                            0.0f);
                    }
                    break;
                case compiler::ValueOperation::generated:
                    value = make_float4(point.generated, 1.0f);
                    break;
                case compiler::ValueOperation::object_position:
                    value = make_float4(
                        point.object_position, 1.0f);
                    break;
                case compiler::ValueOperation::object_location:
                    value = make_float4(
                        point.object_location, 1.0f);
                    break;
                case compiler::ValueOperation::object_random:
                    value = make_float4(point.object_random);
                    break;
                case compiler::ValueOperation::particle_index:
                    value = make_float4(
                        cast<float>(point.particle_index));
                    break;
                case compiler::ValueOperation::particle_random:
                    value = make_float4(
                        cycles_noise::uint_to_float_inclusive(
                            cycles_noise::hash_uint2(
                                point.particle_index, 0u)));
                    break;
                case compiler::ValueOperation::back_facing:
                    value = make_float4(select(
                        0.0f, 1.0f, point.back_facing));
                    break;
                case compiler::ValueOperation::random_per_island:
                    value = make_float4(
                        point.random_per_island);
                    break;
                case compiler::ValueOperation::path_is_camera:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_visibility &
                         camera_ray_visibility) != 0u));
                    break;
                case compiler::ValueOperation::path_is_shadow:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_visibility &
                         shadow_ray_visibility) != 0u));
                    break;
                case compiler::ValueOperation::path_is_diffuse:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_visibility &
                         diffuse_ray_visibility) != 0u));
                    break;
                case compiler::ValueOperation::path_is_glossy:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_visibility &
                         glossy_ray_visibility) != 0u));
                    break;
                case compiler::ValueOperation::path_is_singular:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_events &
                         static_cast<std::uint32_t>(
                             contract::event_singular)) != 0u));
                    break;
                case compiler::ValueOperation::path_is_reflection:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_events &
                         static_cast<std::uint32_t>(
                             contract::event_reflection)) != 0u));
                    break;
                case compiler::ValueOperation::path_is_transmission:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_visibility &
                         transmission_ray_visibility) != 0u));
                    break;
                case compiler::ValueOperation::path_is_volume_scatter:
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        (point.ray_visibility &
                         volume_ray_visibility) != 0u));
                    break;
                case compiler::ValueOperation::path_ray_length:
                    value = make_float4(point.ray_length);
                    break;
                case compiler::ValueOperation::path_ray_depth:
                    value = make_float4(
                        cast<float>(point.ray_depth));
                    break;
                case compiler::ValueOperation::path_diffuse_depth:
                    value = make_float4(
                        cast<float>(point.diffuse_depth));
                    break;
                case compiler::ValueOperation::path_glossy_depth:
                    value = make_float4(
                        cast<float>(point.glossy_depth));
                    break;
                case compiler::ValueOperation::path_transparent_depth:
                    value = make_float4(
                        cast<float>(
                            point.transparent_depth));
                    break;
                case compiler::ValueOperation::path_transmission_depth:
                    value = make_float4(
                        cast<float>(
                            point.transmission_depth));
                    break;
                case compiler::ValueOperation::fresnel: {
                    auto eta = max(
                        scalar(instruction.a, result),
                        1.0e-5f);
                    eta = select(
                        eta,
                        1.0f / eta,
                        point.back_facing);
                    auto normal = safe_normalize(
                        vector(instruction.b, result),
                        result.shading_normal);
                    value = make_float4(
                        fresnel_dielectric_cos(
                            dot(point.incoming, normal),
                            eta));
                    break;
                }
                case compiler::ValueOperation::layer_weight_fresnel: {
                    auto blend = scalar(instruction.a, result);
                    auto normal =
                        instruction.static_u0 != 0u
                            ? vector(instruction.b, result)
                            : result.shading_normal;
                    auto eta = max(1.0f - blend, 1.0e-5f);
                    eta = select(
                        1.0f / eta,
                        eta,
                        point.back_facing);
                    value = make_float4(
                        fresnel_dielectric_cos(
                            dot(point.incoming, normal),
                            eta));
                    break;
                }
                case compiler::ValueOperation::layer_weight_facing: {
                    auto blend = clamp(
                        scalar(instruction.a, result),
                        0.0f,
                        1.0f - 1.0e-5f);
                    auto normal =
                        instruction.static_u0 != 0u
                            ? vector(instruction.b, result)
                            : result.shading_normal;
                    auto facing = abs(dot(
                        point.incoming, normal));
                    auto exponent = select(
                        0.5f / (1.0f - blend),
                        2.0f * blend,
                        blend < 0.5f);
                    value = make_float4(
                        1.0f - pow(facing, exponent));
                    break;
                }
                case compiler::ValueOperation::mapping: {
                    auto input = vector(instruction.a, result);
                    auto location = vector(instruction.b, result);
                    auto rotation = vector(instruction.c, result);
                    auto scale = vector(instruction.d, result);
                    Float3 mapped = input;
                    if (instruction.static_u0 == 1u) {
                        mapped = safe_divide_components(
                            rotate_euler_transposed(
                                input - location,
                                rotation),
                            scale);
                    } else if (
                        instruction.static_u0 == 3u) {
                        mapped = rotate_euler(
                            safe_divide_components(
                                input, scale),
                            rotation);
                        auto mapped_length = length(mapped);
                        mapped /= select(
                            1.0f,
                            mapped_length,
                            mapped_length != 0.0f);
                    } else {
                        mapped = rotate_euler(
                            input * scale, rotation);
                        if (instruction.static_u0 == 0u) {
                            mapped += location;
                        }
                    }
                    value = make_float4(mapped, 0.0f);
                    break;
                }
