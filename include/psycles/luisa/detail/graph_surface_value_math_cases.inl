// Scalar, vector-math, and mix node evaluation cases.
// Included by <psycles/luisa/graph_surface.h>; not a standalone header.

                case compiler::ValueOperation::parameter:
                    value = make_float4(
                        services.parameter_float3(
                            point.parameter_block,
                            instruction.parameter.value),
                        services.parameter_float(
                            point.parameter_block,
                            instruction.parameter.value));
                    break;
                case compiler::ValueOperation::passthrough:
                    value = get(instruction.a, result.values);
                    break;
                case compiler::ValueOperation::scalar_to_color: {
                    auto x = scalar(instruction.a, result);
                    value = make_float4(x, x, x, 1.0f);
                    break;
                }
                case compiler::ValueOperation::color_to_scalar: {
                    auto color = vector(instruction.a, result);
                    value = make_float4(
                        dot(
                            color,
                            // Blender 4.5 default scene-linear
                            // Film::rgb_to_y coefficients.
                            make_float3(
                                0.21267404f,
                                0.7151516f,
                                0.07217542f)));
                    break;
                }
                case compiler::ValueOperation::vector_to_scalar: {
                    auto vector_value =
                        vector(instruction.a, result);
                    value = make_float4(
                        (vector_value.x +
                         vector_value.y +
                         vector_value.z) /
                        3.0f);
                    break;
                }
                case compiler::ValueOperation::add:
                    value = make_float4(
                        scalar(instruction.a, result) +
                        scalar(instruction.b, result));
                    break;
                case compiler::ValueOperation::subtract:
                    value = make_float4(
                        scalar(instruction.a, result) -
                        scalar(instruction.b, result));
                    break;
                case compiler::ValueOperation::multiply:
                    value = make_float4(
                        scalar(instruction.a, result) *
                        scalar(instruction.b, result));
                    break;
                case compiler::ValueOperation::divide: {
                    auto denominator =
                        scalar(instruction.b, result);
                    value = make_float4(select(
                        0.0f,
                        scalar(instruction.a, result) /
                            denominator,
                        abs(denominator) > 1.0e-20f));
                    break;
                }
                case compiler::ValueOperation::minimum:
                    value = make_float4(min(
                        scalar(instruction.a, result),
                        scalar(instruction.b, result)));
                    break;
                case compiler::ValueOperation::maximum:
                    value = make_float4(max(
                        scalar(instruction.a, result),
                        scalar(instruction.b, result)));
                    break;
                case compiler::ValueOperation::power:
                    value = make_float4(pow(
                        max(
                            scalar(instruction.a, result),
                            0.0f),
                        scalar(instruction.b, result)));
                    break;
                case compiler::ValueOperation::math: {
                    auto a = scalar(instruction.a, result);
                    auto b = scalar(instruction.b, result);
                    auto c = scalar(instruction.c, result);
                    Float evaluated = 0.0f;
                    switch (static_cast<compiler::MathOperation>(
                        instruction.static_u0)) {
                        case compiler::MathOperation::add:
                            evaluated = a + b;
                            break;
                        case compiler::MathOperation::subtract:
                            evaluated = a - b;
                            break;
                        case compiler::MathOperation::multiply:
                            evaluated = a * b;
                            break;
                        case compiler::MathOperation::divide:
                            evaluated = select(
                                0.0f, a / b, b != 0.0f);
                            break;
                        case compiler::MathOperation::multiply_add:
                            evaluated = a * b + c;
                            break;
                        case compiler::MathOperation::power: {
                            auto integer_exponent = b == trunc(b);
                            auto powered = pow(abs(a), b);
                            auto odd_exponent =
                                fmod(abs(b), 2.0f) != 0.0f;
                            powered = select(
                                powered,
                                -powered,
                                (a < 0.0f) & odd_exponent);
                            evaluated = select(
                                0.0f,
                                powered,
                                (a >= 0.0f) | integer_exponent);
                            break;
                        }
                        case compiler::MathOperation::logarithm: {
                            auto denominator = log(b);
                            evaluated = select(
                                0.0f,
                                log(a) / denominator,
                                (a > 0.0f) &
                                    (b > 0.0f) &
                                    (denominator != 0.0f));
                            break;
                        }
                        case compiler::MathOperation::square_root:
                            evaluated = sqrt(max(a, 0.0f));
                            break;
                        case compiler::MathOperation::
                            inverse_square_root:
                            evaluated = select(
                                0.0f,
                                1.0f / sqrt(a),
                                a > 0.0f);
                            break;
                        case compiler::MathOperation::absolute:
                            evaluated = abs(a);
                            break;
                        case compiler::MathOperation::exponent:
                            evaluated = exp(a);
                            break;
                        case compiler::MathOperation::minimum:
                            evaluated = min(a, b);
                            break;
                        case compiler::MathOperation::maximum:
                            evaluated = max(a, b);
                            break;
                        case compiler::MathOperation::less_than:
                            evaluated = select(
                                0.0f, 1.0f, a < b);
                            break;
                        case compiler::MathOperation::greater_than:
                            evaluated = select(
                                0.0f, 1.0f, a > b);
                            break;
                        case compiler::MathOperation::sign:
                            evaluated = select(
                                select(1.0f, -1.0f, a < 0.0f),
                                0.0f,
                                a == 0.0f);
                            break;
                        case compiler::MathOperation::compare:
                            evaluated = select(
                                0.0f,
                                1.0f,
                                (a == b) |
                                    (abs(a - b) <=
                                     max(
                                         c,
                                         1.1920928955078125e-7f)));
                            break;
                        case compiler::MathOperation::smooth_minimum: {
                            auto nonzero = c != 0.0f;
                            auto h =
                                max(c - abs(a - b), 0.0f) / c;
                            auto smooth =
                                min(a, b) -
                                h * h * h * c *
                                    (1.0f / 6.0f);
                            evaluated = select(
                                min(a, b), smooth, nonzero);
                            break;
                        }
                        case compiler::MathOperation::smooth_maximum: {
                            auto nonzero = c != 0.0f;
                            auto h =
                                max(c - abs(a - b), 0.0f) / c;
                            auto smooth =
                                max(a, b) +
                                h * h * h * c *
                                    (1.0f / 6.0f);
                            evaluated = select(
                                max(a, b), smooth, nonzero);
                            break;
                        }
                        case compiler::MathOperation::round:
                            evaluated = floor(a + 0.5f);
                            break;
                        case compiler::MathOperation::floor:
                            evaluated = floor(a);
                            break;
                        case compiler::MathOperation::ceil:
                            evaluated = ceil(a);
                            break;
                        case compiler::MathOperation::trunc:
                            evaluated = trunc(a);
                            break;
                        case compiler::MathOperation::fraction:
                            evaluated = a - floor(a);
                            break;
                        case compiler::MathOperation::modulo:
                            evaluated = select(
                                0.0f, fmod(a, b), b != 0.0f);
                            break;
                        case compiler::MathOperation::floored_modulo:
                            evaluated = select(
                                0.0f,
                                a - floor(a / b) * b,
                                b != 0.0f);
                            break;
                        case compiler::MathOperation::wrap: {
                            auto range = b - c;
                            evaluated = select(
                                c,
                                a - range *
                                        floor((a - c) / range),
                                range != 0.0f);
                            break;
                        }
                        case compiler::MathOperation::snap:
                            evaluated = floor(select(
                                            0.0f,
                                            a / b,
                                            b != 0.0f)) *
                                        b;
                            break;
                        case compiler::MathOperation::ping_pong:
                            evaluated = select(
                                0.0f,
                                abs(
                                    fract(
                                        (a - b) /
                                        (b * 2.0f)) *
                                        b * 2.0f -
                                    b),
                                b != 0.0f);
                            break;
                        case compiler::MathOperation::sine:
                            evaluated = sin(a);
                            break;
                        case compiler::MathOperation::cosine:
                            evaluated = cos(a);
                            break;
                        case compiler::MathOperation::tangent:
                            evaluated = tan(a);
                            break;
                        case compiler::MathOperation::arcsine:
                            evaluated = asin(clamp(a, -1.0f, 1.0f));
                            break;
                        case compiler::MathOperation::arccosine:
                            evaluated = acos(clamp(a, -1.0f, 1.0f));
                            break;
                        case compiler::MathOperation::arctangent:
                            evaluated = atan(a);
                            break;
                        case compiler::MathOperation::arctangent2:
                            evaluated = select(
                                atan2(a, b),
                                0.0f,
                                (a == 0.0f) & (b == 0.0f));
                            break;
                        case compiler::MathOperation::
                            hyperbolic_sine:
                            evaluated = sinh(a);
                            break;
                        case compiler::MathOperation::
                            hyperbolic_cosine:
                            evaluated = cosh(a);
                            break;
                        case compiler::MathOperation::
                            hyperbolic_tangent:
                            evaluated = tanh(a);
                            break;
                        case compiler::MathOperation::radians:
                            evaluated = a * (pi / 180.0f);
                            break;
                        case compiler::MathOperation::degrees:
                            evaluated = a * (180.0f / pi);
                            break;
                    }
                    value = make_float4(evaluated);
                    break;
                }
                case compiler::ValueOperation::absolute:
                    value = make_float4(abs(
                        scalar(instruction.a, result)));
                    break;
                case compiler::ValueOperation::clamp01:
                    value = make_float4(clamp(
                        scalar(instruction.a, result),
                        0.0f,
                        1.0f));
                    break;
                case compiler::ValueOperation::clamp_range: {
                    auto input = scalar(instruction.a, result);
                    auto minimum =
                        scalar(instruction.b, result);
                    auto maximum =
                        scalar(instruction.c, result);
                    if (instruction.static_u0 == 1u) {
                        auto reverse = minimum > maximum;
                        auto original_minimum = minimum;
                        minimum = select(
                            minimum, maximum, reverse);
                        maximum = select(
                            maximum,
                            original_minimum,
                            reverse);
                    }
                    value = make_float4(
                        min(max(input, minimum), maximum));
                    break;
                }
                case compiler::ValueOperation::map_range_float: {
                    auto input = scalar(instruction.a, result);
                    auto from_min =
                        scalar(instruction.b, result);
                    auto from_max =
                        scalar(instruction.c, result);
                    auto to_min =
                        scalar(instruction.d, result);
                    auto to_max =
                        scalar(instruction.e, result);
                    auto steps =
                        scalar(instruction.f, result);
                    auto denominator = from_max - from_min;
                    auto has_range = denominator != 0.0f;
                    auto factor =
                        (input - from_min) /
                        select(1.0f, denominator, has_range);
                    if (instruction.static_u0 == 1u) {
                        factor = select(
                            0.0f,
                            floor(
                                factor * (steps + 1.0f)) /
                                select(
                                    1.0f,
                                    steps,
                                    steps > 0.0f),
                            steps > 0.0f);
                    } else if (
                        instruction.static_u0 == 2u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            (3.0f - 2.0f * factor) *
                            (factor * factor);
                    } else if (
                        instruction.static_u0 == 3u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            factor * factor * factor *
                            (factor *
                                     (factor * 6.0f - 15.0f) +
                             10.0f);
                    }
                    auto mapped =
                        to_min + factor * (to_max - to_min);
                    mapped = select(
                        0.0f, mapped, has_range);
                    if (instruction.static_u1 != 0u) {
                        auto minimum = min(to_min, to_max);
                        auto maximum = max(to_min, to_max);
                        mapped = min(
                            max(mapped, minimum), maximum);
                    }
                    value = make_float4(mapped);
                    break;
                }
                case compiler::ValueOperation::map_range_vector: {
                    auto input = vector(instruction.a, result);
                    auto from_min =
                        vector(instruction.b, result);
                    auto from_max =
                        vector(instruction.c, result);
                    auto to_min =
                        vector(instruction.d, result);
                    auto to_max =
                        vector(instruction.e, result);
                    auto steps =
                        vector(instruction.f, result);
                    auto numerator = input - from_min;
                    auto denominator = from_max - from_min;
                    auto safe_divide = [](
                                           Float numerator_component,
                                           Float denominator_component) {
                        auto nonzero =
                            denominator_component != 0.0f;
                        return select(
                            0.0f,
                            numerator_component /
                                select(
                                    1.0f,
                                    denominator_component,
                                    nonzero),
                            nonzero);
                    };
                    auto factor = make_float3(
                        safe_divide(
                            numerator.x, denominator.x),
                        safe_divide(
                            numerator.y, denominator.y),
                        safe_divide(
                            numerator.z, denominator.z));
                    if (instruction.static_u0 == 1u) {
                        auto stepped = [](
                                           Float factor_component,
                                           Float steps_component) {
                            auto valid =
                                steps_component > 0.0f;
                            return select(
                                0.0f,
                                floor(
                                    factor_component *
                                    (steps_component + 1.0f)) /
                                    select(
                                        1.0f,
                                        steps_component,
                                        valid),
                                valid);
                        };
                        factor = make_float3(
                            stepped(factor.x, steps.x),
                            stepped(factor.y, steps.y),
                            stepped(factor.z, steps.z));
                    } else if (
                        instruction.static_u0 == 2u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            (make_float3(3.0f) -
                             2.0f * factor) *
                            (factor * factor);
                    } else if (
                        instruction.static_u0 == 3u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            factor * factor * factor *
                            (factor *
                                     (factor * 6.0f - 15.0f) +
                             10.0f);
                    }
                    auto mapped =
                        to_min + factor * (to_max - to_min);
                    if (instruction.static_u1 != 0u &&
                        instruction.static_u0 < 2u) {
                        mapped = min(
                            max(mapped, min(to_min, to_max)),
                            max(to_min, to_max));
                    }
                    value = make_float4(mapped, 0.0f);
                    break;
                }
                case compiler::ValueOperation::vector_math_value:
                case compiler::ValueOperation::vector_math_vector: {
                    auto a = vector(instruction.a, result);
                    auto b = vector(instruction.b, result);
                    auto c = vector(instruction.c, result);
                    auto scale =
                        scalar(instruction.d, result);
                    auto safe_divide = [](
                                           Float numerator,
                                           Float denominator) {
                        auto valid = denominator != 0.0f;
                        return select(
                            0.0f,
                            numerator /
                                select(
                                    1.0f,
                                    denominator,
                                    valid),
                            valid);
                    };
                    auto safe_divide_vector =
                        [&](Float3 numerator, Float3 denominator) {
                            return make_float3(
                                safe_divide(
                                    numerator.x,
                                    denominator.x),
                                safe_divide(
                                    numerator.y,
                                    denominator.y),
                                safe_divide(
                                    numerator.z,
                                    denominator.z));
                        };
                    auto safe_normalize_zero =
                        [](Float3 input) {
                            auto input_length =
                                sqrt(dot(input, input));
                            auto valid =
                                input_length != 0.0f;
                            return select(
                                input,
                                input /
                                    select(
                                        1.0f,
                                        input_length,
                                        valid),
                                valid);
                        };
                    auto safe_power = [](Float base, Float exponent) {
                        auto integer_exponent =
                            exponent == trunc(exponent);
                        auto powered =
                            pow(abs(base), exponent);
                        auto odd_exponent =
                            fmod(abs(exponent), 2.0f) != 0.0f;
                        powered = select(
                            powered,
                            -powered,
                            (base < 0.0f) & odd_exponent);
                        return select(
                            0.0f,
                            powered,
                            (base >= 0.0f) |
                                integer_exponent);
                    };
                    auto wrap_component = [](
                                              Float input,
                                              Float maximum,
                                              Float minimum) {
                        auto range = maximum - minimum;
                        auto valid = range != 0.0f;
                        return select(
                            minimum,
                            input -
                                range *
                                    floor(
                                        (input - minimum) /
                                        select(
                                            1.0f,
                                            range,
                                            valid)),
                            valid);
                    };

                    Float scalar_result = 0.0f;
                    Float3 vector_result =
                        make_float3(0.0f);
                    switch (
                        static_cast<compiler::VectorMathOperation>(
                            instruction.static_u0)) {
                        case compiler::VectorMathOperation::add:
                            vector_result = a + b;
                            break;
                        case compiler::VectorMathOperation::subtract:
                            vector_result = a - b;
                            break;
                        case compiler::VectorMathOperation::multiply:
                            vector_result = a * b;
                            break;
                        case compiler::VectorMathOperation::divide:
                            vector_result =
                                safe_divide_vector(a, b);
                            break;
                        case compiler::VectorMathOperation::
                            multiply_add:
                            vector_result = a * b + c;
                            break;
                        case compiler::VectorMathOperation::
                            cross_product:
                            vector_result = cross(a, b);
                            break;
                        case compiler::VectorMathOperation::project: {
                            auto length_squared = dot(b, b);
                            auto valid =
                                length_squared != 0.0f;
                            vector_result = select(
                                make_float3(0.0f),
                                safe_divide(
                                    dot(a, b),
                                    length_squared) *
                                    b,
                                valid);
                            break;
                        }
                        case compiler::VectorMathOperation::reflect: {
                            auto normal =
                                safe_normalize_zero(b);
                            vector_result =
                                a -
                                2.0f * normal *
                                    dot(a, normal);
                            break;
                        }
                        case compiler::VectorMathOperation::refract: {
                            auto normal =
                                safe_normalize_zero(b);
                            auto cosine = dot(normal, a);
                            auto k =
                                1.0f -
                                scale * scale *
                                    (1.0f -
                                     cosine * cosine);
                            vector_result = select(
                                make_float3(0.0f),
                                scale * a -
                                    (scale * cosine +
                                     sqrt(max(k, 0.0f))) *
                                        normal,
                                k >= 0.0f);
                            break;
                        }
                        case compiler::VectorMathOperation::
                            faceforward:
                            vector_result = select(
                                -a,
                                a,
                                dot(c, b) < 0.0f);
                            break;
                        case compiler::VectorMathOperation::
                            dot_product:
                            scalar_result = dot(a, b);
                            break;
                        case compiler::VectorMathOperation::distance: {
                            auto delta = a - b;
                            scalar_result =
                                sqrt(dot(delta, delta));
                            break;
                        }
                        case compiler::VectorMathOperation::length:
                            scalar_result = sqrt(dot(a, a));
                            break;
                        case compiler::VectorMathOperation::scale:
                            vector_result = a * scale;
                            break;
                        case compiler::VectorMathOperation::normalize:
                            vector_result =
                                safe_normalize_zero(a);
                            break;
                        case compiler::VectorMathOperation::absolute:
                            vector_result = abs(a);
                            break;
                        case compiler::VectorMathOperation::power:
                            vector_result = make_float3(
                                safe_power(a.x, b.x),
                                safe_power(a.y, b.y),
                                safe_power(a.z, b.z));
                            break;
                        case compiler::VectorMathOperation::sign: {
                            auto sign_component = [](Float input) {
                                return select(
                                    select(
                                        1.0f,
                                        -1.0f,
                                        input < 0.0f),
                                    0.0f,
                                    input == 0.0f);
                            };
                            vector_result = make_float3(
                                sign_component(a.x),
                                sign_component(a.y),
                                sign_component(a.z));
                            break;
                        }
                        case compiler::VectorMathOperation::minimum:
                            vector_result = min(a, b);
                            break;
                        case compiler::VectorMathOperation::maximum:
                            vector_result = max(a, b);
                            break;
                        case compiler::VectorMathOperation::floor:
                            vector_result = floor(a);
                            break;
                        case compiler::VectorMathOperation::ceil:
                            vector_result = ceil(a);
                            break;
                        case compiler::VectorMathOperation::fraction:
                            vector_result = a - floor(a);
                            break;
                        case compiler::VectorMathOperation::modulo:
                            vector_result = make_float3(
                                select(
                                    0.0f,
                                    fmod(a.x, b.x),
                                    b.x != 0.0f),
                                select(
                                    0.0f,
                                    fmod(a.y, b.y),
                                    b.y != 0.0f),
                                select(
                                    0.0f,
                                    fmod(a.z, b.z),
                                    b.z != 0.0f));
                            break;
                        case compiler::VectorMathOperation::wrap:
                            vector_result = make_float3(
                                wrap_component(
                                    a.x, b.x, c.x),
                                wrap_component(
                                    a.y, b.y, c.y),
                                wrap_component(
                                    a.z, b.z, c.z));
                            break;
                        case compiler::VectorMathOperation::snap:
                            vector_result =
                                floor(
                                    safe_divide_vector(a, b)) *
                                b;
                            break;
                        case compiler::VectorMathOperation::sine:
                            vector_result = make_float3(
                                sin(a.x), sin(a.y), sin(a.z));
                            break;
                        case compiler::VectorMathOperation::cosine:
                            vector_result = make_float3(
                                cos(a.x), cos(a.y), cos(a.z));
                            break;
                        case compiler::VectorMathOperation::tangent:
                            vector_result = make_float3(
                                tan(a.x), tan(a.y), tan(a.z));
                            break;
                    }
                    value =
                        instruction.operation ==
                                compiler::ValueOperation::
                                    vector_math_value
                            ? make_float4(scalar_result)
                            : make_float4(
                                  vector_result, 0.0f);
                    break;
                }
                case compiler::ValueOperation::mix_float: {
                    auto t = scalar(instruction.c, result);
                    if (instruction.static_u0 != 0u) {
                        t = clamp(t, 0.0f, 1.0f);
                    }
                    value = make_float4(lerp(
                        scalar(instruction.a, result),
                        scalar(instruction.b, result),
                        t));
                    break;
                }
                case compiler::ValueOperation::mix_vector: {
                    auto t = instruction.static_u0 != 0u
                                 ? vector(instruction.c, result)
                                 : make_float3(
                                       scalar(
                                           instruction.c,
                                           result));
                    if (instruction.static_u1 != 0u) {
                        t = clamp(t, 0.0f, 1.0f);
                    }
                    value = make_float4(
                        lerp(
                            vector(instruction.a, result),
                            vector(instruction.b, result),
                            t),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::mix: {
                    auto t = scalar(instruction.c, result);
                    if ((instruction.static_u1 & 1u) != 0u) {
                        t = clamp(t, 0.0f, 1.0f);
                    }
                    auto a = vector(instruction.a, result);
                    auto b = vector(instruction.b, result);
                    Float3 mixed = a;
                    switch (static_cast<compiler::BlendOperation>(
                        instruction.static_u0)) {
                        case compiler::BlendOperation::mix:
                            mixed = lerp(a, b, t);
                            break;
                        case compiler::BlendOperation::darken:
                            mixed = lerp(a, min(a, b), t);
                            break;
                        case compiler::BlendOperation::multiply:
                            mixed = lerp(a, a * b, t);
                            break;
                        case compiler::BlendOperation::burn: {
                            auto denominator =
                                1.0f - t + t * b;
                            auto burned = clamp(
                                1.0f -
                                    (make_float3(1.0f) - a) /
                                        denominator,
                                0.0f,
                                1.0f);
                            mixed = select(
                                burned,
                                make_float3(0.0f),
                                denominator <= 0.0f);
                            break;
                        }
                        case compiler::BlendOperation::lighten:
                            mixed = lerp(a, max(a, b), t);
                            break;
                        case compiler::BlendOperation::screen:
                            mixed =
                                1.0f -
                                (1.0f - t +
                                 t * (make_float3(1.0f) - b)) *
                                    (make_float3(1.0f) - a);
                            break;
                        case compiler::BlendOperation::dodge: {
                            auto denominator = 1.0f - t * b;
                            auto dodged = min(
                                a / denominator,
                                make_float3(1.0f));
                            dodged = select(
                                dodged,
                                make_float3(1.0f),
                                denominator <= 0.0f);
                            mixed = select(
                                a, dodged, a != 0.0f);
                            break;
                        }
                        case compiler::BlendOperation::add:
                            mixed = lerp(a, a + b, t);
                            break;
                        case compiler::BlendOperation::overlay: {
                            auto low =
                                a * (1.0f - t + 2.0f * t * b);
                            auto high =
                                1.0f -
                                (1.0f - t +
                                 2.0f * t *
                                     (make_float3(1.0f) - b)) *
                                    (make_float3(1.0f) - a);
                            mixed = select(
                                high, low, a < 0.5f);
                            break;
                        }
                        case compiler::BlendOperation::soft_light: {
                            auto screen =
                                1.0f -
                                (make_float3(1.0f) - b) *
                                    (make_float3(1.0f) - a);
                            mixed =
                                (1.0f - t) * a +
                                t * ((make_float3(1.0f) - a) *
                                         b * a +
                                     a * screen);
                            break;
                        }
                        case compiler::BlendOperation::linear_light:
                            mixed =
                                a + t * (2.0f * b - 1.0f);
                            break;
                        case compiler::BlendOperation::difference:
                            mixed = lerp(a, abs(a - b), t);
                            break;
                        case compiler::BlendOperation::exclusion:
                            mixed = max(
                                lerp(
                                    a,
                                    a + b - 2.0f * a * b,
                                    t),
                                make_float3(0.0f));
                            break;
                        case compiler::BlendOperation::subtract:
                            mixed = lerp(a, a - b, t);
                            break;
                        case compiler::BlendOperation::divide: {
                            auto divided =
                                (1.0f - t) * a + t * a / b;
                            mixed = select(
                                a, divided, b != 0.0f);
                            break;
                        }
                        case compiler::BlendOperation::hue: {
                            auto hsv_b = rgb_to_hsv(b);
                            auto hsv = rgb_to_hsv(a);
                            hsv.x = hsv_b.x;
                            auto recolored = hsv_to_rgb(hsv);
                            mixed = select(
                                a,
                                lerp(a, recolored, t),
                                hsv_b.y != 0.0f);
                            break;
                        }
                        case compiler::BlendOperation::saturation: {
                            auto hsv = rgb_to_hsv(a);
                            auto hsv_b = rgb_to_hsv(b);
                            auto has_saturation = hsv.y != 0.0f;
                            hsv.y = lerp(hsv.y, hsv_b.y, t);
                            mixed = select(
                                a,
                                hsv_to_rgb(hsv),
                                has_saturation);
                            break;
                        }
                        case compiler::BlendOperation::color: {
                            auto hsv_b = rgb_to_hsv(b);
                            auto hsv = rgb_to_hsv(a);
                            hsv.x = hsv_b.x;
                            hsv.y = hsv_b.y;
                            auto recolored = hsv_to_rgb(hsv);
                            mixed = select(
                                a,
                                lerp(a, recolored, t),
                                hsv_b.y != 0.0f);
                            break;
                        }
                        case compiler::BlendOperation::value: {
                        auto hsv = rgb_to_hsv(a);
                        auto hsv_b = rgb_to_hsv(b);
                        hsv.z = lerp(hsv.z, hsv_b.z, t);
                        mixed = hsv_to_rgb(hsv);
                            break;
                        }
                    }
                    if ((instruction.static_u1 & 2u) != 0u) {
                        mixed = clamp(mixed, 0.0f, 1.0f);
                    }
                    value = make_float4(
                        mixed,
                        1.0f);
                    break;
                }
