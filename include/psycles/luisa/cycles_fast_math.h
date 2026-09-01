#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_fast_math.h> through the Psycles::luisa target."
#endif

#include <limits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_fast_math {

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float inverse_pi = 0.31830988618379067154f;
inline constexpr float half_pi = 1.57079632679489661923f;
inline constexpr float quarter_pi = 0.78539816339744830962f;
inline constexpr float ln_two = 0.69314718055994530942f;

struct SineCosine {
    luisa::compute::Float sine;
    luisa::compute::Float cosine;
};

// These functions are part of the Cycles numerical contract used by its
// closures and fitted volume phase functions. They are expressed as Luisa DSL
// operations so fallback, HIP, and Vulkan execute the same range reduction
// and polynomial, instead of inheriting three different native "fast math"
// implementations.

[[nodiscard]] inline luisa::compute::Int
round_to_int(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    return cast<int>(
        value + copysign(0.5f, value));
}

[[nodiscard]] inline luisa::compute::Float
sine(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    const auto quadrant =
        round_to_int(value * inverse_pi);
    const auto quadrant_float = cast<float>(quadrant);
    value = quadrant_float * (-0.78515625f * 4.0f) +
            value;
    value = quadrant_float *
                (-0.00024187564849853515625f * 4.0f) +
            value;
    value = quadrant_float *
                (-3.7747668102383613586e-08f * 4.0f) +
            value;
    value = quadrant_float *
                (-1.2816720341285448015e-12f * 4.0f) +
            value;
    value = half_pi - (half_pi - value);
    const auto squared = value * value;
    value = select(
        value,
        -value,
        (quadrant & 1) != 0);
    Float polynomial =
        2.6083159809786593541503e-06f;
    polynomial =
        polynomial * squared -
        0.0001981069071916863322258f;
    polynomial =
        polynomial * squared +
        0.00833307858556509017944336f;
    polynomial =
        polynomial * squared -
        0.166666597127914428710938f;
    polynomial =
        squared * (polynomial * value) +
        value;
    return clamp(polynomial, -1.0f, 1.0f);
}

[[nodiscard]] inline luisa::compute::Float
cosine(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    const auto quadrant =
        round_to_int(value * inverse_pi);
    const auto quadrant_float = cast<float>(quadrant);
    value = quadrant_float * (-0.78515625f * 4.0f) +
            value;
    value = quadrant_float *
                (-0.00024187564849853515625f * 4.0f) +
            value;
    value = quadrant_float *
                (-3.7747668102383613586e-08f * 4.0f) +
            value;
    value = quadrant_float *
                (-1.2816720341285448015e-12f * 4.0f) +
            value;
    value = half_pi - (half_pi - value);
    const auto squared = value * value;
    Float polynomial =
        -2.71811842367242206819355e-07f;
    polynomial =
        polynomial * squared +
        2.47990446951007470488548e-05f;
    polynomial =
        polynomial * squared -
        0.00138888787478208541870117f;
    polynomial =
        polynomial * squared +
        0.0416666641831398010253906f;
    polynomial = polynomial * squared - 0.5f;
    polynomial = polynomial * squared + 1.0f;
    polynomial = select(
        polynomial,
        -polynomial,
        (quadrant & 1) != 0);
    return clamp(polynomial, -1.0f, 1.0f);
}

[[nodiscard]] inline SineCosine
sine_cosine(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    const auto quadrant =
        round_to_int(value * inverse_pi);
    const auto quadrant_float = cast<float>(quadrant);
    value = quadrant_float * (-0.78515625f * 4.0f) +
            value;
    value = quadrant_float *
                (-0.00024187564849853515625f * 4.0f) +
            value;
    value = quadrant_float *
                (-3.7747668102383613586e-08f * 4.0f) +
            value;
    value = quadrant_float *
                (-1.2816720341285448015e-12f * 4.0f) +
            value;
    value = half_pi - (half_pi - value);
    const auto squared = value * value;
    value = select(
        value,
        -value,
        (quadrant & 1) != 0);

    Float sine_polynomial =
        2.6083159809786593541503e-06f;
    sine_polynomial =
        sine_polynomial * squared -
        0.0001981069071916863322258f;
    sine_polynomial =
        sine_polynomial * squared +
        0.00833307858556509017944336f;
    sine_polynomial =
        sine_polynomial * squared -
        0.166666597127914428710938f;
    sine_polynomial =
        squared * (sine_polynomial * value) +
        value;

    Float cosine_polynomial =
        -2.71811842367242206819355e-07f;
    cosine_polynomial =
        cosine_polynomial * squared +
        2.47990446951007470488548e-05f;
    cosine_polynomial =
        cosine_polynomial * squared -
        0.00138888787478208541870117f;
    cosine_polynomial =
        cosine_polynomial * squared +
        0.0416666641831398010253906f;
    cosine_polynomial =
        cosine_polynomial * squared - 0.5f;
    cosine_polynomial =
        cosine_polynomial * squared + 1.0f;
    cosine_polynomial = select(
        cosine_polynomial,
        -cosine_polynomial,
        (quadrant & 1) != 0);
    return {
        .sine = clamp(sine_polynomial, -1.0f, 1.0f),
        .cosine = clamp(cosine_polynomial, -1.0f, 1.0f)};
}

[[nodiscard]] inline luisa::compute::Float
arc_cosine(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    const auto magnitude = abs(value);
    const auto clamped = select(
        1.0f,
        1.0f - (1.0f - magnitude),
        magnitude < 1.0f);
    const auto approximation =
        sqrt(1.0f - clamped) *
        (1.5707963267f +
         clamped *
             (-0.213300989f +
              clamped *
                  (0.077980478f +
                   clamped * -0.02164095f)));
    return select(
        approximation,
        pi - approximation,
        value < 0.0f);
}

[[nodiscard]] inline luisa::compute::Float
arc_tangent2(luisa::compute::Float y,
             luisa::compute::Float x) noexcept {
    using namespace luisa::compute;
    const auto abs_x = abs(x);
    const auto abs_y = abs(y);
    const auto ratio = select(
        abs_y / abs_x,
        abs_x / abs_y,
        abs_y > abs_x);
    const auto nonzero_ratio = select(
        ratio,
        1.0f,
        abs_x == abs_y);
    const auto k = select(
        nonzero_ratio,
        0.0f,
        abs_y == 0.0f);
    const auto crushed = 1.0f - (1.0f - k);
    const auto squared = crushed * crushed;
    Float result =
        crushed * (0.43157974f * squared + 1.0f) /
        (((0.05831938f * squared + 0.76443945f) *
          squared) +
         1.0f);
    result = select(
        result,
        half_pi - result,
        abs_y > abs_x);
    result = select(
        result,
        pi - result,
        (as<uint>(x) & 0x80000000u) != 0u);
    return copysign(result, y);
}

[[nodiscard]] inline luisa::compute::Float
safe_arc_sine(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    return asin(clamp(value, -1.0f, 1.0f));
}

[[nodiscard]] inline luisa::compute::Float
tangent(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    const auto quadrant =
        round_to_int(value * (2.0f * inverse_pi));
    const auto quadrant_float = cast<float>(quadrant);
    value = quadrant_float * (-0.78515625f * 2.0f) +
            value;
    value = quadrant_float *
                (-0.00024187564849853515625f * 2.0f) +
            value;
    value = quadrant_float *
                (-3.7747668102383613586e-08f * 2.0f) +
            value;
    value = quadrant_float *
                (-1.2816720341285448015e-12f * 2.0f) +
            value;
    value = select(
        value,
        quarter_pi - (quarter_pi - value),
        (quadrant & 1) == 0);
    const auto squared = value * value;
    Float polynomial =
        0.00927245803177356719970703f;
    polynomial =
        polynomial * squared +
        0.00331984995864331722259521f;
    polynomial =
        polynomial * squared +
        0.0242998078465461730957031f;
    polynomial =
        polynomial * squared +
        0.0534495301544666290283203f;
    polynomial =
        polynomial * squared +
        0.133383005857467651367188f;
    polynomial =
        polynomial * squared +
        0.333331853151321411132812f;
    polynomial =
        squared * (polynomial * value) +
        value;
    return select(
        polynomial,
        -1.0f / polynomial,
        (quadrant & 1) != 0);
}

[[nodiscard]] inline luisa::compute::Float
log2(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    value = clamp(
        value,
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max());
    const auto bits = as<uint>(value);
    const auto exponent =
        cast<int>(bits >> 23u) - 127;
    const auto fraction =
        as<float>(
            (bits & 0x007fffffu) |
            0x3f800000u) -
        1.0f;
    const auto fraction2 = fraction * fraction;
    const auto fraction4 = fraction2 * fraction2;
    Float high =
        fraction * -0.00931049621349f +
        0.05206469089414f;
    Float low =
        fraction * 0.47868480909345f -
        0.72116591947498f;
    high = fraction * high - 0.13753123777116f;
    high = fraction * high + 0.24187369696082f;
    high = fraction * high - 0.34730547155299f;
    low = fraction * low + 1.442689881667200f;
    return fraction4 * high +
           fraction * low +
           cast<float>(exponent);
}

[[nodiscard]] inline luisa::compute::Float
log(luisa::compute::Float value) noexcept {
    return log2(value) * ln_two;
}

[[nodiscard]] inline luisa::compute::Float
exp2(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    value = clamp(value, -126.0f, 126.0f);
    const auto exponent = cast<int>(value);
    value -= cast<float>(exponent);
    value = 1.0f - (1.0f - value);
    Float polynomial = 1.33336498402e-3f;
    polynomial =
        value * polynomial +
        9.810352697968e-3f;
    polynomial =
        value * polynomial +
        5.551834031939e-2f;
    polynomial =
        value * polynomial +
        0.2401793301105f;
    polynomial =
        value * polynomial +
        0.693144857883f;
    polynomial = value * polynomial + 1.0f;
    const auto exponent_bits =
        cast<uint>(exponent) << 23u;
    return as<float>(
        as<uint>(polynomial) + exponent_bits);
}

[[nodiscard]] inline luisa::compute::Float
exp(luisa::compute::Float value) noexcept {
    return exp2(value / ln_two);
}

[[nodiscard]] inline luisa::compute::Float
inverse_cube_root(luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    Float result = as<float>(
        0x54a24242 -
        as<int>(value) / 3);
    result =
        (2.0f / 3.0f) * result +
        1.0f /
            (3.0f * result * result * value);
    result =
        (2.0f / 3.0f) * result +
        1.0f /
            (3.0f * result * result * value);
    return result;
}

}// namespace psycles::luisa_backend::cycles_fast_math
