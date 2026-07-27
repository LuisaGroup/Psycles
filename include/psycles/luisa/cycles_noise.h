#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_noise.h> through the Psycles::luisa target."
#endif

#include <cstdint>
#include <type_traits>

#include <psycles/luisa/surface.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_noise {

// This is a Luisa DSL lowering of Blender 4.5.10 Cycles' SVM Noise
// Texture. The hash, Perlin gradients, output scales, fractal recurrence,
// coordinate precision correction, distortion seeds, and color seeds are
// intentionally kept in the same order as Cycles so every Luisa backend
// executes the same material semantics.

enum class Type : std::uint32_t {
    multifractal,
    fbm,
    hybrid_multifractal,
    ridged_multifractal,
    hetero_terrain
};

struct Result {
    Float value;
    Float3 color;
};

[[nodiscard]] inline UInt rotate_left(
    UInt value,
    std::uint32_t amount) noexcept {
    return (value << amount) |
           (value >> (32u - amount));
}

inline void hash_final(UInt &a, UInt &b, UInt &c) noexcept {
    c ^= b;
    c -= rotate_left(b, 14u);
    a ^= c;
    a -= rotate_left(c, 11u);
    b ^= a;
    b -= rotate_left(a, 25u);
    c ^= b;
    c -= rotate_left(b, 16u);
    a ^= c;
    a -= rotate_left(c, 4u);
    b ^= a;
    b -= rotate_left(a, 14u);
    c ^= b;
    c -= rotate_left(b, 24u);
}

inline void hash_mix(UInt &a, UInt &b, UInt &c) noexcept {
    a -= c;
    a ^= rotate_left(c, 4u);
    c += b;
    b -= a;
    b ^= rotate_left(a, 6u);
    a += c;
    c -= b;
    c ^= rotate_left(b, 8u);
    b += a;
    a -= c;
    a ^= rotate_left(c, 16u);
    c += b;
    b -= a;
    b ^= rotate_left(a, 19u);
    a += c;
    c -= b;
    c ^= rotate_left(b, 4u);
    b += a;
}

[[nodiscard]] inline UInt hash_uint(UInt x) noexcept {
    UInt a = 0xdeadbeefu + (1u << 2u) + 13u;
    UInt b = a;
    UInt c = a;
    a += x;
    hash_final(a, b, c);
    return c;
}

[[nodiscard]] inline UInt hash_uint2(UInt x, UInt y) noexcept {
    UInt a = 0xdeadbeefu + (2u << 2u) + 13u;
    UInt b = a;
    UInt c = a;
    a += x;
    b += y;
    hash_final(a, b, c);
    return c;
}

[[nodiscard]] inline UInt hash_uint3(
    UInt x,
    UInt y,
    UInt z) noexcept {
    UInt a = 0xdeadbeefu + (3u << 2u) + 13u;
    UInt b = a;
    UInt c = a;
    a += x;
    b += y;
    c += z;
    hash_final(a, b, c);
    return c;
}

[[nodiscard]] inline UInt hash_uint4(
    UInt x,
    UInt y,
    UInt z,
    UInt w) noexcept {
    UInt a = 0xdeadbeefu + (4u << 2u) + 13u;
    UInt b = a;
    UInt c = a;
    a += x;
    b += y;
    c += z;
    hash_mix(a, b, c);
    a += w;
    hash_final(a, b, c);
    return c;
}

[[nodiscard]] inline Float uint_to_float_inclusive(
    UInt value) noexcept {
    return cast<float>(value) *
           (1.0f / static_cast<float>(0xffffffffu));
}

[[nodiscard]] inline Float hash_float2(Float2 value) noexcept {
    return uint_to_float_inclusive(hash_uint2(
        as<luisa::uint>(value.x),
        as<luisa::uint>(value.y)));
}

[[nodiscard]] inline Float hash_float(Float value) noexcept {
    return uint_to_float_inclusive(
        hash_uint(as<luisa::uint>(value)));
}

[[nodiscard]] inline Float hash_float3(Float3 value) noexcept {
    return uint_to_float_inclusive(hash_uint3(
        as<luisa::uint>(value.x),
        as<luisa::uint>(value.y),
        as<luisa::uint>(value.z)));
}

[[nodiscard]] inline Float hash_float4(Float4 value) noexcept {
    return uint_to_float_inclusive(hash_uint4(
        as<luisa::uint>(value.x),
        as<luisa::uint>(value.y),
        as<luisa::uint>(value.z),
        as<luisa::uint>(value.w)));
}

[[nodiscard]] inline Float3 hash_float_to_color(
    Float value) noexcept {
    return make_float3(
        hash_float(value),
        hash_float2(make_float2(value, 1.0f)),
        hash_float2(make_float2(value, 2.0f)));
}

[[nodiscard]] inline Float3 hash_float2_to_color(
    Float2 value) noexcept {
    return make_float3(
        hash_float2(value),
        hash_float3(make_float3(value, 1.0f)),
        hash_float3(make_float3(value, 2.0f)));
}

[[nodiscard]] inline Float3 hash_float3_to_color(
    Float3 value) noexcept {
    return make_float3(
        hash_float3(value),
        hash_float4(make_float4(value, 1.0f)),
        hash_float4(make_float4(value, 2.0f)));
}

[[nodiscard]] inline Float3 hash_float4_to_color(
    Float4 value) noexcept {
    return make_float3(
        hash_float4(value),
        hash_float4(make_float4(
            value.z, value.x, value.w, value.y)),
        hash_float4(make_float4(
            value.w, value.z, value.y, value.x)));
}

[[nodiscard]] inline Result evaluate_white(
    Float3 vector,
    Float w,
    std::uint32_t dimensions,
    bool color_needed) noexcept {
    Result result{
        .value = 0.0f,
        .color = make_float3(0.0f)};
    switch (dimensions) {
        case 1u:
            result.value = hash_float(w);
            if (color_needed) {
                result.color = hash_float_to_color(w);
            }
            break;
        case 2u: {
            const auto value =
                make_float2(vector.x, vector.y);
            result.value = hash_float2(value);
            if (color_needed) {
                result.color =
                    hash_float2_to_color(value);
            }
            break;
        }
        case 4u: {
            const auto value = make_float4(vector, w);
            result.value = hash_float4(value);
            if (color_needed) {
                result.color =
                    hash_float4_to_color(value);
            }
            break;
        }
        case 3u:
        default:
            result.value = hash_float3(vector);
            if (color_needed) {
                result.color =
                    hash_float3_to_color(vector);
            }
            break;
    }
    return result;
}

[[nodiscard]] inline Float random_offset(Float seed) noexcept {
    return 100.0f +
           uint_to_float_inclusive(
               hash_uint(as<luisa::uint>(seed))) *
               100.0f;
}

[[nodiscard]] inline Float2 random_offset2(Float seed) noexcept {
    return make_float2(
        100.0f + hash_float2(make_float2(seed, 0.0f)) * 100.0f,
        100.0f + hash_float2(make_float2(seed, 1.0f)) * 100.0f);
}

[[nodiscard]] inline Float3 random_offset3(Float seed) noexcept {
    return make_float3(
        100.0f + hash_float2(make_float2(seed, 0.0f)) * 100.0f,
        100.0f + hash_float2(make_float2(seed, 1.0f)) * 100.0f,
        100.0f + hash_float2(make_float2(seed, 2.0f)) * 100.0f);
}

[[nodiscard]] inline Float4 random_offset4(Float seed) noexcept {
    return make_float4(
        100.0f + hash_float2(make_float2(seed, 0.0f)) * 100.0f,
        100.0f + hash_float2(make_float2(seed, 1.0f)) * 100.0f,
        100.0f + hash_float2(make_float2(seed, 2.0f)) * 100.0f,
        100.0f + hash_float2(make_float2(seed, 3.0f)) * 100.0f);
}

[[nodiscard]] inline Float fade(Float value) noexcept {
    return value * value * value *
           (value * (value * 6.0f - 15.0f) + 10.0f);
}

[[nodiscard]] inline Float negate_if(
    Float value,
    Bool condition) noexcept {
    return select(value, -value, condition);
}

[[nodiscard]] inline Float gradient1(
    UInt hash,
    Float x) noexcept {
    auto h = hash & 15u;
    auto gradient = cast<float>(1u + (h & 7u));
    return negate_if(gradient, (h & 8u) != 0u) * x;
}

[[nodiscard]] inline Float gradient2(
    UInt hash,
    Float x,
    Float y) noexcept {
    auto h = hash & 7u;
    auto u = select(y, x, h < 4u);
    auto v = 2.0f * select(x, y, h < 4u);
    return negate_if(u, (h & 1u) != 0u) +
           negate_if(v, (h & 2u) != 0u);
}

[[nodiscard]] inline Float gradient3(
    UInt hash,
    Float x,
    Float y,
    Float z) noexcept {
    auto h = hash & 15u;
    auto u = select(y, x, h < 8u);
    auto alternate =
        select(z, x, (h == 12u) | (h == 14u));
    auto v = select(alternate, y, h < 4u);
    return negate_if(u, (h & 1u) != 0u) +
           negate_if(v, (h & 2u) != 0u);
}

[[nodiscard]] inline Float gradient4(
    UInt hash,
    Float x,
    Float y,
    Float z,
    Float w) noexcept {
    auto h = hash & 31u;
    auto u = select(y, x, h < 24u);
    auto v = select(z, y, h < 16u);
    auto s = select(w, z, h < 8u);
    return negate_if(u, (h & 1u) != 0u) +
           negate_if(v, (h & 2u) != 0u) +
           negate_if(s, (h & 4u) != 0u);
}

[[nodiscard]] inline Float bilinear(
    Float v0,
    Float v1,
    Float v2,
    Float v3,
    Float x,
    Float y) noexcept {
    auto one_minus_x = 1.0f - x;
    return (1.0f - y) *
               (v0 * one_minus_x + v1 * x) +
           y * (v2 * one_minus_x + v3 * x);
}

[[nodiscard]] inline Float trilinear(
    Float v0,
    Float v1,
    Float v2,
    Float v3,
    Float v4,
    Float v5,
    Float v6,
    Float v7,
    Float x,
    Float y,
    Float z) noexcept {
    auto one_minus_x = 1.0f - x;
    auto one_minus_y = 1.0f - y;
    auto one_minus_z = 1.0f - z;
    return one_minus_z *
               (one_minus_y *
                    (v0 * one_minus_x + v1 * x) +
                y * (v2 * one_minus_x + v3 * x)) +
           z *
               (one_minus_y *
                    (v4 * one_minus_x + v5 * x) +
                y * (v6 * one_minus_x + v7 * x));
}

[[nodiscard]] inline Float perlin(Float p) noexcept {
    auto floored = floor(p);
    auto x = cast<int>(floored);
    auto fx = p - floored;
    auto u = fade(fx);
    return lerp(
        gradient1(hash_uint(cast<luisa::uint>(x)), fx),
        gradient1(
            hash_uint(cast<luisa::uint>(x + 1)),
            fx - 1.0f),
        u);
}

[[nodiscard]] inline Float perlin(Float2 p) noexcept {
    auto floor_x = floor(p.x);
    auto floor_y = floor(p.y);
    auto x = cast<int>(floor_x);
    auto y = cast<int>(floor_y);
    auto fx = p.x - floor_x;
    auto fy = p.y - floor_y;
    auto u = fade(fx);
    auto v = fade(fy);
    auto x0 = cast<luisa::uint>(x);
    auto x1 = cast<luisa::uint>(x + 1);
    auto y0 = cast<luisa::uint>(y);
    auto y1 = cast<luisa::uint>(y + 1);
    return bilinear(
        gradient2(hash_uint2(x0, y0), fx, fy),
        gradient2(hash_uint2(x1, y0), fx - 1.0f, fy),
        gradient2(hash_uint2(x0, y1), fx, fy - 1.0f),
        gradient2(
            hash_uint2(x1, y1),
            fx - 1.0f,
            fy - 1.0f),
        u,
        v);
}

[[nodiscard]] inline Float perlin(Float3 p) noexcept {
    auto floor_x = floor(p.x);
    auto floor_y = floor(p.y);
    auto floor_z = floor(p.z);
    auto x = cast<int>(floor_x);
    auto y = cast<int>(floor_y);
    auto z = cast<int>(floor_z);
    auto fx = p.x - floor_x;
    auto fy = p.y - floor_y;
    auto fz = p.z - floor_z;
    auto u = fade(fx);
    auto v = fade(fy);
    auto w = fade(fz);
    auto x0 = cast<luisa::uint>(x);
    auto x1 = cast<luisa::uint>(x + 1);
    auto y0 = cast<luisa::uint>(y);
    auto y1 = cast<luisa::uint>(y + 1);
    auto z0 = cast<luisa::uint>(z);
    auto z1 = cast<luisa::uint>(z + 1);
    return trilinear(
        gradient3(hash_uint3(x0, y0, z0), fx, fy, fz),
        gradient3(
            hash_uint3(x1, y0, z0),
            fx - 1.0f,
            fy,
            fz),
        gradient3(
            hash_uint3(x0, y1, z0),
            fx,
            fy - 1.0f,
            fz),
        gradient3(
            hash_uint3(x1, y1, z0),
            fx - 1.0f,
            fy - 1.0f,
            fz),
        gradient3(
            hash_uint3(x0, y0, z1),
            fx,
            fy,
            fz - 1.0f),
        gradient3(
            hash_uint3(x1, y0, z1),
            fx - 1.0f,
            fy,
            fz - 1.0f),
        gradient3(
            hash_uint3(x0, y1, z1),
            fx,
            fy - 1.0f,
            fz - 1.0f),
        gradient3(
            hash_uint3(x1, y1, z1),
            fx - 1.0f,
            fy - 1.0f,
            fz - 1.0f),
        u,
        v,
        w);
}

[[nodiscard]] inline Float perlin(Float4 p) noexcept {
    auto floor_x = floor(p.x);
    auto floor_y = floor(p.y);
    auto floor_z = floor(p.z);
    auto floor_w = floor(p.w);
    auto x = cast<int>(floor_x);
    auto y = cast<int>(floor_y);
    auto z = cast<int>(floor_z);
    auto w = cast<int>(floor_w);
    auto fx = p.x - floor_x;
    auto fy = p.y - floor_y;
    auto fz = p.z - floor_z;
    auto fw = p.w - floor_w;
    auto u = fade(fx);
    auto v = fade(fy);
    auto t = fade(fz);
    auto s = fade(fw);
    auto x0 = cast<luisa::uint>(x);
    auto x1 = cast<luisa::uint>(x + 1);
    auto y0 = cast<luisa::uint>(y);
    auto y1 = cast<luisa::uint>(y + 1);
    auto z0 = cast<luisa::uint>(z);
    auto z1 = cast<luisa::uint>(z + 1);
    auto w0 = cast<luisa::uint>(w);
    auto w1 = cast<luisa::uint>(w + 1);
    auto sample = [&](UInt sx,
                      UInt sy,
                      UInt sz,
                      UInt sw,
                      Float dx,
                      Float dy,
                      Float dz,
                      Float dw) noexcept {
        return gradient4(
            hash_uint4(sx, sy, sz, sw),
            dx,
            dy,
            dz,
            dw);
    };
    auto lower = trilinear(
        sample(x0, y0, z0, w0, fx, fy, fz, fw),
        sample(x1, y0, z0, w0, fx - 1.0f, fy, fz, fw),
        sample(x0, y1, z0, w0, fx, fy - 1.0f, fz, fw),
        sample(
            x1,
            y1,
            z0,
            w0,
            fx - 1.0f,
            fy - 1.0f,
            fz,
            fw),
        sample(x0, y0, z1, w0, fx, fy, fz - 1.0f, fw),
        sample(
            x1,
            y0,
            z1,
            w0,
            fx - 1.0f,
            fy,
            fz - 1.0f,
            fw),
        sample(
            x0,
            y1,
            z1,
            w0,
            fx,
            fy - 1.0f,
            fz - 1.0f,
            fw),
        sample(
            x1,
            y1,
            z1,
            w0,
            fx - 1.0f,
            fy - 1.0f,
            fz - 1.0f,
            fw),
        u,
        v,
        t);
    auto upper = trilinear(
        sample(x0, y0, z0, w1, fx, fy, fz, fw - 1.0f),
        sample(
            x1,
            y0,
            z0,
            w1,
            fx - 1.0f,
            fy,
            fz,
            fw - 1.0f),
        sample(
            x0,
            y1,
            z0,
            w1,
            fx,
            fy - 1.0f,
            fz,
            fw - 1.0f),
        sample(
            x1,
            y1,
            z0,
            w1,
            fx - 1.0f,
            fy - 1.0f,
            fz,
            fw - 1.0f),
        sample(
            x0,
            y0,
            z1,
            w1,
            fx,
            fy,
            fz - 1.0f,
            fw - 1.0f),
        sample(
            x1,
            y0,
            z1,
            w1,
            fx - 1.0f,
            fy,
            fz - 1.0f,
            fw - 1.0f),
        sample(
            x0,
            y1,
            z1,
            w1,
            fx,
            fy - 1.0f,
            fz - 1.0f,
            fw - 1.0f),
        sample(
            x1,
            y1,
            z1,
            w1,
            fx - 1.0f,
            fy - 1.0f,
            fz - 1.0f,
            fw - 1.0f),
        u,
        v,
        t);
    return lerp(lower, upper, s);
}

[[nodiscard]] inline Float signed_noise(Float p) noexcept {
    auto correction = select(
        0.0f, 0.5f, abs(p) >= 1000000.0f);
    return 0.25f *
           perlin(fmod(p, 100000.0f) + correction);
}

[[nodiscard]] inline Float signed_noise(Float2 p) noexcept {
    auto correction = make_float2(
        select(0.0f, 0.5f, abs(p.x) >= 1000000.0f),
        select(0.0f, 0.5f, abs(p.y) >= 1000000.0f));
    return 0.6616f *
           perlin(
               fmod(p, make_float2(100000.0f)) +
               correction);
}

[[nodiscard]] inline Float signed_noise(Float3 p) noexcept {
    auto correction = make_float3(
        select(0.0f, 0.5f, abs(p.x) >= 1000000.0f),
        select(0.0f, 0.5f, abs(p.y) >= 1000000.0f),
        select(0.0f, 0.5f, abs(p.z) >= 1000000.0f));
    return 0.9820f *
           perlin(
               fmod(p, make_float3(100000.0f)) +
               correction);
}

[[nodiscard]] inline Float signed_noise(Float4 p) noexcept {
    auto correction = make_float4(
        select(0.0f, 0.5f, abs(p.x) >= 1000000.0f),
        select(0.0f, 0.5f, abs(p.y) >= 1000000.0f),
        select(0.0f, 0.5f, abs(p.z) >= 1000000.0f),
        select(0.0f, 0.5f, abs(p.w) >= 1000000.0f));
    return 0.8344f *
           perlin(
               fmod(p, make_float4(100000.0f)) +
               correction);
}

template<typename P>
[[nodiscard]] inline Float fbm(
    P p,
    Float detail,
    Float roughness,
    Float lacunarity,
    bool normalize) noexcept {
    Float frequency = 1.0f;
    Float amplitude = 1.0f;
    Float maximum_amplitude = 0.0f;
    Float sum = 0.0f;
    auto whole_octaves =
        cast<luisa::uint>(floor(detail)) + 1u;
    $for (octave, whole_octaves) {
        static_cast<void>(octave);
        sum += signed_noise(frequency * p) * amplitude;
        maximum_amplitude += amplitude;
        amplitude *= roughness;
        frequency *= lacunarity;
    };
    auto result = normalize
                      ? 0.5f * sum /
                                max(maximum_amplitude, 1.0e-20f) +
                            0.5f
                      : sum;
    auto remainder = detail - floor(detail);
    $if (remainder != 0.0f) {
        auto extended_sum =
            sum + signed_noise(frequency * p) * amplitude;
        auto extended_result =
            normalize
                ? 0.5f * extended_sum /
                          max(
                              maximum_amplitude + amplitude,
                              1.0e-20f) +
                      0.5f
                : extended_sum;
        result = lerp(result, extended_result, remainder);
    };
    return result;
}

template<typename P>
[[nodiscard]] inline Float multifractal(
    P p,
    Float detail,
    Float roughness,
    Float lacunarity) noexcept {
    Float value = 1.0f;
    Float power = 1.0f;
    auto whole_octaves =
        cast<luisa::uint>(floor(detail)) + 1u;
    $for (octave, whole_octaves) {
        static_cast<void>(octave);
        value *= power * signed_noise(p) + 1.0f;
        power *= roughness;
        p *= lacunarity;
    };
    auto remainder = detail - floor(detail);
    $if (remainder != 0.0f) {
        value *=
            remainder * power * signed_noise(p) + 1.0f;
    };
    return value;
}

template<typename P>
[[nodiscard]] inline Float hetero_terrain(
    P p,
    Float detail,
    Float roughness,
    Float lacunarity,
    Float offset) noexcept {
    Float power = roughness;
    Float value = offset + signed_noise(p);
    p *= lacunarity;
    auto whole_octaves =
        cast<luisa::uint>(floor(detail));
    $for (octave, whole_octaves) {
        static_cast<void>(octave);
        auto increment =
            (signed_noise(p) + offset) * power * value;
        value += increment;
        power *= roughness;
        p *= lacunarity;
    };
    auto remainder = detail - floor(detail);
    $if (remainder != 0.0f) {
        auto increment =
            (signed_noise(p) + offset) * power * value;
        value += remainder * increment;
    };
    return value;
}

template<typename P>
[[nodiscard]] inline Float hybrid_multifractal(
    P p,
    Float detail,
    Float roughness,
    Float lacunarity,
    Float offset,
    Float gain) noexcept {
    Float power = 1.0f;
    Float value = 0.0f;
    Float weight = 1.0f;
    auto whole_octaves =
        cast<luisa::uint>(floor(detail)) + 1u;
    $for (octave, whole_octaves) {
        static_cast<void>(octave);
        $if (weight > 0.001f) {
            weight = min(weight, 1.0f);
            auto signal =
                (signed_noise(p) + offset) * power;
            power *= roughness;
            value += weight * signal;
            weight *= gain * signal;
            p *= lacunarity;
        };
    };
    auto remainder = detail - floor(detail);
    $if ((remainder != 0.0f) & (weight > 0.001f)) {
        weight = min(weight, 1.0f);
        auto signal = (signed_noise(p) + offset) * power;
        value += remainder * weight * signal;
    };
    return value;
}

template<typename P>
[[nodiscard]] inline Float ridged_multifractal(
    P p,
    Float detail,
    Float roughness,
    Float lacunarity,
    Float offset,
    Float gain) noexcept {
    Float power = roughness;
    Float signal = offset - abs(signed_noise(p));
    signal *= signal;
    Float value = signal;
    auto remaining_octaves =
        cast<luisa::uint>(floor(detail));
    $for (octave, remaining_octaves) {
        static_cast<void>(octave);
        p *= lacunarity;
        auto weight = clamp(signal * gain, 0.0f, 1.0f);
        signal = offset - abs(signed_noise(p));
        signal *= signal;
        signal *= weight;
        value += signal * power;
        power *= roughness;
    };
    return value;
}

template<typename P>
[[nodiscard]] inline Float select_fractal(
    P p,
    Float detail,
    Float roughness,
    Float lacunarity,
    Float offset,
    Float gain,
    Type type,
    bool normalize) noexcept {
    switch (type) {
        case Type::multifractal:
            return multifractal(
                p, detail, roughness, lacunarity);
        case Type::fbm:
            return fbm(
                p,
                detail,
                roughness,
                lacunarity,
                normalize);
        case Type::hybrid_multifractal:
            return hybrid_multifractal(
                p,
                detail,
                roughness,
                lacunarity,
                offset,
                gain);
        case Type::ridged_multifractal:
            return ridged_multifractal(
                p,
                detail,
                roughness,
                lacunarity,
                offset,
                gain);
        case Type::hetero_terrain:
            return hetero_terrain(
                p,
                detail,
                roughness,
                lacunarity,
                offset);
    }
    return 0.0f;
}

template<typename P, typename Offset>
[[nodiscard]] inline Result texture(
    P p,
    Float detail,
    Float roughness,
    Float lacunarity,
    Float offset,
    Float gain,
    Float distortion,
    Type type,
    bool normalize,
    bool color_needed,
    Offset offset_function,
    float first_color_seed) noexcept {
    $if (distortion != 0.0f) {
        if constexpr (std::is_same_v<P, Float>) {
            p += signed_noise(
                     p + offset_function(0.0f)) *
                 distortion;
        } else {
            P displacement = p;
            if constexpr (std::is_same_v<P, Float2>) {
                displacement = make_float2(
                    signed_noise(
                        p + offset_function(0.0f)),
                    signed_noise(
                        p + offset_function(1.0f)));
            } else if constexpr (
                std::is_same_v<P, Float3>) {
                displacement = make_float3(
                    signed_noise(
                        p + offset_function(0.0f)),
                    signed_noise(
                        p + offset_function(1.0f)),
                    signed_noise(
                        p + offset_function(2.0f)));
            } else {
                displacement = make_float4(
                    signed_noise(
                        p + offset_function(0.0f)),
                    signed_noise(
                        p + offset_function(1.0f)),
                    signed_noise(
                        p + offset_function(2.0f)),
                    signed_noise(
                        p + offset_function(3.0f)));
            }
            p += displacement * distortion;
        }
    };
    auto value = select_fractal(
        p,
        detail,
        roughness,
        lacunarity,
        offset,
        gain,
        type,
        normalize);
    Float3 color = make_float3(value);
    if (color_needed) {
        color.y = select_fractal(
            p + offset_function(first_color_seed),
            detail,
            roughness,
            lacunarity,
            offset,
            gain,
            type,
            normalize);
        color.z = select_fractal(
            p + offset_function(first_color_seed + 1.0f),
            detail,
            roughness,
            lacunarity,
            offset,
            gain,
            type,
            normalize);
    }
    return {.value = value, .color = color};
}

[[nodiscard]] inline Result evaluate(
    Float3 vector,
    Float w,
    Float detail,
    Float roughness,
    Float lacunarity,
    Float offset,
    Float gain,
    Float distortion,
    std::uint32_t dimensions,
    Type type,
    bool normalize,
    bool color_needed) noexcept {
    detail = clamp(detail, 0.0f, 15.0f);
    roughness = max(roughness, 0.0f);
    switch (dimensions) {
        case 1u:
            return texture(
                w,
                detail,
                roughness,
                lacunarity,
                offset,
                gain,
                distortion,
                type,
                normalize,
                color_needed,
                [](Float seed) noexcept {
                    return random_offset(seed);
                },
                1.0f);
        case 2u:
            {
            Float2 coordinate = vector.xy();
            return texture(
                coordinate,
                detail,
                roughness,
                lacunarity,
                offset,
                gain,
                distortion,
                type,
                normalize,
                color_needed,
                [](Float seed) noexcept {
                    return random_offset2(seed);
                },
                2.0f);
            }
        case 4u:
            {
            Float4 coordinate = make_float4(vector, w);
            return texture(
                coordinate,
                detail,
                roughness,
                lacunarity,
                offset,
                gain,
                distortion,
                type,
                normalize,
                color_needed,
                [](Float seed) noexcept {
                    return random_offset4(seed);
                },
                4.0f);
            }
        default:
            return texture(
                vector,
                detail,
                roughness,
                lacunarity,
                offset,
                gain,
                distortion,
                type,
                normalize,
                color_needed,
                [](Float seed) noexcept {
                    return random_offset3(seed);
                },
                3.0f);
    }
}

}// namespace psycles::luisa_backend::cycles_noise
