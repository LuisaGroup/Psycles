#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_volume_phase.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/cycles_fast_math.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_volume_phase {

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float inverse_pi = 0.31830988618379067154f;
inline constexpr float inverse_four_pi =
    0.07957747154594766788f;
inline constexpr float cube_root_two =
    1.25992104989487316477f;

enum class Type : std::uint32_t {
    henyey_greenstein,
    fournier_forand,
    draine,
    rayleigh
};

struct Closure {
    Type type{Type::henyey_greenstein};
    luisa::compute::Float3 parameters;
};

struct Sample {
    luisa::compute::Float3 direction;
    luisa::compute::Float pdf;
};

struct MieParameters {
    luisa::compute::Float henyey_greenstein_g;
    luisa::compute::Float draine_g;
    luisa::compute::Float draine_alpha;
    luisa::compute::Float draine_weight;
};

[[nodiscard]] inline luisa::compute::Float
clamp_anisotropy(luisa::compute::Float anisotropy) noexcept {
    using namespace luisa::compute;
    return sign(anisotropy) *
           min(abs(anisotropy), 1.0f - 1.0e-3f);
}

[[nodiscard]] inline Closure henyey_greenstein(
    luisa::compute::Float anisotropy) noexcept {
    using namespace luisa::compute;
    return {
        .type = Type::henyey_greenstein,
        .parameters = make_float3(
            clamp_anisotropy(anisotropy),
            0.0f,
            0.0f)};
}

[[nodiscard]] inline Closure rayleigh() noexcept {
    using namespace luisa::compute;
    return {
        .type = Type::rayleigh,
        .parameters = make_float3(0.0f)};
}

[[nodiscard]] inline Closure draine(
    luisa::compute::Float anisotropy,
    luisa::compute::Float alpha) noexcept {
    using namespace luisa::compute;
    return {
        .type = Type::draine,
        .parameters = make_float3(
            clamp_anisotropy(anisotropy),
            alpha,
            0.0f)};
}

[[nodiscard]] inline luisa::compute::Float3
fournier_forand_coefficients(
    luisa::compute::Float backscatter,
    luisa::compute::Float ior) noexcept {
    using namespace luisa::compute;
    backscatter =
        min(abs(backscatter), 0.5f - 1.0e-3f);
    ior = max(ior, 1.0f + 1.0e-3f);
    const auto ior_difference = ior - 1.0f;
    const auto delta_90 =
        2.0f /
        (3.0f * ior_difference * ior_difference);
    const auto delta_180 =
        4.0f /
        (3.0f * ior_difference * ior_difference);
    const auto exponent =
        -log(
            2.0f * backscatter *
                    (delta_90 - 1.0f) +
                1.0f) /
        log(delta_90);
    return make_float3(
        ior,
        exponent,
        (pow(delta_180, -exponent) - 1.0f) /
            (delta_180 - 1.0f));
}

[[nodiscard]] inline Closure fournier_forand(
    luisa::compute::Float backscatter,
    luisa::compute::Float ior) noexcept {
    return {
        .type = Type::fournier_forand,
        .parameters =
            fournier_forand_coefficients(
                backscatter, ior)};
}

[[nodiscard]] inline luisa::compute::Float
henyey_greenstein_pdf(
    luisa::compute::Float cosine,
    luisa::compute::Float anisotropy) noexcept {
    using namespace luisa::compute;
    Float result = inverse_four_pi;
    $if(abs(anisotropy) >= 1.0e-3f) {
        const auto factor =
            1.0f +
            anisotropy *
                (anisotropy - 2.0f * cosine);
        result =
            (1.0f - anisotropy * anisotropy) /
            (4.0f * pi * factor *
             sqrt(max(factor, 0.0f)));
    };
    return result;
}

[[nodiscard]] inline luisa::compute::Float
rayleigh_pdf(luisa::compute::Float cosine) noexcept {
    return (0.1875f * inverse_pi) *
           (1.0f + cosine * cosine);
}

[[nodiscard]] inline luisa::compute::Float
draine_pdf(
    luisa::compute::Float cosine,
    luisa::compute::Float anisotropy,
    luisa::compute::Float alpha) noexcept {
    using namespace luisa::compute;
    Float result = 0.0f;
    $if((abs(anisotropy) < 1.0e-3f) &
        (alpha > 0.999f)) {
        result = rayleigh_pdf(cosine);
    }
    $elif(abs(alpha) < 1.0e-3f) {
        result = henyey_greenstein_pdf(
            cosine, anisotropy);
    }
    $else {
        const auto g2 = anisotropy * anisotropy;
        const auto factor =
            1.0f + g2 -
            2.0f * anisotropy * cosine;
        result =
            ((1.0f - g2) *
             (1.0f + alpha * cosine * cosine)) /
            ((1.0f +
              alpha * (1.0f + 2.0f * g2) /
                  3.0f) *
             4.0f * pi * factor *
             sqrt(factor));
    };
    return result;
}

[[nodiscard]] inline luisa::compute::Float
fournier_forand_delta(
    luisa::compute::Float ior,
    luisa::compute::Float half_angle_sine_squared) noexcept {
    return 4.0f * half_angle_sine_squared /
           (3.0f * (ior - 1.0f) *
            (ior - 1.0f));
}

[[nodiscard]] inline luisa::compute::Float
fournier_forand_pdf_impl(
    luisa::compute::Float cosine,
    luisa::compute::Float delta,
    luisa::compute::Float delta_power,
    luisa::compute::Float exponent,
    luisa::compute::Float half_angle_sine_squared,
    luisa::compute::Float correction) noexcept {
    using namespace luisa::compute;
    const auto one_minus_delta = 1.0f - delta;
    const auto one_minus_power = 1.0f - delta_power;
    Float result = 0.0f;
    $if(abs(one_minus_delta) < 1.0e-3f) {
        result =
            exponent *
                ((exponent - 1.0f) -
                 (exponent + 1.0f) /
                     half_angle_sine_squared) /
                (8.0f * pi) +
            exponent * (exponent + 1.0f) *
                one_minus_delta *
                (2.0f * (exponent - 1.0f) -
                 (2.0f * exponent + 1.0f) /
                     half_angle_sine_squared) /
                (24.0f * pi);
    }
    $else {
        result =
            (exponent * one_minus_delta -
             one_minus_power +
             (delta * one_minus_power -
              exponent * one_minus_delta) /
                 half_angle_sine_squared) /
            (4.0f * pi *
             one_minus_delta * one_minus_delta *
             delta_power);
    };
    return result +
           correction *
               (3.0f * cosine * cosine - 1.0f);
}

[[nodiscard]] inline luisa::compute::Float
fournier_forand_pdf(
    luisa::compute::Float cosine,
    luisa::compute::Float3 coefficients) noexcept {
    using namespace luisa::compute;
    Float result = 0.0f;
    $if(abs(cosine) < 1.0f) {
        const auto half_angle_sine_squared =
            0.5f * (1.0f - cosine);
        const auto delta = fournier_forand_delta(
            coefficients.x,
            half_angle_sine_squared);
        result = fournier_forand_pdf_impl(
            cosine,
            delta,
            pow(delta, coefficients.y),
            coefficients.y,
            half_angle_sine_squared,
            coefficients.z / (16.0f * pi));
    };
    return result;
}

[[nodiscard]] inline luisa::compute::Float
evaluate(
    const Closure &closure,
    luisa::compute::Float cosine) noexcept {
    switch (closure.type) {
        case Type::henyey_greenstein:
            return henyey_greenstein_pdf(
                cosine, closure.parameters.x);
        case Type::fournier_forand:
            return fournier_forand_pdf(
                cosine, closure.parameters);
        case Type::draine:
            return draine_pdf(
                cosine,
                closure.parameters.x,
                closure.parameters.y);
        case Type::rayleigh:
            return rayleigh_pdf(cosine);
    }
    return luisa::compute::Float{0.0f};
}

[[nodiscard]] inline luisa::compute::Float3
sample_direction(
    luisa::compute::Float3 axis,
    luisa::compute::Float cosine,
    luisa::compute::Float azimuth_sample) noexcept {
    using namespace luisa::compute;
    const auto phi = 2.0f * pi * azimuth_sample;
    const auto sine =
        sqrt(max(1.0f - cosine * cosine, 0.0f));
    const auto local = make_float3(
        sine * cos(phi),
        sine * sin(phi),
        cosine);
    const auto basis =
        cycles_sample_mapping::make_orthonormals(axis);
    return local.x * basis.tangent +
           local.y * basis.bitangent +
           local.z * axis;
}

[[nodiscard]] inline luisa::compute::Float
sample_henyey_greenstein_cosine(
    luisa::compute::Float anisotropy,
    luisa::compute::Float random) noexcept {
    using namespace luisa::compute;
    Float cosine =
        1.0f - 2.0f * random;
    $if(abs(anisotropy) >= 1.0e-3f) {
        const auto ratio =
            (1.0f - anisotropy * anisotropy) /
            (1.0f - anisotropy * cosine);
        cosine =
            (1.0f + anisotropy * anisotropy -
             ratio * ratio) /
            (2.0f * anisotropy);
    };
    return cosine;
}

[[nodiscard]] inline luisa::compute::Float
sample_rayleigh_cosine(
    luisa::compute::Float random) noexcept {
    using namespace luisa::compute;
    const auto a = 2.0f - 4.0f * random;
    const auto inverse_u =
        -cycles_fast_math::inverse_cube_root(
            sqrt(1.0f + a * a) + a);
    return 1.0f / inverse_u - inverse_u;
}

[[nodiscard]] inline luisa::compute::Float
sample_draine_cosine(
    luisa::compute::Float anisotropy,
    luisa::compute::Float alpha,
    luisa::compute::Float random) noexcept {
    using namespace luisa::compute;
    Float cosine = 0.0f;
    $if(abs(anisotropy) < 1.0e-2f) {
        const auto inverse_alpha = 1.0f / alpha;
        const auto b2 =
            (3.0f + alpha) * inverse_alpha *
            (0.5f - random);
        const auto inverse_u =
            -cycles_fast_math::inverse_cube_root(
                b2 +
                sqrt(
                    b2 * b2 +
                    inverse_alpha *
                        inverse_alpha *
                        inverse_alpha));
        cosine =
            1.0f / inverse_u -
            inverse_u / alpha;
    }
    $else {
        const auto g2 = anisotropy * anisotropy;
        const auto g3 = anisotropy * g2;
        const auto g4 = g2 * g2;
        const auto g6 = g2 * g4;
        const auto g_plus_one_squared =
            (1.0f + g2) * (1.0f + g2);
        const auto t1a = alpha * (g4 - 1.0f);
        const auto t1a3 = t1a * t1a * t1a;
        const auto t2 =
            -1296.0f * (g2 - 1.0f) *
            (alpha - alpha * g2) * t1a *
            (4.0f * g2 +
             alpha * g_plus_one_squared);
        const auto t9 =
            2.0f + g2 +
            g3 * (1.0f + 2.0f * g2) *
                (2.0f * random - 1.0f);
        const auto t3 =
            3.0f * g2 *
                (1.0f +
                 anisotropy *
                     (2.0f * random - 1.0f)) +
            alpha * t9;
        const auto t4a =
            432.0f * t1a3 + t2 +
            432.0f * alpha * (1.0f - g2) *
                t3 * t3;
        const auto t10 =
            alpha * (2.0f * g4 - g2 - g6);
        const auto t4b = 144.0f * t10;
        const auto t4 =
            t4a +
            sqrt(
                -4.0f * t4b * t4b * t4b +
                t4a * t4a);
        const auto inverse_t4_cubed =
            cycles_fast_math::inverse_cube_root(t4);
        const auto t8 =
            48.0f * cube_root_two * t10;
        const auto t6 =
            (2.0f * t1a +
             t8 * inverse_t4_cubed +
             1.0f /
                 (3.0f * cube_root_two *
                  inverse_t4_cubed)) /
            (alpha * (1.0f - g2));
        const auto t5 =
            6.0f * (1.0f + g2) + t6;
        const auto t7 =
            6.0f * (1.0f + g2) -
            8.0f * t3 /
                (alpha * (g2 - 1.0f) *
                 sqrt(t5)) -
            t6;
        const auto root_difference =
            sqrt(t7) - sqrt(t5);
        cosine =
            (1.0f + g2 -
             0.25f *
                 root_difference *
                 root_difference) /
            (2.0f * anisotropy);
    };
    return cosine;
}

[[nodiscard]] inline luisa::compute::Float
sample_fournier_forand_cosine(
    luisa::compute::Float random,
    luisa::compute::Float3 coefficients) noexcept {
    using namespace luisa::compute;
    const auto ior = coefficients.x;
    const auto exponent = coefficients.y;
    const auto cdf_correction =
        coefficients.z / 8.0f;
    const auto pdf_correction =
        coefficients.z / (16.0f * pi);
    Float cosine = 0.64278760968f;
    $for(iteration, 20u) {
        const auto half_angle_sine_squared =
            0.5f * (1.0f - cosine);
        const auto delta = fournier_forand_delta(
            ior, half_angle_sine_squared);
        const auto delta_power =
            pow(delta, exponent);
        const auto one_minus_delta =
            1.0f - delta;
        const auto one_minus_power =
            1.0f - delta_power;
        Float cdf = 0.0f;
        $if(abs(one_minus_delta) < 1.0e-3f) {
            cdf =
                1.0f +
                exponent *
                    (1.0f -
                     half_angle_sine_squared) *
                    (1.0f -
                     0.5f *
                         (exponent + 1.0f) *
                         one_minus_delta);
        }
        $else {
            cdf =
                (1.0f -
                 delta_power * delta -
                 one_minus_power *
                     half_angle_sine_squared) /
                (one_minus_delta * delta_power);
        };
        cdf += cdf_correction * cosine *
               (1.0f - cosine * cosine);
        const auto pdf =
            fournier_forand_pdf_impl(
                cosine,
                delta,
                delta_power,
                exponent,
                half_angle_sine_squared,
                pdf_correction);
        auto next =
            cosine +
            (0.5f * inverse_pi) *
                (cdf - random) / pdf;
        next = select(
            next,
            max(
                lerp(cosine, 1.0f, 0.5f),
                0.99f),
            next >= 1.0f);
        const auto done =
            (abs(cosine - next) < 1.0e-6f) |
            (next == 1.0f);
        cosine = next;
        $if(done) {
            $break;
        };
    };
    return cosine;
}

[[nodiscard]] inline Sample sample(
    const Closure &closure,
    luisa::compute::Float3 axis,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    Float cosine = 0.0f;
    switch (closure.type) {
        case Type::henyey_greenstein:
            cosine =
                sample_henyey_greenstein_cosine(
                    closure.parameters.x,
                    random.x);
            break;
        case Type::fournier_forand:
            cosine =
                sample_fournier_forand_cosine(
                    random.x,
                    closure.parameters);
            break;
        case Type::draine: {
            $if((abs(closure.parameters.x) <
                 1.0e-3f) &
                (closure.parameters.y > 0.999f)) {
                cosine =
                    sample_rayleigh_cosine(
                        random.x);
            }
            $elif(abs(closure.parameters.y) <
                  1.0e-3f) {
                cosine =
                    sample_henyey_greenstein_cosine(
                        closure.parameters.x,
                        random.x);
            }
            $else {
                cosine = sample_draine_cosine(
                    closure.parameters.x,
                    closure.parameters.y,
                    random.x);
            };
            break;
        }
        case Type::rayleigh:
            cosine =
                sample_rayleigh_cosine(random.x);
            break;
    }
    return {
        .direction =
            sample_direction(
                axis, cosine, random.y),
        .pdf = evaluate(closure, cosine)};
}

[[nodiscard]] inline luisa::compute::Float evaluate(
    luisa::compute::UInt type,
    luisa::compute::Float3 parameters,
    luisa::compute::Float cosine) noexcept {
    using namespace luisa::compute;
    Float result = 0.0f;
    $if(type ==
        static_cast<std::uint32_t>(
            Type::henyey_greenstein)) {
        result = evaluate(
            Closure{
                .type = Type::henyey_greenstein,
                .parameters = parameters},
            cosine);
    }
    $elif(type ==
          static_cast<std::uint32_t>(
              Type::fournier_forand)) {
        result = evaluate(
            Closure{
                .type = Type::fournier_forand,
                .parameters = parameters},
            cosine);
    }
    $elif(type ==
          static_cast<std::uint32_t>(
              Type::draine)) {
        result = evaluate(
            Closure{
                .type = Type::draine,
                .parameters = parameters},
            cosine);
    }
    $elif(type ==
          static_cast<std::uint32_t>(
              Type::rayleigh)) {
        result = evaluate(
            Closure{
                .type = Type::rayleigh,
                .parameters = parameters},
            cosine);
    };
    return result;
}

[[nodiscard]] inline Sample sample(
    luisa::compute::UInt type,
    luisa::compute::Float3 parameters,
    luisa::compute::Float3 axis,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    Float3 direction = axis;
    Float pdf = 0.0f;
    const auto assign =
        [&](Type static_type) noexcept {
            const auto result = sample(
                Closure{
                    .type = static_type,
                    .parameters = parameters},
                axis,
                random);
            direction = result.direction;
            pdf = result.pdf;
        };
    $if(type ==
        static_cast<std::uint32_t>(
            Type::henyey_greenstein)) {
        assign(Type::henyey_greenstein);
    }
    $elif(type ==
          static_cast<std::uint32_t>(
              Type::fournier_forand)) {
        assign(Type::fournier_forand);
    }
    $elif(type ==
          static_cast<std::uint32_t>(
              Type::draine)) {
        assign(Type::draine);
    }
    $elif(type ==
          static_cast<std::uint32_t>(
              Type::rayleigh)) {
        assign(Type::rayleigh);
    };
    return {
        .direction = direction,
        .pdf = pdf};
}

[[nodiscard]] inline luisa::compute::Float sampled_roughness(
    luisa::compute::UInt type,
    luisa::compute::Float3 parameters) noexcept {
    using namespace luisa::compute;
    Float anisotropy = 0.0f;
    $if(type ==
            static_cast<std::uint32_t>(
                Type::fournier_forand)) {
        anisotropy = 1.0f;
    }
    $elif((type ==
           static_cast<std::uint32_t>(
               Type::henyey_greenstein)) |
          (type ==
           static_cast<std::uint32_t>(
               Type::draine))) {
        anisotropy = parameters.x;
    };
    return 1.0f - abs(anisotropy);
}

[[nodiscard]] inline MieParameters mie_parameters(
    luisa::compute::Float diameter) noexcept {
    using namespace luisa::compute;
    diameter = max(diameter, 0.0f);
    Float henyey_greenstein_g = 0.0f;
    Float draine_g = 0.0f;
    Float draine_alpha = 0.0f;
    Float draine_weight = 0.0f;
    $if(diameter <= 0.1f) {
        henyey_greenstein_g =
            13.8f * diameter * diameter;
        draine_g =
            1.1456f * diameter *
            cycles_fast_math::sine(
                9.29044f * diameter);
        draine_alpha = 250.0f;
        draine_weight =
            0.252977f -
            312.983f * pow(diameter, 4.3f);
    }
    $elif(diameter < 1.5f) {
        const auto logarithm =
            cycles_fast_math::log(diameter);
        henyey_greenstein_g =
            0.862f -
            0.143f * logarithm * logarithm;
        const auto a =
            (logarithm - 0.238604f) *
            (logarithm + 1.00667f);
        const auto b =
            0.507522f -
            0.15677f * logarithm;
        const auto c =
            1.19692f *
                cycles_fast_math::cosine(a / b) +
            1.37932f * logarithm +
            0.0625835f;
        draine_g =
            0.379685f *
                cycles_fast_math::cosine(c) +
            0.344213f;
        draine_alpha = 250.0f;
        draine_weight =
            0.146209f *
                cycles_fast_math::cosine(
                    3.38707f * logarithm +
                    2.11193f) +
            0.316072f +
            0.0778917f * logarithm;
    }
    $elif(diameter < 5.0f) {
        const auto logarithm =
            cycles_fast_math::log(diameter);
        const auto log_logarithm =
            cycles_fast_math::log(logarithm);
        henyey_greenstein_g =
            0.0604931f * log_logarithm +
            0.940256f;
        draine_g =
            0.500411f -
            0.081287f /
                (-2.0f * logarithm +
                 cycles_fast_math::tangent(
                     logarithm) +
                 1.27551f);
        draine_alpha =
            7.30354f * logarithm +
            6.31675f;
        const auto temporary =
            cycles_fast_math::cosine(
                5.68947f *
                (log_logarithm - 0.0292149f));
        draine_weight =
            0.026914f *
                (logarithm - temporary) +
            0.3764f;
    }
    $else {
        henyey_greenstein_g =
            cycles_fast_math::exp(
                -0.0990567f /
                (diameter - 1.67154f));
        draine_g =
            cycles_fast_math::exp(
                -2.20679f /
                    (diameter + 3.91029f) -
                0.428934f);
        draine_alpha =
            cycles_fast_math::exp(
                3.62489f -
                8.29288f /
                    (diameter + 5.52825f));
        draine_weight =
            cycles_fast_math::exp(
                -0.599085f /
                    (diameter - 0.641583f) -
                0.665888f);
    };
    return {
        .henyey_greenstein_g =
            henyey_greenstein_g,
        .draine_g = draine_g,
        .draine_alpha = draine_alpha,
        .draine_weight = draine_weight};
}

}// namespace psycles::luisa_backend::cycles_volume_phase
