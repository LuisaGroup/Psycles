#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/graph_surface.h> through the Psycles::luisa target."
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/surface.h>

#include <luisa/core/stl/vector.h>

namespace psycles::luisa_backend {

class GraphSurface final : public Surface {

private:
    static constexpr float pi = 3.14159265358979323846f;
    static constexpr float inverse_pi = 0.31830988618379067154f;
    static constexpr float two_pi = 6.28318530717958647692f;
    static constexpr std::uint32_t camera_ray_visibility = 1u << 0u;
    static constexpr std::uint32_t diffuse_ray_visibility = 1u << 1u;
    static constexpr std::uint32_t glossy_ray_visibility = 1u << 2u;
    static constexpr std::uint32_t transmission_ray_visibility = 1u << 3u;
    static constexpr std::uint32_t shadow_ray_visibility = 1u << 4u;
    static constexpr std::uint32_t volume_ray_visibility = 1u << 5u;

    struct TracedValues {
        luisa::vector<Float4> values;
        Float3 shading_normal;
    };

    struct TracedClosure {
        compiler::ClosureOperation operation{
            compiler::ClosureOperation::diffuse};
        Float3 weight;
        Float3 color;
        Float3 normal;
        Float roughness;
        Float diffuse_roughness;
        Float metallic;
        Float ior;
        Float specular_ior_level;
        Float3 specular_tint;
        bool preserve_ggx_energy{};
    };

    struct AdjustedIor {
        Float eta;
        Float f0;
    };

    struct GgxEnergy {
        Float3 darkening;
        Float3 evaluation_scale;
    };

    struct PrincipledState {
        Float eta;
        Float3 dielectric_f0;
        Float3 metallic_f0;
        Float3 metallic_b;
        Float3 dielectric_evaluation_scale;
        Float3 metallic_evaluation_scale;
        Float3 diffuse_sample_weight;
        Float3 glossy_sample_weight;
        Float3 diffuse_albedo;
    };

private:
    std::shared_ptr<const compiler::SurfaceProgram> _program;
    SurfaceCapabilities _capabilities;

private:
    template<typename Id, typename Values>
    [[nodiscard]] static const auto &get(
        Id id,
        const Values &values) noexcept {
        return values[id.value];
    }

    [[nodiscard]] static Float scalar(
        compiler::ValueExpressionId id,
        const TracedValues &values) noexcept {
        return get(id, values.values).x;
    }

    [[nodiscard]] static Float3 vector(
        compiler::ValueExpressionId id,
        const TracedValues &values) noexcept {
        return get(id, values.values).xyz();
    }

    [[nodiscard]] static Float sample_weight(Float3 value) noexcept {
        return (abs(value.x) + abs(value.y) + abs(value.z)) /
               3.0f;
    }

    [[nodiscard]] static Float max_component(
        Float3 value) noexcept {
        return max(value.x, max(value.y, value.z));
    }

    [[nodiscard]] static Float srgb_to_linear(
        Float value) noexcept {
        auto linear_segment =
            max(value, 0.0f) * (1.0f / 12.92f);
        auto power_segment = pow(
            (value + 0.055f) * (1.0f / 1.055f),
            2.4f);
        return select(
            power_segment,
            linear_segment,
            value < 0.04045f);
    }

    [[nodiscard]] static Float3 srgb_to_linear(
        Float3 value) noexcept {
        return make_float3(
            srgb_to_linear(value.x),
            srgb_to_linear(value.y),
            srgb_to_linear(value.z));
    }

    [[nodiscard]] static Float cycles_table_1d(
        const ShaderServices &services,
        Float x,
        Expr<std::uint32_t> offset,
        std::uint32_t size) noexcept {
        auto coordinate =
            clamp(x, 0.0f, 1.0f) *
            static_cast<float>(size - 1u);
        auto index = min(
            cast<luisa::uint>(coordinate),
            size - 1u);
        auto next = min(index + 1u, size - 1u);
        auto t = coordinate - cast<float>(index);
        auto data0 =
            services.cycles_bsdf_data(index + offset);
        auto data1 =
            services.cycles_bsdf_data(next + offset);
        return lerp(data0, data1, t);
    }

    [[nodiscard]] static Float cycles_table_2d(
        const ShaderServices &services,
        Float x,
        Float y,
        Expr<std::uint32_t> offset,
        std::uint32_t x_size,
        std::uint32_t y_size) noexcept {
        auto coordinate =
            clamp(y, 0.0f, 1.0f) *
            static_cast<float>(y_size - 1u);
        auto index = min(
            cast<luisa::uint>(coordinate),
            y_size - 1u);
        auto next = min(index + 1u, y_size - 1u);
        auto t = coordinate - cast<float>(index);
        auto data0 = cycles_table_1d(
            services,
            x,
            offset + x_size * index,
            x_size);
        auto data1 = cycles_table_1d(
            services,
            x,
            offset + x_size * next,
            x_size);
        return lerp(data0, data1, t);
    }

    [[nodiscard]] static Float cycles_table_3d(
        const ShaderServices &services,
        Float x,
        Float y,
        Float z,
        Expr<std::uint32_t> offset,
        std::uint32_t x_size,
        std::uint32_t y_size,
        std::uint32_t z_size) noexcept {
        auto coordinate =
            clamp(z, 0.0f, 1.0f) *
            static_cast<float>(z_size - 1u);
        auto index = min(
            cast<luisa::uint>(coordinate),
            z_size - 1u);
        auto next = min(index + 1u, z_size - 1u);
        auto t = coordinate - cast<float>(index);
        auto slice_stride = x_size * y_size;
        auto data0 = cycles_table_2d(
            services,
            x,
            y,
            offset + slice_stride * index,
            x_size,
            y_size);
        auto data1 = cycles_table_2d(
            services,
            x,
            y,
            offset + slice_stride * next,
            x_size,
            y_size);
        return lerp(data0, data1, t);
    }

    [[nodiscard]] static Float3 safe_normalize(
        Float3 value,
        Float3 fallback) noexcept {
        auto valid = dot(value, value) > 1.0e-20f;
        auto selected = select(fallback, value, valid);
        auto fallback_valid =
            dot(selected, selected) > 1.0e-20f;
        selected = select(
            make_float3(0.0f, 0.0f, 1.0f),
            selected,
            fallback_valid);
        return normalize(selected);
    }

    [[nodiscard]] static Float3 rgb_to_hsv(
        Float3 rgb) noexcept {
        auto cmax = max(rgb.x, max(rgb.y, rgb.z));
        auto cmin = min(rgb.x, min(rgb.y, rgb.z));
        auto delta = cmax - cmin;
        auto saturation = select(
            0.0f,
            delta /
                select(1.0f, cmax, cmax != 0.0f),
            cmax != 0.0f);
        auto safe_delta =
            select(1.0f, delta, delta != 0.0f);
        auto c = (make_float3(cmax) - rgb) / safe_delta;
        auto hue = 4.0f + c.y - c.x;
        hue = select(
            hue,
            2.0f + c.x - c.z,
            rgb.y == cmax);
        hue = select(
            hue,
            c.z - c.y,
            rgb.x == cmax);
        hue /= 6.0f;
        hue = select(hue, hue + 1.0f, hue < 0.0f);
        hue = select(0.0f, hue, saturation != 0.0f);
        return make_float3(hue, saturation, cmax);
    }

    [[nodiscard]] static Float3 hsv_to_rgb(
        Float3 hsv) noexcept {
        auto h = select(hsv.x, 0.0f, hsv.x == 1.0f);
        h *= 6.0f;
        auto sector = floor(h);
        auto f = h - sector;
        auto p = hsv.z * (1.0f - hsv.y);
        auto q = hsv.z * (1.0f - hsv.y * f);
        auto t = hsv.z *
                 (1.0f - hsv.y * (1.0f - f));
        auto rgb = make_float3(hsv.z, p, q);
        rgb = select(
            rgb,
            make_float3(t, p, hsv.z),
            sector == 4.0f);
        rgb = select(
            rgb,
            make_float3(p, q, hsv.z),
            sector == 3.0f);
        rgb = select(
            rgb,
            make_float3(p, hsv.z, t),
            sector == 2.0f);
        rgb = select(
            rgb,
            make_float3(q, hsv.z, p),
            sector == 1.0f);
        rgb = select(
            rgb,
            make_float3(hsv.z, t, p),
            sector == 0.0f);
        return select(
            make_float3(hsv.z),
            rgb,
            hsv.y != 0.0f);
    }

    [[nodiscard]] static Float3 rgb_to_hsl(
        Float3 rgb) noexcept {
        auto cmax = max(rgb.x, max(rgb.y, rgb.z));
        auto cmin = min(rgb.x, min(rgb.y, rgb.z));
        auto lightness = min(
            1.0f, (cmax + cmin) * 0.5f);
        auto delta = cmax - cmin;
        auto chromatic = cmax != cmin;
        auto denominator = select(
            cmax + cmin,
            2.0f - cmax - cmin,
            lightness > 0.5f);
        auto saturation = select(
            0.0f,
            delta /
                select(
                    1.0f,
                    denominator,
                    abs(denominator) > 1.0e-20f),
            chromatic);
        auto safe_delta = select(
            1.0f, delta, abs(delta) > 1.0e-20f);
        auto hue =
            (rgb.x - rgb.y) / safe_delta + 4.0f;
        hue = select(
            hue,
            (rgb.z - rgb.x) / safe_delta + 2.0f,
            cmax == rgb.y);
        hue = select(
            hue,
            (rgb.y - rgb.z) / safe_delta +
                select(0.0f, 6.0f, rgb.y < rgb.z),
            cmax == rgb.x);
        hue = select(0.0f, hue / 6.0f, chromatic);
        return make_float3(hue, saturation, lightness);
    }

    [[nodiscard]] static Float3 hsl_to_rgb(
        Float3 hsl) noexcept {
        auto hue6 = hsl.x * 6.0f;
        auto nr = clamp(
            abs(hue6 - 3.0f) - 1.0f,
            0.0f,
            1.0f);
        auto ng = clamp(
            2.0f - abs(hue6 - 2.0f),
            0.0f,
            1.0f);
        auto nb = clamp(
            2.0f - abs(hue6 - 4.0f),
            0.0f,
            1.0f);
        auto chroma =
            (1.0f - abs(2.0f * hsl.z - 1.0f)) *
            hsl.y;
        return make_float3(
            (nr - 0.5f) * chroma + hsl.z,
            (ng - 0.5f) * chroma + hsl.z,
            (nb - 0.5f) * chroma + hsl.z);
    }

    [[nodiscard]] static Float3 separate_color(
        Float3 color,
        std::uint64_t mode) noexcept {
        return mode == 1u
                   ? rgb_to_hsv(color)
                   : mode == 2u ? rgb_to_hsl(color) : color;
    }

    [[nodiscard]] static Float3 combine_color(
        Float3 channels,
        std::uint64_t mode) noexcept {
        return mode == 1u
                   ? hsv_to_rgb(channels)
                   : mode == 2u ? hsl_to_rgb(channels) : channels;
    }

    [[nodiscard]] static Float fresnel_dielectric_cos(
        Float cosine,
        Float eta) noexcept {
        auto c = abs(cosine);
        auto g_squared = eta * eta - 1.0f + c * c;
        auto g = sqrt(max(g_squared, 0.0f));
        auto a = (g - c) / max(g + c, 1.0e-20f);
        auto b =
            (c * (g + c) - 1.0f) /
            select(
                1.0e-20f,
                c * (g - c) + 1.0f,
                abs(c * (g - c) + 1.0f) >
                    1.0e-20f);
        auto regular =
            0.5f * a * a * (1.0f + b * b);
        return select(1.0f, regular, g_squared > 0.0f);
    }

    [[nodiscard]] static Float f0_from_ior(
        Float ior) noexcept {
        auto ratio =
            (ior - 1.0f) / max(ior + 1.0f, 1.0e-20f);
        return ratio * ratio;
    }

    [[nodiscard]] static Float ior_from_f0(
        Float f0) noexcept {
        auto root = sqrt(clamp(f0, 0.0f, 0.99f));
        return (1.0f + root) / max(1.0f - root, 1.0e-20f);
    }

    [[nodiscard]] static Float fresnel_dielectric_fss(
        Float eta) noexcept {
        auto below_one =
            0.997118f +
            eta *
                (0.1014f -
                 eta * (0.965241f + eta * 0.130607f));
        auto above_one =
            (eta - 1.0f) /
            max(4.08567f + 1.00071f * eta, 1.0e-20f);
        return select(above_one, below_one, eta < 1.0f);
    }

    [[nodiscard]] static AdjustedIor adjusted_ior(
        const TracedClosure &closure) noexcept {
        auto original_eta = max(closure.ior, 1.0e-5f);
        auto original_f0 = f0_from_ior(original_eta);
        auto adjusted_f0 =
            original_f0 *
            (2.0f * max(closure.specular_ior_level, 0.0f));
        auto eta_from_adjusted = ior_from_f0(adjusted_f0);
        eta_from_adjusted = select(
            eta_from_adjusted,
            1.0f / max(eta_from_adjusted, 1.0e-20f),
            original_eta < 1.0f);
        auto should_adjust =
            closure.specular_ior_level != 0.5f;
        return {
            .eta = select(
                original_eta,
                eta_from_adjusted,
                should_adjust),
            .f0 = select(
                original_f0,
                adjusted_f0,
                should_adjust)};
    }

    [[nodiscard]] static Float3 generalized_dielectric_fresnel(
        Float cosine,
        Float eta,
        Float3 f0) noexcept {
        auto real_fresnel =
            fresnel_dielectric_cos(cosine, eta);
        auto real_f0 = f0_from_ior(eta);
        auto interpolation = clamp(
            (real_fresnel - real_f0) /
                max(1.0f - real_f0, 1.0e-20f),
            0.0f,
            1.0f);
        return lerp(f0, make_float3(1.0f), interpolation);
    }

    [[nodiscard]] static Float3 fresnel_f82_b(
        Float3 f0,
        Float3 tint) noexcept {
        constexpr float f = 6.0f / 7.0f;
        constexpr float f5 = f * f * f * f * f;
        auto schlick =
            lerp(f0, make_float3(1.0f), f5);
        return schlick *
               (7.0f / (f5 * f)) *
               (make_float3(1.0f) - tint);
    }

    [[nodiscard]] static Float3 fresnel_f82(
        Float cosine,
        Float3 f0,
        Float3 b) noexcept {
        auto mu = clamp(1.0f - cosine, 0.0f, 1.0f);
        auto mu2 = mu * mu;
        auto mu5 = mu2 * mu2 * mu;
        auto schlick =
            lerp(f0, make_float3(1.0f), mu5);
        return clamp(
            schlick - b * cosine * mu5 * mu,
            make_float3(0.0f),
            make_float3(1.0f));
    }

    [[nodiscard]] static GgxEnergy ggx_energy(
        const ShaderServices &services,
        const TracedClosure &closure,
        Float incoming_cosine,
        Float3 fss) noexcept {
        if (!closure.preserve_ggx_energy) {
            return {
                .darkening = make_float3(1.0f),
                .evaluation_scale = make_float3(1.0f)};
        }

        auto roughness =
            clamp(closure.roughness, 0.0f, 1.0f);
        auto energy = max(
            cycles_table_2d(
                services,
                roughness,
                incoming_cosine,
                UInt{cycles45_tables::ggx_e_offset},
                32u,
                32u),
            1.0e-20f);
        auto average_energy = cycles_table_1d(
            services,
            roughness,
            UInt{cycles45_tables::ggx_eavg_offset},
            32u);
        auto missing_factor =
            (1.0f - energy) / energy;
        auto energy_scale = 1.0f / energy;
        auto fms =
            fss * average_energy /
            max(
                make_float3(1.0f) -
                    fss * (1.0f - average_energy),
                make_float3(1.0e-20f));
        auto darkening =
            (make_float3(1.0f) +
             fms * missing_factor) /
            energy_scale;
        return {
            .darkening = darkening,
            .evaluation_scale =
                darkening * energy_scale};
    }

    [[nodiscard]] static PrincipledState principled_state(
        const ShaderServices &services,
        const TracedClosure &closure,
        Float3 incoming) noexcept {
        auto adjusted = adjusted_ior(closure);
        auto tint = max(
            closure.specular_tint,
            make_float3(0.0f));
        auto dielectric_f0 = clamp(
            make_float3(adjusted.f0) * tint,
            make_float3(0.0f),
            make_float3(1.0f));
        auto metallic_f0 = clamp(
            closure.color,
            make_float3(0.0f),
            make_float3(1.0f));
        auto metallic_tint = min(
            tint, make_float3(1.0f));
        auto metallic_b = fresnel_f82_b(
            metallic_f0, metallic_tint);

        auto real_f0 = f0_from_ior(adjusted.eta);
        auto real_fss =
            fresnel_dielectric_fss(adjusted.eta);
        auto fss_interpolation = clamp(
            (real_fss - real_f0) /
                max(1.0f - real_f0, 1.0e-20f),
            0.0f,
            1.0f);
        auto dielectric_fss = lerp(
            dielectric_f0,
            make_float3(1.0f),
            fss_interpolation);
        auto metallic_fss =
            lerp(
                metallic_f0,
                make_float3(1.0f),
                1.0f / 21.0f) -
            metallic_b * (1.0f / 126.0f);

        auto incoming_cosine = clamp(
            dot(closure.normal, incoming), 0.0f, 1.0f);
        auto dielectric_energy = ggx_energy(
            services,
            closure,
            incoming_cosine,
            dielectric_fss);
        auto metallic_energy = ggx_energy(
            services,
            closure,
            incoming_cosine,
            metallic_fss);

        auto roughness =
            clamp(closure.roughness, 0.0f, 1.0f);
        auto dielectric_z = sqrt(abs(
            (adjusted.eta - 1.0f) /
            max(adjusted.eta + 1.0f, 1.0e-20f)));
        auto dielectric_s = cycles_table_3d(
            services,
            roughness,
            incoming_cosine,
            dielectric_z,
            UInt{
                cycles45_tables::
                    ggx_gen_schlick_ior_s_offset},
            16u,
            16u,
            16u);
        auto metallic_s = cycles_table_3d(
            services,
            roughness,
            incoming_cosine,
            0.5f,
            UInt{
                cycles45_tables::
                    ggx_gen_schlick_s_offset},
            16u,
            16u,
            16u);
        auto dielectric_albedo = lerp(
            dielectric_f0,
            make_float3(1.0f),
            dielectric_s);
        auto metallic_albedo = lerp(
            metallic_f0,
            make_float3(1.0f),
            metallic_s);
        auto lower_layer_factor = clamp(
            1.0f -
                max_component(
                    dielectric_energy.darkening *
                    dielectric_albedo),
            0.0f,
            1.0f);
        auto metallic =
            clamp(closure.metallic, 0.0f, 1.0f);
        auto dielectric_weight = 1.0f - metallic;
        auto diffuse_albedo =
            closure.weight *
            max(closure.color, make_float3(0.0f)) *
            dielectric_weight *
            lower_layer_factor;
        auto glossy_sample_weight =
            closure.weight *
            (metallic *
                 metallic_energy.darkening *
                 metallic_albedo +
             dielectric_weight *
                 dielectric_energy.darkening *
                 dielectric_albedo);
        return {
            .eta = adjusted.eta,
            .dielectric_f0 = dielectric_f0,
            .metallic_f0 = metallic_f0,
            .metallic_b = metallic_b,
            .dielectric_evaluation_scale =
                dielectric_energy.evaluation_scale,
            .metallic_evaluation_scale =
                metallic_energy.evaluation_scale,
            .diffuse_sample_weight = diffuse_albedo,
            .glossy_sample_weight = glossy_sample_weight,
            .diffuse_albedo = diffuse_albedo};
    }

    [[nodiscard]] static Float oren_nayar_g(
        Float cosine) noexcept {
        auto c = clamp(cosine, 0.0f, 1.0f);
        auto sine = sqrt(max(1.0f - c * c, 0.0f));
        auto theta = acos(c);
        auto safe_cosine = max(c, 1.0e-6f);
        auto regular =
            sine * (theta - 2.0f / 3.0f - sine * c) +
            2.0f / 3.0f * (sine / safe_cosine) *
                (1.0f - sine * sine * sine);
        auto taylor = (pi * 0.5f - 2.0f / 3.0f) - c;
        return select(regular, taylor, c < 1.0e-6f);
    }

    [[nodiscard]] static Float3 diffuse_intensity(
        const TracedClosure &closure,
        Float3 incoming,
        Float3 outgoing) noexcept {
        auto nl = max(dot(closure.normal, outgoing), 0.0f);
        auto lambert = make_float3(nl * inverse_pi);

        auto sigma = clamp(
            closure.operation ==
                    compiler::ClosureOperation::principled
                ? closure.diffuse_roughness
                : closure.roughness,
            0.0f,
            1.0f);
        auto a = 1.0f /
                 (pi + sigma * (pi * 0.5f - 2.0f / 3.0f));
        auto b = sigma * a;
        auto nv = max(dot(closure.normal, incoming), 0.0f);
        auto t = dot(outgoing, incoming) - nl * nv;
        auto positive_t = t > 0.0f;
        t = select(
            t,
            t / (max(nl, nv) + 1.17549435e-38f),
            positive_t);

        auto single_scatter = a + b * t;
        auto albedo = clamp(
            closure.color,
            make_float3(0.0f),
            make_float3(1.0f));
        auto e_average =
            a * pi + ((two_pi - 5.6f) / 3.0f) * b;
        auto albedo_squared = albedo * albedo;
        auto e_ms =
            inverse_pi * albedo_squared *
            (e_average / (1.0f - e_average)) /
            (make_float3(1.0f) -
             albedo * (1.0f - e_average));
        auto e_incoming =
            a * pi + b * oren_nayar_g(nv);
        auto multiscatter_term =
            e_ms * (1.0f - e_incoming);
        auto e_outgoing =
            a * pi + b * oren_nayar_g(nl);
        auto oren_nayar =
            nl * (make_float3(single_scatter) +
                  multiscatter_term * (1.0f - e_outgoing));

        return select(
            oren_nayar,
            lambert,
            sigma < 1.0e-5f);
    }

    [[nodiscard]] static Float ggx_distribution(
        Float n_dot_h,
        Float alpha) noexcept {
        auto alpha2 = alpha * alpha;
        auto denominator =
            n_dot_h * n_dot_h * (alpha2 - 1.0f) + 1.0f;
        return alpha2 /
               max(pi * denominator * denominator, 1.0e-20f);
    }

    [[nodiscard]] static Float smith_g1(
        Float n_dot_v,
        Float alpha) noexcept {
        auto cosine = max(n_dot_v, 1.0e-6f);
        auto tangent2 = max(
            1.0f / (cosine * cosine) - 1.0f,
            0.0f);
        auto lambda =
            0.5f *
            (sqrt(
                 1.0f +
                 alpha * alpha * tangent2) -
             1.0f);
        return 1.0f / (1.0f + lambda);
    }

    [[nodiscard]] static Float3 specular_f0(
        const TracedClosure &closure) noexcept {
        auto dielectric =
            (closure.ior - 1.0f) /
            max(closure.ior + 1.0f, 1.0e-20f);
        dielectric *= dielectric;
        return lerp(
            make_float3(dielectric),
            clamp(
                closure.color,
                make_float3(0.0f),
                make_float3(1.0f)),
            closure.metallic);
    }

    [[nodiscard]] static Float3 microfacet_intensity(
        const ShaderServices &services,
        const TracedClosure &closure,
        Float3 incoming,
        Float3 outgoing) noexcept {
        auto n_dot_v =
            max(dot(closure.normal, incoming), 0.0f);
        auto n_dot_l =
            max(dot(closure.normal, outgoing), 0.0f);
        auto half_vector = safe_normalize(
            incoming + outgoing,
            closure.normal);
        auto n_dot_h =
            max(dot(closure.normal, half_vector), 0.0f);
        auto v_dot_h =
            max(dot(incoming, half_vector), 0.0f);
        auto alpha = max(
            closure.roughness * closure.roughness,
            1.0e-3f);
        auto distribution =
            ggx_distribution(n_dot_h, alpha);
        auto lambda_v =
            1.0f / smith_g1(n_dot_v, alpha) - 1.0f;
        auto lambda_l =
            1.0f / smith_g1(n_dot_l, alpha) - 1.0f;
        auto geometry =
            1.0f / (1.0f + lambda_v + lambda_l);
        Float3 fresnel;
        if (closure.operation ==
            compiler::ClosureOperation::principled) {
            auto state =
                principled_state(services, closure, incoming);
            auto dielectric_fresnel =
                generalized_dielectric_fresnel(
                    v_dot_h,
                    state.eta,
                    state.dielectric_f0);
            auto metallic_fresnel = fresnel_f82(
                v_dot_h,
                state.metallic_f0,
                state.metallic_b);
            auto metallic =
                clamp(closure.metallic, 0.0f, 1.0f);
            fresnel =
                metallic *
                    metallic_fresnel *
                    state.metallic_evaluation_scale +
                (1.0f - metallic) *
                    dielectric_fresnel *
                    state.dielectric_evaluation_scale;
        } else {
            auto f0 = specular_f0(closure);
            fresnel =
                f0 +
                (make_float3(1.0f) - f0) *
                    pow(1.0f - v_dot_h, 5.0f);
        }
        auto intensity =
            fresnel * distribution * geometry /
            max(4.0f * n_dot_v, 1.0e-20f);
        return select(
            make_float3(0.0f),
            intensity,
            (n_dot_v > 0.0f) &
                (n_dot_l > 0.0f) &
                (n_dot_h > 0.0f) &
                (v_dot_h > 0.0f));
    }

    [[nodiscard]] static Float microfacet_pdf(
        const TracedClosure &closure,
        Float3 incoming,
        Float3 outgoing) noexcept {
        auto half_vector = safe_normalize(
            incoming + outgoing,
            closure.normal);
        auto n_dot_h =
            max(dot(closure.normal, half_vector), 0.0f);
        auto v_dot_h =
            max(dot(incoming, half_vector), 0.0f);
        auto alpha = max(
            closure.roughness * closure.roughness,
            1.0e-3f);
        auto n_dot_v =
            max(dot(closure.normal, incoming), 0.0f);
        auto n_dot_l =
            max(dot(closure.normal, outgoing), 0.0f);
        auto pdf =
            ggx_distribution(n_dot_h, alpha) *
            smith_g1(n_dot_v, alpha) /
            max(4.0f * n_dot_v, 1.0e-20f);
        return select(
            0.0f,
            pdf,
            (n_dot_v > 0.0f) &
                (n_dot_l > 0.0f) &
                (n_dot_h > 0.0f) &
                (v_dot_h > 0.0f));
    }

    [[nodiscard]] static Float3 sample_ggx(
        const TracedClosure &closure,
        Float3 incoming,
        Float2 random) noexcept {
        auto alpha = max(
            closure.roughness * closure.roughness,
            1.0e-3f);
        auto normal = safe_normalize(
            closure.normal,
            make_float3(0.0f, 0.0f, 1.0f));
        auto helper = select(
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            abs(normal.z) < 0.999f);
        auto tangent = safe_normalize(
            cross(helper, normal),
            make_float3(1.0f, 0.0f, 0.0f));
        auto bitangent = cross(normal, tangent);

        // Heitz 2018 GGX visible-normal sampling, matching the
        // distribution used by Cycles. Sampling the full NDF instead
        // produces rare, extremely large weights at grazing angles.
        auto local_incoming = make_float3(
            dot(tangent, incoming),
            dot(bitangent, incoming),
            max(dot(normal, incoming), 1.0e-6f));
        auto stretched_incoming = safe_normalize(
            make_float3(
                alpha * local_incoming.x,
                alpha * local_incoming.y,
                local_incoming.z),
            make_float3(0.0f, 0.0f, 1.0f));
        auto projected_length2 =
            stretched_incoming.x * stretched_incoming.x +
            stretched_incoming.y * stretched_incoming.y;
        auto projected_tangent =
            make_float3(
                -stretched_incoming.y,
                stretched_incoming.x,
                0.0f) /
            sqrt(max(projected_length2, 1.0e-20f));
        auto basis_x = select(
            make_float3(1.0f, 0.0f, 0.0f),
            projected_tangent,
            projected_length2 > 1.0e-7f);
        auto basis_y = cross(stretched_incoming, basis_x);

        auto disk_radius =
            sqrt(clamp(random.x, 0.0f, 1.0f));
        auto disk_phi = two_pi * random.y;
        auto disk = make_float2(
            disk_radius * cos(disk_phi),
            disk_radius * sin(disk_phi));
        auto projected_area =
            0.5f * (1.0f + stretched_incoming.z);
        disk.y = lerp(
            sqrt(max(1.0f - disk.x * disk.x, 0.0f)),
            disk.y,
            projected_area);
        auto hemisphere_z = sqrt(max(
            1.0f - disk.x * disk.x - disk.y * disk.y,
            0.0f));
        auto stretched_half = safe_normalize(
            basis_x * disk.x +
                basis_y * disk.y +
                stretched_incoming * hemisphere_z,
            make_float3(0.0f, 0.0f, 1.0f));
        auto local_half = safe_normalize(
            make_float3(
                alpha * stretched_half.x,
                alpha * stretched_half.y,
                max(stretched_half.z, 0.0f)),
            make_float3(0.0f, 0.0f, 1.0f));
        auto half_vector = safe_normalize(
            tangent * local_half.x +
                bitangent * local_half.y +
                normal * local_half.z,
            normal);
        return safe_normalize(
            -incoming +
                2.0f * dot(incoming, half_vector) *
                    half_vector,
            normal);
    }

    [[nodiscard]] static Float3 sample_cosine_hemisphere(
        Float3 normal,
        Float2 random) noexcept {
        auto radius = sqrt(clamp(random.x, 0.0f, 1.0f));
        auto phi = two_pi * random.y;
        auto x = radius * cos(phi);
        auto y = radius * sin(phi);
        auto z = sqrt(max(1.0f - radius * radius, 0.0f));
        auto helper = select(
            make_float3(1.0f, 0.0f, 0.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            abs(normal.z) < 0.999f);
        auto tangent = safe_normalize(
            cross(helper, normal),
            make_float3(1.0f, 0.0f, 0.0f));
        auto bitangent = cross(normal, tangent);
        return safe_normalize(
            tangent * x + bitangent * y + normal * z,
            normal);
    }

    [[nodiscard]] static Float3 rotate_euler(
        Float3 value,
        Float3 rotation) noexcept {
        auto sx = sin(rotation.x);
        auto cx = cos(rotation.x);
        auto sy = sin(rotation.y);
        auto cy = cos(rotation.y);
        auto sz = sin(rotation.z);
        auto cz = cos(rotation.z);
        auto x_rotated = make_float3(
            value.x,
            cx * value.y - sx * value.z,
            sx * value.y + cx * value.z);
        auto y_rotated = make_float3(
            cy * x_rotated.x + sy * x_rotated.z,
            x_rotated.y,
            -sy * x_rotated.x + cy * x_rotated.z);
        return make_float3(
            cz * y_rotated.x - sz * y_rotated.y,
            sz * y_rotated.x + cz * y_rotated.y,
            y_rotated.z);
    }

    [[nodiscard]] std::vector<bool> value_dependency_mask(
        compiler::ValueExpressionId root) const {
        const auto instruction_count =
            _program->value_instructions().size();
        std::vector<bool> active(instruction_count, false);
        std::vector<compiler::ValueExpressionId> pending;
        pending.emplace_back(root);
        while (!pending.empty()) {
            const auto id = pending.back();
            pending.pop_back();
            if (!id.valid() ||
                id.value >= instruction_count ||
                active[id.value]) {
                continue;
            }
            active[id.value] = true;
            const auto &instruction =
                _program->value_instructions()[id.value];
            const std::array dependencies{
                instruction.a,
                instruction.b,
                instruction.c,
                instruction.d,
                instruction.e,
                instruction.f,
                instruction.g,
                instruction.h,
                instruction.i,
                instruction.j};
            for (const auto dependency : dependencies) {
                if (dependency.valid()) {
                    pending.emplace_back(dependency);
                }
            }
        }
        return active;
    }

    [[nodiscard]] TracedValues trace_values(
        const ShaderServices &services,
        const SurfacePoint &point,
        const std::vector<bool> *active_mask = nullptr) const noexcept {
        TracedValues result;
        result.shading_normal = point.shading_normal;
        const auto &instructions =
            _program->value_instructions();
        result.values.reserve(instructions.size());
        for (std::size_t instruction_index = 0u;
             instruction_index < instructions.size();
             ++instruction_index) {
            if (active_mask != nullptr &&
                !(*active_mask)[instruction_index]) {
                result.values.emplace_back(
                    make_float4(0.0f));
                continue;
            }
            const auto &instruction =
                instructions[instruction_index];
            Float4 value = make_float4(0.0f);
            switch (instruction.operation) {
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
                case compiler::ValueOperation::mix: {
                    auto t = clamp(
                        scalar(instruction.c, result),
                        0.0f,
                        1.0f);
                    auto a = vector(instruction.a, result);
                    auto b = vector(instruction.b, result);
                    Float3 mixed;
                    if (instruction.static_u0 == 1u) {
                        auto hsv = rgb_to_hsv(a);
                        auto hsv_b = rgb_to_hsv(b);
                        hsv.z = lerp(hsv.z, hsv_b.z, t);
                        mixed = hsv_to_rgb(hsv);
                    } else if (instruction.static_u0 == 2u) {
                        auto hsv_b = rgb_to_hsv(b);
                        auto hsv = rgb_to_hsv(a);
                        hsv.x = hsv_b.x;
                        hsv.y = hsv_b.y;
                        auto recolored = hsv_to_rgb(hsv);
                        mixed = select(
                            a,
                            lerp(a, recolored, t),
                            hsv_b.y != 0.0f);
                    } else {
                        mixed = lerp(a, b, t);
                    }
                    value = make_float4(
                        mixed,
                        1.0f);
                    break;
                }
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
                    auto factor = clamp(
                        scalar(instruction.b, result),
                        0.0f,
                        1.0f);
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
                    value = make_float4(
                        point.uv.x, point.uv.y, 0.0f, 0.0f);
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
                case compiler::ValueOperation::layer_weight_fresnel: {
                    auto blend = scalar(instruction.a, result);
                    auto normal = safe_normalize(
                        vector(instruction.b, result),
                        result.shading_normal);
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
                    auto normal = safe_normalize(
                        vector(instruction.b, result),
                        result.shading_normal);
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
                        mapped = rotate_euler(
                            input - location, -rotation);
                        mapped /= select(
                            make_float3(1.0f),
                            scale,
                            abs(scale) > 1.0e-20f);
                    } else {
                        mapped = rotate_euler(
                            input * scale, rotation);
                        if (instruction.static_u0 == 0u) {
                            mapped += location;
                        }
                        if (instruction.static_u0 == 3u) {
                            mapped = safe_normalize(
                                mapped,
                                point.shading_normal);
                        }
                    }
                    value = make_float4(mapped, 0.0f);
                    break;
                }
                case compiler::ValueOperation::image_color:
                case compiler::ValueOperation::image_alpha: {
                    auto uv = vector(instruction.a, result).xy();
                    Bool valid = true;
                    const auto address =
                        instruction.static_u1 & 0xffu;
                    if (address == 0u) {
                        uv = fract(uv);
                    } else if (address == 1u) {
                        valid =
                            all(uv >= make_float2(0.0f)) &
                            all(uv <= make_float2(1.0f));
                        uv = clamp(
                            uv,
                            make_float2(0.0f),
                            make_float2(1.0f));
                    } else if (address == 2u) {
                        uv = clamp(
                            uv,
                            make_float2(0.0f),
                            make_float2(1.0f));
                    } else {
                        auto period = fract(uv * 0.5f) * 2.0f;
                        uv = 1.0f - abs(period - 1.0f);
                    }
                    // Blender UVs use a bottom-left origin while decoded
                    // host images are uploaded in top-to-bottom row order.
                    uv.y = 1.0f - uv.y;
                    auto sampled = services.texture_2d(
                        static_cast<std::uint32_t>(
                            instruction.static_u0),
                        uv,
                        make_float2(0.0f),
                        make_float2(0.0f));
                    sampled = select(
                        make_float4(0.0f),
                        sampled,
                        valid);
                    if (instruction.operation ==
                        compiler::ValueOperation::image_color) {
                        const auto unassociate_alpha =
                            ((instruction.static_u1 >> 9u) & 1u) !=
                            0u;
                        if (unassociate_alpha) {
                            auto alpha = sampled.w;
                            auto should_unassociate =
                                (alpha != 0.0f) &
                                (alpha != 1.0f);
                            auto safe_alpha = select(
                                1.0f,
                                alpha,
                                should_unassociate);
                            sampled = make_float4(
                                select(
                                    sampled.xyz(),
                                    sampled.xyz() / safe_alpha,
                                    should_unassociate),
                                alpha);
                        }
                        const auto encoded_as_srgb =
                            ((instruction.static_u1 >> 8u) & 1u) !=
                            0u;
                        auto color = sampled.xyz();
                        if (encoded_as_srgb) {
                            // Match Cycles' svm_image_texture ordering:
                            // filter associated encoded texels, optionally
                            // unassociate, then decode sRGB.
                            color = srgb_to_linear(color);
                        }
                        value = make_float4(
                            color, sampled.w);
                    } else {
                        value = make_float4(sampled.w);
                    }
                    break;
                }
                case compiler::ValueOperation::attribute_color:
                case compiler::ValueOperation::attribute_alpha: {
                    auto attribute = services.attribute(
                        instruction.static_u0, point);
                    value =
                        instruction.operation ==
                                compiler::ValueOperation::
                                    attribute_alpha
                            ? make_float4(attribute.w)
                            : attribute;
                    break;
                }
                case compiler::ValueOperation::normal_map: {
                    auto mapped =
                        vector(instruction.a, result) * 2.0f -
                        1.0f;
                    auto strength =
                        scalar(instruction.b, result);
                    const auto space =
                        static_cast<compiler::NormalMapSpace>(
                            instruction.static_u0);
                    const auto transform_object_normal =
                        [&](Float3 object_normal) noexcept {
                            return safe_normalize(
                                point.normal_to_world_x *
                                        object_normal.x +
                                    point.normal_to_world_y *
                                        object_normal.y +
                                    point.normal_to_world_z *
                                        object_normal.z,
                                point.shading_normal);
                        };
                    Float3 world;
                    if (space ==
                        compiler::NormalMapSpace::tangent) {
                        // This is the Cycles SVM tangent-space path:
                        // construct from Blender's MikkTSpace tangent/sign
                        // and the unnormalized interpolated object normal,
                        // then apply the inverse-transpose normal transform.
                        mapped.x *= strength;
                        mapped.y *= strength;
                        mapped.z =
                            1.0f +
                            (mapped.z - 1.0f) *
                                clamp(strength, 0.0f, 1.0f);
                        auto object_bitangent =
                            point.tangent_sign *
                            cross(
                                point.object_shading_normal,
                                point.object_tangent);
                        auto object_normal = safe_normalize(
                            point.object_tangent * mapped.x +
                                object_bitangent * mapped.y +
                                point.object_shading_normal *
                                    mapped.z,
                            point.object_shading_normal);
                        world =
                            transform_object_normal(object_normal);
                        world = select(
                            world, -world, point.back_facing);
                        auto tangent_available =
                            (length_squared(
                                 point.object_tangent) >
                             1.0e-20f) &
                            (abs(point.tangent_sign) >
                             1.0e-20f);
                        world = select(
                            point.shading_normal,
                            world,
                            tangent_available);
                    } else {
                        if (space ==
                                compiler::NormalMapSpace::
                                    blender_object ||
                            space ==
                                compiler::NormalMapSpace::
                                    blender_world) {
                            mapped.y = -mapped.y;
                            mapped.z = -mapped.z;
                        }
                        world =
                            space ==
                                        compiler::NormalMapSpace::
                                            object ||
                                    space ==
                                        compiler::NormalMapSpace::
                                            blender_object
                                ? transform_object_normal(mapped)
                                : safe_normalize(
                                      mapped,
                                      point.shading_normal);
                        world = select(
                            world, -world, point.back_facing);
                        auto nonnegative_strength =
                            max(strength, 0.0f);
                        world = safe_normalize(
                            point.shading_normal +
                                (world -
                                 point.shading_normal) *
                                    nonnegative_strength,
                            point.shading_normal);
                    }
                    value = make_float4(world, 0.0f);
                    break;
                }
                case compiler::ValueOperation::bump: {
                    auto normal_in = safe_normalize(
                        vector(instruction.e, result),
                        result.shading_normal);
                    auto filter_width = max(
                        scalar(instruction.d, result), 0.0f);

                    auto point_x = point;
                    point_x.position =
                        point.position +
                        point.dPdx * filter_width;
                    point_x.object_position =
                        point.object_position +
                        point.object_dPdx * filter_width;
                    point_x.generated =
                        point.generated +
                        point.generated_dx * filter_width;
                    point_x.uv =
                        point.uv + point.uv_dx * filter_width;
                    point_x.barycentric =
                        point.barycentric +
                        point.barycentric_dx * filter_width;

                    auto point_y = point;
                    point_y.position =
                        point.position +
                        point.dPdy * filter_width;
                    point_y.object_position =
                        point.object_position +
                        point.object_dPdy * filter_width;
                    point_y.generated =
                        point.generated +
                        point.generated_dy * filter_width;
                    point_y.uv =
                        point.uv + point.uv_dy * filter_width;
                    point_y.barycentric =
                        point.barycentric +
                        point.barycentric_dy * filter_width;

                    const auto height_dependencies =
                        value_dependency_mask(instruction.a);
                    auto values_x = trace_values(
                        services,
                        point_x,
                        &height_dependencies);
                    auto values_y = trace_values(
                        services,
                        point_y,
                        &height_dependencies);
                    auto height_center =
                        scalar(instruction.a, result);
                    auto height_x =
                        scalar(instruction.a, values_x);
                    auto height_y =
                        scalar(instruction.a, values_y);
                    auto rx = cross(point.dPdy, normal_in);
                    auto ry = cross(normal_in, point.dPdx);
                    auto determinant =
                        dot(point.dPdx, rx);
                    auto surface_gradient =
                        (height_x - height_center) * rx +
                        (height_y - height_center) * ry;
                    auto distance =
                        scalar(instruction.c, result);
                    if (instruction.static_u0 != 0u) {
                        distance = -distance;
                    }
                    auto determinant_sign = select(
                        -1.0f,
                        1.0f,
                        determinant >= 0.0f);
                    auto perturbed = safe_normalize(
                        filter_width * abs(determinant) *
                                normal_in -
                            distance * determinant_sign *
                                surface_gradient,
                        normal_in);
                    auto strength =
                        max(scalar(instruction.b, result), 0.0f);
                    auto normal_out = safe_normalize(
                        strength * perturbed +
                            (1.0f - strength) * normal_in,
                        normal_in);
                    value = make_float4(normal_out, 0.0f);
                    result.shading_normal = normal_out;
                    break;
                }
                case compiler::ValueOperation::noise_factor:
                case compiler::ValueOperation::noise_color: {
                    const auto color_needed =
                        instruction.operation ==
                        compiler::ValueOperation::noise_color;
                    const auto normalize =
                        (instruction.static_u1 & 1u) != 0u;
                    const auto noise_type =
                        static_cast<cycles_noise::Type>(
                            (instruction.static_u1 >> 8u) &
                            0xffu);
                    auto scale =
                        scalar(instruction.b, result);
                    value = cycles_noise::
                        evaluate_texture_shared(
                        static_cast<std::uint32_t>(
                            instruction.static_u0),
                        noise_type,
                        normalize,
                        color_needed,
                        vector(instruction.a, result) * scale,
                        scalar(instruction.g, result) * scale,
                        scalar(instruction.c, result),
                        scalar(instruction.d, result),
                        scalar(instruction.e, result),
                        scalar(instruction.h, result),
                        scalar(instruction.i, result),
                        scalar(instruction.f, result));
                    break;
                }
                case compiler::ValueOperation::white_noise_value:
                case compiler::ValueOperation::white_noise_color: {
                    const auto color_needed =
                        instruction.operation ==
                        compiler::ValueOperation::
                            white_noise_color;
                    value =
                        cycles_noise::evaluate_white_shared(
                            static_cast<std::uint32_t>(
                                instruction.static_u0),
                            color_needed,
                            vector(instruction.a, result),
                            scalar(instruction.b, result));
                    break;
                }
                case compiler::ValueOperation::brick_color:
                case compiler::ValueOperation::brick_factor: {
                    auto p =
                        vector(instruction.a, result) *
                        scalar(instruction.e, result);
                    auto mortar_size = max(
                        scalar(instruction.f, result),
                        0.0f);
                    auto mortar_smooth = max(
                        scalar(instruction.g, result),
                        0.0f);
                    auto bias =
                        scalar(instruction.h, result);
                    auto brick_width = max(
                        abs(scalar(instruction.i, result)),
                        1.0e-20f);
                    auto row_height = max(
                        abs(scalar(instruction.j, result)),
                        1.0e-20f);
                    auto row = cast<int>(
                        floor(p.y / row_height));
                    Float offset = 0.0f;
                    if (instruction.static_u0 != 0u &&
                        instruction.static_u1 != 0u) {
                        auto squash_row =
                            (row %
                             static_cast<int>(
                                 instruction.static_u1)) ==
                            0;
                        brick_width *= select(
                            1.0f,
                            instruction.static_f1,
                            squash_row);
                        auto offset_row =
                            (row %
                             static_cast<int>(
                                 instruction.static_u0)) ==
                            0;
                        offset = select(
                            0.0f,
                            brick_width *
                                instruction.static_f0,
                            offset_row);
                    }
                    auto brick = cast<int>(floor(
                        (p.x + offset) / brick_width));
                    auto x =
                        p.x + offset -
                        brick_width * cast<float>(brick);
                    auto y =
                        p.y - row_height * cast<float>(row);
                    auto n =
                        (cast<uint>(row) << 16u) +
                        (cast<uint>(brick) & 0xffffu);
                    n = (n + 1013u) & 0x7fffffffu;
                    n = (n >> 13u) ^ n;
                    auto nn =
                        (n * (n * n * 60493u + 19990303u) +
                         1376312589u) &
                        0x7fffffffu;
                    auto tint = clamp(
                        0.5f *
                                cast<float>(nn) /
                                1073741824.0f +
                            bias,
                        0.0f,
                        1.0f);
                    auto min_distance = min(
                        min(x, y),
                        min(
                            brick_width - x,
                            row_height - y));
                    Float mortar = select(
                        1.0f,
                        0.0f,
                        min_distance >= mortar_size);
                    auto edge =
                        1.0f -
                        min_distance /
                            max(mortar_size, 1.0e-20f);
                    auto smooth = clamp(
                        edge /
                            max(mortar_smooth, 1.0e-20f),
                        0.0f,
                        1.0f);
                    smooth =
                        smooth * smooth *
                        (3.0f - 2.0f * smooth);
                    auto smooth_mortar = select(
                        smooth,
                        0.0f,
                        min_distance >= mortar_size);
                    mortar = select(
                        mortar,
                        smooth_mortar,
                        mortar_smooth != 0.0f);
                    if (instruction.operation ==
                        compiler::ValueOperation::brick_factor) {
                        value = make_float4(mortar);
                    } else {
                        auto brick_color = lerp(
                            vector(instruction.b, result),
                            vector(instruction.c, result),
                            tint);
                        value = make_float4(
                            lerp(
                                brick_color,
                                vector(
                                    instruction.d, result),
                                mortar),
                            1.0f);
                    }
                    break;
                }
                case compiler::ValueOperation::gradient: {
                    auto p = vector(instruction.a, result);
                    Float gradient = p.x;
                    if (instruction.static_u0 == 1u) {
                        gradient = max(p.x, 0.0f);
                        gradient *= gradient;
                    } else if (instruction.static_u0 == 2u) {
                        auto t = clamp(p.x, 0.0f, 1.0f);
                        gradient = t * t * (3.0f - 2.0f * t);
                    } else if (instruction.static_u0 == 3u) {
                        gradient = (p.x + p.y) * 0.5f;
                    } else if (instruction.static_u0 == 4u) {
                        gradient =
                            atan2(p.y, p.x) / two_pi + 0.5f;
                    } else if (instruction.static_u0 == 5u) {
                        gradient = max(
                            0.999999f - length(p),
                            0.0f);
                    } else if (instruction.static_u0 == 6u) {
                        gradient = max(
                            0.999999f - length(p),
                            0.0f);
                        gradient *= gradient;
                    }
                    gradient = clamp(
                        gradient, 0.0f, 1.0f);
                    value = make_float4(gradient);
                    break;
                }
                case compiler::ValueOperation::color_ramp: {
                    auto factor =
                        scalar(instruction.a, result);
                    Float3 color = make_float3(0.0f);
                    Float alpha = 1.0f;
                    const auto count =
                        instruction.static_table.size() / 5u;
                    if (count != 0u) {
                        color = make_float3(
                            instruction.static_table[1u],
                            instruction.static_table[2u],
                            instruction.static_table[3u]);
                        alpha = instruction.static_table[4u];
                        for (std::size_t i = 1u; i < count; ++i) {
                            const auto p0 =
                                instruction.static_table[
                                    (i - 1u) * 5u];
                            const auto p1 =
                                instruction.static_table[i * 5u];
                            auto t = clamp(
                                (factor - p0) /
                                    std::max(p1 - p0, 1.0e-20f),
                                0.0f,
                                1.0f);
                            if (instruction.static_u0 == 1u) {
                                t = 0.0f;
                            }
                            auto c0 = make_float3(
                                instruction.static_table[
                                    (i - 1u) * 5u + 1u],
                                instruction.static_table[
                                    (i - 1u) * 5u + 2u],
                                instruction.static_table[
                                    (i - 1u) * 5u + 3u]);
                            auto c1 = make_float3(
                                instruction.static_table[
                                    i * 5u + 1u],
                                instruction.static_table[
                                    i * 5u + 2u],
                                instruction.static_table[
                                    i * 5u + 3u]);
                            auto a0 =
                                instruction.static_table[
                                    (i - 1u) * 5u + 4u];
                            auto a1 =
                                instruction.static_table[
                                    i * 5u + 4u];
                            auto use =
                                factor >= p0;
                            color = select(
                                color, lerp(c0, c1, t), use);
                            alpha = select(
                                alpha, lerp(a0, a1, t), use);
                        }
                        const auto last =
                            (count - 1u) * 5u;
                        auto use_last =
                            factor >=
                            instruction.static_table[last];
                        color = select(
                            color,
                            make_float3(
                                instruction.static_table[
                                    last + 1u],
                                instruction.static_table[
                                    last + 2u],
                                instruction.static_table[
                                    last + 3u]),
                            use_last);
                        alpha = select(
                            alpha,
                            instruction.static_table[last + 4u],
                            use_last);
                    }
                    value =
                        instruction.static_u1 != 0u
                            ? make_float4(alpha)
                            : make_float4(color, alpha);
                    break;
                }
                case compiler::ValueOperation::rgb_curve: {
                    // The normalized adapter stores a compact sampled
                    // transfer table. Until a bindless LUT is warranted,
                    // the identity path remains exact and non-identity
                    // tables use a host-unrolled linear interpolation.
                    auto input = vector(instruction.a, result);
                    auto factor = clamp(
                        scalar(instruction.b, result),
                        0.0f,
                        1.0f);
                    Float3 mapped = input;
                    const auto count =
                        instruction.static_table.size() / 4u;
                    if (count >= 2u) {
                        mapped = make_float3(0.0f);
                        for (std::size_t i = 1u; i < count; ++i) {
                            const auto x0 =
                                instruction.static_table[
                                    (i - 1u) * 4u];
                            const auto x1 =
                                instruction.static_table[i * 4u];
                            auto t = clamp(
                                (input - x0) /
                                    std::max(x1 - x0, 1.0e-20f),
                                make_float3(0.0f),
                                make_float3(1.0f));
                            auto y0 = make_float3(
                                instruction.static_table[
                                    (i - 1u) * 4u + 1u],
                                instruction.static_table[
                                    (i - 1u) * 4u + 2u],
                                instruction.static_table[
                                    (i - 1u) * 4u + 3u]);
                            auto y1 = make_float3(
                                instruction.static_table[
                                    i * 4u + 1u],
                                instruction.static_table[
                                    i * 4u + 2u],
                                instruction.static_table[
                                    i * 4u + 3u]);
                            mapped = select(
                                mapped,
                                lerp(y0, y1, t),
                                input >= x0);
                        }
                    }
                    value = make_float4(
                        lerp(input, mapped, factor),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::separate_r:
                    value = make_float4(
                        separate_color(
                            vector(instruction.a, result),
                            instruction.static_u0)
                            .x);
                    break;
                case compiler::ValueOperation::separate_g:
                    value = make_float4(
                        separate_color(
                            vector(instruction.a, result),
                            instruction.static_u0)
                            .y);
                    break;
                case compiler::ValueOperation::separate_b:
                    value = make_float4(
                        separate_color(
                            vector(instruction.a, result),
                            instruction.static_u0)
                            .z);
                    break;
                case compiler::ValueOperation::combine_color: {
                    auto channels = make_float3(
                        scalar(instruction.a, result),
                        scalar(instruction.b, result),
                        scalar(instruction.c, result));
                    value = make_float4(
                        combine_color(
                            channels,
                            instruction.static_u0),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::nishita_sky: {
                    auto direction = safe_normalize(
                        -point.incoming,
                        make_float3(0.0f, 0.0f, 1.0f));
                    value = make_float4(
                        services.nishita_sky(
                            point.parameter_block,
                            static_cast<std::uint32_t>(
                                instruction.static_u0),
                            direction,
                            scalar(instruction.a, result),
                            scalar(instruction.b, result),
                            scalar(instruction.c, result),
                            scalar(instruction.d, result)),
                        1.0f);
                    break;
                }
            }
            result.values.emplace_back(value);
        }
        return result;
    }

    template<typename Function>
    void for_each_closure(
        const TracedValues &values,
        Function &&function) const noexcept {
        auto visit =
            [&](auto &&self,
                compiler::ClosureExpressionId id,
                Float mix_weight) noexcept -> void {
            const auto &closure =
                _program->closure_instructions()[id.value];
            switch (closure.operation) {
                case compiler::ClosureOperation::add:
                    self(self, closure.a, mix_weight);
                    self(self, closure.b, mix_weight);
                    return;
                case compiler::ClosureOperation::mix: {
                    auto factor = clamp(
                        scalar(closure.factor, values),
                        0.0f,
                        1.0f);
                    self(
                        self,
                        closure.a,
                        mix_weight * (1.0f - factor));
                    self(
                        self,
                        closure.b,
                        mix_weight * factor);
                    return;
                }
                case compiler::ClosureOperation::translucent: {
                    auto color = vector(
                        closure.color, values);
                    function(TracedClosure{
                        .operation =
                            compiler::ClosureOperation::translucent,
                        .weight = color * mix_weight,
                        .color = color,
                        .normal = safe_normalize(
                            vector(closure.normal, values),
                            values.shading_normal),
                        .roughness = 0.0f,
                        .diffuse_roughness = 0.0f,
                        .metallic = 0.0f,
                        .ior = 1.0f,
                        .specular_ior_level = 0.0f,
                        .specular_tint = make_float3(1.0f)});
                    return;
                }
                case compiler::ClosureOperation::diffuse:
                case compiler::ClosureOperation::principled:
                case compiler::ClosureOperation::glossy: {
                    auto color = vector(
                        closure.color, values);
                    auto metallic =
                        closure.operation ==
                                compiler::ClosureOperation::principled
                            ? clamp(
                                  scalar(
                                      closure.metallic, values),
                                  0.0f,
                                  1.0f)
                            : closure.operation ==
                                      compiler::ClosureOperation::glossy
                                  ? Float{1.0f}
                                  : Float{0.0f};
                    auto ior =
                        closure.operation ==
                                compiler::ClosureOperation::principled
                            ? max(
                                  scalar(closure.ior, values),
                                  1.0e-5f)
                            : Float{1.5f};
                    auto diffuse_roughness =
                        closure.operation ==
                                compiler::ClosureOperation::principled
                            ? scalar(
                                  closure.diffuse_roughness,
                                  values)
                            : scalar(
                                  closure.roughness, values);
                    auto specular_ior_level =
                        closure.operation ==
                                compiler::ClosureOperation::principled
                            ? max(
                                  scalar(
                                      closure.specular_ior_level,
                                      values),
                                  0.0f)
                            : Float{0.5f};
                    auto specular_tint =
                        closure.operation ==
                                compiler::ClosureOperation::principled
                            ? max(
                                  vector(
                                      closure.specular_tint,
                                      values),
                                  make_float3(0.0f))
                            : make_float3(1.0f);
                    function(TracedClosure{
                        .operation = closure.operation,
                        .weight =
                            closure.operation ==
                                    compiler::ClosureOperation::diffuse
                                ? color * mix_weight
                                : make_float3(mix_weight),
                        .color = color,
                        .normal = safe_normalize(
                            vector(closure.normal, values),
                            values.shading_normal),
                        .roughness = scalar(
                            closure.roughness, values),
                        .diffuse_roughness =
                            diffuse_roughness,
                        .metallic = metallic,
                        .ior = ior,
                        .specular_ior_level =
                            specular_ior_level,
                        .specular_tint = specular_tint,
                        .preserve_ggx_energy =
                            closure.preserve_ggx_energy});
                    return;
                }
                case compiler::ClosureOperation::emission: {
                    auto color = vector(
                        closure.color, values);
                    function(TracedClosure{
                        .operation =
                            compiler::ClosureOperation::emission,
                        .weight =
                            color *
                            scalar(closure.strength, values) *
                            mix_weight,
                        .color = color,
                        .normal = make_float3(0.0f, 0.0f, 1.0f),
                        .roughness = 0.0f,
                        .metallic = 0.0f,
                        .ior = 1.0f});
                    return;
                }
                case compiler::ClosureOperation::transparent: {
                    auto color = vector(
                        closure.color, values);
                    function(TracedClosure{
                        .operation =
                            compiler::ClosureOperation::transparent,
                        .weight = color * mix_weight,
                        .color = color,
                        .normal = make_float3(0.0f, 0.0f, 1.0f),
                        .roughness = 0.0f,
                        .metallic = 0.0f,
                        .ior = 1.0f});
                    return;
                }
            }
        };
        visit(visit, _program->root(), 1.0f);
    }

public:
    explicit GraphSurface(
        std::shared_ptr<const compiler::SurfaceProgram> program) noexcept
        : _program{std::move(program)} {
        if (_program) {
            for (const auto &instruction :
                 _program->value_instructions()) {
                if (
                    instruction.operation ==
                        compiler::ValueOperation::noise_factor ||
                    instruction.operation ==
                        compiler::ValueOperation::noise_color) {
                    const auto color_needed =
                        instruction.operation ==
                        compiler::ValueOperation::noise_color;
                    const auto normalize =
                        (instruction.static_u1 & 1u) != 0u;
                    const auto noise_type =
                        static_cast<cycles_noise::Type>(
                            (instruction.static_u1 >> 8u) &
                            0xffu);
                    cycles_noise::prepare_texture(
                        static_cast<std::uint32_t>(
                            instruction.static_u0),
                        noise_type,
                        normalize,
                        color_needed);
                } else if (
                    instruction.operation ==
                        compiler::ValueOperation::
                            white_noise_value ||
                    instruction.operation ==
                        compiler::ValueOperation::
                            white_noise_color) {
                    cycles_noise::prepare_white_texture(
                        static_cast<std::uint32_t>(
                            instruction.static_u0),
                        instruction.operation ==
                            compiler::ValueOperation::
                                white_noise_color);
                }
            }
            for (const auto &closure :
                 _program->closure_instructions()) {
                _capabilities.may_emit |=
                    closure.operation ==
                    compiler::ClosureOperation::emission;
                _capabilities.may_be_transparent |=
                    closure.operation ==
                    compiler::ClosureOperation::transparent;
            }
        }
    }

    [[nodiscard]] SurfaceCapabilities capabilities()
        const noexcept override {
        return _capabilities;
    }

    [[nodiscard]] SurfaceEvaluation evaluate_traced(
        const ShaderServices &services,
        const TracedValues &values,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing_expression,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceEvaluation::zero();
        Float total_sample_weight = 0.0f;
        Float weighted_pdf = 0.0f;
        auto outgoing = safe_normalize(
            Float3{outgoing_expression},
            point.shading_normal);
        auto incoming = safe_normalize(
            point.incoming,
            -outgoing);
        auto diffuse_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_diffuse)) != 0u;
        auto glossy_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_glossy)) != 0u;
        auto transparent_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_transparent)) != 0u;
        auto transmission_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_transmission)) != 0u;
        Bool has_diffuse = false;
        Bool has_translucent = false;
        Bool has_glossy = false;

        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                if (closure.operation ==
                    compiler::ClosureOperation::transparent) {
                    auto weight = sample_weight(closure.weight);
                    total_sample_weight += select(
                        0.0f, weight, transparent_enabled);
                    return;
                }
                const auto is_diffuse =
                    closure.operation ==
                    compiler::ClosureOperation::diffuse;
                const auto is_translucent =
                    closure.operation ==
                    compiler::ClosureOperation::translucent;
                const auto is_principled =
                    closure.operation ==
                    compiler::ClosureOperation::principled;
                const auto is_glossy =
                    closure.operation ==
                    compiler::ClosureOperation::glossy;
                if (!is_diffuse && !is_translucent &&
                    !is_principled &&
                    !is_glossy) {
                    return;
                }
                auto diffuse_normal =
                    is_translucent
                        ? -closure.normal
                        : closure.normal;
                auto diffuse_pdf =
                    max(dot(diffuse_normal, outgoing), 0.0f) *
                    inverse_pi;
                auto glossy_pdf = microfacet_pdf(
                    closure, incoming, outgoing);
                auto translucent_allowed =
                    diffuse_enabled &
                    transmission_enabled &
                    is_translucent;
                auto diffuse_allowed =
                    (diffuse_enabled &
                     (is_diffuse || is_principled)) |
                    translucent_allowed;
                auto glossy_allowed =
                    glossy_enabled &
                    (is_principled || is_glossy);
                Float specular_chance;
                Float3 diffuse_contribution;
                Float3 glossy_contribution;
                Float3 selection_color;
                if (is_principled) {
                    auto state = principled_state(
                        services, closure, incoming);
                    auto diffuse_weight =
                        sample_weight(
                            state.diffuse_sample_weight);
                    auto glossy_weight =
                        sample_weight(
                            state.glossy_sample_weight);
                    specular_chance =
                        glossy_weight /
                        max(
                            diffuse_weight +
                                glossy_weight,
                            1.0e-20f);
                    diffuse_contribution =
                        state.diffuse_albedo *
                        diffuse_intensity(
                            closure, incoming, outgoing);
                    glossy_contribution =
                        closure.weight *
                        microfacet_intensity(
                            services,
                            closure,
                            incoming,
                            outgoing);
                    selection_color =
                        state.diffuse_sample_weight +
                        state.glossy_sample_weight;
                } else if (is_glossy) {
                    specular_chance = 1.0f;
                    diffuse_contribution =
                        make_float3(0.0f);
                    glossy_contribution =
                        closure.weight *
                        microfacet_intensity(
                            services,
                            closure,
                            incoming,
                            outgoing);
                    selection_color =
                        closure.weight *
                        max(
                            closure.color,
                            make_float3(0.04f));
                } else if (is_translucent) {
                    specular_chance = 0.0f;
                    auto cosine =
                        max(
                            dot(-closure.normal, outgoing),
                            0.0f) *
                        inverse_pi;
                    diffuse_contribution =
                        closure.weight * cosine;
                    glossy_contribution =
                        make_float3(0.0f);
                    selection_color = closure.weight;
                } else {
                    specular_chance = 0.0f;
                    diffuse_contribution =
                        closure.weight *
                        diffuse_intensity(
                            closure, incoming, outgoing);
                    glossy_contribution =
                        make_float3(0.0f);
                    selection_color = closure.weight;
                }
                specular_chance = select(
                    specular_chance,
                    0.0f,
                    !glossy_allowed);
                specular_chance = select(
                    specular_chance,
                    1.0f,
                    glossy_allowed & (!diffuse_allowed));
                auto pdf = lerp(
                    diffuse_pdf,
                    glossy_pdf,
                    specular_chance);
                auto contribution =
                    select(
                        make_float3(0.0f),
                        diffuse_contribution,
                        diffuse_allowed) +
                    select(
                        make_float3(0.0f),
                        glossy_contribution,
                        glossy_allowed);
                contribution = select(
                    make_float3(0.0f),
                    contribution,
                    diffuse_allowed | glossy_allowed);
                auto enabled_pdf = select(
                    0.0f,
                    pdf,
                    diffuse_allowed | glossy_allowed);
                result.f += contribution;
                result.diffuse_f += select(
                    make_float3(0.0f),
                    diffuse_contribution,
                    diffuse_allowed);
                auto weight =
                    sample_weight(selection_color);
                weight = select(
                    0.0f,
                    weight,
                    diffuse_allowed | glossy_allowed);
                total_sample_weight += weight;
                weighted_pdf += weight * enabled_pdf;
                has_diffuse =
                    has_diffuse |
                    ((diffuse_allowed & (!is_translucent)) &
                     (sample_weight(diffuse_contribution) >
                      0.0f));
                has_translucent =
                    has_translucent |
                    (translucent_allowed &
                     (sample_weight(diffuse_contribution) >
                      0.0f));
                has_glossy =
                    has_glossy |
                    (glossy_allowed &
                     (sample_weight(glossy_contribution) >
                      0.0f));
            });

        auto has_pdf = total_sample_weight > 0.0f;
        result.pdf = select(
            0.0f,
            weighted_pdf / max(total_sample_weight, 1.0e-20f),
            has_pdf);
        result.diffuse_pdf = select(
            0.0f,
            result.pdf,
            has_diffuse | has_translucent);
        auto has_diffuse_pdf = weighted_pdf > 0.0f;
        UInt events =
            static_cast<std::uint32_t>(event_none);
        events = select(
            events,
            events |
                static_cast<std::uint32_t>(
                    event_diffuse | event_reflection),
            has_diffuse);
        events = select(
            events,
            events |
                static_cast<std::uint32_t>(
                    event_glossy | event_reflection),
            has_glossy);
        events = select(
            events,
            events |
                static_cast<std::uint32_t>(
                    event_diffuse | event_transmission),
            has_translucent);
        result.events = select(
            static_cast<std::uint32_t>(event_none),
            events,
            has_diffuse_pdf);
        return result;
    }

    [[nodiscard]] SurfaceEvaluation evaluate(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing_expression,
        const SurfaceQuery &query) const noexcept override {
        if (!_program) {
            return SurfaceEvaluation::zero();
        }
        auto values = trace_values(services, point);
        return evaluate_traced(
            services,
            values,
            point,
            outgoing_expression,
            query);
    }

    [[nodiscard]] SurfaceSample sample(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe_expression,
        Expr<luisa::float2> u_direction_expression,
        const SurfaceQuery &query) const noexcept override {
        auto result = SurfaceSample::zero();
        if (!_program) {
            return result;
        }

        auto values = trace_values(services, point);
        auto incoming = safe_normalize(
            point.incoming,
            point.shading_normal);
        Float total_weight = 0.0f;
        auto diffuse_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_diffuse)) != 0u;
        auto glossy_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_glossy)) != 0u;
        auto transparent_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_transparent)) != 0u;
        auto transmission_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(event_transmission)) != 0u;
        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                auto is_diffuse =
                    closure.operation ==
                    compiler::ClosureOperation::diffuse;
                auto is_translucent =
                    closure.operation ==
                    compiler::ClosureOperation::translucent;
                auto is_principled =
                    closure.operation ==
                    compiler::ClosureOperation::principled;
                auto is_glossy =
                    closure.operation ==
                    compiler::ClosureOperation::glossy;
                auto is_transparent =
                    closure.operation ==
                    compiler::ClosureOperation::transparent;
                auto eligible =
                    is_transparent
                        ? transparent_enabled
                        : is_translucent
                              ? (diffuse_enabled &
                                 transmission_enabled)
                              : is_diffuse
                              ? diffuse_enabled
                              : is_principled
                                    ? (diffuse_enabled |
                                       glossy_enabled)
                                    : is_glossy
                                          ? glossy_enabled
                                          : Bool{false};
                Float3 selection_color;
                if (is_principled) {
                    auto state = principled_state(
                        services, closure, incoming);
                    selection_color =
                        state.diffuse_sample_weight +
                        state.glossy_sample_weight;
                } else {
                    selection_color =
                        is_diffuse || is_translucent ||
                                is_transparent
                            ? closure.weight
                            : closure.weight *
                                  max(
                                      closure.color,
                                      make_float3(0.04f));
                }
                total_weight += select(
                    0.0f,
                    sample_weight(selection_color),
                    eligible);
            });

        auto random_lobe = clamp(
            Float{u_lobe_expression}, 0.0f, 0.99999994f);
        auto target = random_lobe * total_weight;
        auto random_direction = Float2{u_direction_expression};
        Float accumulated = 0.0f;
        Bool selected = false;
        Bool selected_transparent = false;
        Bool selected_translucent = false;
        Bool selected_glossy = false;
        Float3 transparent_weight = make_float3(0.0f);
        Float transparent_sample_weight = 0.0f;

        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                auto is_diffuse =
                    closure.operation ==
                    compiler::ClosureOperation::diffuse;
                auto is_translucent =
                    closure.operation ==
                    compiler::ClosureOperation::translucent;
                auto is_principled =
                    closure.operation ==
                    compiler::ClosureOperation::principled;
                auto is_glossy =
                    closure.operation ==
                    compiler::ClosureOperation::glossy;
                auto is_transparent =
                    closure.operation ==
                    compiler::ClosureOperation::transparent;
                auto eligible =
                    is_transparent
                        ? transparent_enabled
                        : is_translucent
                              ? (diffuse_enabled &
                                 transmission_enabled)
                              : is_diffuse
                              ? diffuse_enabled
                              : is_principled
                                    ? (diffuse_enabled |
                                       glossy_enabled)
                                    : is_glossy
                                          ? glossy_enabled
                                          : Bool{false};
                Float3 selection_color;
                Float principled_specular_chance = 0.0f;
                if (is_principled) {
                    auto state = principled_state(
                        services, closure, incoming);
                    selection_color =
                        state.diffuse_sample_weight +
                        state.glossy_sample_weight;
                    auto diffuse_weight = sample_weight(
                        state.diffuse_sample_weight);
                    auto glossy_weight = sample_weight(
                        state.glossy_sample_weight);
                    principled_specular_chance =
                        glossy_weight /
                        max(
                            diffuse_weight +
                                glossy_weight,
                            1.0e-20f);
                } else {
                    selection_color =
                        is_diffuse || is_translucent ||
                                is_transparent
                            ? closure.weight
                            : closure.weight *
                                  max(
                                      closure.color,
                                      make_float3(0.04f));
                }
                auto weight = select(
                    0.0f,
                    sample_weight(selection_color),
                    eligible);
                auto next = accumulated + weight;
                auto choose =
                    (!selected) &
                    (weight > 0.0f) &
                    (target < next);
                Bool closure_diffuse_enabled = false;
                if (is_diffuse || is_principled) {
                    closure_diffuse_enabled = true;
                } else if (is_translucent) {
                    closure_diffuse_enabled =
                        transmission_enabled;
                }
                auto local_diffuse_enabled =
                    diffuse_enabled &
                    closure_diffuse_enabled;
                auto local_glossy_enabled =
                    glossy_enabled &
                    (is_glossy || is_principled);
                Float specular_chance =
                    is_glossy
                        ? 1.0f
                        : is_diffuse
                              ? 0.0f
                              : principled_specular_chance;
                specular_chance = select(
                    specular_chance,
                    0.0f,
                    !local_glossy_enabled);
                specular_chance = select(
                    specular_chance,
                    1.0f,
                    local_glossy_enabled &
                        (!local_diffuse_enabled));
                auto remapped_specular =
                    random_direction.x /
                    max(specular_chance, 1.0e-20f);
                auto remapped_diffuse =
                    (random_direction.x - specular_chance) /
                    max(1.0f - specular_chance, 1.0e-20f);
                auto remapped_random = make_float2(
                    select(
                        remapped_diffuse,
                        remapped_specular,
                        random_direction.x <
                            specular_chance),
                    random_direction.y);
                auto diffuse_direction =
                    sample_cosine_hemisphere(
                        is_translucent
                            ? -closure.normal
                            : closure.normal,
                        remapped_random);
                auto glossy_direction = sample_ggx(
                    closure,
                    incoming,
                    remapped_random);
                auto transparent_direction = -point.incoming;
                auto sample_glossy =
                    local_glossy_enabled &
                    ((!local_diffuse_enabled) |
                     (random_direction.x <
                      specular_chance));
                auto candidate_direction =
                    is_transparent
                        ? transparent_direction
                        : select(
                              diffuse_direction,
                              glossy_direction,
                              sample_glossy);
                result.wi = select(
                    result.wi,
                    candidate_direction,
                    choose);
                result.roughness = select(
                    result.roughness,
                    !is_transparent
                        ? make_float2(closure.roughness)
                        : make_float2(0.0f),
                    choose);
                transparent_weight = select(
                    transparent_weight,
                    closure.weight,
                    is_transparent ? choose : Bool{false});
                transparent_sample_weight = select(
                    transparent_sample_weight,
                    weight,
                    is_transparent ? choose : Bool{false});
                selected_transparent =
                    selected_transparent |
                    (is_transparent ? choose : Bool{false});
                selected_translucent =
                    selected_translucent |
                    (is_translucent ? choose : Bool{false});
                selected_glossy =
                    selected_glossy |
                    ((!is_transparent) & choose &
                     sample_glossy);
                selected = selected | choose;
                accumulated = next;
            });

        auto diffuse_evaluation = evaluate_traced(
            services, values, point, result.wi, query);
        auto reflection_geometric_valid =
            dot(point.geometric_normal, result.wi) > 0.0f;
        auto transmission_geometric_valid =
            dot(point.geometric_normal, result.wi) < 0.0f;
        auto geometric_valid = select(
            reflection_geometric_valid,
            transmission_geometric_valid,
            selected_translucent);
        auto diffuse_valid =
            selected & (!selected_transparent) & geometric_valid;
        auto transparent_valid =
            selected & selected_transparent;
        result.valid = diffuse_valid | transparent_valid;
        result.evaluation.f = select(
            diffuse_evaluation.f,
            transparent_weight * 1.0e6f,
            selected_transparent);
        result.evaluation.pdf = select(
            diffuse_evaluation.pdf,
            1.0e6f * transparent_sample_weight /
                max(total_weight, 1.0e-20f),
            selected_transparent);
        result.evaluation.diffuse_f = select(
            diffuse_evaluation.diffuse_f,
            make_float3(0.0f),
            selected_transparent);
        result.evaluation.diffuse_pdf = select(
            diffuse_evaluation.diffuse_pdf,
            0.0f,
            selected_transparent);
        auto sampled_surface_events = select(
            static_cast<std::uint32_t>(
                event_diffuse | event_reflection),
            static_cast<std::uint32_t>(
                event_glossy | event_reflection),
            selected_glossy);
        sampled_surface_events = select(
            sampled_surface_events,
            static_cast<std::uint32_t>(
                event_diffuse | event_transmission),
            selected_translucent);
        result.evaluation.events = select(
            sampled_surface_events,
            static_cast<std::uint32_t>(
                event_transmission | event_transparent),
            selected_transparent);
        result.eta = 1.0f;
        return result;
    }

    [[nodiscard]] Float3 emission(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3>) const noexcept override {
        if (!_program) {
            return make_float3(0.0f);
        }
        auto values = trace_values(services, point);
        Float3 result = make_float3(0.0f);
        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                if (closure.operation ==
                    compiler::ClosureOperation::emission) {
                    result += closure.weight;
                }
            });
        return result;
    }

    [[nodiscard]] Float3 transparent_extinction(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept override {
        if (!_program) {
            return make_float3(0.0f);
        }
        auto values = trace_values(services, point);
        Float3 result = make_float3(0.0f);
        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                if (closure.operation ==
                    compiler::ClosureOperation::transparent) {
                    result += closure.weight;
                }
            });
        return result;
    }

    [[nodiscard]] SurfaceAov aov(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept override {
        auto result = SurfaceAov{
            .albedo = make_float3(0.0f),
            .roughness = make_float2(0.0f),
            .normal = point.shading_normal,
            .transparency = make_float3(0.0f)};
        if (!_program) {
            return result;
        }
        auto values = trace_values(services, point);
        Float total_weight = 0.0f;
        Float roughness = 0.0f;
        Float3 normal = make_float3(0.0f);
        auto incoming = safe_normalize(
            point.incoming,
            point.shading_normal);
        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                if (closure.operation ==
                    compiler::ClosureOperation::transparent) {
                    result.transparency += closure.weight;
                    return;
                }
                const auto is_diffuse =
                    closure.operation ==
                    compiler::ClosureOperation::diffuse;
                const auto is_translucent =
                    closure.operation ==
                    compiler::ClosureOperation::translucent;
                const auto is_principled =
                    closure.operation ==
                    compiler::ClosureOperation::principled;
                const auto is_glossy =
                    closure.operation ==
                    compiler::ClosureOperation::glossy;
                if (!is_diffuse && !is_translucent &&
                    !is_principled &&
                    !is_glossy) {
                    return;
                }
                Float3 albedo;
                if (is_principled) {
                    albedo = principled_state(
                                 services,
                                 closure,
                                 incoming)
                                 .diffuse_albedo;
                } else {
                    albedo =
                        is_diffuse || is_translucent
                            ? closure.weight
                            : closure.weight * closure.color;
                }
                auto weight = sample_weight(albedo);
                total_weight += weight;
                result.albedo += albedo;
                roughness += weight * closure.roughness;
                normal += weight * closure.normal;
            });
        auto valid = total_weight > 0.0f;
        result.roughness = make_float2(select(
            0.0f,
            roughness / max(total_weight, 1.0e-20f),
            valid));
        result.normal = safe_normalize(
            select(
                point.shading_normal,
                normal,
                valid),
            point.shading_normal);
        return result;
    }
};

}// namespace psycles::luisa_backend
