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
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_color_nodes.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_sample_mapping.h>
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
        Float3 glossy_closure_weight;
        Float3 diffuse_albedo;
    };

    struct ClosureSelectionState {
        Bool eligible;
        Float weight;
        Float principled_specular_chance;
        Float3 glossy_normal;
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

    [[nodiscard]] static Float3 bsdf_allocated_weight(
        Float3 value) noexcept {
        // Cycles bsdf_alloc() removes negative spectral weight and skips
        // closures whose average weight is below CLOSURE_WEIGHT_CUTOFF.
        value = max(value, make_float3(0.0f));
        auto average = (value.x + value.y + value.z) / 3.0f;
        return select(
            make_float3(0.0f),
            value,
            average >= 1.0e-5f);
    }

    [[nodiscard]] static Float pass_weight(Float3 value) noexcept {
        // Cycles data passes use fabsf(average(sc->weight)), which differs
        // from the lobe-selection weight when signed closure weights are
        // present.
        return abs((value.x + value.y + value.z) / 3.0f);
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

    [[nodiscard]] static Float3
    ensure_valid_specular_reflection(
        Float3 geometric_normal,
        Float3 incoming,
        Float3 shading_normal) noexcept {
        auto reflected =
            2.0f * dot(shading_normal, incoming) *
                shading_normal -
            incoming;
        auto incoming_geometric_cosine =
            max(dot(incoming, geometric_normal), 0.0f);
        auto threshold = min(
            0.9f * incoming_geometric_cosine, 0.01f);
        auto reflection_is_valid =
            dot(geometric_normal, reflected) >= threshold;

        auto tangent = safe_normalize(
            shading_normal -
                dot(shading_normal, geometric_normal) *
                    geometric_normal,
            shading_normal);
        auto incoming_tangent = dot(incoming, tangent);
        auto a =
            incoming_tangent * incoming_tangent +
            incoming_geometric_cosine *
                incoming_geometric_cosine;
        auto b =
            2.0f *
            (a + incoming_geometric_cosine * threshold);
        auto c =
            (threshold + incoming_geometric_cosine) *
            (threshold + incoming_geometric_cosine);
        auto discriminant = max(
            b * b - 4.0f * a * c, 0.0f);
        auto signed_root = select(
            b - sqrt(discriminant),
            b + sqrt(discriminant),
            incoming_tangent < 0.0f);
        auto normal_z_squared = clamp(
            0.25f * signed_root / max(a, 1.0e-20f),
            0.0f,
            1.0f);
        auto corrected = safe_normalize(
            sqrt(max(1.0f - normal_z_squared, 0.0f)) *
                    tangent +
                sqrt(normal_z_squared) * geometric_normal,
            geometric_normal);
        return select(
            shading_normal,
            corrected,
            (!reflection_is_valid) & (a > 1.0e-20f));
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
        Float3 incoming,
        Float3 glossy_normal) noexcept {
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
            dot(glossy_normal, incoming), 0.0f, 1.0f);
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
        // Cycles' Normal and Roughness passes weight each closure by its
        // actual closure weight, not by the estimated albedo used for BSDF
        // sampling. The microfacet energy compensation darkens that weight
        // for MULTI_GGX; plain GGX leaves it unchanged.
        auto glossy_closure_weight =
            closure.weight *
            (metallic * metallic_energy.darkening +
             dielectric_weight * dielectric_energy.darkening);
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
            .glossy_closure_weight = glossy_closure_weight,
            .diffuse_albedo = diffuse_albedo};
    }

    [[nodiscard]] static bool is_scattering_operation(
        compiler::ClosureOperation operation) noexcept {
        switch (operation) {
            case compiler::ClosureOperation::diffuse:
            case compiler::ClosureOperation::translucent:
            case compiler::ClosureOperation::principled:
            case compiler::ClosureOperation::glossy:
            case compiler::ClosureOperation::transparent:
                return true;
            case compiler::ClosureOperation::null_closure:
            case compiler::ClosureOperation::add:
            case compiler::ClosureOperation::mix:
            case compiler::ClosureOperation::emission:
                return false;
        }
        return false;
    }

    [[nodiscard]] static Float closure_sample_weight(
        const TracedClosure &closure) noexcept {
        return sample_weight(closure.weight);
    }

    [[nodiscard]] static Bool closure_allocated(
        const TracedClosure &closure) noexcept {
        if (!is_scattering_operation(closure.operation)) {
            return false;
        }
        return closure_sample_weight(closure) > 0.0f;
    }

    [[nodiscard]] static UInt cycles_closure_type(
        const TracedClosure &closure) noexcept {
        switch (closure.operation) {
            case compiler::ClosureOperation::diffuse:
                return select(
                    cycles_closure::type_oren_nayar,
                    cycles_closure::type_diffuse,
                    closure.roughness < 1.0e-5f);
            case compiler::ClosureOperation::translucent:
                return cycles_closure::type_translucent;
            case compiler::ClosureOperation::glossy:
                return cycles_closure::type_microfacet_ggx;
            case compiler::ClosureOperation::transparent:
                return cycles_closure::type_transparent;
            case compiler::ClosureOperation::principled:
                // This virtual ID makes the still-aggregated Principled
                // representation visible to the differential oracle. Cycles
                // expands it into physical closures before sampling, so a
                // comparison cannot accidentally pass until Psycles does the
                // same.
                return cycles_closure::
                    type_principled_virtual;
            case compiler::ClosureOperation::null_closure:
            case compiler::ClosureOperation::add:
            case compiler::ClosureOperation::mix:
            case compiler::ClosureOperation::emission:
                return cycles_closure::type_none;
        }
        return cycles_closure::type_none;
    }

    [[nodiscard]] static ClosureSelectionState
    closure_selection_state(
        const ShaderServices &services,
        const SurfacePoint &point,
        const TracedClosure &closure,
        Float3 incoming,
        const SurfaceQuery &query) noexcept {
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
        const auto is_transparent =
            closure.operation ==
            compiler::ClosureOperation::transparent;
        const auto diffuse_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(
                 event_diffuse)) != 0u;
        const auto glossy_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(
                 event_glossy)) != 0u;
        const auto transparent_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(
                 event_transparent)) != 0u;
        const auto transmission_enabled =
            (query.lobe_mask &
             static_cast<std::uint32_t>(
                 event_transmission)) != 0u;
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
        eligible &= closure_allocated(closure);
        const auto glossy_normal =
            ensure_valid_specular_reflection(
                point.geometric_normal,
                incoming,
                closure.normal);
        Float3 selection_color;
        Float principled_specular_chance = 0.0f;
        if (is_principled) {
            const auto state = principled_state(
                services,
                closure,
                incoming,
                glossy_normal);
            selection_color =
                state.diffuse_sample_weight +
                state.glossy_sample_weight;
            const auto diffuse_weight = sample_weight(
                state.diffuse_sample_weight);
            const auto glossy_weight = sample_weight(
                state.glossy_sample_weight);
            principled_specular_chance =
                glossy_weight /
                max(
                    diffuse_weight + glossy_weight,
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
        return {
            .eligible = eligible,
            .weight = select(
                0.0f,
                sample_weight(selection_color),
                eligible),
            .principled_specular_chance =
                principled_specular_chance,
            .glossy_normal = glossy_normal};
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

        auto use_lambert =
            closure.operation ==
                    compiler::ClosureOperation::principled
                ? sigma < 1.0e-5f
                : sigma == 0.0f;
        return select(oren_nayar, lambert, use_lambert);
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

    [[nodiscard]] static Float microfacet_alpha(
        const TracedClosure &closure,
        Float glossy_filter_roughness) noexcept {
        // Cycles applies bsdf_microfacet_blur after closure setup. Keep the
        // original closure roughness for sample weights, layering, and energy
        // compensation; only evaluation and sampling see this widened alpha.
        auto setup_alpha =
            clamp(closure.roughness, 0.0f, 1.0f);
        setup_alpha *= setup_alpha;
        return max(
            max(setup_alpha, glossy_filter_roughness),
            1.0e-3f);
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
        Float3 outgoing,
        Float3 glossy_normal,
        Float glossy_filter_roughness) noexcept {
        auto n_dot_v =
            max(dot(glossy_normal, incoming), 0.0f);
        auto n_dot_l =
            max(dot(glossy_normal, outgoing), 0.0f);
        auto half_vector = safe_normalize(
            incoming + outgoing,
            glossy_normal);
        auto n_dot_h =
            max(dot(glossy_normal, half_vector), 0.0f);
        auto v_dot_h =
            max(dot(incoming, half_vector), 0.0f);
        auto alpha = microfacet_alpha(
            closure, glossy_filter_roughness);
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
                principled_state(
                    services,
                    closure,
                    incoming,
                    glossy_normal);
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
        Float3 outgoing,
        Float3 glossy_normal,
        Float glossy_filter_roughness) noexcept {
        auto half_vector = safe_normalize(
            incoming + outgoing,
            glossy_normal);
        auto n_dot_h =
            max(dot(glossy_normal, half_vector), 0.0f);
        auto v_dot_h =
            max(dot(incoming, half_vector), 0.0f);
        auto alpha = microfacet_alpha(
            closure, glossy_filter_roughness);
        auto n_dot_v =
            max(dot(glossy_normal, incoming), 0.0f);
        auto n_dot_l =
            max(dot(glossy_normal, outgoing), 0.0f);
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
        Float2 random,
        Float3 glossy_normal,
        Float glossy_filter_roughness) noexcept {
        auto alpha = microfacet_alpha(
            closure, glossy_filter_roughness);
        auto normal = safe_normalize(
            glossy_normal,
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
        return cycles_sample_mapping::
            sample_cosine_hemisphere(normal, random)
                .direction;
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

    [[nodiscard]] static Float3 rotate_euler_transposed(
        Float3 value,
        Float3 rotation) noexcept {
        auto sx = sin(rotation.x);
        auto cx = cos(rotation.x);
        auto sy = sin(rotation.y);
        auto cy = cos(rotation.y);
        auto sz = sin(rotation.z);
        auto cz = cos(rotation.z);
        return make_float3(
            cy * cz * value.x +
                cy * sz * value.y -
                sy * value.z,
            (sy * sx * cz - cx * sz) * value.x +
                (sy * sx * sz + cx * cz) * value.y +
                cy * sx * value.z,
            (sy * cx * cz + sx * sz) * value.x +
                (sy * cx * sz - sx * cz) * value.y +
                cy * cx * value.z);
    }

    [[nodiscard]] static Float3 safe_divide_components(
        Float3 numerator,
        Float3 denominator) noexcept {
        return make_float3(
            select(
                0.0f,
                numerator.x / denominator.x,
                denominator.x != 0.0f),
            select(
                0.0f,
                numerator.y / denominator.y,
                denominator.y != 0.0f),
            select(
                0.0f,
                numerator.z / denominator.z,
                denominator.z != 0.0f));
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
                            instruction.static_u1, point);
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
                case compiler::ValueOperation::image_color:
                case compiler::ValueOperation::image_alpha: {
                    const auto extension =
                        static_cast<std::uint32_t>(
                            instruction.static_u1 & 0xffu);
                    const auto interpolation =
                        static_cast<std::uint32_t>(
                            (instruction.static_u1 >> 10u) &
                            0x03u);
                    const auto projection =
                        static_cast<std::uint32_t>(
                            (instruction.static_u1 >> 12u) &
                            0x03u);
                    const auto unassociate_alpha =
                        ((instruction.static_u1 >> 9u) & 1u) !=
                        0u;
                    const auto encoded_as_srgb =
                        ((instruction.static_u1 >> 8u) & 1u) !=
                        0u;
                    const auto decode_sample =
                        [&](Float4 sampled) noexcept {
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
                        if (encoded_as_srgb) {
                            // Match Cycles' svm_image_texture ordering:
                            // filter associated encoded texels, optionally
                            // unassociate, then decode sRGB.
                            sampled = make_float4(
                                srgb_to_linear(sampled.xyz()),
                                sampled.w);
                        }
                        return sampled;
                    };
                    const auto sample_uv =
                        [&](Float2 uv) noexcept {
                        // Blender UVs use a bottom-left origin while decoded
                        // host images are uploaded in top-to-bottom row
                        // order.
                        uv.y = 1.0f - uv.y;
                        return decode_sample(
                            services.texture_2d(
                                static_cast<std::uint32_t>(
                                    instruction.static_u0),
                                uv,
                                make_float2(0.0f),
                                make_float2(0.0f),
                                interpolation,
                                extension));
                    };

                    auto coordinate =
                        vector(instruction.a, result);
                    Float4 sampled;
                    if (projection == 1u) {
                        // Cycles' object-normal weighted box projection.
                        auto signed_normal =
                            point.object_shading_normal;
                        auto normal = abs(signed_normal);
                        auto normal_sum =
                            normal.x + normal.y + normal.z;
                        normal /= select(
                            1.0f,
                            normal_sum,
                            normal_sum != 0.0f);
                        Float3 weight =
                            make_float3(0.0f);
                        const auto blend =
                            instruction.static_f0;
                        const auto limit =
                            0.5f * (1.0f + blend);
                        $if ((normal.x >
                              limit *
                                  (normal.x + normal.y)) &
                             (normal.x >
                              limit *
                                  (normal.x + normal.z))) {
                            weight.x = 1.0f;
                        }
                        $elif ((normal.y >
                                limit *
                                    (normal.x + normal.y)) &
                               (normal.y >
                                limit *
                                    (normal.y + normal.z))) {
                            weight.y = 1.0f;
                        }
                        $elif ((normal.z >
                                limit *
                                    (normal.x + normal.z)) &
                               (normal.z >
                                limit *
                                    (normal.y + normal.z))) {
                            weight.z = 1.0f;
                        }
                        $elif (blend > 0.0f) {
                            $if (
                                normal.z <
                                (1.0f - limit) *
                                    (normal.y + normal.x)) {
                                weight.x =
                                    normal.x /
                                    (normal.x + normal.y);
                                weight.x = luisa::compute::clamp(
                                    (weight.x -
                                     0.5f *
                                         (1.0f - blend)) /
                                        blend,
                                    0.0f,
                                    1.0f);
                                weight.y = 1.0f - weight.x;
                            }
                            $elif (
                                normal.x <
                                (1.0f - limit) *
                                    (normal.y + normal.z)) {
                                weight.y =
                                    normal.y /
                                    (normal.y + normal.z);
                                weight.y = luisa::compute::clamp(
                                    (weight.y -
                                     0.5f *
                                         (1.0f - blend)) /
                                        blend,
                                    0.0f,
                                    1.0f);
                                weight.z = 1.0f - weight.y;
                            }
                            $elif (
                                normal.y <
                                (1.0f - limit) *
                                    (normal.x + normal.z)) {
                                weight.x =
                                    normal.x /
                                    (normal.x + normal.z);
                                weight.x = luisa::compute::clamp(
                                    (weight.x -
                                     0.5f *
                                         (1.0f - blend)) /
                                        blend,
                                    0.0f,
                                    1.0f);
                                weight.z = 1.0f - weight.x;
                            }
                            $else {
                                weight.x =
                                    ((2.0f - limit) *
                                         normal.x +
                                     (limit - 1.0f)) /
                                    (2.0f * limit - 1.0f);
                                weight.y =
                                    ((2.0f - limit) *
                                         normal.y +
                                     (limit - 1.0f)) /
                                    (2.0f * limit - 1.0f);
                                weight.z =
                                    ((2.0f - limit) *
                                         normal.z +
                                     (limit - 1.0f)) /
                                    (2.0f * limit - 1.0f);
                            };
                        }
                        $else {
                            weight.x = 1.0f;
                        };

                        auto uv_x = make_float2(
                            select(
                                coordinate.y,
                                1.0f - coordinate.y,
                                signed_normal.x < 0.0f),
                            coordinate.z);
                        auto uv_y = make_float2(
                            select(
                                coordinate.x,
                                1.0f - coordinate.x,
                                signed_normal.y > 0.0f),
                            coordinate.z);
                        auto uv_z = make_float2(
                            select(
                                coordinate.y,
                                1.0f - coordinate.y,
                                signed_normal.z > 0.0f),
                            coordinate.x);
                        sampled =
                            weight.x * sample_uv(uv_x) +
                            weight.y * sample_uv(uv_y) +
                            weight.z * sample_uv(uv_z);
                    } else {
                        Float2 uv = coordinate.xy();
                        if (projection == 2u) {
                            auto direction =
                                (coordinate - 0.5f) * 2.0f;
                            auto length_squared =
                                dot(direction, direction);
                            Float2 spherical =
                                make_float2(0.0f);
                            $if (length_squared > 0.0f) {
                                Float u = 0.0f;
                                $if ((direction.x != 0.0f) |
                                     (direction.y != 0.0f)) {
                                    u =
                                        0.5f -
                                        atan2(
                                            direction.x,
                                            direction.y) /
                                            (2.0f * pi);
                                };
                                auto z = luisa::compute::clamp(
                                    direction.z /
                                        sqrt(length_squared),
                                    -1.0f,
                                    1.0f);
                                spherical = make_float2(
                                    u,
                                    1.0f -
                                        acos(z) / pi);
                            };
                            uv = spherical;
                        } else if (projection == 3u) {
                            auto direction =
                                (coordinate - 0.5f) * 2.0f;
                            auto radial_length = sqrt(
                                direction.x * direction.x +
                                direction.y * direction.y);
                            Float2 tube =
                                make_float2(0.0f);
                            $if (radial_length > 0.0f) {
                                tube = make_float2(
                                    (1.0f -
                                     atan2(
                                         direction.x /
                                             radial_length,
                                         direction.y /
                                             radial_length) /
                                         pi) *
                                        0.5f,
                                    (direction.z + 1.0f) *
                                        0.5f);
                            };
                            uv = tube;
                        }
                        sampled = sample_uv(uv);
                    }
                    value =
                        instruction.operation ==
                                compiler::ValueOperation::
                                    image_color
                            ? sampled
                            : make_float4(sampled.w);
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
                            instruction.static_u0 & 0xffu);
                    auto object_tangent =
                        point.object_tangent;
                    auto tangent_sign =
                        point.tangent_sign;
                    if ((instruction.static_u0 & 0x100u) !=
                        0u) {
                        auto named_tangent =
                            services.attribute(
                                instruction.static_u1,
                                point);
                        object_tangent =
                            named_tangent.xyz();
                        tangent_sign =
                            named_tangent.w;
                    }
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
                            tangent_sign *
                            cross(
                                point.object_shading_normal,
                                object_tangent);
                        auto object_normal = safe_normalize(
                            object_tangent * mapped.x +
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
                                 object_tangent) >
                             1.0e-20f) &
                            (abs(tangent_sign) >
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
                    auto normal_in =
                        (instruction.static_u0 & 2u) != 0u
                            ? vector(instruction.e, result)
                            : result.shading_normal;
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
                    if ((instruction.static_u0 & 1u) != 0u) {
                        distance = -distance;
                    }
                    auto determinant_sign = select(
                        -1.0f,
                        1.0f,
                        determinant >= 0.0f);
                    auto perturbed_vector =
                        filter_width * abs(determinant) *
                            normal_in -
                        distance * determinant_sign *
                            surface_gradient;
                    auto perturbed_valid =
                        length_squared(perturbed_vector) >
                        0.0f;
                    auto perturbed = safe_normalize(
                        perturbed_vector,
                        make_float3(0.0f));
                    auto strength =
                        max(scalar(instruction.b, result), 0.0f);
                    auto blended = safe_normalize(
                        strength * perturbed +
                            (1.0f - strength) * normal_in,
                        make_float3(0.0f));
                    auto normal_out = select(
                        normal_in, blended, perturbed_valid);
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
                case compiler::ValueOperation::checker_color:
                case compiler::ValueOperation::checker_factor: {
                    auto scaled_raw =
                        vector(instruction.a, result) *
                        scalar(instruction.d, result);
                    auto p = select(
                        scaled_raw,
                        make_float3(0.0f),
                        luisa::compute::dsl::isnan(
                            scaled_raw));
                    // Cycles computes `(p + 1e-6f) * 0.999999f`.
                    // Express the second factor as `1 - 0x1.1p-20f`,
                    // which is bit-identical in float32 but remains below
                    // the unit boundary even if a backend contracts the
                    // arithmetic.
                    auto shifted_raw = p + 0.000001f;
                    auto shifted = select(
                        shifted_raw,
                        make_float3(0.0f),
                        luisa::compute::dsl::isnan(
                            shifted_raw));
                    p =
                        shifted -
                        shifted * 0x1.1p-20f;
                    auto xi = abs(cast<int>(floor(p.x)));
                    auto yi = abs(cast<int>(floor(p.y)));
                    auto zi = abs(cast<int>(floor(p.z)));
                    auto xy_same = select(
                        0,
                        1,
                        (xi % 2) == (yi % 2));
                    auto is_first =
                        xy_same == (zi % 2);
                    auto factor = select(
                        0.0f, 1.0f, is_first);
                    if (instruction.operation ==
                        compiler::ValueOperation::
                            checker_color) {
                        value = make_float4(
                            lerp(
                                vector(
                                    instruction.c, result),
                                vector(
                                    instruction.b, result),
                                factor),
                            1.0f);
                    } else {
                        value = make_float4(factor);
                    }
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
                    if (
                        count >= 2u &&
                        (instruction.static_u0 & 2u) != 0u) {
                        std::vector<luisa::float4> samples;
                        samples.reserve(count);
                        for (std::size_t i = 0u;
                             i < count;
                             ++i) {
                            samples.emplace_back(
                                instruction.static_table[
                                    i * 5u + 1u],
                                instruction.static_table[
                                    i * 5u + 2u],
                                instruction.static_table[
                                    i * 5u + 3u],
                                instruction.static_table[
                                    i * 5u + 4u]);
                        }
                        luisa::compute::Constant<luisa::float4>
                            table{samples};
                        auto scaled =
                            clamp(factor, 0.0f, 1.0f) *
                            static_cast<float>(count - 1u);
                        auto index = min(
                            cast<luisa::uint>(scaled),
                            static_cast<std::uint32_t>(
                                count - 1u));
                        auto t =
                            scaled - cast<float>(index);
                        auto sampled = table.read(index);
                        if ((instruction.static_u0 & 1u) == 0u) {
                            auto next = table.read(min(
                                index + 1u,
                                static_cast<std::uint32_t>(
                                    count - 1u)));
                            sampled = select(
                                sampled,
                                lerp(sampled, next, t),
                                t > 0.0f);
                        }
                        color = sampled.xyz();
                        alpha = sampled.w;
                    } else if (count != 0u) {
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
                            if (
                                (instruction.static_u0 & 1u) !=
                                0u) {
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
                    auto input = vector(instruction.a, result);
                    auto factor = scalar(instruction.b, result);
                    Float3 mapped = input;
                    const auto count =
                        instruction.static_table.size() / 4u;
                    if (
                        count >= 2u &&
                        (instruction.static_u0 & 1u) != 0u) {
                        std::vector<luisa::float3> samples;
                        samples.reserve(count);
                        for (std::size_t i = 0u;
                             i < count;
                             ++i) {
                            samples.emplace_back(
                                instruction.static_table[
                                    i * 4u + 1u],
                                instruction.static_table[
                                    i * 4u + 2u],
                                instruction.static_table[
                                    i * 4u + 3u]);
                        }
                        luisa::compute::Constant<luisa::float3>
                            table{samples};
                        const auto component =
                            [](Float3 value,
                               std::uint32_t channel) {
                                return channel == 0u
                                           ? value.x
                                           : channel == 1u
                                                 ? value.y
                                                 : value.z;
                            };
                        const auto lookup =
                            [&](Float coordinate,
                                std::uint32_t channel) {
                                auto scaled =
                                    clamp(
                                        coordinate,
                                        0.0f,
                                        1.0f) *
                                    static_cast<float>(
                                        count - 1u);
                                auto index = min(
                                    cast<luisa::uint>(scaled),
                                    static_cast<std::uint32_t>(
                                        count - 1u));
                                auto t =
                                    scaled - cast<float>(index);
                                auto sampled = component(
                                    table.read(index), channel);
                                auto next = component(
                                    table.read(min(
                                        index + 1u,
                                        static_cast<
                                            std::uint32_t>(
                                            count - 1u))),
                                    channel);
                                sampled = select(
                                    sampled,
                                    lerp(sampled, next, t),
                                    t > 0.0f);
                                if (
                                    (instruction.static_u0 &
                                     2u) != 0u) {
                                    auto first = component(
                                        table.read(0u),
                                        channel);
                                    auto second = component(
                                        table.read(1u),
                                        channel);
                                    auto last = component(
                                        table.read(
                                            static_cast<
                                                std::uint32_t>(
                                                count - 1u)),
                                        channel);
                                    auto previous = component(
                                        table.read(
                                            static_cast<
                                                std::uint32_t>(
                                                count - 2u)),
                                        channel);
                                    auto below =
                                        first +
                                        (first - second) *
                                            (-coordinate) *
                                            static_cast<float>(
                                                count - 1u);
                                    auto above =
                                        last +
                                        (last - previous) *
                                            (coordinate - 1.0f) *
                                            static_cast<float>(
                                                count - 1u);
                                    sampled = select(
                                        sampled,
                                        below,
                                        coordinate < 0.0f);
                                    sampled = select(
                                        sampled,
                                        above,
                                        coordinate > 1.0f);
                                }
                                return sampled;
                            };
                        const auto range =
                            instruction.static_f1 -
                            instruction.static_f0;
                        auto relative =
                            (input -
                             instruction.static_f0) /
                            range;
                        mapped = make_float3(
                            lookup(relative.x, 0u),
                            lookup(relative.y, 1u),
                            lookup(relative.z, 2u));
                    } else if (count >= 2u) {
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
                        vector(instruction.i, result),
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
                case compiler::ClosureOperation::null_closure:
                    return;
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
                        .weight = bsdf_allocated_weight(
                            color * mix_weight),
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
                                ? bsdf_allocated_weight(
                                      color * mix_weight)
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
                auto glossy_normal =
                    ensure_valid_specular_reflection(
                        point.geometric_normal,
                        incoming,
                        closure.normal);
                auto diffuse_normal =
                    is_translucent
                        ? -glossy_normal
                        : closure.normal;
                auto diffuse_pdf =
                    max(dot(diffuse_normal, outgoing), 0.0f) *
                    inverse_pi;
                auto glossy_pdf = microfacet_pdf(
                    closure,
                    incoming,
                    outgoing,
                    glossy_normal,
                    query.glossy_filter_roughness);
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
                        services,
                        closure,
                        incoming,
                        glossy_normal);
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
                            outgoing,
                            glossy_normal,
                            query.glossy_filter_roughness);
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
                            outgoing,
                            glossy_normal,
                            query.glossy_filter_roughness);
                    selection_color =
                        closure.weight *
                        max(
                            closure.color,
                            make_float3(0.04f));
                } else if (is_translucent) {
                    specular_chance = 0.0f;
                    auto cosine =
                        max(
                            dot(-glossy_normal, outgoing),
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

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<std::uint32_t>
            requested_index_expression) const noexcept override {
        auto requested_index =
            UInt{requested_index_expression};
        auto result =
            SurfaceClosureTrace::zero(requested_index);
        if (!_program) {
            return result;
        }
        auto values = trace_values(services, point);
        UInt closure_count = 0u;
        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                auto allocated =
                    closure_allocated(closure);
                auto match =
                    allocated &
                    (closure_count == requested_index);
                result.type = select(
                    result.type,
                    cycles_closure_type(closure),
                    match);
                result.sample_weight = select(
                    result.sample_weight,
                    closure_sample_weight(closure),
                    match);
                result.weight = select(
                    result.weight,
                    closure.weight,
                    match);
                result.normal = select(
                    result.normal,
                    closure.normal,
                    match);
                result.valid = result.valid | match;
                closure_count +=
                    select(0u, 1u, allocated);
            });
        result.count = closure_count;
        return result;
    }

private:
    template<bool TraceSelection>
    [[nodiscard]] SurfaceSampleTrace
    sample_with_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe_expression,
        Expr<luisa::float2> u_direction_expression,
        const SurfaceQuery &query) const noexcept {
        auto trace = SurfaceSampleTrace::zero();
        auto &result = trace.sample;
        if (!_program) {
            return trace;
        }

        auto values = trace_values(services, point);
        auto incoming = safe_normalize(
            point.incoming,
            point.shading_normal);
        Float total_weight = 0.0f;
        UInt closure_count = 0u;
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
                const auto selection =
                    closure_selection_state(
                        services,
                        point,
                        closure,
                        incoming,
                        query);
                total_weight += selection.weight;
                closure_count += select(
                    0u,
                    1u,
                    closure_allocated(closure));
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
        UInt closure_index = 0u;

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
                const auto allocated =
                    closure_allocated(closure);
                const auto current_closure_index =
                    closure_index;
                const auto selection =
                    closure_selection_state(
                        services,
                        point,
                        closure,
                        incoming,
                        query);
                const auto glossy_normal =
                    selection.glossy_normal;
                const auto weight = selection.weight;
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
                              : selection
                                    .principled_specular_chance;
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
                            ? -glossy_normal
                            : closure.normal,
                        remapped_random);
                auto glossy_direction = sample_ggx(
                    closure,
                    incoming,
                    remapped_random,
                    glossy_normal,
                    query.glossy_filter_roughness);
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
                auto nontransparent_roughness =
                    select(
                        make_float2(1.0f),
                        make_float2(closure.roughness),
                        sample_glossy);
                result.roughness = select(
                    result.roughness,
                    is_transparent
                        ? make_float2(0.0f)
                        : nontransparent_roughness,
                    choose);
                if constexpr (TraceSelection) {
                    const auto rescaled_selection =
                        select(
                            random_lobe,
                            (target - accumulated) /
                                max(weight, 1.0e-20f),
                            closure_count > 1u);
                    trace.closure_index = select(
                        trace.closure_index,
                        current_closure_index,
                        choose);
                    trace.closure_type = select(
                        trace.closure_type,
                        cycles_closure_type(closure),
                        choose);
                    trace.closure_sample_weight = select(
                        trace.closure_sample_weight,
                        closure_sample_weight(closure),
                        choose);
                    trace.selection_rescaled = select(
                        trace.selection_rescaled,
                        rescaled_selection,
                        choose);
                    trace.closure_weight = select(
                        trace.closure_weight,
                        closure.weight,
                        choose);
                    trace.closure_normal = select(
                        trace.closure_normal,
                        closure.normal,
                        choose);
                }
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
                closure_index += select(
                    0u, 1u, allocated);
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
        if constexpr (TraceSelection) {
            trace.closure_valid = selected;
        }
        return trace;
    }

public:
    [[nodiscard]] SurfaceSample sample(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe_expression,
        Expr<luisa::float2> u_direction_expression,
        const SurfaceQuery &query) const noexcept override {
        return sample_with_trace<false>(
                   services,
                   point,
                   u_lobe_expression,
                   u_direction_expression,
                   query)
            .sample;
    }

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe_expression,
        Expr<luisa::float2> u_direction_expression,
        const SurfaceQuery &query) const noexcept override {
        return sample_with_trace<true>(
            services,
            point,
            u_lobe_expression,
            u_direction_expression,
            query);
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

    [[nodiscard]] Float3 shading_normal(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept override {
        if (!_program) {
            return point.shading_normal;
        }
        return trace_values(services, point).shading_normal;
    }

    [[nodiscard]] SurfaceAov aov(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept override {
        auto result = SurfaceAov{
            .albedo = make_float3(0.0f),
            .glossy_albedo = make_float3(0.0f),
            .transmission_albedo = make_float3(0.0f),
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
                auto glossy_normal =
                    ensure_valid_specular_reflection(
                        point.geometric_normal,
                        incoming,
                        closure.normal);
                Float3 diffuse_albedo = make_float3(0.0f);
                Float diffuse_weight = 0.0f;
                Float glossy_weight = 0.0f;
                if (is_principled) {
                    const auto state = principled_state(
                        services,
                        closure,
                        incoming,
                        glossy_normal);
                    diffuse_albedo = state.diffuse_albedo;
                    result.glossy_albedo +=
                        state.glossy_sample_weight;
                    diffuse_weight =
                        pass_weight(state.diffuse_albedo);
                    glossy_weight =
                        pass_weight(
                            state.glossy_closure_weight);
                } else if (is_diffuse || is_translucent) {
                    diffuse_albedo = closure.weight;
                    diffuse_weight =
                        pass_weight(closure.weight);
                } else {
                    auto glossy_albedo =
                        closure.weight *
                        max(closure.color, make_float3(0.0f));
                    result.glossy_albedo += glossy_albedo;
                    glossy_weight =
                        pass_weight(closure.weight);
                }
                const auto weight =
                    diffuse_weight + glossy_weight;
                total_weight += weight;
                // Cycles Diffuse Color includes only diffuse/BSSRDF
                // closures. Glossy closure weights still contribute to the
                // Normal and Roughness passes, but never to Diffuse Color.
                result.albedo += diffuse_albedo;
                roughness += weight * closure.roughness;
                normal +=
                    diffuse_weight *
                        (is_translucent
                             ? glossy_normal
                             : closure.normal) +
                    glossy_weight * glossy_normal;
            });
        auto valid = total_weight > 0.0f;
        result.roughness = make_float2(select(
            1.0f,
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
