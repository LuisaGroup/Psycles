// GraphSurface state, value accessors, and color-space utilities.
// Included by <psycles/luisa/graph_surface.h>; not a standalone header.

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
        // Cycles stores fabsf(average(weight)) in ShaderClosure. Ordinary
        // BSDF allocation has already clamped negative spectral components;
        // transparent closure setup applies the absolute value directly.
        return abs((value.x + value.y + value.z) / 3.0f);
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
