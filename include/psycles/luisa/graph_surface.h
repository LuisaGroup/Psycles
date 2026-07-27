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
        Float metallic;
        Float ior;
    };

private:
    std::shared_ptr<const compiler::SurfaceProgram> _program;
    std::uint32_t _parameter_block{};
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
            delta / max(abs(cmax), 1.0e-20f),
            cmax != 0.0f);
        auto safe_delta = max(abs(delta), 1.0e-20f);
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

        auto sigma = clamp(closure.roughness, 0.0f, 1.0f);
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
            closure.roughness < 1.0e-5f);
    }

    [[nodiscard]] static Float specular_probability(
        const TracedClosure &closure) noexcept {
        if (closure.operation ==
            compiler::ClosureOperation::glossy) {
            return 1.0f;
        }
        if (closure.operation !=
            compiler::ClosureOperation::principled) {
            return 0.0f;
        }
        return lerp(0.25f, 0.8f, closure.metallic);
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
        auto fresnel =
            specular_f0(closure) +
            (make_float3(1.0f) -
             specular_f0(closure)) *
                pow(1.0f - v_dot_h, 5.0f);
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

    [[nodiscard]] static Float3 srgb_to_linear(
        Float3 value) noexcept {
        auto low = value / 12.92f;
        auto high = pow(
            (value + 0.055f) / 1.055f,
            make_float3(2.4f));
        return select(low, high, value > 0.04045f);
    }

    [[nodiscard]] static Float hash_noise(Float3 p) noexcept {
        return fract(
            sin(dot(
                p,
                make_float3(
                    127.1f, 311.7f, 74.7f))) *
            43758.5453123f);
    }

    [[nodiscard]] static Float value_noise(Float3 p) noexcept {
        auto cell = floor(p);
        auto f = fract(p);
        auto u = f * f * (make_float3(3.0f) - 2.0f * f);
        auto n000 = hash_noise(cell);
        auto n100 = hash_noise(
            cell + make_float3(1.0f, 0.0f, 0.0f));
        auto n010 = hash_noise(
            cell + make_float3(0.0f, 1.0f, 0.0f));
        auto n110 = hash_noise(
            cell + make_float3(1.0f, 1.0f, 0.0f));
        auto n001 = hash_noise(
            cell + make_float3(0.0f, 0.0f, 1.0f));
        auto n101 = hash_noise(
            cell + make_float3(1.0f, 0.0f, 1.0f));
        auto n011 = hash_noise(
            cell + make_float3(0.0f, 1.0f, 1.0f));
        auto n111 = hash_noise(
            cell + make_float3(1.0f, 1.0f, 1.0f));
        auto x00 = lerp(n000, n100, u.x);
        auto x10 = lerp(n010, n110, u.x);
        auto x01 = lerp(n001, n101, u.x);
        auto x11 = lerp(n011, n111, u.x);
        return lerp(
            lerp(x00, x10, u.y),
            lerp(x01, x11, u.y),
            u.z);
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
                            _parameter_block,
                            instruction.parameter.value),
                        services.parameter_float(
                            _parameter_block,
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
                        (color.x + color.y + color.z) /
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
                    auto color = vector(instruction.a, result);
                    auto hue =
                        scalar(instruction.b, result) - 0.5f;
                    auto saturation = max(
                        scalar(instruction.c, result), 0.0f);
                    auto brightness =
                        scalar(instruction.d, result);
                    auto factor = clamp(
                        scalar(instruction.e, result),
                        0.0f,
                        1.0f);
                    auto axis = make_float3(
                        0.57735026919f);
                    auto angle = hue * two_pi;
                    auto rotated =
                        color * cos(angle) +
                        cross(axis, color) * sin(angle) +
                        axis * dot(axis, color) *
                            (1.0f - cos(angle));
                    auto luminance = dot(
                        rotated,
                        make_float3(
                            0.2126f, 0.7152f, 0.0722f));
                    auto adjusted =
                        (make_float3(luminance) +
                         (rotated -
                          make_float3(luminance)) *
                             saturation) *
                        brightness;
                    value = make_float4(
                        lerp(color, adjusted, factor),
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
                        auto color = sampled.xyz();
                        if (((instruction.static_u1 >> 8u) &
                             1u) != 0u) {
                            color = srgb_to_linear(color);
                        }
                        value = make_float4(color, sampled.w);
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
                    auto tangent_normal =
                        vector(instruction.a, result) * 2.0f -
                        1.0f;
                    auto strength = max(
                        scalar(instruction.b, result), 0.0f);
                    tangent_normal.x *= strength;
                    tangent_normal.y *= strength;
                    tangent_normal = safe_normalize(
                        tangent_normal,
                        make_float3(0.0f, 0.0f, 1.0f));
                    auto tangent = safe_normalize(
                        point.dpdu,
                        make_float3(1.0f, 0.0f, 0.0f));
                    auto bitangent = safe_normalize(
                        point.dpdv,
                        cross(point.shading_normal, tangent));
                    auto world = safe_normalize(
                        tangent * tangent_normal.x +
                            bitangent * tangent_normal.y +
                            point.shading_normal *
                                tangent_normal.z,
                        point.shading_normal);
                    value = make_float4(world, 0.0f);
                    break;
                }
                case compiler::ValueOperation::bump: {
                    auto normal_in = safe_normalize(
                        vector(instruction.e, result),
                        result.shading_normal);

                    auto point_x = point;
                    point_x.position =
                        point.position + point.dPdx;
                    point_x.object_position =
                        point.object_position +
                        point.object_dPdx;
                    point_x.generated =
                        point.generated + point.generated_dx;
                    point_x.uv = point.uv + point.uv_dx;
                    point_x.barycentric =
                        point.barycentric +
                        point.barycentric_dx;

                    auto point_y = point;
                    point_y.position =
                        point.position + point.dPdy;
                    point_y.object_position =
                        point.object_position +
                        point.object_dPdy;
                    point_y.generated =
                        point.generated + point.generated_dy;
                    point_y.uv = point.uv + point.uv_dy;
                    point_y.barycentric =
                        point.barycentric +
                        point.barycentric_dy;

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
                    auto filter_width = max(
                        scalar(instruction.d, result), 0.0f);
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
                    auto p = vector(instruction.a, result) *
                             scalar(instruction.b, result);
                    auto detail = clamp(
                        scalar(instruction.c, result),
                        0.0f,
                        8.0f);
                    auto roughness = clamp(
                        scalar(instruction.d, result),
                        0.0f,
                        1.0f);
                    Float amplitude = 1.0f;
                    Float frequency = 1.0f;
                    Float sum = 0.0f;
                    Float weight = 0.0f;
                    for (std::uint32_t octave = 0u;
                         octave < 8u;
                         ++octave) {
                        auto enabled =
                            detail >= static_cast<float>(octave);
                        auto contribution =
                            value_noise(p * frequency);
                        sum += select(
                            0.0f,
                            contribution * amplitude,
                            enabled);
                        weight += select(
                            0.0f, amplitude, enabled);
                        amplitude *= roughness;
                        frequency *= 2.0f;
                    }
                    auto noise =
                        sum / max(weight, 1.0e-20f);
                    value =
                        instruction.operation ==
                                compiler::ValueOperation::noise_color
                            ? make_float4(
                                  noise,
                                  value_noise(p + 19.19f),
                                  value_noise(p + 47.47f),
                                  1.0f)
                            : make_float4(noise);
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
                        gradient = length(p);
                    } else if (instruction.static_u0 == 6u) {
                        gradient = length(p);
                        gradient *= gradient;
                    }
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
                        vector(instruction.a, result).x);
                    break;
                case compiler::ValueOperation::separate_g:
                    value = make_float4(
                        vector(instruction.a, result).y);
                    break;
                case compiler::ValueOperation::separate_b:
                    value = make_float4(
                        vector(instruction.a, result).z);
                    break;
                case compiler::ValueOperation::combine_color:
                    value = make_float4(
                        scalar(instruction.a, result),
                        scalar(instruction.b, result),
                        scalar(instruction.c, result),
                        1.0f);
                    break;
                case compiler::ValueOperation::nishita_sky: {
                    auto direction = safe_normalize(
                        -point.incoming,
                        make_float3(0.0f, 0.0f, 1.0f));
                    auto elevation =
                        scalar(instruction.a, result);
                    auto rotation =
                        scalar(instruction.b, result);
                    auto sun_size = max(
                        scalar(instruction.c, result),
                        1.0e-5f);
                    auto sun_intensity = max(
                        scalar(instruction.d, result),
                        0.0f);
                    auto air = max(
                        scalar(instruction.f, result),
                        0.0f);
                    auto dust = max(
                        scalar(instruction.g, result),
                        0.0f);
                    auto ozone = max(
                        scalar(instruction.h, result),
                        0.0f);
                    auto sun_direction = make_float3(
                        -cos(elevation) * sin(rotation),
                        cos(elevation) * cos(rotation),
                        sin(elevation));
                    auto height = clamp(
                        direction.z, 0.0f, 1.0f);
                    auto horizon = make_float3(
                        0.42f, 0.55f, 0.78f) /
                        max(0.5f + 0.35f * dust, 0.1f);
                    auto zenith = make_float3(
                        0.08f, 0.24f, 0.72f) /
                        max(0.65f + 0.2f * air, 0.1f);
                    auto sky = lerp(
                        horizon,
                        zenith,
                        pow(height, 0.35f));
                    auto sunset = pow(
                        max(
                            dot(direction, sun_direction),
                            0.0f),
                        8.0f);
                    sky += make_float3(
                               0.9f, 0.28f, 0.04f) *
                           sunset *
                           max(0.0f, 0.35f - sun_direction.z) *
                           (0.5f + dust);
                    sky *= make_float3(
                        1.0f,
                        max(1.0f - 0.05f * ozone, 0.5f),
                        max(1.0f - 0.12f * ozone, 0.35f));
                    auto sun_cos =
                        cos(sun_size * 0.5f);
                    auto disc =
                        dot(direction, sun_direction) >= sun_cos;
                    auto sun = make_float3(
                                   18.0f, 15.5f, 12.0f) *
                               sun_intensity;
                    value = make_float4(
                        select(sky, sky + sun, disc),
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
                                  1.0f)
                            : Float{1.5f};
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
                        .metallic = metallic,
                        .ior = ior});
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
    GraphSurface(
        std::shared_ptr<const compiler::SurfaceProgram> program,
        std::uint32_t parameter_block) noexcept
        : _program{std::move(program)},
          _parameter_block{parameter_block} {
        if (_program) {
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
        Bool has_diffuse = false;
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
                const auto is_principled =
                    closure.operation ==
                    compiler::ClosureOperation::principled;
                const auto is_glossy =
                    closure.operation ==
                    compiler::ClosureOperation::glossy;
                if (!is_diffuse && !is_principled &&
                    !is_glossy) {
                    return;
                }
                auto diffuse_pdf =
                    max(dot(closure.normal, outgoing), 0.0f) *
                    inverse_pi;
                auto glossy_pdf = microfacet_pdf(
                    closure, incoming, outgoing);
                auto diffuse_allowed =
                    diffuse_enabled &
                    (is_diffuse || is_principled);
                auto glossy_allowed =
                    glossy_enabled &
                    (is_principled || is_glossy);
                Float specular_chance =
                    is_glossy
                        ? 1.0f
                        : is_diffuse
                              ? 0.0f
                              : specular_probability(closure);
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
                auto diffuse_contribution =
                    closure.weight *
                    diffuse_intensity(
                        closure, incoming, outgoing);
                if (is_principled) {
                    diffuse_contribution =
                        closure.weight *
                        closure.color *
                        (1.0f - closure.metallic) *
                        diffuse_intensity(
                            closure, incoming, outgoing);
                }
                if (is_glossy) {
                    diffuse_contribution =
                        make_float3(0.0f);
                }
                auto glossy_contribution =
                    is_diffuse
                        ? make_float3(0.0f)
                        : closure.weight *
                              microfacet_intensity(
                                  closure,
                                  incoming,
                                  outgoing);
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
                auto selection_color =
                    is_diffuse
                        ? closure.weight
                        : closure.weight *
                              max(
                                  closure.color,
                                  make_float3(0.04f));
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
                    (diffuse_allowed &
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
            0.0f, result.pdf, has_diffuse);
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
            values, point, outgoing_expression, query);
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
        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                auto is_diffuse =
                    closure.operation ==
                    compiler::ClosureOperation::diffuse;
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
                        : is_diffuse
                              ? diffuse_enabled
                              : is_principled
                                    ? (diffuse_enabled |
                                       glossy_enabled)
                                    : is_glossy
                                          ? glossy_enabled
                                          : Bool{false};
                auto selection_color =
                    is_diffuse || is_transparent
                        ? closure.weight
                        : closure.weight *
                              max(
                                  closure.color,
                                  make_float3(0.04f));
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
        Bool selected_glossy = false;
        Float3 transparent_weight = make_float3(0.0f);
        Float transparent_sample_weight = 0.0f;

        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                auto is_diffuse =
                    closure.operation ==
                    compiler::ClosureOperation::diffuse;
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
                        : is_diffuse
                              ? diffuse_enabled
                              : is_principled
                                    ? (diffuse_enabled |
                                       glossy_enabled)
                                    : is_glossy
                                          ? glossy_enabled
                                          : Bool{false};
                auto selection_color =
                    is_diffuse || is_transparent
                        ? closure.weight
                        : closure.weight *
                              max(
                                  closure.color,
                                  make_float3(0.04f));
                auto weight = select(
                    0.0f,
                    sample_weight(selection_color),
                    eligible);
                auto next = accumulated + weight;
                auto choose =
                    (!selected) &
                    (weight > 0.0f) &
                    (target < next);
                auto local_diffuse_enabled =
                    diffuse_enabled &
                    (is_diffuse || is_principled);
                auto local_glossy_enabled =
                    glossy_enabled &
                    (is_glossy || is_principled);
                Float specular_chance =
                    is_glossy
                        ? 1.0f
                        : is_diffuse
                              ? 0.0f
                              : specular_probability(closure);
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
                        closure.normal, remapped_random);
                auto glossy_direction = sample_ggx(
                    closure,
                    point.incoming,
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
                selected_glossy =
                    selected_glossy |
                    ((!is_transparent) & choose &
                     sample_glossy);
                selected = selected | choose;
                accumulated = next;
            });

        auto diffuse_evaluation = evaluate_traced(
            values, point, result.wi, query);
        auto geometric_valid =
            dot(point.geometric_normal, result.wi) > 0.0f;
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
            .normal = point.shading_normal};
        if (!_program) {
            return result;
        }
        auto values = trace_values(services, point);
        Float total_weight = 0.0f;
        Float roughness = 0.0f;
        Float3 normal = make_float3(0.0f);
        for_each_closure(
            values,
            [&](const TracedClosure &closure) noexcept {
                const auto is_diffuse =
                    closure.operation ==
                    compiler::ClosureOperation::diffuse;
                const auto is_principled =
                    closure.operation ==
                    compiler::ClosureOperation::principled;
                const auto is_glossy =
                    closure.operation ==
                    compiler::ClosureOperation::glossy;
                if (!is_diffuse && !is_principled &&
                    !is_glossy) {
                    return;
                }
                auto albedo =
                    is_diffuse
                        ? closure.weight
                        : closure.weight * closure.color;
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
