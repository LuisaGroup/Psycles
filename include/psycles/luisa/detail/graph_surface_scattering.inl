// Cycles closure setup, Fresnel, BSDF evaluation, and direction sampling.
// Included by <psycles/luisa/graph_surface.h>; not a standalone header.

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
        return closure_sample_weight(closure) >=
               cycles_closure::closure_weight_cutoff;
    }

    [[nodiscard]] static UInt cycles_runtime_flags(
        const TracedClosure &closure,
        Float glossy_filter_roughness = 0.0f) noexcept {
        UInt flags = 0u;
        switch (closure.operation) {
            case compiler::ClosureOperation::diffuse:
                flags =
                    cycles_closure::runtime_bsdf |
                    cycles_closure::runtime_bsdf_has_eval;
                break;
            case compiler::ClosureOperation::translucent:
                flags =
                    cycles_closure::runtime_bsdf |
                    cycles_closure::runtime_bsdf_has_eval |
                    cycles_closure::
                        runtime_bsdf_has_transmission;
                break;
            case compiler::ClosureOperation::glossy: {
                auto alpha =
                    clamp(closure.roughness, 0.0f, 1.0f);
                alpha *= alpha;
                alpha = max(
                    alpha,
                    glossy_filter_roughness);
                flags =
                    cycles_closure::runtime_bsdf |
                    select(
                        0u,
                        cycles_closure::
                            runtime_bsdf_has_eval,
                        alpha * alpha >
                            cycles_closure::
                                microfacet_singular_alpha_product);
                break;
            }
            case compiler::ClosureOperation::transparent:
                flags =
                    cycles_closure::runtime_bsdf |
                    cycles_closure::runtime_transparent;
                break;
            case compiler::ClosureOperation::principled:
                // The aggregate closure is deliberately incomplete until
                // physical Principled expansion. Its currently implemented
                // reflective lobes are evaluable BSDFs.
                flags =
                    cycles_closure::runtime_bsdf |
                    cycles_closure::runtime_bsdf_has_eval;
                break;
            case compiler::ClosureOperation::emission:
                return cycles_closure::runtime_emission;
            case compiler::ClosureOperation::null_closure:
            case compiler::ClosureOperation::add:
            case compiler::ClosureOperation::mix:
                return 0u;
        }
        flags |= select(
            0u,
            cycles_closure::runtime_bsdf_has_eval,
            glossy_filter_roughness *
                    glossy_filter_roughness >
                cycles_closure::
                    microfacet_singular_alpha_product);
        return select(
            0u,
            flags,
            closure_allocated(closure));
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
