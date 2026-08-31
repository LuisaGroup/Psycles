#include "subsurface_random_walk_component.h"
#include "path_kernel_subsurface_intersection.h"

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/cycles_volume_phase.h>
#include <psycles/luisa/surface_ray.h>

#include <psycles/sampling/tabulated_sobol.h>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::uint32_t maximum_bounces = 256u;
inline constexpr std::uint32_t similarity_level = 9u;
inline constexpr std::uint32_t random_walk_scramble = 0xdeadbeefu;
inline constexpr float throughput_epsilon = 1.0e-6f;
inline constexpr float minimum_alpha = 0.2f;

[[nodiscard]] Float3 square(Float3 value) noexcept {
    return value * value;
}

[[nodiscard]] Float3 safe_divide_color(
    Float3 numerator, Float3 denominator) noexcept {
    return make_float3(
        select(0.0f, numerator.x / denominator.x, denominator.x != 0.0f),
        select(0.0f, numerator.y / denominator.y, denominator.y != 0.0f),
        select(0.0f, numerator.z / denominator.z, denominator.z != 0.0f));
}

[[nodiscard]] Float component(Float3 value, UInt channel) noexcept {
    auto result = select(value.z, value.y, channel == 1u);
    return select(result, value.x, channel == 0u);
}

struct ScalarRemap {
    Float sigma_t;
    Float alpha;
};

[[nodiscard]] ScalarRemap legacy_remap(
    Float albedo, Float radius, Float anisotropy) noexcept {
    const auto g2 = anisotropy * anisotropy;
    const auto g3 = g2 * anisotropy;
    const auto g4 = g3 * anisotropy;
    const auto g5 = g4 * anisotropy;
    const auto g6 = g5 * anisotropy;
    const auto g7 = g6 * anisotropy;
    const auto a = 1.8260523782f - 1.28451056436f * anisotropy -
                   1.79904629312f * g2 + 9.19393289202f * g3 -
                   22.8215585862f * g4 + 32.0234874259f * g5 -
                   23.6264803333f * g6 + 7.21067002658f * g7;
    const auto b = 4.98511194385f +
                   0.127355959438f *
                       exp(31.1491581433f * anisotropy -
                           201.847017512f * g2 + 841.576016723f * g3 -
                           2018.09288505f * g4 + 2731.71560286f * g5 -
                           1935.41424244f * g6 + 559.009054474f * g7);
    const auto c = 1.09686102424f - 0.394704063468f * anisotropy +
                   1.05258115941f * g2 - 8.83963712726f * g3 +
                   28.8643230661f * g4 - 46.8802913581f * g5 +
                   38.5402837518f * g6 - 12.7181042538f * g7;
    const auto d = 0.496310210422f + 0.360146581622f * anisotropy -
                   2.15139309747f * g2 + 17.8896899217f * g3 -
                   55.2984010333f * g4 + 82.065982243f * g5 -
                   58.5106008578f * g6 + 15.8478295021f * g7;
    const auto e = 4.23190299701f +
                   0.00310603949088f *
                       exp(76.7316253952f * anisotropy -
                           594.356773233f * g2 + 2448.8834203f * g3 -
                           5576.68528998f * g4 + 7116.60171912f * g5 -
                           4763.54467887f * g6 + 1303.5318055f * g7);
    const auto f = 2.40602999408f - 2.51814844609f * anisotropy +
                   9.18494908356f * g2 - 79.2191708682f * g3 +
                   259.082868209f * g4 - 403.613804597f * g5 +
                   302.85712436f * g6 - 87.4370473567f * g7;
    const auto blend = pow(albedo, 0.25f);
    auto alpha = (1.0f - blend) * a * pow(atan(b * albedo), c) +
                 blend * d * pow(atan(e * albedo), f);
    alpha = clamp(alpha, 0.0f, 0.999999f);
    const auto sigma_t_prime = 1.0f / max(radius, 1.0e-16f);
    return {
        .sigma_t = sigma_t_prime / (1.0f - anisotropy),
        .alpha = alpha};
}

struct VectorRemap {
    Float3 sigma_t;
    Float3 alpha;
};

[[nodiscard]] VectorRemap van_de_hulst_remap(
    Float3 color, Float3 radius, Float anisotropy) noexcept {
    const auto s_squared = square(
        4.20863f * color -
        sqrt(9.59217f + 41.6808f * color + 17.7126f * square(color)) +
        4.09712f);
    const auto alpha = clamp(
        safe_divide_color(
            1.0f - s_squared,
            1.0f - anisotropy * s_squared),
        make_float3(0.0f),
        make_float3(0.999999f));
    return {
        .sigma_t = 1.0f / max(radius, make_float3(1.0e-16f)),
        .alpha = alpha};
}

[[nodiscard]] Float diffusion_length(Float alpha) noexcept {
    return rsqrt(
        1.0f -
        pow(alpha,
            2.44294f - 0.0215813f * alpha + 0.578637f / alpha));
}

[[nodiscard]] Float dwivedi_pdf(
    Float value, Float phase_log, Float cosine) noexcept {
    return 1.0f / ((value - cosine) * phase_log);
}

[[nodiscard]] Float sample_dwivedi(
    Float value, Float phase_log, Float random) noexcept {
    return value - (value + 1.0f) * exp(-random * phase_log);
}

[[nodiscard]] Float3 distance_pdf(
    Float3 sigma_t,
    Float distance,
    Bool hit,
    Float3 *transmittance = nullptr) noexcept {
    const auto value = exp(-sigma_t * distance);
    if (transmittance != nullptr) {
        *transmittance = value;
    }
    return select(sigma_t * value, value, hit);
}

struct ChannelSample {
    UInt channel;
    Float3 pdf;
};

[[nodiscard]] ChannelSample sample_channel(
    Float3 alpha, Float3 throughput, Float random) noexcept {
    const auto weights = abs(throughput * alpha);
    const auto sum = weights.x + weights.y + weights.z;
    const auto weighted = weights / max(sum, 1.0e-30f);
    const auto use_weighted = (1.0f - sum) < 1.0f;
    const auto pdf = select(make_float3(1.0f / 3.0f), weighted, use_weighted);
    const auto choose_x = random < pdf.x;
    const auto choose_y = !choose_x & (random < pdf.x + pdf.y);
    auto channel = select(2u, 1u, choose_y);
    channel = select(channel, 0u, choose_x);
    return {.channel = channel, .pdf = pdf};
}

[[nodiscard]] Float3 sample_phase_direction(
    Float3 axis, Float cosine, Float azimuth_random) noexcept {
    return cycles_volume_phase::sample_direction(
        axis, cosine, azimuth_random);
}

}// namespace

SubsurfaceRandomWalkCoefficients
SubsurfaceRandomWalkComponent::coefficients(
    UInt method,
    Float3 albedo,
    Float3 radius,
    Float anisotropy,
    Float3 throughput) const noexcept {
    const auto standard = method == static_cast<std::uint32_t>(
                                      SurfaceBssrdfMethod::random_walk);
    const auto van_de_hulst = standard | (anisotropy < 0.0f);

    const auto modern = van_de_hulst_remap(
        albedo, radius, anisotropy);
    const auto legacy_x = legacy_remap(
        albedo.x, radius.x, anisotropy);
    const auto legacy_y = legacy_remap(
        albedo.y, radius.y, anisotropy);
    const auto legacy_z = legacy_remap(
        albedo.z, radius.z, anisotropy);
    const auto legacy_sigma_t = make_float3(
        legacy_x.sigma_t, legacy_y.sigma_t, legacy_z.sigma_t);
    const auto legacy_alpha = make_float3(
        legacy_x.alpha, legacy_y.alpha, legacy_z.alpha);
    auto sigma_t = select(
        legacy_sigma_t, modern.sigma_t, van_de_hulst);
    auto alpha = select(
        legacy_alpha, modern.alpha, van_de_hulst);
    throughput = safe_divide_color(throughput, albedo);

    const auto low_x = alpha.x < minimum_alpha;
    const auto low_y = alpha.y < minimum_alpha;
    const auto low_z = alpha.z < minimum_alpha;
    throughput = make_float3(
        select(throughput.x,
            throughput.x * alpha.x / minimum_alpha,
            low_x),
        select(throughput.y,
            throughput.y * alpha.y / minimum_alpha,
            low_y),
        select(throughput.z,
            throughput.z * alpha.z / minimum_alpha,
            low_z));
    alpha = make_float3(
        select(alpha.x, minimum_alpha, low_x),
        select(alpha.y, minimum_alpha, low_y),
        select(alpha.z, minimum_alpha, low_z));
    const auto sigma_s = sigma_t * alpha;
    const auto length = diffusion_length(max(alpha.x, max(alpha.y, alpha.z)));
    return {
        .sigma_t = sigma_t,
        .alpha = alpha,
        .sigma_s = sigma_s,
        .throughput = throughput,
        .valid = length != 1.0f};
}

SubsurfaceRandomWalkEntry
SubsurfaceRandomWalkComponent::sample_entry(
    const SurfacePoint &point,
    const SurfaceSample &closure,
    Float2 random) const noexcept {
    const auto skin = closure.bssrdf_method ==
                      static_cast<std::uint32_t>(
                          SurfaceBssrdfMethod::random_walk_skin);
    const auto diffuse_entry = skin & (random.x < 0.5f);
    const auto diffuse_random = make_float2(random.x * 2.0f, random.y);
    const auto refractive_random = make_float2(
        select(random.x,
            2.0f * (random.x - 0.5f),
            skin),
        random.y);
    const auto diffuse_direction =
        cycles_sample_mapping::sample_cosine_hemisphere(
            -closure.bssrdf_normal, diffuse_random)
            .direction;

    const auto cosine = dot(closure.bssrdf_normal, point.incoming);
    const auto half = cycles_sample_mapping::sample_ggx_visible_normal(
        closure.bssrdf_normal,
        point.incoming,
        closure.bssrdf_roughness,
        refractive_random);
    const auto cosine_half_incoming = dot(half, point.incoming);
    const auto inverse_ior = 1.0f / closure.bssrdf_ior;
    const auto argument = 1.0f -
                          inverse_ior * inverse_ior *
                              (1.0f - cosine_half_incoming *
                                          cosine_half_incoming);
    const auto transmitted_cosine = max(sqrt(max(argument, 0.0f)), 1.0e-7f);
    const auto coefficient = inverse_ior * cosine_half_incoming -
                             transmitted_cosine;
    const auto refractive_direction =
        -inverse_ior * point.incoming + coefficient * half;
    const auto direction = select(
        refractive_direction, diffuse_direction, diffuse_entry);
    const auto refractive_valid = cosine > 0.0f;
    const auto valid = (diffuse_entry | refractive_valid) &
                       (dot(point.geometric_normal, direction) < 0.0f);
    return {.direction = direction, .valid = valid};
}

Bool SubsurfaceRandomWalkComponent::transport(
    PathSampleContext &path,
    const SubsurfaceTransportState &state) const noexcept {
    const auto &invocation = path.invocation;
    const auto &scene = invocation.config.scene;
    const auto &parameters = invocation.parameters;
    auto &throughput = path.throughput;
    auto &outer_rng_offset = path.cycles_rng_offset;

    if (!scene->subsurface_accel) {
        return false;
    }
    const auto &subsurface_accel = *scene->subsurface_accel;

    const auto legacy_encoded = state.encoded_anisotropy >= 1.0f;
    const auto anisotropy = select(
        state.encoded_anisotropy,
        state.encoded_anisotropy - 2.0f,
        legacy_encoded);
    const auto method = select(
        static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk),
        static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk_legacy),
        legacy_encoded);
    const auto mapped = coefficients(
        method,
        state.albedo,
        state.radius,
        anisotropy,
        throughput);

    const auto guide_normal = state.normal;
    const auto start_position = path.ray->origin();
    Float3 walk_position = start_position;
    Float3 walk_direction = path.ray->direction();
    Float3 walk_throughput = mapped.throughput;
    const auto original_sigma_t = mapped.sigma_t;
    const auto original_sigma_s = mapped.sigma_s;
    const auto original_anisotropy = anisotropy;
    const auto original_guided_fraction =
        1.0f - max(0.5f, pow(abs(original_anisotropy), 0.125f));
    const auto reduced_sigma_s =
        original_sigma_s * (1.0f - original_anisotropy);
    const auto reduced_sigma_t = original_sigma_t - original_sigma_s +
                                 reduced_sigma_s;
    const auto guide_length = diffusion_length(
        max(mapped.alpha.x, max(mapped.alpha.y, mapped.alpha.z)));
    const auto phase_log = log(
        (guide_length + 1.0f) / (guide_length - 1.0f));

    UInt local_rng_offset = cycles_sampler::scramble_path_offset(
        outer_rng_offset, random_walk_scramble);
    Bool hit = false;
    Bool have_opposite_interface = false;
    Float opposite_distance = 0.0f;
    Float exit_distance = 0.0f;
    Var<luisa::compute::CommittedHit> exit_hit;
    exit_hit.inst = surface_ray::invalid_primitive;
    exit_hit.prim = surface_ray::invalid_primitive;
    exit_hit.bary = make_float2(0.0f);
    exit_hit.hit_type = static_cast<std::uint32_t>(
        luisa::compute::HitType::Miss);
    exit_hit.committed_ray_t = 0.0f;

    $if(mapped.valid) {
        $for(bounce, maximum_bounces) {
            local_rng_offset += cycles_path_state::bounce_dimension_count;
            const auto use_original = bounce <= similarity_level;
            const auto anisotropy = select(
                0.0f, original_anisotropy, use_original);
            const auto guided_fraction = select(
                0.75f, original_guided_fraction, use_original);
            const auto sigma_t = select(
                reduced_sigma_t, original_sigma_t, use_original);
            const auto sigma_s = select(
                reduced_sigma_s, original_sigma_s, use_original);

            const auto channel_random = cycles_sampler::sample_1d(
                invocation.sobol_table,
                parameters.sobol_sequence_size,
                path.sample_index,
                path.rng_hash,
                cycles_sampler::path_state_dimension(
                    local_rng_offset,
                    sampling::tabulated_sobol::
                        subsurface_color_channel_dimension));
            const auto selected_channel = sample_channel(
                mapped.alpha, walk_throughput, channel_random);
            Float sample_sigma_t = component(
                sigma_t, selected_channel.channel);
            const auto distance_random = cycles_sampler::sample_1d(
                invocation.sobol_table,
                parameters.sobol_sequence_size,
                path.sample_index,
                path.rng_hash,
                cycles_sampler::path_state_dimension(
                    local_rng_offset,
                    sampling::tabulated_sobol::
                        subsurface_scatter_distance_dimension));

            Float backward_fraction = 0.0f;
            Float forward_pdf_factor = 0.0f;
            Float forward_stretching = 1.0f;
            Float backward_pdf_factor = 0.0f;
            Float backward_stretching = 1.0f;
            $if(bounce > 0u) {
                const auto strategy_random = cycles_sampler::sample_1d(
                    invocation.sobol_table,
                    parameters.sobol_sequence_size,
                    path.sample_index,
                    path.rng_hash,
                    cycles_sampler::path_state_dimension(
                        local_rng_offset,
                        sampling::tabulated_sobol::
                            subsurface_guide_strategy_dimension));
                const auto guided = strategy_random < guided_fraction;
                Bool guide_backward = false;
                $if(have_opposite_interface) {
                    const auto progress = clamp(
                        dot(walk_position - start_position, -guide_normal),
                        0.0f,
                        opposite_distance);
                    backward_fraction =
                        1.0f /
                        (1.0f +
                         exp((opposite_distance - 2.0f * progress) /
                             guide_length));
                    const auto direction_random = cycles_sampler::sample_1d(
                        invocation.sobol_table,
                        parameters.sobol_sequence_size,
                        path.sample_index,
                        path.rng_hash,
                        cycles_sampler::path_state_dimension(
                            local_rng_offset,
                            sampling::tabulated_sobol::
                                subsurface_guide_direction_dimension));
                    guide_backward = direction_random < backward_fraction;
                };

                const auto scatter_random = cycles_sampler::sample_2d(
                    invocation.sobol_table,
                    parameters.sobol_sequence_size,
                    path.sample_index,
                    path.rng_hash,
                    cycles_sampler::path_state_dimension(
                        local_rng_offset,
                        sampling::tabulated_sobol::subsurface_bsdf_dimension));
                const auto hg_cosine =
                    cycles_volume_phase::sample_henyey_greenstein_cosine(
                        anisotropy, scatter_random.x);
                const auto hg_direction = sample_phase_direction(
                    walk_direction, hg_cosine, scatter_random.y);
                auto guided_cosine = sample_dwivedi(
                    guide_length, phase_log, scatter_random.x);
                guided_cosine = select(
                    guided_cosine, -guided_cosine, guide_backward);
                const auto guided_direction = sample_phase_direction(
                    guide_normal, guided_cosine, scatter_random.y);
                const auto next_direction = select(
                    hg_direction, guided_direction, guided);
                const auto cosine = select(
                    dot(next_direction, guide_normal),
                    guided_cosine,
                    guided);
                const auto hg_pdf = cycles_volume_phase::henyey_greenstein_pdf(
                    dot(walk_direction, next_direction), anisotropy);
                walk_direction = next_direction;

                forward_pdf_factor =
                    cycles_sample_mapping::inverse_two_pi *
                    dwivedi_pdf(guide_length, phase_log, cosine) / hg_pdf;
                backward_pdf_factor =
                    cycles_sample_mapping::inverse_two_pi *
                    dwivedi_pdf(guide_length, phase_log, -cosine) / hg_pdf;
                forward_stretching = 1.0f - cosine / guide_length;
                backward_stretching = 1.0f + cosine / guide_length;
                const auto guided_sigma_t = sample_sigma_t * select(
                    forward_stretching,
                    backward_stretching,
                    guide_backward);
                sample_sigma_t = select(
                    sample_sigma_t,
                    guided_sigma_t,
                    guided);
            };

            Float distance = -log(1.0f - distance_random) / sample_sigma_t;
            const auto minimum_sigma_t = min(
                sigma_t.x, min(sigma_t.y, sigma_t.z));
            const auto trace_maximum = select(
                distance,
                max(distance, 10.0f / minimum_sigma_t),
                bounce == 0u);
            Var<luisa::compute::Ray> query_ray = make_ray(
                walk_position,
                walk_direction,
                0.0f,
                trace_maximum);
            const auto reject_entry_primitive = bounce == 0u;
            auto candidate_hit = subsurface_accel
                                     ->traverse(
                                         query_ray,
                                         {.visibility_mask = ~0u})
                                     .on_surface_candidate(
                                         [&](luisa::compute::SurfaceCandidate
                                                 &candidate) noexcept {
                                             const auto candidate_value =
                                                 candidate.hit();
                                             const auto primary_instance =
                                                 subsurface_primary_instance(
                                                     scene,
                                                     candidate_value->inst);
                                             const auto instance =
                                                 scene->instance_buffer->read(
                                                     primary_instance);
                                             const auto object = select(
                                                 primary_instance,
                                                 instance.cycles_object_index,
                                                 instance.cycles_object_index !=
                                                     ~0u);
                                             const auto self =
                                                 surface_ray::same_primitive(
                                                     primary_instance,
                                                     candidate_value->prim,
                                                     path.pending_subsurface_hit
                                                         .instance,
                                                     path.pending_subsurface_hit
                                                         .primitive);
                                             $if((object ==
                                                  path.ray_source_object) &
                                                 !(reject_entry_primitive &
                                                   self)) {
                                                 candidate.commit();
                                             };
                                         })
                                     .on_procedural_candidate(
                                         [](luisa::compute::ProceduralCandidate
                                                &) noexcept {})
                                     .trace();
            const auto query_hit = !candidate_hit->miss();
            UInt primary_hit_instance = surface_ray::invalid_primitive;
            $if(query_hit) {
                primary_hit_instance = subsurface_primary_instance(
                    scene, candidate_hit->inst);
            };
            const auto query_distance = select(
                trace_maximum,
                candidate_hit->committed_ray_t,
                query_hit);
            $if((bounce == 0u) & query_hit) {
                have_opposite_interface = true;
                opposite_distance = dot(
                    walk_position + query_distance * walk_direction -
                        start_position,
                    -guide_normal);
            };
            const auto step_hit = query_hit &
                                  select(true,
                                      query_distance < distance,
                                      bounce == 0u);
            distance = select(distance, query_distance, step_hit);
            walk_position += distance * walk_direction;

            Float3 transmittance;
            auto pdf = distance_pdf(
                sigma_t, distance, step_hit, &transmittance);
            $if(bounce > 0u) {
                auto guided_pdf = distance_pdf(
                    forward_stretching * sigma_t,
                    distance,
                    step_hit);
                $if(have_opposite_interface) {
                    const auto backward_pdf = distance_pdf(
                        backward_stretching * sigma_t,
                        distance,
                        step_hit);
                    guided_pdf = lerp(
                        guided_pdf * forward_pdf_factor,
                        backward_pdf * backward_pdf_factor,
                        backward_fraction);
                }
                $else {
                    guided_pdf *= forward_pdf_factor;
                };
                pdf = lerp(pdf, guided_pdf, guided_fraction);
            };
            const auto numerator = select(
                sigma_s * transmittance,
                transmittance,
                step_hit);
            walk_throughput *= numerator /
                               dot(selected_channel.pdf, pdf);

            $if(step_hit) {
                hit = true;
                exit_distance = query_distance;
                exit_hit.inst = primary_hit_instance;
                exit_hit.prim = candidate_hit->prim;
                exit_hit.bary = candidate_hit->bary;
                exit_hit.hit_type = candidate_hit->hit_type;
                exit_hit.committed_ray_t =
                    candidate_hit->committed_ray_t;
                $break;
            };
            $if(max(walk_throughput.x,
                    max(walk_throughput.y, walk_throughput.z)) <
                throughput_epsilon) {
                $break;
            };
        };
    };

    $if(hit) {
        throughput = walk_throughput;
        path.pending_subsurface_hit.store_surface(exit_hit);
        path.pending_subsurface_exit = true;
        path.ray = make_ray(
            walk_position + walk_direction * (2.0f * exit_distance),
            -walk_direction,
            0.0f,
            exit_distance);
        path.ray_dD = 0.0f;
        outer_rng_offset += cycles_path_state::bounce_dimension_count;
    };
    return hit;
}

}// namespace psycles::luisa_backend::detail
