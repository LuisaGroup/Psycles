#include "path_kernel_builder.h"
#include "path_kernel_triangle_geometry.h"

#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::uint32_t maximum_hits = 4u;
inline constexpr float burley_truncate = 16.0f;
inline constexpr float burley_truncate_cdf =
    0.9963790093708328f;
inline constexpr std::uint32_t local_intersection_scramble =
    0x68bc21ebu;

[[nodiscard]] Float burley_root(Float xi) noexcept {
    Float radius = select(
        15.0f,
        exp(xi * xi * 2.4f) - 1.0f,
        xi <= 0.9f);
    Bool active = true;
    for (auto iteration = 0u; iteration < 10u; ++iteration) {
        const auto exp_third = exp(-radius / 3.0f);
        const auto exp_full = exp_third * exp_third * exp_third;
        const auto value = 1.0f - 0.25f * exp_full -
                           0.75f * exp_third - xi;
        const auto derivative =
            0.25f * exp_full + 0.25f * exp_third;
        const auto update = active & (abs(value) >= 1.0e-6f) &
                            (derivative != 0.0f);
        radius = select(
            radius,
            max(radius - value / max(derivative, 1.0e-20f), 0.0f),
            update);
        active &= update;
    }
    return radius;
}

[[nodiscard]] Float burley_pdf(Float radius, Float distance) noexcept {
    const auto valid = (radius > 0.0f) & (distance != 0.0f) &
                       (distance < burley_truncate * radius);
    const auto safe_radius = max(radius, 1.0e-20f);
    const auto exp_third = exp(-distance / (3.0f * safe_radius));
    const auto exp_full = exp_third * exp_third * exp_third;
    const auto value = (exp_full + exp_third) /
                       (4.0f * safe_radius * burley_truncate_cdf);
    return select(0.0f, value, valid);
}

[[nodiscard]] Float3 burley_profile(
    Float3 radius, Float distance) noexcept {
    return make_float3(
        burley_pdf(radius.x, distance),
        burley_pdf(radius.y, distance),
        burley_pdf(radius.z, distance));
}

[[nodiscard]] Float channel_count(Float3 radius) noexcept {
    return select(0.0f, 1.0f, radius.x > 0.0f) +
           select(0.0f, 1.0f, radius.y > 0.0f) +
           select(0.0f, 1.0f, radius.z > 0.0f);
}

struct BurleyDiskSample {
    Float radius;
    Float height;
};

[[nodiscard]] BurleyDiskSample sample_burley_disk(
    Float3 radius, Float random) noexcept {
    const auto channels = channel_count(radius);
    Float channel_random = random * channels;
    Float selected_radius = 0.0f;
    Float profile_random = 0.0f;
    Float accumulated = 0.0f;
    Bool selected = false;
    const auto consider = [&](Float value) noexcept {
        const auto enabled = value > 0.0f;
        const auto next = accumulated + select(0.0f, 1.0f, enabled);
        const auto choose = !selected & enabled & (channel_random < next);
        selected_radius = select(selected_radius, value, choose);
        profile_random = select(
            profile_random, channel_random - accumulated, choose);
        selected |= choose;
        accumulated = next;
    };
    consider(radius.x);
    consider(radius.y);
    consider(radius.z);
    const auto scaled = burley_root(
        profile_random * burley_truncate_cdf);
    const auto disk_radius = scaled * selected_radius;
    const auto maximum_radius = burley_truncate * selected_radius;
    return {
        .radius = disk_radius,
        .height = sqrt(max(maximum_radius * maximum_radius -
                               disk_radius * disk_radius,
                           0.0f))};
}

[[nodiscard]] UInt lcg_step(UInt &state) noexcept {
    state = 1103515245u * state + 12345u;
    return state;
}

template<typename Value>
void conditional_swap(Value &left, Value &right, Bool predicate) noexcept {
    const auto left_value = left;
    const auto right_value = right;
    left = select(left_value, right_value, predicate);
    right = select(right_value, left_value, predicate);
}

class SubsurfaceTransportStageImpl final
    : public SubsurfaceTransportStage {

private:
    std::shared_ptr<const TriangleGeometryComponent>
        _triangle_geometry{
            make_triangle_geometry_component()};

public:
    Bool emit(DirectLightingContext &context,
              const SurfaceSample &closure_sample) const noexcept override {
        auto &bounce = context.bounce;
        auto &path = bounce.sample;
        auto &invocation = path.invocation;
        const auto &scene = invocation.config.scene;
        const auto &parameters = invocation.parameters;
        auto &surface = context.surface;
        auto &rng_offset = path.cycles_rng_offset;
        auto &ray = path.ray;
        auto &ray_dD = path.ray_dD;
        auto &throughput = path.throughput;
        auto &pending_exit = path.pending_subsurface_exit;
        auto &pending_hit = path.pending_subsurface_hit;

        Bool success = false;
        const auto is_burley =
            closure_sample.bssrdf_method ==
            static_cast<std::uint32_t>(SurfaceBssrdfMethod::burley);
        $if(is_burley) {
            const auto random_disk = cycles_sampler::sample_2d(
                invocation.sobol_table,
                parameters.sobol_sequence_size,
                path.sample_index,
                path.rng_hash,
                cycles_sampler::path_state_dimension(rng_offset, 0u));
            auto disk_random_y = random_disk.y;

            const auto basis = cycles_sample_mapping::make_orthonormals(
                surface.point.geometric_normal);
            Float3 disk_normal = surface.point.geometric_normal;
            Float3 disk_tangent = basis.tangent;
            Float3 disk_bitangent = basis.bitangent;
            Float pick_pdf_normal = 0.5f;
            Float pick_pdf_tangent = 0.25f;
            Float pick_pdf_bitangent = 0.25f;

            const auto tangent_axis =
                (disk_random_y >= 0.5f) & (disk_random_y < 0.75f);
            const auto bitangent_axis = disk_random_y >= 0.75f;
            disk_normal = select(disk_normal, basis.tangent, tangent_axis);
            disk_tangent = select(disk_tangent,
                                  surface.point.geometric_normal,
                                  tangent_axis);
            pick_pdf_normal = select(
                pick_pdf_normal, 0.25f, tangent_axis);
            pick_pdf_tangent = select(
                pick_pdf_tangent, 0.5f, tangent_axis);
            disk_random_y = select(
                disk_random_y,
                (disk_random_y - 0.5f) * 4.0f,
                tangent_axis);

            disk_normal = select(
                disk_normal, basis.bitangent, bitangent_axis);
            disk_bitangent = select(
                disk_bitangent,
                surface.point.geometric_normal,
                bitangent_axis);
            pick_pdf_normal = select(
                pick_pdf_normal, 0.25f, bitangent_axis);
            pick_pdf_bitangent = select(
                pick_pdf_bitangent, 0.5f, bitangent_axis);
            disk_random_y = select(
                disk_random_y,
                (disk_random_y - 0.75f) * 4.0f,
                bitangent_axis);
            disk_random_y = select(
                disk_random_y,
                disk_random_y * 2.0f,
                (!tangent_axis) & (!bitangent_axis));

            const auto disk = sample_burley_disk(
                closure_sample.bssrdf_radius, random_disk.x);
            const auto phi = 2.0f * cycles_sample_mapping::pi *
                             disk_random_y;
            const auto disk_offset =
                disk_tangent * (disk.radius * cos(phi)) +
                disk_bitangent * (disk.radius * sin(phi));
            Var<luisa::compute::Ray> probe = make_ray(
                surface.point.position + disk_normal * disk.height +
                    disk_offset,
                -disk_normal,
                0.0f,
                2.0f * disk.height);

            luisa::compute::ArrayUInt<maximum_hits> hit_instances;
            luisa::compute::ArrayUInt<maximum_hits> hit_primitives;
            luisa::compute::ArrayFloat2<maximum_hits> hit_barycentrics;
            luisa::compute::ArrayFloat<maximum_hits> hit_distances;
            for (auto index = 0u; index < maximum_hits; ++index) {
                hit_instances[index] = bounce.hit->inst;
                hit_primitives[index] = bounce.hit->prim;
                hit_barycentrics[index] = bounce.hit->bary;
                hit_distances[index] = 0.0f;
            }

            UInt hit_count = 0u;
            UInt lcg_state = cycles_noise::hash_uint3(
                path.rng_hash ^ local_intersection_scramble,
                rng_offset,
                path.sample_index);
            const auto ignored = scene->accel
                                     ->traverse(
                                         probe,
                                         {.visibility_mask = ~0u})
                                     .on_surface_candidate(
                                         [&](luisa::compute::SurfaceCandidate
                                                 &candidate) noexcept {
                                             const auto hit = candidate.hit();
                                             const auto instance =
                                                 scene->instance_buffer->read(
                                                     hit->inst);
                                             const auto object = select(
                                                 hit->inst,
                                                 instance.cycles_object_index,
                                                 instance.cycles_object_index !=
                                                     ~0u);
                                             Bool duplicate = false;
                                             for (auto index = 0u;
                                                  index < maximum_hits;
                                                  ++index) {
                                                 duplicate |=
                                                     (index < min(hit_count,
                                                                  maximum_hits)) &
                                                     (hit_distances[index] ==
                                                      hit->committed_ray_t);
                                             }
                                             $if((object ==
                                                  surface.cycles_object_index) &
                                                 !duplicate) {
                                                 hit_count += 1u;
                                                 UInt record = hit_count - 1u;
                                                 $if(hit_count > maximum_hits) {
                                                     record =
                                                         lcg_step(lcg_state) %
                                                         hit_count;
                                                 };
                                                 $if(record < maximum_hits) {
                                                     hit_instances[record] =
                                                         hit->inst;
                                                     hit_primitives[record] =
                                                         hit->prim;
                                                     hit_barycentrics[record] =
                                                         hit->bary;
                                                     hit_distances[record] =
                                                         hit->committed_ray_t;
                                                 };
                                             };
                                         })
                                     .on_procedural_candidate(
                                         [](luisa::compute::ProceduralCandidate
                                                &) noexcept {})
                                     .trace();
            static_cast<void>(ignored);
            const auto evaluated_hits = min(hit_count, maximum_hits);

            // Cycles sorts the retained local reservoir so CPU, HIP and
            // Vulkan do not inherit their BVH traversal order downstream.
            for (auto pass = 0u; pass + 1u < maximum_hits; ++pass) {
                for (auto index = 0u;
                     index + 1u < maximum_hits - pass;
                     ++index) {
                    const auto swap =
                        (index + 1u < evaluated_hits) &
                        (hit_distances[index] > hit_distances[index + 1u]);
                    conditional_swap(
                        hit_instances[index], hit_instances[index + 1u], swap);
                    conditional_swap(
                        hit_primitives[index], hit_primitives[index + 1u], swap);
                    conditional_swap(hit_barycentrics[index],
                                     hit_barycentrics[index + 1u],
                                     swap);
                    conditional_swap(
                        hit_distances[index], hit_distances[index + 1u], swap);
                }
            }

            luisa::compute::ArrayFloat3<maximum_hits> hit_normals;
            luisa::compute::ArrayFloat3<maximum_hits> hit_weights;
            Float weight_sum = 0.0f;
            const auto channels = channel_count(
                closure_sample.bssrdf_radius);
            const auto disk_profile = burley_profile(
                closure_sample.bssrdf_radius, disk.radius);
            const auto disk_pdf =
                (disk_profile.x + disk_profile.y + disk_profile.z) /
                max(channels, 1.0f);
            for (auto index = 0u; index < maximum_hits; ++index) {
                const auto geometry = _triangle_geometry->emit(
                    scene,
                    hit_instances[index],
                    hit_primitives[index]);
                const auto normal_to_world = transpose(inverse(
                    scene->accel->instance_transform(hit_instances[index])));
                const auto object_normal = normalize(cross(
                    geometry.p1 - geometry.p0,
                    geometry.p2 - geometry.p0));
                auto hit_normal = normalize(
                    (normal_to_world * make_float4(object_normal, 0.0f)).xyz());
                hit_normal = select(
                    hit_normal, -hit_normal, surface.point.back_facing);
                hit_normals[index] = hit_normal;

                const auto hit_position =
                    probe->origin() +
                    probe->direction() * hit_distances[index];
                const auto distance = luisa::compute::length(
                    hit_position - surface.point.position);
                const auto pdf_normal =
                    pick_pdf_normal * abs(dot(disk_normal, hit_normal));
                const auto pdf_tangent =
                    pick_pdf_tangent * abs(dot(disk_tangent, hit_normal));
                const auto pdf_bitangent =
                    pick_pdf_bitangent * abs(dot(disk_bitangent, hit_normal));
                auto mis_weight = pdf_normal /
                                  max(pdf_normal * pdf_normal +
                                          pdf_tangent * pdf_tangent +
                                          pdf_bitangent * pdf_bitangent,
                                      1.0e-20f);
                mis_weight *= select(
                    1.0f,
                    cast<float>(hit_count) /
                        static_cast<float>(maximum_hits),
                    hit_count > maximum_hits);
                auto weight = burley_profile(
                                  closure_sample.bssrdf_radius, distance) *
                              (mis_weight / max(disk_pdf, 1.0e-20f));
                weight = select(
                    make_float3(0.0f),
                    weight,
                    (index < evaluated_hits) & (disk_pdf > 0.0f));
                hit_weights[index] = weight;
                weight_sum +=
                    (abs(weight.x) + abs(weight.y) + abs(weight.z)) /
                    3.0f;
            }

            const auto resample_random = cycles_sampler::sample_1d(
                invocation.sobol_table,
                parameters.sobol_sequence_size,
                path.sample_index,
                path.rng_hash,
                cycles_sampler::path_state_dimension(rng_offset, 1u));
            const auto target = resample_random * weight_sum;
            Float partial = 0.0f;
            UInt selected_index = 0u;
            Float selected_sample_weight = 0.0f;
            Bool selected = false;
            for (auto index = 0u; index < maximum_hits; ++index) {
                const auto weight = hit_weights[index];
                const auto sample_weight =
                    (abs(weight.x) + abs(weight.y) + abs(weight.z)) /
                    3.0f;
                const auto next = partial + sample_weight;
                const auto choose = !selected &
                                    (index < evaluated_hits) &
                                    (target < next);
                selected_index = select(
                    selected_index,
                    static_cast<std::uint32_t>(index),
                    choose);
                selected_sample_weight = select(
                    selected_sample_weight, sample_weight, choose);
                selected |= choose;
                partial = next;
            }

            success = selected & (weight_sum > 0.0f);
            $if(success) {
                const auto selected_weight = hit_weights[selected_index];
                throughput *= selected_weight * weight_sum /
                              max(selected_sample_weight, 1.0e-20f);
                pending_hit.inst = hit_instances[selected_index];
                pending_hit.prim = hit_primitives[selected_index];
                pending_hit.bary = hit_barycentrics[selected_index];
                pending_hit.hit_type = static_cast<std::uint32_t>(
                    luisa::compute::HitType::Surface);
                pending_hit.committed_ray_t =
                    hit_distances[selected_index];
                pending_exit = true;

                const auto exit_normal = hit_normals[selected_index];
                const auto exit_position =
                    probe->origin() + probe->direction() *
                                          hit_distances[selected_index];
                ray = make_ray(exit_position + 2.0f * exit_normal,
                               -exit_normal,
                               0.0f,
                               1.0f);
                ray_dD = 0.0f;
                rng_offset += cycles_path_state::bounce_dimension_count;
            };
        };
        return success;
    }
};

}// namespace

std::unique_ptr<SubsurfaceTransportStage>
make_subsurface_transport_stage() {
    return std::make_unique<SubsurfaceTransportStageImpl>();
}

}// namespace psycles::luisa_backend::detail
