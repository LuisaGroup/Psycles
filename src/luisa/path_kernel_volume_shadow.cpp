#include "path_kernel_volume_shadow.h"

#include "path_kernel_volume_boundary.h"
#include "path_kernel_volume_majorant_provider.h"
#include "path_kernel_volume_random.h"
#include "path_tracer_cycles_svm_volume.h"

#include <psycles/luisa/cycles_noise.h>

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/heterogeneous_volume_collision.h>
#include <psycles/luisa/heterogeneous_volume_shadow.h>
#include <psycles/luisa/surface_ray.h>
#include <psycles/luisa/volume_majorant_overlap.h>
#include <psycles/luisa/volume_shadow_interval.h>

#include <algorithm>
#include <cstdlib>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::uint32_t resolve_max_volume_shadow_steps(
    std::uint32_t configured) noexcept {
    if (const auto *raw = std::getenv("PSYCLES_VOLUME_SHADOW_MAX_STEPS");
        raw != nullptr && raw[0] != '\0') {
        char *end = nullptr;
        const auto parsed = std::strtoul(raw, &end, 10);
        if (end != raw && parsed > 0ul) {
            return static_cast<std::uint32_t>(parsed);
        }
    }
    return std::max(configured, 1u);
}

class VolumeShadowComponentImpl final
    : public VolumeShadowComponent {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<const VolumeMajorantRuntime> _majorants;
    std::size_t _stack_size;
    std::uint32_t _max_volume_shadow_steps;
    std::shared_ptr<
        const TriangleVolumeBoundaryComponent>
        _boundary;
    std::unique_ptr<
        HeterogeneousVolumeShadowComponent>
        _heterogeneous;

    [[nodiscard]] Bool _stack_is_heterogeneous(
        const VolumeStack &stack)
        const noexcept {
        return stack.any(
            [this](const VolumeStackEntry &entry) noexcept {
                return !cycles_svm_volume_is_homogeneous(_scene, entry);
            });
    }

    [[nodiscard]] static bool _may_have_heterogeneous_volume(
        const std::shared_ptr<LuisaSceneData> &scene,
        const std::shared_ptr<const VolumeMajorantRuntime> &majorants) noexcept {
        if (!majorants || majorants->node_count == 0u ||
            majorants->root_count == 0u || majorants->range_count == 0u ||
            majorants->world_range >= majorants->range_count) {
            return false;
        }
        if (scene && scene->cycles_svm) {
            constexpr auto het_mask = static_cast<unsigned>(
                compiler::cycles_svm::SD_HETEROGENEOUS_VOLUME |
                compiler::cycles_svm::SD_NEED_VOLUME_ATTRIBUTES);
            for (const auto &ks : scene->cycles_svm->compilation.kernel_shaders) {
                if ((static_cast<unsigned>(ks.flags) & het_mask) != 0u) {
                    return true;
                }
            }
            return false;
        }
        return true;
    }

  public:
    explicit VolumeShadowComponentImpl(
        const PathKernelConfig &config)
        : _scene{config.scene}, _majorants{config.volume_majorants},
          _stack_size{
              std::max(
                  std::size_t{
                      config.volume_stack_size},
                  std::size_t{1u})},
          _max_volume_shadow_steps{
              resolve_max_volume_shadow_steps(
                  config.max_volume_shadow_steps)},
          _boundary{
              make_triangle_volume_boundary_component()},
          _heterogeneous{
              _may_have_heterogeneous_volume(config.scene, config.volume_majorants)
                  ? make_heterogeneous_volume_shadow_component()
                  : nullptr} {}

    Float3 emit(
        const PathSampleContext &sample,
        const VolumeStack &path_stack,
        Var<luisa::compute::Ray> shadow_ray,
        UInt light_instance,
        UInt light_primitive)
        const noexcept override {
        VolumeStack shadow_stack{
            _stack_size};
        // Cycles copies the complete path stack. Shadow-invisible objects
        // remain present for homogeneity selection, majorant traversal, and
        // RNG consumption; native volume entry evaluation suppresses
        // only their raw closure evaluation.
        shadow_stack.copy_from(path_stack);
        const auto ray_origin =
            shadow_ray->origin();
        const auto ray_direction =
            shadow_ray->direction();
        VolumeShadowIntervalCursor
            interval{
                shadow_ray->t_min()};
        UInt shadow_rng_offset =
            sample.cycles_rng_offset;
        Float3 transmittance =
            make_float3(1.0f);
        Bool active = true;
        UInt last_instance = surface_ray::invalid_primitive;
        UInt last_primitive = surface_ray::invalid_primitive;
        UInt step_count = 0u;

        const bool use_volume_boundary_accel =
            _scene->volume_boundary_accel.has_value() &&
            _scene->primary_to_volume_local_buffer.has_value();
        UInt local_light_inst = surface_ray::invalid_primitive;
        if (use_volume_boundary_accel) {
            const auto n_primary = static_cast<std::uint32_t>(
                _scene->primary_to_volume_local_host.size());
            $if(light_instance < n_primary) {
                local_light_inst =
                    Expr<Buffer<luisa::uint>>{
                        *_scene->primary_to_volume_local_buffer}
                        .read(light_instance);
            };
        }

        // Reduce to the nearest boundary on every iteration. Luisa ray-query
        // candidate order is unspecified; an any-hit stack update would
        // therefore be non-formal and backend-dependent.
        $while(active & (step_count < _max_volume_shadow_steps)) {
            const auto committed = [&]() noexcept {
                if (use_volume_boundary_accel) {
                    return (*_scene->volume_boundary_accel)
                        ->traverse(
                            shadow_ray,
                            {.visibility_mask = shadow_visibility})
                        .on_surface_candidate(
                            [&](luisa::compute::SurfaceCandidate
                                    &candidate) noexcept {
                                const auto hit = candidate.hit();
                                auto accept =
                                    !surface_ray::same_primitive(
                                        hit->inst, hit->prim,
                                        local_light_inst, light_primitive) &
                                    !surface_ray::same_primitive(
                                        hit->inst, hit->prim,
                                        last_instance, last_primitive);
                                if (!_scene->all_volume_boundary_primitives_have_volume) {
                                    const auto p_inst =
                                        (*_scene->volume_boundary_accel)
                                            ->instance_user_id(hit->inst);
                                    accept = accept & _boundary->has_volume(
                                                          _scene, p_inst, hit->prim);
                                }
                                $if(accept) { candidate.commit(); };
                            })
                        .trace();
                }
                return surface_ray::
                    closest_shadow_intersection(
                        _scene->accel,
                        shadow_ray,
                        last_instance,
                        last_primitive,
                        light_instance,
                        light_primitive,
                        shadow_visibility);
            }();
            Float segment_end =
                shadow_ray->t_max();
            $if(!committed->miss()) {
                segment_end =
                    committed
                        ->committed_ray_t;
            };
            const auto segment_length =
                max(
                    segment_end -
                        interval.minimum(),
                    0.0f);
            $if((!shadow_stack.empty()) & (segment_length > 0.0f)) {
                const cycles_svm::PathState shader_state{
                    cycles_svm::path_ray_visibility_shadow, 0u,
                    sample.path_depth, sample.transparent_depth, sample.diffuse_depth,
                    sample.glossy_depth, sample.transmission_depth, sample.portal_depth};
                const PathCyclesSvmVolumeShader volume_shader{
                    _scene, sample.invocation.parameters, ray_origin, ray_direction,
                    interval.minimum(), 0.5f, shadow_stack.entry(0u).object, shader_state,
                    cycles_noise::hash_uint3(sample.rng_hash ^ 0xd9111870u,
                                            shadow_rng_offset, sample.sample_index), true};
                const auto apply_homogeneous_transmittance = [&]() noexcept {
                    const auto coefficients =
                        volume_shader.evaluate(
                            shadow_stack, ray_origin + ray_direction * interval.minimum());
                    // Prevent DXC fast-math from rewriting exp(-0 * FLT_MAX)
                    // into exp2(0 * (-FLT_MAX * log2(e))) = NaN on empty/zero channels.
                    const auto active_extinction = coefficients.sigma_t > 0.0f;
                    const auto safe_sigma_t = select(
                        make_float3(0.0f), coefficients.sigma_t, active_extinction);
                    const auto clamped_optical_depth = min(
                        safe_sigma_t * segment_length,
                        make_float3(105.0f));
                    transmittance *= select(
                        make_float3(1.0f),
                        exp(-clamped_optical_depth),
                        active_extinction);
                };
                if (_heterogeneous) {
                    const auto heterogeneous =
                        _stack_is_heterogeneous(
                            shadow_stack);
                    $if(heterogeneous) {
                        const auto tracking_rng_offset =
                            cycles_sampler::
                                scramble_path_offset(
                                    shadow_rng_offset,
                                    heterogeneous_shadow_scramble_seed);
                        PathVolumeTrackingRandomSource random{
                            sample.invocation.sobol_table,
                            sample.invocation.parameters
                                .sobol_sequence_size,
                            sample.sample_index,
                            sample.rng_hash};
                        auto majorant_provider =
                            make_scene_volume_majorant_entry_provider(
                                volume_shader);
                        Expr<
                            Buffer<
                                VolumeMajorantNodeGpu>>
                            nodes{
                                _majorants->node_buffer};
                        Expr<
                            Buffer<
                                VolumeMajorantRootGpu>>
                            roots{
                                _majorants->root_buffer};
                        Expr<
                            Buffer<
                                VolumeMajorantRootRangeGpu>>
                            ranges{
                                _majorants->range_buffer};
                        VolumeMajorantOverlapTraversal
                            traversal{
                                std::move(nodes),
                                std::move(roots),
                                std::move(ranges),
                                _majorants->node_count,
                                _majorants->root_count,
                                _majorants->range_count,
                                _majorants->world_range,
                                shadow_stack,
                                *majorant_provider,
                                ray_origin,
                                ray_direction,
                                interval.minimum(),
                                segment_end,
                                random.shade_offset(
                                    tracking_rng_offset)};
                        auto collisions =
                            make_stacked_heterogeneous_volume_collision_provider(
                                volume_shader,
                                shadow_stack,
                                ray_origin,
                                ray_direction);
                        const auto estimate =
                            _heterogeneous->emit(
                                traversal,
                                random,
                                *collisions,
                                transmittance,
                                tracking_rng_offset);
                        transmittance =
                            estimate.throughput;
                    }
                    $else {
                        apply_homogeneous_transmittance();
                    };
                } else {
                    apply_homogeneous_transmittance();
                }
            };

            $if(committed->miss()) {
                active = false;
            }
            $else {
                const auto resolved_inst =
                    use_volume_boundary_accel
                        ? (*_scene->volume_boundary_accel)
                              ->instance_user_id(committed->inst)
                        : committed->inst;
                const auto boundary =
                    _boundary->resolve(
                        _scene,
                        resolved_inst,
                        committed->prim,
                        ray_direction);
                shadow_stack.cross_boundary(
                    boundary.primitive
                        .volume_stack_entry(),
                    boundary.back_facing,
                    boundary.primitive
                        .has_volume,
                    true);
                last_instance = committed->inst;
                last_primitive = committed->prim;
                const auto advanced_t = interval.advance(
                    max(shadow_ray->t_min(), committed->committed_ray_t));
                shadow_ray->set_t_min(advanced_t);
                step_count += 1u;
                $if(advanced_t >= shadow_ray->t_max()) {
                    active = false;
                };
                shadow_rng_offset +=
                    cycles_path_state::
                        bounce_dimension_count;
            };
        };
        $if(active) {
            if (_scene && _scene->volume_shadow_guard_hits) {
                Expr<Buffer<luisa::uint>>{*_scene->volume_shadow_guard_hits}
                    .atomic(0u)
                    .fetch_add(1u);
            }
        };
        return transmittance;
    }
};

}// namespace

std::unique_ptr<
    VolumeShadowComponent>
make_volume_shadow_component(
    const PathKernelConfig &config) {
    return std::make_unique<
        VolumeShadowComponentImpl>(
        config);
}

}// namespace psycles::luisa_backend::detail
