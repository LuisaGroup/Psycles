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
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class VolumeShadowComponentImpl final
    : public VolumeShadowComponent {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<const VolumeMajorantRuntime> _majorants;
    std::size_t _stack_size;
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

  public:
    explicit VolumeShadowComponentImpl(
        const PathKernelConfig &config)
        : _scene{config.scene}, _majorants{config.volume_majorants},
          _stack_size{
              std::max(
                  std::size_t{
                      config.volume_stack_size},
                  std::size_t{1u})},
          _boundary{
              make_triangle_volume_boundary_component()},
          _heterogeneous{
              make_heterogeneous_volume_shadow_component()} {}

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

        // Reduce to the nearest boundary on every iteration. Luisa ray-query
        // candidate order is unspecified; an any-hit stack update would
        // therefore be non-formal and backend-dependent.
        $while(active) {
            const auto committed =
                surface_ray::
                    closest_shadow_intersection(
                        _scene->accel,
                        shadow_ray,
                        surface_ray::
                            invalid_primitive,
                        surface_ray::
                            invalid_primitive,
                        light_instance,
                        light_primitive,
                        shadow_visibility);
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
            const cycles_svm::PathState shader_state{
                cycles_svm::path_ray_visibility_shadow, 0u,
                sample.path_depth, sample.transparent_depth, sample.diffuse_depth,
                sample.glossy_depth, sample.transmission_depth, sample.portal_depth};
            const PathCyclesSvmVolumeShader volume_shader{
                _scene, sample.invocation.parameters, ray_origin, ray_direction,
                interval.minimum(), 0.5f, shadow_stack.entry(0u).object, shader_state,
                cycles_noise::hash_uint3(sample.rng_hash ^ 0xd9111870u,
                                        shadow_rng_offset, sample.sample_index), true};
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
                const auto coefficients =
                    volume_shader.evaluate(
                        shadow_stack, ray_origin + ray_direction * interval.minimum());
                transmittance *=
                    exp(
                        -coefficients.sigma_t *
                        segment_length);
            };

            $if(committed->miss()) {
                active = false;
            }
            $else {
                const auto boundary =
                    _boundary->resolve(
                        _scene,
                        committed->inst,
                        committed->prim,
                        ray_direction);
                shadow_stack.cross_boundary(
                    boundary.primitive
                        .volume_stack_entry(),
                    boundary.back_facing,
                    boundary.primitive
                        .has_volume,
                    true);
                // Cycles shades the next volume interval from the raw
                // previous hit t. Only the next surface query is ULP-offset.
                shadow_ray->set_t_min(
                    interval.advance(
                        committed
                            ->committed_ray_t));
                shadow_rng_offset +=
                    cycles_path_state::
                        bounce_dimension_count;
            };
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
