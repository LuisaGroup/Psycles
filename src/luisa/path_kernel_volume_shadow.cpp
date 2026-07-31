#include "path_kernel_volume_shadow.h"

#include "path_kernel_volume_boundary.h"
#include "path_kernel_volume_majorant_provider.h"
#include "path_kernel_volume_point.h"
#include "path_kernel_volume_random.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_volume_capabilities.h"

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/heterogeneous_volume_collision.h>
#include <psycles/luisa/heterogeneous_volume_shadow.h>
#include <psycles/luisa/stacked_volume.h>
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
    std::size_t _stack_size;
    std::shared_ptr<
        const VolumeStackEntryPointProvider>
        _points;
    std::shared_ptr<
        const TriangleVolumeBoundaryComponent>
        _boundary;
    std::unique_ptr<
        HeterogeneousVolumeShadowComponent>
        _heterogeneous;

    [[nodiscard]] UInt _surface_flags(
        const VolumeStackEntry &entry)
        const noexcept {
        UInt flags = 0u;
        $if(entry.surface_tag <
            _scene->volume_surface_flag_count) {
            flags =
                _scene->volume_surface_flag_buffer
                    ->read(entry.surface_tag);
        };
        return flags;
    }

    [[nodiscard]] Bool _stack_is_heterogeneous(
        const VolumeStack &stack)
        const noexcept {
        Bool heterogeneous = false;
        for (std::size_t index = 0u;
             index <
             stack.maximum_entries();
             ++index) {
            const auto device_index =
                static_cast<std::uint32_t>(
                    index);
            $if(device_index < stack.count()) {
                const auto entry =
                    stack.entry(device_index);
                heterogeneous |=
                    (_surface_flags(entry) &
                     volume_surface_flag_heterogeneous) !=
                    0u;
            };
        }
        return heterogeneous;
    }

  public:
    explicit VolumeShadowComponentImpl(
        const PathKernelConfig &config)
        : _scene{config.scene},
          _stack_size{
              std::max(
                  std::size_t{
                      config.volume_stack_size},
                  std::size_t{1u})},
          _points{
              make_scene_volume_stack_entry_point_provider(
                  _scene)},
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
        // RNG consumption; SceneVolumeStackEntryPointProvider suppresses
        // only their raw closure evaluation.
        shadow_stack.copy_from(path_stack);
        const StackedVolumeEvaluator evaluator{
            _scene->surfaces,
            *_points};
        BufferShaderServices services{
            _scene->parameter_buffer,
            _scene->cycles_bsdf_table_buffer,
            _scene->texture_heap,
            _scene->heap,
            _scene->attribute_binding_slot,
            _scene->attribute_range_slot,
            _scene->nishita_texture_bindings,
            _scene->shader_color_space};
        const auto shader_state =
            cycles_path_state::
                shadow_shader_state(
                    sample.path_depth,
                    sample.diffuse_depth,
                    sample.glossy_depth,
                    sample.transparent_depth,
                    sample.transmission_depth);

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
            const VolumeShadingState state{
                .position =
                    ray_origin +
                    ray_direction *
                        interval.minimum(),
                .incoming =
                    -ray_direction,
                .ray_visibility =
                    shader_state
                        .ray_visibility,
                .ray_events =
                    shader_state.ray_events,
                .ray_depth =
                    shader_state.ray_depth,
                .diffuse_depth =
                    shader_state.diffuse_depth,
                .glossy_depth =
                    shader_state.glossy_depth,
                .transparent_depth =
                    shader_state
                        .transparent_depth,
                .transmission_depth =
                    shader_state
                        .transmission_depth,
                // shader_setup_from_volume() initializes ray_length for each
                // transparent-shadow interval, independent of distance from
                // the original light sample.
                .ray_length =
                    interval
                        .shader_ray_length(),
                .time = 0.0f};
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
                        _scene,
                        _points,
                        services,
                        state,
                        false);
                Expr<
                    Buffer<
                        VolumeMajorantNodeGpu>>
                    nodes{
                        _scene
                            ->volume_majorant_node_buffer};
                Expr<
                    Buffer<
                        VolumeMajorantRootGpu>>
                    roots{
                        _scene
                            ->volume_majorant_root_buffer};
                Expr<
                    Buffer<
                        VolumeMajorantRootRangeGpu>>
                    ranges{
                        _scene
                            ->volume_majorant_range_buffer};
                VolumeMajorantOverlapTraversal
                    traversal{
                        std::move(nodes),
                        std::move(roots),
                        std::move(ranges),
                        _scene
                            ->volume_majorant_node_count,
                        _scene
                            ->volume_majorant_root_count,
                        _scene
                            ->volume_majorant_range_count,
                        _scene
                            ->volume_majorant_world_range,
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
                        _scene->surfaces,
                        _points,
                        shadow_stack,
                        services,
                        state,
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
                    evaluator.evaluate(
                        shadow_stack,
                        services,
                        state,
                        false);
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
