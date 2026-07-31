#include "path_kernel_heterogeneous_volume.h"

#include "path_kernel_volume_majorant_provider.h"
#include "path_tracer_volume_capabilities.h"

#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/volume_majorant_overlap.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::uint32_t
    heterogeneous_tracking_scramble_seed =
        0xe35fad82u;

class PathVolumeTrackingRandomSource final
    : public HeterogeneousVolumeTrackingRandomSource {

  private:
    const BufferFloat4 &_sobol_table;
    UInt _sequence_size;
    UInt _sample_index;
    UInt _rng_hash;

    [[nodiscard]] UInt _dimension(
        UInt rng_offset,
        std::uint32_t dimension) const noexcept {
        return cycles_sampler::
            path_state_dimension(
                rng_offset, dimension);
    }

  public:
    PathVolumeTrackingRandomSource(
        const BufferFloat4 &sobol_table,
        UInt sequence_size,
        UInt sample_index,
        UInt rng_hash) noexcept
        : _sobol_table{sobol_table},
          _sequence_size{
              std::move(sequence_size)},
          _sample_index{
              std::move(sample_index)},
          _rng_hash{std::move(rng_hash)} {}

    Float scatter_distance(
        UInt rng_offset)
        const noexcept override {
        return cycles_sampler::sample_1d(
            _sobol_table,
            _sequence_size,
            _sample_index,
            _rng_hash,
            _dimension(
                rng_offset,
                sampling::tabulated_sobol::
                    volume_scatter_distance_dimension));
    }

    Float shade_offset(
        UInt rng_offset)
        const noexcept override {
        return cycles_sampler::sample_2d(
                   _sobol_table,
                   _sequence_size,
                   _sample_index,
                   _rng_hash,
                   _dimension(
                       rng_offset,
                       sampling::tabulated_sobol::
                           volume_shade_offset_dimension))
            .y;
    }

    Float expansion_order(
        UInt rng_offset)
        const noexcept override {
        return cycles_sampler::
            sample_volume_expansion_order(
                _sobol_table,
                _sequence_size,
                _sample_index,
                rng_offset);
    }

    Float transmittance_shade_offset(
        UInt rng_offset)
        const noexcept override {
        return cycles_sampler::sample_1d(
            _sobol_table,
            _sequence_size,
            _sample_index,
            _rng_hash,
            _dimension(
                rng_offset,
                sampling::tabulated_sobol::
                    volume_shade_offset_dimension));
    }
};

class PathHeterogeneousVolumeComponentImpl final
    : public PathHeterogeneousVolumeComponent {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<
        const VolumeStackEntryPointProvider>
        _points;
    std::unique_ptr<
        HeterogeneousVolumeSegmentComponent>
        _segment;
    HeterogeneousVolumeScatterProbability
        _scatter_probability;

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

  public:
    PathHeterogeneousVolumeComponentImpl(
        std::shared_ptr<LuisaSceneData> scene,
        std::shared_ptr<
            const VolumeStackEntryPointProvider>
            points,
        std::size_t closure_allocation_budget)
        : _scene{std::move(scene)},
          _points{std::move(points)},
          _segment{
              make_heterogeneous_volume_segment_component(
                  closure_allocation_budget)} {}

    Bool stack_is_heterogeneous(
        const VolumeStack &stack)
        const noexcept override {
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

    HeterogeneousVolumeSegmentResult
    emit(
        const PathHeterogeneousVolumeInput &input)
        const noexcept override {
        PathVolumeTrackingRandomSource random{
            input.sobol_table,
            input.sobol_sequence_size,
            input.sample_index,
            input.rng_hash};

        // Cycles initializes the octree before scrambling its copied tracking
        // state. Every later setup/advance reads from the current scrambled
        // offset, which the candidate walker advances by one bounce block.
        const auto initial_shade_offset =
            random.shade_offset(
                input.path_rng_offset);
        const auto direct_random =
            random.scatter_distance(
                input.path_rng_offset);
        const auto tracking_rng_offset =
            cycles_sampler::
                scramble_path_offset(
                    input.path_rng_offset,
                    heterogeneous_tracking_scramble_seed);

        auto majorant_provider =
            make_scene_volume_majorant_entry_provider(
                _scene,
                _points,
                input.services,
                input.state,
                true);
        Expr<Buffer<VolumeMajorantNodeGpu>>
            nodes{
                _scene
                    ->volume_majorant_node_buffer};
        Expr<Buffer<VolumeMajorantRootGpu>>
            roots{
                _scene
                    ->volume_majorant_root_buffer};
        Expr<
            Buffer<
                VolumeMajorantRootRangeGpu>>
            ranges{
                _scene
                    ->volume_majorant_range_buffer};
        VolumeMajorantOverlapTraversal traversal{
            std::move(nodes),
            std::move(roots),
            std::move(ranges),
            _scene->volume_majorant_node_count,
            _scene->volume_majorant_root_count,
            _scene->volume_majorant_range_count,
            _scene->volume_majorant_world_range,
            input.stack,
            *majorant_provider,
            input.ray_origin,
            input.ray_direction,
            input.ray_minimum,
            input.ray_maximum,
            initial_shade_offset};
        auto collisions =
            make_stacked_heterogeneous_volume_collision_provider(
                _scene->surfaces,
                _points,
                input.stack,
                input.services,
                input.state,
                input.ray_origin,
                input.ray_direction);
        return _segment->emit(
            {.segments = traversal,
             .random = random,
             .collisions = *collisions,
             .guiding =
                 _scatter_probability
                     .evaluate(
                         input.guiding),
             .direct = input.direct,
             .direct_direction =
                 input.direct_direction,
             .ray_minimum =
                 input.ray_minimum,
             .ray_maximum =
                 input.ray_maximum,
             .segment_origin =
                 input.ray_origin +
                 input.ray_direction *
                     input.ray_minimum,
             .phase_axis =
                 input.ray_direction,
             .throughput =
                 input.throughput,
             .direct_random =
                 direct_random,
             .reservoir_random =
                 input.reservoir_random,
             .phase_random =
                 input.phase_random,
             .tracking_rng_offset =
                 tracking_rng_offset,
             .terminate =
                 input.terminate});
    }
};

}// namespace

std::unique_ptr<PathHeterogeneousVolumeComponent>
make_path_heterogeneous_volume_component(
    std::shared_ptr<LuisaSceneData> scene,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    std::size_t closure_allocation_budget) {
    return std::make_unique<
        PathHeterogeneousVolumeComponentImpl>(
        std::move(scene),
        std::move(points),
        closure_allocation_budget);
}

}// namespace psycles::luisa_backend::detail
