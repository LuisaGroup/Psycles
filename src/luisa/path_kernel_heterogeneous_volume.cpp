#include "path_kernel_heterogeneous_volume.h"

#include "path_kernel_volume_majorant_provider.h"
#include "path_kernel_volume_random.h"

#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/volume_majorant_overlap.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathHeterogeneousVolumeComponentImpl final
    : public PathHeterogeneousVolumeComponent {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<const VolumeMajorantRuntime> _majorants;
    std::unique_ptr<
        HeterogeneousVolumeSegmentComponent>
        _segment;
    HeterogeneousVolumeScatterProbability
        _scatter_probability;

  public:
    PathHeterogeneousVolumeComponentImpl(
        std::shared_ptr<LuisaSceneData> scene,
        std::shared_ptr<const VolumeMajorantRuntime> majorants)
        : _scene{std::move(scene)}, _majorants{std::move(majorants)},
          _segment{
              make_heterogeneous_volume_segment_component(
                  8u)} {}

    Bool stack_is_heterogeneous(
        const VolumeStack &stack)
        const noexcept override {
        return stack.any(
            [this](const VolumeStackEntry &entry) noexcept {
                return !cycles_svm_volume_is_homogeneous(_scene, entry);
            });
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
                input.shader);
        Expr<Buffer<VolumeMajorantNodeGpu>>
            nodes{
                _majorants->node_buffer};
        Expr<Buffer<VolumeMajorantRootGpu>>
            roots{
                _majorants->root_buffer};
        Expr<
            Buffer<
                VolumeMajorantRootRangeGpu>>
            ranges{
                _majorants->range_buffer};
        VolumeMajorantOverlapTraversal traversal{
            std::move(nodes),
            std::move(roots),
            std::move(ranges),
            _majorants->node_count,
            _majorants->root_count,
            _majorants->range_count,
            _majorants->world_range,
            input.stack,
            *majorant_provider,
            input.ray_origin,
            input.ray_direction,
            input.ray_minimum,
            input.ray_maximum,
            initial_shade_offset};
        auto collisions =
            make_stacked_heterogeneous_volume_collision_provider(
                input.shader,
                input.stack,
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
             .direct_light =
                 input.direct_light,
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
    std::shared_ptr<const VolumeMajorantRuntime> majorants) {
    return std::make_unique<
        PathHeterogeneousVolumeComponentImpl>(
        std::move(scene), std::move(majorants));
}

}// namespace psycles::luisa_backend::detail
