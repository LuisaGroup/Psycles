#include "path_kernel_builder.h"
#include "path_kernel_scene_traversal.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathBounceSetupStageImpl final
    : public PathBounceSetupStage {

  private:
    std::shared_ptr<const SceneTraversalComponent> _traversal;

  public:
    explicit PathBounceSetupStageImpl(
        SceneTraversalStagePlan plan)
        : _traversal{
              make_scene_traversal_component(
                  plan)} {}

    PathBounceContext
    emit(PathSampleContext &sample,
         const UInt &path_step) const noexcept override {
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &scene = config.scene;
        const auto &kernel_parameters =
            invocation.parameters;
        const auto &sobol_table =
            invocation.sobol_table;
        const auto &sample_index =
            sample.sample_index;
        const auto &rng_hash = sample.rng_hash;
        auto &cycles_rng_offset =
            sample.cycles_rng_offset;
        auto &continuation_probability =
            sample.continuation_probability;
        auto &continuation_decided_in_volume =
            sample.continuation_decided_in_volume;
        auto &ray = sample.ray;
        auto &ray_visibility =
            sample.ray_visibility;
        auto &ray_source_object =
            sample.ray_source_object;
        auto &ray_source_primitive =
            sample.ray_source_primitive;
        auto &pending_subsurface_exit =
            sample.pending_subsurface_exit;
        auto &pending_subsurface_hit =
            sample.pending_subsurface_hit;

        continuation_probability = 1.0f;
        continuation_decided_in_volume =
            false;
        const auto terminate_sample =
            cycles_sampler::sample_1d(
                sobol_table,
                kernel_parameters
                    .sobol_sequence_size,
                sample_index,
                rng_hash,
                cycles_sampler::
                    path_state_dimension(
                        cycles_rng_offset,
                        tabulated_sobol::
                            terminate_dimension));
        const auto light_sample =
            cycles_sampler::sample_3d(
                sobol_table,
                kernel_parameters
                    .sobol_sequence_size,
                sample_index,
                rng_hash,
                cycles_sampler::
                    path_state_dimension(
                        cycles_rng_offset,
                        tabulated_sobol::
                            light_dimension));
        Var<LightDistributionGpu> selected_light;
        if (config.use_light_tree) {
            selected_light.cumulative = 0.0f;
            selected_light.selection_pdf = 0.0f;
            selected_light.kind = static_cast<std::uint32_t>(
                sampling::LightDistributionEmitterKind::sentinel);
            selected_light.index = 0u;
            selected_light.emitter_id =
                ~std::uint32_t{0u};
        } else {
            selected_light = config.light_distribution_sample(
                light_sample.z);
        }
        const auto light_terminate_sample =
            cycles_sampler::sample_1d(
                sobol_table,
                kernel_parameters
                    .sobol_sequence_size,
                sample_index,
                rng_hash,
                cycles_sampler::
                    path_state_dimension(
                        cycles_rng_offset,
                        tabulated_sobol::
                            light_terminate_dimension));
        const Bool subsurface_exit = pending_subsurface_exit;
        Var<luisa::compute::CommittedHit> hit;
        Float closest_surface_distance =
            ray->t_max();
        $if(subsurface_exit) {
            // The local BSSRDF traversal has already selected the exact
            // intersection. Preserve it directly, as Cycles does between
            // INTERSECT_SUBSURFACE and SHADE_SURFACE.
            hit = pending_subsurface_hit;
            closest_surface_distance = hit->committed_ray_t;
            pending_subsurface_exit = false;
        }
        $else {
            // Match Cycles' RaySelfPrimitives contract: the previous
            // committed primitive is rejected by identity during traversal.
            // This remains independent of the geometric origin offset.
            hit = _traversal->closest(
                scene, ray, ray_visibility,
                {.object = ray_source_object,
                 .primitive = ray_source_primitive});
            $if(!hit->miss()) {
                closest_surface_distance = hit->committed_ray_t;
            };
        };

        return {
            sample,
            path_step,
            std::move(terminate_sample),
            std::move(light_sample),
            std::move(selected_light),
            std::move(light_terminate_sample),
            std::move(hit),
            std::move(
                closest_surface_distance),
            std::move(subsurface_exit)};
    }
};

}// namespace

std::unique_ptr<PathBounceSetupStage>
make_path_bounce_setup_stage(
    SceneTraversalStagePlan plan) {
    return std::make_unique<
        PathBounceSetupStageImpl>(
            plan);
}

}// namespace psycles::luisa_backend::detail
