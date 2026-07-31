#include "path_kernel_builder.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathBounceSetupStageImpl final
    : public PathBounceSetupStage {

  public:
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
        auto &ray_source_instance =
            sample.ray_source_instance;
        auto &ray_source_primitive =
            sample.ray_source_primitive;

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
        Var<LightDistributionGpu> selected_light =
            config.light_distribution_sample(
                light_sample.z);
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
        const auto bsdf_sample =
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
                            surface_bsdf_dimension));

        // Match Cycles' RaySelfPrimitives contract: the previous committed
        // primitive is rejected by identity during traversal. This is
        // independent of origin offset and remains active for transparent
        // and non-transparent bounces.
        Var<luisa::compute::CommittedHit> hit =
            scene->accel
                ->traverse(
                    ray,
                    {.visibility_mask =
                         ray_visibility})
                .on_surface_candidate(
                    [&](luisa::compute::
                            SurfaceCandidate
                                &candidate) noexcept {
                        auto candidate_hit =
                            candidate.hit();
                        $if(!surface_ray::
                                same_primitive(
                                    candidate_hit
                                        ->inst,
                                    candidate_hit
                                        ->prim,
                                    ray_source_instance,
                                    ray_source_primitive)) {
                            candidate.commit();
                        };
                    })
                .on_procedural_candidate(
                    [](luisa::compute::
                           ProceduralCandidate &) noexcept {})
                .trace();
        Float closest_surface_distance =
            ray->t_max();
        $if(!hit->miss()) {
            closest_surface_distance =
                hit->committed_ray_t;
        };

        return {
            sample,
            path_step,
            std::move(terminate_sample),
            std::move(light_sample),
            std::move(selected_light),
            std::move(light_terminate_sample),
            std::move(bsdf_sample),
            std::move(hit),
            std::move(
                closest_surface_distance)};
    }
};

}// namespace

std::unique_ptr<PathBounceSetupStage>
make_path_bounce_setup_stage() {
    return std::make_unique<
        PathBounceSetupStageImpl>();
}

}// namespace psycles::luisa_backend::detail
