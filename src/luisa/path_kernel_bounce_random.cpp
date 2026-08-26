#include "path_kernel_builder.h"

#include <psycles/luisa/cycles_sampler.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathBounceRandomStageImpl final : public PathBounceRandomStage {

public:
    PathBounceRandomState
    emit(PathSampleContext &sample) const noexcept override {
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &parameters = invocation.parameters;
        const auto &sobol_table = invocation.sobol_table;
        const auto &sample_index = sample.sample_index;
        const auto &rng_hash = sample.rng_hash;
        const auto &rng_offset = sample.cycles_rng_offset;

        const auto light_sample = cycles_sampler::sample_3d(
            sobol_table, parameters.sobol_sequence_size, sample_index, rng_hash,
            cycles_sampler::path_state_dimension(rng_offset,
                                                 tabulated_sobol::light_dimension));
        light_sample.set_name("path_bounce_light_sample");
        Var<LightDistributionGpu> selected_light;
        if (config.use_light_tree) {
            selected_light.cumulative = 0.0f;
            selected_light.selection_pdf = 0.0f;
            selected_light.kind = static_cast<std::uint32_t>(
                sampling::LightDistributionEmitterKind::sentinel);
            selected_light.index = 0u;
            selected_light.emitter_id = ~std::uint32_t{0u};
        } else {
            selected_light = config.light_distribution_sample(light_sample.z);
        }
        selected_light.set_name("path_bounce_selected_light");
        const auto light_terminate_sample = cycles_sampler::sample_1d(
            sobol_table, parameters.sobol_sequence_size, sample_index, rng_hash,
            cycles_sampler::path_state_dimension(
                rng_offset, tabulated_sobol::light_terminate_dimension));
        light_terminate_sample.set_name("path_bounce_light_terminate_sample");

        return {.light_sample = std::move(light_sample),
                .selected_light = std::move(selected_light),
                .light_terminate_sample = std::move(light_terminate_sample)};
    }
};

}// namespace

std::unique_ptr<PathBounceRandomStage> make_path_bounce_random_stage() {
    return std::make_unique<PathBounceRandomStageImpl>();
}

}// namespace psycles::luisa_backend::detail
