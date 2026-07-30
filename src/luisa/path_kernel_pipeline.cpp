#include "path_kernel_builder.h"

#include <utility>
#include <vector>

namespace psycles::luisa_backend::detail {

class PathKernelPipeline::Impl {

  public:
    std::unique_ptr<ClosestEventStage> closest_event{
        make_closest_event_stage()};
    std::unique_ptr<SurfaceGeometryStage> surface_geometry{
        make_surface_geometry_stage()};
    std::unique_ptr<SurfaceShadingStage> surface_shading{
        make_surface_shading_stage()};
    std::vector<std::unique_ptr<DirectLightingComponent>> direct_lighting;
    std::unique_ptr<SurfaceScatterStage> surface_scatter{
        make_surface_scatter_stage()};

    Impl() {
        direct_lighting.emplace_back(make_environment_lighting_component());
        direct_lighting.emplace_back(make_emissive_mesh_lighting_component());
        direct_lighting.emplace_back(make_analytic_lighting_component());
    }
};

PathKernelPipeline::PathKernelPipeline() : _impl{std::make_unique<Impl>()} {}

PathKernelPipeline::~PathKernelPipeline() noexcept = default;
PathKernelPipeline::PathKernelPipeline(PathKernelPipeline &&) noexcept =
    default;
PathKernelPipeline &
PathKernelPipeline::operator=(PathKernelPipeline &&) noexcept = default;

void PathKernelPipeline::emit(PathSampleContext &sample) const noexcept {
    $for(path_step, sample.invocation.parameters.max_path_steps) {
        auto bounce = _impl->closest_event->emit(sample, path_step);
        auto surface = _impl->surface_geometry->emit(bounce);
        auto shading = _impl->surface_shading->emit(surface);
        DirectLightingContext lighting{
            .bounce = bounce, .surface = surface, .shading = shading};
        if (sample.invocation.config.next_event_estimation) {
            for (const auto &component : _impl->direct_lighting) {
                component->emit(lighting);
            }
        }
        _impl->surface_scatter->emit(lighting);
    };
}

RenderKernel build_path_kernel(const PathKernelConfig &config) noexcept {
    PathKernelPipeline pipeline;
    RenderKernel kernel =
        [&config, &pipeline](BufferFloat4 combined,
                             BufferFloat4 normal,
                             BufferFloat4 albedo,
                             BufferFloat4 light_passes,
                             BufferUInt sample_count,
                             BufferFloat4 path_trace,
                             UInt sample_first,
                             UInt samples,
                             BufferFloat4 sobol_table,
                             BufferFloat filter_table,
                             Var<RenderKernelParameters> parameters) noexcept {
            auto invocation = begin_path_kernel(config,
                                                combined,
                                                normal,
                                                albedo,
                                                light_passes,
                                                sample_count,
                                                path_trace,
                                                sample_first,
                                                samples,
                                                sobol_table,
                                                filter_table,
                                                parameters);
            $for(sample_offset, samples) {
                auto sample = begin_path_sample(invocation, sample_offset);
                pipeline.emit(sample);
                accumulate_path_sample(sample);
            };
            invocation.write_film();
        };
    return kernel;
}

} // namespace psycles::luisa_backend::detail
