#include "path_kernel_builder.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_environment_light.h"

#include <psycles/luisa/cycles_light.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class EnvironmentDirectLightProvider final : public DirectLightProvider {

  private:
    DirectLightingContext &_context;
    DirectLightSampleState &_result;
    std::shared_ptr<const EnvironmentLightComponent> _environment_light;
    std::shared_ptr<const DirectLightTraceRecorder> _trace;

  public:
    EnvironmentDirectLightProvider(
        DirectLightingContext &context,
        DirectLightSampleState &result,
        std::shared_ptr<const EnvironmentLightComponent> environment_light,
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _context{context}, _result{result},
          _environment_light{std::move(environment_light)},
          _trace{std::move(trace)} {}

    void sample() const noexcept override {
        auto &bounce = _context.bounce;
        auto &sample = bounce.sample;
        const auto &config = sample.invocation.config;
        auto &surface = _context.surface;
        auto &selected_light = bounce.random().selected_light;
        auto &light_sample = bounce.random().light_sample;
        $if(selected_light.kind ==
            static_cast<std::uint32_t>(
                sampling::LightDistributionEmitterKind::environment)) {
            const auto light =
                _environment_light
                    ->from_position(
                        config.scene,
                        surface.hit_position,
                        light_sample.xy(),
                        selected_light
                            .selection_pdf,
                        sample.invocation.parameters
                            .analytic_light_count,
                        sample.invocation.parameters
                            .portal_count);
            const auto reached_max_bounces =
                cycles_light::select_reached_max_bounces(
                    sample.path_depth,
                    config.scene->world_max_bounces);
            $if(!reached_max_bounces & light.valid) {
                _trace->record_sample(
                    bounce,
                    {.type = 2u,
                     .emitter_id =
                         selected_light.emitter_id,
                     .primitive =
                         config.scene->light_count,
                     .object =
                         config.scene
                             ->cycles_background_object_index,
                     .light_group =
                         config.scene
                             ->cycles_background_light_group,
                     .shader =
                         config.scene
                             ->cycles_background_shader_id,
                     .pdf = light.pdf,
                     .selection_pdf =
                         selected_light.selection_pdf,
                     .evaluation_factor = 1.0f,
                     .direction = light.direction,
                     .position = -light.direction,
                     .geometric_normal =
                         -light.direction,
                     .distance = ray_maximum});
                Float3 light_shader = make_float3(0.0f);
                const auto constant_emission =
                    config.scene->environment_emission_is_constant;
                if (config.scene
                        ->environment_emission_is_constant) {
                    light_shader =
                        _environment_light
                            ->evaluate_constant_emission(
                                sample);
                }
                _result.accept(
                    {.direction = light.direction,
                     .target_position = -light.direction,
                     .light_normal = -light.direction,
                     .light_uv = make_float2(0.5f),
                     .barycentric = make_float2(0.0f),
                     .radiometric_weight = make_float3(1.0f),
                     .light_shader = light_shader,
                     .pdf = light.pdf,
                     .normalization_pdf = light.pdf,
                     .distance = ray_maximum,
                     .emitter_kind = static_cast<std::uint32_t>(
                         sampling::LightDistributionEmitterKind::environment),
                     .emitter_index = selected_light.index,
                     .light_object = surface_ray::invalid_primitive,
                     .light_primitive = surface_ray::invalid_primitive,
                     .shader_flags =
                         config.scene->cycles_background_shader_flags,
                     .apply_mis = true,
                     .constant_light_shader = Bool{constant_emission},
                     .distant = true,
                     .valid = true});
            }
            $else {
                _trace->record_failed_sample(
                    bounce,
                    {.emitter_id = selected_light.emitter_id,
                     .primitive =
                         ~cast<int>(config.scene->light_count),
                     .object = cast<int>(
                         config.scene->cycles_background_object_index),
                     .visibility_flag = 0u,
                     .selection_pdf = selected_light.selection_pdf});
            };
        };
    }

    void evaluate_deferred_emission(
        Bool receiving_nonzero) const noexcept override {
        auto &sample = _context.bounce.sample;
        if (!sample.invocation.config.scene
                 ->environment_emission_is_constant) {
            const auto owns_sample =
                _result.valid &
                (_result.emitter_kind ==
                 static_cast<std::uint32_t>(
                     sampling::LightDistributionEmitterKind::environment));
            $if(owns_sample & receiving_nonzero) {
                _result.light_shader =
                    _environment_light->evaluate_emission(
                        sample,
                        _result.direction,
                        cycles_path_state::light_emission_shader_state(
                            sample.path_depth,
                            sample.diffuse_depth,
                            sample.glossy_depth,
                            sample.transparent_depth,
                            sample.transmission_depth));
            };
        }
    }
};

class EnvironmentLightingComponent final : public DirectLightingComponent {

  private:
    std::shared_ptr<const EnvironmentLightComponent> _environment_light{
        make_environment_light_component()};
    std::shared_ptr<const DirectLightTraceRecorder> _trace;

  public:
    explicit EnvironmentLightingComponent(
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _trace{std::move(trace)} {}

    [[nodiscard]] std::unique_ptr<DirectLightProvider>
    make_light_provider(DirectLightingContext &context,
                        DirectLightSampleState &sample) const override {
        return std::make_unique<EnvironmentDirectLightProvider>(
            context, sample, _environment_light, _trace);
    }
};

} // namespace

std::unique_ptr<DirectLightingComponent>
make_environment_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace) {
    return std::make_unique<EnvironmentLightingComponent>(
        std::move(trace));
}

} // namespace psycles::luisa_backend::detail
