#include "path_kernel_builder.h"

#include "path_kernel_area_light.h"
#include "path_kernel_direct_light_trace.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_light.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class AnalyticDirectLightProvider final : public DirectLightProvider {

  private:
    DirectLightingContext &_context;
    DirectLightSampleState &_result;
    AreaLightSampling _area_sampling;
    std::shared_ptr<const DirectLightTraceRecorder> _trace;

  public:
    AnalyticDirectLightProvider(
        DirectLightingContext &context,
        DirectLightSampleState &result,
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _context{context}, _result{result}, _trace{std::move(trace)} {}

    void sample() const noexcept override {
        auto &bounce = _context.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &scene = config.scene;
        auto &surface = _context.surface;
        auto &selected_light = bounce.random().selected_light;
        auto &light_sample = bounce.random().light_sample;
        auto &hit_position = surface.hit_position;
        auto &cycles_surface_runtime_flags =
            _context.shading.cycles_surface_runtime_flags;
        auto &path_depth = sample.path_depth;
        $if(selected_light.kind ==
            static_cast<std::uint32_t>(
                sampling::LightDistributionEmitterKind::analytic_light)) {
            UInt light_index = selected_light.index;
            Var<LightGpu> light = scene->light_buffer->read(light_index);
            const auto reached_max_bounces =
                cycles_light::select_reached_max_bounces(
                    path_depth, light.max_bounces);
            Float3 wi = make_float3(0.0f);
            Float3 light_radiance = make_float3(0.0f);
            Float3 light_position = light.position;
            Float3 light_normal = make_float3(0.0f, 0.0f, 1.0f);
            Float2 light_uv = make_float2(0.5f);
            Float light_distance = ray_maximum;
            Float light_pdf = 0.0f;
            Float light_eval_factor = 0.0f;
            Bool light_valid = false;

            $if(light.type == static_cast<std::uint32_t>(LightType::area)) {
                const auto finite_sample =
                    _area_sampling.from_position(
                        area_light_sample_input(
                            light,
                            hit_position,
                            light_sample.xy()));
                wi = finite_sample.direction;
                light_position =
                    finite_sample.position;
                light_normal =
                    finite_sample.normal;
                light_uv = finite_sample.uv;
                light_distance =
                    finite_sample.distance;
                light_pdf =
                    finite_sample
                        .conditional_pdf;
                light_eval_factor =
                    finite_sample
                        .evaluation_factor;
                light_radiance =
                    light.color *
                    (light.power *
                     light_eval_factor);
                light_valid =
                    finite_sample.valid;
            }
            $elif(light.type ==
                  static_cast<std::uint32_t>(LightType::distant)) {
                Bool normalize_power =
                    (light.flags & light_flag_normalize) != 0u;
                const auto distant_sample =
                    analytic_light_sampling::sample_distant_light(
                        light.axis_z,
                        light.angle,
                        light_sample.xy(),
                        normalize_power);
                wi = distant_sample.direction;
                light_pdf = distant_sample.conditional_pdf;
                light_eval_factor = distant_sample.evaluation_factor;
                light_radiance =
                    light.color *
                    (light.power * distant_sample.evaluation_factor);
                light_position = -wi;
                light_normal = -wi;
                light_valid = true;
            }
            $else {
                const auto sphere = (light.flags & light_flag_sphere) != 0u;
                const auto inside_sphere =
                    sphere & (light.radius > 0.0f) &
                    (length_squared(hit_position - light.position) <=
                     light.radius * light.radius);
                // The unified pre-NEE closure capability summary also
                // supplies Cycles' transmission bit for the special case of
                // a shading point inside spherical emitter geometry.
                const auto has_transmission =
                    (cycles_surface_runtime_flags &
                     cycles_closure::runtime_bsdf_has_transmission) != 0u;
                const auto normalize_power =
                    (light.flags & light_flag_normalize) != 0u;
                analytic_light_sampling::FiniteLightSample finite_sample{
                    .valid = false,
                    .direction = make_float3(0.0f),
                    .position = light.position,
                    .normal = make_float3(0.0f),
                    .uv = make_float2(0.0f),
                    .distance = 0.0f,
                    .conditional_pdf = 0.0f,
                    .evaluation_factor = 0.0f};
                const auto spot =
                    light.type == static_cast<std::uint32_t>(LightType::spot);
                $if(spot) {
                    finite_sample = analytic_light_sampling::sample_spot_light(
                        hit_position,
                        _context.shading.shading_normal,
                        has_transmission,
                        light.position,
                        light.radius,
                        sphere,
                        light.axis_x,
                        light.axis_y,
                        light.axis_z,
                        light.axis_scale,
                        light.spot_angle,
                        light.spot_smooth,
                        light_sample.xy(),
                        normalize_power);
                }
                $else {
                    finite_sample = analytic_light_sampling::sample_point_light(
                        hit_position,
                        _context.shading.shading_normal,
                        has_transmission,
                        light.position,
                        light.radius,
                        sphere,
                        light.axis_x,
                        light.axis_y,
                        light.axis_z,
                        light.axis_scale,
                        light_sample.xy(),
                        normalize_power);
                };
                wi = finite_sample.direction;
                light_position = finite_sample.position;
                light_normal = finite_sample.normal;
                light_uv = finite_sample.uv;
                light_distance = finite_sample.distance;
                light_pdf = finite_sample.conditional_pdf;
                light_eval_factor = finite_sample.evaluation_factor;
                light_radiance =
                    light.color * (light.power * light_eval_factor);
                light_valid = finite_sample.valid;
            };

            light_pdf *= selected_light.selection_pdf;
            $if(!reached_max_bounces & light_valid & (light_pdf > 0.0f)) {
                _trace->record_sample(
                    bounce,
                    {.type = light.cycles_type,
                     .emitter_id = selected_light.emitter_id,
                     .primitive = light_index,
                     .object = light.cycles_object_index,
                     .light_group = light.cycles_light_group,
                     .shader = light.cycles_shader_id,
                     .pdf = light_pdf,
                     .selection_pdf = selected_light.selection_pdf,
                     .evaluation_factor = light_eval_factor,
                     .direction = wi,
                     .position = light_position,
                     .geometric_normal = light_normal,
                     .distance = light_distance});
                const auto constant_emission =
                    (light.flags &
                     light_flag_constant_emission) != 0u;
                Float3 light_shader = make_float3(0.0f);
                $if(constant_emission) {
                    light_shader =
                        sample.analytic_light_constant_shader(light);
                };
                const auto forward_intersectable =
                    (light.flags & light_flag_forward_intersectable) != 0u;
                _result.accept(
                    {.direction = wi,
                     .target_position = light_position,
                     .light_normal = light_normal,
                     .light_uv = light_uv,
                     .barycentric = make_float2(0.0f),
                     .radiometric_weight = light_radiance,
                     .light_shader = light_shader,
                     .pdf = light_pdf,
                     .normalization_pdf = max(light_pdf, 1.0e-20f),
                     .distance = light_distance,
                     .emitter_kind = static_cast<std::uint32_t>(
                         sampling::LightDistributionEmitterKind::analytic_light),
                     .emitter_index = light_index,
                     .light_object = light.cycles_object_index,
                     .light_primitive = surface_ray::invalid_primitive,
                     .shader_flags = light.cycles_shader_flags,
                     .apply_mis = forward_intersectable,
                     .constant_light_shader = constant_emission,
                     .distant =
                         light.type == static_cast<std::uint32_t>(
                                           LightType::distant),
                     .valid = true});
            }
            $else {
                _trace->record_failed_sample(bounce);
            };
        };
    }

    void evaluate_deferred_emission(
        Bool receiving_nonzero) const noexcept override {
        const auto owns_sample =
            _result.valid &
            (_result.emitter_kind ==
             static_cast<std::uint32_t>(
                 sampling::LightDistributionEmitterKind::analytic_light));
        $if(owns_sample & !_result.constant_light_shader & receiving_nonzero) {
            Var<LightGpu> light =
                _context.bounce.sample.invocation.config.scene
                    ->light_buffer->read(_result.emitter_index);
            _result.light_shader =
                _context.bounce.sample.analytic_light_shader(
                    light,
                    _result.emitter_index,
                    _result.target_position,
                    _result.light_normal,
                    _result.light_uv,
                    -_result.direction,
                    _result.distance);
        };
    }
};

class AnalyticLightingComponent final : public DirectLightingComponent {

  private:
    std::shared_ptr<const DirectLightTraceRecorder> _trace;

  public:
    explicit AnalyticLightingComponent(
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _trace{std::move(trace)} {}

    [[nodiscard]] std::unique_ptr<DirectLightProvider>
    make_light_provider(DirectLightingContext &context,
                        DirectLightSampleState &sample) const override {
        return std::make_unique<AnalyticDirectLightProvider>(
            context, sample, _trace);
    }
};

} // namespace

std::unique_ptr<DirectLightingComponent>
make_analytic_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace) {
    return std::make_unique<AnalyticLightingComponent>(
        std::move(trace));
}

} // namespace psycles::luisa_backend::detail
