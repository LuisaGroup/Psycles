#include "path_kernel_builder.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_emissive_triangle.h"
#include "cycles_shader_identity.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class EmissiveMeshDirectLightProvider final : public DirectLightProvider {

  private:
    DirectLightingContext &_context;
    DirectLightSampleState &_result;
    std::shared_ptr<const EmissiveTriangleComponent> _emissive_triangle;
    std::shared_ptr<const DirectLightTraceRecorder> _trace;

  public:
    EmissiveMeshDirectLightProvider(
        DirectLightingContext &context,
        DirectLightSampleState &result,
        std::shared_ptr<const EmissiveTriangleComponent> emissive_triangle,
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _context{context}, _result{result},
          _emissive_triangle{std::move(emissive_triangle)},
          _trace{std::move(trace)} {}

    void sample() const noexcept override {
        auto &bounce = _context.bounce;
        auto &sample = bounce.sample;
        const auto &config = sample.invocation.config;
        auto &surface = _context.surface;
        auto &selected_light =
            bounce.random().selected_light;
        const auto selected_mesh =
            selected_light.kind ==
            static_cast<std::uint32_t>(
                sampling::
                    LightDistributionEmitterKind::
                        emissive_triangle);
        $if(selected_mesh) {
            const auto light =
                _emissive_triangle
                    ->from_position(
                        config.scene,
                        selected_light
                            .index,
                        surface
                            .hit_position,
                        bounce.random()
                            .light_sample
                            .xy(),
                        selected_light
                            .selection_pdf);
            $if(light.valid) {
                const auto &emitter =
                    light.geometry.emitter;
                _trace->record_sample(
                    bounce,
                    {.type = 5u,
                     .emitter_id =
                         selected_light.emitter_id,
                     .primitive =
                         emitter.cycles_primitive_index,
                     .object =
                         emitter.cycles_object_index,
                     .light_group =
                         emitter.cycles_light_group,
                     .shader =
                         emitter.cycles_shader_id,
                     .pdf = light.pdf,
                     .selection_pdf =
                         selected_light.selection_pdf,
                     .evaluation_factor = 1.0f,
                     .direction =
                         light.light.direction,
                     .position =
                         light.light.position,
                     .geometric_normal =
                         light.geometry.geometric_normal,
                     .distance =
                         light.light.distance});
                const auto is_transmission =
                    dot(
                        light.light.direction,
                        _context.shading.shading_normal) <
                    0.0f;
                const auto same_primitive =
                    (emitter.cycles_primitive_index !=
                     surface_ray::invalid_primitive) &
                    (emitter.cycles_primitive_index ==
                     surface.cycles_primitive_index) &
                    (emitter.cycles_object_index ==
                     surface.cycles_object_index);
                const auto oriented_normal =
                    select(
                        surface.point.geometric_normal,
                        -surface.point.geometric_normal,
                        is_transmission);
                const auto reject_self =
                    same_primitive &
                    (dot(
                         light.light.direction,
                         oriented_normal) >
                     0.0f);
                $if(!reject_self) {
                    const auto constant_emission =
                        emitter.emission_is_constant != 0u;
                    Float3 light_shader = make_float3(0.0f);
                    // Cycles' constant factor is evaluated after geometric
                    // rejection but before the receiving BSDF.
                    $if(constant_emission) {
                        light_shader =
                            _emissive_triangle
                                ->evaluate_constant_emission(
                                    sample,
                                    light);
                    };
                    _result.accept(
                        {.direction = light.light.direction,
                         .target_position = light.light.position,
                         .light_normal = light.geometry.geometric_normal,
                         .light_uv = make_float2(0.0f),
                         .barycentric = light.light.barycentric,
                         .radiometric_weight = make_float3(1.0f),
                         .light_shader = light_shader,
                         .pdf = light.pdf,
                         .normalization_pdf = light.pdf,
                         .distance = light.light.distance,
                         .emitter_kind = static_cast<std::uint32_t>(
                             sampling::LightDistributionEmitterKind::
                                 emissive_triangle),
                         .emitter_index = selected_light.index,
                         .light_object = emitter.cycles_object_index,
                         .light_primitive = emitter.cycles_primitive_index,
                         .shader_flags = emitter.cycles_shader_flags,
                         .apply_mis = true,
                         .constant_light_shader = constant_emission,
                         .distant = false,
                         .valid = true});
                };
            }
            $else {
                const auto emitter =
                    config.scene->emissive_triangle_buffer->read(
                        selected_light.index);
                _trace->record_failed_sample(
                    bounce,
                    {.emitter_id = selected_light.emitter_id,
                     .primitive = cast<int>(
                         emitter.cycles_primitive_index),
                     .object = cast<int>(
                         emitter.cycles_object_index),
                     .visibility_flag =
                         emitter.cycles_shader_flags &
                         cycles_shader_identity::exclude_any,
                     .selection_pdf = selected_light.selection_pdf});
            };
        };
    }

    void evaluate_deferred_emission(
        Bool receiving_nonzero) const noexcept override {
        const auto owns_sample =
            _result.valid &
            (_result.emitter_kind ==
             static_cast<std::uint32_t>(
                 sampling::LightDistributionEmitterKind::emissive_triangle));
        $if(owns_sample & !_result.constant_light_shader & receiving_nonzero) {
            _result.light_shader =
                _emissive_triangle->evaluate_emission_from_sample(
                    _context.bounce.sample,
                    _result.emitter_index,
                    _result.target_position,
                    _result.barycentric,
                    _result.direction,
                    _result.distance);
        };
    }
};

class EmissiveMeshLightingComponent final
    : public DirectLightingComponent {

  private:
    std::shared_ptr<const EmissiveTriangleComponent> _emissive_triangle{
        make_emissive_triangle_component()};
    std::shared_ptr<const DirectLightTraceRecorder> _trace;

  public:
    explicit EmissiveMeshLightingComponent(
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _trace{std::move(trace)} {}

    [[nodiscard]] std::unique_ptr<DirectLightProvider>
    make_light_provider(DirectLightingContext &context,
                        DirectLightSampleState &sample) const override {
        return std::make_unique<EmissiveMeshDirectLightProvider>(
            context, sample, _emissive_triangle, _trace);
    }
};

}// namespace

std::unique_ptr<
    DirectLightingComponent>
make_emissive_mesh_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace) {
    return std::make_unique<EmissiveMeshLightingComponent>(
        std::move(trace));
}

}// namespace psycles::luisa_backend::detail
