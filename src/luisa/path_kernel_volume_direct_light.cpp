#include "path_kernel_volume_direct_light.h"

#include "path_kernel_area_light.h"
#include "path_kernel_volume_environment_light.h"
#include "path_kernel_volume_mesh_light.h"
#include "path_kernel_volume_shadow.h"

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/spherical_geometry.h>
#include <psycles/luisa/surface_ray.h>
#include <psycles/luisa/volume_analytic_light_sampling.h>
#include <psycles/luisa/volume_light_interval.h>
#include <psycles/sampling/light_distribution.h>

#include <utility>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr auto analytic_emitter_kind =
    static_cast<std::uint32_t>(
        sampling::
            LightDistributionEmitterKind::
                analytic_light);

[[nodiscard]] VolumePointLightSampleInput
volume_point_light_input(
    Var<LightGpu> light,
    Float3 reference,
    Float2 random) noexcept {
    return {
        .reference = std::move(reference),
        .center = light.position,
        .radius = light.radius,
        .sphere =
            (light.flags &
             light_flag_sphere) != 0u,
        .axis_x = light.axis_x,
        .axis_y = light.axis_y,
        .axis_z = light.axis_z,
        .axis_scale = light.axis_scale,
        .random = std::move(random),
        .normalize_power =
            (light.flags &
             light_flag_normalize) != 0u};
}

[[nodiscard]] VolumeSpotLightSampleInput
volume_spot_light_input(
    Var<LightGpu> light,
    Float3 reference,
    Float2 random) noexcept {
    return {
        .point =
            volume_point_light_input(
                light,
                std::move(reference),
                std::move(random)),
        .spot_angle = light.spot_angle,
        .spot_smooth =
            light.spot_smooth};
}

class AnalyticVolumeDirectionProvider final
    : public VolumeDirectDirectionProvider {

  private:
    const PathKernelConfig &_config;
    ClosestPathEvent &_event;
    VolumeDirectLightProposal _proposal;
    Float3 _segment_position;
    Float3 _segment_direction;
    VolumeDirectLightSample &_result;
    VolumeAnalyticLightSampling
        _light_sampling;
    AreaLightSampling
        _area_sampling;

    void _sample_distant(
        Var<LightGpu> light,
        UInt light_index,
        Float3 random,
        Bool active) const noexcept {
        const auto half_angle =
            0.5f * max(light.angle, 0.0f);
        const auto finite_sun =
            half_angle > 0.0f;
        const auto cap_height =
            spherical_geometry::
                unit_cap_height(
                    half_angle);
        const auto sun_axis =
            light.axis_z;
        const auto basis_reference =
            select(
                make_float3(
                    0.0f, 0.0f, 1.0f),
                make_float3(
                    0.0f, 1.0f, 0.0f),
                abs(sun_axis.z) >
                    0.999f);
        const auto sun_tangent =
            _config.light_transport
                .safe_normalize(
                    cross(
                        basis_reference,
                        sun_axis),
                    make_float3(
                        1.0f, 0.0f, 0.0f));
        const auto sun_bitangent =
            cross(
                sun_axis,
                sun_tangent);
        const auto cosine_theta =
            1.0f -
            random.x * cap_height;
        const auto sine_theta =
            sqrt(max(
                1.0f -
                    cosine_theta *
                        cosine_theta,
                0.0f));
        const auto phi =
            2.0f * pi * random.y;
        const auto cone_direction =
            sun_tangent *
                (cos(phi) * sine_theta) +
            sun_bitangent *
                (sin(phi) * sine_theta) +
            sun_axis * cosine_theta;
        const auto direction =
            select(
                sun_axis,
                cone_direction,
                finite_sun);
        const auto solid_angle =
            spherical_geometry::two_pi *
            cap_height;
        const auto conditional_pdf =
            select(
                1.0f,
                1.0f /
                    max(
                        solid_angle,
                        1.0e-20f),
                finite_sun);
        const auto normalize_power =
            (light.flags &
             light_flag_normalize) != 0u;
        const auto disk_area =
            pi *
            sin(half_angle) *
            sin(half_angle);
        const auto evaluation_factor =
            select(
                1.0f,
                1.0f /
                    max(
                        disk_area,
                        1.0e-20f),
                normalize_power &
                    finite_sun);
        auto radiance =
            light.color *
            (light.power *
             evaluation_factor);
        radiance *=
            _event.bounce.sample
                .analytic_light_shader(
                    light,
                    light_index,
                    direction,
                    -direction,
                    make_float2(0.5f),
                    -direction,
                    ray_maximum);
        const auto pdf =
            conditional_pdf *
            _event.bounce.selected_light
                .selection_pdf;
        const auto valid =
            active & (pdf > 0.0f);
        _result.direction =
            select(
                _result.direction,
                direction,
                valid);
        _result.radiance =
            select(
                _result.radiance,
                radiance,
                valid);
        _result.pdf =
            select(
                _result.pdf,
                pdf,
                valid);
        _result.maximum_distance =
            select(
                _result.maximum_distance,
                ray_maximum,
                valid);
        _result.use_mis =
            select(
                _result.use_mis,
                (light.flags &
                 light_flag_use_mis) != 0u,
                valid);
        _result.valid |= valid;
    }

    void _commit_finite_sample(
        Var<LightGpu> light,
        UInt light_index,
        const analytic_light_sampling::
            FiniteLightSample &finite,
        Bool active) const noexcept {
        auto radiance =
            light.color *
            (light.power *
             finite.evaluation_factor);
        radiance *=
            _event.bounce.sample
                .analytic_light_shader(
                    light,
                    light_index,
                    finite.position,
                    finite.normal,
                    finite.uv,
                    -finite.direction,
                    finite.distance);
        const auto pdf =
            finite.conditional_pdf *
            _event.bounce.selected_light
                .selection_pdf;
        const auto valid =
            active &
            finite.valid &
            (pdf > 0.0f);
        _result.direction =
            select(
                _result.direction,
                finite.direction,
                valid);
        _result.radiance =
            select(
                _result.radiance,
                radiance,
                valid);
        _result.pdf =
            select(
                _result.pdf,
                pdf,
                valid);
        _result.maximum_distance =
            select(
                _result.maximum_distance,
                finite.distance,
                valid);
        _result.use_mis =
            select(
                _result.use_mis,
                (light.flags &
                 light_flag_use_mis) != 0u,
                valid);
        _result.valid |= valid;
    }

    void _sample_point(
        Var<LightGpu> light,
        UInt light_index,
        Float3 position,
        Float3 random,
        Bool active) const noexcept {
        const auto finite =
            _light_sampling.point(
                volume_point_light_input(
                    light,
                    std::move(position),
                    random.xy()));
        _commit_finite_sample(
            light,
            light_index,
            finite,
            active);
    }

    void _sample_spot(
        Var<LightGpu> light,
        UInt light_index,
        Float3 position,
        Float3 random,
        Bool active) const noexcept {
        const auto finite =
            _light_sampling
                .spot_from_position(
                    volume_spot_light_input(
                        light,
                        std::move(position),
                        random.xy()));
        _commit_finite_sample(
            light,
            light_index,
            finite,
            active);
    }

    void _sample_area(
        Var<LightGpu> light,
        UInt light_index,
        Float3 position,
        Float3 random,
        Bool active) const noexcept {
        const auto finite =
            _area_sampling.from_position(
                area_light_sample_input(
                    light,
                    std::move(position),
                    random.xy()));
        _commit_finite_sample(
            light,
            light_index,
            finite,
            active);
    }

  public:
    AnalyticVolumeDirectionProvider(
        const PathKernelConfig &config,
        ClosestPathEvent &event,
        const VolumeDirectLightProposal
            &proposal,
        Float3 segment_position,
        Float3 segment_direction,
        VolumeDirectLightSample &result) noexcept
        : _config{config},
          _event{event},
          _proposal{proposal},
          _segment_position{
              std::move(segment_position)},
          _segment_direction{
              std::move(segment_direction)},
          _result{result} {}

    VolumeDirectDirectionSample emit(
        Float distance) const noexcept override {
        const auto position =
            _segment_position +
            _segment_direction *
                distance;
        const auto random =
            _event.bounce.light_sample;
        const auto active =
            _proposal.valid &
            (_proposal.emitter_kind ==
             analytic_emitter_kind);
        const auto light_index =
            select(
                0u,
                _proposal.emitter_index,
                active);
        Var<LightGpu> light =
            _config.scene->light_buffer->read(
                light_index);
        const auto distant =
            light.type ==
            static_cast<std::uint32_t>(
                contract::LightType::
                    distant);
        const auto point =
            light.type ==
            static_cast<std::uint32_t>(
                contract::LightType::
                    point);
        const auto spot =
            light.type ==
            static_cast<std::uint32_t>(
                contract::LightType::
                    spot);
        const auto area =
            light.type ==
            static_cast<std::uint32_t>(
                contract::LightType::
                    area);
        $if(active & distant) {
            _sample_distant(
                light,
                light_index,
                random,
                true);
        }
        $elif(active & point) {
            _sample_point(
                light,
                light_index,
                position,
                random,
                true);
        }
        $elif(active & spot) {
            _sample_spot(
                light,
                light_index,
                position,
                random,
                true);
        }
        $elif(active & area) {
            _sample_area(
                light,
                light_index,
                position,
                random,
                true);
        };
        return {
            .direction =
                _result.direction,
            .valid = _result.valid};
    }
};

class CombinedVolumeDirectionProvider final
    : public VolumeDirectDirectionProvider {

  private:
    std::vector<std::unique_ptr<
        VolumeDirectDirectionProvider>>
        _providers;
    VolumeDirectLightSample &_result;

  public:
    CombinedVolumeDirectionProvider(
        std::vector<std::unique_ptr<
            VolumeDirectDirectionProvider>>
            providers,
        VolumeDirectLightSample
            &result) noexcept
        : _providers{
              std::move(providers)},
          _result{result} {}

    VolumeDirectDirectionSample emit(
        Float distance)
        const noexcept override {
        for (const auto &provider :
             _providers) {
            static_cast<void>(
                provider->emit(
                    distance));
        }
        return {
            .direction =
                _result.direction,
            .valid =
                _result.valid};
    }
};

class PathVolumeDirectLightingComponent final
    : public VolumeDirectLightingComponent {

  private:
    PathKernelConfig _config;
    std::unique_ptr<
        HomogeneousVolumeShadowComponent>
        _volume_shadow;
    std::unique_ptr<
        VolumeMeshLightComponent>
        _mesh_light;
    std::unique_ptr<
        VolumeEnvironmentLightComponent>
        _environment_light;
    VolumeAnalyticLightSampling
        _light_sampling;
    AreaLightSampling
        _area_sampling;
    VolumeLightInterval
        _light_interval;

    void _propose_distant(
        UInt light_index,
        Float segment_length,
        VolumeDirectLightProposal
            &result) const noexcept {
        result.emitter_kind =
            analytic_emitter_kind;
        result.emitter_index =
            light_index;
        result.requested_method =
            volume_sample_distance;
        result.light_position =
            make_float3(0.0f);
        result.interval = {
            .minimum = 0.0f,
            .maximum = segment_length};
        result.valid = true;
    }

    void _propose_point(
        Var<LightGpu> light,
        UInt light_index,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Float2 random,
        Float segment_length,
        VolumeDirectLightProposal
            &result) const noexcept {
        const auto sample =
            _light_sampling.point(
                volume_point_light_input(
                    light,
                    std::move(
                        segment_position),
                    std::move(random)));
        $if(sample.valid) {
            result.emitter_kind =
                analytic_emitter_kind;
            result.emitter_index =
                light_index;
            result.requested_method =
                path_stack.sample_method();
            result.light_position =
                sample.position;
            result.interval = {
                .minimum = 0.0f,
                .maximum =
                    segment_length};
            result.valid = true;
        };
    }

    void _propose_spot(
        Var<LightGpu> light,
        UInt light_index,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Float3 segment_direction,
        Float2 random,
        Float segment_length,
        VolumeDirectLightProposal
            &result) const noexcept {
        const auto sample =
            _light_sampling
                .spot_from_segment(
                    volume_spot_light_input(
                        light,
                        segment_position,
                        std::move(random)));
        $if(sample.valid) {
            const auto interval =
                _light_interval.spot(
                    {.ray_origin =
                         segment_position,
                     .ray_direction =
                         segment_direction,
                     .interval =
                         {.minimum = 0.0f,
                          .maximum =
                              segment_length},
                     .center =
                         light.position,
                     .axis_x =
                         light.axis_x,
                     .axis_y =
                         light.axis_y,
                     .axis_z =
                         light.axis_z,
                     .axis_scale =
                         light.axis_scale,
                     .radius =
                         light.radius,
                     .spot_angle =
                         light.spot_angle});
            $if(interval.valid) {
                result.emitter_kind =
                    analytic_emitter_kind;
                result.emitter_index =
                    light_index;
                result.requested_method =
                    path_stack
                        .sample_method();
                result.light_position =
                    sample.position;
                result.interval =
                    interval.interval;
                result.valid = true;
            };
        };
    }

    void _propose_area(
        Var<LightGpu> light,
        UInt light_index,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Float3 segment_direction,
        Float2 random,
        Float segment_length,
        VolumeDirectLightProposal
            &result) const noexcept {
        const auto sample =
            _area_sampling.from_segment(
                area_light_sample_input(
                    light,
                    segment_position,
                    std::move(random)));
        $if(sample.valid) {
            const auto interval =
                _light_interval.area(
                    {.ray_origin =
                         segment_position,
                     .ray_direction =
                         segment_direction,
                     .interval =
                         {.minimum = 0.0f,
                          .maximum =
                              segment_length},
                     .center =
                         light.position,
                     .axis_u =
                         light.axis_x,
                     .axis_v =
                         light.axis_y,
                     .axis_z =
                         light.axis_z,
                     .length_u =
                         light.size_u,
                     .length_v =
                         light.size_v,
                     .spread =
                         light.spread,
                     .ellipse =
                         (light.flags &
                          light_flag_ellipse) !=
                         0u});
            $if(interval.valid) {
                result.emitter_kind =
                    analytic_emitter_kind;
                result.emitter_index =
                    light_index;
                result.requested_method =
                    path_stack
                        .sample_method();
                result.light_position =
                    sample.position;
                result.interval =
                    interval.interval;
                result.valid = true;
            };
        };
    }

  public:
    explicit PathVolumeDirectLightingComponent(
        const PathKernelConfig &config)
        : _config{config},
          _volume_shadow{
              make_homogeneous_volume_shadow_component(
                  config)},
          _mesh_light{
              make_volume_mesh_light_component(
                  config)},
          _environment_light{
              make_volume_environment_light_component(
                  config)} {}

    VolumeDirectLightProposal propose(
        const ClosestPathEvent &event,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Float3 segment_direction,
        Float segment_length)
        const noexcept override {
        const auto &selected =
            event.bounce.selected_light;
        const auto selected_analytic =
            selected.kind ==
            analytic_emitter_kind;
        VolumeDirectLightProposal result{
            .emitter_kind =
                ~0u,
            .emitter_index = 0u,
            .requested_method =
                volume_sample_none,
            .light_position =
                make_float3(0.0f),
            .interval =
                {.minimum = 0.0f,
                 .maximum =
                     segment_length},
            .valid = false};
        // Non-analytic distribution entries do not index light_buffer.
        $if(selected_analytic) {
            const auto light_index =
                selected.index;
            Var<LightGpu> light =
                _config.scene->light_buffer
                    ->read(light_index);
            const auto distant =
                light.type ==
                static_cast<std::uint32_t>(
                    contract::LightType::
                        distant);
            const auto point =
                light.type ==
                static_cast<std::uint32_t>(
                    contract::LightType::
                        point);
            const auto spot =
                light.type ==
                static_cast<std::uint32_t>(
                    contract::LightType::
                        spot);
            const auto area =
                light.type ==
                static_cast<std::uint32_t>(
                    contract::LightType::
                        area);
            const auto visible_to_volume =
                (light.visibility_mask &
                 contract::visibility_bit(
                     contract::
                         RayVisibility::
                             volume_scatter)) !=
                0u;
            const auto eligible =
                visible_to_volume &
                (segment_length > 0.0f);
            $if(eligible & distant) {
                _propose_distant(
                    light_index,
                    segment_length,
                    result);
            }
            $elif(eligible & point) {
                _propose_point(
                    light,
                    light_index,
                    path_stack,
                    segment_position,
                    event.bounce
                        .light_sample.xy(),
                    segment_length,
                    result);
            }
            $elif(eligible & spot) {
                _propose_spot(
                    light,
                    light_index,
                    path_stack,
                    segment_position,
                    segment_direction,
                    event.bounce
                        .light_sample.xy(),
                    segment_length,
                    result);
            }
            $elif(eligible & area) {
                _propose_area(
                    light,
                    light_index,
                    path_stack,
                    segment_position,
                    segment_direction,
                    event.bounce
                        .light_sample.xy(),
                    segment_length,
                    result);
            };
        };
        _mesh_light->propose(
            event,
            path_stack,
            segment_position,
            segment_direction,
            segment_length,
            result);
        _environment_light->propose(
            event,
            path_stack,
            segment_length,
            result);
        return result;
    }

    std::unique_ptr<
        VolumeDirectDirectionProvider>
    make_direction_provider(
        ClosestPathEvent &event,
        const VolumeDirectLightProposal
            &proposal,
        Float3 segment_position,
        Float3 segment_direction,
        VolumeDirectLightSample &result)
        const override {
        std::vector<std::unique_ptr<
            VolumeDirectDirectionProvider>>
            providers;
        providers.emplace_back(
            std::make_unique<
                AnalyticVolumeDirectionProvider>(
                    _config,
                    event,
                    proposal,
                    segment_position,
                    segment_direction,
                    result));
        providers.emplace_back(
            _mesh_light
                ->make_direction_provider(
                    event,
                    proposal,
                    segment_position,
                    segment_direction,
                    result));
        providers.emplace_back(
            _environment_light
                ->make_direction_provider(
                    event,
                    proposal,
                    std::move(
                        segment_position),
                    std::move(
                        segment_direction),
                    result));
        return std::make_unique<
            CombinedVolumeDirectionProvider>(
                std::move(providers),
                result);
    }

    void accumulate(
        ClosestPathEvent &event,
        const VolumeDirectLightSample &light,
        const VolumeDirectScatter &volume,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Bool inside_volume)
        const noexcept override {
        auto &bounce = event.bounce;
        auto &sample = bounce.sample;
        auto &invocation =
            sample.invocation;
        const auto eligible =
            inside_volume &
            light.valid &
            volume.scattered &
            volume.phase.valid &
            (sample
                 .continuation_probability >
             0.0f);
        $if(eligible) {
            const auto position =
                segment_position +
                sample.ray->direction() *
                    volume.distance;
            Var<luisa::compute::Ray>
                surface_shadow_ray =
                    make_ray(
                        position,
                        light.direction,
                        0.0f,
                        light.maximum_distance);
            Var<luisa::compute::Ray>
                volume_shadow_ray =
                    make_ray(
                        position,
                        light.direction,
                        0.0f,
                        light.maximum_distance);
            const auto surface_transmittance =
                _config.trace_shadow(
                    surface_shadow_ray,
                    surface_ray::
                        invalid_primitive,
                    surface_ray::
                        invalid_primitive,
                    light.light_instance,
                    light.light_primitive,
                    invocation.parameters
                        .transparent_max_bounces,
                    pack_shader_evaluation_state(
                        cycles_path_state::
                            shadow_shader_state(
                                sample.path_depth,
                                sample.diffuse_depth,
                                sample.glossy_depth,
                                sample
                                    .transparent_depth,
                                sample
                                    .transmission_depth)));
            const auto volume_transmittance =
                _volume_shadow->emit(
                    sample,
                    path_stack,
                    volume_shadow_ray,
                    light.light_instance,
                    light.light_primitive);
            const auto phase_pdf =
                select(
                    0.0f,
                    volume.phase.pdf,
                    light.use_mis);
            const auto mis_weight =
                _config.light_transport
                    .nee_light_weight(
                        light.pdf,
                        phase_pdf);
            const auto unshadowed =
                make_float3(
                    volume.phase.value) *
                light.radiance *
                (mis_weight /
                 max(
                     light.pdf,
                     1.0e-20f));
            const auto roulette_weight =
                invocation
                    .sample_light_roulette(
                        unshadowed,
                        bounce
                            .light_terminate_sample);
            const auto continuation =
                volume.throughput /
                sample
                    .continuation_probability;
            const auto contribution =
                invocation
                    .clamp_contribution(
                        continuation *
                            unshadowed *
                            surface_transmittance *
                            volume_transmittance *
                            roulette_weight,
                        sample.path_depth);
            sample.accumulate_radiance(
                contribution,
                true);
            const auto primary_volume =
                (sample.path_flags &
                 cycles_path_state::
                     flag_any_pass) == 0u;
            sample.sample_volume_direct +=
                select(
                    make_float3(0.0f),
                    contribution,
                    primary_volume);
            sample.accumulate_scattered_light(
                select(
                    contribution,
                    make_float3(0.0f),
                    primary_volume));
        };
    }
};

}// namespace

std::unique_ptr<
    VolumeDirectLightingComponent>
make_volume_direct_lighting_component(
    const PathKernelConfig &config) {
    return std::make_unique<
        PathVolumeDirectLightingComponent>(
        config);
}

}// namespace psycles::luisa_backend::detail
