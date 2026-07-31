#include "path_kernel_volume_direct_light.h"

#include "path_kernel_volume_shadow.h"

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/spherical_geometry.h>
#include <psycles/luisa/surface_ray.h>
#include <psycles/sampling/light_distribution.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class DistantVolumeDirectLightingComponent final
    : public VolumeDirectLightingComponent {

  private:
    PathKernelConfig _config;
    std::unique_ptr<
        HomogeneousVolumeShadowComponent>
        _volume_shadow;

  public:
    explicit DistantVolumeDirectLightingComponent(
        const PathKernelConfig &config)
        : _config{config},
          _volume_shadow{
              make_homogeneous_volume_shadow_component(
                  config)} {}

    VolumeDirectLightSample sample(
        const ClosestPathEvent &event)
        const noexcept override {
        const auto &bounce = event.bounce;
        const auto &path_sample =
            bounce.sample;
        const auto &scene =
            _config.scene;
        const auto &selected =
            bounce.selected_light;
        const auto &random =
            bounce.light_sample;

        const auto selected_analytic =
            selected.kind ==
            static_cast<std::uint32_t>(
                sampling::
                    LightDistributionEmitterKind::
                        analytic_light);
        VolumeDirectLightSample result{
            .direction =
                make_float3(0.0f),
            .radiance =
                make_float3(0.0f),
            .pdf = 0.0f,
            .maximum_distance =
                ray_maximum,
            .light_instance =
                surface_ray::
                    invalid_primitive,
            .light_primitive =
                surface_ray::
                    invalid_primitive,
            .use_mis = false,
            .valid = false};
        // Distribution entries for emissive meshes and the environment do
        // not index light_buffer. Keep the device read structurally inside
        // the analytic-emitter arm instead of masking an out-of-range read
        // after the fact.
        $if(selected_analytic) {
            const auto light_index =
                selected.index;
            Var<LightGpu> light =
                scene->light_buffer->read(
                    light_index);
            const auto distant =
                light.type ==
                static_cast<std::uint32_t>(
                    contract::LightType::
                        distant);

            const auto half_angle =
                0.5f *
                max(light.angle, 0.0f);
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
                sqrt(
                    max(
                        1.0f -
                            cosine_theta *
                                cosine_theta,
                        0.0f));
            const auto phi =
                2.0f * pi * random.y;
            const auto cone_direction =
                sun_tangent *
                    (cos(phi) *
                     sine_theta) +
                sun_bitangent *
                    (sin(phi) *
                     sine_theta) +
                sun_axis * cosine_theta;
            const auto direction =
                select(
                    sun_axis,
                    cone_direction,
                    finite_sun);
            const auto solid_angle =
                spherical_geometry::
                    two_pi *
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
                 light_flag_normalize) !=
                0u;
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
                path_sample
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
                selected.selection_pdf;
            const auto visible_to_volume =
                (light.visibility_mask &
                 contract::visibility_bit(
                     contract::
                         RayVisibility::
                             volume_scatter)) !=
                0u;
            const auto valid =
                distant &
                visible_to_volume &
                (pdf > 0.0f);
            result.direction =
                select(
                    make_float3(0.0f),
                    direction,
                    valid);
            result.radiance =
                select(
                    make_float3(0.0f),
                    radiance,
                    valid);
            result.pdf =
                select(
                    0.0f,
                    pdf,
                    valid);
            result.use_mis =
                valid &
                ((light.flags &
                  light_flag_use_mis) !=
                 0u);
            result.valid = valid;
        };
        return result;
    }

    void accumulate(
        ClosestPathEvent &event,
        const VolumeDirectLightSample &light,
        const HomogeneousVolumeSegmentResult &volume,
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
            volume.direct_transport
                .scattered &
            volume.direct_phase.valid &
            (sample
                 .continuation_probability >
             0.0f);
        $if(eligible) {
            const auto position =
                segment_position +
                sample.ray->direction() *
                    volume.direct_transport
                        .distance;
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
                    volume.direct_phase.pdf,
                    light.use_mis);
            const auto mis_weight =
                _config.light_transport
                    .nee_light_weight(
                        light.pdf,
                        phase_pdf);
            const auto unshadowed =
                make_float3(
                    volume.direct_phase
                        .value) *
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
                volume.direct_transport
                    .throughput /
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
        DistantVolumeDirectLightingComponent>(
        config);
}

}// namespace psycles::luisa_backend::detail
