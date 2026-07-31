#include "path_kernel_builder.h"
#include "path_kernel_triangle_geometry.h"

#include <psycles/luisa/spherical_geometry.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class EmissiveMeshLightingComponent final : public DirectLightingComponent {

private:
    std::shared_ptr<const TriangleGeometryComponent>
        _triangle_geometry;

  public:
    EmissiveMeshLightingComponent()
        : _triangle_geometry{
              make_triangle_geometry_component()} {}

    void emit(DirectLightingContext &context) const noexcept override {
        auto &bounce = context.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &scene = config.scene;
        auto &surface = context.surface;
        auto &selected_light = bounce.selected_light;
        auto &light_sample = bounce.light_sample;
        auto &light_terminate_sample = bounce.light_terminate_sample;
        auto &hit = bounce.hit;
        auto &hit_position = surface.hit_position;
        auto &surface_tag = surface.surface_tag;
        auto &point = surface.point;
        auto &path_surface_query = surface.path_surface_query;
        auto &throughput = sample.throughput;
        auto &path_depth = sample.path_depth;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transparent_depth = sample.transparent_depth;
        auto &transmission_depth = sample.transmission_depth;
        auto &path_diffuse_weight = sample.path_diffuse_weight;
        auto &path_glossy_weight = sample.path_glossy_weight;
        const auto &kernel_parameters = invocation.parameters;
        const auto &safe_normalize = config.light_transport.safe_normalize;
        const auto &emissive_triangle_pdf = config.emissive_triangle_pdf;
        const auto &trace_shadow = config.trace_shadow;
        const auto &nee_light_weight = config.light_transport.nee_light_weight;
        const auto &split_nee_light = config.light_transport.split_nee_light;
        auto make_surface_shadow_origin = [&](Float3 direction) noexcept {
            return surface.make_shadow_origin(direction);
        };
        auto surface_emission = [&](UInt tag,
                                    const SurfacePoint &surface_point,
                                    Float3 outgoing) noexcept {
            return invocation.surface_emission(tag, surface_point, outgoing);
        };
        auto evaluate_surface = [&](UInt tag,
                                    const SurfacePoint &surface_point,
                                    Float3 outgoing,
                                    const SurfaceQuery &query) noexcept {
            return invocation.evaluate_surface(
                tag, surface_point, outgoing, query);
        };
        auto sample_light_roulette = [&](Float3 contribution,
                                         Float random) noexcept {
            return invocation.sample_light_roulette(contribution, random);
        };
        auto clamp_contribution = [&](Float3 contribution,
                                      UInt depth) noexcept {
            return invocation.clamp_contribution(contribution, depth);
        };
        auto accumulate_light_pass =
            [&](Var<LightPassContributionCall> contribution) noexcept {
                sample.accumulate_light_pass(std::move(contribution));
            };
        $if(selected_light.kind ==
            static_cast<std::uint32_t>(
                sampling::LightDistributionEmitterKind::emissive_triangle)) {
            Var<EmissiveTriangleGpu> emitter =
                scene->emissive_triangle_buffer->read(selected_light.index);
            auto light_attributes =
                _triangle_geometry->emit(
                    scene,
                    emitter.instance_index,
                    emitter.primitive_index);
            auto &light_primitive =
                light_attributes.primitive;
            auto &lp0 = light_attributes.p0;
            auto &lp1 = light_attributes.p1;
            auto &lp2 = light_attributes.p2;
            auto &ln0 = light_attributes.n0;
            auto &ln1 = light_attributes.n1;
            auto &ln2 = light_attributes.n2;
            auto &luv0 = light_attributes.uv0;
            auto &luv1 = light_attributes.uv1;
            auto &luv2 = light_attributes.uv2;
            auto &light_tangent0 =
                light_attributes.tangent0;
            auto &light_tangent1 =
                light_attributes.tangent1;
            auto &light_tangent2 =
                light_attributes.tangent2;
            auto &light_generated0 =
                light_attributes.generated0;
            auto &light_generated1 =
                light_attributes.generated1;
            auto &light_generated2 =
                light_attributes.generated2;
            auto &light_random_per_island =
                light_attributes.random_per_island;
            auto &light_instance =
                light_primitive.instance;
            Float3 local_lp0 = lp0;
            Float3 local_lp1 = lp1;
            Float3 local_lp2 = lp2;
            auto light_object_to_world =
                scene->accel->instance_transform(emitter.instance_index);
            auto light_normal_to_world =
                transpose(inverse(light_object_to_world));
            lp0 = (light_object_to_world * make_float4(lp0, 1.0f)).xyz();
            lp1 = (light_object_to_world * make_float4(lp1, 1.0f)).xyz();
            lp2 = (light_object_to_world * make_float4(lp2, 1.0f)).xyz();
            auto triangle_sample = spherical_geometry::sample_triangle(
                hit_position, lp0, lp1, lp2, light_sample.xy());
            Float2 light_barycentric = triangle_sample.barycentric;
            Float3 light_position = triangle_sample.position;
            Float3 light_object_geometric_normal = safe_normalize(
                cross(local_lp1 - local_lp0, local_lp2 - local_lp0),
                make_float3(0.0f, 0.0f, 1.0f));
            Float3 light_geometric_normal = safe_normalize(
                (light_normal_to_world *
                 make_float4(light_object_geometric_normal, 0.0f))
                    .xyz(),
                make_float3(0.0f, 0.0f, 1.0f));
            Float3 light_object_shading_normal = select(
                light_object_geometric_normal,
                triangle_interpolate(
                    light_barycentric, ln0, ln1, ln2),
                light_primitive.smooth);
            Float4 light_object_tangent =
                triangle_interpolate(light_barycentric,
                                     light_tangent0,
                                     light_tangent1,
                                     light_tangent2);
            Float3 light_shading_normal =
                safe_normalize((light_normal_to_world *
                                make_float4(light_object_shading_normal, 0.0f))
                                   .xyz(),
                               light_geometric_normal);
            Float light_distance = triangle_sample.distance;
            Float3 wi = triangle_sample.direction;
            Bool light_back_facing = dot(light_geometric_normal, -wi) < 0.0f;
            light_geometric_normal = select(light_geometric_normal,
                                            -light_geometric_normal,
                                            light_back_facing);
            light_shading_normal = select(
                light_shading_normal, -light_shading_normal, light_back_facing);
            light_shading_normal = select(
                light_shading_normal,
                -light_shading_normal,
                dot(light_shading_normal, light_geometric_normal) < 0.0f);
            Float3 light_tangent = safe_normalize(
                (light_object_to_world *
                 make_float4(light_object_tangent.xyz(), 0.0f))
                    .xyz(),
                safe_normalize((lp1 - lp0) -
                                   light_geometric_normal *
                                       dot(lp1 - lp0, light_geometric_normal),
                               make_float3(1.0f, 0.0f, 0.0f)));
            SurfacePoint light_point{
                .position = light_position,
                .object_position = triangle_interpolate(
                    light_barycentric, local_lp0, local_lp1, local_lp2),
                .object_location = (light_object_to_world *
                                    make_float4(0.0f, 0.0f, 0.0f, 1.0f))
                                       .xyz(),
                .generated = triangle_interpolate(light_barycentric,
                                                  light_generated0,
                                                  light_generated1,
                                                  light_generated2),
                .geometric_normal = light_geometric_normal,
                .shading_normal = light_shading_normal,
                .object_shading_normal = light_object_shading_normal,
                .object_tangent = light_object_tangent.xyz(),
                .tangent_sign = light_object_tangent.w,
                .normal_to_world_x = (light_normal_to_world *
                                      make_float4(1.0f, 0.0f, 0.0f, 0.0f))
                                         .xyz(),
                .normal_to_world_y = (light_normal_to_world *
                                      make_float4(0.0f, 1.0f, 0.0f, 0.0f))
                                         .xyz(),
                .normal_to_world_z = (light_normal_to_world *
                                      make_float4(0.0f, 0.0f, 1.0f, 0.0f))
                                         .xyz(),
                .dpdu = light_tangent,
                .dpdv = cross(light_shading_normal, light_tangent),
                .dPdx = make_float3(0.0f),
                .dPdy = make_float3(0.0f),
                .object_dPdx = make_float3(0.0f),
                .object_dPdy = make_float3(0.0f),
                .generated_dx = make_float3(0.0f),
                .generated_dy = make_float3(0.0f),
                .incoming = -wi,
                .uv = triangle_interpolate(light_barycentric, luv0, luv1, luv2),
                .uv_dx = make_float2(0.0f),
                .uv_dy = make_float2(0.0f),
                .geometry_index = emitter.geometry_index,
                .barycentric = light_barycentric,
                .barycentric_dx = make_float2(0.0f),
                .barycentric_dy = make_float2(0.0f),
                .instance_id = emitter.instance_index,
                .primitive_id = emitter.primitive_index,
                .parameter_block = emitter.parameter_block,
                .object_random = light_instance.object_random,
                .particle_index = light_instance.particle_index,
                .random_per_island = light_random_per_island,
                .ray_visibility = shadow_visibility,
                .ray_events = 0u,
                .ray_depth = path_depth,
                .diffuse_depth = diffuse_depth,
                .glossy_depth = glossy_depth,
                .transparent_depth = transparent_depth,
                .transmission_depth = transmission_depth,
                .ray_length = light_distance,
                .time = 0.5f,
                .back_facing = light_back_facing};
            cycles_path_state::apply_shader_state(
                light_point,
                cycles_path_state::light_emission_shader_state(
                    path_depth,
                    diffuse_depth,
                    glossy_depth,
                    transparent_depth,
                    transmission_depth));
            Float3 light_radiance =
                surface_emission(emitter.surface_tag, light_point, -wi);
            Float light_pdf = emissive_triangle_pdf(emitter.instance_index,
                                                    emitter.primitive_index,
                                                    hit_position,
                                                    light_position,
                                                    lp0,
                                                    lp1,
                                                    lp2);
            $if(triangle_sample.valid & (light_pdf > 0.0f) &
                any(light_radiance > 0.0f)) {
                const auto shadow = make_surface_shadow_origin(wi);
                const auto shadow_offset = light_position - shadow.position;
                const auto shadow_distance =
                    sqrt(max(length_squared(shadow_offset), 1.0e-20f));
                const auto shadow_direction = shadow_offset / shadow_distance;
                Var<luisa::compute::Ray> mesh_light_shadow_ray = make_ray(
                    shadow.position, shadow_direction, 0.0f, shadow_distance);
                Float3 shadow_transmittance =
                    trace_shadow(mesh_light_shadow_ray,
                                 select(surface_ray::invalid_primitive,
                                        hit->inst,
                                        shadow.skip_self),
                                 select(surface_ray::invalid_primitive,
                                        hit->prim,
                                        shadow.skip_self),
                                 emitter.instance_index,
                                 emitter.primitive_index,
                                 kernel_parameters.transparent_max_bounces,
                                 pack_shader_evaluation_state(
                                     cycles_path_state::shadow_shader_state(
                                         path_depth,
                                         diffuse_depth,
                                         glossy_depth,
                                         transparent_depth,
                                         transmission_depth)));
                $if(any(shadow_transmittance > 0.0f)) {
                    auto evaluation = evaluate_surface(
                        surface_tag, point, wi, path_surface_query);
                    Float mis_weight =
                        nee_light_weight(light_pdf, evaluation.pdf);
                    Float3 unshadowed_contribution = evaluation.f *
                                                     light_radiance *
                                                     (mis_weight / light_pdf);
                    Float roulette_weight = sample_light_roulette(
                        unshadowed_contribution, light_terminate_sample);
                    Float3 contribution = clamp_contribution(
                        throughput * unshadowed_contribution *
                            shadow_transmittance * roulette_weight,
                        path_depth);
                    sample.accumulate_radiance(
                        contribution);
                    accumulate_light_pass(split_nee_light(contribution,
                                                          evaluation.f,
                                                          evaluation.diffuse_f,
                                                          path_diffuse_weight,
                                                          path_glossy_weight,
                                                          path_depth));
                };
            };
        };
    }
};

} // namespace

std::unique_ptr<DirectLightingComponent>
make_emissive_mesh_lighting_component() {
    return std::make_unique<EmissiveMeshLightingComponent>();
}

} // namespace psycles::luisa_backend::detail
