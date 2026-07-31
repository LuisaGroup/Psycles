#include "path_kernel_builder.h"
#include "path_kernel_triangle_geometry.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceGeometryStageImpl final : public SurfaceGeometryStage {

private:
    std::shared_ptr<const TriangleGeometryComponent>
        _triangle_geometry;

  public:
    SurfaceGeometryStageImpl()
        : _triangle_geometry{
              make_triangle_geometry_component()} {}

    SurfaceGeometryContext
    emit(PathBounceContext &bounce) const noexcept override {
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &scene = config.scene;
        auto &hit = bounce.hit;
        auto &ray = sample.ray;
        auto &ray_dP = sample.ray_dP;
        auto &ray_dD = sample.ray_dD;
        auto &ray_visibility = sample.ray_visibility;
        auto &ray_events = sample.ray_events;
        auto &path_depth = sample.path_depth;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transparent_depth = sample.transparent_depth;
        auto &transmission_depth = sample.transmission_depth;
        auto &terminate_after_transparent = sample.terminate_after_transparent;
        auto &minimum_bsdf_pdf = sample.minimum_bsdf_pdf;
        const auto &surface_query = invocation.surface_query;
        const auto &kernel_parameters = invocation.parameters;
        const auto &safe_normalize = config.light_transport.safe_normalize;
        const auto reflective_caustics = config.reflective_caustics;
        const auto refractive_caustics = config.refractive_caustics;
        auto surface_shading_normal = [&](UInt surface_tag,
                                          const SurfacePoint &point) noexcept {
            return invocation.surface_shading_normal(surface_tag, point);
        };
        auto triangle_attributes =
            _triangle_geometry->emit(
                scene,
                hit->inst,
                hit->prim);
        auto &primitive =
            triangle_attributes.primitive;
        auto &instance = primitive.instance;
        auto &p0 = triangle_attributes.p0;
        auto &p1 = triangle_attributes.p1;
        auto &p2 = triangle_attributes.p2;
        auto &n0 = triangle_attributes.n0;
        auto &n1 = triangle_attributes.n1;
        auto &n2 = triangle_attributes.n2;
        auto &uv0 = triangle_attributes.uv0;
        auto &uv1 = triangle_attributes.uv1;
        auto &uv2 = triangle_attributes.uv2;
        auto &tangent0 = triangle_attributes.tangent0;
        auto &tangent1 = triangle_attributes.tangent1;
        auto &tangent2 = triangle_attributes.tangent2;
        auto &generated0 = triangle_attributes.generated0;
        auto &generated1 = triangle_attributes.generated1;
        auto &generated2 = triangle_attributes.generated2;
        auto &random_per_island =
            triangle_attributes.random_per_island;
        auto &triangle_smooth =
            primitive.smooth;

        auto object_to_world = scene->accel->instance_transform(hit->inst);
        auto world_to_object = inverse(object_to_world);
        auto normal_to_world = transpose(world_to_object);
        Float3 wp0 = (object_to_world * make_float4(p0, 1.0f)).xyz();
        Float3 wp1 = (object_to_world * make_float4(p1, 1.0f)).xyz();
        Float3 wp2 = (object_to_world * make_float4(p2, 1.0f)).xyz();
        Float3 object_geometric_normal = safe_normalize(
            cross(p1 - p0, p2 - p0), make_float3(0.0f, 0.0f, 1.0f));
        Float3 geometric_normal = safe_normalize(
            (normal_to_world * make_float4(object_geometric_normal, 0.0f))
                .xyz(),
            -ray->direction());
        Float3 object_shading_normal = select(
            object_geometric_normal,
            triangle_interpolate(hit->bary, n0, n1, n2),
            triangle_smooth);
        Float4 object_tangent =
            triangle_interpolate(hit->bary, tangent0, tangent1, tangent2);
        Float3 shading_normal = safe_normalize(
            (normal_to_world * make_float4(object_shading_normal, 0.0f)).xyz(),
            geometric_normal);
        Bool back_facing = dot(geometric_normal, -ray->direction()) < 0.0f;
        geometric_normal =
            select(geometric_normal, -geometric_normal, back_facing);
        shading_normal = select(shading_normal, -shading_normal, back_facing);
        shading_normal = select(shading_normal,
                                -shading_normal,
                                dot(shading_normal, geometric_normal) < 0.0f);
        // Reconstruct static-triangle shading points from the
        // committed barycentrics. This is both more accurate than
        // origin + t * direction at large world coordinates and
        // provides the same geometric point used by Cycles before
        // spawning secondary and shadow rays.
        Float3 object_shading_position =
            p0 + hit->bary.x * (p1 - p0) + hit->bary.y * (p2 - p0);
        Float3 hit_position =
            (object_to_world * make_float4(object_shading_position, 1.0f))
                .xyz();
        Float3 object_hit_position =
            (world_to_object * make_float4(hit_position, 1.0f)).xyz();
        Float3 tangent = safe_normalize(
            (object_to_world * make_float4(object_tangent.xyz(), 0.0f)).xyz(),
            safe_normalize((wp1 - wp0) - geometric_normal *
                                             dot(wp1 - wp0, geometric_normal),
                           make_float3(1.0f, 0.0f, 0.0f)));
        Float surface_radius = ray_dP + hit->committed_ray_t * ray_dD;
        // Cycles stores a compact scalar ray differential.
        // ShaderData reconstructs its two surface directions
        // with make_orthonormals(sd->Ng), rather than retaining
        // the full camera-ray differential vectors.
        auto normal_components_differ =
            (geometric_normal.x != geometric_normal.y) |
            (geometric_normal.x != geometric_normal.z);
        Float3 compact_dx =
            select(make_float3(geometric_normal.z - geometric_normal.y,
                               geometric_normal.x + geometric_normal.z,
                               -geometric_normal.y - geometric_normal.x),
                   make_float3(geometric_normal.z - geometric_normal.y,
                               geometric_normal.x - geometric_normal.z,
                               geometric_normal.y - geometric_normal.x),
                   normal_components_differ);
        compact_dx = safe_normalize(compact_dx, tangent);
        Float3 compact_dy = cross(geometric_normal, compact_dx);
        Float3 dPdx = compact_dx * surface_radius;
        Float3 dPdy = compact_dy * surface_radius;
        Float differential_radius = surface_radius;
        Float3 edge1 = wp1 - wp0;
        Float3 edge2 = wp2 - wp0;
        Float gram00 = dot(edge1, edge1);
        Float gram01 = dot(edge1, edge2);
        Float gram11 = dot(edge2, edge2);
        Float gram_determinant = gram00 * gram11 - gram01 * gram01;
        Bool valid_gram = abs(gram_determinant) > 1.0e-20f;
        Float safe_gram_determinant =
            select(1.0f, gram_determinant, valid_gram);
        auto barycentric_differential = [&](Float3 differential) noexcept {
            Float projected1 = dot(differential, edge1);
            Float projected2 = dot(differential, edge2);
            Float2 delta =
                make_float2((projected1 * gram11 - projected2 * gram01) /
                                safe_gram_determinant,
                            (projected2 * gram00 - projected1 * gram01) /
                                safe_gram_determinant);
            delta = select(make_float2(0.0f), delta, valid_gram);
            // Cycles flips dPdu/dPdv when ShaderData is
            // oriented to a backfacing hit. Solving the
            // compact surface differential in that basis
            // therefore reverses arbitrary attribute
            // derivatives while leaving world P unchanged.
            return select(delta, -delta, back_facing);
        };
        Float2 barycentric_dx = barycentric_differential(dPdx);
        Float2 barycentric_dy = barycentric_differential(dPdy);
        Float3 object_dPdx = (world_to_object * make_float4(dPdx, 0.0f)).xyz();
        Float3 object_dPdy = (world_to_object * make_float4(dPdy, 0.0f)).xyz();
        Float3 generated_dx = (generated1 - generated0) * barycentric_dx.x +
                              (generated2 - generated0) * barycentric_dx.y;
        Float3 generated_dy = (generated1 - generated0) * barycentric_dy.x +
                              (generated2 - generated0) * barycentric_dy.y;
        Float2 uv = triangle_interpolate(hit->bary, uv0, uv1, uv2);
        Float2 uv_dx =
            (uv1 - uv0) * barycentric_dx.x + (uv2 - uv0) * barycentric_dx.y;
        Float2 uv_dy =
            (uv1 - uv0) * barycentric_dy.x + (uv2 - uv0) * barycentric_dy.y;

        auto &material_binding =
            primitive.material_binding;
        UInt surface_tag = material_binding.surface_tag;
        UInt cycles_surface_shader =
            primitive.cycles_surface_shader;
        UInt cycles_object_index =
            primitive.cycles_object_index;
        auto volume_stack_entry =
            primitive.volume_stack_entry();
        Bool surface_has_volume =
            primitive.has_volume;

        SurfacePoint point{
            .position = hit_position,
            .object_position = object_shading_position,
            .object_location =
                (object_to_world * make_float4(0.0f, 0.0f, 0.0f, 1.0f)).xyz(),
            .generated = triangle_interpolate(
                hit->bary, generated0, generated1, generated2),
            .geometric_normal = geometric_normal,
            .shading_normal = shading_normal,
            .object_shading_normal = object_shading_normal,
            .object_tangent = object_tangent.xyz(),
            .tangent_sign = object_tangent.w,
            .normal_to_world_x =
                (normal_to_world * make_float4(1.0f, 0.0f, 0.0f, 0.0f)).xyz(),
            .normal_to_world_y =
                (normal_to_world * make_float4(0.0f, 1.0f, 0.0f, 0.0f)).xyz(),
            .normal_to_world_z =
                (normal_to_world * make_float4(0.0f, 0.0f, 1.0f, 0.0f)).xyz(),
            .dpdu = tangent,
            .dpdv = cross(shading_normal, tangent),
            .dPdx = dPdx,
            .dPdy = dPdy,
            .object_dPdx = object_dPdx,
            .object_dPdy = object_dPdy,
            .generated_dx = generated_dx,
            .generated_dy = generated_dy,
            .incoming = -ray->direction(),
            .uv = uv,
            .uv_dx = uv_dx,
            .uv_dy = uv_dy,
            .geometry_index = instance.geometry_index,
            .barycentric = hit->bary,
            .barycentric_dx = barycentric_dx,
            .barycentric_dy = barycentric_dy,
            .instance_id = hit->inst,
            .primitive_id = hit->prim,
            .parameter_block = material_binding.parameter_block,
            .object_random = instance.object_random,
            .particle_index = instance.particle_index,
            .random_per_island = random_per_island,
            .ray_visibility = ray_visibility,
            .ray_events = ray_events,
            .ray_depth = path_depth,
            .diffuse_depth = diffuse_depth,
            .glossy_depth = glossy_depth,
            .transparent_depth = transparent_depth,
            .transmission_depth = transmission_depth,
            .ray_length = hit->committed_ray_t,
            .time = 0.0f,
            .back_facing = back_facing};
        Float3 shadow_shading_normal = shading_normal;
        $if(triangle_smooth &
            (instance.shadow_terminator_geometry_offset > 0.0f)) {
            shadow_shading_normal = surface_shading_normal(surface_tag, point);
        };
        UInt path_lobe_mask = surface_query.lobe_mask;
        Bool previous_ray_was_diffuse =
            (ray_events &
             static_cast<std::uint32_t>(contract::event_diffuse)) != 0u;
        if (!reflective_caustics) {
            path_lobe_mask =
                select(path_lobe_mask,
                       path_lobe_mask &
                           ~static_cast<std::uint32_t>(contract::event_glossy),
                       previous_ray_was_diffuse);
        }
        if (!refractive_caustics) {
            path_lobe_mask =
                select(path_lobe_mask,
                       path_lobe_mask & ~static_cast<std::uint32_t>(
                                            contract::event_transmission),
                       previous_ray_was_diffuse);
        }
        // Cycles' PATH_RAY_TERMINATE_AFTER_TRANSPARENT
        // evaluates emission but allocates only transparent
        // closures. Filtering the query here also renormalizes
        // mixed transparent/opaque closure selection instead of
        // probabilistically losing the transparent branch.
        path_lobe_mask =
            select(path_lobe_mask,
                   static_cast<std::uint32_t>(contract::event_transparent),
                   terminate_after_transparent);
        SurfaceQuery path_surface_query{.lobe_mask = path_lobe_mask,
                                        .transport_mode =
                                            surface_query.transport_mode,
                                        .glossy_filter_roughness = 0.0f};
        auto blur_pdf = kernel_parameters.filter_glossy * minimum_bsdf_pdf;
        auto filter_glossy_enabled =
            kernel_parameters.filter_glossy < std::numeric_limits<float>::max();
        path_surface_query.glossy_filter_roughness =
            select(0.0f,
                   sqrt(max(1.0f - blur_pdf, 0.0f)) * 0.5f,
                   filter_glossy_enabled & (blur_pdf < 1.0f));
        return {bounce,
                std::move(instance),
                std::move(p0),
                std::move(p1),
                std::move(p2),
                std::move(n0),
                std::move(n1),
                std::move(n2),
                std::move(object_to_world),
                std::move(world_to_object),
                std::move(wp0),
                std::move(wp1),
                std::move(wp2),
                std::move(geometric_normal),
                std::move(hit_position),
                std::move(object_hit_position),
                std::move(differential_radius),
                std::move(triangle_smooth),
                std::move(shadow_shading_normal),
                std::move(surface_tag),
                std::move(cycles_surface_shader),
                std::move(cycles_object_index),
                std::move(volume_stack_entry),
                std::move(surface_has_volume),
                std::move(point),
                std::move(path_surface_query)};
    }
};

} // namespace

Float3
SurfaceGeometryContext::make_ray_origin(Float3 direction) const noexcept {
    Float3 object_direction =
        (world_to_object * make_float4(direction, 0.0f)).xyz();
    return surface_ray::origin_with_explicit_self_exclusion(hit_position,
                                                            geometric_normal,
                                                            object_hit_position,
                                                            object_direction,
                                                            p0,
                                                            p1,
                                                            p2);
}

surface_ray::ShadowOrigin
SurfaceGeometryContext::make_shadow_origin(Float3 direction) const noexcept {
    return surface_ray::surface_shadow_origin(
        hit_position,
        shadow_shading_normal,
        geometric_normal,
        direction,
        instance.shadow_terminator_geometry_offset,
        triangle_smooth,
        object_to_world,
        world_to_object,
        bounce.hit->bary,
        p0,
        p1,
        p2,
        n0,
        n1,
        n2);
}

std::unique_ptr<SurfaceGeometryStage> make_surface_geometry_stage() {
    return std::make_unique<SurfaceGeometryStageImpl>();
}

} // namespace psycles::luisa_backend::detail
