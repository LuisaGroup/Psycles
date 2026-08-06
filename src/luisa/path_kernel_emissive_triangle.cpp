#include "path_kernel_emissive_triangle.h"

#include <psycles/luisa/cycles_path_state.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3 normalize_or(
    Float3 value,
    Float3 fallback) noexcept {
    const auto magnitude_squared =
        dot(value, value);
    return select(
        fallback,
        value /
            sqrt(max(
                magnitude_squared,
                1.0e-20f)),
        magnitude_squared >
            1.0e-20f);
}

[[nodiscard]] Bool samples_front(
    UInt sampling) noexcept {
    return
        (sampling ==
         static_cast<std::uint32_t>(
             contract::EmissionSampling::
                 automatic)) |
        (sampling ==
         static_cast<std::uint32_t>(
             contract::EmissionSampling::
                 front)) |
        (sampling ==
         static_cast<std::uint32_t>(
             contract::EmissionSampling::
                 front_back));
}

[[nodiscard]] Bool samples_back(
    UInt sampling) noexcept {
    return
        (sampling ==
         static_cast<std::uint32_t>(
             contract::EmissionSampling::
                 automatic)) |
        (sampling ==
         static_cast<std::uint32_t>(
             contract::EmissionSampling::
                 back)) |
        (sampling ==
         static_cast<std::uint32_t>(
             contract::EmissionSampling::
                 front_back));
}

[[nodiscard]] Bool sampled_side(
    Bool sample_front,
    Bool sample_back,
    Float3 geometric_normal,
    Float3 direction) noexcept {
    const auto back_facing =
        dot(
            geometric_normal,
            -direction) < 0.0f;
    return select(
        sample_front,
        sample_back,
        back_facing);
}

class PathEmissiveTriangleComponent final
    : public EmissiveTriangleComponent {

  private:
    std::shared_ptr<const TriangleGeometryComponent>
        _triangle_geometry{
            make_triangle_geometry_component()};
    TriangleLightSampling _sampling;

    [[nodiscard]]
    EmissiveTriangleGeometryContext
    _geometry(
        const std::shared_ptr<LuisaSceneData> &scene,
        UInt emitter_index) const noexcept {
        Var<EmissiveTriangleGpu> emitter =
            scene->emissive_triangle_buffer
                ->read(emitter_index);
        auto triangle =
            _triangle_geometry->emit(
                scene,
                emitter.instance_index,
                emitter.primitive_index);
        const auto object_to_world =
            scene->accel->instance_transform(
                emitter.instance_index);
        const auto normal_to_world =
            transpose(
                inverse(
                    object_to_world));
        const auto local_p0 =
            triangle.p0;
        const auto local_p1 =
            triangle.p1;
        const auto local_p2 =
            triangle.p2;
        const auto p0 =
            (object_to_world *
             make_float4(
                 local_p0, 1.0f))
                .xyz();
        const auto p1 =
            (object_to_world *
             make_float4(
                 local_p1, 1.0f))
                .xyz();
        const auto p2 =
            (object_to_world *
             make_float4(
                 local_p2, 1.0f))
                .xyz();
        const auto local_normal =
            normalize_or(
                cross(
                    local_p1 - local_p0,
                    local_p2 - local_p0),
                make_float3(
                    0.0f, 0.0f, 1.0f));
        const auto geometric_normal =
            normalize_or(
                (normal_to_world *
                 make_float4(
                     local_normal, 0.0f))
                    .xyz(),
                make_float3(
                    0.0f, 0.0f, 1.0f));
        const auto doubled_area =
            length(
                cross(
                    p1 - p0,
                    p2 - p0));
        const auto sample_front =
            samples_front(
                emitter.emission_sampling);
        const auto sample_back =
            samples_back(
                emitter.emission_sampling);
        return {
            .emitter =
                std::move(emitter),
            .triangle =
                std::move(triangle),
            .p0 = p0,
            .p1 = p1,
            .p2 = p2,
            .geometric_normal =
                geometric_normal,
            .area =
                0.5f *
                doubled_area,
            .sample_front =
                sample_front,
            .sample_back =
                sample_back};
    }

    [[nodiscard]] Float3
    _evaluate_emission(
        PathSampleContext &sample,
        const EmissiveTriangleGeometryContext
            &geometry,
        const TriangleLightSample
            &light) const noexcept {
        auto &invocation =
            sample.invocation;
        const auto &scene =
            invocation.config.scene;
        const auto &attributes =
            geometry.triangle;
        const auto &primitive =
            attributes.primitive;
        const auto &instance =
            primitive.instance;
        const auto object_to_world =
            scene->accel
                ->instance_transform(
                    geometry.emitter
                        .instance_index);
        const auto normal_to_world =
            transpose(
                inverse(
                    object_to_world));
        const auto local_geometric_normal =
            normalize_or(
                cross(
                    attributes.p1 -
                        attributes.p0,
                    attributes.p2 -
                        attributes.p0),
                make_float3(
                    0.0f, 0.0f, 1.0f));
        auto local_shading_normal =
            select(
                local_geometric_normal,
                triangle_interpolate(
                    light.barycentric,
                    attributes.n0,
                    attributes.n1,
                    attributes.n2),
                primitive.smooth);
        const auto local_tangent =
            triangle_interpolate(
                light.barycentric,
                attributes.tangent0,
                attributes.tangent1,
                attributes.tangent2);
        const auto undisplaced_local_geometric_normal =
            normalize_or(
                cross(
                    attributes.undisplaced_p1 -
                        attributes.undisplaced_p0,
                    attributes.undisplaced_p2 -
                        attributes.undisplaced_p0),
                make_float3(0.0f, 0.0f, 1.0f));
        const auto undisplaced_local_shading_normal =
            select(
                undisplaced_local_geometric_normal,
                triangle_interpolate(
                    light.barycentric,
                    attributes.undisplaced_n0,
                    attributes.undisplaced_n1,
                    attributes.undisplaced_n2),
                primitive.smooth);
        const auto undisplaced_local_position =
            triangle_interpolate(
                light.barycentric,
                attributes.undisplaced_p0,
                attributes.undisplaced_p1,
                attributes.undisplaced_p2);
        const auto undisplaced_position =
            (object_to_world *
             make_float4(undisplaced_local_position, 1.0f))
                .xyz();
        const auto undisplaced_local_tangent =
            triangle_interpolate(
                light.barycentric,
                attributes.undisplaced_tangent0,
                attributes.undisplaced_tangent1,
                attributes.undisplaced_tangent2);
        auto geometric_normal =
            geometry.geometric_normal;
        auto shading_normal =
            normalize_or(
                (normal_to_world *
                 make_float4(
                     local_shading_normal,
                     0.0f))
                    .xyz(),
                geometric_normal);
        const auto back_facing =
            dot(
                geometric_normal,
                -light.direction) <
            0.0f;
        geometric_normal =
            select(
                geometric_normal,
                -geometric_normal,
                back_facing);
        shading_normal =
            select(
                shading_normal,
                -shading_normal,
                back_facing);
        shading_normal =
            select(
                shading_normal,
                -shading_normal,
                dot(
                    shading_normal,
                    geometric_normal) <
                    0.0f);
        auto undisplaced_shading_normal =
            normalize_or(
                (normal_to_world *
                 make_float4(
                     undisplaced_local_shading_normal,
                     0.0f))
                    .xyz(),
                geometric_normal);
        undisplaced_shading_normal =
            select(
                undisplaced_shading_normal,
                -undisplaced_shading_normal,
                back_facing);
        const auto tangent =
            normalize_or(
                (object_to_world *
                 make_float4(
                     local_tangent.xyz(),
                     0.0f))
                    .xyz(),
                normalize_or(
                    (geometry.p1 -
                     geometry.p0) -
                        geometric_normal *
                            dot(
                                geometry.p1 -
                                    geometry.p0,
                                geometric_normal),
                    make_float3(
                        1.0f, 0.0f, 0.0f)));
        SurfacePoint point{
            .position =
                light.position,
            .object_position =
                triangle_interpolate(
                    light.barycentric,
                    attributes.p0,
                    attributes.p1,
                    attributes.p2),
            .object_location =
                (object_to_world *
                 make_float4(
                     0.0f,
                     0.0f,
                     0.0f,
                     1.0f))
                    .xyz(),
            .generated =
                triangle_interpolate(
                    light.barycentric,
                    attributes.generated0,
                    attributes.generated1,
                    attributes.generated2),
            .geometric_normal =
                geometric_normal,
            .shading_normal =
                shading_normal,
            .object_shading_normal =
                local_shading_normal,
            .object_tangent =
                local_tangent.xyz(),
            .tangent_sign =
                local_tangent.w,
            .undisplaced_position =
                undisplaced_position,
            .undisplaced_object_position =
                undisplaced_local_position,
            .undisplaced_shading_normal =
                undisplaced_shading_normal,
            .undisplaced_object_shading_normal =
                undisplaced_local_shading_normal,
            .undisplaced_object_tangent =
                undisplaced_local_tangent.xyz(),
            .undisplaced_tangent_sign =
                undisplaced_local_tangent.w,
            .normal_to_world_x =
                (normal_to_world *
                 make_float4(
                     1.0f,
                     0.0f,
                     0.0f,
                     0.0f))
                    .xyz(),
            .normal_to_world_y =
                (normal_to_world *
                 make_float4(
                     0.0f,
                     1.0f,
                     0.0f,
                     0.0f))
                    .xyz(),
            .normal_to_world_z =
                (normal_to_world *
                 make_float4(
                     0.0f,
                     0.0f,
                     1.0f,
                     0.0f))
                    .xyz(),
            .dpdu = tangent,
            .dpdv =
                cross(
                    shading_normal,
                    tangent),
            .dPdx =
                make_float3(0.0f),
            .dPdy =
                make_float3(0.0f),
            .object_dPdx =
                make_float3(0.0f),
            .object_dPdy =
                make_float3(0.0f),
            .undisplaced_dPdx =
                make_float3(0.0f),
            .undisplaced_dPdy =
                make_float3(0.0f),
            .undisplaced_object_dPdx =
                make_float3(0.0f),
            .undisplaced_object_dPdy =
                make_float3(0.0f),
            .generated_dx =
                make_float3(0.0f),
            .generated_dy =
                make_float3(0.0f),
            .incoming =
                -light.direction,
            .uv =
                triangle_interpolate(
                    light.barycentric,
                    attributes.uv0,
                    attributes.uv1,
                    attributes.uv2),
            .uv_dx =
                make_float2(0.0f),
            .uv_dy =
                make_float2(0.0f),
            .geometry_index =
                geometry.emitter
                    .geometry_index,
            .barycentric =
                light.barycentric,
            .barycentric_dx =
                make_float2(0.0f),
            .barycentric_dy =
                make_float2(0.0f),
            .instance_id =
                geometry.emitter
                    .instance_index,
            .primitive_id =
                geometry.emitter
                    .primitive_index,
            .parameter_block =
                geometry.emitter
                    .parameter_block,
            .object_random =
                instance.object_random,
            .particle_index =
                instance.particle_index,
            .random_per_island =
                attributes
                    .random_per_island,
            .triangle_smooth = primitive.smooth,
            .is_curve = false,
            .curve_intercept = 0.0f,
            .curve_length = 0.0f,
            .curve_thickness = 0.0f,
            .curve_tangent_normal = make_float3(0.0f),
            .curve_random = 0.0f,
            .ray_visibility =
                shadow_visibility,
            .ray_events = 0u,
            .ray_depth =
                sample.path_depth,
            .diffuse_depth =
                sample.diffuse_depth,
            .glossy_depth =
                sample.glossy_depth,
            .transparent_depth =
                sample.transparent_depth,
            .transmission_depth =
                sample.transmission_depth,
            .ray_length =
                light.distance,
            .time = 0.5f,
            // This point is evaluated only for emission. Its material
            // binding is not carried in EmissiveTriangleGpu because bump
            // correction cannot affect an emission closure.
            .use_bump_map_correction = false,
            .back_facing =
                back_facing};
        cycles_path_state::
            apply_shader_state(
                point,
                cycles_path_state::
                    light_emission_shader_state(
                        sample.path_depth,
                        sample.diffuse_depth,
                        sample.glossy_depth,
                        sample
                            .transparent_depth,
                        sample
                            .transmission_depth));
        return invocation
            .surface_emission(
                geometry.emitter
                    .surface_tag,
                point,
                -light.direction);
    }

  public:
    EmissiveTriangleSegmentSample
    from_segment(
        const std::shared_ptr<LuisaSceneData> &scene,
        UInt emitter_index,
        Float3 reference,
        Float2 random) const noexcept override {
        auto geometry =
            _geometry(
                scene,
                emitter_index);
        auto light =
            _sampling.from_segment(
                {.reference =
                     std::move(reference),
                 .p0 = geometry.p0,
                 .p1 = geometry.p1,
                 .p2 = geometry.p2,
                 .random =
                     std::move(random)});
        const auto valid =
            light.valid &
            (geometry.area > 0.0f);
        return {
            .geometry =
                std::move(geometry),
            .light =
                std::move(light),
            .valid = valid};
    }

    EmissiveTriangleLightProposal
    from_position(
        const std::shared_ptr<LuisaSceneData> &scene,
        UInt emitter_index,
        Float3 reference,
        Float2 random,
        Float selection_pdf) const noexcept override {
        auto geometry =
            _geometry(
                scene,
                emitter_index);
        auto light =
            _sampling.from_position(
                {.reference =
                     std::move(reference),
                 .p0 = geometry.p0,
                 .p1 = geometry.p1,
                 .p2 = geometry.p2,
                 .random =
                     std::move(random)});
        const auto side_valid =
            sampled_side(
                geometry.sample_front,
                geometry.sample_back,
                geometry
                    .geometric_normal,
                light.direction);
        const auto pdf =
            light.conditional_pdf *
            selection_pdf;
        const auto valid =
            light.valid &
            side_valid &
            (geometry.area > 0.0f) &
            (pdf > 0.0f);
        return {
            .geometry =
                std::move(geometry),
            .light =
                std::move(light),
            .pdf = pdf,
            .valid = valid};
    }

    Float3 evaluate_emission(
        PathSampleContext &sample,
        const EmissiveTriangleLightProposal
            &proposal) const noexcept override {
        return _evaluate_emission(
            sample,
            proposal.geometry,
            proposal.light);
    }

    Float3 evaluate_constant_emission(
        PathSampleContext &sample,
        const EmissiveTriangleLightProposal
            &proposal) const noexcept override {
        return sample.invocation
            .constant_surface_emission(
                proposal.geometry.emitter.surface_tag,
                proposal.geometry.emitter.parameter_block);
    }

    EmissiveTrianglePdf
    from_intersection(
        Float selection_pdf,
        UInt emission_sampling,
        Float3 reference,
        Float3 light_position,
        Float3 p0,
        Float3 p1,
        Float3 p2,
        Float3 oriented_geometric_normal)
        const noexcept override {
        const auto pdf =
            _sampling.from_intersection(
                {.reference = reference,
                 .p0 = p0,
                 .p1 = p1,
                 .p2 = p2,
                 .random =
                     make_float2(0.0f)},
                light_position);
        const auto offset =
            light_position - reference;
        const auto direction =
            normalize_or(
                offset,
                make_float3(
                    0.0f, 0.0f, 1.0f));
        const auto side_valid =
            sampled_side(
                samples_front(
                    emission_sampling),
                samples_back(
                    emission_sampling),
                oriented_geometric_normal,
                direction);
        const auto area =
            0.5f *
            length(cross(
                p1 - p0,
                p2 - p0));
        const auto value =
            pdf.value *
            selection_pdf;
        const auto valid =
            (emission_sampling !=
             static_cast<std::uint32_t>(
                 contract::EmissionSampling::none)) &
            side_valid &
            pdf.valid &
            (area > 0.0f) &
            (value > 0.0f);
        return {
            .value =
                select(0.0f, value, valid),
            .valid = valid};
    }
};

}// namespace

std::shared_ptr<
    const EmissiveTriangleComponent>
make_emissive_triangle_component() {
    return std::make_shared<
        PathEmissiveTriangleComponent>();
}

}// namespace psycles::luisa_backend::detail
