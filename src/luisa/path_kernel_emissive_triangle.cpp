#include "path_kernel_emissive_triangle.h"
#include "cycles_shader_identity.h"
#include "path_tracer_cycles_svm_emission.h"

#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/native_vector_math.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3 normalize_or(
    Float3 value,
    Float3 fallback) noexcept {
    return native_vector_math::normalize_above_or(
        value, fallback, 1.0e-20f);
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
        const auto transform_applied =
            triangle.primitive.instance.cycles_transform_applied != 0u;
        const auto normal_to_world = instance_normal_to_world(
            scene,
            emitter.instance_index,
            triangle.primitive.instance);
        const auto local_p0 =
            triangle.p0;
        const auto local_p1 =
            triangle.p1;
        const auto local_p2 =
            triangle.p2;
        Float3 p0 = triangle.cycles_intersection_p0;
        Float3 p1 = triangle.cycles_intersection_p1;
        Float3 p2 = triangle.cycles_intersection_p2;
        $if(!transform_applied) {
            const auto object_to_world =
                scene->accel->instance_transform(
                    emitter.instance_index);
            p0 =
                (object_to_world *
                 make_float4(
                     local_p0, 1.0f))
                    .xyz();
            p1 =
                (object_to_world *
                 make_float4(
                     local_p1, 1.0f))
                    .xyz();
            p2 =
                (object_to_world *
                 make_float4(
                     local_p2, 1.0f))
                    .xyz();
        };
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

  public:
    EmissiveTriangleSegmentSample
    from_segment(
        const std::shared_ptr<LuisaSceneData> &scene,
        UInt emitter_index,
        Float3 reference,
        Float2 random) const noexcept override {
        if (scene->emissive_triangle_geometry_buffer.has_value()) {
            const auto geom =
                (*scene->emissive_triangle_geometry_buffer)->read(emitter_index);
            const auto p0 = make_float3(
                as<float>(geom.p0_area.x),
                as<float>(geom.p0_area.y),
                as<float>(geom.p0_area.z));
            const auto area = as<float>(geom.p0_area.w);
            const auto p1 = make_float3(
                as<float>(geom.p1_packed_meta.x),
                as<float>(geom.p1_packed_meta.y),
                as<float>(geom.p1_packed_meta.z));
            const auto packed_meta = geom.p1_packed_meta.w;
            const auto p2 = make_float3(
                as<float>(geom.p2_object.x),
                as<float>(geom.p2_object.y),
                as<float>(geom.p2_object.z));
            const auto geometric_normal = make_float3(
                as<float>(geom.normal_primitive.x),
                as<float>(geom.normal_primitive.y),
                as<float>(geom.normal_primitive.z));
            const auto emission_sampling =
                (packed_meta >> emissive_geom_sampling_shift) & 7u;
            const auto sample_front = samples_front(emission_sampling);
            const auto sample_back = samples_back(emission_sampling);
            auto light = _sampling.from_segment(
                {.reference = std::move(reference),
                 .p0 = p0,
                 .p1 = p1,
                 .p2 = p2,
                 .random = std::move(random)});
            const auto valid = light.valid & (area > 0.0f);
            Var<EmissiveTriangleGpu> emitter;
            $if(valid) {
                emitter = scene->emissive_triangle_buffer->read(emitter_index);
            };
            return {
                .geometry =
                    {.emitter = std::move(emitter),
                     .p0 = p0,
                     .p1 = p1,
                     .p2 = p2,
                     .geometric_normal = geometric_normal,
                     .area = area,
                     .sample_front = sample_front,
                     .sample_back = sample_back},
                .light = std::move(light),
                .valid = valid};
        }
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
        Float selection_pdf,
        bool require_full_emitter,
        std::optional<Float3> receiver_upper_hemisphere_normal,
        std::optional<Bool> receiver_upper_hemisphere_only) const noexcept override {
        if (scene->emissive_triangle_geometry_buffer.has_value()) {
            const auto geom =
                (*scene->emissive_triangle_geometry_buffer)->read(emitter_index);
            const auto p0 = make_float3(
                as<float>(geom.p0_area.x),
                as<float>(geom.p0_area.y),
                as<float>(geom.p0_area.z));
            const auto area = as<float>(geom.p0_area.w);
            const auto p1 = make_float3(
                as<float>(geom.p1_packed_meta.x),
                as<float>(geom.p1_packed_meta.y),
                as<float>(geom.p1_packed_meta.z));
            const auto packed_meta = geom.p1_packed_meta.w;
            const auto p2 = make_float3(
                as<float>(geom.p2_object.x),
                as<float>(geom.p2_object.y),
                as<float>(geom.p2_object.z));
            const auto geometric_normal = make_float3(
                as<float>(geom.normal_primitive.x),
                as<float>(geom.normal_primitive.y),
                as<float>(geom.normal_primitive.z));
            const auto emission_sampling =
                (packed_meta >> emissive_geom_sampling_shift) & 7u;
            const auto sample_front = samples_front(emission_sampling);
            const auto sample_back = samples_back(emission_sampling);

            Bool can_sample = (area > 0.0f) & (sample_front | sample_back);
            const auto rel0 = reference - p0;
            const auto plane_dot = dot(geometric_normal, rel0);
            const auto plane_dot_sq = plane_dot * plane_dot;
            const auto rel0_sq = dot(rel0, rel0);
            const auto strictly_back =
                (plane_dot < -1.0e-4f) & (plane_dot_sq > 1.0e-8f * rel0_sq);
            const auto strictly_front =
                (plane_dot > 1.0e-4f) & (plane_dot_sq > 1.0e-8f * rel0_sq);
            can_sample = can_sample &
                !((sample_front & !sample_back & strictly_back) |
                  (!sample_front & sample_back & strictly_front));
            if (receiver_upper_hemisphere_normal.has_value() &&
                receiver_upper_hemisphere_only.has_value()) {
                const auto &rn = *receiver_upper_hemisphere_normal;
                const auto d0 = p0 - reference;
                const auto d1 = p1 - reference;
                const auto d2 = p2 - reference;
                const auto c0 = dot(d0, rn);
                const auto c1 = dot(d1, rn);
                const auto c2 = dot(d2, rn);
                const auto all_strictly_below =
                    (c0 < 0.0f) & (c0 * c0 > 1.0e-8f * dot(d0, d0)) &
                    (c1 < 0.0f) & (c1 * c1 > 1.0e-8f * dot(d1, d1)) &
                    (c2 < 0.0f) & (c2 * c2 > 1.0e-8f * dot(d2, d2));
                can_sample = can_sample &
                    !((*receiver_upper_hemisphere_only) & all_strictly_below);
            }

            TriangleLightSample light{
                .position = make_float3(0.0f),
                .barycentric = make_float2(0.0f),
                .direction = make_float3(0.0f, 0.0f, 1.0f),
                .distance = 0.0f,
                .conditional_pdf = 0.0f,
                .uses_solid_angle = false,
                .valid = false};
            $if(can_sample) {
                light = _sampling.from_position(
                    {.reference = std::move(reference),
                     .p0 = p0,
                     .p1 = p1,
                     .p2 = p2,
                     .random = std::move(random)});
            };
            const auto side_valid = sampled_side(
                sample_front, sample_back, geometric_normal, light.direction);
            const auto pdf = light.conditional_pdf * selection_pdf;
            const auto valid =
                can_sample & light.valid & side_valid & (pdf > 0.0f);
            Var<EmissiveTriangleGpu> emitter;
            if (!require_full_emitter &&
                scene->emissive_triangle_geometry_has_packed_meta) {
                emitter.emission_sampling = emission_sampling;
                emitter.emission_is_constant = (packed_meta >> 29u) & 1u;
                emitter.cycles_object_index = geom.p2_object.w;
                emitter.cycles_primitive_index = geom.normal_primitive.w;
                const auto shader_flags =
                    packed_meta & (~cycles_shader_identity::shader_mask);
                emitter.cycles_shader_flags = shader_flags;
                emitter.cycles_shader_id =
                    (packed_meta & emissive_geom_shader_mask) | shader_flags;
            } else {
                $if(valid) {
                    emitter =
                        scene->emissive_triangle_buffer->read(emitter_index);
                };
            }
            return {
                .geometry =
                    {.emitter = std::move(emitter),
                     .p0 = p0,
                     .p1 = p1,
                     .p2 = p2,
                     .geometric_normal = geometric_normal,
                     .area = area,
                     .sample_front = sample_front,
                     .sample_back = sample_back},
                .light = std::move(light),
                .pdf = pdf,
                .valid = valid};
        }
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

    Float3 evaluate_constant_emission(
        PathSampleContext &sample,
        const EmissiveTriangleLightProposal
            &proposal) const noexcept override {
        Float3 emission = make_float3(1.0f);
        static_cast<void>(cycles_svm_constant_emission(
            *sample.invocation.config.scene,
            proposal.geometry.emitter.cycles_shader_id, emission));
        return emission;
    }

    Float4 evaluate_constant_emission_spectral(
        PathSampleContext &sample,
        const EmissiveTriangleLightProposal
            &proposal) const noexcept override {
        const Float3 rgb = evaluate_constant_emission(sample, proposal);
        if (!sample.is_spectral()) {
            return make_float4(rgb, 0.0f);
        }
        const auto ctx = sample.spectral_context();
        return cycles_spectrum::rgb_to_spectrum_illuminant(
            *ctx->tables, rgb, ctx->wavelengths.lambdas);
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
