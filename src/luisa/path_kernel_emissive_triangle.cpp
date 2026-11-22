#include "path_kernel_emissive_triangle.h"
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
