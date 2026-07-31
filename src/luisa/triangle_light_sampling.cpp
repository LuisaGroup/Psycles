#include <psycles/luisa/triangle_light_sampling.h>

#include <psycles/luisa/spherical_geometry.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

namespace {

[[nodiscard]] TriangleLightSample area_sample(
    const TriangleLightSampleInput &input) noexcept {
    const auto barycentric =
        spherical_geometry::low_distortion_triangle_barycentric(
            input.random);
    const auto position =
        (1.0f - barycentric.x - barycentric.y) * input.p0 +
        barycentric.x * input.p1 +
        barycentric.y * input.p2;
    const auto offset = position - input.reference;
    const auto distance_squared = dot(offset, offset);
    const auto distance =
        sqrt(max(distance_squared, 1.0e-20f));
    const auto direction = offset / distance;
    const auto unnormalized_normal =
        cross(input.p1 - input.p0, input.p2 - input.p0);
    const auto doubled_area =
        sqrt(max(dot(unnormalized_normal, unnormalized_normal), 0.0f));
    const auto area = 0.5f * doubled_area;
    const auto normal =
        unnormalized_normal / max(doubled_area, 1.0e-20f);
    const auto cosine = abs(dot(normal, -direction));
    const auto conditional_pdf =
        distance_squared / max(cosine * area, 1.0e-20f);
    const auto valid =
        (area > 0.0f) &
        (distance_squared > 1.0e-12f) &
        (cosine > 0.0f);
    return {
        .position = position,
        .barycentric = barycentric,
        .direction = direction,
        .distance = distance,
        .conditional_pdf = conditional_pdf,
        .uses_solid_angle = false,
        .valid = valid};
}

}// namespace

TriangleLightSample
TriangleLightSampling::from_segment(
    const TriangleLightSampleInput &input) const noexcept {
    return area_sample(input);
}

TriangleLightSample
TriangleLightSampling::from_position(
    const TriangleLightSampleInput &input) const noexcept {
    const auto sample =
        spherical_geometry::sample_triangle(
            input.reference,
            input.p0,
            input.p1,
            input.p2,
            input.random);
    return {
        .position = sample.position,
        .barycentric = sample.barycentric,
        .direction = sample.direction,
        .distance = sample.distance,
        .conditional_pdf = sample.conditional_pdf,
        .uses_solid_angle = sample.uses_solid_angle,
        .valid = sample.valid};
}

TriangleLightPdf
TriangleLightSampling::from_intersection(
    const TriangleLightSampleInput &input,
    Float3 light_position) const noexcept {
    const auto pdf =
        spherical_geometry::triangle_directional_pdf(
            input.reference,
            light_position,
            input.p0,
            input.p1,
            input.p2);
    return {
        .value = pdf.value,
        .uses_solid_angle = pdf.uses_solid_angle,
        .valid = pdf.valid};
}

}// namespace psycles::luisa_backend
