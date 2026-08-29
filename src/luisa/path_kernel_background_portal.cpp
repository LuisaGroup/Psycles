#include "path_kernel_background_portal.h"

#include <psycles/luisa/analytic_light_sampling.h>
#include <psycles/luisa/area_light_sampling.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

namespace sampling = analytic_light_sampling;

[[nodiscard]] Bool possible_from(Var<LightGpu> light,
                                 Float3 reference) noexcept {
  const auto portal_normal = -light.axis_z;
  return dot(portal_normal, reference - light.position) > 1.0e-4f;
}

[[nodiscard]] AreaLightSampleInput portal_sample_input(Var<LightGpu> light,
                                                       Float3 reference,
                                                       Float2 random) noexcept {
  return {.reference = std::move(reference),
          .center = light.position,
          .axis_u = light.axis_x,
          .axis_v = light.axis_y,
          .axis_z = light.axis_z,
          .length_u = light.size_u,
          .length_v = light.size_v,
          .spread = sampling::pi,
          .ellipse = (light.flags & light_flag_ellipse) != 0u,
          .full_spread = true,
          .random = std::move(random),
          // Cycles forces portal inverse area independently of lamp power
          // normalization. This flag affects only evaluation, while retaining
          // the same positive-validity contract as the source implementation.
          .normalize_power = true};
}

[[nodiscard]] Float portal_direction_pdf(Var<LightGpu> light, Float3 reference,
                                         Float3 direction) noexcept {
  const auto normal = -light.axis_z;
  const auto direction_normal = dot(direction, normal);
  const auto distance = sampling::safe_divide(
      dot(light.position - reference, normal), direction_normal);
  const auto position = reference + distance * direction;
  const auto inplane = position - light.position;
  const auto u =
      sampling::safe_divide(dot(inplane, light.axis_x), light.size_u);
  const auto v =
      sampling::safe_divide(dot(inplane, light.axis_y), light.size_v);
  const auto ellipse = (light.flags & light_flag_ellipse) != 0u;
  const auto inside = (u >= -0.5f) & (u <= 0.5f) & (v >= -0.5f) & (v <= 0.5f) &
                      (!ellipse | (u * u + v * v <= 0.25f));
  const auto valid = possible_from(light, reference) &
                     (direction_normal < 0.0f) & (distance > 1.0e-4f) & inside;

  Float result = sampling::rectangle_solid_angle_pdf(
      reference, light.position, light.axis_x, light.size_u, light.axis_y,
      light.size_v);
  $if(ellipse) {
    // Match Cycles background_portal_pdf: round portals evaluate the
    // area-to-solid-angle Jacobian at the portal center.
    const auto center_offset = light.position - reference;
    const auto center_distance_squared = dot(center_offset, center_offset);
    const auto center_distance = sqrt(max(center_distance_squared, 1.0e-30f));
    const auto center_direction = center_offset / center_distance;
    const auto cosine = dot(normal, -center_direction);
    const auto inverse_area =
        sampling::safe_divide(4.0f, sampling::pi * light.size_u * light.size_v);
    result =
        sampling::safe_divide(inverse_area * center_distance_squared, cosine);
  };
  return select(0.0f, result, valid);
}

struct PortalPdfResult {
  Float pdf;
  UInt possible_count;
};

[[nodiscard]] PortalPdfResult
portal_pdf_ignoring(const Buffer<LightGpu> &lights, UInt portal_offset,
                    UInt portal_count, Float3 reference, Float3 direction,
                    Int ignored_portal) noexcept {
  UInt possible_count = 0u;
  Float sum = 0.0f;
  $for(portal_index, portal_count) {
    const auto light = lights->read(portal_offset + portal_index);
    const auto possible = possible_from(light, reference);
    possible_count += cast<std::uint32_t>(possible);
    const auto ignored = (ignored_portal >= 0) &
                         (cast<std::int32_t>(portal_index) == ignored_portal);
    $if(possible & !ignored) {
      sum += portal_direction_pdf(light, reference, direction);
    };
  };
  return {.pdf = select(0.0f, sum / max(cast<float>(possible_count), 1.0f),
                        possible_count > 0u),
          .possible_count = possible_count};
}

} // namespace

UInt BackgroundPortalSampling::count_possible(const Buffer<LightGpu> &lights,
                                              UInt portal_offset,
                                              UInt portal_count,
                                              Float3 reference) const noexcept {
  UInt result = 0u;
  $for(portal_index, portal_count) {
    const auto light = lights->read(portal_offset + portal_index);
    result += cast<std::uint32_t>(possible_from(light, reference));
  };
  return result;
}

BackgroundPortalSample BackgroundPortalSampling::sample(
    const Buffer<LightGpu> &lights, UInt portal_offset, UInt portal_count,
    Float3 reference, Float2 random) const noexcept {
  const auto possible_count =
      count_possible(lights, portal_offset, portal_count, reference);
  const auto scaled = random.y * cast<float>(possible_count);
  UInt desired = min(cast<std::uint32_t>(scaled),
                     select(0u, possible_count - 1u, possible_count > 0u));
  const auto portal_random = make_float2(random.x, scaled - floor(scaled));

  Float3 direction = make_float3(0.0f);
  Float selected_pdf = 0.0f;
  Int selected_portal = -1;
  Bool selected = false;
  Bool valid = false;
  const AreaLightSampling area_sampling;
  $for(portal_index, portal_count) {
    const auto light = lights->read(portal_offset + portal_index);
    const auto possible = possible_from(light, reference);
    $if(possible & !selected) {
      $if(desired == 0u) {
        const auto sampled = area_sampling.from_position(
            portal_sample_input(light, reference, portal_random));
        direction = sampled.direction;
        selected_pdf =
            sampled.conditional_pdf / max(cast<float>(possible_count), 1.0f);
        selected_portal = cast<std::int32_t>(portal_index);
        selected = true;
        valid = sampled.valid;
      }
      $else { desired -= 1u; };
    };
  };
  const auto other = portal_pdf_ignoring(lights, portal_offset, portal_count,
                                         reference, direction, selected_portal);
  const auto result_pdf = selected_pdf + other.pdf;
  valid &= (possible_count > 0u) & (result_pdf > 0.0f);
  return {.direction = direction,
          .pdf = select(0.0f, result_pdf, valid),
          .valid = valid};
}

Float BackgroundPortalSampling::pdf(const Buffer<LightGpu> &lights,
                                    UInt portal_offset, UInt portal_count,
                                    Float3 reference,
                                    Float3 direction) const noexcept {
  return evaluate_pdf(lights, portal_offset, portal_count, std::move(reference),
                      std::move(direction))
      .pdf;
}

BackgroundPortalPdf BackgroundPortalSampling::evaluate_pdf(
    const Buffer<LightGpu> &lights, UInt portal_offset, UInt portal_count,
    Float3 reference, Float3 direction) const noexcept {
  const auto result =
      portal_pdf_ignoring(lights, portal_offset, portal_count,
                          std::move(reference), std::move(direction), -1);
  return {.pdf = result.pdf, .possible = result.possible_count > 0u};
}

} // namespace psycles::luisa_backend::detail
