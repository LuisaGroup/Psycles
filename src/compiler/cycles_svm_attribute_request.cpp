#include <psycles/compiler/cycles_svm_attribute_request.h>

#include <algorithm>
#include <ranges>

namespace psycles::compiler::cycles_svm {

std::string_view attribute_standard_name(AttributeStandard standard) noexcept {
  switch (standard) {
  case ATTR_STD_POSITION:
    return "P";
  case ATTR_STD_RADIUS:
    return "radius";
  case ATTR_STD_VERTEX_NORMAL:
  case ATTR_STD_CORNER_NORMAL:
    return "N";
  case ATTR_STD_UV:
    return "uv";
  case ATTR_STD_GENERATED:
    return "generated";
  case ATTR_STD_GENERATED_TRANSFORM:
    return "generated_transform";
  case ATTR_STD_UV_TANGENT:
    return "tangent";
  case ATTR_STD_UV_TANGENT_SIGN:
    return "tangent_sign";
  case ATTR_STD_UV_TANGENT_UNDISPLACED:
    return "undisplaced_tangent";
  case ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED:
    return "undisplaced_tangent_sign";
  case ATTR_STD_VERTEX_COLOR:
    return "vertex_color";
  case ATTR_STD_POSITION_UNDEFORMED:
    return "undeformed";
  case ATTR_STD_POSITION_UNDISPLACED:
    return "undisplaced";
  case ATTR_STD_NORMAL_UNDISPLACED:
    return "undisplaced_N";
  case ATTR_STD_PARTICLE:
    return "particle";
  case ATTR_STD_CURVE_INTERCEPT:
    return "curve_intercept";
  case ATTR_STD_CURVE_LENGTH:
    return "curve_length";
  case ATTR_STD_CURVE_RANDOM:
    return "curve_random";
  case ATTR_STD_POINT_RANDOM:
    return "point_random";
  case ATTR_STD_PTEX_FACE_ID:
    return "ptex_face_id";
  case ATTR_STD_PTEX_UV:
    return "ptex_uv";
  case ATTR_STD_VOLUME_DENSITY:
    return "density";
  case ATTR_STD_VOLUME_COLOR:
    return "color";
  case ATTR_STD_VOLUME_FLAME:
    return "flame";
  case ATTR_STD_VOLUME_HEAT:
    return "heat";
  case ATTR_STD_VOLUME_TEMPERATURE:
    return "temperature";
  case ATTR_STD_VOLUME_VELOCITY:
    return "velocity";
  case ATTR_STD_VOLUME_VELOCITY_X:
    return "velocity_x";
  case ATTR_STD_VOLUME_VELOCITY_Y:
    return "velocity_y";
  case ATTR_STD_VOLUME_VELOCITY_Z:
    return "velocity_z";
  case ATTR_STD_POINTINESS:
    return "pointiness";
  case ATTR_STD_RANDOM_PER_ISLAND:
    return "random_per_island";
  case ATTR_STD_SHADOW_TRANSPARENCY:
    return "shadow_transparency";
  case ATTR_STD_NOT_FOUND:
  case ATTR_STD_NONE:
  case ATTR_STD_NUM:
    return {};
  }
  return {};
}

AttributeStandard attribute_standard_from_name(std::string_view name) noexcept {
  for (auto standard = static_cast<int>(ATTR_STD_NONE);
       standard < static_cast<int>(ATTR_STD_NUM); ++standard) {
    const auto value = static_cast<AttributeStandard>(standard);
    if (name == attribute_standard_name(value)) {
      return value;
    }
  }
  return ATTR_STD_NONE;
}

void AttributeRequestSet::add(std::string_view name) {
  if (std::ranges::any_of(_requests, [name](const AttributeRequest &request) {
        return request.name == name;
      })) {
    return;
  }
  _requests.emplace_back(
      AttributeRequest{.standard = ATTR_STD_NONE, .name = std::string{name}});
}

void AttributeRequestSet::add(AttributeStandard standard) {
  if (std::ranges::any_of(_requests,
                          [standard](const AttributeRequest &request) {
                            return request.standard == standard;
                          })) {
    return;
  }
  _requests.emplace_back(AttributeRequest{.standard = standard, .name = {}});
}

void AttributeRequestSet::add_standard(std::string_view name) {
  if (name.empty()) {
    return;
  }
  const auto standard = attribute_standard_from_name(name);
  if (standard != ATTR_STD_NONE) {
    add(standard);
  } else {
    add(name);
  }
}

std::vector<AttributeRequest> AttributeRequestSet::canonical_requests() const {
  auto result = _requests;
  std::ranges::sort(result);
  return result;
}

} // namespace psycles::compiler::cycles_svm
