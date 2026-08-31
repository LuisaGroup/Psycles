#include <psycles/compiler/cycles_transform.h>

#include <limits>

namespace psycles::compiler {
namespace {

[[nodiscard]] Vec3f cross(Vec3f a, Vec3f b) noexcept {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

[[nodiscard]] float dot(Vec3f a, Vec3f b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3f divide(Vec3f value, float divisor) noexcept {
  const auto inverse = 1.0f / divisor;
  return {value.x * inverse, value.y * inverse, value.z * inverse};
}

} // namespace

Mat4f cycles_inverse_affine_transform(const Mat4f &transform) noexcept {
  const auto &e = transform.elements;
  auto x = Vec3f{e[0u], e[1u], e[2u]};
  auto y = Vec3f{e[4u], e[5u], e[6u]};
  auto z = Vec3f{e[8u], e[9u], e[10u]};
  const auto w = Vec3f{e[12u], e[13u], e[14u]};

  auto determinant = dot(x, cross(y, z));
  if (determinant == 0.0f) {
    x.x += 1.0e-8f;
    y.y += 1.0e-8f;
    z.z += 1.0e-8f;
    determinant = dot(x, cross(y, z));
    if (determinant == 0.0f) {
      determinant = std::numeric_limits<float>::max();
    }
  }

  const auto inverse_x = divide(cross(y, z), determinant);
  const auto inverse_y = divide(cross(z, x), determinant);
  const auto inverse_z = divide(cross(x, y), determinant);
  Mat4f result;
  result.elements = {
      inverse_x.x, inverse_y.x, inverse_z.x, 0.0f,
      inverse_x.y, inverse_y.y, inverse_z.y, 0.0f,
      inverse_x.z, inverse_y.z, inverse_z.z, 0.0f,
      -dot(inverse_x, w), -dot(inverse_y, w), -dot(inverse_z, w), 1.0f};
  return result;
}

} // namespace psycles::compiler
