#include "surface_displacement.h"

#include "surface_math.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float
displacement_amount(const SurfaceDisplacementInput &input) noexcept {
  return (input.height - input.midlevel) * input.scale;
}

} // namespace

Float3
displacement_world_inline(const SurfaceDisplacementInput &input) noexcept {
  return input.normal * displacement_amount(input);
}

Float3
displacement_object_inline(const SurfaceDisplacementInput &input) noexcept {
  // SurfacePoint stores A = (M^-1)^T by columns, where M maps object
  // directions to world directions. Therefore:
  //
  //   object normal = normalize(A^-1 * world normal)
  //   world offset  = A^-T * object normal * amount
  //
  // The three cofactors below are simultaneously the rows of A^-1 and the
  // columns of A^-T. This realizes Cycles' object_inverse_normal_transform
  // followed by object_dir_transform without adding another transform to
  // SurfacePoint or reconstructing it from unrelated geometry values.
  const auto column_x = input.normal_to_world_x;
  const auto column_y = input.normal_to_world_y;
  const auto column_z = input.normal_to_world_z;
  const auto cofactor_x = cross(column_y, column_z);
  const auto cofactor_y = cross(column_z, column_x);
  const auto cofactor_z = cross(column_x, column_y);
  const auto determinant = dot(column_x, cofactor_x);
  const auto transform_valid = abs(determinant) > 1.0e-20f;
  const auto inverse_determinant =
      1.0f / select(1.0f, determinant, transform_valid);
  const auto object_normal = safe_normalize(
      make_float3(dot(cofactor_x, input.normal), dot(cofactor_y, input.normal),
                  dot(cofactor_z, input.normal)) *
          inverse_determinant,
      make_float3(0.0f));
  const auto world_direction =
      (cofactor_x * object_normal.x + cofactor_y * object_normal.y +
       cofactor_z * object_normal.z) *
      inverse_determinant;
  return select(make_float3(0.0f), world_direction * displacement_amount(input),
                transform_valid);
}

} // namespace psycles::luisa_backend::detail
