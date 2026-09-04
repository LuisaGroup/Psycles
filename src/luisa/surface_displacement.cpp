#include "surface_displacement.h"

#include "surface_math.h"

#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float
displacement_amount(const SurfaceDisplacementInput &input) noexcept {
  return (input.height - input.midlevel) * input.scale;
}

struct ObjectDirectionFrame {
  Float3 cofactor_x;
  Float3 cofactor_y;
  Float3 cofactor_z;
  Float inverse_determinant;
  Bool valid;
};

[[nodiscard]] ObjectDirectionFrame object_direction_frame(
    Float3 column_x, Float3 column_y, Float3 column_z) noexcept {
  const auto cofactor_x = cross(column_y, column_z);
  const auto cofactor_y = cross(column_z, column_x);
  const auto cofactor_z = cross(column_x, column_y);
  const auto determinant = dot(column_x, cofactor_x);
  const auto valid = determinant != 0.0f;
  return {.cofactor_x = cofactor_x,
          .cofactor_y = cofactor_y,
          .cofactor_z = cofactor_z,
          .inverse_determinant =
              1.0f / select(1.0f, determinant, valid),
          .valid = valid};
}

[[nodiscard]] Float3 object_inverse_normal(
    const ObjectDirectionFrame &frame, Float3 normal) noexcept {
  return native_vector_math::safe_normalize_nonzero(
      make_float3(dot(frame.cofactor_x, normal),
                  dot(frame.cofactor_y, normal),
                  dot(frame.cofactor_z, normal)) *
          frame.inverse_determinant);
}

[[nodiscard]] Float3 object_direction_to_world(
    const ObjectDirectionFrame &frame, Float3 direction) noexcept {
  const auto transformed =
      (frame.cofactor_x * direction.x +
       frame.cofactor_y * direction.y +
       frame.cofactor_z * direction.z) *
      frame.inverse_determinant;
  return select(make_float3(0.0f), transformed, frame.valid);
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
  const auto frame = object_direction_frame(
      input.normal_to_world_x, input.normal_to_world_y,
      input.normal_to_world_z);
  const auto object_normal = object_inverse_normal(frame, input.normal);
  return object_direction_to_world(frame, object_normal) *
         displacement_amount(input);
}

SurfaceVectorDisplacementTangent
vector_displacement_default_tangent(Float3 object_tangent, Float tangent_sign,
                                    Bool is_curve,
                                    UInt geometry_index) noexcept {
  const auto attribute_found =
      !is_curve & (geometry_index != ~static_cast<std::uint32_t>(0u));
  return {.object_tangent = object_tangent,
          .tangent_sign = tangent_sign,
          .tangent_attribute_found = attribute_found,
          .tangent_sign_found = attribute_found};
}

Float3 vector_displacement_inline(
    const SurfaceVectorDisplacementInput &input, UInt space) noexcept {
  const auto frame = object_direction_frame(
      input.normal_to_world_x, input.normal_to_world_y,
      input.normal_to_world_z);
  Float3 displacement =
      (input.vector - make_float3(input.midlevel)) * input.scale;
  $if(space == static_cast<std::uint32_t>(
                    compiler::VectorDisplacementSpace::tangent)) {
    const auto normal = object_inverse_normal(frame, input.shading_normal);
    auto tangent = native_vector_math::normalize_unchecked(input.dpdu);
    tangent = select(tangent, input.object_tangent,
                     input.tangent_attribute_found);
    auto bitangent = native_vector_math::safe_normalize_nonzero(
        cross(normal, tangent));
    bitangent *= select(1.0f, input.tangent_sign,
                        input.tangent_sign_found);
    displacement = tangent * displacement.x + normal * displacement.y +
                   bitangent * displacement.z;
  };
  $if(space != static_cast<std::uint32_t>(
                    compiler::VectorDisplacementSpace::world)) {
    displacement = object_direction_to_world(frame, displacement);
  };
  return displacement;
}

} // namespace psycles::luisa_backend::detail
