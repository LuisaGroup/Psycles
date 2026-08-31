/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include "surface_image_sampling.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

constexpr auto pi = 3.14159265358979323846f;
constexpr auto inverse_pi = 1.0f / pi;
constexpr auto inverse_two_pi = 0.5f / pi;

[[nodiscard]] Bool is_zero(Expr<luisa::float3> value) noexcept {
  return (value.x == 0.0f) & (value.y == 0.0f) & (value.z == 0.0f);
}

[[nodiscard]] Dual3 load_coordinate(Stack &stack,
                                    Expr<std::uint32_t> offset,
                                    bool use_derivatives) noexcept {
  Dual3 result{.val = stack_load_float3(stack, offset),
               .dx = make_float3(0.0f),
               .dy = make_float3(0.0f)};
  if (use_derivatives) {
    result.dx = stack_load_float3(stack, offset + 3u);
    result.dy = stack_load_float3(stack, offset + 6u);
  }
  return result;
}

[[nodiscard]] Dual1 component(const Dual3 &value,
                              std::uint32_t index) noexcept {
  if (index == 0u) {
    return {.val = value.val.x, .dx = value.dx.x, .dy = value.dy.x};
  }
  if (index == 1u) {
    return {.val = value.val.y, .dx = value.dx.y, .dy = value.dy.y};
  }
  return {.val = value.val.z, .dx = value.dx.z, .dy = value.dy.z};
}

[[nodiscard]] Dual1 add(const Dual1 &lhs, const Dual1 &rhs) noexcept {
  return {.val = lhs.val + rhs.val,
          .dx = lhs.dx + rhs.dx,
          .dy = lhs.dy + rhs.dy};
}

[[nodiscard]] Dual1 add(const Dual1 &value, float scalar) noexcept {
  return {.val = value.val + scalar, .dx = value.dx, .dy = value.dy};
}

[[nodiscard]] Dual1 subtract(const Dual1 &lhs,
                             const Dual1 &rhs) noexcept {
  return {.val = lhs.val - rhs.val,
          .dx = lhs.dx - rhs.dx,
          .dy = lhs.dy - rhs.dy};
}

[[nodiscard]] Dual1 scale(const Dual1 &value, float scalar) noexcept {
  return {.val = value.val * scalar,
          .dx = value.dx * scalar,
          .dy = value.dy * scalar};
}

[[nodiscard]] Dual1 multiply(const Dual1 &lhs,
                             const Dual1 &rhs) noexcept {
  return {.val = lhs.val * rhs.val,
          .dx = lhs.dx * rhs.val + lhs.val * rhs.dx,
          .dy = lhs.dy * rhs.val + lhs.val * rhs.dy};
}

[[nodiscard]] Dual1 divide(const Dual1 &lhs, const Dual1 &rhs) noexcept {
  const auto inverse = 1.0f / rhs.val;
  const auto value = lhs.val * inverse;
  return {.val = value,
          .dx = (lhs.dx - value * rhs.dx) * inverse,
          .dy = (lhs.dy - value * rhs.dy) * inverse};
}

[[nodiscard]] Dual1 inverse_sqrt(const Dual1 &value) noexcept {
  const auto result = rsqrt(value.val);
  // Cycles dual inversesqrt uses safe_divide for df/du. This is observable at
  // mirror-ball poles as a finite/invalid derivative distinction, not merely
  // a last-bit arithmetic choice.
  const auto derivative = select(
      0.0f, -0.5f * result / value.val, value.val != 0.0f);
  return {.val = result,
          .dx = derivative * value.dx,
          .dy = derivative * value.dy};
}

[[nodiscard]] Dual1 square_root(const Dual1 &value) noexcept {
  const auto result = sqrt(value.val);
  const auto derivative = 0.5f / result;
  return {.val = result,
          .dx = derivative * value.dx,
          .dy = derivative * value.dy};
}

[[nodiscard]] Dual1 length_squared(const Dual3 &value) noexcept {
  return {.val = dot(value.val, value.val),
          .dx = 2.0f * dot(value.val, value.dx),
          .dy = 2.0f * dot(value.val, value.dy)};
}

[[nodiscard]] Dual1 atan2_dual(const Dual1 &y, const Dual1 &x) noexcept {
  const auto denominator = x.val * x.val + y.val * y.val;
  const auto nonzero = denominator != 0.0f;
  const auto inverse = select(0.0f, 1.0f / denominator, nonzero);
  Float value = 0.0f;
  $if(nonzero) { value = atan2(y.val, x.val); };
  const auto derivative_x = -y.val * inverse;
  const auto derivative_y = x.val * inverse;
  return {.val = value,
          .dx = derivative_x * x.dx + derivative_y * y.dx,
          .dy = derivative_x * x.dy + derivative_y * y.dy};
}

[[nodiscard]] Dual1 acos_dual(const Dual1 &value, bool safe) noexcept {
  Float result = acos(safe ? clamp(value.val, -1.0f, 1.0f) : value.val);
  Float derivative = 0.0f;
  if (safe) {
    $if(abs(value.val) < 1.0f) {
      derivative = -rsqrt(1.0f - value.val * value.val);
    };
  } else {
    derivative = -rsqrt(1.0f - value.val * value.val);
  }
  return {.val = result,
          .dx = derivative * value.dx,
          .dy = derivative * value.dy};
}

[[nodiscard]] Dual1 select_dual(Bool condition,
                                const Dual1 &when_true,
                                const Dual1 &when_false) noexcept {
  return {.val = select(when_false.val, when_true.val, condition),
          .dx = select(when_false.dx, when_true.dx, condition),
          .dy = select(when_false.dy, when_true.dy, condition)};
}

[[nodiscard]] Dual2 make_dual2(const Dual1 &x, const Dual1 &y) noexcept {
  return {.val = make_float2(x.val, y.val),
          .dx = make_float2(x.dx, y.dx),
          .dy = make_float2(x.dy, y.dy)};
}

[[nodiscard]] Dual3 remap_square(const Dual3 &coordinate) noexcept {
  return {.val = (coordinate.val - 0.5f) * 2.0f,
          .dx = coordinate.dx * 2.0f,
          .dy = coordinate.dy * 2.0f};
}

[[nodiscard]] Float compatible_atan2(Float y, Float x) noexcept {
  Float result = 0.0f;
  $if((x != 0.0f) | (y != 0.0f)) { result = atan2(y, x); };
  return result;
}

[[nodiscard]] Float2 map_to_sphere(Float3 coordinate) noexcept {
  const auto l = dot(coordinate, coordinate);
  Float2 result = make_float2(0.0f);
  $if(l > 0.0f) {
    const auto u =
        0.5f - compatible_atan2(coordinate.x, coordinate.y) * inverse_two_pi;
    const auto v =
        1.0f - acos(clamp(coordinate.z * rsqrt(l), -1.0f, 1.0f)) * inverse_pi;
    result = make_float2(u, v);
  };
  return result;
}

[[nodiscard]] Dual2 map_to_sphere(const Dual3 &coordinate) noexcept {
  const auto l = length_squared(coordinate);
  Dual1 u{.val = 0.0f, .dx = 0.0f, .dy = 0.0f};
  Dual1 v{.val = 0.0f, .dx = 0.0f, .dy = 0.0f};
  $if(l.val > 0.0f) {
    const auto x = component(coordinate, 0u);
    const auto y = component(coordinate, 1u);
    const auto z = component(coordinate, 2u);
    u = add(scale(atan2_dual(x, y), -inverse_two_pi), 0.5f);
    v = add(scale(acos_dual(multiply(z, inverse_sqrt(l)), true),
                  -inverse_pi),
            1.0f);
  };
  return make_dual2(u, v);
}

[[nodiscard]] Float2 map_to_tube(Float3 coordinate) noexcept {
  const auto length =
      sqrt(coordinate.x * coordinate.x + coordinate.y * coordinate.y);
  Float2 result = make_float2(0.0f);
  $if(length > 0.0f) {
    result = make_float2(
        (1.0f - atan2(coordinate.x / length, coordinate.y / length) *
                    inverse_pi) *
            0.5f,
        (coordinate.z + 1.0f) * 0.5f);
  };
  return result;
}

[[nodiscard]] Dual2 map_to_tube(const Dual3 &coordinate) noexcept {
  const auto x = component(coordinate, 0u);
  const auto y = component(coordinate, 1u);
  const auto z = component(coordinate, 2u);
  const auto radial_squared = add(multiply(x, x), multiply(y, y));
  Dual1 u{.val = 0.0f, .dx = 0.0f, .dy = 0.0f};
  Dual1 v{.val = 0.0f, .dx = 0.0f, .dy = 0.0f};
  $if(radial_squared.val > 0.0f) {
    const auto radial = square_root(radial_squared);
    u = scale(
        add(scale(atan2_dual(divide(x, radial), divide(y, radial)),
                  -inverse_pi),
            1.0f),
        0.5f);
    v = scale(add(z, 1.0f), 0.5f);
  };
  return make_dual2(u, v);
}

[[nodiscard]] Float2 image_mapping(Float3 coordinate,
                                   Expr<std::uint32_t> projection) noexcept {
  Float2 result = coordinate.xy();
  const auto mapped = (coordinate - 0.5f) * 2.0f;
  $if(projection == static_cast<std::uint32_t>(NODE_IMAGE_PROJ_SPHERE)) {
    result = map_to_sphere(mapped);
  }
  $elif(projection == static_cast<std::uint32_t>(NODE_IMAGE_PROJ_TUBE)) {
    result = map_to_tube(mapped);
  };
  return result;
}

[[nodiscard]] Dual2 image_mapping(const Dual3 &coordinate,
                                  Expr<std::uint32_t> projection) noexcept {
  Dual2 result{.val = coordinate.val.xy(),
               .dx = coordinate.dx.xy(),
               .dy = coordinate.dy.xy()};
  const auto mapped = remap_square(coordinate);
  $if(projection == static_cast<std::uint32_t>(NODE_IMAGE_PROJ_SPHERE)) {
    result = map_to_sphere(mapped);
  }
  $elif(projection == static_cast<std::uint32_t>(NODE_IMAGE_PROJ_TUBE)) {
    result = map_to_tube(mapped);
  };
  return result;
}

[[nodiscard]] Float2 direction_to_equirectangular(Float3 direction) noexcept {
  Float2 result = make_float2(0.0f);
  $if(!is_zero(direction)) {
    const auto u =
        (compatible_atan2(direction.y, direction.x) - pi) / (-2.0f * pi);
    const auto v =
        (acos(direction.z / sqrt(dot(direction, direction))) - pi) / -pi;
    result = make_float2(u, v);
  };
  return result;
}

[[nodiscard]] Dual2 direction_to_equirectangular(
    const Dual3 &direction) noexcept {
  Dual1 u{.val = 0.0f, .dx = 0.0f, .dy = 0.0f};
  Dual1 v{.val = 0.0f, .dx = 0.0f, .dy = 0.0f};
  $if(!is_zero(direction.val)) {
    u = scale(add(atan2_dual(component(direction, 1u),
                             component(direction, 0u)),
                  -pi),
              -inverse_two_pi);
    v = scale(add(acos_dual(divide(component(direction, 2u),
                                   square_root(length_squared(direction))),
                            false),
                  -pi),
              -inverse_pi);
  };
  return make_dual2(u, v);
}

[[nodiscard]] Float2 direction_to_mirrorball(Float3 direction) noexcept {
  direction.y -= 1.0f;
  const auto divisor =
      2.0f * sqrt(max(-0.5f * direction.y, 0.0f));
  $if(divisor > 0.0f) { direction /= divisor; };
  return 0.5f * (direction.xz() + 1.0f);
}

[[nodiscard]] Dual2 direction_to_mirrorball(const Dual3 &input) noexcept {
  auto direction = input;
  direction.val.y -= 1.0f;
  const auto factor =
      scale(inverse_sqrt(scale(component(direction, 1u), -0.5f)), 0.5f);
  const auto x = add(multiply(component(direction, 0u), factor), 1.0f);
  const auto z = add(multiply(component(direction, 2u), factor), 1.0f);
  return make_dual2(scale(x, 0.5f), scale(z, 0.5f));
}

[[nodiscard]] Float4 sample_image(const KernelGlobals &kernel_globals,
                                  ShaderData &shader_data,
                                  Expr<std::int32_t> image_id,
                                  const Dual2 &uv,
                                  Expr<std::uint32_t> flags) noexcept {
  const auto sampled = kernel_globals.kernel_image_interp_with_udim(
      shader_data, image_id, uv);
  return ::psycles::luisa_backend::detail::decode_surface_image_sample(
      sampled,
      (flags & static_cast<std::uint32_t>(NODE_IMAGE_ALPHA_UNASSOCIATE)) != 0u,
      (flags & static_cast<std::uint32_t>(NODE_IMAGE_COMPRESS_AS_SRGB)) != 0u);
}

void store_image_sample(Stack &stack, Expr<std::uint32_t> color_offset,
                        Expr<std::uint32_t> alpha_offset,
                        Expr<luisa::float4> value) noexcept {
  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, color_offset, value.xyz());
  };
  $if(alpha_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, alpha_offset, value.w);
  };
}

} // namespace

void node_tex_image(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals,
                    ShaderData &shader_data,
                    bool use_derivatives) noexcept {
  const auto image_id = cursor.word().bitcast<std::int32_t>();
  const auto projection = cursor.word();
  const auto packed = cursor.word();
  const auto flags = cursor.byte(packed, 0u);
  const auto coordinate_offset = cursor.byte(packed, 1u);
  const auto color_offset = cursor.byte(packed, 2u);
  const auto alpha_offset = cursor.byte(packed, 3u);
  const auto coordinate =
      load_coordinate(stack, coordinate_offset, use_derivatives);
  Dual2 uv{.val = image_mapping(coordinate.val, projection),
           .dx = make_float2(0.0f),
           .dy = make_float2(0.0f)};
  if (use_derivatives) {
    uv = image_mapping(coordinate, projection);
  }
  store_image_sample(
      stack, color_offset, alpha_offset,
      sample_image(kernel_globals, shader_data, image_id, uv, flags));
}

void node_tex_image_box(Cursor &cursor, Stack &stack,
                        const KernelGlobals &kernel_globals,
                        const TransformState &transform_state,
                        ShaderData &shader_data,
                        bool use_derivatives,
                        bool object_motion_enabled) noexcept {
  const auto image_id = cursor.word().bitcast<std::int32_t>();
  const auto blend = cursor.floating();
  const auto packed = cursor.word();
  const auto flags = cursor.byte(packed, 0u);
  const auto coordinate_offset = cursor.byte(packed, 1u);
  const auto color_offset = cursor.byte(packed, 2u);
  const auto alpha_offset = cursor.byte(packed, 3u);

  Float3 normal = shader_data.N;
  object_inverse_normal_transform(normal, transform_state, shader_data,
                                  object_motion_enabled);
  const Float3 signed_normal = normal;
  normal = abs(normal);
  normal /= normal.x + normal.y + normal.z;

  Float3 weight = make_float3(0.0f);
  const Float limit = 0.5f * (1.0f + blend);
  $if((normal.x > limit * (normal.x + normal.y)) &
      (normal.x > limit * (normal.x + normal.z))) {
    weight.x = 1.0f;
  }
  $elif((normal.y > limit * (normal.x + normal.y)) &
        (normal.y > limit * (normal.y + normal.z))) {
    weight.y = 1.0f;
  }
  $elif((normal.z > limit * (normal.x + normal.z)) &
        (normal.z > limit * (normal.y + normal.z))) {
    weight.z = 1.0f;
  }
  $elif(blend > 0.0f) {
    $if(normal.z < (1.0f - limit) * (normal.y + normal.x)) {
      weight.x = normal.x / (normal.x + normal.y);
      weight.x = clamp(
          (weight.x - 0.5f * (1.0f - blend)) / blend, 0.0f, 1.0f);
      weight.y = 1.0f - weight.x;
    }
    $elif(normal.x < (1.0f - limit) * (normal.y + normal.z)) {
      weight.y = normal.y / (normal.y + normal.z);
      weight.y = clamp(
          (weight.y - 0.5f * (1.0f - blend)) / blend, 0.0f, 1.0f);
      weight.z = 1.0f - weight.y;
    }
    $elif(normal.y < (1.0f - limit) * (normal.x + normal.z)) {
      weight.x = normal.x / (normal.x + normal.z);
      weight.x = clamp(
          (weight.x - 0.5f * (1.0f - blend)) / blend, 0.0f, 1.0f);
      weight.z = 1.0f - weight.x;
    }
    $else {
      weight.x = ((2.0f - limit) * normal.x + (limit - 1.0f)) /
                 (2.0f * limit - 1.0f);
      weight.y = ((2.0f - limit) * normal.y + (limit - 1.0f)) /
                 (2.0f * limit - 1.0f);
      weight.z = ((2.0f - limit) * normal.z + (limit - 1.0f)) /
                 (2.0f * limit - 1.0f);
    };
  }
  $else { weight.x = 1.0f; };

  const auto coordinate =
      load_coordinate(stack, coordinate_offset, use_derivatives);
  const auto x = component(coordinate, 0u);
  const auto y = component(coordinate, 1u);
  const auto z = component(coordinate, 2u);
  const Dual1 one{.val = 1.0f, .dx = 0.0f, .dy = 0.0f};
  Float4 sampled = make_float4(0.0f);
  $if(weight.x > 0.0f) {
    const auto u = select_dual(signed_normal.x < 0.0f,
                               subtract(one, y), y);
    sampled += weight.x * sample_image(
                              kernel_globals, shader_data, image_id,
                              make_dual2(u, z), flags);
  };
  $if(weight.y > 0.0f) {
    const auto u = select_dual(signed_normal.y > 0.0f,
                               subtract(one, x), x);
    sampled += weight.y * sample_image(
                              kernel_globals, shader_data, image_id,
                              make_dual2(u, z), flags);
  };
  $if(weight.z > 0.0f) {
    const auto u = select_dual(signed_normal.z > 0.0f,
                               subtract(one, y), y);
    sampled += weight.z * sample_image(
                              kernel_globals, shader_data, image_id,
                              make_dual2(u, x), flags);
  };
  store_image_sample(stack, color_offset, alpha_offset, sampled);
}

void node_tex_environment(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          ShaderData &shader_data,
                          bool use_derivatives) noexcept {
  const auto image_id = cursor.word().bitcast<std::int32_t>();
  const auto projection = cursor.word();
  const auto packed = cursor.word();
  const auto flags = cursor.byte(packed, 0u);
  const auto coordinate_offset = cursor.byte(packed, 1u);
  const auto color_offset = cursor.byte(packed, 2u);
  const auto alpha_offset = cursor.byte(packed, 3u);
  const auto coordinate =
      load_coordinate(stack, coordinate_offset, use_derivatives);

  Dual2 uv;
  if (use_derivatives) {
    const auto direction = safe_normalize_dual(coordinate);
    uv = direction_to_equirectangular(direction);
    $if(projection ==
        static_cast<std::uint32_t>(NODE_ENVIRONMENT_MIRROR_BALL)) {
      uv = direction_to_mirrorball(direction);
    };
  } else {
    const auto direction = safe_normalize_cycles(coordinate.val);
    uv = {.val = direction_to_equirectangular(direction),
          .dx = make_float2(0.0f),
          .dy = make_float2(0.0f)};
    $if(projection ==
        static_cast<std::uint32_t>(NODE_ENVIRONMENT_MIRROR_BALL)) {
      uv.val = direction_to_mirrorball(direction);
    };
  }
  store_image_sample(
      stack, color_offset, alpha_offset,
      sample_image(kernel_globals, shader_data, image_id, uv, flags));
}

} // namespace psycles::luisa_backend::cycles_svm::detail
