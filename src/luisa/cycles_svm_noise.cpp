/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_noise.h>

#include <type_traits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

static_assert(static_cast<std::uint32_t>(NODE_NOISE_MULTIFRACTAL) ==
              static_cast<std::uint32_t>(cycles_noise::Type::multifractal));
static_assert(static_cast<std::uint32_t>(NODE_NOISE_FBM) ==
              static_cast<std::uint32_t>(cycles_noise::Type::fbm));
static_assert(
    static_cast<std::uint32_t>(NODE_NOISE_HYBRID_MULTIFRACTAL) ==
    static_cast<std::uint32_t>(cycles_noise::Type::hybrid_multifractal));
static_assert(
    static_cast<std::uint32_t>(NODE_NOISE_RIDGED_MULTIFRACTAL) ==
    static_cast<std::uint32_t>(cycles_noise::Type::ridged_multifractal));
static_assert(static_cast<std::uint32_t>(NODE_NOISE_HETERO_TERRAIN) ==
              static_cast<std::uint32_t>(cycles_noise::Type::hetero_terrain));

template<typename Coordinate>
[[nodiscard]] const auto &noise_select_callable() noexcept {
  static Callable callable{
      [](Coordinate coordinate, Float detail, Float roughness,
         Float lacunarity, Float offset, Float gain, UInt type,
         Bool normalize) noexcept {
        Float result = 0.0f;
        $switch (type) {
          $case(static_cast<std::uint32_t>(NODE_NOISE_MULTIFRACTAL)) {
            result = cycles_noise::multifractal(
                coordinate, detail, roughness, lacunarity);
          };
          $case(static_cast<std::uint32_t>(NODE_NOISE_FBM)) {
            result = cycles_noise::fbm(
                coordinate, detail, roughness, lacunarity, normalize);
          };
          $case(static_cast<std::uint32_t>(
              NODE_NOISE_HYBRID_MULTIFRACTAL)) {
            result = cycles_noise::hybrid_multifractal(
                coordinate, detail, roughness, lacunarity, offset, gain);
          };
          $case(static_cast<std::uint32_t>(
              NODE_NOISE_RIDGED_MULTIFRACTAL)) {
            result = cycles_noise::ridged_multifractal(
                coordinate, detail, roughness, lacunarity, offset, gain);
          };
          $case(static_cast<std::uint32_t>(NODE_NOISE_HETERO_TERRAIN)) {
            result = cycles_noise::hetero_terrain(
                coordinate, detail, roughness, lacunarity, offset);
          };
          $default {
            luisa::compute::dsl::unreachable(
                "invalid Cycles SVM Noise type");
          };
        };
        return result;
      }};
  return callable;
}

template<typename Coordinate>
[[nodiscard]] auto random_offset(Float seed) noexcept {
  if constexpr (std::is_same_v<Coordinate, Float>) {
    return cycles_noise::random_offset(seed);
  } else if constexpr (std::is_same_v<Coordinate, Float2>) {
    return cycles_noise::random_offset2(seed);
  } else if constexpr (std::is_same_v<Coordinate, Float3>) {
    return cycles_noise::random_offset3(seed);
  } else {
    return cycles_noise::random_offset4(seed);
  }
}

template<typename Coordinate>
[[nodiscard]] constexpr float first_color_seed() noexcept {
  if constexpr (std::is_same_v<Coordinate, Float>) {
    return 1.0f;
  } else if constexpr (std::is_same_v<Coordinate, Float2>) {
    return 2.0f;
  } else if constexpr (std::is_same_v<Coordinate, Float3>) {
    return 3.0f;
  } else {
    return 4.0f;
  }
}

template<typename Coordinate>
[[nodiscard]] const auto &noise_texture_callable() noexcept {
  static Callable callable{
      [](Coordinate coordinate, Float detail, Float roughness,
         Float lacunarity, Float offset, Float gain, Float distortion,
         UInt type, Bool normalize, Bool color_needed) noexcept {
        $if (distortion != 0.0f) {
          if constexpr (std::is_same_v<Coordinate, Float>) {
            coordinate +=
                cycles_noise::signed_noise(
                    coordinate + random_offset<Coordinate>(0.0f)) *
                distortion;
          } else if constexpr (std::is_same_v<Coordinate, Float2>) {
            coordinate +=
                make_float2(
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(0.0f)),
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(1.0f))) *
                distortion;
          } else if constexpr (std::is_same_v<Coordinate, Float3>) {
            coordinate +=
                make_float3(
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(0.0f)),
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(1.0f)),
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(2.0f))) *
                distortion;
          } else {
            coordinate +=
                make_float4(
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(0.0f)),
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(1.0f)),
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(2.0f)),
                    cycles_noise::signed_noise(
                        coordinate + random_offset<Coordinate>(3.0f))) *
                distortion;
          }
        };

        const auto &select_noise = noise_select_callable<Coordinate>();
        const Float value = select_noise(
            coordinate, detail, roughness, lacunarity, offset, gain, type,
            normalize);
        Float3 color = make_float3(value);
        $if (color_needed) {
          constexpr auto seed = first_color_seed<Coordinate>();
          color.y = select_noise(
              coordinate + random_offset<Coordinate>(seed), detail,
              roughness, lacunarity, offset, gain, type, normalize);
          color.z = select_noise(
              coordinate + random_offset<Coordinate>(seed + 1.0f), detail,
              roughness, lacunarity, offset, gain, type, normalize);
        };
        return make_float4(color, value);
      }};
  return callable;
}

[[nodiscard]] Float4 evaluate_noise(
    Expr<std::uint32_t> dimensions, Float3 vector, Float w, Float detail,
    Float roughness, Float lacunarity, Float offset, Float gain,
    Float distortion, Expr<std::uint32_t> type, Bool normalize,
    Bool color_needed) noexcept {
  Float4 result = make_float4(0.0f);
  $switch (dimensions) {
    $case(1u) {
      result = noise_texture_callable<Float>()(
          w, detail, roughness, lacunarity, offset, gain, distortion, type,
          normalize, color_needed);
    };
    $case(2u) {
      result = noise_texture_callable<Float2>()(
          vector.xy(), detail, roughness, lacunarity, offset, gain,
          distortion, type, normalize, color_needed);
    };
    $case(3u) {
      result = noise_texture_callable<Float3>()(
          vector, detail, roughness, lacunarity, offset, gain, distortion,
          type, normalize, color_needed);
    };
    $case(4u) {
      result = noise_texture_callable<Float4>()(
          make_float4(vector, w), detail, roughness, lacunarity, offset, gain,
          distortion, type, normalize, color_needed);
    };
    $default {
      luisa::compute::dsl::unreachable(
          "invalid Cycles SVM Noise dimensions");
    };
  };
  return result;
}

} // namespace

void node_tex_noise(Cursor &cursor, Stack &stack) noexcept {
  const auto dimensions = cursor.word();
  const auto noise_type = cursor.word();
  const auto normalize = cursor.word() != 0u;
  const auto w_bits = cursor.word();
  const auto scale_bits = cursor.word();
  const auto detail_bits = cursor.word();
  const auto roughness_bits = cursor.word();
  const auto lacunarity_bits = cursor.word();
  const auto offset_bits = cursor.word();
  const auto gain_bits = cursor.word();
  const auto distortion_bits = cursor.word();
  const auto packed = cursor.word();
  const auto vector_offset = cursor.byte(packed, 0u);
  const auto value_offset = cursor.byte(packed, 1u);
  const auto color_offset = cursor.byte(packed, 2u);

  auto vector = stack_load_float3(stack, vector_offset);
  auto w = stack_load_input_float(stack, w_bits);
  const auto scale = stack_load_input_float(stack, scale_bits);
  auto detail = stack_load_input_float(stack, detail_bits);
  auto roughness = stack_load_input_float(stack, roughness_bits);
  const auto lacunarity = stack_load_input_float(stack, lacunarity_bits);
  const auto offset = stack_load_input_float(stack, offset_bits);
  const auto gain = stack_load_input_float(stack, gain_bits);
  const auto distortion = stack_load_input_float(stack, distortion_bits);

  detail = clamp(detail, 0.0f, 15.0f);
  roughness = max(roughness, 0.0f);
  vector *= scale;
  w *= scale;

  const Bool color_needed =
      color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID);
  const auto result = evaluate_noise(
      dimensions, vector, w, detail, roughness, lacunarity, offset, gain,
      distortion, noise_type, normalize, color_needed);
  $if (value_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, value_offset, result.w);
  };
  $if (color_needed) {
    stack_store_float3(stack, color_offset, result.xyz());
  };
}

void node_tex_white_noise(Cursor &cursor, Stack &stack) noexcept {
  const auto dimensions = cursor.word();
  const auto vector_x = cursor.word();
  const auto vector_y = cursor.word();
  const auto vector_z = cursor.word();
  const auto w_bits = cursor.word();
  const auto packed_outputs = cursor.word();
  const auto value_offset = cursor.byte(packed_outputs, 0u);
  const auto color_offset = cursor.byte(packed_outputs, 1u);
  const auto vector =
      stack_load_input_float3(stack, vector_x, vector_y, vector_z);
  const auto w = stack_load_input_float(stack, w_bits);

  $if (color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    Float3 color = make_float3(1.0f, 0.0f, 1.0f);
    $switch (dimensions) {
      $case(1u) { color = cycles_noise::hash_float_to_color(w); };
      $case(2u) {
        color = cycles_noise::hash_float2_to_color(vector.xy());
      };
      $case(3u) { color = cycles_noise::hash_float3_to_color(vector); };
      $case(4u) {
        color = cycles_noise::hash_float4_to_color(make_float4(vector, w));
      };
      // Cycles keeps the magenta initializer after its debug kernel_assert.
      $default {};
    };
    stack_store_float3(stack, color_offset, color);
  };

  $if (value_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    Float value = 0.0f;
    $switch (dimensions) {
      $case(1u) { value = cycles_noise::hash_float(w); };
      $case(2u) { value = cycles_noise::hash_float2(vector.xy()); };
      $case(3u) { value = cycles_noise::hash_float3(vector); };
      $case(4u) {
        value = cycles_noise::hash_float4(make_float4(vector, w));
      };
      // Cycles keeps the zero initializer after its debug kernel_assert.
      $default {};
    };
    stack_store_float(stack, value_offset, value);
  };
}

void node_tex_gradient(Cursor &cursor, Stack &stack) noexcept {
  const auto gradient_type = cursor.word();
  const auto packed = cursor.word();
  const auto coordinate_offset = cursor.byte(packed, 0u);
  const auto factor_offset = cursor.byte(packed, 1u);
  const auto color_offset = cursor.byte(packed, 2u);
  const auto point = stack_load_float3(stack, coordinate_offset);

  Float factor = 0.0f;
  $if(gradient_type == static_cast<std::uint32_t>(NODE_BLEND_LINEAR)) {
    factor = point.x;
  }
  $elif(gradient_type ==
        static_cast<std::uint32_t>(NODE_BLEND_QUADRATIC)) {
    const auto r = max(point.x, 0.0f);
    factor = r * r;
  }
  $elif(gradient_type == static_cast<std::uint32_t>(NODE_BLEND_EASING)) {
    const auto r = clamp(point.x, 0.0f, 1.0f);
    const auto t = r * r;
    factor = 3.0f * t - 2.0f * t * r;
  }
  $elif(gradient_type == static_cast<std::uint32_t>(NODE_BLEND_DIAGONAL)) {
    factor = (point.x + point.y) * 0.5f;
  }
  $elif(gradient_type == static_cast<std::uint32_t>(NODE_BLEND_RADIAL)) {
    factor = atan2(point.y, point.x) / 6.2831853071795864f + 0.5f;
  }
  $else {
    const auto r = max(
        0.999999f -
            sqrt(point.x * point.x + point.y * point.y + point.z * point.z),
        0.0f);
    $if(gradient_type ==
        static_cast<std::uint32_t>(NODE_BLEND_QUADRATIC_SPHERE)) {
      factor = r * r;
    }
    $elif(gradient_type ==
          static_cast<std::uint32_t>(NODE_BLEND_SPHERICAL)) {
      factor = r;
    };
  };
  factor = clamp(factor, 0.0f, 1.0f);

  $if(factor_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, factor_offset, factor);
  };
  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, color_offset, make_float3(factor));
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail
