/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_checker.h>
#include <psycles/luisa/cycles_magic.h>
#include <psycles/luisa/cycles_wave.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

inline constexpr auto pi_over_two = 1.57079632679489661923f;
inline constexpr auto two_pi = 6.28318530717958647692f;

[[nodiscard]] Float wave_coordinate(UInt wave_type, UInt bands_direction,
                                    UInt rings_direction,
                                    Float3 point) noexcept {
  Float coordinate = 0.0f;
  $if(wave_type == static_cast<std::uint32_t>(NODE_WAVE_BANDS)) {
    $if(bands_direction ==
        static_cast<std::uint32_t>(NODE_WAVE_BANDS_DIRECTION_X)) {
      coordinate = point.x * 20.0f;
    }
    $elif(bands_direction ==
          static_cast<std::uint32_t>(NODE_WAVE_BANDS_DIRECTION_Y)) {
      coordinate = point.y * 20.0f;
    }
    $elif(bands_direction ==
          static_cast<std::uint32_t>(NODE_WAVE_BANDS_DIRECTION_Z)) {
      coordinate = point.z * 20.0f;
    }
    $else { coordinate = (point.x + point.y + point.z) * 10.0f; };
  }
  $else {
    Float3 radial = point;
    $if(rings_direction ==
        static_cast<std::uint32_t>(NODE_WAVE_RINGS_DIRECTION_X)) {
      radial *= make_float3(0.0f, 1.0f, 1.0f);
    }
    $elif(rings_direction ==
          static_cast<std::uint32_t>(NODE_WAVE_RINGS_DIRECTION_Y)) {
      radial *= make_float3(1.0f, 0.0f, 1.0f);
    }
    $elif(rings_direction ==
          static_cast<std::uint32_t>(NODE_WAVE_RINGS_DIRECTION_Z)) {
      radial *= make_float3(1.0f, 1.0f, 0.0f);
    };
    coordinate = length(radial) * 20.0f;
  };
  return coordinate;
}

[[nodiscard]] Float wave_profile(UInt profile, Float coordinate) noexcept {
  Float factor = 0.0f;
  $if(profile == static_cast<std::uint32_t>(NODE_WAVE_PROFILE_SIN)) {
    factor = 0.5f + 0.5f * sin(coordinate - pi_over_two);
  }
  $elif(profile == static_cast<std::uint32_t>(NODE_WAVE_PROFILE_SAW)) {
    coordinate /= two_pi;
    factor = coordinate - floor(coordinate);
  }
  $else {
    coordinate /= two_pi;
    factor = abs(coordinate - floor(coordinate + 0.5f)) * 2.0f;
  };
  return factor;
}

[[nodiscard]] Float smoothstep_cycles(Float value) noexcept {
  Float result = 0.0f;
  $if(value >= 1.0f) { result = 1.0f; }
  $elif(value > 0.0f) {
    const auto squared = value * value;
    result = 3.0f * squared - 2.0f * squared * value;
  };
  return result;
}

[[nodiscard]] Float brick_noise(UInt value) noexcept {
  value = (value + 1013u) & 0x7fffffffu;
  value = (value >> 13u) ^ value;
  const auto noise =
      (value * (value * value * 60493u + 19990303u) + 1376312589u) &
      0x7fffffffu;
  return 0.5f * noise.cast<float>() / 1073741824.0f;
}

struct BrickResult {
  Float tint;
  Float mortar;
};

[[nodiscard]] BrickResult
evaluate_brick(Float3 point, Float mortar_size, Float mortar_smooth, Float bias,
               Float brick_width, Float row_height, Float offset_amount,
               Int offset_frequency, Float squash_amount,
               Int squash_frequency) noexcept {
  const Int row = floor(point.y / row_height).cast<std::int32_t>();
  Float offset = 0.0f;
  $if((offset_frequency != 0) & (squash_frequency != 0)) {
    $if((row % squash_frequency) == 0) { brick_width *= squash_amount; };
    $if((row % offset_frequency) == 0) {
      offset = brick_width * offset_amount;
    };
  };

  const Int brick =
      floor((point.x + offset) / brick_width).cast<std::int32_t>();
  const Float x = point.x + offset - brick_width * brick.cast<float>();
  const Float y = point.y - row_height * row.cast<float>();
  const UInt hash = (row.cast<std::uint32_t>() << 16u) +
                    (brick.cast<std::uint32_t>() & 0xffffu);
  const Float tint = clamp(brick_noise(hash) + bias, 0.0f, 1.0f);
  Float minimum_distance =
      luisa::compute::min(luisa::compute::min(x, y),
                          luisa::compute::min(brick_width - x, row_height - y));
  Float mortar = 0.0f;
  $if(minimum_distance < mortar_size) {
    $if(mortar_smooth == 0.0f) { mortar = 1.0f; }
    $else {
      minimum_distance = 1.0f - minimum_distance / mortar_size;
      mortar = smoothstep_cycles(minimum_distance / mortar_smooth);
    };
  };
  return {.tint = tint, .mortar = mortar};
}

} // namespace

void node_tex_wave(Cursor &cursor, Stack &stack) noexcept {
  const auto wave_type = cursor.word();
  const auto bands_direction = cursor.word();
  const auto rings_direction = cursor.word();
  const auto profile = cursor.word();
  const auto scale_bits = cursor.word();
  const auto distortion_bits = cursor.word();
  const auto detail_bits = cursor.word();
  const auto detail_scale_bits = cursor.word();
  const auto detail_roughness_bits = cursor.word();
  const auto phase_bits = cursor.word();
  const auto packed = cursor.word();
  const auto coordinate_offset = cursor.byte(packed, 0u);
  const auto color_offset = cursor.byte(packed, 1u);
  const auto factor_offset = cursor.byte(packed, 2u);

  auto point = stack_load_float3(stack, coordinate_offset) *
               stack_load_input_float(stack, scale_bits);
  point = (point + 0.000001f) * 0.999999f;
  auto coordinate =
      wave_coordinate(wave_type, bands_direction, rings_direction, point);
  coordinate += stack_load_input_float(stack, phase_bits);
  const auto distortion = stack_load_input_float(stack, distortion_bits);
  $if(distortion != 0.0f) {
    const auto detail = stack_load_input_float(stack, detail_bits);
    const auto detail_scale = stack_load_input_float(stack, detail_scale_bits);
    const auto detail_roughness =
        stack_load_input_float(stack, detail_roughness_bits);
    coordinate +=
        distortion * (cycles_wave::distortion_noise(point * detail_scale,
                                                    detail, detail_roughness) *
                          2.0f -
                      1.0f);
  };
  const auto factor = wave_profile(profile, coordinate);
  $if(factor_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, factor_offset, factor);
  };
  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, color_offset, make_float3(factor));
  };
}

void node_tex_magic(Cursor &cursor, Stack &stack) noexcept {
  const auto scale_bits = cursor.word();
  const auto distortion_bits = cursor.word();
  const auto packed = cursor.word();
  const auto depth = cursor.byte(packed, 0u);
  const auto coordinate_offset = cursor.byte(packed, 1u);
  const auto color_offset = cursor.byte(packed, 2u);
  const auto factor_offset = cursor.byte(packed, 3u);
  const auto color =
      cycles_magic::evaluate(depth, stack_load_float3(stack, coordinate_offset),
                             stack_load_input_float(stack, scale_bits),
                             stack_load_input_float(stack, distortion_bits));
  $if(factor_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, factor_offset,
                      (color.x + color.y + color.z) * (1.0f / 3.0f));
  };
  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, color_offset, color);
  };
}

void node_tex_checker(Cursor &cursor, Stack &stack) noexcept {
  const auto color1_x = cursor.word();
  const auto color1_y = cursor.word();
  const auto color1_z = cursor.word();
  const auto color2_x = cursor.word();
  const auto color2_y = cursor.word();
  const auto color2_z = cursor.word();
  const auto scale_bits = cursor.word();
  const auto packed = cursor.word();
  const auto coordinate_offset = cursor.byte(packed, 0u);
  const auto color_offset = cursor.byte(packed, 1u);
  const auto factor_offset = cursor.byte(packed, 2u);

  const auto point = stack_load_float3(stack, coordinate_offset) *
                     stack_load_input_float(stack, scale_bits);
  const auto factor = cycles_checker::evaluate(point);
  const auto first = factor == 1.0f;
  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const auto color1 =
        stack_load_input_float3(stack, color1_x, color1_y, color1_z);
    const auto color2 =
        stack_load_input_float3(stack, color2_x, color2_y, color2_z);
    stack_store_float3(stack, color_offset, select(color2, color1, first));
  };
  $if(factor_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, factor_offset, factor);
  };
}

void node_tex_brick(Cursor &cursor, Stack &stack) noexcept {
  const auto color1_x = cursor.word();
  const auto color1_y = cursor.word();
  const auto color1_z = cursor.word();
  const auto color2_x = cursor.word();
  const auto color2_y = cursor.word();
  const auto color2_z = cursor.word();
  const auto mortar_x = cursor.word();
  const auto mortar_y = cursor.word();
  const auto mortar_z = cursor.word();
  const auto scale_bits = cursor.word();
  const auto mortar_size_bits = cursor.word();
  const auto bias_bits = cursor.word();
  const auto brick_width_bits = cursor.word();
  const auto row_height_bits = cursor.word();
  const auto mortar_smooth_bits = cursor.word();
  const auto offset_amount = cursor.floating();
  const auto squash_amount = cursor.floating();
  const auto packed = cursor.word();
  const auto offset_frequency = cursor.byte(packed, 0u).cast<std::int32_t>();
  const auto squash_frequency = cursor.byte(packed, 1u).cast<std::int32_t>();
  const auto coordinate_offset = cursor.byte(packed, 2u);
  const auto color_offset = cursor.byte(packed, 3u);
  const auto tail = cursor.word();
  const auto factor_offset = cursor.byte(tail, 0u);

  const auto point = stack_load_float3(stack, coordinate_offset) *
                     stack_load_input_float(stack, scale_bits);
  const auto evaluated = evaluate_brick(
      point, stack_load_input_float(stack, mortar_size_bits),
      stack_load_input_float(stack, mortar_smooth_bits),
      stack_load_input_float(stack, bias_bits),
      stack_load_input_float(stack, brick_width_bits),
      stack_load_input_float(stack, row_height_bits), offset_amount,
      offset_frequency, squash_amount, squash_frequency);

  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    auto color1 = stack_load_input_float3(stack, color1_x, color1_y, color1_z);
    const auto color2 =
        stack_load_input_float3(stack, color2_x, color2_y, color2_z);
    const auto mortar =
        stack_load_input_float3(stack, mortar_x, mortar_y, mortar_z);
    $if(evaluated.mortar != 1.0f) {
      color1 = (1.0f - evaluated.tint) * color1 + evaluated.tint * color2;
    };
    stack_store_float3(stack, color_offset,
                       color1 * (1.0f - evaluated.mortar) +
                           mortar * evaluated.mortar);
  };
  $if(factor_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, factor_offset, evaluated.mortar);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail
