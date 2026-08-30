/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

[[nodiscard]] Float fract(Float value) noexcept {
  return value - floor(value);
}

[[nodiscard]] Float3 rgb_to_hsv(Expr<luisa::float3> rgb_value) noexcept {
  const Float3 rgb = rgb_value;
  const Float cmax = luisa::compute::max(
      rgb.x, luisa::compute::max(rgb.y, rgb.z));
  const Float cmin = luisa::compute::min(
      rgb.x, luisa::compute::min(rgb.y, rgb.z));
  const Float cdelta = cmax - cmin;
  Float h = 0.0f;
  Float s = 0.0f;
  const Float v = cmax;

  $if (cmax != 0.0f) { s = cdelta / cmax; };

  $if (s != 0.0f) {
    const Float3 c = (make_float3(cmax) - rgb) / cdelta;
    $if (rgb.x == cmax) {
      h = c.z - c.y;
    }
    $elif (rgb.y == cmax) { h = 2.0f + c.x - c.z; }
    $else { h = 4.0f + c.y - c.x; };
    h /= 6.0f;
    $if (h < 0.0f) { h += 1.0f; };
  };
  return make_float3(h, s, v);
}

[[nodiscard]] Float3 hsv_to_rgb(Expr<luisa::float3> hsv_value) noexcept {
  const Float3 hsv = hsv_value;
  Float h = hsv.x;
  const Float s = hsv.y;
  const Float v = hsv.z;
  Float3 rgb = make_float3(v);

  $if (s != 0.0f) {
    $if (h == 1.0f) { h = 0.0f; };
    h *= 6.0f;
    const Float i = floor(h);
    const Float f = h - i;
    const Float p = v * (1.0f - s);
    const Float q = v * (1.0f - s * f);
    const Float t = v * (1.0f - s * (1.0f - f));

    $if (i == 0.0f) {
      rgb = make_float3(v, t, p);
    }
    $elif (i == 1.0f) { rgb = make_float3(q, v, p); }
    $elif (i == 2.0f) { rgb = make_float3(p, v, t); }
    $elif (i == 3.0f) { rgb = make_float3(p, q, v); }
    $elif (i == 4.0f) { rgb = make_float3(t, p, v); }
    $else { rgb = make_float3(v, p, q); };
  };
  return rgb;
}

[[nodiscard]] Float3 rgb_to_hsl(Expr<luisa::float3> rgb_value) noexcept {
  const Float3 rgb = rgb_value;
  const Float cmax =
      luisa::compute::max(rgb.x, luisa::compute::max(rgb.y, rgb.z));
  const Float cmin =
      luisa::compute::min(rgb.x, luisa::compute::min(rgb.y, rgb.z));
  Float h = 0.0f;
  Float s = 0.0f;
  const Float l = luisa::compute::min(1.0f, (cmax + cmin) / 2.0f);

  $if(cmax != cmin) {
    const Float cdelta = cmax - cmin;
    $if(l > 0.5f) { s = cdelta / (2.0f - cmax - cmin); }
    $else { s = cdelta / (cmax + cmin); };
    $if(cmax == rgb.x) {
      h = (rgb.y - rgb.z) / cdelta;
      $if(rgb.y < rgb.z) { h += 6.0f; };
    }
    $elif(cmax == rgb.y) { h = (rgb.z - rgb.x) / cdelta + 2.0f; }
    $else { h = (rgb.x - rgb.y) / cdelta + 4.0f; };
  };
  h /= 6.0f;
  return make_float3(h, s, l);
}

[[nodiscard]] Float3 hsl_to_rgb(Expr<luisa::float3> hsl_value) noexcept {
  const Float3 hsl = hsl_value;
  Float nr = abs(hsl.x * 6.0f - 3.0f) - 1.0f;
  Float ng = 2.0f - abs(hsl.x * 6.0f - 2.0f);
  Float nb = 2.0f - abs(hsl.x * 6.0f - 4.0f);
  nr = luisa::compute::clamp(nr, 0.0f, 1.0f);
  nb = luisa::compute::clamp(nb, 0.0f, 1.0f);
  ng = luisa::compute::clamp(ng, 0.0f, 1.0f);
  const Float chroma = (1.0f - abs(2.0f * hsl.z - 1.0f)) * hsl.y;
  return make_float3((nr - 0.5f) * chroma + hsl.z, (ng - 0.5f) * chroma + hsl.z,
                     (nb - 0.5f) * chroma + hsl.z);
}

[[nodiscard]] Float3 combine_color(UInt color_type,
                                   Expr<luisa::float3> color_value) noexcept {
  Float3 color = color_value;
  $switch(color_type) {
    $case(static_cast<std::uint32_t>(NODE_COMBSEP_COLOR_HSV)) {
      color = hsv_to_rgb(color);
    };
    $case(static_cast<std::uint32_t>(NODE_COMBSEP_COLOR_HSL)) {
      color = hsl_to_rgb(color);
    };
    $default{};
  };
  return color;
}

[[nodiscard]] Float3 separate_color(UInt color_type,
                                    Expr<luisa::float3> color_value) noexcept {
  Float3 color = color_value;
  $switch(color_type) {
    $case(static_cast<std::uint32_t>(NODE_COMBSEP_COLOR_HSV)) {
      color = rgb_to_hsv(color);
    };
    $case(static_cast<std::uint32_t>(NODE_COMBSEP_COLOR_HSL)) {
      color = rgb_to_hsl(color);
    };
    $default{};
  };
  return color;
}

[[nodiscard]] Float3 gamma_color(Expr<luisa::float3> color_value,
                                 Expr<float> gamma_value) noexcept {
  Float3 color = color_value;
  const Float gamma = gamma_value;
  $if (gamma == 0.0f) {
    color = make_float3(1.0f);
  }
  $else {
    $if (color.x > 0.0f) { color.x = pow(color.x, gamma); };
    $if (color.y > 0.0f) { color.y = pow(color.y, gamma); };
    $if (color.z > 0.0f) { color.z = pow(color.z, gamma); };
  };
  return color;
}

[[nodiscard]] Float3 brightness_contrast(
    Expr<luisa::float3> color_value, Expr<float> brightness_value,
    Expr<float> contrast_value) noexcept {
  Float3 color = color_value;
  const Float brightness = brightness_value;
  const Float contrast = contrast_value;
  const Float a = 1.0f + contrast;
  const Float b = brightness - contrast * 0.5f;
  color.x = luisa::compute::max(a * color.x + b, 0.0f);
  color.y = luisa::compute::max(a * color.y + b, 0.0f);
  color.z = luisa::compute::max(a * color.z + b, 0.0f);
  return color;
}

[[nodiscard]] Float clamp_value(Float value, Float minimum,
                                Float maximum) noexcept {
  return luisa::compute::min(luisa::compute::max(value, minimum), maximum);
}

} // namespace

void node_hsv(Cursor &cursor, Stack &stack) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto hue_input = cursor.word();
  const auto saturation_input = cursor.word();
  const auto value_input = cursor.word();
  const auto factor_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);

  const Float factor = stack_load_input_float(stack, factor_input);
  const Float3 input_color =
      stack_load_input_float3(stack, color_x, color_y, color_z);
  Float3 color = rgb_to_hsv(input_color);
  const Float hue = stack_load_input_float(stack, hue_input);
  const Float saturation = stack_load_input_float(stack, saturation_input);
  const Float value = stack_load_input_float(stack, value_input);

  color.x = fract(color.x + hue + 0.5f);
  color.y = luisa::compute::clamp(color.y * saturation, 0.0f, 1.0f);
  color.z *= value;
  color = hsv_to_rgb(color);
  color = factor * color + (1.0f - factor) * input_color;
  color = luisa::compute::max(color, make_float3(0.0f));

  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, output_offset, color);
  };
}

void node_gamma(Cursor &cursor, Stack &stack) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto gamma_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const auto color = gamma_color(
      stack_load_input_float3(stack, color_x, color_y, color_z),
      stack_load_input_float(stack, gamma_input));
  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, output_offset, color);
  };
}

void node_brightness(Cursor &cursor, Stack &stack) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto brightness_input = cursor.word();
  const auto contrast_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const auto color = brightness_contrast(
      stack_load_input_float3(stack, color_x, color_y, color_z),
      stack_load_input_float(stack, brightness_input),
      stack_load_input_float(stack, contrast_input));
  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, output_offset, color);
  };
}

void node_invert(Cursor &cursor, Stack &stack) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto factor_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const Float factor = stack_load_input_float(stack, factor_input);
  const Float3 input_color =
      stack_load_input_float3(stack, color_x, color_y, color_z);
  const Float3 color =
      factor * (make_float3(1.0f) - input_color) +
      (1.0f - factor) * input_color;
  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, output_offset, color);
  };
}

void node_separate_color(Cursor &cursor, Stack &stack) noexcept {
  const auto color_type = cursor.word();
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto packed_outputs = cursor.word();
  const auto red_offset = cursor.byte(packed_outputs, 0u);
  const auto green_offset = cursor.byte(packed_outputs, 1u);
  const auto blue_offset = cursor.byte(packed_outputs, 2u);
  const auto color = separate_color(
      color_type, stack_load_input_float3(stack, color_x, color_y, color_z));
  $if(red_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, red_offset, color.x);
  };
  $if(green_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, green_offset, color.y);
  };
  $if(blue_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, blue_offset, color.z);
  };
}

void node_combine_color(Cursor &cursor, Stack &stack) noexcept {
  const auto color_type = cursor.word();
  const auto red_input = cursor.word();
  const auto green_input = cursor.word();
  const auto blue_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const auto color = combine_color(
      color_type, make_float3(stack_load_input_float(stack, red_input),
                              stack_load_input_float(stack, green_input),
                              stack_load_input_float(stack, blue_input)));
  $if(output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, output_offset, color);
  };
}

void node_clamp(Cursor &cursor, Stack &stack) noexcept {
  const auto clamp_type = cursor.word();
  const auto minimum_input = cursor.word();
  const auto maximum_input = cursor.word();
  const auto value_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const Float value = stack_load_input_float(stack, value_input);
  const Float minimum = stack_load_input_float(stack, minimum_input);
  const Float maximum = stack_load_input_float(stack, maximum_input);
  Float result = clamp_value(value, minimum, maximum);
  $if ((clamp_type == static_cast<std::uint32_t>(NODE_CLAMP_RANGE)) &
       (minimum > maximum)) {
    result = clamp_value(value, maximum, minimum);
  };
  stack_store_float(stack, output_offset, result);
}

} // namespace psycles::luisa_backend::cycles_svm::detail
