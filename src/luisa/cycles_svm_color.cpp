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

[[nodiscard]] Float3 mix_interp(Expr<luisa::float3> a,
                                Expr<luisa::float3> b, Float t) noexcept {
  return a + t * (b - a);
}

[[nodiscard]] Float3 mix_blend(Float t, Expr<luisa::float3> color1,
                               Expr<luisa::float3> color2) noexcept {
  return mix_interp(color1, color2, t);
}

[[nodiscard]] Float3 mix_add(Float t, Expr<luisa::float3> color1,
                             Expr<luisa::float3> color2) noexcept {
  return mix_interp(color1, color1 + color2, t);
}

[[nodiscard]] Float3 mix_multiply(Float t, Expr<luisa::float3> color1,
                                  Expr<luisa::float3> color2) noexcept {
  return mix_interp(color1, color1 * color2, t);
}

[[nodiscard]] Float3 mix_screen(Float t, Expr<luisa::float3> color1_value,
                                Expr<luisa::float3> color2_value) noexcept {
  const Float3 color1 = color1_value;
  const Float3 color2 = color2_value;
  const Float tm = 1.0f - t;
  const Float3 one = make_float3(1.0f);
  return one - (make_float3(tm) + t * (one - color2)) * (one - color1);
}

[[nodiscard]] Float3 mix_overlay(Float t, Expr<luisa::float3> color1,
                                 Expr<luisa::float3> color2_value) noexcept {
  const Float3 color2 = color2_value;
  const Float tm = 1.0f - t;
  Float3 result = color1;
  $if (result.x < 0.5f) {
    result.x *= tm + 2.0f * t * color2.x;
  }
  $else {
    result.x =
        1.0f - (tm + 2.0f * t * (1.0f - color2.x)) * (1.0f - result.x);
  };
  $if (result.y < 0.5f) {
    result.y *= tm + 2.0f * t * color2.y;
  }
  $else {
    result.y =
        1.0f - (tm + 2.0f * t * (1.0f - color2.y)) * (1.0f - result.y);
  };
  $if (result.z < 0.5f) {
    result.z *= tm + 2.0f * t * color2.z;
  }
  $else {
    result.z =
        1.0f - (tm + 2.0f * t * (1.0f - color2.z)) * (1.0f - result.z);
  };
  return result;
}

[[nodiscard]] Float3 mix_subtract(Float t, Expr<luisa::float3> color1,
                                  Expr<luisa::float3> color2) noexcept {
  return mix_interp(color1, color1 - color2, t);
}

[[nodiscard]] Float3 mix_divide(Float t, Expr<luisa::float3> color1,
                                Expr<luisa::float3> color2_value) noexcept {
  const Float3 color2 = color2_value;
  const Float tm = 1.0f - t;
  Float3 result = color1;
  $if (color2.x != 0.0f) {
    result.x = tm * result.x + t * result.x / color2.x;
  };
  $if (color2.y != 0.0f) {
    result.y = tm * result.y + t * result.y / color2.y;
  };
  $if (color2.z != 0.0f) {
    result.z = tm * result.z + t * result.z / color2.z;
  };
  return result;
}

[[nodiscard]] Float3 mix_difference(Float t, Expr<luisa::float3> color1,
                                    Expr<luisa::float3> color2) noexcept {
  return mix_interp(color1, abs(color1 - color2), t);
}

[[nodiscard]] Float3 mix_exclusion(Float t, Expr<luisa::float3> color1,
                                   Expr<luisa::float3> color2) noexcept {
  return luisa::compute::max(
      mix_interp(color1, color1 + color2 - 2.0f * color1 * color2, t),
      make_float3(0.0f));
}

[[nodiscard]] Float3 mix_darken(Float t, Expr<luisa::float3> color1,
                                Expr<luisa::float3> color2) noexcept {
  return mix_interp(color1, luisa::compute::min(color1, color2), t);
}

[[nodiscard]] Float3 mix_lighten(Float t, Expr<luisa::float3> color1,
                                 Expr<luisa::float3> color2) noexcept {
  return mix_interp(color1, luisa::compute::max(color1, color2), t);
}

[[nodiscard]] Float mix_dodge_component(Float t, Float color1,
                                        Float color2) noexcept {
  Float result = color1;
  $if (result != 0.0f) {
    Float value = 1.0f - t * color2;
    $if (value <= 0.0f) { result = 1.0f; }
    $else {
      value = result / value;
      $if (value > 1.0f) { result = 1.0f; }
      $else { result = value; };
    };
  };
  return result;
}

[[nodiscard]] Float3 mix_dodge(Float t, Expr<luisa::float3> color1_value,
                               Expr<luisa::float3> color2_value) noexcept {
  const Float3 color1 = color1_value;
  const Float3 color2 = color2_value;
  return make_float3(mix_dodge_component(t, color1.x, color2.x),
                     mix_dodge_component(t, color1.y, color2.y),
                     mix_dodge_component(t, color1.z, color2.z));
}

[[nodiscard]] Float mix_burn_component(Float t, Float color1,
                                       Float color2) noexcept {
  Float result = color1;
  Float value = 1.0f - t + t * color2;
  $if (value <= 0.0f) { result = 0.0f; }
  $else {
    value = 1.0f - (1.0f - result) / value;
    $if (value < 0.0f) { result = 0.0f; }
    $elif (value > 1.0f) { result = 1.0f; }
    $else { result = value; };
  };
  return result;
}

[[nodiscard]] Float3 mix_burn(Float t, Expr<luisa::float3> color1_value,
                              Expr<luisa::float3> color2_value) noexcept {
  const Float3 color1 = color1_value;
  const Float3 color2 = color2_value;
  return make_float3(mix_burn_component(t, color1.x, color2.x),
                     mix_burn_component(t, color1.y, color2.y),
                     mix_burn_component(t, color1.z, color2.z));
}

[[nodiscard]] Float3 mix_hue(Float t, Expr<luisa::float3> color1,
                             Expr<luisa::float3> color2) noexcept {
  Float3 result = color1;
  const Float3 hsv2 = rgb_to_hsv(color2);
  $if (hsv2.y != 0.0f) {
    Float3 hsv = rgb_to_hsv(result);
    hsv.x = hsv2.x;
    result = mix_interp(result, hsv_to_rgb(hsv), t);
  };
  return result;
}

[[nodiscard]] Float3 mix_saturation(Float t, Expr<luisa::float3> color1,
                                    Expr<luisa::float3> color2) noexcept {
  const Float tm = 1.0f - t;
  Float3 result = color1;
  Float3 hsv = rgb_to_hsv(result);
  $if (hsv.y != 0.0f) {
    const Float3 hsv2 = rgb_to_hsv(color2);
    hsv.y = tm * hsv.y + t * hsv2.y;
    result = hsv_to_rgb(hsv);
  };
  return result;
}

[[nodiscard]] Float3 mix_value(Float t, Expr<luisa::float3> color1,
                               Expr<luisa::float3> color2) noexcept {
  const Float tm = 1.0f - t;
  Float3 hsv = rgb_to_hsv(color1);
  const Float3 hsv2 = rgb_to_hsv(color2);
  hsv.z = tm * hsv.z + t * hsv2.z;
  return hsv_to_rgb(hsv);
}

[[nodiscard]] Float3 mix_color(Float t, Expr<luisa::float3> color1,
                               Expr<luisa::float3> color2) noexcept {
  Float3 result = color1;
  const Float3 hsv2 = rgb_to_hsv(color2);
  $if (hsv2.y != 0.0f) {
    Float3 hsv = rgb_to_hsv(result);
    hsv.x = hsv2.x;
    hsv.y = hsv2.y;
    result = mix_interp(result, hsv_to_rgb(hsv), t);
  };
  return result;
}

[[nodiscard]] Float3 mix_soft_light(Float t,
                                    Expr<luisa::float3> color1_value,
                                    Expr<luisa::float3> color2_value) noexcept {
  const Float3 color1 = color1_value;
  const Float3 color2 = color2_value;
  const Float tm = 1.0f - t;
  const Float3 one = make_float3(1.0f);
  const Float3 screen = one - (one - color2) * (one - color1);
  return tm * color1 +
         t * ((one - color1) * color2 * color1 + color1 * screen);
}

[[nodiscard]] Float3 mix_linear_light(Float t,
                                      Expr<luisa::float3> color1,
                                      Expr<luisa::float3> color2) noexcept {
  return color1 + t * (2.0f * color2 + make_float3(-1.0f));
}

[[nodiscard]] Float3 svm_mix(UInt type, Float t,
                             Expr<luisa::float3> color1,
                             Expr<luisa::float3> color2) noexcept {
  Float3 result = make_float3(0.0f);
  $switch(type) {
    $case(static_cast<std::uint32_t>(NODE_MIX_BLEND)) {
      result = mix_blend(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_ADD)) {
      result = mix_add(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_MUL)) {
      result = mix_multiply(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_SCREEN)) {
      result = mix_screen(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_OVERLAY)) {
      result = mix_overlay(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_SUB)) {
      result = mix_subtract(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_DIV)) {
      result = mix_divide(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_DIFF)) {
      result = mix_difference(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_EXCLUSION)) {
      result = mix_exclusion(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_DARK)) {
      result = mix_darken(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_LIGHT)) {
      result = mix_lighten(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_DODGE)) {
      result = mix_dodge(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_BURN)) {
      result = mix_burn(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_HUE)) {
      result = mix_hue(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_SAT)) {
      result = mix_saturation(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_VAL)) {
      result = mix_value(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_COL)) {
      result = mix_color(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_SOFT)) {
      result = mix_soft_light(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_LINEAR)) {
      result = mix_linear_light(t, color1, color2);
    };
    $case(static_cast<std::uint32_t>(NODE_MIX_CLAMP)) {
      result = luisa::compute::clamp(color1, 0.0f, 1.0f);
    };
    $default{};
  };
  return result;
}

[[nodiscard]] Float3 svm_mix_clamped_factor(
    UInt type, Float t, Expr<luisa::float3> color1,
    Expr<luisa::float3> color2) noexcept {
  return svm_mix(type, luisa::compute::clamp(t, 0.0f, 1.0f), color1, color2);
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

void node_mix(Cursor &cursor, Stack &stack) noexcept {
  const auto mix_type = cursor.word();
  const auto color1_x = cursor.word();
  const auto color1_y = cursor.word();
  const auto color1_z = cursor.word();
  const auto color2_x = cursor.word();
  const auto color2_y = cursor.word();
  const auto color2_z = cursor.word();
  const auto factor_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const Float factor = stack_load_input_float(stack, factor_input);
  const Float3 color1 =
      stack_load_input_float3(stack, color1_x, color1_y, color1_z);
  const Float3 color2 =
      stack_load_input_float3(stack, color2_x, color2_y, color2_z);
  const Float3 result =
      svm_mix_clamped_factor(mix_type, factor, color1, color2);
  stack_store_float3(stack, output_offset, result);
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
