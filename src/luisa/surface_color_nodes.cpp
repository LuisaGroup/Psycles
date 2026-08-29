#include "surface_color_nodes.h"

#include "surface_color_transforms.h"

#include <array>
#include <cstdlib>
#include <utility>

#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/luisa/cycles_color_nodes.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr std::size_t color_mode_count = 3u;

[[nodiscard]] auto
active_color_modes(std::span<const std::uint16_t> immediate_domain) noexcept {
  std::array<bool, color_mode_count> active{};
  if (immediate_domain.empty()) {
    std::abort();
  }
  for (const auto encoded : immediate_domain) {
    if ((encoded & ~compiler::surface_value_color_mode_mask) != 0u ||
        encoded >= active.size()) {
      std::abort();
    }
    active[encoded] = true;
  }
  return active;
}

template <typename Evaluate>
[[nodiscard]] Float3
evaluate_color_mode_svm(UInt immediate,
                        std::span<const std::uint16_t> immediate_domain,
                        Evaluate &&evaluate) noexcept {
  const auto active = active_color_modes(immediate_domain);
  Float3 result = make_float3(0.0f);
  luisa::compute::detail::SwitchStmtBuilder{
      immediate & compiler::surface_value_color_mode_mask} %
      [&] {
        for (auto mode = std::size_t{}; mode < active.size(); ++mode) {
          if (!active[mode]) {
            continue;
          }
          luisa::compute::detail::SwitchCaseStmtBuilder{
              static_cast<luisa::uint>(mode)} %
              [&, mode] {
                result = evaluate(static_cast<std::uint32_t>(mode));
              };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
          luisa::compute::dsl::unreachable(
              "invalid compact surface color mode");
        };
      };
  return result;
}

} // namespace

Float3 rgb_to_hsv(const ShaderServices &services, Float3 rgb) noexcept {
  if (const auto provider = services.surface_color_transform_provider()) {
    return provider->rgb_to_hsv(rgb);
  }
  return rgb_to_hsv_inline(rgb);
}

Float3 hsv_to_rgb(const ShaderServices &services, Float3 hsv) noexcept {
  if (const auto provider = services.surface_color_transform_provider()) {
    return provider->hsv_to_rgb(hsv);
  }
  return hsv_to_rgb_inline(hsv);
}

Float3 rgb_to_hsl(const ShaderServices &services, Float3 rgb) noexcept {
  if (const auto provider = services.surface_color_transform_provider()) {
    return provider->rgb_to_hsl(rgb);
  }
  return rgb_to_hsl_inline(rgb);
}

Float3 hsl_to_rgb(const ShaderServices &services, Float3 hsl) noexcept {
  if (const auto provider = services.surface_color_transform_provider()) {
    return provider->hsl_to_rgb(hsl);
  }
  return hsl_to_rgb_inline(hsl);
}

Float3 evaluate_surface_hsv(const ShaderServices &services, Float3 color,
                            Float hue, Float saturation, Float value,
                            Float factor) noexcept {
  auto adjusted = rgb_to_hsv(services, color);
  adjusted.x = fract(adjusted.x + hue + 0.5f);
  adjusted.y = clamp(adjusted.y * saturation, 0.0f, 1.0f);
  adjusted.z *= value;
  adjusted = hsv_to_rgb(services, adjusted);
  // Cycles deliberately leaves Fac unclamped and clamps only the final RGB
  // lower bound after interpolation.
  return max(lerp(color, adjusted, factor), make_float3(0.0f));
}

Float3 evaluate_surface_invert(Float3 color, Float factor) noexcept {
  return lerp(color, make_float3(1.0f) - color, factor);
}

Float3 evaluate_surface_gamma(Float3 color, Float exponent) noexcept {
  const auto positive = color > 0.0f;
  const auto safe = select(make_float3(1.0f), color, positive);
  auto result = select(color, pow(safe, exponent), positive);
  return select(result, make_float3(1.0f), exponent == 0.0f);
}

Float3 evaluate_surface_brightness_contrast(Float3 color, Float brightness,
                                            Float contrast) noexcept {
  const auto a = 1.0f + contrast;
  const auto b = brightness - contrast * 0.5f;
  return max(a * color + make_float3(b), make_float3(0.0f));
}

Float3 evaluate_surface_blackbody(const ShaderServices &services,
                                  Float temperature) noexcept {
  return max(
      services.rec709_to_rgb(cycles_color_nodes::blackbody_rec709(temperature)),
      make_float3(0.0f));
}

Float3 evaluate_surface_wavelength(const ShaderServices &services,
                                   Float nanometers) noexcept {
  return max(
      services.xyz_to_rgb(cycles_color_nodes::wavelength_xyz(nanometers)) *
          (1.0f / 2.52f),
      make_float3(0.0f));
}

Float3 separate_color(const ShaderServices &services, Float3 color,
                      std::uint32_t mode) noexcept {
  switch (mode) {
  case 1u:
    return rgb_to_hsv(services, color);
  case 2u:
    return rgb_to_hsl(services, color);
  case 0u:
  default:
    return color;
  }
}

Float3 combine_color(const ShaderServices &services, Float3 channels,
                     std::uint32_t mode) noexcept {
  switch (mode) {
  case 1u:
    return hsv_to_rgb(services, channels);
  case 2u:
    return hsl_to_rgb(services, channels);
  case 0u:
  default:
    return channels;
  }
}

Float3 separate_color_svm(const ShaderServices &services, UInt immediate,
                          std::span<const std::uint16_t> immediate_domain,
                          Float3 color) noexcept {
  return evaluate_color_mode_svm(std::move(immediate), immediate_domain,
                                 [&](std::uint32_t mode) {
                                   return separate_color(services, color, mode);
                                 });
}

Float3 combine_color_svm(const ShaderServices &services, UInt immediate,
                         std::span<const std::uint16_t> immediate_domain,
                         Float3 channels) noexcept {
  return evaluate_color_mode_svm(
      std::move(immediate), immediate_domain, [&](std::uint32_t mode) {
        return combine_color(services, channels, mode);
      });
}

} // namespace psycles::luisa_backend::detail
