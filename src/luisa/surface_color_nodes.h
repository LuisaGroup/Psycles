#pragma once

#include <cstdint>
#include <span>

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Shared Cycles 5.2 color-node semantics. GraphSurface and the compact typed
// SVM evaluator call the same strongly typed operations; neither path owns a
// second formula or a weak float4 register protocol.
[[nodiscard]] Float3 rgb_to_hsv(const ShaderServices &services,
                                Float3 rgb) noexcept;
[[nodiscard]] Float3 hsv_to_rgb(const ShaderServices &services,
                                Float3 hsv) noexcept;
[[nodiscard]] Float3 rgb_to_hsl(const ShaderServices &services,
                                Float3 rgb) noexcept;
[[nodiscard]] Float3 hsl_to_rgb(const ShaderServices &services,
                                Float3 hsl) noexcept;

[[nodiscard]] Float3 evaluate_surface_hsv(const ShaderServices &services,
                                          Float3 color, Float hue,
                                          Float saturation, Float value,
                                          Float factor) noexcept;
[[nodiscard]] Float3 evaluate_surface_invert(Float3 color,
                                             Float factor) noexcept;
[[nodiscard]] Float3 evaluate_surface_gamma(Float3 color,
                                            Float exponent) noexcept;
[[nodiscard]] Float3
evaluate_surface_brightness_contrast(Float3 color, Float brightness,
                                     Float contrast) noexcept;
[[nodiscard]] Float3 evaluate_surface_blackbody(const ShaderServices &services,
                                                Float temperature) noexcept;
[[nodiscard]] Float3 evaluate_surface_wavelength(const ShaderServices &services,
                                                 Float nanometers) noexcept;

[[nodiscard]] Float3 separate_color(const ShaderServices &services,
                                    Float3 color, std::uint32_t mode) noexcept;
[[nodiscard]] Float3 combine_color(const ShaderServices &services,
                                   Float3 channels,
                                   std::uint32_t mode) noexcept;

// The finite immediate image is part of the host proof: only scene-reachable
// RGB/HSV/HSL cases are recorded, while the instruction remains runtime data.
[[nodiscard]] Float3
separate_color_svm(const ShaderServices &services, UInt immediate,
                   std::span<const std::uint16_t> immediate_domain,
                   Float3 color) noexcept;
[[nodiscard]] Float3
combine_color_svm(const ShaderServices &services, UInt immediate,
                  std::span<const std::uint16_t> immediate_domain,
                  Float3 channels) noexcept;

} // namespace psycles::luisa_backend::detail
