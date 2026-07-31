#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_guiding.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::volume_guiding {

inline constexpr std::uint32_t raw_scatter_slot = 0u;
inline constexpr std::uint32_t raw_transmit_slot = 1u;
inline constexpr std::uint32_t optical_depth_slot = 2u;
inline constexpr std::uint32_t raw_pixel_stride = 3u;

inline constexpr std::uint32_t denoised_scatter_slot = 0u;
inline constexpr std::uint32_t denoised_transmit_slot = 1u;
inline constexpr std::uint32_t denoised_pixel_stride = 2u;

inline constexpr std::uint32_t filter_radius = 5u;
inline constexpr std::uint32_t filter_width =
    filter_radius * 2u + 1u;

// Exact signed 8/8/8/5 RGBE representation used by current Cycles volume
// guiding passes. The remaining three exponent-byte bits carry RGB signs.
[[nodiscard]] luisa::compute::UInt
encode_rgbe(luisa::compute::Float3 rgb) noexcept;

[[nodiscard]] luisa::compute::Float3
decode_rgbe(luisa::compute::UInt rgbe) noexcept;

}// namespace psycles::luisa_backend::volume_guiding
