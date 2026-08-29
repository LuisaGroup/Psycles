#pragma once

#include <cstdint>
#include <span>

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// These are distinct Cycles execution shapes, not authored projection modes.
// FLAT/SPHERE/TUBE share `image`; BOX may issue three samples; environment
// reuses projection bit one for MIRROR_BALL rather than BOX.
enum class SurfaceImageSvmShape : std::uint8_t {
    image,
    image_box,
    environment,
};

[[nodiscard]] Float4 evaluate_surface_image_svm(
    const ShaderServices &services, const SurfacePoint &point,
    SurfaceImageSvmShape shape, UInt immediate,
    std::span<const std::uint16_t> immediate_domain, Float3 coordinate,
    UInt texture_handle, Float projection_blend) noexcept;

} // namespace psycles::luisa_backend::detail
