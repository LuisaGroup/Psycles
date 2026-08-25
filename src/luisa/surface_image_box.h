#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Host-stage sampling endpoint used while recording the canonical BOX
// operation. Implementations bind one static interpolation/extension mode;
// the virtual call itself never survives into the device program.
class SurfaceImageBoxTextureSampler {

public:
    virtual ~SurfaceImageBoxTextureSampler() noexcept = default;

    [[nodiscard]] virtual Float4 sample(
        Expr<std::uint32_t> texture_handle,
        Float2 uv) const noexcept = 0;
};

// Canonical pure BOX operation. This is deliberately independent of graph
// storage and SVM execution state so it can be recorded either inline or as a
// shared typed callable without changing the node transaction.
[[nodiscard]] Float4 evaluate_surface_image_box(
    const SurfaceImageBoxInput &input,
    const SurfaceImageBoxTextureSampler &sampler) noexcept;

}// namespace psycles::luisa_backend::detail
