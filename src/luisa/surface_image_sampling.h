#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Canonical Cycles Image Texture post-filter transform. Keeping this as one
// operation prevents ordinary and BOX projections from drifting in alpha and
// color-space ordering.
[[nodiscard]] Float4 decode_surface_image_sample(
    Float4 sampled,
    Bool unassociate_alpha,
    Bool encoded_as_srgb) noexcept;

}// namespace psycles::luisa_backend::detail
