#include "surface_image_sampling.h"

#include "surface_color_encoding.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

Float4 decode_surface_image_sample(
    Float4 sampled,
    Bool unassociate_alpha,
    Bool encoded_as_srgb) noexcept {
    $if(unassociate_alpha) {
        const auto alpha = sampled.w;
        const auto should_unassociate =
            (alpha != 0.0f) & (alpha != 1.0f);
        const auto safe_alpha =
            select(1.0f, alpha, should_unassociate);
        sampled = make_float4(
            select(
                sampled.xyz(),
                sampled.xyz() / safe_alpha,
                should_unassociate),
            alpha);
    };
    $if(encoded_as_srgb) {
        // Cycles filters associated encoded texels first, optionally
        // unassociates alpha, and only then decodes the RGB channels.
        sampled = make_float4(
            srgb_to_linear(sampled.xyz()),
            sampled.w);
    };
    return sampled;
}

}// namespace psycles::luisa_backend::detail
