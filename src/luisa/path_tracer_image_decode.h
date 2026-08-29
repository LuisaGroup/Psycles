#pragma once

#include <psycles/contract/scene.h>
#include <psycles/io/image.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace psycles::luisa_backend::detail {

[[nodiscard]] std::optional<io::DecodedImageRgba>
decode_scene_image(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint);

void apply_scene_image_alpha(
    io::DecodedImageRgba &image,
    contract::ImageAlphaType alpha_type,
    contract::ImageColorSpace color_space) noexcept;

}// namespace psycles::luisa_backend::detail
