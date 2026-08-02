#pragma once

#include <psycles/io/image.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace psycles::luisa_backend::detail {

[[nodiscard]] std::optional<io::DecodedImageRgba8>
decode_scene_image_rgba8(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint);

}// namespace psycles::luisa_backend::detail
