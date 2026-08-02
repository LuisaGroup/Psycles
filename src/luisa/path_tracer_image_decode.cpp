#include "path_tracer_image_decode.h"

#include <cstring>
#include <limits>

#include <stb/stb_image.h>

namespace psycles::luisa_backend::detail {

std::optional<io::DecodedImageRgba8>
decode_scene_image_rgba8(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint) {
    if (encoded.empty() ||
        encoded.size() > static_cast<std::size_t>(
                             std::numeric_limits<int>::max())) {
        return std::nullopt;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    auto *pixels = stbi_load_from_memory(
        encoded.data(),
        static_cast<int>(encoded.size()),
        &width,
        &height,
        &channels,
        STBI_rgb_alpha);
    if (pixels != nullptr && width > 0 && height > 0) {
        io::DecodedImageRgba8 image{
            .width = static_cast<std::uint32_t>(width),
            .height = static_cast<std::uint32_t>(height),
            .pixels = {}};
        image.pixels.resize(
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) * 4u);
        std::memcpy(
            image.pixels.data(), pixels, image.pixels.size());
        stbi_image_free(pixels);
        return image;
    }
    stbi_image_free(pixels);

#if defined(PSYCLES_WITH_OPENIMAGEIO)
    io::DecodedImageRgba8 image;
    if (io::decode_image_rgba8(
            encoded, filename_hint, image)) {
        return image;
    }
#else
    static_cast<void>(filename_hint);
#endif
    return std::nullopt;
}

}// namespace psycles::luisa_backend::detail
