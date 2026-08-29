#include "path_tracer_image_decode.h"

#include <cstring>
#include <limits>

#include <stb/stb_image.h>

namespace psycles::luisa_backend::detail {

std::optional<io::DecodedImageRgba>
decode_scene_image(
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
    const auto is_hdr =
        stbi_is_hdr_from_memory(
            encoded.data(),
            static_cast<int>(encoded.size())) != 0;
    const auto is_sixteen_bit =
        stbi_is_16_bit_from_memory(
            encoded.data(),
            static_cast<int>(encoded.size())) != 0;
#if defined(PSYCLES_WITH_OPENIMAGEIO)
    if (is_hdr || is_sixteen_bit) {
        io::DecodedImageRgba image;
        if (io::decode_image_rgba(
                encoded, filename_hint, image)) {
            return image;
        }
    }
#else
    static_cast<void>(filename_hint);
#endif
    if (is_hdr) {
        auto *pixels = stbi_loadf_from_memory(
            encoded.data(),
            static_cast<int>(encoded.size()),
            &width,
            &height,
            &channels,
            STBI_rgb_alpha);
        if (pixels != nullptr && width > 0 && height > 0) {
            io::DecodedImageRgba image{
                .width = static_cast<std::uint32_t>(width),
                .height = static_cast<std::uint32_t>(height),
                .storage = io::DecodedImageStorage::float32,
                .unorm8_pixels = {},
                .float_pixels = {}};
            image.float_pixels.assign(
                pixels,
                pixels +
                    static_cast<std::size_t>(width) *
                        static_cast<std::size_t>(height) * 4u);
            stbi_image_free(pixels);
            return image;
        }
        stbi_image_free(pixels);
        return std::nullopt;
    }
    if (is_sixteen_bit) {
        auto *pixels = stbi_load_16_from_memory(
            encoded.data(),
            static_cast<int>(encoded.size()),
            &width,
            &height,
            &channels,
            STBI_rgb_alpha);
        if (pixels != nullptr && width > 0 && height > 0) {
            io::DecodedImageRgba image{
                .width = static_cast<std::uint32_t>(width),
                .height = static_cast<std::uint32_t>(height),
                .storage = io::DecodedImageStorage::float32,
                .unorm8_pixels = {},
                .float_pixels = {}};
            const auto value_count =
                static_cast<std::size_t>(width) *
                static_cast<std::size_t>(height) * 4u;
            image.float_pixels.reserve(value_count);
            for (std::size_t index = 0u;
                 index < value_count;
                 ++index) {
                image.float_pixels.emplace_back(
                    static_cast<float>(pixels[index]) /
                    65535.0f);
            }
            stbi_image_free(pixels);
            return image;
        }
        stbi_image_free(pixels);
        return std::nullopt;
    }
    auto *pixels = stbi_load_from_memory(
        encoded.data(),
        static_cast<int>(encoded.size()),
        &width,
        &height,
        &channels,
        STBI_rgb_alpha);
    if (pixels != nullptr && width > 0 && height > 0) {
        io::DecodedImageRgba image{
            .width = static_cast<std::uint32_t>(width),
            .height = static_cast<std::uint32_t>(height),
            .storage = io::DecodedImageStorage::unorm8,
            .unorm8_pixels = {},
            .float_pixels = {}};
        image.unorm8_pixels.resize(
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) * 4u);
        std::memcpy(
            image.unorm8_pixels.data(),
            pixels,
            image.unorm8_pixels.size());
        stbi_image_free(pixels);
        return image;
    }
    stbi_image_free(pixels);
#if defined(PSYCLES_WITH_OPENIMAGEIO)
    io::DecodedImageRgba image;
    if (io::decode_image_rgba(
            encoded, filename_hint, image)) {
        return image;
    }
#endif
    return std::nullopt;
}

void apply_scene_image_alpha(
    io::DecodedImageRgba &image,
    contract::ImageAlphaType alpha_type,
    contract::ImageColorSpace color_space) noexcept {
    const auto premultiply =
        alpha_type == contract::ImageAlphaType::straight &&
        color_space != contract::ImageColorSpace::data;
    const auto ignore =
        alpha_type == contract::ImageAlphaType::ignore;
    if (image.storage ==
        io::DecodedImageStorage::unorm8) {
        for (std::size_t offset = 0u;
             offset < image.unorm8_pixels.size();
             offset += 4u) {
            if (premultiply) {
                const auto alpha =
                    static_cast<std::uint32_t>(
                        image.unorm8_pixels[offset + 3u]);
                for (std::size_t channel = 0u;
                     channel < 3u;
                     ++channel) {
                    const auto value =
                        static_cast<std::uint32_t>(
                            image.unorm8_pixels[offset + channel]);
                    image.unorm8_pixels[offset + channel] =
                        static_cast<std::uint8_t>(
                            (value * alpha) / 255u);
                }
            } else if (ignore) {
                image.unorm8_pixels[offset + 3u] = 255u;
            }
        }
        return;
    }
    for (std::size_t offset = 0u;
         offset < image.float_pixels.size();
         offset += 4u) {
        auto &alpha = image.float_pixels[offset + 3u];
        if (ignore) {
            alpha = 1.0f;
        } else if (premultiply) {
            image.float_pixels[offset] *= alpha;
            image.float_pixels[offset + 1u] *= alpha;
            image.float_pixels[offset + 2u] *= alpha;
        }
    }
}

}// namespace psycles::luisa_backend::detail
