#pragma once

#include <psycles/contract/scene.h>
#include <psycles/io/image.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace psycles::luisa_backend::detail {

enum class DecodedImageChannelMode : std::uint8_t {
    rgba = 0u,
    gray_opaque = 1u,
    gray_alpha = 2u,
};

struct CompactDecodedImage {
    std::uint32_t width{};
    std::uint32_t height{};
    bool is_unorm8{false};
    std::uint8_t channels{4u};
    DecodedImageChannelMode channel_mode{DecodedImageChannelMode::rgba};
    std::vector<std::uint8_t> byte_pixels;
    std::vector<float> float_pixels;
};

[[nodiscard]] std::optional<io::ProbedImageInfo>
probe_scene_image_info(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint);

[[nodiscard]] std::optional<CompactDecodedImage>
decode_scene_image_gray8(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint);

[[nodiscard]] CompactDecodedImage
compact_decoded_scene_image(
    io::DecodedImageRgba &&image,
    bool allow_opaque_gray) noexcept;

[[nodiscard]] std::optional<io::DecodedImageRgba>
decode_scene_image(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint);

void apply_scene_image_alpha(
    io::DecodedImageRgba &image,
    contract::ImageAlphaType alpha_type,
    contract::ImageColorSpace color_space) noexcept;

}// namespace psycles::luisa_backend::detail
