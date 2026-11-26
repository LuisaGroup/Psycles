#include "path_tracer_image_decode.h"

#include <psycles/core/parallel.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>

#include <stb/stb_image.h>

namespace psycles::luisa_backend::detail {

namespace {

[[nodiscard]] bool png_has_trns_chunk(
    std::span<const std::uint8_t> encoded,
    std::uint8_t &ihdr_color_type_out) noexcept {
    constexpr std::array<std::uint8_t, 8u> kPngSig{
        0x89u, 0x50u, 0x4eu, 0x47u, 0x0du, 0x0au, 0x1au, 0x0au};
    if (encoded.size() < 33u ||
        std::memcmp(encoded.data(), kPngSig.data(), kPngSig.size()) != 0) {
        return false;
    }
    ihdr_color_type_out = encoded[25u];
    std::size_t pos = 8u;
    while (pos + 12u <= encoded.size()) {
        const std::uint32_t len =
            (static_cast<std::uint32_t>(encoded[pos + 0u]) << 24u) |
            (static_cast<std::uint32_t>(encoded[pos + 1u]) << 16u) |
            (static_cast<std::uint32_t>(encoded[pos + 2u]) << 8u) |
            static_cast<std::uint32_t>(encoded[pos + 3u]);
        const std::uint8_t *t = encoded.data() + pos + 4u;
        if (t[0] == 't' && t[1] == 'R' && t[2] == 'N' && t[3] == 'S') {
            return true;
        }
        if ((t[0] == 'I' && t[1] == 'D' && t[2] == 'A' && t[3] == 'T') ||
            (t[0] == 'I' && t[1] == 'E' && t[2] == 'N' && t[3] == 'D')) {
            return false;
        }
        const std::size_t step = static_cast<std::size_t>(len) + 12u;
        if (step > encoded.size() - pos) {
            return false;
        }
        pos += step;
    }
    return false;
}

template<typename Pixel, typename Accum, typename ReduceFn>
void downscale_rgba_box(
    const Pixel *src,
    std::uint32_t src_w,
    std::uint32_t src_h,
    std::uint32_t dst_w,
    std::uint32_t dst_h,
    std::uint64_t step,
    Pixel *dst,
    ReduceFn &&reduce) {
    // Cycles stores images bottom-up and truncates top remainder rows on odd heights.
    const std::uint64_t rem_h =
        (static_cast<std::uint64_t>(src_h) > static_cast<std::uint64_t>(dst_h) * step)
            ? (static_cast<std::uint64_t>(src_h) - static_cast<std::uint64_t>(dst_h) * step)
            : 0ull;
    auto process_row = [&](std::size_t dy) {
        const std::uint64_t y0 = rem_h + static_cast<std::uint64_t>(dy) * step;
        const std::uint64_t y1 = std::min<std::uint64_t>(src_h, y0 + step);
        for (std::uint32_t dx = 0u; dx < dst_w; ++dx) {
            const std::uint64_t x0 = static_cast<std::uint64_t>(dx) * step;
            const std::uint64_t x1 = std::min<std::uint64_t>(src_w, x0 + step);
            std::array<Accum, 4u> sum{};
            std::uint64_t count = 0u;
            for (std::uint64_t y = y0; y < y1; ++y) {
                const auto row_off = static_cast<std::size_t>(y) * src_w;
                for (std::uint64_t x = x0; x < x1; ++x) {
                    const std::size_t off =
                        (row_off + static_cast<std::size_t>(x)) * 4u;
                    for (std::size_t c = 0u; c < 4u; ++c) {
                        sum[c] += static_cast<Accum>(src[off + c]);
                    }
                    ++count;
                }
            }
            if (count == 0u) {
                continue;
            }
            const std::size_t dst_off =
                (dy * dst_w + static_cast<std::size_t>(dx)) * 4u;
            for (std::size_t c = 0u; c < 4u; ++c) {
                dst[dst_off + c] = reduce(sum[c], count);
            }
        }
    };
    if (dst_h >= 64u) {
        psycles::parallel_for(static_cast<std::size_t>(dst_h), process_row);
    } else {
        for (std::uint32_t dy = 0u; dy < dst_h; ++dy) {
            process_row(dy);
        }
    }
}

} // namespace

std::optional<io::ProbedImageInfo>
probe_scene_image_info(
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
    const int encoded_size = static_cast<int>(encoded.size());
    if (stbi_info_from_memory(
            encoded.data(), encoded_size, &width, &height, &channels) != 0 &&
        width > 0 && height > 0) {
        const auto w = static_cast<std::size_t>(width);
        const auto h = static_cast<std::size_t>(height);
        if (w > std::numeric_limits<std::size_t>::max() / h ||
            w * h > std::numeric_limits<std::size_t>::max() / 4u) {
            return std::nullopt;
        }
        const auto is_hdr =
            stbi_is_hdr_from_memory(encoded.data(), encoded_size) != 0;
        const auto is_sixteen_bit =
            stbi_is_16_bit_from_memory(encoded.data(), encoded_size) != 0;
        auto ch = static_cast<std::uint8_t>(std::clamp(channels, 1, 4));
        bool has_alpha = (ch == 2u || ch == 4u);
        std::uint8_t png_ct = 0u;
        if (png_has_trns_chunk(encoded, png_ct)) {
            if (png_ct == 0u) {
                ch = 2u;
                has_alpha = true;
            } else if (png_ct == 2u || png_ct == 3u) {
                ch = 4u;
                has_alpha = true;
            }
        }
        return io::ProbedImageInfo{
            .width = static_cast<std::uint32_t>(width),
            .height = static_cast<std::uint32_t>(height),
            .storage = (is_hdr || is_sixteen_bit)
                           ? io::DecodedImageStorage::float32
                           : io::DecodedImageStorage::unorm8,
            .channels = ch,
            .bit_depth = is_hdr ? static_cast<std::uint8_t>(32u)
                                : (is_sixteen_bit
                                       ? static_cast<std::uint8_t>(16u)
                                       : static_cast<std::uint8_t>(8u)),
            .is_floating_point = is_hdr,
            .has_alpha = has_alpha};
    }
#if defined(PSYCLES_WITH_OPENIMAGEIO)
    io::ProbedImageInfo info;
    if (io::probe_image_info(encoded, filename_hint, info)) {
        return info;
    }
#else
    static_cast<void>(filename_hint);
#endif
    return std::nullopt;
}

std::optional<CompactDecodedImage>
decode_scene_image_gray8(
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
    if (is_hdr || is_sixteen_bit) {
        return std::nullopt;
    }
#if defined(PSYCLES_WITH_OPENIMAGEIO)
    // Preserve libjpeg integer IDCT parity with Blender ImBuf on JPEGs.
    const auto is_jpeg = encoded.size() >= 3u && encoded[0] == 0xffu &&
                         encoded[1] == 0xd8u && encoded[2] == 0xffu;
    if (is_jpeg) {
        io::DecodedImageRgba rgba;
        if (io::decode_image_rgba(
                encoded,
                filename_hint.empty() ? std::string_view{"texture.jpg"}
                                      : filename_hint,
                rgba) &&
            rgba.storage == io::DecodedImageStorage::unorm8) {
            const auto pixel_count =
                static_cast<std::size_t>(rgba.width) *
                static_cast<std::size_t>(rgba.height);
            if (rgba.unorm8_pixels.size() == pixel_count * 4u) {
                CompactDecodedImage gray{
                    .width = rgba.width,
                    .height = rgba.height,
                    .is_unorm8 = true,
                    .channels = 1u,
                    .channel_mode = DecodedImageChannelMode::gray_opaque};
                gray.byte_pixels.resize(pixel_count);
                for (std::size_t i = 0u; i < pixel_count; ++i) {
                    gray.byte_pixels[i] = rgba.unorm8_pixels[i * 4u];
                }
                return gray;
            }
        }
        return std::nullopt;
    }
#else
    static_cast<void>(filename_hint);
#endif
    auto *pixels = stbi_load_from_memory(
        encoded.data(),
        static_cast<int>(encoded.size()),
        &width,
        &height,
        &channels,
        STBI_grey);
    if (pixels != nullptr && width > 0 && height > 0) {
        const auto pixel_count =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height);
        CompactDecodedImage gray{
            .width = static_cast<std::uint32_t>(width),
            .height = static_cast<std::uint32_t>(height),
            .is_unorm8 = true,
            .channels = 1u,
            .channel_mode = DecodedImageChannelMode::gray_opaque};
        gray.byte_pixels.resize(pixel_count);
        std::memcpy(gray.byte_pixels.data(), pixels, pixel_count);
        stbi_image_free(pixels);
        return gray;
    }
    stbi_image_free(pixels);
    return std::nullopt;
}

CompactDecodedImage
compact_decoded_scene_image(
    io::DecodedImageRgba &&image,
    bool allow_opaque_gray) noexcept {
    CompactDecodedImage out{
        .width = image.width,
        .height = image.height,
        .is_unorm8 = image.storage == io::DecodedImageStorage::unorm8};
    const std::size_t pixel_count =
        static_cast<std::size_t>(image.width) *
        static_cast<std::size_t>(image.height);
    if (out.is_unorm8) {
        if (pixel_count != 0u &&
            image.unorm8_pixels.size() == pixel_count * 4u) {
            bool all_gray = true;
            bool all_opaque = true;
            const auto *src = image.unorm8_pixels.data();
            for (std::size_t i = 0u; i < pixel_count; ++i) {
                const auto off = i * 4u;
                const auto r = src[off];
                if (r != src[off + 1u] || r != src[off + 2u]) {
                    all_gray = false;
                    break;
                }
                all_opaque = all_opaque && (src[off + 3u] == 255u);
            }
            if (all_gray && all_opaque && allow_opaque_gray) {
                out.channels = 1u;
                out.channel_mode = DecodedImageChannelMode::gray_opaque;
                out.byte_pixels.resize(pixel_count);
                for (std::size_t i = 0u; i < pixel_count; ++i) {
                    out.byte_pixels[i] = src[i * 4u];
                }
                return out;
            }
            if (all_gray) {
                out.channels = 2u;
                out.channel_mode = DecodedImageChannelMode::gray_alpha;
                out.byte_pixels.resize(pixel_count * 2u);
                for (std::size_t i = 0u; i < pixel_count; ++i) {
                    out.byte_pixels[i * 2u] = src[i * 4u];
                    out.byte_pixels[i * 2u + 1u] = src[i * 4u + 3u];
                }
                return out;
            }
        }
        out.channels = 4u;
        out.channel_mode = DecodedImageChannelMode::rgba;
        out.byte_pixels = std::move(image.unorm8_pixels);
        return out;
    }

    if (pixel_count != 0u && image.float_pixels.size() == pixel_count * 4u) {
        bool all_gray = true;
        bool all_opaque = true;
        const auto *src = image.float_pixels.data();
        for (std::size_t i = 0u; i < pixel_count; ++i) {
            const auto off = i * 4u;
            const float r = src[off];
            if (r != src[off + 1u] || r != src[off + 2u]) {
                all_gray = false;
                break;
            }
            all_opaque = all_opaque && (src[off + 3u] == 1.0f);
        }
        if (all_gray && all_opaque && allow_opaque_gray) {
            out.channels = 1u;
            out.channel_mode = DecodedImageChannelMode::gray_opaque;
            out.float_pixels.resize(pixel_count);
            for (std::size_t i = 0u; i < pixel_count; ++i) {
                out.float_pixels[i] = src[i * 4u];
            }
            return out;
        }
        if (all_gray) {
            out.channels = 2u;
            out.channel_mode = DecodedImageChannelMode::gray_alpha;
            out.float_pixels.resize(pixel_count * 2u);
            for (std::size_t i = 0u; i < pixel_count; ++i) {
                out.float_pixels[i * 2u] = src[i * 4u];
                out.float_pixels[i * 2u + 1u] = src[i * 4u + 3u];
            }
            return out;
        }
    }
    out.channels = 4u;
    out.channel_mode = DecodedImageChannelMode::rgba;
    out.float_pixels = std::move(image.float_pixels);
    return out;
}

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
    // Blender ImBuf uses libjpeg's default integer IDCT. STB is a different
    // decoder: even an identical JPEG payload can change texture bytes and
    // therefore closure weights, normals and subsequent path decisions.
    // Use the existing libjpeg-backed loader before STB for JPEG resources.
    const auto is_jpeg = encoded.size() >= 3u && encoded[0] == 0xffu &&
                         encoded[1] == 0xd8u && encoded[2] == 0xffu;
    if (is_hdr || is_sixteen_bit || is_jpeg) {
        io::DecodedImageRgba image;
        if (io::decode_image_rgba(
                encoded, is_jpeg ? std::string_view{"texture.jpg"} : filename_hint,
                image)) {
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

std::uint32_t compute_texture_limit_shift(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t texture_limit) noexcept {
    if (texture_limit == 0u || width == 0u || height == 0u) {
        return 0u;
    }
    const auto max_size = static_cast<std::uint64_t>(std::max(width, height));
    const auto limit = static_cast<std::uint64_t>(texture_limit);
    std::uint32_t shift = 0u;
    while (shift < 31u && max_size > (limit << shift)) {
        ++shift;
    }
    return shift;
}

std::pair<std::uint32_t, std::uint32_t> compute_texture_limit_extent(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t texture_limit) noexcept {
    if (width == 0u || height == 0u) {
        return {width, height};
    }
    const std::uint32_t shift =
        compute_texture_limit_shift(width, height, texture_limit);
    if (shift == 0u) {
        return {width, height};
    }
    return {
        std::max(1u, width >> shift),
        std::max(1u, height >> shift)};
}

void downscale_scene_image_to_limit(
    io::DecodedImageRgba &image,
    std::uint32_t texture_limit) {
    const std::uint32_t shift =
        compute_texture_limit_shift(image.width, image.height, texture_limit);
    if (shift == 0u) {
        return;
    }
    const auto [dst_w, dst_h] =
        compute_texture_limit_extent(image.width, image.height, texture_limit);
    const std::size_t expected_src =
        static_cast<std::size_t>(image.width) * image.height * 4u;
    const std::uint64_t step = 1ull << shift;
    if (image.storage == io::DecodedImageStorage::unorm8) {
        if (image.unorm8_pixels.size() != expected_src) {
            return;
        }
        std::vector<std::uint8_t> dst(
            static_cast<std::size_t>(dst_w) * dst_h * 4u);
        downscale_rgba_box<std::uint8_t, std::uint64_t>(
            image.unorm8_pixels.data(),
            image.width,
            image.height,
            dst_w,
            dst_h,
            step,
            dst.data(),
            [](std::uint64_t sum, std::uint64_t count) noexcept {
                return static_cast<std::uint8_t>((sum + count / 2u) / count);
            });
        image.unorm8_pixels = std::move(dst);
    } else {
        if (image.float_pixels.size() != expected_src) {
            return;
        }
        std::vector<float> dst(
            static_cast<std::size_t>(dst_w) * dst_h * 4u);
        downscale_rgba_box<float, double>(
            image.float_pixels.data(),
            image.width,
            image.height,
            dst_w,
            dst_h,
            step,
            dst.data(),
            [](double sum, std::uint64_t count) noexcept {
                return static_cast<float>(sum / static_cast<double>(count));
            });
        image.float_pixels = std::move(dst);
    }
    image.width = dst_w;
    image.height = dst_h;
}

}// namespace psycles::luisa_backend::detail
