#include <psycles/io/image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>

namespace psycles::io {

using contract::PassKind;
using contract::PassTile;
using contract::RenderSettings;

void MemoryOutputSink::begin(const RenderSettings &settings) {
    _settings = settings;
    _images.clear();
    _cancelled = false;
    _images.reserve(settings.passes.size());
    for (const auto &pass : settings.passes) {
        const auto count =
            static_cast<std::size_t>(settings.full_extent.width) *
            static_cast<std::size_t>(settings.full_extent.height);
        _images.emplace_back(PassImage{
            .pass = pass,
            .extent = settings.full_extent,
            .channels = pass.channels,
            .pixels = std::vector<float>(
                count * static_cast<std::size_t>(pass.channels))});
    }
}

void MemoryOutputSink::write(const PassTile &tile) {
    auto image = std::find_if(
        _images.begin(),
        _images.end(),
        [&](const PassImage &candidate) {
            return candidate.pass.kind == tile.pass.kind &&
                   candidate.pass.name == tile.pass.name &&
                   candidate.pass.light_group ==
                       tile.pass.light_group;
        });
    if (image == _images.end() ||
        image->channels != tile.pass.channels) {
        return;
    }
    const auto expected =
        static_cast<std::size_t>(tile.window.width) *
        static_cast<std::size_t>(tile.window.height) *
        static_cast<std::size_t>(tile.pass.channels);
    if (tile.pixels.size() != expected) {
        return;
    }
    for (std::uint32_t y = 0u; y < tile.window.height; ++y) {
        const auto source =
            static_cast<std::size_t>(y) * tile.window.width *
            tile.pass.channels;
        const auto destination =
            (static_cast<std::size_t>(tile.window.y + y) *
                 image->extent.width +
             tile.window.x) *
            image->channels;
        std::copy_n(
            tile.pixels.begin() +
                static_cast<std::ptrdiff_t>(source),
            static_cast<std::size_t>(tile.window.width) *
                image->channels,
            image->pixels.begin() +
                static_cast<std::ptrdiff_t>(destination));
    }
}

void MemoryOutputSink::end(bool cancelled) {
    _cancelled = cancelled;
}

const PassImage *MemoryOutputSink::find(
    PassKind kind,
    std::string_view name) const noexcept {
    auto iter = std::find_if(
        _images.begin(),
        _images.end(),
        [&](const PassImage &image) {
            return image.pass.kind == kind &&
                   (name.empty() || image.pass.name == name);
        });
    return iter == _images.end() ? nullptr : &*iter;
}

bool write_ppm(
    const PassImage &image,
    const std::filesystem::path &path,
    PpmWriteOptions options) {
    if (image.extent.width == 0u ||
        image.extent.height == 0u ||
        image.channels < 3u) {
        return false;
    }
    const auto expected =
        static_cast<std::size_t>(image.extent.width) *
        static_cast<std::size_t>(image.extent.height) *
        image.channels;
    if (image.pixels.size() != expected) {
        return false;
    }
    std::ofstream stream{path, std::ios::binary};
    if (!stream) {
        return false;
    }
    stream << "P6\n"
           << image.extent.width << ' '
           << image.extent.height << "\n255\n";
    const auto exposure = std::exp2(options.exposure_stops);
    auto convert = [&](float value) {
        value = std::max(value * exposure, 0.0f);
        if (options.apply_aces_tonemap) {
            value = std::clamp(
                value * (2.51f * value + 0.03f) /
                    (value * (2.43f * value + 0.59f) + 0.14f),
                0.0f,
                1.0f);
        } else {
            value = std::clamp(value, 0.0f, 1.0f);
        }
        if (options.apply_srgb_transfer) {
            value =
                value <= 0.0031308f
                    ? 12.92f * value
                    : 1.055f *
                              std::pow(value, 1.0f / 2.4f) -
                          0.055f;
        }
        return static_cast<unsigned char>(
            std::clamp(
                static_cast<int>(
                    std::lround(value * 255.0f)),
                0,
                255));
    };
    const auto count =
        static_cast<std::size_t>(image.extent.width) *
        static_cast<std::size_t>(image.extent.height);
    for (std::size_t pixel = 0u; pixel < count; ++pixel) {
        const auto base = pixel * image.channels;
        const std::array<unsigned char, 3u> rgb{
            convert(image.pixels[base]),
            convert(image.pixels[base + 1u]),
            convert(image.pixels[base + 2u])};
        stream.write(
            reinterpret_cast<const char *>(rgb.data()),
            static_cast<std::streamsize>(rgb.size()));
    }
    return static_cast<bool>(stream);
}

bool write_pfm(
    const PassImage &image,
    const std::filesystem::path &path) {
    if (image.extent.width == 0u ||
        image.extent.height == 0u ||
        image.channels == 0u) {
        return false;
    }
    const auto expected =
        static_cast<std::size_t>(image.extent.width) *
        static_cast<std::size_t>(image.extent.height) *
        image.channels;
    if (image.pixels.size() != expected) {
        return false;
    }
    const auto color = image.channels >= 3u;
    std::ofstream stream{path, std::ios::binary};
    if (!stream) {
        return false;
    }
    stream << (color ? "PF\n" : "Pf\n")
           << image.extent.width << ' '
           << image.extent.height
           << "\n-1.0\n";
    const auto output_channels = color ? 3u : 1u;
    for (std::uint32_t y = image.extent.height; y > 0u; --y) {
        for (std::uint32_t x = 0u; x < image.extent.width; ++x) {
            const auto base =
                (static_cast<std::size_t>(y - 1u) *
                     image.extent.width +
                 x) *
                image.channels;
            stream.write(
                reinterpret_cast<const char *>(
                    image.pixels.data() +
                    static_cast<std::ptrdiff_t>(base)),
                static_cast<std::streamsize>(
                    output_channels * sizeof(float)));
        }
    }
    return static_cast<bool>(stream);
}

}// namespace psycles::io
