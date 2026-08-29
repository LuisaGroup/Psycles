#include <psycles/io/image.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/filesystem.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace psycles::io {
namespace {

[[nodiscard]] bool fail(
    std::string message,
    std::string *error) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

[[nodiscard]] std::string pass_name(
    const PassImage &image) {
    return image.pass.name;
}

[[nodiscard]] std::string component_name(
    const PassImage &image,
    std::uint32_t component) {
    constexpr std::array rgba{"R", "G", "B", "A"};
    constexpr std::array xyz{"X", "Y", "Z"};
    if (image.pass.kind == contract::PassKind::normal &&
        component < xyz.size()) {
        return xyz[component];
    }
    if (image.pass.kind == contract::PassKind::depth &&
        component == 0u) {
        return "Z";
    }
    if (image.channels <= rgba.size()) {
        if (image.channels <= 2u) {
            return xyz[component];
        }
        return rgba[component];
    }
    return "C" + std::to_string(component);
}

[[nodiscard]] bool source_is_uint8(
    const OIIO::ImageSpec &specification,
    int channel_count) noexcept {
    for (auto channel = 0; channel < channel_count; ++channel) {
        const auto format =
            static_cast<std::size_t>(channel) >=
                    specification.channelformats.size()
                ? specification.format
                : specification.channelformats[
                      static_cast<std::size_t>(channel)];
        if (format != OIIO::TypeDesc::UINT8) {
            return false;
        }
    }
    return true;
}

template<typename Source, typename Destination>
void expand_rgba(
    std::span<const Source> source,
    int source_channels,
    std::span<Destination> destination,
    Destination alpha_one) noexcept {
    const auto pixel_count =
        destination.size() / 4u;
    const auto channels =
        static_cast<std::size_t>(source_channels);
    for (std::size_t pixel = 0u;
         pixel < pixel_count; ++pixel) {
        const auto source_offset = pixel * channels;
        const auto destination_offset = pixel * 4u;
        if (source_channels == 1) {
            destination[destination_offset] =
                source[source_offset];
            destination[destination_offset + 1u] =
                source[source_offset];
            destination[destination_offset + 2u] =
                source[source_offset];
            destination[destination_offset + 3u] =
                alpha_one;
        } else if (source_channels == 2) {
            destination[destination_offset] =
                source[source_offset];
            destination[destination_offset + 1u] =
                source[source_offset];
            destination[destination_offset + 2u] =
                source[source_offset];
            destination[destination_offset + 3u] =
                source[source_offset + 1u];
        } else {
            destination[destination_offset] =
                source[source_offset];
            destination[destination_offset + 1u] =
                source[source_offset + 1u];
            destination[destination_offset + 2u] =
                source[source_offset + 2u];
            destination[destination_offset + 3u] =
                source_channels == 4
                    ? source[source_offset + 3u]
                    : alpha_one;
        }
    }
}

}// namespace

bool decode_image_rgba(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint,
    DecodedImageRgba &image,
    std::string *error) {
    image = {};
    if (error != nullptr) {
        error->clear();
    }
    if (encoded.empty()) {
        return fail("the encoded image payload is empty", error);
    }

    OIIO::Filesystem::IOMemReader reader{
        encoded.data(), encoded.size()};
    auto input = OIIO::ImageInput::open(
        std::string{filename_hint}, nullptr, &reader);
    if (!input) {
        return fail(
            "OpenImageIO could not identify the encoded image",
            error);
    }
    const auto specification = input->spec();
    if (specification.width <= 0 ||
        specification.height <= 0 ||
        specification.nchannels <= 0) {
        return fail("the encoded image has an invalid extent", error);
    }
    const auto width =
        static_cast<std::size_t>(specification.width);
    const auto height =
        static_cast<std::size_t>(specification.height);
    if (width > std::numeric_limits<std::size_t>::max() / height ||
        width * height >
            std::numeric_limits<std::size_t>::max() / 4u) {
        return fail("the encoded image is too large", error);
    }
    const auto pixel_count = width * height;
    const auto source_channels =
        std::min(specification.nchannels, 4);
    image.width = static_cast<std::uint32_t>(width);
    image.height = static_cast<std::uint32_t>(height);
    if (source_is_uint8(
            specification,
            source_channels)) {
        image.storage =
            DecodedImageStorage::unorm8;
        std::vector<std::uint8_t> source_pixels(
            pixel_count *
            static_cast<std::size_t>(source_channels));
        if (!input->read_image(
                0, 0, 0, source_channels,
                OIIO::TypeDesc::UINT8,
                source_pixels.data())) {
            auto message = input->geterror();
            return fail(
                message.empty()
                    ? "OpenImageIO could not read the encoded image"
                    : std::move(message),
                error);
        }
        image.unorm8_pixels.resize(
            pixel_count * 4u);
        expand_rgba<std::uint8_t, std::uint8_t>(
            std::span<const std::uint8_t>{source_pixels},
            source_channels,
            std::span<std::uint8_t>{image.unorm8_pixels},
            std::uint8_t{255u});
    } else {
        image.storage =
            DecodedImageStorage::float32;
        std::vector<float> source_pixels(
            pixel_count *
            static_cast<std::size_t>(source_channels));
        if (!input->read_image(
                0, 0, 0, source_channels,
                OIIO::TypeDesc::FLOAT,
                source_pixels.data())) {
            auto message = input->geterror();
            return fail(
                message.empty()
                    ? "OpenImageIO could not read the encoded image"
                    : std::move(message),
                error);
        }
        image.float_pixels.resize(
            pixel_count * 4u);
        expand_rgba<float, float>(
            std::span<const float>{source_pixels},
            source_channels,
            std::span<float>{image.float_pixels},
            1.0f);
    }
    return true;
}

bool decode_image_rgba8(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint,
    DecodedImageRgba8 &image,
    std::string *error) {
    DecodedImageRgba decoded;
    if (!decode_image_rgba(
            encoded,
            filename_hint,
            decoded,
            error)) {
        image = {};
        return false;
    }
    image.width = decoded.width;
    image.height = decoded.height;
    if (decoded.storage ==
        DecodedImageStorage::unorm8) {
        image.pixels =
            std::move(decoded.unorm8_pixels);
        return true;
    }
    image.pixels.resize(
        decoded.float_pixels.size());
    std::transform(
        decoded.float_pixels.begin(),
        decoded.float_pixels.end(),
        image.pixels.begin(),
        [](float value) noexcept {
            const auto scaled =
                std::clamp(
                    value, 0.0f, 1.0f) *
                    255.0f;
            return static_cast<std::uint8_t>(
                scaled + 0.5f);
        });
    return true;
}

bool write_multilayer_exr(
    std::span<const PassImage> images,
    const std::filesystem::path &path,
    std::string_view view_layer,
    std::string *error) {
    if (error != nullptr) {
        error->clear();
    }
    if (images.empty()) {
        return fail("no render passes were supplied", error);
    }
    if (view_layer.empty()) {
        return fail("the OpenEXR view-layer name is empty", error);
    }
    if (path.extension() != ".exr" &&
        path.extension() != ".EXR") {
        return fail("the output path does not have an .exr extension", error);
    }

    const auto extent = images.front().extent;
    if (extent.width == 0u || extent.height == 0u ||
        extent.width >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max()) ||
        extent.height >
            static_cast<std::uint32_t>(
                std::numeric_limits<int>::max())) {
        return fail("the render extent is invalid for OpenEXR", error);
    }

    std::size_t total_channels = 0u;
    std::unordered_set<std::string> channel_names;
    std::vector<std::string> ordered_channel_names;
    for (const auto &image : images) {
        if (image.extent.width != extent.width ||
            image.extent.height != extent.height) {
            return fail(
                "all OpenEXR passes must have the same extent",
                error);
        }
        if (image.channels == 0u) {
            return fail(
                "an OpenEXR pass has no channels",
                error);
        }
        const auto pixel_count =
            static_cast<std::size_t>(extent.width) *
            static_cast<std::size_t>(extent.height);
        if (image.channels >
                std::numeric_limits<std::size_t>::max() /
                    pixel_count ||
            image.pixels.size() !=
                pixel_count *
                    static_cast<std::size_t>(image.channels)) {
            return fail(
                "an OpenEXR pass has an invalid pixel buffer",
                error);
        }
        const auto name = pass_name(image);
        if (name.empty()) {
            return fail(
                "an OpenEXR pass has an empty name",
                error);
        }
        if (image.channels >
            std::numeric_limits<std::size_t>::max() -
                total_channels) {
            return fail(
                "the OpenEXR channel count overflows size_t",
                error);
        }
        total_channels += image.channels;
        for (std::uint32_t component = 0u;
             component < image.channels;
             ++component) {
            auto channel =
                std::string{view_layer} + "." + name + "." +
                component_name(image, component);
            if (!channel_names.emplace(channel).second) {
                return fail(
                    "duplicate OpenEXR channel name: " + channel,
                    error);
            }
            ordered_channel_names.emplace_back(std::move(channel));
        }
    }
    if (total_channels >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        return fail(
            "the OpenEXR channel count exceeds the format API limit",
            error);
    }

    OIIO::ImageSpec specification{
        static_cast<int>(extent.width),
        static_cast<int>(extent.height),
        static_cast<int>(total_channels),
        OIIO::TypeDesc::FLOAT};
    specification.channelnames = std::move(ordered_channel_names);
    specification.attribute("compression", "zip");
    // Render-pass RGB values use the same scene-linear Rec.709 primaries as
    // Cycles. "scene_linear" is an OCIO role, not a stable interchange
    // identity: with Blender's current config OpenImageIO resolves it to
    // lin_ap1_scene and would therefore mislabel unchanged Rec.709 values.
    specification.attribute(
        "oiio:ColorSpace",
        "lin_rec709_scene");
    specification.attribute(
        "Software",
        "Psycles (LuisaCompute)");

    const auto filename = path.string();
    auto output = OIIO::ImageOutput::create(filename);
    if (!output) {
        auto reason = OIIO::geterror();
        return fail(
            reason.empty()
                ? "OpenImageIO could not create an OpenEXR writer"
                : std::move(reason),
            error);
    }
    if (!output->open(filename, specification)) {
        auto reason = output->geterror();
        return fail(
            reason.empty()
                ? "OpenImageIO could not open the OpenEXR output"
                : std::move(reason),
            error);
    }

    const auto row_floats =
        static_cast<std::size_t>(extent.width) *
        total_channels;
    constexpr std::size_t target_buffer_bytes = 8u * 1024u * 1024u;
    const auto rows_per_batch = std::max<std::size_t>(
        1u,
        std::min<std::size_t>(
            extent.height,
            target_buffer_bytes /
                std::max<std::size_t>(
                    row_floats * sizeof(float), 1u)));
    std::vector<float> interleaved(
        row_floats * rows_per_batch);

    for (std::uint32_t y_begin = 0u;
         y_begin < extent.height;) {
        const auto row_count = std::min<std::size_t>(
            rows_per_batch,
            static_cast<std::size_t>(extent.height - y_begin));
        for (std::size_t local_y = 0u;
             local_y < row_count;
             ++local_y) {
            const auto y =
                static_cast<std::size_t>(y_begin) + local_y;
            for (std::size_t x = 0u; x < extent.width; ++x) {
                auto destination =
                    interleaved.begin() +
                    static_cast<std::ptrdiff_t>(
                        (local_y * extent.width + x) *
                        total_channels);
                for (const auto &image : images) {
                    const auto source =
                        image.pixels.begin() +
                        static_cast<std::ptrdiff_t>(
                            (y * extent.width + x) *
                            image.channels);
                    destination = std::copy_n(
                        source,
                        image.channels,
                        destination);
                }
            }
        }
        const auto y_end =
            y_begin + static_cast<std::uint32_t>(row_count);
        if (!output->write_scanlines(
                static_cast<int>(y_begin),
                static_cast<int>(y_end),
                0,
                OIIO::TypeDesc::FLOAT,
                interleaved.data())) {
            auto reason = output->geterror();
            output->close();
            return fail(
                reason.empty()
                    ? "OpenImageIO could not write OpenEXR scanlines"
                    : std::move(reason),
                error);
        }
        y_begin = y_end;
    }
    if (!output->close()) {
        auto reason = output->geterror();
        return fail(
            reason.empty()
                ? "OpenImageIO could not finalize the OpenEXR output"
                : std::move(reason),
            error);
    }
    return true;
}

}// namespace psycles::io
