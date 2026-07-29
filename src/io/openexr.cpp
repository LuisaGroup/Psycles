#include <psycles/io/image.h>

#include <OpenImageIO/imageio.h>

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

}// namespace

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
