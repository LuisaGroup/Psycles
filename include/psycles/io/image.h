#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <psycles/contract/render.h>

namespace psycles::io {

struct PassImage {
    contract::PassRequest pass;
    contract::ImageExtent extent;
    std::uint32_t channels{};
    std::vector<float> pixels;
};

class MemoryOutputSink final : public contract::OutputSink {

private:
    contract::RenderSettings _settings;
    std::vector<PassImage> _images;
    bool _cancelled{false};

public:
    void begin(const contract::RenderSettings &settings) override;
    void write(const contract::PassTile &tile) override;
    void end(bool cancelled) override;

    [[nodiscard]] bool cancelled() const noexcept {
        return _cancelled;
    }
    [[nodiscard]] const std::vector<PassImage> &images() const noexcept {
        return _images;
    }
    [[nodiscard]] const PassImage *find(
        contract::PassKind kind,
        std::string_view name = {}) const noexcept;
};

struct PpmWriteOptions {
    float exposure_stops{};
    bool apply_aces_tonemap{true};
    bool apply_srgb_transfer{true};
};

struct DecodedImageRgba8 {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> pixels;
};

enum class DecodedImageStorage : std::uint8_t {
    unorm8,
    float32
};

// Precision-preserving decoded texture payload. Exactly one pixel vector is
// populated: native UINT8 sources remain compact UNORM8, while every source
// with a wider or floating-point channel domain is represented as float32.
// This prevents HDR radiance and 16-bit texture data from being silently
// projected onto [0, 1] at the image-I/O boundary.
struct DecodedImageRgba {
    std::uint32_t width{};
    std::uint32_t height{};
    DecodedImageStorage storage{DecodedImageStorage::unorm8};
    std::vector<std::uint8_t> unorm8_pixels;
    std::vector<float> float_pixels;
};

[[nodiscard]] bool write_ppm(
    const PassImage &image,
    const std::filesystem::path &path,
    PpmWriteOptions options = {});

// Portable Float Map output is kept linear and is therefore suitable for
// direct numerical comparison with Cycles' linear EXR passes.
[[nodiscard]] bool write_pfm(
    const PassImage &image,
    const std::filesystem::path &path);

#if defined(PSYCLES_WITH_OPENIMAGEIO)
// Decodes an encoded image from memory without reducing its native numeric
// domain. UINT8 sources use RGBA8; all wider integer and floating-point
// sources use RGBA32F. The filename is a format hint only.
[[nodiscard]] bool decode_image_rgba(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint,
    DecodedImageRgba &image,
    std::string *error = nullptr);

// Decodes an encoded image from memory through OpenImageIO and expands it to
// tightly packed RGBA8. This explicitly quantizing compatibility API is for
// presentation/tests; renderer texture upload uses decode_image_rgba().
[[nodiscard]] bool decode_image_rgba8(
    std::span<const std::uint8_t> encoded,
    std::string_view filename_hint,
    DecodedImageRgba8 &image,
    std::string *error = nullptr);

// Writes every pass into one scanline-oriented, full-float OpenEXR part.
// Channel names follow Cycles' <view-layer>.<pass>.<component> convention,
// allowing the result to be compared or composited without per-pass files.
[[nodiscard]] bool write_multilayer_exr(
    std::span<const PassImage> images,
    const std::filesystem::path &path,
    std::string_view view_layer = "ViewLayer",
    std::string *error = nullptr);
#endif

}// namespace psycles::io
