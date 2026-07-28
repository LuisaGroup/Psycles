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
