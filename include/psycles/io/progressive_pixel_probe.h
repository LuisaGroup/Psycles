#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <psycles/contract/render.h>

namespace psycles::io {

struct PixelPassCapture {
    contract::PassRequest pass;
    std::vector<float> values;
};

// Captures one absolute raster pixel from every pass tile written by a render
// session. The requested coordinate uses OutputSink's upper-left raster
// convention, rather than Cycles' lower-left film convention.
class PixelOutputSink final : public contract::OutputSink {

private:
    std::uint32_t _pixel_x{};
    std::uint32_t _pixel_y{};
    std::vector<PixelPassCapture> _passes;
    bool _cancelled{false};

public:
    PixelOutputSink(
        std::uint32_t pixel_x,
        std::uint32_t pixel_y) noexcept;

    void begin(const contract::RenderSettings &settings) override;
    void write(const contract::PassTile &tile) override;
    void end(bool cancelled) override;

    [[nodiscard]] bool cancelled() const noexcept {
        return _cancelled;
    }

    [[nodiscard]] const std::vector<PixelPassCapture> &
    passes() const noexcept {
        return _passes;
    }
};

struct ProgressivePassDelta {
    contract::PassRequest pass;
    std::vector<double> values;
};

struct ProgressiveSampleChunk {
    std::uint32_t sample_first{};
    std::uint32_t sample_count{};
    std::vector<ProgressivePassDelta> passes;
};

// Reconstructs delta(progressive_output * rendered_sample_count) for each
// pass. This is an exact additive sample-chunk contribution only for linear
// outputs such as Combined. Passes normalized by another accumulated pass
// retain the stated scaled-output-delta semantics and must not be interpreted
// as independent per-sample radiance.
class ProgressivePixelAccumulator {

private:
    std::uint32_t _rendered_samples{};
    std::vector<contract::PassRequest> _layout;
    std::vector<std::vector<double>> _previous_sums;

public:
    void reset() noexcept;

    [[nodiscard]] std::uint32_t rendered_samples() const noexcept {
        return _rendered_samples;
    }

    // Validation is transactional: an invalid chunk returns nullopt without
    // changing the accumulated layout, sample count, or previous sums.
    [[nodiscard]] std::optional<ProgressiveSampleChunk> append(
        std::uint32_t sample_first,
        std::uint32_t sample_count,
        const std::vector<PixelPassCapture> &captures);
};

}// namespace psycles::io
