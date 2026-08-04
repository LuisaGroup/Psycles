#include <psycles/io/progressive_pixel_probe.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace psycles::io {
namespace {

[[nodiscard]] bool same_pass(
    const contract::PassRequest &lhs,
    const contract::PassRequest &rhs) noexcept {
    return lhs.kind == rhs.kind &&
           lhs.name == rhs.name &&
           lhs.light_group == rhs.light_group &&
           lhs.channels == rhs.channels;
}

}// namespace

PixelOutputSink::PixelOutputSink(
    std::uint32_t pixel_x,
    std::uint32_t pixel_y) noexcept
    : _pixel_x{pixel_x}, _pixel_y{pixel_y} {}

void PixelOutputSink::begin(
    const contract::RenderSettings &) {
    _passes.clear();
    _cancelled = false;
}

void PixelOutputSink::write(
    const contract::PassTile &tile) {
    const auto pixel_end_x =
        static_cast<std::uint64_t>(tile.window.x) +
        tile.window.width;
    const auto pixel_end_y =
        static_cast<std::uint64_t>(tile.window.y) +
        tile.window.height;
    const auto expected =
        static_cast<std::size_t>(tile.window.width) *
        tile.window.height * tile.pass.channels;
    if (_pixel_x < tile.window.x ||
        static_cast<std::uint64_t>(_pixel_x) >= pixel_end_x ||
        _pixel_y < tile.window.y ||
        static_cast<std::uint64_t>(_pixel_y) >= pixel_end_y ||
        tile.pass.channels == 0u ||
        tile.pixels.size() != expected) {
        _cancelled = true;
        return;
    }
    const auto pixel =
        (static_cast<std::size_t>(_pixel_y - tile.window.y) *
             tile.window.width +
         (_pixel_x - tile.window.x)) *
        tile.pass.channels;
    _passes.emplace_back(PixelPassCapture{
        .pass = tile.pass,
        .values = std::vector<float>{
            tile.pixels.begin() +
                static_cast<std::ptrdiff_t>(pixel),
            tile.pixels.begin() +
                static_cast<std::ptrdiff_t>(
                    pixel + tile.pass.channels)}});
}

void PixelOutputSink::end(bool cancelled) {
    _cancelled = _cancelled || cancelled;
}

void ProgressivePixelAccumulator::reset() noexcept {
    _rendered_samples = 0u;
    _layout.clear();
    _previous_sums.clear();
}

std::optional<ProgressiveSampleChunk>
ProgressivePixelAccumulator::append(
    std::uint32_t sample_first,
    std::uint32_t sample_count,
    const std::vector<PixelPassCapture> &captures) {
    if (sample_count == 0u || captures.empty() ||
        sample_count >
            std::numeric_limits<std::uint32_t>::max() -
                _rendered_samples) {
        return std::nullopt;
    }
    for (std::size_t index = 0u;
         index < captures.size();
         ++index) {
        const auto &capture = captures[index];
        if (capture.pass.channels == 0u ||
            capture.values.size() != capture.pass.channels ||
            (!_layout.empty() &&
             (index >= _layout.size() ||
              !same_pass(_layout[index], capture.pass)))) {
            return std::nullopt;
        }
    }
    if (!_layout.empty() && captures.size() != _layout.size()) {
        return std::nullopt;
    }

    auto next_sums = _previous_sums;
    if (_layout.empty()) {
        next_sums.reserve(captures.size());
        for (const auto &capture : captures) {
            next_sums.emplace_back(capture.values.size(), 0.0);
        }
    }
    const auto next_rendered_samples =
        _rendered_samples + sample_count;
    const auto next_rendered_samples_real =
        static_cast<double>(next_rendered_samples);
    ProgressiveSampleChunk record{
        .sample_first = sample_first,
        .sample_count = sample_count,
        .passes = {}};
    record.passes.reserve(captures.size());
    for (std::size_t pass = 0u;
         pass < captures.size();
         ++pass) {
        const auto &capture = captures[pass];
        ProgressivePassDelta delta{
            .pass = capture.pass,
            .values = {}};
        delta.values.reserve(capture.values.size());
        for (std::size_t channel = 0u;
             channel < capture.values.size();
             ++channel) {
            const auto sum =
                static_cast<double>(capture.values[channel]) *
                next_rendered_samples_real;
            delta.values.emplace_back(
                sum - next_sums[pass][channel]);
            next_sums[pass][channel] = sum;
        }
        record.passes.emplace_back(std::move(delta));
    }

    if (_layout.empty()) {
        _layout.reserve(captures.size());
        for (const auto &capture : captures) {
            _layout.emplace_back(capture.pass);
        }
    }
    _previous_sums = std::move(next_sums);
    _rendered_samples = next_rendered_samples;
    return record;
}

}// namespace psycles::io
