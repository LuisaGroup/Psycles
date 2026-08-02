#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace psycles::luisa_backend::detail {

struct SampleDispatchBatch {
    std::uint32_t first{};
    std::uint32_t count{};
    bool filter_volume_guiding{};
};

struct PixelRowDispatchBatch {
    std::uint32_t first_row{};
    std::uint32_t row_count{};
    std::uint32_t first_pixel{};
    std::uint32_t pixel_count{};
};

// Splits a rectangular dispatch into complete, contiguous row bands while
// bounding width * rows * samples. Complete rows keep film coordinates and
// buffer views contiguous, so tiling does not alter sample identities.
class PixelRowDispatchPartition final {

private:
    std::uint32_t _width{};
    std::uint32_t _height{};
    std::uint32_t _cursor{};
    std::uint32_t _max_rows{};

    PixelRowDispatchPartition(
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t max_rows) noexcept
        : _width{width},
          _height{height},
          _max_rows{max_rows} {}

public:
    [[nodiscard]] static std::optional<
        PixelRowDispatchPartition>
    make(
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t samples,
        std::uint32_t max_pixel_samples) noexcept {
        if (width == 0u || height == 0u || samples == 0u ||
            max_pixel_samples == 0u) {
            return std::nullopt;
        }
        const auto pixel_count =
            static_cast<std::uint64_t>(width) *
            static_cast<std::uint64_t>(height);
        if (pixel_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max())) {
            return std::nullopt;
        }
        const auto work_per_row =
            static_cast<std::uint64_t>(width) *
            static_cast<std::uint64_t>(samples);
        const auto bounded_rows = std::max<std::uint64_t>(
            1u,
            static_cast<std::uint64_t>(max_pixel_samples) /
                work_per_row);
        return PixelRowDispatchPartition{
            width,
            height,
            static_cast<std::uint32_t>(std::min<std::uint64_t>(
                height, bounded_rows))};
    }

    [[nodiscard]] std::optional<PixelRowDispatchBatch>
    next() noexcept {
        if (_cursor == _height) {
            return std::nullopt;
        }
        const auto rows = std::min(
            _height - _cursor, _max_rows);
        const PixelRowDispatchBatch batch{
            .first_row = _cursor,
            .row_count = rows,
            .first_pixel = _cursor * _width,
            .pixel_count = rows * _width};
        _cursor += rows;
        return batch;
    }
};

// A valid partition of [first, first + count) satisfies:
//
//   B[0].first == first
//   B[i + 1].first == B[i].first + B[i].count
//   0 < B[i].count <= max_samples_per_dispatch
//   B[last].first + B[last].count == first + count
//
// These invariants make the batches an exact, ordered cover of the requested
// sample interval. They preserve the device sampler's sequence while bounding
// the amount of work submitted in one synchronized dispatch.
class SampleDispatchPartition final {

private:
    std::uint64_t _cursor{};
    std::uint64_t _end{};
    std::uint32_t _max_samples_per_dispatch{};
    std::uint64_t _completed_samples{};
    std::uint64_t _total_samples{};
    bool _split_volume_guiding{};

private:
    SampleDispatchPartition(
        std::uint64_t first,
        std::uint64_t end,
        std::uint32_t max_samples_per_dispatch,
        std::uint64_t completed_samples,
        std::uint64_t total_samples,
        bool split_volume_guiding) noexcept
        : _cursor{first},
          _end{end},
          _max_samples_per_dispatch{
              max_samples_per_dispatch},
          _completed_samples{
              completed_samples},
          _total_samples{total_samples},
          _split_volume_guiding{
              split_volume_guiding} {}

public:
    [[nodiscard]] static std::optional<
        SampleDispatchPartition>
    make(
        std::uint32_t first,
        std::uint32_t count,
        std::uint32_t max_samples_per_dispatch) noexcept {
        const auto end =
            static_cast<std::uint64_t>(first) +
            static_cast<std::uint64_t>(count);
        if (max_samples_per_dispatch == 0u ||
            end > static_cast<std::uint64_t>(
                      std::numeric_limits<
                          std::uint32_t>::max())) {
            return std::nullopt;
        }
        return SampleDispatchPartition{
            first,
            end,
            max_samples_per_dispatch,
            0u,
            0u,
            false};
    }

    // Cycles updates VSPG history after cumulative sample counts 1, 2, 4,
    // 8, ... while more samples remain. This constructor preserves the
    // requested sample interval but also makes every such update point a
    // dispatch boundary, independent of the ordinary batch-size limit.
    [[nodiscard]] static std::optional<
        SampleDispatchPartition>
    make_volume_guided(
        std::uint32_t first,
        std::uint32_t count,
        std::uint32_t max_samples_per_dispatch,
        std::uint32_t completed_samples,
        std::uint32_t total_samples) noexcept {
        const auto end =
            static_cast<std::uint64_t>(first) +
            static_cast<std::uint64_t>(count);
        const auto completed_end =
            static_cast<std::uint64_t>(
                completed_samples) +
            static_cast<std::uint64_t>(count);
        if (max_samples_per_dispatch == 0u ||
            total_samples == 0u ||
            end > static_cast<std::uint64_t>(
                      std::numeric_limits<
                          std::uint32_t>::max()) ||
            completed_end >
                static_cast<std::uint64_t>(
                    total_samples)) {
            return std::nullopt;
        }
        return SampleDispatchPartition{
            first,
            end,
            max_samples_per_dispatch,
            completed_samples,
            total_samples,
            true};
    }

    [[nodiscard]] std::optional<SampleDispatchBatch>
    next() noexcept {
        if (_cursor == _end) {
            return std::nullopt;
        }
        auto batch_count = std::min(
            _end - _cursor,
            static_cast<std::uint64_t>(
                _max_samples_per_dispatch));
        if (_split_volume_guiding) {
            auto next_power_of_two =
                std::uint64_t{1u};
            while (next_power_of_two <=
                   _completed_samples) {
                next_power_of_two <<= 1u;
            }
            batch_count = std::min(
                batch_count,
                next_power_of_two -
                    _completed_samples);
        }
        const auto completed_after =
            _completed_samples +
            batch_count;
        const auto power_of_two =
            completed_after != 0u &&
            (completed_after &
             (completed_after - 1u)) == 0u;
        const SampleDispatchBatch batch{
            .first = static_cast<std::uint32_t>(_cursor),
            .count =
                static_cast<std::uint32_t>(batch_count),
            .filter_volume_guiding =
                _split_volume_guiding &&
                power_of_two &&
                completed_after <
                    _total_samples};
        _cursor += batch_count;
        _completed_samples =
            completed_after;
        return batch;
    }
};

}// namespace psycles::luisa_backend::detail
