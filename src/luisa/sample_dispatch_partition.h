#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace psycles::luisa_backend::detail {

struct SampleDispatchBatch {
    std::uint32_t first{};
    std::uint32_t count{};
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

private:
    SampleDispatchPartition(
        std::uint64_t first,
        std::uint64_t end,
        std::uint32_t max_samples_per_dispatch) noexcept
        : _cursor{first},
          _end{end},
          _max_samples_per_dispatch{
              max_samples_per_dispatch} {}

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
            first, end, max_samples_per_dispatch};
    }

    [[nodiscard]] std::optional<SampleDispatchBatch>
    next() noexcept {
        if (_cursor == _end) {
            return std::nullopt;
        }
        const auto batch_count = std::min(
            _end - _cursor,
            static_cast<std::uint64_t>(
                _max_samples_per_dispatch));
        const SampleDispatchBatch batch{
            .first = static_cast<std::uint32_t>(_cursor),
            .count =
                static_cast<std::uint32_t>(batch_count)};
        _cursor += batch_count;
        return batch;
    }
};

}// namespace psycles::luisa_backend::detail
