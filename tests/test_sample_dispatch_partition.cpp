#include "../src/luisa/sample_dispatch_partition.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace {

using psycles::luisa_backend::detail::
    SampleDispatchBatch;
using psycles::luisa_backend::detail::
    SampleDispatchPartition;

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

[[nodiscard]] std::optional<
    std::vector<SampleDispatchBatch>>
collect(
    std::uint32_t first,
    std::uint32_t count,
    std::uint32_t limit) {
    auto partition =
        SampleDispatchPartition::make(first, count, limit);
    if (!partition) {
        return std::nullopt;
    }
    std::vector<SampleDispatchBatch> batches;
    while (const auto batch = partition->next()) {
        batches.emplace_back(*batch);
    }
    require(
        !partition->next().has_value(),
        "an exhausted partition produced another batch");
    return batches;
}

[[nodiscard]] std::optional<
    std::vector<SampleDispatchBatch>>
collect_volume_guided(
    std::uint32_t first,
    std::uint32_t count,
    std::uint32_t limit,
    std::uint32_t completed,
    std::uint32_t total) {
    auto partition =
        SampleDispatchPartition::
            make_volume_guided(
                first,
                count,
                limit,
                completed,
                total);
    if (!partition) {
        return std::nullopt;
    }
    std::vector<SampleDispatchBatch> batches;
    while (const auto batch =
               partition->next()) {
        batches.emplace_back(*batch);
    }
    return batches;
}

void require_exact_partition(
    std::uint32_t first,
    std::uint32_t count,
    std::uint32_t limit) {
    const auto batches = collect(first, count, limit);
    require(batches.has_value(), "valid partition rejected");

    auto cursor = static_cast<std::uint64_t>(first);
    auto covered = std::uint64_t{0u};
    for (const auto batch : *batches) {
        require(
            batch.count > 0u,
            "partition emitted an empty batch");
        require(
            batch.count <= limit,
            "partition exceeded the dispatch limit");
        require(
            static_cast<std::uint64_t>(batch.first) ==
                cursor,
            "partition has a gap, overlap, or reordering");
        cursor += batch.count;
        covered += batch.count;
    }
    require(
        covered == static_cast<std::uint64_t>(count),
        "partition did not cover the requested count");
    require(
        cursor ==
            static_cast<std::uint64_t>(first) +
                static_cast<std::uint64_t>(count),
        "partition ended at the wrong sample");

    const auto expected_batch_count =
        count == 0u ?
            std::uint64_t{0u} :
            (static_cast<std::uint64_t>(count) +
             static_cast<std::uint64_t>(limit) - 1u) /
                static_cast<std::uint64_t>(limit);
    require(
        batches->size() == expected_batch_count,
        "partition is not the minimal bounded cover");
}

}// namespace

int main() {
    require(
        !SampleDispatchPartition::make(0u, 1u, 0u)
             .has_value(),
        "zero dispatch limit was accepted");

    constexpr auto max_sample =
        std::numeric_limits<std::uint32_t>::max();
    require(
        !SampleDispatchPartition::make(
             max_sample - 3u, 4u, 1u)
             .has_value(),
        "overflowing half-open interval was accepted");
    require(
        !SampleDispatchPartition::
             make_volume_guided(
                 0u, 2u, 8u, 7u, 8u)
             .has_value(),
        "volume-guided partition exceeded total samples");

    for (auto first = 0u; first <= 16u; ++first) {
        for (auto count = 0u; count <= 16u; ++count) {
            for (auto limit = 1u; limit <= 16u; ++limit) {
                require_exact_partition(
                    first, count, limit);
            }
        }
    }

    require_exact_partition(0u, 256u, 8u);
    require_exact_partition(
        max_sample - 17u, 17u, 8u);

    const auto quality_batches = collect(0u, 256u, 8u);
    require(
        quality_batches &&
            quality_batches->size() == 32u,
        "256 samples were not split into 32 bounded batches");
    require(
        quality_batches->front().first == 0u &&
            quality_batches->back().first == 248u &&
            quality_batches->back().count == 8u,
        "quality partition endpoints changed");

    const auto guided =
        collect_volume_guided(
            0u, 8u, 8u, 0u, 8u);
    require(
        guided &&
            guided->size() == 4u,
        "power-of-two VSPG schedule was not split");
    constexpr std::array expected_counts{
        1u, 1u, 2u, 4u};
    constexpr std::array expected_filter{
        true, true, true, false};
    auto guided_first = 0u;
    for (std::size_t index = 0u;
         index < expected_counts.size();
         ++index) {
        require(
            (*guided)[index].first ==
                    guided_first &&
                (*guided)[index].count ==
                    expected_counts[index] &&
                (*guided)[index]
                        .filter_volume_guiding ==
                    expected_filter[index],
            "canonical VSPG update schedule changed");
        guided_first +=
            expected_counts[index];
    }

    const auto resumed =
        collect_volume_guided(
            17u, 5u, 8u, 3u, 8u);
    require(
        resumed &&
            resumed->size() == 2u &&
            (*resumed)[0].first == 17u &&
            (*resumed)[0].count == 1u &&
            (*resumed)[0]
                .filter_volume_guiding &&
            (*resumed)[1].first == 18u &&
            (*resumed)[1].count == 4u &&
            !(*resumed)[1]
                 .filter_volume_guiding,
        "resumed VSPG schedule lost its cumulative boundary");

    const auto limited =
        collect_volume_guided(
            0u, 9u, 3u, 0u, 9u);
    constexpr std::array limited_counts{
        1u, 1u, 2u, 3u, 1u, 1u};
    constexpr std::array limited_filter{
        true, true, true, false, true, false};
    require(
        limited &&
            limited->size() ==
                limited_counts.size(),
        "dispatch limit and VSPG boundaries did not compose");
    for (std::size_t index = 0u;
         index < limited_counts.size();
         ++index) {
        require(
            (*limited)[index].count ==
                    limited_counts[index] &&
                (*limited)[index]
                        .filter_volume_guiding ==
                    limited_filter[index],
            "bounded VSPG schedule changed");
    }

    std::cout
        << "All sample-dispatch partition invariants passed.\n";
    return EXIT_SUCCESS;
}
