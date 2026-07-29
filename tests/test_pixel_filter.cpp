#include <psycles/sampling/pixel_filter.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

using psycles::contract::PixelFilter;
using psycles::sampling::make_pixel_filter_table;
using psycles::sampling::pixel_filter_table_size;
using psycles::sampling::PixelFilterTable;

[[nodiscard]] bool approximately_equal(float actual,
                                       float expected,
                                       float tolerance = 2.0e-5f) noexcept {
    return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool check_table(const PixelFilterTable &table,
                               float expected_minimum,
                               float expected_maximum,
                               const char *name) {
    constexpr auto half = (pixel_filter_table_size - 1u) / 2u;
    if (!std::all_of(table.begin(),
                     table.end(),
                     [](float value) { return std::isfinite(value); }) ||
        !approximately_equal(table[half], 0.5f) ||
        !approximately_equal(table.front(), expected_minimum) ||
        !approximately_equal(table[pixel_filter_table_size - 2u],
                             expected_maximum) ||
        table.back() != 0.0f) {
        std::cerr << name << " inverse-CDF endpoint contract failed\n";
        return false;
    }
    for (std::size_t i = 0u; i <= half; ++i) {
        if (!approximately_equal(table[half - i] + table[half + i], 1.0f) ||
            (i != 0u && (table[half - i] > table[half - i + 1u] ||
                         table[half + i] < table[half + i - 1u]))) {
            std::cerr << name
                      << " symmetric monotonic inverse-CDF invariant "
                         "failed at "
                      << i << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    const auto box = make_pixel_filter_table(PixelFilter::box, 1.0f);
    if (!check_table(box, 0.0f, 1.0f, "BOX") ||
        !approximately_equal(box[510u], 0.5f - 0.5f / 511.0f) ||
        !approximately_equal(box[512u], 0.5f + 0.5f / 511.0f)) {
        std::cerr << "BOX table no longer preserves Cycles' even-table "
                     "sample positions\n";
        return EXIT_FAILURE;
    }

    // Cycles expands the configured width by three for Gaussian and by two
    // for Blackman-Harris before constructing the symmetric table.
    const auto gaussian = make_pixel_filter_table(PixelFilter::gaussian, 2.25f);
    if (!check_table(gaussian,
                     0.5f - 0.5f * (2.25f * 3.0f),
                     0.5f + 0.5f * (2.25f * 3.0f),
                     "GAUSSIAN")) {
        return EXIT_FAILURE;
    }

    const auto blackman_harris =
        make_pixel_filter_table(PixelFilter::blackman_harris, 1.5f);
    if (!check_table(blackman_harris, -1.0f, 2.0f, "BLACKMAN_HARRIS")) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
