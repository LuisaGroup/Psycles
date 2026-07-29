#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/pixel_filter.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/sampling/pixel_filter.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::pixel_filter {

[[nodiscard]] inline luisa::compute::Float
sample(const luisa::compute::BufferFloat &table,
       luisa::compute::Float random) noexcept {
    constexpr auto last_index =
        static_cast<std::uint32_t>(sampling::pixel_filter_table_size - 1u);
    const auto coordinate = luisa::compute::clamp(random, 0.0f, 1.0f) *
                            static_cast<float>(last_index);
    const auto index = luisa::compute::min(
        luisa::compute::cast<std::uint32_t>(coordinate), last_index);
    const auto next_index = luisa::compute::min(index + 1u, last_index);
    const auto interpolation = coordinate - luisa::compute::cast<float>(index);
    return luisa::compute::lerp(
        table.read(index), table.read(next_index), interpolation);
}

} // namespace psycles::luisa_backend::pixel_filter
