#include <psycles/luisa/cycles_bsdf_tables.h>

namespace psycles::luisa_backend::detail {

using namespace luisa::compute;

Float cycles_table_1d(const CyclesBsdfTableReader &reader, Float x,
                      Expr<std::uint32_t> offset, std::uint32_t size) noexcept {
    const auto coordinate = clamp(x, 0.0f, 1.0f) * static_cast<float>(size - 1u);
    const auto index = min(cast<luisa::uint>(coordinate), size - 1u);
    const auto next = min(index + 1u, size - 1u);
    const auto t = coordinate - cast<float>(index);
    const auto data0 = reader.cycles_bsdf_data(index + offset);
    const auto data1 = reader.cycles_bsdf_data(next + offset);
    return lerp(data0, data1, t);
}

Float cycles_table_2d(const CyclesBsdfTableReader &reader, Float x, Float y,
                      Expr<std::uint32_t> offset, std::uint32_t x_size,
                      std::uint32_t y_size) noexcept {
    const auto coordinate =
        clamp(y, 0.0f, 1.0f) * static_cast<float>(y_size - 1u);
    const auto index = min(cast<luisa::uint>(coordinate), y_size - 1u);
    const auto next = min(index + 1u, y_size - 1u);
    const auto t = coordinate - cast<float>(index);
    const auto data0 =
        cycles_table_1d(reader, x, offset + x_size * index, x_size);
    const auto data1 = cycles_table_1d(reader, x, offset + x_size * next, x_size);
    return lerp(data0, data1, t);
}

Float cycles_table_3d(const CyclesBsdfTableReader &reader, Float x, Float y,
                      Float z, Expr<std::uint32_t> offset, std::uint32_t x_size,
                      std::uint32_t y_size, std::uint32_t z_size) noexcept {
    const auto coordinate =
        clamp(z, 0.0f, 1.0f) * static_cast<float>(z_size - 1u);
    const auto index = min(cast<luisa::uint>(coordinate), z_size - 1u);
    const auto next = min(index + 1u, z_size - 1u);
    const auto t = coordinate - cast<float>(index);
    const auto slice_stride = x_size * y_size;
    const auto data0 = cycles_table_2d(
        reader, x, y, offset + slice_stride * index, x_size, y_size);
    const auto data1 = cycles_table_2d(reader, x, y, offset + slice_stride * next,
                                       x_size, y_size);
    return lerp(data0, data1, t);
}

}// namespace psycles::luisa_backend::detail
