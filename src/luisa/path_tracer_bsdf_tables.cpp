#include "path_tracer_bsdf_tables.h"

#include <psycles/luisa/cycles_bsdf_tables.h>

#include <iterator>

#include <luisa/core/logging.h>

#include "cycles_shader_tables_4_5_10.inl"
#include "cycles_thin_film_table_5_2_1.h"

namespace psycles::luisa_backend::detail {

luisa::vector<float> make_cycles_bsdf_table_values(
    const contract::ShaderColorSpace &color_space) {
    using namespace cycles45_tables;
    static_assert(std::size(table_ggx_E) == ggx_e_size);
    static_assert(std::size(table_ggx_Eavg) == ggx_eavg_size);
    static_assert(std::size(table_ggx_glass_E) == ggx_glass_e_size);
    static_assert(std::size(table_ggx_glass_Eavg) == ggx_glass_eavg_size);
    static_assert(std::size(table_ggx_glass_inv_E) == ggx_glass_inv_e_size);
    static_assert(
        std::size(table_ggx_glass_inv_Eavg) == ggx_glass_inv_eavg_size);
    static_assert(std::size(table_sheen_ltc) == sheen_ltc_size);
    static_assert(
        std::size(table_ggx_gen_schlick_ior_s) ==
        ggx_gen_schlick_ior_s_size);
    static_assert(
        std::size(table_ggx_gen_schlick_s) == ggx_gen_schlick_s_size);
    static_assert(
        std::size(cycles52_thin_film_tables::table_thin_film_cmf) ==
        thin_film_table_size);

    luisa::vector<float> values;
    values.reserve(total_size);
    const auto append = [&values](const auto &table) noexcept {
        values.insert(values.end(), std::begin(table), std::end(table));
    };
    append(table_ggx_E);
    append(table_ggx_Eavg);
    append(table_ggx_glass_E);
    append(table_ggx_glass_Eavg);
    append(table_ggx_glass_inv_E);
    append(table_ggx_glass_inv_Eavg);
    append(table_sheen_ltc);
    append(table_ggx_gen_schlick_ior_s);
    append(table_ggx_gen_schlick_s);

    const auto transform = [&color_space](const float *xyz) noexcept {
        const auto dot_row = [xyz](Vec3f row) noexcept {
            return row.x * xyz[0] + row.y * xyz[1] + row.z * xyz[2];
        };
        return Vec3f{
            dot_row(color_space.xyz_to_r),
            dot_row(color_space.xyz_to_g),
            dot_row(color_space.xyz_to_b)};
    };
    const auto &cmf = cycles52_thin_film_tables::table_thin_film_cmf;
    const auto dc = transform(cmf[0]);
    const Vec3f normalization{
        1.0f / dc.x,
        1.0f / dc.y,
        1.0f / dc.z};
    const auto channel = [](Vec3f value,
                            std::uint32_t index) noexcept {
        return index == 0u ? value.x : index == 1u ? value.y : value.z;
    };
    for (auto component = std::uint32_t{0u};
         component < thin_film_channel_count;
         ++component) {
        const auto imaginary = component >= 3u;
        const auto rgb_channel = component % 3u;
        for (auto index = std::uint32_t{0u};
             index < thin_film_table_size;
             ++index) {
            const auto rgb = transform(
                cmf[index] + (imaginary ? 3u : 0u));
            values.emplace_back(
                channel(rgb, rgb_channel) *
                channel(normalization, rgb_channel));
        }
    }
    LUISA_ASSERT(values.size() == total_size,
                 "Cycles BSDF table ABI size mismatch.");
    return values;
}

}// namespace psycles::luisa_backend::detail
