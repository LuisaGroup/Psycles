#include "path_tracer_bsdf_tables.h"

#include <psycles/luisa/cycles_bsdf_tables.h>

#include <iterator>

#include "cycles_shader_tables_4_5_10.inl"

namespace psycles::luisa_backend::detail {

luisa::vector<float> make_cycles_bsdf_table_values() {
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
    return values;
}

}// namespace psycles::luisa_backend::detail
