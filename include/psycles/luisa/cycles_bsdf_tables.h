#pragma once

#include <cstdint>

namespace psycles::luisa_backend::cycles45_tables {

// Offsets into the contiguous Blender 4.5.10 Cycles BSDF lookup-table
// buffer. The table payload is copied verbatim from
// intern/cycles/scene/shader.tables and versioned with the compatibility
// contract.
inline constexpr std::uint32_t ggx_e_offset = 0u;
inline constexpr std::uint32_t ggx_e_size = 1024u;

inline constexpr std::uint32_t ggx_eavg_offset =
    ggx_e_offset + ggx_e_size;
inline constexpr std::uint32_t ggx_eavg_size = 32u;

inline constexpr std::uint32_t ggx_glass_e_offset =
    ggx_eavg_offset + ggx_eavg_size;
inline constexpr std::uint32_t ggx_glass_e_size = 4096u;

inline constexpr std::uint32_t ggx_glass_eavg_offset =
    ggx_glass_e_offset + ggx_glass_e_size;
inline constexpr std::uint32_t ggx_glass_eavg_size = 256u;

inline constexpr std::uint32_t ggx_glass_inv_e_offset =
    ggx_glass_eavg_offset + ggx_glass_eavg_size;
inline constexpr std::uint32_t ggx_glass_inv_e_size = 4096u;

inline constexpr std::uint32_t ggx_glass_inv_eavg_offset =
    ggx_glass_inv_e_offset + ggx_glass_inv_e_size;
inline constexpr std::uint32_t ggx_glass_inv_eavg_size = 256u;

inline constexpr std::uint32_t sheen_ltc_offset =
    ggx_glass_inv_eavg_offset + ggx_glass_inv_eavg_size;
inline constexpr std::uint32_t sheen_ltc_size = 3072u;

inline constexpr std::uint32_t ggx_gen_schlick_ior_s_offset =
    sheen_ltc_offset + sheen_ltc_size;
inline constexpr std::uint32_t ggx_gen_schlick_ior_s_size = 4096u;

inline constexpr std::uint32_t ggx_gen_schlick_s_offset =
    ggx_gen_schlick_ior_s_offset +
    ggx_gen_schlick_ior_s_size;
inline constexpr std::uint32_t ggx_gen_schlick_s_size = 4096u;

// The albedo tables above are byte-for-byte unchanged in Cycles 5.2.1. The
// thin-film CMF source added in 5.2 is transformed to the scene-linear RGB
// space on the host and appended in six channel-major arrays.
inline constexpr std::uint32_t thin_film_offset =
    ggx_gen_schlick_s_offset + ggx_gen_schlick_s_size;
inline constexpr std::uint32_t thin_film_table_size = 512u;
inline constexpr std::uint32_t thin_film_channel_count = 6u;
inline constexpr std::uint32_t thin_film_size =
    thin_film_table_size * thin_film_channel_count;

inline constexpr std::uint32_t total_size =
    thin_film_offset + thin_film_size;

}// namespace psycles::luisa_backend::cycles45_tables
