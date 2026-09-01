#pragma once

#include <cstdint>

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend {

/* Minimal device-side projection of Cycles' contiguous BSDF lookup-table
 * buffer. Both the native SVM KernelGlobals and the legacy graph surface
 * services implement this interface; table interpolation therefore has one
 * implementation and does not depend on either shading architecture. */
class CyclesBsdfTableReader {
public:
    virtual ~CyclesBsdfTableReader() noexcept = default;

    [[nodiscard]] virtual luisa::compute::Float cycles_bsdf_data(
        luisa::compute::Expr<std::uint32_t> index) const noexcept = 0;
};

namespace detail {

[[nodiscard]] luisa::compute::Float cycles_table_1d(
    const CyclesBsdfTableReader &reader,
    luisa::compute::Float x,
    luisa::compute::Expr<std::uint32_t> offset,
    std::uint32_t size) noexcept;

[[nodiscard]] luisa::compute::Float cycles_table_2d(
    const CyclesBsdfTableReader &reader,
    luisa::compute::Float x,
    luisa::compute::Float y,
    luisa::compute::Expr<std::uint32_t> offset,
    std::uint32_t x_size,
    std::uint32_t y_size) noexcept;

[[nodiscard]] luisa::compute::Float cycles_table_3d(
    const CyclesBsdfTableReader &reader,
    luisa::compute::Float x,
    luisa::compute::Float y,
    luisa::compute::Float z,
    luisa::compute::Expr<std::uint32_t> offset,
    std::uint32_t x_size,
    std::uint32_t y_size,
    std::uint32_t z_size) noexcept;

}// namespace detail

namespace cycles45_tables {

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

}// namespace cycles45_tables

}// namespace psycles::luisa_backend
