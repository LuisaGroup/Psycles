#pragma once

#include "path_tracer_internal.h"

#include <array>

namespace psycles::luisa_backend::detail {

inline constexpr auto texture_interpolation_family_count =
    std::size_t{3u};
inline constexpr auto texture_extension_mode_count =
    std::size_t{4u};
inline constexpr auto texture_sampling_specialization_count =
    texture_interpolation_family_count *
    texture_extension_mode_count;

using Texture2DSamplingCallable =
    Callable<luisa::float4(
        BindlessArray,
        luisa::uint,
        luisa::float2)>;

struct Texture2DSamplingCallables {
    std::array<Texture2DSamplingCallable,
               texture_sampling_specialization_count>
        specializations;

    [[nodiscard]] const Texture2DSamplingCallable &
    specialization(std::uint32_t interpolation,
                   std::uint32_t extension) const noexcept;
};

[[nodiscard]] Texture2DSamplingCallables
make_texture_2d_sampling_callables() noexcept;

// Host-stage adapter from ShaderServices' semantic interpolation/extension
// metadata to the corresponding shared typed callable. No mode tag is
// evaluated on the device.
class CallableTexture2DSamplingProvider final {

private:
    const BindlessVar &_textures;
    const Texture2DSamplingCallables &_callables;

public:
    CallableTexture2DSamplingProvider(
        const BindlessVar &textures,
        const Texture2DSamplingCallables &callables) noexcept;

    [[nodiscard]] Float4 sample(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept;
};

}// namespace psycles::luisa_backend::detail
