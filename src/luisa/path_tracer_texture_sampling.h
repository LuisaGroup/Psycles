#pragma once

#include "path_tracer_internal.h"

#include <array>
#include <optional>

namespace psycles::luisa_backend::detail {

inline constexpr auto texture_interpolation_family_count =
    static_cast<std::size_t>(
        compiler::surface_value_image_interpolation_family_count);
inline constexpr auto texture_extension_mode_count =
    static_cast<std::size_t>(
        compiler::surface_value_image_extension_mode_count);
inline constexpr auto texture_sampling_specialization_count =
    static_cast<std::size_t>(
        compiler::surface_value_image_sampling_key_count);

using Texture2DSamplingCallable =
    Callable<luisa::float4(
        BindlessArray,
        luisa::uint,
        luisa::float2)>;

using SurfaceImageBoxCallable =
    Callable<luisa::float4(
        BindlessArray,
        luisa::float3,
        luisa::float3,
        float,
        luisa::uint,
        bool,
        bool)>;

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
class CallableTexture2DSamplingProvider final
    : public SurfaceImageBoxProvider {

private:
    Expr<BindlessArray> _textures;
    const Texture2DSamplingCallables &_callables;
    // BOX is a larger pure operation than one texture lookup. Lazily retain
    // one callable for each canonical sampling mode that is actually reached
    // while recording the scene. Each callable contains three references to
    // the corresponding typed sampler, never a weak SVM register ABI.
    mutable std::array<
        std::optional<SurfaceImageBoxCallable>,
        texture_sampling_specialization_count>
        _image_box_callables;

public:
    CallableTexture2DSamplingProvider(
        Expr<BindlessArray> textures,
        const Texture2DSamplingCallables &callables) noexcept;

    [[nodiscard]] Float4 sample(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept;

    [[nodiscard]] Float4 evaluate(
        const SurfaceImageBoxInput &input,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept override;
};

}// namespace psycles::luisa_backend::detail
