#include "path_tracer_texture_sampling.h"

#include "cycles_texture_sampling.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] constexpr std::uint32_t
canonical_interpolation_family(
    std::uint32_t interpolation) noexcept {
    return interpolation == 0u ? 0u :
           interpolation == 1u ? 1u : 2u;
}

[[nodiscard]] constexpr std::uint32_t
canonical_extension_mode(
    std::uint32_t extension) noexcept {
    return extension == 0u ? 0u :
           extension == 2u ? 2u :
           extension == 3u ? 3u : 1u;
}

[[nodiscard]] constexpr std::size_t specialization_index(
    std::uint32_t interpolation,
    std::uint32_t extension) noexcept {
    return static_cast<std::size_t>(
               canonical_interpolation_family(interpolation)) *
               texture_extension_mode_count +
           static_cast<std::size_t>(
               canonical_extension_mode(extension));
}

[[nodiscard]] const char *interpolation_name(
    std::uint32_t interpolation) noexcept {
    switch (canonical_interpolation_family(interpolation)) {
        case 0u: return "nearest";
        case 1u: return "linear";
        default: return "cubic";
    }
}

[[nodiscard]] const char *extension_name(
    std::uint32_t extension) noexcept {
    switch (canonical_extension_mode(extension)) {
        case 0u: return "repeat";
        case 2u: return "extend";
        case 3u: return "mirror";
        default: return "clip";
    }
}

[[nodiscard]] Texture2DSamplingCallable
make_texture_2d_sampling_callable(
    std::uint32_t interpolation,
    std::uint32_t extension) noexcept {
    Texture2DSamplingCallable callable =
        [interpolation, extension](
            BindlessVar textures,
            UInt handle,
            Float2 uv) noexcept {
            return sample_cycles_texture_2d(
                textures,
                Expr<std::uint32_t>{handle.expression()},
                Expr<luisa::float2>{uv.expression()},
                interpolation,
                extension);
        };
    callable.set_name(luisa::format(
        "cycles_texture_{}_{}",
        interpolation_name(interpolation),
        extension_name(extension)));
    return callable;
}

template<std::size_t... Indices>
[[nodiscard]] Texture2DSamplingCallables
make_texture_2d_sampling_callables_impl(
    std::index_sequence<Indices...>) noexcept {
    return {{make_texture_2d_sampling_callable(
        static_cast<std::uint32_t>(
            Indices / texture_extension_mode_count),
        static_cast<std::uint32_t>(
            Indices % texture_extension_mode_count))...}};
}

}// namespace

const Texture2DSamplingCallable &
Texture2DSamplingCallables::specialization(
    std::uint32_t interpolation,
    std::uint32_t extension) const noexcept {
    return specializations[
        specialization_index(interpolation, extension)];
}

Texture2DSamplingCallables
make_texture_2d_sampling_callables() noexcept {
    return make_texture_2d_sampling_callables_impl(
        std::make_index_sequence<
            texture_sampling_specialization_count>{});
}

CallableTexture2DSamplingProvider::
    CallableTexture2DSamplingProvider(
        const BindlessVar &textures,
        const Texture2DSamplingCallables &callables) noexcept
    : _textures{textures},
      _callables{callables} {}

Float4 CallableTexture2DSamplingProvider::sample(
    Expr<std::uint32_t> handle,
    Expr<luisa::float2> uv,
    std::uint32_t interpolation,
    std::uint32_t extension) const noexcept {
    return _callables
        .specialization(interpolation, extension)(
            _textures,
            handle,
            uv);
}

}// namespace psycles::luisa_backend::detail
