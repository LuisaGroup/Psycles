#include "path_tracer_texture_sampling.h"

#include "cycles_texture_sampling.h"
#include "surface_image_box.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] constexpr std::uint32_t
canonical_interpolation_family(
    std::uint32_t interpolation) noexcept {
    return compiler::canonical_surface_value_image_interpolation(
        interpolation);
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
        compiler::make_surface_value_image_sampling_key(
            interpolation,
            canonical_extension_mode(extension)));
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

class CallableImageBoxTextureSampler final
    : public SurfaceImageBoxTextureSampler {

private:
    Expr<BindlessArray> _textures;
    const Texture2DSamplingCallable &_sampler;

public:
    CallableImageBoxTextureSampler(
        Expr<BindlessArray> textures,
        const Texture2DSamplingCallable &sampler) noexcept
        : _textures{std::move(textures)},
          _sampler{sampler} {}

    [[nodiscard]] Float4 sample(
        Expr<std::uint32_t> texture_handle,
        Float2 uv) const noexcept override {
        return _sampler(
            _textures,
            texture_handle,
            uv);
    }
};

[[nodiscard]] SurfaceImageBoxCallable
make_surface_image_box_callable(
    const Texture2DSamplingCallable &texture_sampler,
    std::uint32_t interpolation,
    std::uint32_t extension) noexcept {
    SurfaceImageBoxCallable callable =
        [texture_sampler](
            BindlessVar textures,
            Float3 coordinate,
            Float3 signed_normal,
            Float blend,
            UInt texture_handle,
            Bool unassociate_alpha,
            Bool encoded_as_srgb) noexcept {
            CallableImageBoxTextureSampler sampler{
                textures,
                texture_sampler};
            return evaluate_surface_image_box(
                SurfaceImageBoxInput{
                    .coordinate = coordinate,
                    .signed_normal = signed_normal,
                    .blend = blend,
                    .texture_handle = texture_handle,
                    .unassociate_alpha = unassociate_alpha,
                    .encoded_as_srgb = encoded_as_srgb},
                sampler);
        };
    callable.set_name(luisa::format(
        "cycles_image_box_{}_{}",
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
        Expr<BindlessArray> textures,
        const Texture2DSamplingCallables &callables) noexcept
    : _textures{std::move(textures)},
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

Float4 CallableTexture2DSamplingProvider::evaluate(
    const SurfaceImageBoxInput &input,
    std::uint32_t interpolation,
    std::uint32_t extension) const noexcept {
    const auto index =
        specialization_index(interpolation, extension);
    auto &box_callable = _image_box_callables[index];
    if (!box_callable) {
        box_callable.emplace(make_surface_image_box_callable(
            _callables.specializations[index],
            interpolation,
            extension));
    }
    return (*box_callable)(
        _textures,
        input.coordinate,
        input.signed_normal,
        input.blend,
        input.texture_handle,
        input.unassociate_alpha,
        input.encoded_as_srgb);
}

}// namespace psycles::luisa_backend::detail
