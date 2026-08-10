#include "path_tracer_color_transforms.h"

#include "surface_color_transforms.h"

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] ShaderColorTransformCallable
make_rgb_to_hsv_callable() noexcept {
    ShaderColorTransformCallable rgb_to_hsv =
        [](Float3 rgb) noexcept {
            return rgb_to_hsv_inline(rgb);
        };
    rgb_to_hsv.set_name("surface_rgb_to_hsv");
    return rgb_to_hsv;
}

[[nodiscard]] ShaderColorTransformCallable
make_hsv_to_rgb_callable() noexcept {
    ShaderColorTransformCallable hsv_to_rgb =
        [](Float3 hsv) noexcept {
            return hsv_to_rgb_inline(hsv);
        };
    hsv_to_rgb.set_name("surface_hsv_to_rgb");
    return hsv_to_rgb;
}

[[nodiscard]] ShaderColorTransformCallable
make_rgb_to_hsl_callable() noexcept {
    ShaderColorTransformCallable rgb_to_hsl =
        [](Float3 rgb) noexcept {
            return rgb_to_hsl_inline(rgb);
        };
    rgb_to_hsl.set_name("surface_rgb_to_hsl");
    return rgb_to_hsl;
}

[[nodiscard]] ShaderColorTransformCallable
make_hsl_to_rgb_callable() noexcept {
    ShaderColorTransformCallable hsl_to_rgb =
        [](Float3 hsl) noexcept {
            return hsl_to_rgb_inline(hsl);
        };
    hsl_to_rgb.set_name("surface_hsl_to_rgb");
    return hsl_to_rgb;
}

}// namespace

Float3 CallableSurfaceColorTransformProvider::rgb_to_hsv(
    Float3 rgb) const noexcept {
    if (!_rgb_to_hsv) {
        _rgb_to_hsv.emplace(make_rgb_to_hsv_callable());
    }
    return (*_rgb_to_hsv)(rgb);
}

Float3 CallableSurfaceColorTransformProvider::hsv_to_rgb(
    Float3 hsv) const noexcept {
    if (!_hsv_to_rgb) {
        _hsv_to_rgb.emplace(make_hsv_to_rgb_callable());
    }
    return (*_hsv_to_rgb)(hsv);
}

Float3 CallableSurfaceColorTransformProvider::rgb_to_hsl(
    Float3 rgb) const noexcept {
    if (!_rgb_to_hsl) {
        _rgb_to_hsl.emplace(make_rgb_to_hsl_callable());
    }
    return (*_rgb_to_hsl)(rgb);
}

Float3 CallableSurfaceColorTransformProvider::hsl_to_rgb(
    Float3 hsl) const noexcept {
    if (!_hsl_to_rgb) {
        _hsl_to_rgb.emplace(make_hsl_to_rgb_callable());
    }
    return (*_hsl_to_rgb)(hsl);
}

}// namespace psycles::luisa_backend::detail
