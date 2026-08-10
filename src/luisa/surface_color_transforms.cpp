#include "surface_color_transforms.h"

namespace psycles::luisa_backend::detail {

Float3 rgb_to_hsv_inline(Float3 rgb) noexcept {
    auto cmax = max(rgb.x, max(rgb.y, rgb.z));
    auto cmin = min(rgb.x, min(rgb.y, rgb.z));
    auto delta = cmax - cmin;
    auto saturation = select(
        0.0f,
        delta / select(1.0f, cmax, cmax != 0.0f),
        cmax != 0.0f);
    auto safe_delta =
        select(1.0f, delta, delta != 0.0f);
    auto c = (make_float3(cmax) - rgb) / safe_delta;
    auto hue = 4.0f + c.y - c.x;
    hue = select(
        hue,
        2.0f + c.x - c.z,
        rgb.y == cmax);
    hue = select(
        hue,
        c.z - c.y,
        rgb.x == cmax);
    hue /= 6.0f;
    hue = select(hue, hue + 1.0f, hue < 0.0f);
    hue = select(0.0f, hue, saturation != 0.0f);
    return make_float3(hue, saturation, cmax);
}

Float3 hsv_to_rgb_inline(Float3 hsv) noexcept {
    auto h = select(hsv.x, 0.0f, hsv.x == 1.0f);
    h *= 6.0f;
    auto sector = floor(h);
    auto f = h - sector;
    auto p = hsv.z * (1.0f - hsv.y);
    auto q = hsv.z * (1.0f - hsv.y * f);
    auto t = hsv.z *
             (1.0f - hsv.y * (1.0f - f));
    auto rgb = make_float3(hsv.z, p, q);
    rgb = select(
        rgb,
        make_float3(t, p, hsv.z),
        sector == 4.0f);
    rgb = select(
        rgb,
        make_float3(p, q, hsv.z),
        sector == 3.0f);
    rgb = select(
        rgb,
        make_float3(p, hsv.z, t),
        sector == 2.0f);
    rgb = select(
        rgb,
        make_float3(q, hsv.z, p),
        sector == 1.0f);
    rgb = select(
        rgb,
        make_float3(hsv.z, t, p),
        sector == 0.0f);
    return select(
        make_float3(hsv.z),
        rgb,
        hsv.y != 0.0f);
}

Float3 rgb_to_hsl_inline(Float3 rgb) noexcept {
    auto cmax = max(rgb.x, max(rgb.y, rgb.z));
    auto cmin = min(rgb.x, min(rgb.y, rgb.z));
    auto lightness = min(
        1.0f, (cmax + cmin) * 0.5f);
    auto delta = cmax - cmin;
    auto chromatic = cmax != cmin;
    auto denominator = select(
        cmax + cmin,
        2.0f - cmax - cmin,
        lightness > 0.5f);
    auto saturation = select(
        0.0f,
        delta /
            select(
                1.0f,
                denominator,
                abs(denominator) > 1.0e-20f),
        chromatic);
    auto safe_delta = select(
        1.0f, delta, abs(delta) > 1.0e-20f);
    auto hue =
        (rgb.x - rgb.y) / safe_delta + 4.0f;
    hue = select(
        hue,
        (rgb.z - rgb.x) / safe_delta + 2.0f,
        cmax == rgb.y);
    hue = select(
        hue,
        (rgb.y - rgb.z) / safe_delta +
            select(0.0f, 6.0f, rgb.y < rgb.z),
        cmax == rgb.x);
    hue = select(0.0f, hue / 6.0f, chromatic);
    return make_float3(hue, saturation, lightness);
}

Float3 hsl_to_rgb_inline(Float3 hsl) noexcept {
    auto hue6 = hsl.x * 6.0f;
    auto nr = clamp(
        abs(hue6 - 3.0f) - 1.0f,
        0.0f,
        1.0f);
    auto ng = clamp(
        2.0f - abs(hue6 - 2.0f),
        0.0f,
        1.0f);
    auto nb = clamp(
        2.0f - abs(hue6 - 4.0f),
        0.0f,
        1.0f);
    auto chroma =
        (1.0f - abs(2.0f * hsl.z - 1.0f)) *
        hsl.y;
    return make_float3(
        (nr - 0.5f) * chroma + hsl.z,
        (ng - 0.5f) * chroma + hsl.z,
        (nb - 0.5f) * chroma + hsl.z);
}

}// namespace psycles::luisa_backend::detail
