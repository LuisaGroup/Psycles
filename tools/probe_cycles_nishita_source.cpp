#include "sky_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr int width = 512;
constexpr int height = 128;
constexpr int channels = 4;
constexpr float pi = 3.14159265358979323846f;
constexpr float half_pi = 1.57079632679489661923f;
constexpr float two_pi = 6.28318530717958647692f;

struct Float3 {
    float x;
    float y;
    float z;
};

[[nodiscard]] Float3 normalize(Float3 value) {
    const auto length = std::sqrt(
        value.x * value.x +
        value.y * value.y +
        value.z * value.z);
    return {
        value.x / length,
        value.y / length,
        value.z / length};
}

[[nodiscard]] Float3 read(
    const std::vector<float> &pixels,
    int x,
    int y) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    const auto offset =
        static_cast<std::size_t>((y * width + x) * channels);
    return {
        pixels[offset],
        pixels[offset + 1u],
        pixels[offset + 2u]};
}

[[nodiscard]] Float3 sample(
    const std::vector<float> &pixels,
    float x,
    float y) {
    const auto pixel_x = x * static_cast<float>(width) - 0.5f;
    const auto pixel_y = y * static_cast<float>(height) - 0.5f;
    const auto x0 = static_cast<int>(std::floor(pixel_x));
    const auto y0 = static_cast<int>(std::floor(pixel_y));
    const auto tx = pixel_x - static_cast<float>(x0);
    const auto ty = pixel_y - static_cast<float>(y0);
    const auto a = read(pixels, x0, y0);
    const auto b = read(pixels, x0 + 1, y0);
    const auto c = read(pixels, x0, y0 + 1);
    const auto d = read(pixels, x0 + 1, y0 + 1);
    return {
        (1.0f - ty) * ((1.0f - tx) * a.x + tx * b.x) +
            ty * ((1.0f - tx) * c.x + tx * d.x),
        (1.0f - ty) * ((1.0f - tx) * a.y + tx * b.y) +
            ty * ((1.0f - tx) * c.y + tx * d.y),
        (1.0f - ty) * ((1.0f - tx) * a.z + tx * b.z) +
            ty * ((1.0f - tx) * c.z + tx * d.z)};
}

[[nodiscard]] Float3 evaluate(
    const std::vector<float> &pixels,
    Float3 direction,
    float sun_rotation) {
    direction = normalize(direction);
    if (direction.z < -0.4f) {
        return {};
    }
    const auto theta =
        std::acos(std::clamp(direction.z, -1.0f, 1.0f));
    const auto phi = std::atan2(direction.y, direction.x);
    auto x =
        (-phi - half_pi + sun_rotation) / two_pi;
    x -= std::floor(x);
    if (direction.z >= 0.0f) {
        const auto elevation = half_pi - theta;
        return sample(
            pixels,
            x,
            std::sqrt(std::max(elevation / half_pi, 0.0f)));
    }
    auto fade = 1.0f + direction.z * 2.5f;
    fade = fade * fade * fade;
    const auto value = sample(pixels, x, -0.5f);
    return {value.x * fade, value.y * fade, value.z * fade};
}

[[nodiscard]] Float3 xyz_to_rgb(Float3 xyz) {
    return {
        std::max(
            3.2404542f * xyz.x -
                1.5371385f * xyz.y -
                0.4985314f * xyz.z,
            0.0f),
        std::max(
            -0.9692660f * xyz.x +
                1.8760108f * xyz.y +
                0.0415560f * xyz.z,
            0.0f),
        std::max(
            0.0556434f * xyz.x -
                0.2040259f * xyz.y +
                1.0572252f * xyz.z,
            0.0f)};
}

}// namespace

int main() {
    constexpr float sun_elevation = 0.9250245094299316f;
    constexpr float sun_rotation = 3.6651914755450647f;
    std::vector<float> pixels(
        static_cast<std::size_t>(width * height * channels));
    SKY_nishita_skymodel_precompute_texture(
        pixels.data(),
        channels,
        0,
        height,
        width,
        height,
        sun_elevation,
        1.0f,
        1.0f,
        1.0f,
        1.0f);

    const std::array probes{
        std::pair{
            "near_zenith",
            Float3{1.0e-3f, 2.0e-3f, 1.0f}},
        std::pair{
            "north_horizon",
            Float3{0.0f, 1.0f, 1.0e-4f}},
        std::pair{
            "east_horizon",
            Float3{1.0f, 0.0f, 1.0e-4f}}};
    std::cout << std::setprecision(10) << "{\n";
    for (std::size_t i = 0u; i < probes.size(); ++i) {
        const auto rgb = xyz_to_rgb(evaluate(
            pixels, probes[i].second, sun_rotation));
        std::cout << "  \"" << probes[i].first << "\": ["
                  << rgb.x << ", " << rgb.y << ", " << rgb.z
                  << "]"
                  << (i + 1u == probes.size() ? "\n" : ",\n");
    }
    std::cout << "}\n";
}
