#pragma once

#include <array>

namespace psycles {

struct Vec2f {
    float x{};
    float y{};

    bool operator==(const Vec2f &) const noexcept = default;
};

struct Vec3f {
    float x{};
    float y{};
    float z{};

    bool operator==(const Vec3f &) const noexcept = default;
};

struct Vec4f {
    float x{};
    float y{};
    float z{};
    float w{};

    bool operator==(const Vec4f &) const noexcept = default;
};

struct Mat4f {
    std::array<float, 16u> elements{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};

    bool operator==(const Mat4f &) const noexcept = default;
};

}// namespace psycles

