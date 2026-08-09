#include "graph_surface_value_expression.h"

#include <cstdlib>

namespace psycles::luisa_backend::detail {

SurfaceValueCategory surface_value_category(
    contract::SocketType type) noexcept {
    using contract::SocketType;
    switch (type) {
        case SocketType::boolean:
        case SocketType::integer:
        case SocketType::floating:
            return SurfaceValueCategory::scalar;
        case SocketType::unsigned_integer:
            return SurfaceValueCategory::unsigned_integer;
        case SocketType::float2:
        case SocketType::float3:
        case SocketType::color:
        case SocketType::spectrum:
        case SocketType::point:
        case SocketType::vector:
        case SocketType::normal:
            return SurfaceValueCategory::vector;
        case SocketType::transform:
        case SocketType::string:
        case SocketType::closure:
        case SocketType::volume_closure:
            std::abort();
    }
    std::abort();
}

SurfaceValueExpression::SurfaceValueExpression(
    Expr<float> value) noexcept
    : _value{value} {}

SurfaceValueExpression::SurfaceValueExpression(
    Expr<luisa::float3> value) noexcept
    : _value{value} {}

SurfaceValueExpression::SurfaceValueExpression(
    Expr<luisa::ulong> value) noexcept
    : _value{value} {}

SurfaceValueExpression SurfaceValueExpression::from_scalar(
    Expr<float> value) noexcept {
    return SurfaceValueExpression{value};
}

SurfaceValueExpression SurfaceValueExpression::from_vector(
    Expr<luisa::float3> value) noexcept {
    return SurfaceValueExpression{value};
}

SurfaceValueExpression SurfaceValueExpression::from_unsigned_integer(
    Expr<luisa::ulong> value) noexcept {
    return SurfaceValueExpression{value};
}

SurfaceValueExpression SurfaceValueExpression::zero(
    contract::SocketType type) noexcept {
    switch (surface_value_category(type)) {
        case SurfaceValueCategory::scalar:
            return from_scalar(Expr<float>{0.0f});
        case SurfaceValueCategory::vector:
            return from_vector(Expr<luisa::float3>{
                luisa::make_float3(0.0f)});
        case SurfaceValueCategory::unsigned_integer:
            return from_unsigned_integer(Expr<luisa::ulong>{0ull});
    }
    std::abort();
}

Float SurfaceValueExpression::scalar() const noexcept {
    if (const auto *value =
            std::get_if<Expr<float>>(&_value)) {
        return Float{*value};
    }
    std::abort();
}

Float3 SurfaceValueExpression::vector() const noexcept {
    if (const auto *value =
            std::get_if<Expr<luisa::float3>>(&_value)) {
        return Float3{*value};
    }
    std::abort();
}

ULong SurfaceValueExpression::unsigned_integer() const noexcept {
    if (const auto *value =
            std::get_if<Expr<luisa::ulong>>(&_value)) {
        return ULong{*value};
    }
    std::abort();
}

SurfaceValueExpression project_surface_value(
    contract::SocketType type,
    Float4 value) noexcept {
    switch (surface_value_category(type)) {
        case SurfaceValueCategory::scalar:
            return SurfaceValueExpression::from_scalar(
                Expr<float>{value.x.expression()});
        case SurfaceValueCategory::vector:
            return SurfaceValueExpression::from_vector(
                Expr<luisa::float3>{value.xyz().expression()});
        case SurfaceValueCategory::unsigned_integer:
            std::abort();
    }
    std::abort();
}

}// namespace psycles::luisa_backend::detail
