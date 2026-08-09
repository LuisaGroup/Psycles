#pragma once

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

#include <variant>

namespace psycles::luisa_backend::detail {

enum class SurfaceValueCategory : std::uint8_t {
    scalar,
    vector,
    unsigned_integer
};

// Classify the compiler IR type while recording the shader AST. Unsupported
// types are construction-time compiler errors; no runtime type tag reaches
// device code.
[[nodiscard]] SurfaceValueCategory surface_value_category(
    contract::SocketType type) noexcept;

// A topologically scheduled shader value retains its original DSL type.
// std::variant is host-stage metadata only: the generated Luisa AST contains
// the selected Expr<float> or Expr<float3>, never a tagged union or a padded
// float4 value.
class SurfaceValueExpression {

  private:
    std::variant<Expr<float>, Expr<luisa::float3>, Expr<luisa::ulong>> _value;

    explicit SurfaceValueExpression(Expr<float> value) noexcept;
    explicit SurfaceValueExpression(
        Expr<luisa::float3> value) noexcept;
    explicit SurfaceValueExpression(
        Expr<luisa::ulong> value) noexcept;

  public:
    [[nodiscard]] static SurfaceValueExpression from_scalar(
        Expr<float> value) noexcept;
    [[nodiscard]] static SurfaceValueExpression from_vector(
        Expr<luisa::float3> value) noexcept;
    [[nodiscard]] static SurfaceValueExpression from_unsigned_integer(
        Expr<luisa::ulong> value) noexcept;
    [[nodiscard]] static SurfaceValueExpression zero(
        contract::SocketType type) noexcept;

    [[nodiscard]] Float scalar() const noexcept;
    [[nodiscard]] Float3 vector() const noexcept;
    [[nodiscard]] ULong unsigned_integer() const noexcept;
};

// Migration boundary for node implementations whose local computation is
// naturally four-channel (for example an RGBA texture sample). The result is
// projected immediately according to the statically known compiler IR type;
// only the strong expression is retained in the topological value stream.
[[nodiscard]] SurfaceValueExpression project_surface_value(
    contract::SocketType type,
    Float4 value) noexcept;

}// namespace psycles::luisa_backend::detail
