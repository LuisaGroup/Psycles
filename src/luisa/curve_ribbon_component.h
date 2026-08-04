#pragma once

#include <cstdint>
#include <memory>

#include <luisa/dsl/rtx/ray.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

struct CurveControlPoints {
    luisa::compute::Float4 before;
    luisa::compute::Float4 begin;
    luisa::compute::Float4 end;
    luisa::compute::Float4 after;
};

struct CurveRibbonIntersection {
    luisa::compute::Bool valid;
    luisa::compute::Float distance;
    luisa::compute::Float u;
    luisa::compute::Float v;
};

// Host-stage polymorphism emits a typed Luisa AST. The shader sees neither a
// virtual call nor a weakly typed parameter block: each operation below is
// expanded while the kernel is being constructed.
class CurveRibbonComponent {

public:
    virtual ~CurveRibbonComponent() noexcept = default;

    [[nodiscard]] virtual luisa::compute::Float4 evaluate(
        const CurveControlPoints &curve,
        luisa::compute::Expr<float> u) const noexcept = 0;

    [[nodiscard]] virtual luisa::compute::Float4 derivative(
        const CurveControlPoints &curve,
        luisa::compute::Expr<float> u) const noexcept = 0;

    [[nodiscard]] virtual CurveRibbonIntersection intersect(
        const luisa::compute::Var<luisa::compute::Ray> &object_ray,
        const CurveControlPoints &curve,
        luisa::compute::Expr<std::uint32_t> subdivision_level)
        const noexcept = 0;
};

[[nodiscard]] std::shared_ptr<const CurveRibbonComponent>
make_curve_ribbon_component();

}// namespace psycles::luisa_backend::detail
