#include <psycles/luisa/surface_closure_operations.h>

#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] Bool allocated(
    const SurfaceClosureExpression &closure) noexcept {
    return (closure.kind != static_cast<std::uint32_t>(
                                SurfaceClosureKind::none)) &
           (closure.allocation_weight >=
               cycles_closure::closure_weight_cutoff);
}

[[nodiscard]] Bool retained(
    const SurfaceClosureExpression &closure,
    UInt allocated_count,
    std::size_t capacity) noexcept {
    return allocated(closure) &
           (allocated_count < static_cast<std::uint32_t>(capacity));
}

[[nodiscard]] luisa::compute::UInt2 classify(
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureExpression &closure,
    Expr<float> glossy_filter_roughness) noexcept {
    return identity(
        closure.kind,
        closure.lobe,
        closure.allocation_weight,
        closure.setup_valid,
        closure.roughness,
        closure.preserve_ggx_energy,
        closure.beckmann,
        glossy_filter_roughness);
}

}// namespace

SurfaceClosureIdentityCallable
make_surface_closure_identity_callable() noexcept {
    return [](
               UInt kind,
               UInt lobe,
               Float allocation_weight,
               Bool setup_valid,
               Float roughness,
               Bool preserve_ggx_energy,
               Bool beckmann,
               Float glossy_filter_roughness) noexcept {
        const auto closure =
            detail::SurfaceClosureIdentityExpression{
                .kind = Expr<std::uint32_t>{kind.expression()},
                .lobe = Expr<std::uint32_t>{lobe.expression()},
                .allocation_weight =
                    Expr<float>{allocation_weight.expression()},
                .setup_valid =
                    Expr<bool>{setup_valid.expression()},
                .roughness = Expr<float>{roughness.expression()},
                .preserve_ggx_energy =
                    Expr<bool>{preserve_ggx_energy.expression()},
                .beckmann = Expr<bool>{beckmann.expression()}};
        return luisa::compute::make_uint2(
            detail::cycles_runtime_flags(
                closure,
                std::move(glossy_filter_roughness)),
            detail::cycles_closure_type(closure));
    };
}

SurfaceRuntimeFlagsVisitor::SurfaceRuntimeFlagsVisitor(
    const SurfacePoint &point,
    Expr<float> glossy_filter_roughness,
    std::size_t capacity,
    const SurfaceClosureIdentityCallable &identity) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _glossy_filter_roughness{glossy_filter_roughness},
      _identity{identity} {}

void SurfaceRuntimeFlagsVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    static_cast<void>(shading_normal);
    _result = select(
        0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    UInt allocated_count = 0u;
    for (const auto &closure : closures) {
        const auto keep = retained(
            closure, allocated_count, capacity());
        $if(keep) {
            _result |= classify(
                _identity,
                closure,
                _glossy_filter_roughness)
                           .x;
        };
        allocated_count += select(0u, 1u, keep);
    }
}

Expr<std::uint32_t> SurfaceRuntimeFlagsVisitor::result() const noexcept {
    return Expr<std::uint32_t>{_result.expression()};
}

SurfaceClosureTraceVisitor::SurfaceClosureTraceVisitor(
    const SurfacePoint &point,
    Expr<std::uint32_t> requested_index,
    std::size_t capacity,
    const SurfaceClosureIdentityCallable &identity) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _requested_index{requested_index},
      _identity{identity},
      _result{SurfaceClosureTrace::zero(requested_index)} {}

void SurfaceClosureTraceVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    static_cast<void>(shading_normal);
    _result.count = 0u;
    _result.runtime_flags = select(
        0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    _result.index = _requested_index;
    _result.type = cycles_closure::type_none;
    _result.sample_weight = 0.0f;
    _result.weight = make_float3(0.0f);
    _result.normal = make_float3(0.0f, 0.0f, 1.0f);
    _result.valid = false;

    UInt allocated_count = 0u;
    for (const auto &closure : closures) {
        const auto keep = retained(
            closure, allocated_count, capacity());
        $if(keep) {
            const auto identity = classify(
                _identity, closure, 0.0f);
            _result.runtime_flags |= identity.x;
            const auto match =
                allocated_count == _requested_index;
            _result.type = select(
                _result.type, identity.y, match);
            _result.sample_weight = select(
                _result.sample_weight,
                closure.sample_weight,
                match);
            _result.weight = select(
                _result.weight, closure.weight, match);
            _result.normal = select(
                _result.normal, closure.normal, match);
        };
        allocated_count += select(0u, 1u, keep);
    }
    _result.count = allocated_count;
    _result.valid = _requested_index < allocated_count;
}

const SurfaceClosureTrace &SurfaceClosureTraceVisitor::result() const noexcept {
    return _result;
}

}// namespace psycles::luisa_backend
