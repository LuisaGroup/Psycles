#include "surface_preparation_accumulator.h"

#include "surface_math.h"

#include <algorithm>
#include <utility>

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

SurfacePreparationAccumulator::SurfacePreparationAccumulator(
    const SurfacePoint &point,
    std::size_t capacity,
    Expr<float> glossy_filter_roughness,
    Expr<bool> include_runtime_flags,
    Expr<bool> include_aov,
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureAovCallable &aov_operation) noexcept
    : _point{point},
      _capacity{std::clamp(
          capacity,
          std::size_t{1u},
          static_cast<std::size_t>(
              maximum_surface_closure_capacity))},
      _glossy_filter_roughness{glossy_filter_roughness},
      _include_runtime_flags{include_runtime_flags},
      _include_aov{include_aov},
      _identity{identity},
      _aov_operation{aov_operation},
      _retained_count{0u},
      _runtime_flags{select(
          0u,
          cycles_closure::runtime_backfacing,
          _include_runtime_flags & point.back_facing)},
      _aov{SurfacePreparation::zero(point).aov},
      _aov_total_weight{0.0f},
      _aov_roughness_weight{0.0f},
      _aov_roughness{0.0f},
      _aov_normal{make_float3(0.0f)} {}

void SurfacePreparationAccumulator::set_shading_normal(
    Expr<luisa::float3> shading_normal) noexcept {
    _point.shading_normal = shading_normal;
    _aov.normal = shading_normal;
}

void SurfacePreparationAccumulator::fold_retained(
    const SurfaceClosureRecord &closure) noexcept {
    $if(_include_runtime_flags) {
        _runtime_flags |= _identity(
            closure.kind,
            closure.lobe,
            closure.bssrdf_method,
            closure.allocation_weight,
            closure.setup_valid,
            closure.roughness,
            closure.preserve_ggx_energy,
            closure.beckmann,
            _glossy_filter_roughness)
                              .x;
    };
    $if(_include_aov) {
        const auto contribution = _aov_operation(
            _point.incoming,
            _point.shading_normal,
            _point.geometric_normal,
            _point.use_bump_map_correction,
            closure.kind,
            closure.lobe,
            closure.weight,
            closure.setup_valid,
            closure.albedo,
            closure.reflection_albedo,
            closure.transmission_albedo,
            closure.normal,
            closure.roughness);
        _aov.albedo += contribution.albedo;
        _aov.glossy_albedo += contribution.glossy_albedo;
        _aov.transmission_albedo +=
            contribution.transmission_albedo;
        _aov.transparency += contribution.transparency;
        _aov_total_weight += contribution.total_weight;
        _aov_roughness_weight += contribution.roughness_weight;
        _aov_roughness += contribution.roughness;
        _aov_normal += contribution.normal;
    };
    _retained_count += 1u;
}

void SurfacePreparationAccumulator::add(
    const SurfaceClosureRecord &closure) noexcept {
    const auto allocated =
        (closure.kind != static_cast<std::uint32_t>(
                             SurfaceClosureKind::none)) &
        (closure.allocation_weight >=
         cycles_closure::closure_weight_cutoff);
    const auto retained =
        allocated &
        (_retained_count < static_cast<std::uint32_t>(_capacity));
    $if(retained) {
        fold_retained(closure);
    };
}

void SurfacePreparationAccumulator::add_retained(
    const SurfaceClosureRecord &closure) noexcept {
    fold_retained(closure);
}

void SurfacePreparationAccumulator::finish() noexcept {
    const auto computed_roughness = make_float2(select(
        1.0f,
        _aov_roughness /
            max(_aov_roughness_weight, 1.0e-20f),
        _aov_roughness_weight > 0.0f));
    _aov.roughness = select(
        make_float2(0.0f),
        computed_roughness,
        _include_aov);
    _aov.normal = safe_normalize(
        select(
            _point.shading_normal,
            _aov_normal,
            _aov_total_weight > 0.0f),
        _point.shading_normal);
}

SurfacePreparation SurfacePreparationAccumulator::preparation(
    Float3 emission) const noexcept {
    return {
        .emission = std::move(emission),
        .runtime_flags = _runtime_flags,
        .aov = _aov};
}

} // namespace psycles::luisa_backend::detail
