#include <psycles/luisa/surface_closure_population.h>

#include "surface_math.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_set.h>

#include <luisa/dsl/sugar.h>

#include <utility>

namespace psycles::luisa_backend {

struct SurfaceClosurePopulationCollector::Impl {
    SurfacePoint point;
    Float glossy_filter_roughness;
    Bool include_runtime_flags;
    Bool include_aov;
    // Own callable handles: the collector's public constructor accepts
    // references for convenience, but its lifetime must not depend on the
    // caller retaining those wrapper objects while the shader is recorded.
    SurfaceClosureIdentityCallable identity;
    SurfaceClosureAovCallable aov_operation;
    SurfaceClosureSet closures;
    UInt runtime_flags;
    SurfaceAov aov;
    Float aov_total_weight{0.0f};
    Float aov_roughness_weight{0.0f};
    Float aov_roughness{0.0f};
    Float3 aov_normal{make_float3(0.0f)};

    Impl(
        const SurfacePoint &point_value,
        std::size_t capacity,
        const SurfacePopulationQuery &query,
        const SurfaceClosureIdentityCallable &identity_value,
        const SurfaceClosureAovCallable &aov_operation_value) noexcept
        : point{point_value},
          glossy_filter_roughness{query.glossy_filter_roughness},
          include_runtime_flags{query.include_runtime_flags},
          include_aov{query.include_aov},
          identity{identity_value},
          aov_operation{aov_operation_value},
          closures{capacity, SurfaceClosureStorageProfile::physical},
          runtime_flags{select(
              0u,
              cycles_closure::runtime_backfacing,
              include_runtime_flags & point.back_facing)},
          aov{SurfacePreparation::zero(point).aov} {}
};

SurfaceClosurePopulationCollector::SurfaceClosurePopulationCollector(
    const SurfacePoint &point,
    std::size_t capacity,
    const SurfacePopulationQuery &query,
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureAovCallable &aov_operation) noexcept
    : _impl{std::make_unique<Impl>(
          point,
          capacity,
          query,
          identity,
          aov_operation)} {}

SurfaceClosurePopulationCollector::~SurfaceClosurePopulationCollector()
    noexcept = default;

void SurfaceClosurePopulationCollector::add(
    const SurfaceClosureRecord &closure) noexcept {
    // Storage and both reductions are recorded inside one canonical retained
    // transaction. Capacity truncation and allocation cutoff therefore cannot
    // make preparation observe a closure which later physical consumers do
    // not observe, or vice versa.
    _impl->closures.append(closure, [&] {
        $if(_impl->include_runtime_flags) {
            _impl->runtime_flags |= _impl->identity(
                closure.kind,
                closure.lobe,
                closure.bssrdf_method,
                closure.allocation_weight,
                closure.setup_valid,
                closure.roughness,
                closure.preserve_ggx_energy,
                closure.beckmann,
                _impl->glossy_filter_roughness)
                                        .x;
        };
        $if(_impl->include_aov) {
            const auto contribution = _impl->aov_operation(
                _impl->point.incoming,
                _impl->point.shading_normal,
                _impl->point.geometric_normal,
                _impl->point.use_bump_map_correction,
                closure.kind,
                closure.lobe,
                closure.weight,
                closure.setup_valid,
                closure.albedo,
                closure.reflection_albedo,
                closure.transmission_albedo,
                closure.normal,
                closure.roughness);
            _impl->aov.albedo += contribution.albedo;
            _impl->aov.glossy_albedo += contribution.glossy_albedo;
            _impl->aov.transmission_albedo +=
                contribution.transmission_albedo;
            _impl->aov.transparency += contribution.transparency;
            _impl->aov_total_weight += contribution.total_weight;
            _impl->aov_roughness_weight += contribution.roughness_weight;
            _impl->aov_roughness += contribution.roughness;
            _impl->aov_normal += contribution.normal;
        };
    });
}

void SurfaceClosurePopulationCollector::finish() noexcept {
    const auto computed_roughness = make_float2(select(
        1.0f,
        _impl->aov_roughness /
            max(_impl->aov_roughness_weight, 1.0e-20f),
        _impl->aov_roughness_weight > 0.0f));
    _impl->aov.roughness = select(
        make_float2(0.0f),
        computed_roughness,
        _impl->include_aov);
    _impl->aov.normal = detail::safe_normalize(
        select(
            _impl->point.shading_normal,
            _impl->aov_normal,
            _impl->aov_total_weight > 0.0f),
        _impl->point.shading_normal);
}

const SurfaceClosureSet &
SurfaceClosurePopulationCollector::closures() const noexcept {
    return _impl->closures;
}

SurfacePreparation SurfaceClosurePopulationCollector::preparation(
    Float3 emission) const noexcept {
    return {
        .emission = std::move(emission),
        .runtime_flags = _impl->runtime_flags,
        .aov = _impl->aov};
}

}// namespace psycles::luisa_backend
