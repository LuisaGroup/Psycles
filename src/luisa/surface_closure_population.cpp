#include <psycles/luisa/surface_closure_population.h>

#include "surface_preparation_accumulator.h"

#include <psycles/luisa/surface_closure_set.h>

#include <utility>

namespace psycles::luisa_backend {

struct SurfaceClosurePopulationCollector::Impl {
    SurfaceClosureSet closures;
    detail::SurfacePreparationAccumulator preparation;

    Impl(
        const SurfacePoint &point_value,
        std::size_t capacity,
        const SurfacePopulationQuery &query,
        const SurfaceClosureIdentityCallable &identity_value,
        const SurfaceClosureAovCallable &aov_operation_value) noexcept
        : closures{capacity, SurfaceClosureStorageProfile::physical},
          preparation{
              point_value,
              capacity,
              query.glossy_filter_roughness,
              query.include_runtime_flags,
              query.include_aov,
              identity_value,
              aov_operation_value} {}
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

void SurfaceClosurePopulationCollector::begin(
    Expr<luisa::float3> shading_normal) noexcept {
    _impl->preparation.set_shading_normal(shading_normal);
}

void SurfaceClosurePopulationCollector::add(
    const SurfaceClosureRecord &closure) noexcept {
    // Storage and both reductions are recorded inside one canonical retained
    // transaction. Capacity truncation and allocation cutoff therefore cannot
    // make preparation observe a closure which later physical consumers do
    // not observe, or vice versa.
    _impl->closures.append(closure, [&] {
        _impl->preparation.add_retained(closure);
    });
}

void SurfaceClosurePopulationCollector::finish() noexcept {
    _impl->preparation.finish();
}

const SurfaceClosureSet &
SurfaceClosurePopulationCollector::closures() const noexcept {
    return _impl->closures;
}

SurfacePreparation SurfaceClosurePopulationCollector::preparation(
    Float3 emission) const noexcept {
    return _impl->preparation.preparation(std::move(emission));
}

} // namespace psycles::luisa_backend
