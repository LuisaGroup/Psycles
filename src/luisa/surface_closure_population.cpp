#include <psycles/luisa/surface_closure_population.h>

#include "surface_preparation_accumulator.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_set.h>

#include <utility>

namespace psycles::luisa_backend {

struct SurfaceClosurePopulationCollector::Impl {
    SurfaceClosureSet closures;
    detail::SurfacePreparationAccumulator preparation;
    UInt transparent_index;
    Float3 shading_normal;

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
              aov_operation_value,
              detail::RuntimeFlagReductionMode::retained_state},
          transparent_index{
              static_cast<std::uint32_t>(closures.capacity())},
          shading_normal{point_value.shading_normal} {}
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
    _impl->transparent_index = static_cast<std::uint32_t>(
        _impl->closures.capacity());
    _impl->shading_normal = shading_normal;
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

bool SurfaceClosurePopulationCollector::
    supports_transparent_closure_finalization() const noexcept {
    return true;
}

void SurfaceClosurePopulationCollector::
    begin_transparent_closure(
    const SurfaceClosureRecord &closure) noexcept {
    const auto allocated =
        (closure.closure_type == cycles_closure::type_transparent) &
        (closure.allocation_weight >=
         cycles_closure::closure_weight_cutoff);
    $if(allocated) {
        // bsdf_transparent_setup updates ShaderData identity and extinction
        // before closure_alloc. Observe that state even when capacity is
        // already exhausted and no physical slot can be retained.
        _impl->preparation.begin_transparent_setup(closure);
        // The capacity value is outside the initialized index domain [0, count)
        // and is therefore an exact not-retained sentinel. This represents the
        // same sum state as a separate Boolean without extending another live
        // scalar through surface population.
        $if(_impl->transparent_index ==
            static_cast<std::uint32_t>(_impl->closures.capacity())) {
            _impl->closures.append(closure, [&] {
                _impl->transparent_index =
                    _impl->closures.count();
                _impl->preparation.retain_transparent_slot();
            });
        };
    };
}

void SurfaceClosurePopulationCollector::
    finalize_transparent_closure(
    Expr<luisa::float3> weight,
    Expr<float> sample_weight) noexcept {
    $if(_impl->transparent_index <
        static_cast<std::uint32_t>(_impl->closures.capacity())) {
        _impl->closures.finalize_physical_transparent(
            _impl->transparent_index,
            weight,
            sample_weight,
            _impl->shading_normal);
    };
    _impl->preparation.finalize_transparent_setup(weight);
}

void SurfaceClosurePopulationCollector::finish() noexcept {
    _impl->preparation.finish();
}

const SurfaceClosureSet &
SurfaceClosurePopulationCollector::closures() const noexcept {
    return _impl->closures;
}

SurfaceClosurePopulationState
SurfaceClosurePopulationCollector::runtime_state() const noexcept {
    return SurfaceClosurePopulationState{
        _impl->preparation.runtime_flags()};
}

SurfacePreparation SurfaceClosurePopulationCollector::preparation(
    Float3 emission) const noexcept {
    return _impl->preparation.preparation(std::move(emission));
}

} // namespace psycles::luisa_backend
