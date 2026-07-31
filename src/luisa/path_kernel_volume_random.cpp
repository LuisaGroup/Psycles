#include "path_kernel_volume_random.h"

#include <psycles/luisa/cycles_sampler.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <utility>

namespace psycles::luisa_backend::detail {

UInt PathVolumeTrackingRandomSource::_dimension(
    UInt rng_offset,
    std::uint32_t dimension) const noexcept {
    return cycles_sampler::path_state_dimension(
        rng_offset,
        dimension);
}

PathVolumeTrackingRandomSource::
    PathVolumeTrackingRandomSource(
        const luisa::compute::BufferFloat4
            &sobol_table,
        UInt sequence_size,
        UInt sample_index,
        UInt rng_hash) noexcept
    : _sobol_table{sobol_table},
      _sequence_size{
          std::move(sequence_size)},
      _sample_index{
          std::move(sample_index)},
      _rng_hash{std::move(rng_hash)} {}

Float PathVolumeTrackingRandomSource::
    scatter_distance(
        UInt rng_offset)
        const noexcept {
    return cycles_sampler::sample_1d(
        _sobol_table,
        _sequence_size,
        _sample_index,
        _rng_hash,
        _dimension(
            rng_offset,
            sampling::tabulated_sobol::
                volume_scatter_distance_dimension));
}

Float PathVolumeTrackingRandomSource::shade_offset(
    UInt rng_offset)
    const noexcept {
    return cycles_sampler::sample_2d(
               _sobol_table,
               _sequence_size,
               _sample_index,
               _rng_hash,
               _dimension(
                   rng_offset,
                   sampling::tabulated_sobol::
                       volume_shade_offset_dimension))
        .y;
}

Float PathVolumeTrackingRandomSource::
    expansion_order(
        UInt rng_offset)
        const noexcept {
    return cycles_sampler::
        sample_volume_expansion_order(
            _sobol_table,
            _sequence_size,
            _sample_index,
            rng_offset);
}

Float PathVolumeTrackingRandomSource::
    transmittance_shade_offset(
        UInt rng_offset)
        const noexcept {
    return cycles_sampler::sample_1d(
        _sobol_table,
        _sequence_size,
        _sample_index,
        _rng_hash,
        _dimension(
            rng_offset,
            sampling::tabulated_sobol::
                volume_shade_offset_dimension));
}

}// namespace psycles::luisa_backend::detail
