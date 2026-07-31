#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/homogeneous_volume_transport.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>
#include <psycles/luisa/volume_direct_sampling.h>

namespace psycles::luisa_backend {

struct VolumeChannelSample {
    UInt channel;
    Float3 pdf;
    Float random;
    Bool matched;
};

struct HomogeneousVolumeSample {
    Float3 transmittance;
    Float3 emission;
    Float3 throughput;
    Float3 scatter_probability;
    Float3 channel_pdf;
    Float distance;
    Float event_pdf;
    Float scatter_random;
    Float reservoir_random;
    UInt channel;
    Bool scattered;
    Bool active;
};

struct HomogeneousVolumeDirectSample {
    Float3 transmittance;
    Float3 throughput;
    Float3 channel_pdf;
    Float distance;
    Float distance_pdf;
    Float equiangular_pdf;
    Float mis_weight;
    UInt channel;
    UInt sample_method;
    Bool use_mis;
    Bool scattered;
};

// Host-stage Luisa AST component for Cycles' analytic homogeneous-volume
// estimator. The caller supplies the two Cycles random dimensions separately:
// PRNG_VOLUME_SCATTER_DISTANCE and PRNG_VOLUME_RESERVOIR. Phase selection uses
// PRNG_VOLUME_PHASE after a real collision and is deliberately kept outside
// this distance estimator.
class HomogeneousVolumeTransport {

  private:
    [[nodiscard]] static Float3 _safe_divide(
        Float3 numerator,
        Float3 denominator,
        float fallback = 0.0f) noexcept;
    [[nodiscard]] static Float3 _safe_divide(
        Float3 numerator,
        Float denominator) noexcept;

  public:
    [[nodiscard]] Float3 transmittance(
        Float3 sigma_t,
        Float distance) const noexcept;

    [[nodiscard]] Float3 emission_integral(
        const VolumeCoefficients &coefficients,
        Float distance) const noexcept;

    [[nodiscard]] Float3 channel_pdf(
        Float3 albedo,
        Float3 throughput) const noexcept;

    [[nodiscard]] VolumeChannelSample sample_channel(
        Float3 albedo,
        Float3 throughput,
        Float random) const noexcept;

    [[nodiscard]] Float bounded_exponential_sample(
        Float random,
        Float rate,
        Float minimum,
        Float maximum) const noexcept;

    [[nodiscard]] Float3 bounded_exponential_pdf(
        Float distance,
        Float3 rate,
        Float minimum,
        Float maximum) const noexcept;

    [[nodiscard]] Float3 unguided_scatter_probability(
        const VolumeCoefficients &coefficients,
        Float distance,
        Bool terminate) const noexcept;

    [[nodiscard]] HomogeneousVolumeSample sample(
        const VolumeCoefficients &coefficients,
        Float distance,
        Float3 throughput,
        Float scatter_random,
        Float channel_random,
        Bool terminate) const noexcept;

    // VSPG changes only the probability of selecting the scatter-vs-transmit
    // estimator. Supplying that probability explicitly keeps the estimator
    // measure identical while allowing the path stage to source Cycles'
    // history-dependent primary-ray probability later.
    [[nodiscard]] HomogeneousVolumeSample sample_with_probability(
        const VolumeCoefficients &coefficients,
        Float distance,
        Float3 throughput,
        Float scatter_random,
        Float channel_random,
        Float3 scatter_probability,
        Bool terminate) const noexcept;

    // Cycles samples direct volume scattering independently from the
    // scatter-vs-transmit decision used by the continuation path. Distant
    // emitters and the environment force this distance measure even when a
    // material requests equiangular or multiple-importance sampling.
    [[nodiscard]] HomogeneousVolumeDirectSample
    sample_direct_distance(
        const VolumeCoefficients &coefficients,
        Float distance,
        Float3 throughput,
        Float scatter_random,
        Float reservoir_random,
        Bool enabled) const noexcept;

    // General finite-emitter direct estimator. `sampling.random` is the
    // technique-rescaled PRNG_VOLUME_SCATTER_DISTANCE value saved before
    // indirect free-flight sampling; `scatter_random` and `reservoir_random`
    // are the values left after that sampling. This distinction is required
    // by Cycles' coupled random-number measure.
    [[nodiscard]] HomogeneousVolumeDirectSample
    sample_direct(
        const VolumeCoefficients &coefficients,
        Float distance,
        Float3 throughput,
        Float scatter_random,
        Float reservoir_random,
        const VolumeDirectSamplingState &sampling,
        Float3 ray_origin,
        Float3 ray_direction,
        const VolumeEquiangularCoefficients
            &equiangular) const noexcept;
};

}// namespace psycles::luisa_backend
