#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/volume_phase_set.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <cstdint>

#include <psycles/luisa/cycles_volume_phase.h>
#include <psycles/luisa/surface.h>

#include <luisa/dsl/local.h>

namespace psycles::luisa_backend {

inline constexpr std::uint32_t
    maximum_volume_phase_closures = 8u;

struct VolumePhaseEntry {
    UInt type;
    Float3 parameters;
    Float3 weight;
    Float sample_weight;
    Bool valid;
};

struct VolumePhaseSetEvaluation {
    Float value;
    Float pdf;
    Float sample_weight;
    Bool valid;
};

struct VolumePhaseSetSample {
    Float3 direction;
    Float pdf;
    Float sampled_roughness;
    Float selection_rescaled;
    UInt closure_index;
    UInt closure_type;
    Bool valid;
};

// Device-local closure array built through the host-stage
// VolumePhaseCollector interface. Capacity is a scene/JIT-stage value: common
// one-phase shaders do not pay for Cycles' global maximum, while complex
// graphs can retain every closure until the explicit merge/truncate step.
class VolumePhaseSet final : public VolumePhaseCollector {

  private:
    std::size_t _capacity;
    luisa::compute::Local<luisa::uint> _types;
    luisa::compute::Local<luisa::float4> _parameters;
    luisa::compute::Local<luisa::float4> _weights;
    UInt _count;

  public:
    explicit VolumePhaseSet(std::size_t capacity) noexcept;

    VolumePhaseSet(const VolumePhaseSet &) = delete;
    VolumePhaseSet(VolumePhaseSet &&) = delete;
    VolumePhaseSet &operator=(const VolumePhaseSet &) = delete;
    VolumePhaseSet &operator=(VolumePhaseSet &&) = delete;

    void add(
        const cycles_volume_phase::Closure &phase,
        Float3 weight) noexcept override;

    [[nodiscard]] UInt count() const noexcept;
    [[nodiscard]] VolumePhaseEntry entry(
        UInt index) const noexcept;

    // Cycles merges after each stack entry. Equality is exact and closure
    // family-specific setup has already canonicalized unused parameters to
    // zero, so an exact three-component comparison matches
    // volume_phase_equal().
    void merge_equal() noexcept;

    void truncate(
        std::uint32_t maximum =
            maximum_volume_phase_closures) noexcept;

    [[nodiscard]] VolumePhaseSetEvaluation evaluate(
        Float3 axis,
        Float3 outgoing) const noexcept;

    [[nodiscard]] VolumePhaseSetSample sample(
        Float3 axis,
        Float2 random) const noexcept;
};

}// namespace psycles::luisa_backend
