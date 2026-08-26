#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_set.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <cstdint>
#include <functional>

#include <psycles/luisa/surface.h>
#include <psycles/luisa/surface_closure_physical_blocks.h>

#include <luisa/dsl/local.h>

namespace psycles::luisa_backend {

// Host-stage projection of Cycles' post-shader closure records. Each profile
// retains the allocation identity plus exactly the fields consumed by the
// named device-stage operation. The complete profile is the lossless form
// used by scattering and round-trip diagnostics.
enum class SurfaceClosureStorageProfile : std::uint32_t {
    complete,
    // Exact dependency cut consumed by BSDF evaluation and sampling after
    // runtime flags and camera AOVs have already been reduced. This is a
    // lossless SurfaceClosurePhysicalRecord, not a baked BSDF response.
    physical,
    runtime_flags,
    closure_trace,
    aov,
};

// Device-local counterpart of Cycles' ShaderData closure array. GraphSurface
// emits records through the host-stage SurfaceClosureCollector interface;
// this class alone owns allocation-budget truncation and runtime indexing.
// Complete scattering records use four matrix blocks per entry so one
// transactional append does not scalarize into thirteen independently
// branched stores. Projected diagnostic profiles remain SoA and therefore
// load only the fields consumed by their evaluator.
class SurfaceClosureSet final : public SurfaceClosureCollector {

  private:
    std::size_t _capacity;
    SurfaceClosureStorageProfile _profile;
    luisa::compute::Local<luisa::float4x4> _complete_0;
    luisa::compute::Local<luisa::float4x4> _complete_1;
    luisa::compute::Local<luisa::float4x4> _complete_2;
    luisa::compute::Local<luisa::float4x4> _complete_3;
    luisa::compute::Local<luisa::float4x4> _physical_0;
    luisa::compute::Local<luisa::float4x4> _physical_1;
    luisa::compute::Local<luisa::uint4> _identity;
    luisa::compute::Local<luisa::float4> _weight;
    luisa::compute::Local<luisa::float4> _albedo;
    luisa::compute::Local<luisa::float4> _reflection_albedo;
    luisa::compute::Local<luisa::float4> _transmission_albedo;
    luisa::compute::Local<luisa::float4> _color;
    luisa::compute::Local<luisa::float4> _normal;
    luisa::compute::Local<luisa::float4> _specular_tint;
    luisa::compute::Local<luisa::float4> _evaluation_scale;
    luisa::compute::Local<luisa::float4> _fresnel_f0;
    luisa::compute::Local<luisa::float4> _fresnel_f90;
    luisa::compute::Local<luisa::float4> _reflection_tint;
    luisa::compute::Local<luisa::float4> _transmission_tint;
    UInt _count;

    void append_impl(
        const SurfaceClosureRecord &closure,
        const std::function<void()> *on_retained) noexcept;

  public:
    explicit SurfaceClosureSet(
        std::size_t capacity,
        SurfaceClosureStorageProfile profile =
            SurfaceClosureStorageProfile::complete) noexcept;

    SurfaceClosureSet(const SurfaceClosureSet &) = delete;
    SurfaceClosureSet(SurfaceClosureSet &&) = delete;
    SurfaceClosureSet &operator=(const SurfaceClosureSet &) = delete;
    SurfaceClosureSet &operator=(SurfaceClosureSet &&) = delete;

    void add(
        const SurfaceClosureRecord &closure) noexcept override;

    // Transactionally appends and records on_retained in the same device
    // conditional. Storage, the streaming fold, and count advancement thus
    // observe exactly the same retained source-order subsequence without a
    // second predicate or a mutable-count snapshot.
    void append(
        const SurfaceClosureRecord &closure,
        const std::function<void()> &on_retained) noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] SurfaceClosureStorageProfile profile() const noexcept;
    [[nodiscard]] UInt count() const noexcept;

    // Staged access to the physical tagged union. Both methods require the
    // physical profile and index < count(); keeping them unchecked prevents a
    // redundant validity select in the two loops which already establish that
    // invariant. Reading common never touches the payload Local. Reading the
    // payload is deliberately a separate operation so callers can record it
    // under a runtime family branch.
    [[nodiscard]] SurfaceClosurePhysicalCommonRecord
    physical_common_entry_unchecked(UInt index) const noexcept;
    [[nodiscard]] SurfaceClosurePhysicalRecord
    physical_payload_entry_unchecked(
        UInt index,
        const SurfaceClosurePhysicalCommonRecord &common) const noexcept;

    [[nodiscard]] SurfaceClosureRecord entry(
        UInt index) const noexcept;
};

}// namespace psycles::luisa_backend
