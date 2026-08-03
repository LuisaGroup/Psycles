#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_set.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <cstdint>

#include <psycles/luisa/surface.h>

#include <luisa/dsl/local.h>

namespace psycles::luisa_backend {

// Device-local counterpart of Cycles' ShaderData closure array. GraphSurface
// emits records through the host-stage SurfaceClosureCollector interface;
// this class alone owns allocation-budget truncation and runtime indexing.
// The structure is SoA because Luisa Local arrays are backend-independent and
// allow each shared evaluator to load only the fields it consumes.
class SurfaceClosureSet final : public SurfaceClosureCollector {

  private:
    std::size_t _capacity;
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

  public:
    explicit SurfaceClosureSet(std::size_t capacity) noexcept;

    SurfaceClosureSet(const SurfaceClosureSet &) = delete;
    SurfaceClosureSet(SurfaceClosureSet &&) = delete;
    SurfaceClosureSet &operator=(const SurfaceClosureSet &) = delete;
    SurfaceClosureSet &operator=(SurfaceClosureSet &&) = delete;

    void add(
        const SurfaceClosureRecord &closure) noexcept override;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] UInt count() const noexcept;
    [[nodiscard]] SurfaceClosureRecord entry(
        UInt index) const noexcept;
};

}// namespace psycles::luisa_backend
