#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/stacked_volume.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>
#include <psycles/luisa/volume_phase_set.h>
#include <psycles/luisa/volume_stack.h>

namespace psycles::luisa_backend {

// Path fields shared by every medium in a stack. Object-dependent coordinates
// and density scaling are deliberately supplied by a host-stage provider:
// mesh volumes, world volumes, motion transforms, and future grid volumes can
// construct exact Cycles ShaderData semantics without changing aggregation.
struct VolumeShadingState {
    Float3 position;
    Float3 incoming;
    UInt ray_visibility;
    UInt ray_events;
    UInt ray_depth;
    UInt diffuse_depth;
    UInt glossy_depth;
    UInt transparent_depth;
    UInt transmission_depth;
    Float ray_length;
    Float time;
};

struct VolumeStackEntryShading {
    SurfacePoint point;
    // Cycles' object_volume_density is an entry-level scale: it is constant
    // over all spatial evaluations of one object and is applied again by
    // runtime transport after being divided out of majorant metadata.
    Float object_density;
};

class VolumeStackEntryPointProvider {

  public:
    virtual ~VolumeStackEntryPointProvider() noexcept = default;

    // Cycles preserves every entry in a copied shadow stack, including
    // objects hidden from shadow rays, so majorant traversal and RNG
    // consumption retain their original structure. The visibility test is
    // applied only when evaluating the entry's raw closure.
    [[nodiscard]] virtual Bool should_evaluate(
        const VolumeStackEntry &entry,
        const VolumeShadingState &state) const noexcept {
        static_cast<void>(entry);
        static_cast<void>(state);
        return true;
    }

    // Cycles stores this as an entry-level object property. Keeping it
    // independently queryable lets majorant traversal apply the same scale
    // without constructing an otherwise unused shading point.
    [[nodiscard]] virtual Float object_density(
        const VolumeStackEntry &entry) const noexcept {
        static_cast<void>(entry);
        return 1.0f;
    }

    [[nodiscard]] virtual VolumeStackEntryShading
    emit(const VolumeStackEntry &entry,
         const VolumeShadingState &state) const noexcept = 0;
};

// Visits every original volume-stack entry in order at one spatial point.
// Entry visibility can suppress raw closure evaluation without changing the
// stack itself. Coefficients are additive. Raw phase closures share one
// collector and are merged only after the second and subsequent stack
// entries, matching Cycles' volume_shader_eval() rather than pre-combining
// material data.
class StackedVolumeEvaluator {

  private:
    const SurfaceDispatch &_surfaces;
    const VolumeStackEntryPointProvider &_points;

  public:
    StackedVolumeEvaluator(
        const SurfaceDispatch &surfaces,
        const VolumeStackEntryPointProvider &points) noexcept;

    [[nodiscard]] VolumeCoefficients evaluate(
        const VolumeStack &stack,
        const ShaderServices &services,
        const VolumeShadingState &state,
        Bool evaluate_emission,
        VolumePhaseSet *phases = nullptr) const noexcept;
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeShadingState)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::VolumeStackEntryShading)
