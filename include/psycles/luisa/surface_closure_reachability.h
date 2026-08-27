#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_reachability.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

using SurfaceClosureKindMask = std::uint32_t;
using SurfaceClosureLobeMask = std::uint32_t;

[[nodiscard]] constexpr SurfaceClosureKindMask
surface_closure_kind_bit(SurfaceClosureKind kind) noexcept {
    return SurfaceClosureKindMask{1u} << static_cast<std::uint32_t>(kind);
}

[[nodiscard]] constexpr SurfaceClosureLobeMask
surface_closure_lobe_bit(SurfaceClosureLobe lobe) noexcept {
    return SurfaceClosureLobeMask{1u} << static_cast<std::uint32_t>(lobe);
}

inline constexpr auto all_surface_closure_kinds =
    surface_closure_kind_bit(SurfaceClosureKind::diffuse) |
    surface_closure_kind_bit(SurfaceClosureKind::translucent) |
    surface_closure_kind_bit(SurfaceClosureKind::principled) |
    surface_closure_kind_bit(SurfaceClosureKind::glossy) |
    surface_closure_kind_bit(SurfaceClosureKind::glass) |
    surface_closure_kind_bit(SurfaceClosureKind::transparent) |
    surface_closure_kind_bit(SurfaceClosureKind::refraction) |
    surface_closure_kind_bit(SurfaceClosureKind::bssrdf) |
    surface_closure_kind_bit(SurfaceClosureKind::rough_translucent) |
    surface_closure_kind_bit(SurfaceClosureKind::thin_glass_transmission);

inline constexpr auto all_surface_closure_lobes =
    surface_closure_lobe_bit(SurfaceClosureLobe::sheen) |
    surface_closure_lobe_bit(SurfaceClosureLobe::coat) |
    surface_closure_lobe_bit(SurfaceClosureLobe::metallic) |
    surface_closure_lobe_bit(SurfaceClosureLobe::transmission) |
    surface_closure_lobe_bit(SurfaceClosureLobe::dielectric);

// Abstract domain for host/JIT physical-closure reachability. The first
// component is a set of canonical SurfaceClosureKind values. The second
// refines the only kind whose directional algorithm also depends on its lobe
// tag. The third is the subset of reachable kinds whose microfacet state may
// be anisotropic. The partial order, join, and meet are component-wise subset,
// union, and intersection, reduced by the invariants
//
//   Principled not in kinds => principled_lobes is empty
//   anisotropic_microfacet_kinds is a subset of kinds.
//
// This is host/JIT capability metadata. It never classifies a device value or
// changes the authored closure graph.
struct SurfaceClosureReachability {
    SurfaceClosureKindMask kinds{};
    SurfaceClosureLobeMask principled_lobes{};
    SurfaceClosureKindMask anisotropic_microfacet_kinds{};

    [[nodiscard]] constexpr bool
    contains(SurfaceClosureKind kind) const noexcept {
        return (kinds & surface_closure_kind_bit(kind)) != 0u;
    }

    [[nodiscard]] constexpr bool
    contains_principled_lobe(SurfaceClosureLobe lobe) const noexcept {
        return contains(SurfaceClosureKind::principled) &&
               (principled_lobes & surface_closure_lobe_bit(lobe)) != 0u;
    }

    [[nodiscard]] constexpr bool
    contains_anisotropic_microfacet(
        SurfaceClosureKind kind) const noexcept {
        return contains(kind) &&
               (anisotropic_microfacet_kinds &
                surface_closure_kind_bit(kind)) != 0u;
    }

private:
    constexpr void normalize() noexcept {
        if (!contains(SurfaceClosureKind::principled)) {
            principled_lobes = 0u;
        }
        anisotropic_microfacet_kinds &= kinds;
    }

public:

    constexpr SurfaceClosureReachability &
    operator|=(SurfaceClosureReachability rhs) noexcept {
        kinds |= rhs.kinds;
        principled_lobes |= rhs.principled_lobes;
        anisotropic_microfacet_kinds |= rhs.anisotropic_microfacet_kinds;
        normalize();
        return *this;
    }

    [[nodiscard]] friend constexpr SurfaceClosureReachability
    operator|(SurfaceClosureReachability lhs,
              SurfaceClosureReachability rhs) noexcept {
        return lhs |= rhs;
    }

    constexpr SurfaceClosureReachability &
    operator&=(SurfaceClosureReachability rhs) noexcept {
        kinds &= rhs.kinds;
        principled_lobes &= rhs.principled_lobes;
        anisotropic_microfacet_kinds &=
            rhs.anisotropic_microfacet_kinds;
        normalize();
        return *this;
    }

    // Meet in the finite reachability lattice. Consumer-directed closure
    // elimination uses this to prove that a family handler cannot observe a
    // kind or Principled lobe outside the tag branch which dominates it.
    [[nodiscard]] friend constexpr SurfaceClosureReachability
    operator&(SurfaceClosureReachability lhs,
              SurfaceClosureReachability rhs) noexcept {
        return lhs &= rhs;
    }

    constexpr bool
    operator==(const SurfaceClosureReachability &) const noexcept = default;
};

inline constexpr auto all_surface_closure_reachability =
    SurfaceClosureReachability{.kinds = all_surface_closure_kinds,
                               .principled_lobes = all_surface_closure_lobes,
                               .anisotropic_microfacet_kinds =
                                   surface_closure_kind_bit(
                                       SurfaceClosureKind::principled) |
                                   surface_closure_kind_bit(
                                       SurfaceClosureKind::glossy)};

// Monotone transfer from the scene image's immutable closure-operation and
// Principled-feature sets to canonical physical identities. The final two
// inputs preserve the correlation between the anisotropy bit and the feature
// mask of the same bytecode leaf; using the scene-wide Principled union there
// would unsoundly conflate distinct material nodes. For recognized and
// consistent bits this is a union homomorphism: no authored parameter is
// inspected and no closure is pre-evaluated. Any unrecognized or inconsistent
// bit maps to top, so schema drift can only disable specialization rather than
// remove required shader code.
[[nodiscard]] SurfaceClosureReachability
reachable_surface_closures(std::uint32_t closure_operations,
                           std::uint32_t principled_features,
                           std::uint32_t anisotropic_closure_operations,
                           std::uint32_t anisotropic_principled_features) noexcept;

} // namespace psycles::luisa_backend
