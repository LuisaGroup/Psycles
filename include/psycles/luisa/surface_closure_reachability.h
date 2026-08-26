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
// tag. The partial order and join are component-wise subset and union.
struct SurfaceClosureReachability {
    SurfaceClosureKindMask kinds{};
    SurfaceClosureLobeMask principled_lobes{};

    [[nodiscard]] constexpr bool
    contains(SurfaceClosureKind kind) const noexcept {
        return (kinds & surface_closure_kind_bit(kind)) != 0u;
    }

    [[nodiscard]] constexpr bool
    contains_principled_lobe(SurfaceClosureLobe lobe) const noexcept {
        return contains(SurfaceClosureKind::principled) &&
               (principled_lobes & surface_closure_lobe_bit(lobe)) != 0u;
    }

    constexpr SurfaceClosureReachability &
    operator|=(SurfaceClosureReachability rhs) noexcept {
        kinds |= rhs.kinds;
        principled_lobes |= rhs.principled_lobes;
        return *this;
    }

    [[nodiscard]] friend constexpr SurfaceClosureReachability
    operator|(SurfaceClosureReachability lhs,
              SurfaceClosureReachability rhs) noexcept {
        return lhs |= rhs;
    }

    constexpr bool
    operator==(const SurfaceClosureReachability &) const noexcept = default;
};

inline constexpr auto all_surface_closure_reachability =
    SurfaceClosureReachability{.kinds = all_surface_closure_kinds,
                               .principled_lobes = all_surface_closure_lobes};

// Monotone transfer from the scene image's immutable closure-operation and
// Principled-feature sets to canonical physical identities. For recognized
// bits this is a union homomorphism: no authored parameter is inspected and no
// closure is pre-evaluated. Any unrecognized bit maps to top, so schema drift
// can only disable specialization rather than remove required shader code.
[[nodiscard]] SurfaceClosureReachability
reachable_surface_closures(std::uint32_t closure_operations,
                           std::uint32_t principled_features) noexcept;

} // namespace psycles::luisa_backend
