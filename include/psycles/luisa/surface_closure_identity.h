#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_identity.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface.h>
#include <psycles/luisa/surface_closure_reachability.h>

namespace psycles::luisa_backend::detail {

// The unique setup -> retained-ABI boundary. Every closure setup supplies its
// exact successful type and Fresnel family here; failed setup canonicalizes
// both discriminants. Pack/eval/sample code is forbidden from reconstructing
// either tag from authoring kind/lobe.
inline void finalize_cycles_closure_identity(
    SurfaceClosureRecord &closure,
    UInt successful_type,
    UInt successful_fresnel = static_cast<std::uint32_t>(
        cycles_closure::MicrofacetFresnel::none)) noexcept {
    closure.closure_type = luisa::compute::select(
        UInt{cycles_closure::type_none},
        successful_type,
        closure.setup_valid);
    closure.microfacet_fresnel = luisa::compute::select(
        UInt{static_cast<std::uint32_t>(
            cycles_closure::MicrofacetFresnel::none)},
        successful_fresnel,
        closure.setup_valid);
}

// Host/JIT-specialized setup classification. `kind` and `lobe` are immutable
// graph metadata, so this switch emits only the device choices that genuinely
// remain dynamic for that closure family (rough/smooth, distribution, or
// BSSRDF method). This is the Luisa multistage equivalent of Cycles assigning
// `sc->type` inside each bsdf_*_setup function.
inline void finalize_cycles_closure_identity(
    SurfaceClosureRecord &closure,
    SurfaceClosureKind kind,
    SurfaceClosureLobe lobe = SurfaceClosureLobe::none) noexcept {
    using Fresnel = cycles_closure::MicrofacetFresnel;
    UInt type = cycles_closure::type_none;
    UInt fresnel = static_cast<std::uint32_t>(Fresnel::none);
    switch (kind) {
        case SurfaceClosureKind::none:
            break;
        case SurfaceClosureKind::diffuse:
            type = luisa::compute::select(
                UInt{cycles_closure::type_oren_nayar},
                UInt{cycles_closure::type_diffuse},
                closure.roughness < 1.0e-5f);
            break;
        case SurfaceClosureKind::translucent:
            type = cycles_closure::type_translucent;
            break;
        case SurfaceClosureKind::rough_translucent:
            type = cycles_closure::type_rough_translucent;
            break;
        case SurfaceClosureKind::principled:
            if (lobe == SurfaceClosureLobe::sheen) {
                type = cycles_closure::type_sheen;
            } else if (lobe == SurfaceClosureLobe::transmission) {
                type = cycles_closure::type_microfacet_ggx_glass;
                fresnel = static_cast<std::uint32_t>(
                    Fresnel::generalized_schlick);
            } else {
                type = cycles_closure::type_microfacet_ggx;
                if (lobe == SurfaceClosureLobe::metallic) {
                    fresnel = static_cast<std::uint32_t>(
                        Fresnel::f82_tint);
                } else if (lobe == SurfaceClosureLobe::coat) {
                    fresnel = static_cast<std::uint32_t>(
                        Fresnel::dielectric);
                } else if (lobe == SurfaceClosureLobe::dielectric) {
                    fresnel = static_cast<std::uint32_t>(
                        Fresnel::generalized_schlick);
                }
            }
            break;
        case SurfaceClosureKind::glossy:
        case SurfaceClosureKind::metallic_f82:
        case SurfaceClosureKind::metallic_conductor:
            type = luisa::compute::select(
                UInt{cycles_closure::type_microfacet_ggx},
                UInt{cycles_closure::type_microfacet_beckmann},
                closure.beckmann);
            if (kind == SurfaceClosureKind::metallic_f82) {
                fresnel = static_cast<std::uint32_t>(Fresnel::f82_tint);
            } else if (kind == SurfaceClosureKind::metallic_conductor) {
                fresnel = static_cast<std::uint32_t>(Fresnel::conductor);
            }
            break;
        case SurfaceClosureKind::sheen_microfiber:
            type = cycles_closure::type_sheen;
            break;
        case SurfaceClosureKind::sheen_ashikhmin:
            type = cycles_closure::type_ashikhmin_velvet;
            break;
        case SurfaceClosureKind::hair_reflection:
            type = cycles_closure::type_hair_reflection;
            break;
        case SurfaceClosureKind::hair_transmission:
            type = cycles_closure::type_hair_transmission;
            break;
        case SurfaceClosureKind::glass: {
            type = luisa::compute::select(
                UInt{cycles_closure::type_microfacet_ggx_glass},
                UInt{cycles_closure::type_microfacet_beckmann_glass},
                closure.beckmann);
            fresnel = static_cast<std::uint32_t>(
                Fresnel::generalized_schlick);
            break;
        }
        case SurfaceClosureKind::transparent:
            type = cycles_closure::type_transparent;
            break;
        case SurfaceClosureKind::refraction:
            type = luisa::compute::select(
                UInt{cycles_closure::type_microfacet_ggx_refraction},
                UInt{cycles_closure::type_microfacet_beckmann_refraction},
                closure.beckmann);
            break;
        case SurfaceClosureKind::bssrdf:
            type = cycles_closure::type_bssrdf_random_walk;
            type = luisa::compute::select(
                type,
                UInt{cycles_closure::type_bssrdf_burley},
                closure.bssrdf_method == static_cast<std::uint32_t>(
                    SurfaceBssrdfMethod::burley));
            type = luisa::compute::select(
                type,
                UInt{cycles_closure::type_bssrdf_random_walk_legacy},
                closure.bssrdf_method == static_cast<std::uint32_t>(
                    SurfaceBssrdfMethod::random_walk_legacy));
            type = luisa::compute::select(
                type,
                UInt{cycles_closure::type_bssrdf_random_walk_skin},
                closure.bssrdf_method == static_cast<std::uint32_t>(
                    SurfaceBssrdfMethod::random_walk_skin));
            break;
        case SurfaceClosureKind::thin_glass_transmission:
            type = cycles_closure::type_thin_glass_transmission;
            break;
    }
    finalize_cycles_closure_identity(closure, type, fresnel);
}

[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureRecord &closure) noexcept;

[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness = 0.0f,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;
// Runtime flags for an already-retained ShaderClosure. Prefix membership
// proves allocation and type_none is the complete setup-failure state, so no
// authoring identity or compatibility flags participate in this overload.
[[nodiscard]] UInt cycles_runtime_flags(
    UInt closure_type,
    Float roughness,
    Float glossy_filter_roughness,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;

} // namespace psycles::luisa_backend::detail
