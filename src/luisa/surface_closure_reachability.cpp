#include <psycles/luisa/surface_closure_reachability.h>

#include <psycles/compiler/surface_program.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] constexpr std::uint32_t
operation_bit(compiler::ClosureOperation operation) noexcept {
    return std::uint32_t{1u} << static_cast<std::uint32_t>(operation);
}

[[nodiscard]] constexpr auto known_operation_mask() noexcept {
    using compiler::ClosureOperation;
    return operation_bit(ClosureOperation::null_closure) |
           operation_bit(ClosureOperation::diffuse) |
           operation_bit(ClosureOperation::translucent) |
           operation_bit(ClosureOperation::principled) |
           operation_bit(ClosureOperation::glossy) |
           operation_bit(ClosureOperation::metallic_f82) |
           operation_bit(ClosureOperation::metallic_conductor) |
           operation_bit(ClosureOperation::sheen_microfiber) |
           operation_bit(ClosureOperation::sheen_ashikhmin) |
           operation_bit(ClosureOperation::hair_reflection) |
           operation_bit(ClosureOperation::hair_transmission) |
           operation_bit(ClosureOperation::glass) |
           operation_bit(ClosureOperation::emission) |
           operation_bit(ClosureOperation::transparent) |
           operation_bit(ClosureOperation::subsurface) |
           operation_bit(ClosureOperation::add) |
           operation_bit(ClosureOperation::mix) |
           operation_bit(ClosureOperation::refraction);
}

[[nodiscard]] constexpr auto known_principled_feature_mask() noexcept {
    using compiler::principled_closure_feature_bit;
    using compiler::PrincipledClosureFeature;
    return principled_closure_feature_bit(PrincipledClosureFeature::alpha) |
           principled_closure_feature_bit(PrincipledClosureFeature::sheen) |
           principled_closure_feature_bit(PrincipledClosureFeature::coat) |
           principled_closure_feature_bit(PrincipledClosureFeature::metallic) |
           principled_closure_feature_bit(
               PrincipledClosureFeature::thick_transmission) |
           principled_closure_feature_bit(
               PrincipledClosureFeature::thin_transmission) |
           principled_closure_feature_bit(PrincipledClosureFeature::dielectric) |
           principled_closure_feature_bit(
               PrincipledClosureFeature::thick_subsurface) |
           principled_closure_feature_bit(
               PrincipledClosureFeature::thin_subsurface) |
           principled_closure_feature_bit(PrincipledClosureFeature::diffuse) |
           principled_closure_feature_bit(PrincipledClosureFeature::emission);
}

} // namespace

SurfaceClosureReachability
reachable_surface_closures(std::uint32_t closure_operations,
                           std::uint32_t principled_features,
                           std::uint32_t anisotropic_closure_operations,
                           std::uint32_t anisotropic_principled_features,
                           std::uint32_t thin_film_closure_operations,
                           std::uint32_t thin_film_principled_features) noexcept {
    static_assert(
        static_cast<std::uint32_t>(compiler::ClosureOperation::refraction) < 32u);
    static_assert(static_cast<std::uint32_t>(
                      SurfaceClosureKind::thin_glass_transmission) < 32u);
    static_assert(static_cast<std::uint32_t>(SurfaceClosureLobe::dielectric) <
                  32u);

    constexpr auto anisotropic_operation_mask =
        operation_bit(compiler::ClosureOperation::principled) |
        operation_bit(compiler::ClosureOperation::glossy) |
        operation_bit(compiler::ClosureOperation::metallic_f82) |
        operation_bit(compiler::ClosureOperation::metallic_conductor);
    constexpr auto thin_film_operation_mask =
        operation_bit(compiler::ClosureOperation::principled) |
        operation_bit(compiler::ClosureOperation::glass) |
        operation_bit(compiler::ClosureOperation::metallic_f82) |
        operation_bit(compiler::ClosureOperation::metallic_conductor);
    if ((closure_operations & ~known_operation_mask()) != 0u ||
        (principled_features & ~known_principled_feature_mask()) != 0u ||
        (anisotropic_closure_operations & ~anisotropic_operation_mask) != 0u ||
        (anisotropic_closure_operations & ~closure_operations) != 0u ||
        (anisotropic_principled_features &
         ~known_principled_feature_mask()) != 0u ||
        (anisotropic_principled_features & ~principled_features) != 0u ||
        (anisotropic_principled_features != 0u &&
         (anisotropic_closure_operations &
          operation_bit(compiler::ClosureOperation::principled)) == 0u) ||
        (thin_film_closure_operations & ~thin_film_operation_mask) != 0u ||
        (thin_film_closure_operations & ~closure_operations) != 0u ||
        (thin_film_principled_features &
         ~known_principled_feature_mask()) != 0u ||
        (thin_film_principled_features & ~principled_features) != 0u ||
        (thin_film_principled_features != 0u &&
         (thin_film_closure_operations &
          operation_bit(compiler::ClosureOperation::principled)) == 0u)) {
        return all_surface_closure_reachability;
    }

    using compiler::ClosureOperation;
    using compiler::principled_closure_feature_bit;
    using compiler::PrincipledClosureFeature;
    SurfaceClosureReachability result;
    const auto has_operation =
        [closure_operations](ClosureOperation operation) noexcept {
            return (closure_operations & operation_bit(operation)) != 0u;
        };
    const auto has_feature = [principled_features](
                                 PrincipledClosureFeature feature) noexcept {
        return (principled_features & principled_closure_feature_bit(feature)) !=
               0u;
    };
    const auto has_anisotropic_operation =
        [anisotropic_closure_operations](ClosureOperation operation) noexcept {
            return (anisotropic_closure_operations & operation_bit(operation)) !=
                   0u;
        };
    const auto has_anisotropic_principled_feature =
        [anisotropic_principled_features](
            PrincipledClosureFeature feature) noexcept {
            return (anisotropic_principled_features &
                    principled_closure_feature_bit(feature)) != 0u;
        };
    const auto has_thin_film_operation =
        [thin_film_closure_operations](ClosureOperation operation) noexcept {
            return (thin_film_closure_operations & operation_bit(operation)) !=
                   0u;
        };
    const auto has_thin_film_principled_feature =
        [thin_film_principled_features](
            PrincipledClosureFeature feature) noexcept {
            return (thin_film_principled_features &
                    principled_closure_feature_bit(feature)) != 0u;
        };
    const auto add_kind = [&result](SurfaceClosureKind kind) noexcept {
        result.kinds |= surface_closure_kind_bit(kind);
    };
    const auto add_principled_lobe =
        [&result, &add_kind](SurfaceClosureLobe lobe) noexcept {
            add_kind(SurfaceClosureKind::principled);
            result.principled_lobes |= surface_closure_lobe_bit(lobe);
        };

    if (has_operation(ClosureOperation::diffuse)) {
        add_kind(SurfaceClosureKind::diffuse);
    }
    if (has_operation(ClosureOperation::translucent)) {
        add_kind(SurfaceClosureKind::translucent);
    }
    if (has_operation(ClosureOperation::glossy)) {
        add_kind(SurfaceClosureKind::glossy);
        if (has_anisotropic_operation(ClosureOperation::glossy)) {
            result.anisotropic_microfacet_kinds |=
                surface_closure_kind_bit(SurfaceClosureKind::glossy);
        }
    }
    const auto add_metallic = [&](ClosureOperation operation,
                                  SurfaceClosureKind kind) noexcept {
        if (!has_operation(operation)) {
            return;
        }
        add_kind(kind);
        if (has_anisotropic_operation(operation)) {
            result.anisotropic_microfacet_kinds |=
                surface_closure_kind_bit(kind);
        }
        if (has_thin_film_operation(operation)) {
            result.thin_film_kinds |= surface_closure_kind_bit(kind);
        }
    };
    add_metallic(ClosureOperation::metallic_f82,
                 SurfaceClosureKind::metallic_f82);
    add_metallic(ClosureOperation::metallic_conductor,
                 SurfaceClosureKind::metallic_conductor);
    if (has_operation(ClosureOperation::sheen_microfiber)) {
        add_kind(SurfaceClosureKind::sheen_microfiber);
    }
    if (has_operation(ClosureOperation::sheen_ashikhmin)) {
        add_kind(SurfaceClosureKind::sheen_ashikhmin);
    }
    if (has_operation(ClosureOperation::hair_reflection)) {
        add_kind(SurfaceClosureKind::hair_reflection);
    }
    if (has_operation(ClosureOperation::hair_transmission)) {
        add_kind(SurfaceClosureKind::hair_transmission);
    }
    if (has_operation(ClosureOperation::glass)) {
        add_kind(SurfaceClosureKind::glass);
        if (has_thin_film_operation(ClosureOperation::glass)) {
            result.thin_film_kinds |=
                surface_closure_kind_bit(SurfaceClosureKind::glass);
        }
    }
    if (has_operation(ClosureOperation::transparent)) {
        add_kind(SurfaceClosureKind::transparent);
    }
    if (has_operation(ClosureOperation::refraction)) {
        add_kind(SurfaceClosureKind::refraction);
    }
    if (has_operation(ClosureOperation::subsurface)) {
        add_kind(SurfaceClosureKind::bssrdf);
        add_kind(SurfaceClosureKind::diffuse);
    }

    if (has_feature(PrincipledClosureFeature::alpha)) {
        add_kind(SurfaceClosureKind::transparent);
    }
    if (has_feature(PrincipledClosureFeature::sheen)) {
        add_principled_lobe(SurfaceClosureLobe::sheen);
    }
    if (has_feature(PrincipledClosureFeature::coat)) {
        add_principled_lobe(SurfaceClosureLobe::coat);
    }
    if (has_feature(PrincipledClosureFeature::metallic)) {
        add_principled_lobe(SurfaceClosureLobe::metallic);
        if (has_thin_film_principled_feature(
                PrincipledClosureFeature::metallic)) {
            result.thin_film_principled_lobes |=
                surface_closure_lobe_bit(SurfaceClosureLobe::metallic);
        }
        if (has_anisotropic_principled_feature(
                PrincipledClosureFeature::metallic)) {
            result.anisotropic_microfacet_kinds |=
                surface_closure_kind_bit(SurfaceClosureKind::principled);
        }
    }
    if (has_feature(PrincipledClosureFeature::thick_transmission)) {
        add_kind(SurfaceClosureKind::glass);
        if (has_thin_film_principled_feature(
                PrincipledClosureFeature::thick_transmission)) {
            result.thin_film_kinds |=
                surface_closure_kind_bit(SurfaceClosureKind::glass);
        }
    }
    if (has_feature(PrincipledClosureFeature::thin_transmission)) {
        add_kind(SurfaceClosureKind::glossy);
        add_kind(SurfaceClosureKind::thin_glass_transmission);
        add_kind(SurfaceClosureKind::transparent);
    }
    if (has_feature(PrincipledClosureFeature::dielectric)) {
        add_principled_lobe(SurfaceClosureLobe::dielectric);
        if (has_thin_film_principled_feature(
                PrincipledClosureFeature::dielectric)) {
            result.thin_film_principled_lobes |=
                surface_closure_lobe_bit(SurfaceClosureLobe::dielectric);
        }
        if (has_anisotropic_principled_feature(
                PrincipledClosureFeature::dielectric)) {
            result.anisotropic_microfacet_kinds |=
                surface_closure_kind_bit(SurfaceClosureKind::principled);
        }
    }
    if (has_feature(PrincipledClosureFeature::thick_subsurface)) {
        add_kind(SurfaceClosureKind::bssrdf);
        add_kind(SurfaceClosureKind::diffuse);
    }
    if (has_feature(PrincipledClosureFeature::thin_subsurface)) {
        add_kind(SurfaceClosureKind::diffuse);
        add_kind(SurfaceClosureKind::translucent);
        add_kind(SurfaceClosureKind::rough_translucent);
    }
    if (has_feature(PrincipledClosureFeature::diffuse)) {
        add_kind(SurfaceClosureKind::diffuse);
    }
    return result;
}

} // namespace psycles::luisa_backend
