#include <psycles/luisa/surface_closure_set.h>

#include <algorithm>

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_blocks.h>
#include <psycles/luisa/surface_closure_physical_blocks.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

inline constexpr std::uint32_t setup_valid_bit = 1u << 0u;
inline constexpr std::uint32_t preserve_ggx_energy_bit = 1u << 1u;
inline constexpr std::uint32_t beckmann_bit = 1u << 2u;
inline constexpr std::uint32_t physical_setup_valid_bit = 1u << 29u;

enum class StorageField : std::uint32_t {
    identity,
    weight,
    albedo,
    reflection_albedo,
    transmission_albedo,
    color,
    normal,
    specular_tint,
    evaluation_scale,
    fresnel_f0,
    fresnel_f90,
    reflection_tint,
    transmission_tint,
};

[[nodiscard]] bool stores(
    SurfaceClosureStorageProfile profile,
    StorageField field) noexcept {
    if (profile == SurfaceClosureStorageProfile::complete) {
        return true;
    }
    if (profile == SurfaceClosureStorageProfile::physical) {
        return false;
    }
    switch (field) {
        case StorageField::identity:
        case StorageField::reflection_albedo:
            return true;
        case StorageField::weight:
        case StorageField::albedo:
        case StorageField::normal:
            return profile !=
                   SurfaceClosureStorageProfile::runtime_flags;
        case StorageField::transmission_albedo:
            return profile == SurfaceClosureStorageProfile::aov;
        case StorageField::color:
        case StorageField::specular_tint:
        case StorageField::evaluation_scale:
        case StorageField::fresnel_f0:
        case StorageField::fresnel_f90:
        case StorageField::reflection_tint:
        case StorageField::transmission_tint:
            return false;
    }
    return false;
}

[[nodiscard]] std::size_t storage_size(
    SurfaceClosureStorageProfile profile,
    StorageField field,
    std::size_t capacity) noexcept {
    return profile != SurfaceClosureStorageProfile::complete &&
                   stores(profile, field)
               ? capacity
               : std::size_t{1u};
}

[[nodiscard]] std::size_t complete_storage_size(
    SurfaceClosureStorageProfile profile,
    std::size_t capacity) noexcept {
    return profile == SurfaceClosureStorageProfile::complete
               ? capacity
               : std::size_t{1u};
}

[[nodiscard]] std::size_t physical_storage_size(
    SurfaceClosureStorageProfile profile,
    std::size_t capacity) noexcept {
    return profile == SurfaceClosureStorageProfile::physical
               ? capacity
               : std::size_t{1u};
}

}// namespace

SurfaceClosureSet::SurfaceClosureSet(
    std::size_t capacity,
    SurfaceClosureStorageProfile profile) noexcept
    : _capacity{std::clamp(
          capacity,
          std::size_t{1u},
          static_cast<std::size_t>(
              maximum_surface_closure_capacity))},
      _profile{profile},
      _complete_0{complete_storage_size(profile, _capacity)},
      _complete_1{complete_storage_size(profile, _capacity)},
      _complete_2{complete_storage_size(profile, _capacity)},
      _complete_3{complete_storage_size(profile, _capacity)},
      _physical_0{physical_storage_size(profile, _capacity)},
      _physical_1{physical_storage_size(profile, _capacity)},
      _identity{storage_size(
          profile, StorageField::identity, _capacity)},
      _weight{storage_size(
          profile, StorageField::weight, _capacity)},
      _albedo{storage_size(
          profile, StorageField::albedo, _capacity)},
      _reflection_albedo{storage_size(
          profile, StorageField::reflection_albedo, _capacity)},
      _transmission_albedo{storage_size(
          profile, StorageField::transmission_albedo, _capacity)},
      _color{storage_size(
          profile, StorageField::color, _capacity)},
      _normal{storage_size(
          profile, StorageField::normal, _capacity)},
      _specular_tint{storage_size(
          profile, StorageField::specular_tint, _capacity)},
      _evaluation_scale{storage_size(
          profile, StorageField::evaluation_scale, _capacity)},
      _fresnel_f0{storage_size(
          profile, StorageField::fresnel_f0, _capacity)},
      _fresnel_f90{storage_size(
          profile, StorageField::fresnel_f90, _capacity)},
      _reflection_tint{storage_size(
          profile, StorageField::reflection_tint, _capacity)},
      _transmission_tint{storage_size(
          profile, StorageField::transmission_tint, _capacity)},
      _count{0u} {
    const auto zero = SurfaceClosureRecord::zero();
    if (_profile == SurfaceClosureStorageProfile::complete) {
        const auto packed = pack_surface_closure(zero);
        _complete_0.write(0u, packed.block_0);
        _complete_1.write(0u, packed.block_1);
        _complete_2.write(0u, packed.block_2);
        _complete_3.write(0u, packed.block_3);
        return;
    }
    if (_profile == SurfaceClosureStorageProfile::physical) {
        const auto packed = pack_surface_closure_physical(zero);
        // Keep the empty-set fallback at slot zero. append_impl writes every
        // field of slot count before publishing count + 1, and every consumer
        // indexes only [0, count) (or selects slot zero for an invalid index).
        // Thus [0, max(count, 1)) is the initialized prefix; clearing the
        // unused suffix would add stores without changing any observable
        // value. physical_access() maps every invalid request to the always
        // initialized slot zero and separately retains its validity bit.
        _physical_0.write(0u, packed.block_0);
        _physical_1.write(0u, packed.block_1);
        return;
    }
    _identity.write(0u,
        make_uint4(
            zero.kind,
            zero.lobe,
            0u,
            0u));
    if (stores(_profile, StorageField::weight)) {
        _weight.write(0u, make_float4(
                               zero.weight,
                               zero.allocation_weight));
    }
    if (stores(_profile, StorageField::albedo)) {
        _albedo.write(0u, make_float4(
                               zero.albedo,
                               zero.sample_weight));
    }
    if (stores(_profile, StorageField::reflection_albedo)) {
        _reflection_albedo.write(0u, make_float4(
                                          zero.reflection_albedo,
                                          zero.roughness));
    }
    if (stores(_profile, StorageField::transmission_albedo)) {
        _transmission_albedo.write(0u, make_float4(
                                            zero.transmission_albedo,
                                            zero.diffuse_roughness));
    }
    if (stores(_profile, StorageField::color)) {
        _color.write(0u, make_float4(
                              zero.color,
                              zero.metallic));
    }
    if (stores(_profile, StorageField::normal)) {
        _normal.write(0u, make_float4(
                               zero.normal,
                               zero.ior));
    }
    if (stores(_profile, StorageField::specular_tint)) {
        _specular_tint.write(0u, make_float4(
                                      zero.specular_tint,
                                      zero.specular_ior_level));
    }
    if (stores(_profile, StorageField::evaluation_scale)) {
        _evaluation_scale.write(0u, make_float4(
                                         zero.evaluation_scale,
                                         zero.sheen_transform_a));
    }
    if (stores(_profile, StorageField::fresnel_f0)) {
        _fresnel_f0.write(0u, make_float4(
                                   zero.fresnel_f0,
                                   zero.sheen_transform_b));
    }
    if (stores(_profile, StorageField::fresnel_f90)) {
        _fresnel_f90.write(0u, make_float4(
                                    zero.fresnel_f90,
                                    0.0f));
    }
    if (stores(_profile, StorageField::reflection_tint)) {
        _reflection_tint.write(0u, make_float4(
                                        zero.reflection_tint,
                                        0.0f));
    }
    if (stores(_profile, StorageField::transmission_tint)) {
        _transmission_tint.write(0u, make_float4(
                                          zero.transmission_tint,
                                          0.0f));
    }
}

void SurfaceClosureSet::add(
    const SurfaceClosureRecord &closure) noexcept {
    append_impl(closure, nullptr);
}

void SurfaceClosureSet::append_impl(
    const SurfaceClosureRecord &closure,
    const std::function<void()> *on_retained) noexcept {
    const auto scattering =
        closure.kind != static_cast<std::uint32_t>(
                            SurfaceClosureKind::none);
    const auto allocated =
        scattering &
        (closure.allocation_weight >=
            cycles_closure::closure_weight_cutoff);
    const auto retained =
        allocated &
        (_count < static_cast<std::uint32_t>(_capacity));
    $if(retained) {
        if (_profile == SurfaceClosureStorageProfile::complete) {
            const auto packed = pack_surface_closure(closure);
            _complete_0.write(_count, packed.block_0);
            _complete_1.write(_count, packed.block_1);
            _complete_2.write(_count, packed.block_2);
            _complete_3.write(_count, packed.block_3);
        } else if (
            _profile == SurfaceClosureStorageProfile::physical) {
            const auto packed = pack_surface_closure_physical(
                static_cast<SurfaceClosurePhysicalRecord>(closure));
            _physical_0.write(_count, packed.block_0);
            _physical_1.write(_count, packed.block_1);
        } else {
            UInt flags = 0u;
            flags |= select(
                0u, setup_valid_bit, closure.setup_valid);
            flags |= select(0u,
                preserve_ggx_energy_bit,
                closure.preserve_ggx_energy);
            flags |= select(
                0u, beckmann_bit, closure.beckmann);
            _identity.write(_count,
                make_uint4(
                    closure.kind,
                    closure.lobe,
                    flags,
                    0u));
            if (stores(_profile, StorageField::weight)) {
                _weight.write(_count,
                    make_float4(
                        closure.weight,
                        closure.allocation_weight));
            }
            if (stores(_profile, StorageField::albedo)) {
                _albedo.write(_count,
                    make_float4(
                        closure.albedo,
                        closure.sample_weight));
            }
            if (stores(
                    _profile,
                    StorageField::reflection_albedo)) {
                _reflection_albedo.write(_count,
                    make_float4(
                        closure.reflection_albedo,
                        closure.roughness));
            }
            if (stores(
                    _profile,
                    StorageField::transmission_albedo)) {
                _transmission_albedo.write(_count,
                    make_float4(
                        closure.transmission_albedo,
                        closure.diffuse_roughness));
            }
            if (stores(_profile, StorageField::normal)) {
                _normal.write(_count,
                    make_float4(
                        closure.normal,
                        closure.ior));
            }
        }
        if (on_retained != nullptr) {
            (*on_retained)();
        }
        _count += 1u;
    };
}

void SurfaceClosureSet::append(
    const SurfaceClosureRecord &closure,
    const std::function<void()> &on_retained) noexcept {
    append_impl(closure, &on_retained);
}

void SurfaceClosureSet::finalize_physical_transparent(
    UInt index,
    Expr<luisa::float3> weight,
    Expr<float> sample_weight,
    Expr<luisa::float3> normal) noexcept {
    LUISA_ASSERT(
        _profile == SurfaceClosureStorageProfile::physical,
        "Transparent closure finalization requires the physical profile.");
    // Transparent is a common-only member of the physical tagged union. Its
    // payload block was initialized by the first append and is unobservable;
    // overwrite only the common block once the additive reduction is final.
    luisa::compute::UInt4 identity = luisa::make_uint4(
        cycles_closure::type_transparent,
        static_cast<std::uint32_t>(
            SurfaceClosureKind::transparent),
        static_cast<std::uint32_t>(
            SurfaceClosureLobe::none),
        physical_setup_valid_bit);
    _physical_0.write(
        index,
        make_float4x4(
            identity.bitcast<luisa::float4>(),
            make_float4(weight, sample_weight),
            make_float4(make_float3(1.0f), sample_weight),
            make_float4(normal, 0.0f)));
}

std::size_t SurfaceClosureSet::capacity() const noexcept {
    return _capacity;
}

SurfaceClosureStorageProfile SurfaceClosureSet::profile() const noexcept {
    return _profile;
}

UInt SurfaceClosureSet::count() const noexcept {
    return _count;
}

SurfaceClosurePhysicalAccess
SurfaceClosureSet::physical_access(
    UInt requested_index) const noexcept {
    LUISA_ASSERT(
        _profile == SurfaceClosureStorageProfile::physical,
        "Staged physical closure access requires the physical profile.");
    const auto valid = requested_index < _count;
    return SurfaceClosurePhysicalAccess{
        this, select(0u, requested_index, valid), valid};
}

SurfaceClosurePhysicalCommonRecord
SurfaceClosureSet::physical_common_entry_unchecked(
    UInt index) const noexcept {
    LUISA_ASSERT(
        _profile == SurfaceClosureStorageProfile::physical,
        "Staged physical closure access requires the physical profile.");
    return unpack_surface_closure_physical_common(
        Expr<luisa::float4x4>{
            _physical_0.read(index).expression()});
}

SurfaceClosurePhysicalCommonRecord
SurfaceClosureSet::physical_common_entry(
    const SurfaceClosurePhysicalAccess &access) const noexcept {
    LUISA_ASSERT(
        access._owner == this,
        "Physical closure access belongs to a different set.");
    return physical_common_entry_unchecked(access._index);
}

SurfaceClosurePhysicalRecord
SurfaceClosureSet::physical_payload_entry_unchecked(
    UInt index,
    const SurfaceClosurePhysicalCommonRecord &common) const noexcept {
    LUISA_ASSERT(
        _profile == SurfaceClosureStorageProfile::physical,
        "Staged physical closure access requires the physical profile.");
    return unpack_surface_closure_physical_payload(
        common,
        Expr<luisa::float4x4>{
            _physical_1.read(index).expression()});
}

SurfaceClosurePhysicalRecord
SurfaceClosureSet::physical_payload_entry(
    const SurfaceClosurePhysicalAccess &access,
    const SurfaceClosurePhysicalCommonRecord &common) const noexcept {
    LUISA_ASSERT(
        access._owner == this,
        "Physical closure access belongs to a different set.");
    return physical_payload_entry_unchecked(access._index, common);
}

luisa::compute::Float4x4 SurfaceClosureSet::physical_payload_block(
    const SurfaceClosurePhysicalAccess &access) const noexcept {
    LUISA_ASSERT(access._owner == this,
                 "Physical closure access belongs to a different set.");
    LUISA_ASSERT(
        _profile == SurfaceClosureStorageProfile::physical,
        "Encoded physical closure access requires the physical profile.");
    return _physical_1.read(access._index);
}

SurfaceClosureRecord SurfaceClosureSet::entry(
    UInt index) const noexcept {
    const auto valid = index < _count;
    const auto safe_index = select(0u, index, valid);
    const auto zero = SurfaceClosureRecord::zero();
    auto identity = make_uint4(
        zero.kind,
        zero.lobe,
        0u,
        0u);
    auto weight = make_float4(
        zero.weight, zero.allocation_weight);
    auto albedo = make_float4(
        zero.albedo, zero.sample_weight);
    auto reflection_albedo = make_float4(
        zero.reflection_albedo, zero.roughness);
    auto transmission_albedo = make_float4(
        zero.transmission_albedo, zero.diffuse_roughness);
    auto color = make_float4(zero.color, zero.metallic);
    auto normal = make_float4(zero.normal, zero.ior);
    auto microfacet_tangent = zero.microfacet_tangent;
    auto microfacet_alpha_x = zero.microfacet_alpha_x;
    auto microfacet_alpha_y = zero.microfacet_alpha_y;
    auto specular_tint = make_float4(
        zero.specular_tint, zero.specular_ior_level);
    auto evaluation_scale = make_float4(
        zero.evaluation_scale, zero.sheen_transform_a);
    auto fresnel_f0 = make_float4(
        zero.fresnel_f0, zero.sheen_transform_b);
    auto fresnel_f90 = make_float4(zero.fresnel_f90, 0.0f);
    auto reflection_tint = make_float4(
        zero.reflection_tint, 0.0f);
    auto transmission_tint = make_float4(
        zero.transmission_tint, zero.bssrdf_ior);
    auto bssrdf_radius = make_float4(
        zero.bssrdf_radius, zero.bssrdf_anisotropy);
    auto bssrdf_albedo = make_float4(
        zero.bssrdf_albedo, zero.bssrdf_roughness);
    Float thin_film_thickness = 0.0f;
    Float thin_film_ior = 0.0f;
    if (_profile == SurfaceClosureStorageProfile::complete) {
        // Decode through the ABI's sole semantic row projection. Going
        // through SurfaceClosureRecord here would immediately repack every
        // field into rows again and needlessly inflate the pre-optimization
        // shader AST.
        const auto complete = unpack_surface_closure_block_rows(
            Expr<luisa::float4x4>{
                _complete_0.read(safe_index).expression()},
            Expr<luisa::float4x4>{
                _complete_1.read(safe_index).expression()},
            Expr<luisa::float4x4>{
                _complete_2.read(safe_index).expression()},
            Expr<luisa::float4x4>{
                _complete_3.read(safe_index).expression()});
        identity = complete.identity;
        weight = complete.weight_allocation_weight;
        albedo = complete.albedo_sample_weight;
        reflection_albedo = complete.reflection_albedo_roughness;
        transmission_albedo =
            complete.transmission_albedo_diffuse_roughness;
        color = complete.color_metallic;
        normal = complete.normal_ior;
        microfacet_tangent =
            complete.microfacet_tangent_reserved.xyz();
        microfacet_alpha_x =
            complete.fresnel_f90_microfacet_alpha_x.w;
        microfacet_alpha_y =
            complete.reflection_tint_microfacet_alpha_y.w;
        specular_tint = complete.specular_tint_ior_level;
        evaluation_scale =
            complete.evaluation_scale_sheen_transform_a;
        fresnel_f0 = complete.fresnel_f0_sheen_transform_b;
        fresnel_f90 = complete.fresnel_f90_microfacet_alpha_x;
        reflection_tint =
            complete.reflection_tint_microfacet_alpha_y;
        transmission_tint = complete.transmission_tint_bssrdf_ior;
        bssrdf_radius = complete.bssrdf_radius_anisotropy;
        bssrdf_albedo = complete.bssrdf_albedo_roughness;
        const auto general_film_payload =
            (complete.identity.x == static_cast<std::uint32_t>(
                                        SurfaceClosureKind::principled)) &
            ((complete.identity.y == static_cast<std::uint32_t>(
                                         SurfaceClosureLobe::metallic)) |
             (complete.identity.y == static_cast<std::uint32_t>(
                                         SurfaceClosureLobe::dielectric))) |
            (complete.identity.x == static_cast<std::uint32_t>(
                                        SurfaceClosureKind::metallic_f82)) |
            (complete.identity.x == static_cast<std::uint32_t>(
                                        SurfaceClosureKind::metallic_conductor));
        const auto film_payload =
            general_film_payload |
            (complete.identity.x == static_cast<std::uint32_t>(
                                        SurfaceClosureKind::glass));
        thin_film_thickness = select(
            0.0f, transmission_albedo.w, film_payload);
        thin_film_ior = select(0.0f, color.w, film_payload);
        transmission_albedo.w = select(
            transmission_albedo.w, 0.0f, film_payload);
        color.w = select(color.w, 0.0f, film_payload);
    } else if (
        _profile == SurfaceClosureStorageProfile::physical) {
        const auto physical = unpack_surface_closure_physical(
            Expr<luisa::float4x4>{
                _physical_0.read(safe_index).expression()},
            Expr<luisa::float4x4>{
                _physical_1.read(safe_index).expression()});
        UInt physical_flags = 0u;
        physical_flags |= select(
            0u, setup_valid_bit, physical.setup_valid);
        physical_flags |= select(
            0u,
            preserve_ggx_energy_bit,
            physical.preserve_ggx_energy);
        physical_flags |= select(
            0u, beckmann_bit, physical.beckmann);
        identity = make_uint4(
            physical.kind,
            physical.lobe,
            physical_flags,
            physical.bssrdf_method);
        weight = make_float4(
            physical.weight, physical.allocation_weight);
        albedo = make_float4(zero.albedo, physical.sample_weight);
        reflection_albedo = make_float4(
            zero.reflection_albedo, physical.roughness);
        transmission_albedo = make_float4(
            zero.transmission_albedo, physical.diffuse_roughness);
        color = make_float4(physical.color, physical.metallic);
        normal = make_float4(physical.normal, physical.ior);
        microfacet_tangent = physical.microfacet_tangent;
        microfacet_alpha_x = physical.microfacet_alpha_x;
        microfacet_alpha_y = physical.microfacet_alpha_y;
        specular_tint = make_float4(
            physical.specular_tint, zero.specular_ior_level);
        evaluation_scale = make_float4(
            physical.evaluation_scale,
            physical.sheen_transform_a);
        fresnel_f0 = make_float4(
            physical.fresnel_f0,
            physical.sheen_transform_b);
        fresnel_f90 = make_float4(physical.fresnel_f90, 0.0f);
        reflection_tint = make_float4(
            physical.reflection_tint, 0.0f);
        transmission_tint = make_float4(
            physical.transmission_tint,
            physical.bssrdf_ior);
        bssrdf_radius = make_float4(
            physical.bssrdf_radius,
            physical.bssrdf_anisotropy);
        bssrdf_albedo = make_float4(
            physical.bssrdf_albedo,
            physical.bssrdf_roughness);
        thin_film_thickness = physical.thin_film_thickness;
        thin_film_ior = physical.thin_film_ior;
    } else {
        identity = _identity.read(safe_index);
        if (stores(_profile, StorageField::weight)) {
            weight = _weight.read(safe_index);
        }
        if (stores(_profile, StorageField::albedo)) {
            albedo = _albedo.read(safe_index);
        }
        if (stores(
                _profile,
                StorageField::reflection_albedo)) {
            reflection_albedo =
                _reflection_albedo.read(safe_index);
        }
        if (stores(
                _profile,
                StorageField::transmission_albedo)) {
            transmission_albedo =
                _transmission_albedo.read(safe_index);
        }
        if (stores(_profile, StorageField::normal)) {
            normal = _normal.read(safe_index);
        }
    }
    const auto flags = identity.z;
    const auto vector_or_zero =
        [&](Float3 value) noexcept {
            return select(
                make_float3(0.0f), value, valid);
        };
    const auto scalar_or_zero =
        [&](Float value) noexcept {
            return select(0.0f, value, valid);
        };
    // add() retains only allocated closures. Profiles which project the
    // weight vector away must still preserve that allocation identity so
    // canonical consumers may apply closure_allocated() without knowing the
    // storage layout. The exact allocation weight is not part of those
    // profiles; the cutoff is the unique minimal representative of the same
    // predicate.
    auto allocation_weight =
        scalar_or_zero(weight.w);
    if (_profile != SurfaceClosureStorageProfile::physical &&
        !stores(_profile, StorageField::weight)) {
        allocation_weight = select(
            0.0f,
            cycles_closure::closure_weight_cutoff,
            valid);
    }
    return {
        .kind = select(
            static_cast<std::uint32_t>(SurfaceClosureKind::none),
            identity.x,
            valid),
        .lobe = select(
            static_cast<std::uint32_t>(SurfaceClosureLobe::none),
            identity.y,
            valid),
        .weight = vector_or_zero(weight.xyz()),
        .allocation_weight = allocation_weight,
        .sample_weight = scalar_or_zero(albedo.w),
        .setup_valid =
            valid & ((flags & setup_valid_bit) != 0u),
        .albedo = vector_or_zero(albedo.xyz()),
        .reflection_albedo =
            vector_or_zero(reflection_albedo.xyz()),
        .transmission_albedo =
            vector_or_zero(transmission_albedo.xyz()),
        .color = vector_or_zero(color.xyz()),
        .normal = select(
            make_float3(0.0f, 0.0f, 1.0f),
            normal.xyz(),
            valid),
        .roughness = scalar_or_zero(reflection_albedo.w),
        .microfacet_tangent =
            vector_or_zero(microfacet_tangent),
        .microfacet_alpha_x =
            scalar_or_zero(microfacet_alpha_x),
        .microfacet_alpha_y =
            scalar_or_zero(microfacet_alpha_y),
        .diffuse_roughness =
            scalar_or_zero(transmission_albedo.w),
        .metallic = scalar_or_zero(color.w),
        .ior = select(1.0f, normal.w, valid),
        .thin_film_thickness =
            scalar_or_zero(thin_film_thickness),
        .thin_film_ior = scalar_or_zero(thin_film_ior),
        .specular_ior_level =
            scalar_or_zero(specular_tint.w),
        .specular_tint =
            vector_or_zero(specular_tint.xyz()),
        .sheen_transform_a =
            scalar_or_zero(evaluation_scale.w),
        .sheen_transform_b =
            scalar_or_zero(fresnel_f0.w),
        .evaluation_scale = select(
            make_float3(1.0f),
            evaluation_scale.xyz(),
            valid),
        .fresnel_f0 = vector_or_zero(fresnel_f0.xyz()),
        .fresnel_f90 = vector_or_zero(fresnel_f90.xyz()),
        .reflection_tint =
            vector_or_zero(reflection_tint.xyz()),
        .transmission_tint =
            vector_or_zero(transmission_tint.xyz()),
        .preserve_ggx_energy =
            valid &
            ((flags & preserve_ggx_energy_bit) != 0u),
        .beckmann =
            valid & ((flags & beckmann_bit) != 0u),
        .bssrdf_method = select(
            static_cast<std::uint32_t>(
                SurfaceBssrdfMethod::random_walk),
            identity.w,
            valid),
        .bssrdf_radius = vector_or_zero(bssrdf_radius.xyz()),
        .bssrdf_albedo = vector_or_zero(bssrdf_albedo.xyz()),
        .bssrdf_ior = select(
            zero.bssrdf_ior, transmission_tint.w, valid),
        .bssrdf_roughness = select(
            zero.bssrdf_roughness, bssrdf_albedo.w, valid),
        .bssrdf_anisotropy = scalar_or_zero(bssrdf_radius.w)};
}

}// namespace psycles::luisa_backend
