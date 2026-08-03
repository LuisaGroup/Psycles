#include <psycles/luisa/surface_closure_set.h>

#include <algorithm>

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

inline constexpr std::uint32_t setup_valid_bit = 1u << 0u;
inline constexpr std::uint32_t preserve_ggx_energy_bit = 1u << 1u;
inline constexpr std::uint32_t beckmann_bit = 1u << 2u;

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
    return stores(profile, field) ? capacity : std::size_t{1u};
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
      _identity{_capacity},
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
    _identity.write(0u, make_uint4(
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
        if (stores(_profile, StorageField::reflection_albedo)) {
            _reflection_albedo.write(_count,
                make_float4(
                    closure.reflection_albedo,
                    closure.roughness));
        }
        if (stores(_profile, StorageField::transmission_albedo)) {
            _transmission_albedo.write(_count,
                make_float4(
                    closure.transmission_albedo,
                    closure.diffuse_roughness));
        }
        if (stores(_profile, StorageField::color)) {
            _color.write(_count,
                make_float4(
                    closure.color,
                    closure.metallic));
        }
        if (stores(_profile, StorageField::normal)) {
            _normal.write(_count,
                make_float4(
                    closure.normal,
                    closure.ior));
        }
        if (stores(_profile, StorageField::specular_tint)) {
            _specular_tint.write(_count,
                make_float4(
                    closure.specular_tint,
                    closure.specular_ior_level));
        }
        if (stores(_profile, StorageField::evaluation_scale)) {
            _evaluation_scale.write(_count,
                make_float4(
                    closure.evaluation_scale,
                    closure.sheen_transform_a));
        }
        if (stores(_profile, StorageField::fresnel_f0)) {
            _fresnel_f0.write(_count,
                make_float4(
                    closure.fresnel_f0,
                    closure.sheen_transform_b));
        }
        if (stores(_profile, StorageField::fresnel_f90)) {
            _fresnel_f90.write(_count,
                make_float4(
                    closure.fresnel_f90,
                    0.0f));
        }
        if (stores(_profile, StorageField::reflection_tint)) {
            _reflection_tint.write(_count,
                make_float4(
                    closure.reflection_tint,
                    0.0f));
        }
        if (stores(_profile, StorageField::transmission_tint)) {
            _transmission_tint.write(_count,
                make_float4(
                    closure.transmission_tint,
                    0.0f));
        }
        _count += 1u;
    };
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

SurfaceClosureRecord SurfaceClosureSet::entry(
    UInt index) const noexcept {
    const auto valid = index < _count;
    const auto safe_index = select(0u, index, valid);
    const auto identity = _identity.read(safe_index);
    const auto zero = SurfaceClosureRecord::zero();
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
        zero.transmission_tint, 0.0f);
    if (stores(_profile, StorageField::weight)) {
        weight = _weight.read(safe_index);
    }
    if (stores(_profile, StorageField::albedo)) {
        albedo = _albedo.read(safe_index);
    }
    if (stores(_profile, StorageField::reflection_albedo)) {
        reflection_albedo = _reflection_albedo.read(safe_index);
    }
    if (stores(_profile, StorageField::transmission_albedo)) {
        transmission_albedo = _transmission_albedo.read(safe_index);
    }
    if (stores(_profile, StorageField::color)) {
        color = _color.read(safe_index);
    }
    if (stores(_profile, StorageField::normal)) {
        normal = _normal.read(safe_index);
    }
    if (stores(_profile, StorageField::specular_tint)) {
        specular_tint = _specular_tint.read(safe_index);
    }
    if (stores(_profile, StorageField::evaluation_scale)) {
        evaluation_scale = _evaluation_scale.read(safe_index);
    }
    if (stores(_profile, StorageField::fresnel_f0)) {
        fresnel_f0 = _fresnel_f0.read(safe_index);
    }
    if (stores(_profile, StorageField::fresnel_f90)) {
        fresnel_f90 = _fresnel_f90.read(safe_index);
    }
    if (stores(_profile, StorageField::reflection_tint)) {
        reflection_tint = _reflection_tint.read(safe_index);
    }
    if (stores(_profile, StorageField::transmission_tint)) {
        transmission_tint = _transmission_tint.read(safe_index);
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
    if (!stores(_profile, StorageField::weight)) {
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
        .diffuse_roughness =
            scalar_or_zero(transmission_albedo.w),
        .metallic = scalar_or_zero(color.w),
        .ior = select(1.0f, normal.w, valid),
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
            valid & ((flags & beckmann_bit) != 0u)};
}

}// namespace psycles::luisa_backend
