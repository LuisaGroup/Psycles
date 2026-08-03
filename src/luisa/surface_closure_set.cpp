#include <psycles/luisa/surface_closure_set.h>

#include <algorithm>

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

inline constexpr std::uint32_t setup_valid_bit = 1u << 0u;
inline constexpr std::uint32_t preserve_ggx_energy_bit = 1u << 1u;
inline constexpr std::uint32_t beckmann_bit = 1u << 2u;

}// namespace

SurfaceClosureSet::SurfaceClosureSet(
    std::size_t capacity) noexcept
    : _capacity{std::clamp(
          capacity,
          std::size_t{1u},
          static_cast<std::size_t>(
              maximum_surface_closure_capacity))},
      _identity{_capacity},
      _weight{_capacity},
      _albedo{_capacity},
      _reflection_albedo{_capacity},
      _transmission_albedo{_capacity},
      _color{_capacity},
      _normal{_capacity},
      _specular_tint{_capacity},
      _evaluation_scale{_capacity},
      _fresnel_f0{_capacity},
      _fresnel_f90{_capacity},
      _reflection_tint{_capacity},
      _transmission_tint{_capacity},
      _count{0u} {
    const auto zero = SurfaceClosureRecord::zero();
    _identity.write(0u, make_uint4(
                            zero.kind,
                            zero.lobe,
                            0u,
                            0u));
    _weight.write(0u, make_float4(
                           zero.weight,
                           zero.allocation_weight));
    _albedo.write(0u, make_float4(
                           zero.albedo,
                           zero.sample_weight));
    _reflection_albedo.write(0u, make_float4(
                                      zero.reflection_albedo,
                                      zero.roughness));
    _transmission_albedo.write(0u, make_float4(
                                        zero.transmission_albedo,
                                        zero.diffuse_roughness));
    _color.write(0u, make_float4(
                          zero.color,
                          zero.metallic));
    _normal.write(0u, make_float4(
                           zero.normal,
                           zero.ior));
    _specular_tint.write(0u, make_float4(
                                  zero.specular_tint,
                                  zero.specular_ior_level));
    _evaluation_scale.write(0u, make_float4(
                                     zero.evaluation_scale,
                                     zero.sheen_transform_a));
    _fresnel_f0.write(0u, make_float4(
                               zero.fresnel_f0,
                               zero.sheen_transform_b));
    _fresnel_f90.write(0u, make_float4(
                                zero.fresnel_f90,
                                0.0f));
    _reflection_tint.write(0u, make_float4(
                                    zero.reflection_tint,
                                    0.0f));
    _transmission_tint.write(0u, make_float4(
                                      zero.transmission_tint,
                                      0.0f));
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
        _weight.write(_count,
            make_float4(
                closure.weight,
                closure.allocation_weight));
        _albedo.write(_count,
            make_float4(
                closure.albedo,
                closure.sample_weight));
        _reflection_albedo.write(_count,
            make_float4(
                closure.reflection_albedo,
                closure.roughness));
        _transmission_albedo.write(_count,
            make_float4(
                closure.transmission_albedo,
                closure.diffuse_roughness));
        _color.write(_count,
            make_float4(
                closure.color,
                closure.metallic));
        _normal.write(_count,
            make_float4(
                closure.normal,
                closure.ior));
        _specular_tint.write(_count,
            make_float4(
                closure.specular_tint,
                closure.specular_ior_level));
        _evaluation_scale.write(_count,
            make_float4(
                closure.evaluation_scale,
                closure.sheen_transform_a));
        _fresnel_f0.write(_count,
            make_float4(
                closure.fresnel_f0,
                closure.sheen_transform_b));
        _fresnel_f90.write(_count,
            make_float4(
                closure.fresnel_f90,
                0.0f));
        _reflection_tint.write(_count,
            make_float4(
                closure.reflection_tint,
                0.0f));
        _transmission_tint.write(_count,
            make_float4(
                closure.transmission_tint,
                0.0f));
        _count += 1u;
    };
}

std::size_t SurfaceClosureSet::capacity() const noexcept {
    return _capacity;
}

UInt SurfaceClosureSet::count() const noexcept {
    return _count;
}

SurfaceClosureRecord SurfaceClosureSet::entry(
    UInt index) const noexcept {
    const auto valid = index < _count;
    const auto safe_index = select(0u, index, valid);
    const auto identity = _identity.read(safe_index);
    const auto weight = _weight.read(safe_index);
    const auto albedo = _albedo.read(safe_index);
    const auto reflection_albedo =
        _reflection_albedo.read(safe_index);
    const auto transmission_albedo =
        _transmission_albedo.read(safe_index);
    const auto color = _color.read(safe_index);
    const auto normal = _normal.read(safe_index);
    const auto specular_tint =
        _specular_tint.read(safe_index);
    const auto evaluation_scale =
        _evaluation_scale.read(safe_index);
    const auto fresnel_f0 =
        _fresnel_f0.read(safe_index);
    const auto fresnel_f90 =
        _fresnel_f90.read(safe_index);
    const auto reflection_tint =
        _reflection_tint.read(safe_index);
    const auto transmission_tint =
        _transmission_tint.read(safe_index);
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
        .allocation_weight = scalar_or_zero(weight.w),
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
