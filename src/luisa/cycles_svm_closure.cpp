/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_bssrdf.h"
#include "cycles_svm_closure_layout.h"
#include "cycles_svm_hair.h"
#include "cycles_svm_internal.h"
#include "cycles_svm_microfacet.h"
#include "cycles_svm_principled.h"
#include "cycles_svm_principled_hair.h"
#include "cycles_svm_ray_portal.h"
#include "cycles_svm_sheen.h"
#include "cycles_svm_simple_closure.h"
#include "cycles_svm_toon.h"

#include <psycles/luisa/native_vector_math.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace layout = detail::closure_layout;

static constexpr auto closure_vec4s = layout::ShaderClosure_sizeof / 16u;
static_assert(layout::ShaderClosure_sizeof % 16u == 0u);

ClosurePool::ClosurePool(std::size_t capacity) noexcept
    : ClosurePool{capacity, Options{}} {}

ClosurePool::ClosurePool(std::size_t capacity, Options options) noexcept
    : _capacity{std::min(capacity, maximum_closure_capacity)},
      _options{options},
      _slot_vec4s{_options.compact_64b_slots
                      ? 4u
                      : static_cast<std::uint32_t>(closure_vec4s)},
      _storage{std::max(_capacity, std::size_t{1u}) * _slot_vec4s},
      _count{0u},
      _left{static_cast<std::uint32_t>(_capacity)} {
  if (_options.spectral) {
    const auto spectral_capacity = std::max(_capacity, std::size_t{1u});
    _spectral_weight.emplace(spectral_capacity);
    _spectral_weight_valid.emplace(spectral_capacity);
  }
}

Expr<std::uint32_t> ClosurePool::slot_storage_index(
    Expr<std::uint32_t> slot, std::uint32_t vec_index) const noexcept {
  LUISA_ASSERT(
      vec_index < _slot_vec4s,
      "ClosurePool vec4 index {} exceeds slot stride {} (byte offset {}).",
      vec_index, _slot_vec4s, vec_index * 16u);
  if (_capacity <= 1u) {
    return Expr<std::uint32_t>{vec_index};
  }
  if (const auto c = detail::try_const_uint(slot)) {
    return Expr<std::uint32_t>{*c * _slot_vec4s + vec_index};
  }
  if (_slot_vec4s == 4u) {
    return (slot << 2u) + vec_index;
  }
  return slot * _slot_vec4s + vec_index;
}

UInt4 ClosurePool::load_vec4(Expr<std::uint32_t> slot,
                             std::uint32_t vec_index) const noexcept {
  if (_capacity > 1u) {
    if (const auto c = detail::try_const_uint(slot)) {
      if (*c >= _capacity) {
        return luisa::compute::make_uint4(0u);
      }
    }
  }
  return _storage[slot_storage_index(slot, vec_index)];
}

void ClosurePool::store_vec4(Expr<std::uint32_t> slot,
                             std::uint32_t vec_index,
                             Expr<luisa::uint4> value) noexcept {
  if (_capacity > 1u) {
    if (const auto c = detail::try_const_uint(slot)) {
      if (*c >= _capacity) {
        return;
      }
    }
  }
  _storage[slot_storage_index(slot, vec_index)] = value;
}

UInt ClosurePool::load_word(Expr<std::uint32_t> slot,
                            std::uint32_t byte_offset) const noexcept {
  const auto vec_offset = static_cast<std::uint32_t>(byte_offset / 16u);
  const auto lane = static_cast<std::uint32_t>((byte_offset / 4u) & 3u);
  if (_capacity > 1u) {
    if (const auto c = detail::try_const_uint(slot)) {
      if (*c >= _capacity) {
        return 0u;
      }
    }
  }
  return _storage[slot_storage_index(slot, vec_offset)][lane];
}

void ClosurePool::store_word(Expr<std::uint32_t> slot,
                             std::uint32_t byte_offset,
                             Expr<std::uint32_t> value) noexcept {
  const auto vec_offset = static_cast<std::uint32_t>(byte_offset / 16u);
  const auto lane = static_cast<std::uint32_t>((byte_offset / 4u) & 3u);
  if (_capacity > 1u) {
    if (const auto c = detail::try_const_uint(slot)) {
      if (*c >= _capacity) {
        return;
      }
    }
  }
  _storage[slot_storage_index(slot, vec_offset)][lane] = value;
}

Float ClosurePool::load_float(Expr<std::uint32_t> slot,
                              std::uint32_t byte_offset) const noexcept {
  return load_word(slot, byte_offset).as<float>();
}

void ClosurePool::store_float(Expr<std::uint32_t> slot,
                              std::uint32_t byte_offset,
                              Expr<float> value) noexcept {
  store_word(slot, byte_offset, value.as<luisa::uint>());
}

Float3 ClosurePool::load_float3(Expr<std::uint32_t> slot,
                                std::uint32_t byte_offset) const noexcept {
  return make_float3(load_float(slot, byte_offset),
                     load_float(slot, byte_offset + 4u),
                     load_float(slot, byte_offset + 8u));
}

void ClosurePool::store_float3(Expr<std::uint32_t> slot,
                               std::uint32_t byte_offset,
                               Expr<luisa::float3> value) noexcept {
  store_float(slot, byte_offset, value.x);
  store_float(slot, byte_offset + 4u, value.y);
  store_float(slot, byte_offset + 8u, value.z);
}

std::size_t ClosurePool::capacity() const noexcept { return _capacity; }

UInt ClosurePool::count() const noexcept { return _count; }

UInt ClosurePool::left() const noexcept { return _left; }

void ClosurePool::reset() noexcept {
  _count = 0u;
  _left = static_cast<std::uint32_t>(_capacity);
}

ClosurePool::Allocation
ClosurePool::allocate(Expr<std::uint32_t> closure_type,
                      Expr<luisa::float3> weight) noexcept {
  Allocation allocation{.index = _count, .valid = false};
  $if(_left != 0u) {
    static_assert(layout::ShaderClosure_weight == 0u &&
                  layout::ShaderClosure_type == 12u);
    store_vec4(allocation.index, 0u,
               make_uint4(weight.x.as<luisa::uint>(),
                          weight.y.as<luisa::uint>(),
                          weight.z.as<luisa::uint>(), closure_type));
    if (_options.spectral && _spectral_weight_valid.has_value()) {
      if (_capacity <= 1u) {
        (*_spectral_weight_valid)[0u] = 0u;
      } else if (const auto c = detail::try_const_uint(allocation.index)) {
        if (*c < _capacity) {
          (*_spectral_weight_valid)[*c] = 0u;
        }
      } else {
        (*_spectral_weight_valid)[allocation.index] = 0u;
      }
    }
    // Native closure_alloc initializes only type/weight. Scattering setup
    // owns sample_weight/N; other closure kinds must not clobber those lanes.
    _count += 1u;
    _left -= 1u;
    allocation.valid = true;
  };
  return allocation;
}

void ClosurePool::set_spectral_weight(Expr<std::uint32_t> index,
                                      Expr<luisa::float4> weight) noexcept {
  if (!_options.spectral || !_spectral_weight.has_value() ||
      !_spectral_weight_valid.has_value()) {
    return;
  }
  if (_capacity <= 1u) {
    (*_spectral_weight)[0u] = weight;
    (*_spectral_weight_valid)[0u] = 1u;
    return;
  }
  if (const auto c = detail::try_const_uint(index)) {
    if (*c < _capacity) {
      (*_spectral_weight)[*c] = weight;
      (*_spectral_weight_valid)[*c] = 1u;
    }
    return;
  }
  $if(index < static_cast<std::uint32_t>(_capacity)) {
    (*_spectral_weight)[index] = weight;
    (*_spectral_weight_valid)[index] = 1u;
  };
}

Bool ClosurePool::has_spectral_weight(
    Expr<std::uint32_t> index) const noexcept {
  if (!_options.spectral || !_spectral_weight_valid.has_value()) {
    return false;
  }
  if (_capacity <= 1u) {
    return (*_spectral_weight_valid)[0u] != 0u;
  }
  if (const auto c = detail::try_const_uint(index)) {
    if (*c >= _capacity) {
      return false;
    }
    return (*_spectral_weight_valid)[*c] != 0u;
  }
  Bool result = false;
  $if(index < static_cast<std::uint32_t>(_capacity)) {
    result = (*_spectral_weight_valid)[index] != 0u;
  };
  return result;
}

Float4 ClosurePool::spectral_weight(
    Expr<std::uint32_t> index) const noexcept {
  if (!_options.spectral || !_spectral_weight.has_value()) {
    return make_float4(0.0f);
  }
  if (_capacity <= 1u) {
    return (*_spectral_weight)[0u];
  }
  if (const auto c = detail::try_const_uint(index)) {
    if (*c >= _capacity) {
      return make_float4(0.0f);
    }
    return (*_spectral_weight)[*c];
  }
  Float4 result = make_float4(0.0f);
  $if(index < static_cast<std::uint32_t>(_capacity)) {
    result = (*_spectral_weight)[index];
  };
  return result;
}

Bool ClosurePool::allocate_extra(const Allocation &owner,
                                 Expr<std::uint32_t> slot_count,
                                 ExtraPayload payload) noexcept {
  Bool allocated = false;
  $if(owner.valid) {
    $if(slot_count <= _left) {
      _left -= slot_count;
      if (payload == ExtraPayload::huang_hair) {
        store_word(owner.index, layout::HuangHairBSDF_extra, _count + _left);
      } else {
        static_assert(layout::MicrofacetBsdf_fresnel_type == 48u &&
                      layout::MicrofacetBsdf_fresnel == 56u);
        const auto fresnel_type = load_vec4(owner.index, 3u).x;
        store_vec4(owner.index, 3u,
                   make_uint4(fresnel_type, 0u, _count + _left, 0u));
      }
      allocated = true;
    }
    $else {
      /* closure_alloc_extra is called immediately after bsdf_alloc. The
       * owner identity makes that source precondition explicit and prevents
       * an unrelated closure from being removed by a malformed caller. */
      $if((_count != 0u) & (owner.index + 1u == _count)) {
        _count -= 1u;
        _left += 1u;
      };
    };
  };
  return allocated;
}

void ClosurePool::rollback_with_extra(
    const Allocation &owner, Expr<std::uint32_t> extra_slot_count) noexcept {
  $if(owner.valid & (_count != 0u) & (owner.index + 1u == _count)) {
    _count -= 1u;
    _left += 1u + extra_slot_count;
  };
}

void ClosurePool::set_type(Expr<std::uint32_t> index,
                           Expr<std::uint32_t> closure_type) noexcept {
  store_word(index, layout::ShaderClosure_type, closure_type);
}

void ClosurePool::set_weight(Expr<std::uint32_t> index,
                             Expr<luisa::float3> weight) noexcept {
  static_assert(layout::ShaderClosure_weight == 0u &&
                layout::ShaderClosure_type == 12u);
  const auto v0 = load_vec4(index, 0u);
  store_vec4(index, 0u,
             make_uint4(weight.x.as<luisa::uint>(),
                        weight.y.as<luisa::uint>(),
                        weight.z.as<luisa::uint>(), v0.w));
}

void ClosurePool::set_weight_and_type(
    Expr<std::uint32_t> index, Expr<luisa::float3> weight,
    Expr<std::uint32_t> closure_type) noexcept {
  static_assert(layout::ShaderClosure_weight == 0u &&
                layout::ShaderClosure_type == 12u);
  store_vec4(index, 0u,
             make_uint4(weight.x.as<luisa::uint>(),
                        weight.y.as<luisa::uint>(),
                        weight.z.as<luisa::uint>(), closure_type));
}

void ClosurePool::add_weight(Expr<std::uint32_t> index,
                             Expr<luisa::float3> weight) noexcept {
  static_assert(layout::ShaderClosure_weight == 0u &&
                layout::ShaderClosure_type == 12u);
  const auto v0 = load_vec4(index, 0u);
  store_vec4(index, 0u,
             make_uint4((v0.x.as<float>() + weight.x).as<luisa::uint>(),
                        (v0.y.as<float>() + weight.y).as<luisa::uint>(),
                        (v0.z.as<float>() + weight.z).as<luisa::uint>(), v0.w));
}

void ClosurePool::set_sample_weight(Expr<std::uint32_t> index,
                                    Expr<float> sample_weight) noexcept {
  store_float(index, layout::ShaderClosure_sample_weight, sample_weight);
}

void ClosurePool::add_sample_weight(Expr<std::uint32_t> index,
                                    Expr<float> sample_weight) noexcept {
  static_assert(layout::ShaderClosure_sample_weight == 16u &&
                layout::ShaderClosure_N == 20u);
  const auto v1 = load_vec4(index, 1u);
  store_vec4(
      index, 1u,
      make_uint4((v1.x.as<float>() + sample_weight).as<luisa::uint>(), v1.y,
                 v1.z, v1.w));
}

void ClosurePool::set_normal(Expr<std::uint32_t> index,
                             Expr<luisa::float3> normal) noexcept {
  static_assert(layout::ShaderClosure_sample_weight == 16u &&
                layout::ShaderClosure_N == 20u);
  const auto v1 = load_vec4(index, 1u);
  store_vec4(index, 1u,
             make_uint4(v1.x, normal.x.as<luisa::uint>(),
                        normal.y.as<luisa::uint>(),
                        normal.z.as<luisa::uint>()));
}

void ClosurePool::set_sample_weight_and_normal(
    Expr<std::uint32_t> index, Expr<float> sample_weight,
    Expr<luisa::float3> normal) noexcept {
  static_assert(layout::ShaderClosure_sample_weight == 16u &&
                layout::ShaderClosure_N == 20u);
  store_vec4(index, 1u,
             make_uint4(sample_weight.as<luisa::uint>(),
                        normal.x.as<luisa::uint>(),
                        normal.y.as<luisa::uint>(),
                        normal.z.as<luisa::uint>()));
}

void ClosurePool::set_oren_nayar_param(Expr<std::uint32_t> index,
                                       const OrenNayarParam &param) noexcept {
  constexpr auto base = layout::OrenNayarBsdf_param;
  store_float(index, base + layout::OrenNayarParam_roughness, param.roughness);
  store_float(index, base + layout::OrenNayarParam_a, param.a);
  store_float(index, base + layout::OrenNayarParam_b, param.b);
  store_float3(index, base + layout::OrenNayarParam_multiscatter_term,
               param.multiscatter_term);
}

void ClosurePool::set_sheen_param(Expr<std::uint32_t> index,
                                  const SheenParam &param) noexcept {
  store_float(index, layout::SheenBsdf_roughness, param.roughness);
  store_float(index, layout::SheenBsdf_transformA, param.transform_a);
  store_float(index, layout::SheenBsdf_transformB, param.transform_b);
  store_float3(index, layout::SheenBsdf_T, param.T);
  store_float3(index, layout::SheenBsdf_B, param.B);
}

void ClosurePool::set_velvet_param(Expr<std::uint32_t> index,
                                   const VelvetParam &param) noexcept {
  store_float(index, layout::VelvetBsdf_sigma, param.sigma);
  store_float(index, layout::VelvetBsdf_invsigma2, param.invsigma2);
}

void ClosurePool::set_toon_param(Expr<std::uint32_t> index,
                                 const ToonParam &param) noexcept {
  store_float(index, layout::ToonBsdf_size, param.size);
  store_float(index, layout::ToonBsdf_smooth, param.smooth);
}

void ClosurePool::set_ray_portal_param(Expr<std::uint32_t> index,
                                       const RayPortalParam &param) noexcept {
  store_float3(index, layout::RayPortalClosure_P, param.P);
  store_float3(index, layout::RayPortalClosure_D, param.D);
}

void ClosurePool::set_hair_param(Expr<std::uint32_t> index,
                                 const HairParam &param) noexcept {
  store_float3(index, layout::HairBsdf_T, param.T);
  store_float(index, layout::HairBsdf_roughness1, param.roughness1);
  store_float(index, layout::HairBsdf_roughness2, param.roughness2);
  store_float(index, layout::HairBsdf_offset, param.offset);
}

void ClosurePool::set_chiang_hair_param(Expr<std::uint32_t> index,
                                        const ChiangHairParam &param) noexcept {
  store_float3(index, layout::ChiangHairBSDF_sigma, param.sigma);
  store_float(index, layout::ChiangHairBSDF_v, param.v);
  store_float(index, layout::ChiangHairBSDF_s, param.s);
  store_float(index, layout::ChiangHairBSDF_alpha, param.alpha);
  store_float(index, layout::ChiangHairBSDF_eta, param.eta);
  store_float(index, layout::ChiangHairBSDF_m0_roughness, param.m0_roughness);
  store_float(index, layout::ChiangHairBSDF_h, param.h);
}

void ClosurePool::set_huang_hair(Expr<std::uint32_t> index,
                                 const HuangHairParam &param,
                                 const HuangHairExtra &extra) noexcept {
  store_float3(index, layout::HuangHairBSDF_sigma, param.sigma);
  store_float(index, layout::HuangHairBSDF_roughness, param.roughness);
  store_float(index, layout::HuangHairBSDF_tilt, param.tilt);
  store_float(index, layout::HuangHairBSDF_eta, param.eta);
  store_float(index, layout::HuangHairBSDF_aspect_ratio, param.aspect_ratio);
  store_float(index, layout::HuangHairBSDF_h, param.h);
  const auto tail = load_word(index, layout::HuangHairBSDF_extra);
  store_float(tail, layout::HuangHairExtra_R, extra.R);
  store_float(tail, layout::HuangHairExtra_TT, extra.TT);
  store_float(tail, layout::HuangHairExtra_TRT, extra.TRT);
  store_float3(tail, layout::HuangHairExtra_Y, extra.Y);
  store_float3(tail, layout::HuangHairExtra_Z, extra.Z);
  store_float3(tail, layout::HuangHairExtra_wi, extra.wi);
  store_float(tail, layout::HuangHairExtra_radius, extra.radius);
  store_float(tail, layout::HuangHairExtra_e2, extra.e2);
  store_float(tail, layout::HuangHairExtra_pixel_coverage,
              extra.pixel_coverage);
}

void ClosurePool::set_bssrdf_param(Expr<std::uint32_t> index,
                                   const BssrdfParam &param) noexcept {
  store_float3(index, layout::Bssrdf_radius, param.radius);
  store_float3(index, layout::Bssrdf_albedo, param.albedo);
  store_float(index, layout::Bssrdf_anisotropy, param.anisotropy);
  store_float(index, layout::Bssrdf_ior, param.ior);
  store_float(index, layout::Bssrdf_alpha, param.alpha);
}

void ClosurePool::set_microfacet_param(Expr<std::uint32_t> index,
                                       const MicrofacetParam &param) noexcept {
  static_assert(layout::MicrofacetBsdf_alpha_x == 32u &&
                layout::MicrofacetBsdf_alpha_y == 36u &&
                layout::MicrofacetBsdf_ior == 40u &&
                layout::MicrofacetBsdf_energy_scale == 44u &&
                layout::MicrofacetBsdf_fresnel_type == 48u &&
                layout::MicrofacetBsdf_fresnel == 56u &&
                layout::MicrofacetBsdf_T == 64u);
  store_vec4(index, 2u,
             make_uint4(param.alpha_x.as<luisa::uint>(),
                        param.alpha_y.as<luisa::uint>(),
                        param.ior.as<luisa::uint>(),
                        param.energy_scale.as<luisa::uint>()));
  if (_options.may_have_extra_fresnel) {
    const auto extra = load_vec4(index, 3u).z;
    store_vec4(index, 3u, make_uint4(param.fresnel_type, 0u, extra, 0u));
  } else {
    store_vec4(index, 3u, make_uint4(param.fresnel_type, 0u, 0u, 0u));
  }
  if (_options.may_have_anisotropic_microfacet) {
    store_vec4(index, 4u,
               make_uint4(param.T.x.as<luisa::uint>(),
                          param.T.y.as<luisa::uint>(),
                          param.T.z.as<luisa::uint>(), 0u));
  }
}

void ClosurePool::blur_microfacet_roughness(Expr<std::uint32_t> index,
                                            Expr<float> roughness) noexcept {
  static_assert(layout::MicrofacetBsdf_alpha_x == 32u &&
                layout::MicrofacetBsdf_alpha_y == 36u);
  const auto v2 = load_vec4(index, 2u);
  const auto alpha_x = max(roughness, v2.x.as<float>());
  const auto alpha_y = max(roughness, v2.y.as<float>());
  store_vec4(index, 2u,
             make_uint4(alpha_x.as<luisa::uint>(), alpha_y.as<luisa::uint>(),
                        v2.z, v2.w));
}

void ClosurePool::set_generalized_schlick(
    Expr<std::uint32_t> index,
    const FresnelGeneralizedSchlick &fresnel) noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  static_assert(layout::FresnelGeneralizedSchlick_f0 == 32u &&
                layout::FresnelGeneralizedSchlick_f90 == 44u &&
                layout::FresnelGeneralizedSchlick_exponent == 56u);
  if (!_options.may_have_thin_film &&
      !_options.may_have_custom_generalized_schlick &&
      !_options.may_have_positive_exponent) {
    store_vec4(tail, 2u,
               make_uint4(fresnel.f0.x.as<luisa::uint>(),
                          fresnel.f0.y.as<luisa::uint>(),
                          fresnel.f0.z.as<luisa::uint>(), 0u));
    return;
  }
  constexpr auto film = layout::FresnelGeneralizedSchlick_thin_film;
  if (_options.may_have_thin_film) {
    store_float(tail, film + layout::FresnelThinFilm_thickness,
                fresnel.thin_film.thickness);
    store_float(tail, film + layout::FresnelThinFilm_ior, fresnel.thin_film.ior);
  }
  if (_options.may_have_custom_generalized_schlick) {
    store_float3(tail, layout::FresnelGeneralizedSchlick_reflection_tint,
                 fresnel.reflection_tint);
    store_float3(tail, layout::FresnelGeneralizedSchlick_transmission_tint,
                 fresnel.transmission_tint);
  }
  store_vec4(tail, 2u,
             make_uint4(fresnel.f0.x.as<luisa::uint>(),
                        fresnel.f0.y.as<luisa::uint>(),
                        fresnel.f0.z.as<luisa::uint>(),
                        fresnel.f90.x.as<luisa::uint>()));
  if (_options.may_have_positive_exponent ||
      _options.may_have_custom_generalized_schlick) {
    store_vec4(tail, 3u,
               make_uint4(fresnel.f90.y.as<luisa::uint>(),
                          fresnel.f90.z.as<luisa::uint>(),
                          fresnel.exponent.as<luisa::uint>(), 0u));
  }
}

void ClosurePool::set_fresnel_conductor(
    Expr<std::uint32_t> index, const FresnelConductor &fresnel) noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelConductor_thin_film;
  if (_options.may_have_thin_film) {
    store_float(tail, film + layout::FresnelThinFilm_thickness,
                fresnel.thin_film.thickness);
    store_float(tail, film + layout::FresnelThinFilm_ior, fresnel.thin_film.ior);
  }
  store_float3(tail, layout::FresnelConductor_ior + layout::complex_Spectrum_re,
               fresnel.ior);
  store_float3(tail, layout::FresnelConductor_ior + layout::complex_Spectrum_im,
               fresnel.extinction);
}

void ClosurePool::set_fresnel_f82_tint(Expr<std::uint32_t> index,
                                       const FresnelF82Tint &fresnel) noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelF82Tint_thin_film;
  if (_options.may_have_thin_film) {
    store_float(tail, film + layout::FresnelThinFilm_thickness,
                fresnel.thin_film.thickness);
    store_float(tail, film + layout::FresnelThinFilm_ior, fresnel.thin_film.ior);
  }
  store_float3(tail, layout::FresnelF82Tint_f0, fresnel.f0);
  store_float3(tail, layout::FresnelF82Tint_b, fresnel.b);
}

void ClosurePool::set_left(Expr<std::uint32_t> left) noexcept { _left = left; }

ShaderClosureCommon
ClosurePool::common(Expr<std::uint32_t> index) const noexcept {
  static_assert(layout::ShaderClosure_weight == 0u &&
                layout::ShaderClosure_type == 12u &&
                layout::ShaderClosure_sample_weight == 16u &&
                layout::ShaderClosure_N == 20u);
  const auto v0 = load_vec4(index, 0u);
  const auto v1 = load_vec4(index, 1u);
  return {.weight = make_float3(v0.x.as<float>(), v0.y.as<float>(),
                                v0.z.as<float>()),
          .type = v0.w,
          .sample_weight = v1.x.as<float>(),
          .N = make_float3(v1.y.as<float>(), v1.z.as<float>(),
                           v1.w.as<float>())};
}

OrenNayarClosure
ClosurePool::oren_nayar(Expr<std::uint32_t> index) const noexcept {
  constexpr auto base = layout::OrenNayarBsdf_param;
  return {
      .common = common(index),
      .param = {.roughness =
                    load_float(index, base + layout::OrenNayarParam_roughness),
                .a = load_float(index, base + layout::OrenNayarParam_a),
                .b = load_float(index, base + layout::OrenNayarParam_b),
                .multiscatter_term = load_float3(
                    index, base + layout::OrenNayarParam_multiscatter_term)}};
}

SheenClosure ClosurePool::sheen(Expr<std::uint32_t> index) const noexcept {
  return {
      .common = common(index),
      .param = {.roughness = load_float(index, layout::SheenBsdf_roughness),
                .transform_a = load_float(index, layout::SheenBsdf_transformA),
                .transform_b = load_float(index, layout::SheenBsdf_transformB),
                .T = load_float3(index, layout::SheenBsdf_T),
                .B = load_float3(index, layout::SheenBsdf_B)}};
}

VelvetClosure ClosurePool::velvet(Expr<std::uint32_t> index) const noexcept {
  return {
      .common = common(index),
      .param = {.sigma = load_float(index, layout::VelvetBsdf_sigma),
                .invsigma2 = load_float(index, layout::VelvetBsdf_invsigma2)}};
}

ToonClosure ClosurePool::toon(Expr<std::uint32_t> index) const noexcept {
  return {.common = common(index),
          .param = {.size = load_float(index, layout::ToonBsdf_size),
                    .smooth = load_float(index, layout::ToonBsdf_smooth)}};
}

RayPortalClosure
ClosurePool::ray_portal(Expr<std::uint32_t> index) const noexcept {
  return {.common = common(index),
          .param = {.P = load_float3(index, layout::RayPortalClosure_P),
                    .D = load_float3(index, layout::RayPortalClosure_D)}};
}

HairClosure ClosurePool::hair(Expr<std::uint32_t> index) const noexcept {
  return {
      .common = common(index),
      .param = {.T = load_float3(index, layout::HairBsdf_T),
                .roughness1 = load_float(index, layout::HairBsdf_roughness1),
                .roughness2 = load_float(index, layout::HairBsdf_roughness2),
                .offset = load_float(index, layout::HairBsdf_offset)}};
}

ChiangHairClosure
ClosurePool::chiang_hair(Expr<std::uint32_t> index) const noexcept {
  return {.common = common(index),
          .param = {.sigma = load_float3(index, layout::ChiangHairBSDF_sigma),
                    .v = load_float(index, layout::ChiangHairBSDF_v),
                    .s = load_float(index, layout::ChiangHairBSDF_s),
                    .alpha = load_float(index, layout::ChiangHairBSDF_alpha),
                    .eta = load_float(index, layout::ChiangHairBSDF_eta),
                    .m0_roughness =
                        load_float(index, layout::ChiangHairBSDF_m0_roughness),
                    .h = load_float(index, layout::ChiangHairBSDF_h)}};
}

HuangHairClosure
ClosurePool::huang_hair(Expr<std::uint32_t> index) const noexcept {
  const auto tail = load_word(index, layout::HuangHairBSDF_extra);
  return {
      .common = common(index),
      .param = {.sigma = load_float3(index, layout::HuangHairBSDF_sigma),
                .roughness = load_float(index, layout::HuangHairBSDF_roughness),
                .tilt = load_float(index, layout::HuangHairBSDF_tilt),
                .eta = load_float(index, layout::HuangHairBSDF_eta),
                .aspect_ratio =
                    load_float(index, layout::HuangHairBSDF_aspect_ratio),
                .h = load_float(index, layout::HuangHairBSDF_h)},
      .extra = {.R = load_float(tail, layout::HuangHairExtra_R),
                .TT = load_float(tail, layout::HuangHairExtra_TT),
                .TRT = load_float(tail, layout::HuangHairExtra_TRT),
                .Y = load_float3(tail, layout::HuangHairExtra_Y),
                .Z = load_float3(tail, layout::HuangHairExtra_Z),
                .wi = load_float3(tail, layout::HuangHairExtra_wi),
                .radius = load_float(tail, layout::HuangHairExtra_radius),
                .e2 = load_float(tail, layout::HuangHairExtra_e2),
                .pixel_coverage =
                    load_float(tail, layout::HuangHairExtra_pixel_coverage)}};
}

BssrdfClosure ClosurePool::bssrdf(Expr<std::uint32_t> index) const noexcept {
  return {.common = common(index),
          .param = {.radius = load_float3(index, layout::Bssrdf_radius),
                    .albedo = load_float3(index, layout::Bssrdf_albedo),
                    .anisotropy = load_float(index, layout::Bssrdf_anisotropy),
                    .ior = load_float(index, layout::Bssrdf_ior),
                    .alpha = load_float(index, layout::Bssrdf_alpha)}};
}

MicrofacetClosure
ClosurePool::microfacet(Expr<std::uint32_t> index) const noexcept {
  /* Do not eagerly project the discriminated extra payload. Cycles stores a
   * pointer here and dereferences it only after testing fresnel_type. The
   * storage handle keeps that branch-local lifetime in the generated DSL. */
  return {.common = common(index),
          .param = microfacet_param(index),
          .generalized_schlick = {.thin_film = {.thickness = 0.0f, .ior = 0.0f},
                                  .reflection_tint = make_float3(0.0f),
                                  .transmission_tint = make_float3(0.0f),
                                  .f0 = make_float3(0.0f),
                                  .f90 = make_float3(0.0f),
                                  .exponent = 0.0f},
          .storage = this,
          .storage_index = index};
}

FresnelGeneralizedSchlick
ClosurePool::generalized_schlick(Expr<std::uint32_t> index) const noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  const auto v2 = load_vec4(tail, 2u);
  const auto f0 = make_float3(v2.x.as<float>(), v2.y.as<float>(),
                              v2.z.as<float>());
  if (!_options.may_have_thin_film &&
      !_options.may_have_custom_generalized_schlick &&
      !_options.may_have_positive_exponent) {
    return {.thin_film = {.thickness = 0.0f, .ior = 0.0f},
            .reflection_tint = make_float3(1.0f),
            .transmission_tint = make_float3(0.0f),
            .f0 = f0,
            .f90 = make_float3(1.0f),
            .exponent = -1.0f};
  }
  constexpr auto film = layout::FresnelGeneralizedSchlick_thin_film;
  FresnelThinFilm tf{.thickness = 0.0f, .ior = 0.0f};
  if (_options.may_have_thin_film) {
    tf.thickness = load_float(tail, film + layout::FresnelThinFilm_thickness);
    tf.ior = load_float(tail, film + layout::FresnelThinFilm_ior);
  }
  Float3 refl_tint = make_float3(1.0f);
  Float3 trans_tint = make_float3(0.0f);
  if (_options.may_have_custom_generalized_schlick) {
    refl_tint =
        load_float3(tail, layout::FresnelGeneralizedSchlick_reflection_tint);
    trans_tint =
        load_float3(tail, layout::FresnelGeneralizedSchlick_transmission_tint);
  }
  Float3 f90 = make_float3(1.0f);
  Float exponent = -1.0f;
  if (_options.may_have_positive_exponent ||
      _options.may_have_custom_generalized_schlick) {
    const auto v3 = load_vec4(tail, 3u);
    f90 = make_float3(v2.w.as<float>(), v3.x.as<float>(), v3.y.as<float>());
    exponent = v3.z.as<float>();
  }
  return {.thin_film = tf,
          .reflection_tint = refl_tint,
          .transmission_tint = trans_tint,
          .f0 = f0,
          .f90 = f90,
          .exponent = exponent};
}

MicrofacetConductorClosure
ClosurePool::microfacet_conductor(Expr<std::uint32_t> index) const noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelConductor_thin_film;
  FresnelThinFilm tf{.thickness = 0.0f, .ior = 0.0f};
  if (_options.may_have_thin_film) {
    tf.thickness = load_float(tail, film + layout::FresnelThinFilm_thickness);
    tf.ior = load_float(tail, film + layout::FresnelThinFilm_ior);
  }
  return {
      .common = common(index),
      .param = microfacet_param(index),
      .conductor = {
          .thin_film = tf,
          .ior = load_float3(tail, layout::FresnelConductor_ior +
                                       layout::complex_Spectrum_re),
          .extinction = load_float3(tail, layout::FresnelConductor_ior +
                                              layout::complex_Spectrum_im)}};
}

MicrofacetF82TintClosure
ClosurePool::microfacet_f82_tint(Expr<std::uint32_t> index) const noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelF82Tint_thin_film;
  FresnelThinFilm tf{.thickness = 0.0f, .ior = 0.0f};
  if (_options.may_have_thin_film) {
    tf.thickness = load_float(tail, film + layout::FresnelThinFilm_thickness);
    tf.ior = load_float(tail, film + layout::FresnelThinFilm_ior);
  }
  return {.common = common(index),
          .param = microfacet_param(index),
          .f82_tint = {
              .thin_film = tf,
              .f0 = load_float3(tail, layout::FresnelF82Tint_f0),
              .b = load_float3(tail, layout::FresnelF82Tint_b)}};
}

MicrofacetParam
ClosurePool::microfacet_param(Expr<std::uint32_t> index) const noexcept {
  const auto v2 = load_vec4(index, 2u);
  Float3 tangent = make_float3(0.0f);
  if (_options.may_have_anisotropic_microfacet) {
    const auto v4 = load_vec4(index, 4u);
    tangent = make_float3(v4.x.as<float>(), v4.y.as<float>(), v4.z.as<float>());
  }
  return {.alpha_x = v2.x.as<float>(),
          .alpha_y = v2.y.as<float>(),
          .ior = v2.z.as<float>(),
          .energy_scale = v2.w.as<float>(),
          .fresnel_type = load_word(index, layout::MicrofacetBsdf_fresnel_type),
          .T = tangent};
}

namespace detail {

void node_closure_set_weight(Cursor &cursor, Float3 &closure_weight,
                             SpectralClosureWeight *spectral_weight) noexcept {
  const auto x = cursor.floating();
  const auto y = cursor.floating();
  const auto z = cursor.floating();
  closure_weight = make_float3(x, y, z);
  if (spectral_weight != nullptr) {
    spectral_weight->valid = 0u;
  }
}

void node_closure_weight(Cursor &cursor, Stack &stack, Float3 &closure_weight,
                         SpectralClosureWeight *spectral_weight) noexcept {
  const auto packed = cursor.word();
  const auto offset = cursor.byte(packed, 0u);
  closure_weight = stack_load_float3(stack, offset);
  if (spectral_weight != nullptr && stack.is_spectral()) {
    const Bool found =
        stack.try_load_spectral_slot(offset, spectral_weight->spd);
    spectral_weight->valid = select(0u, 1u, found);
  }
}

void node_emission_weight(Cursor &cursor, Stack &stack, Float3 &closure_weight,
                          SpectralClosureWeight *spectral_weight) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto strength = cursor.word();
  const auto strength_val = stack_load_input_float(stack, strength);
  closure_weight =
      stack_load_input_float3(stack, color_x, color_y, color_z) * strength_val;
  if (spectral_weight != nullptr && stack.is_spectral()) {
    spectral_weight->valid = 0u;
    spectral_weight->spd = make_float4(0.0f);
    if (const auto cx = try_const_uint(color_x)) {
      if ((*cx >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
        Float4 spd = make_float4(0.0f);
        const Bool found = stack.try_load_spectral_slot(*cx & 0xffu, spd);
        spectral_weight->spd =
            select(spectral_weight->spd, spd * strength_val, found);
        spectral_weight->valid = select(0u, 1u, found);
      }
    } else if (!stack.is_direct_jit()) {
      $if((color_x >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
        Float4 spd = make_float4(0.0f);
        const Bool found = stack.try_load_spectral_slot(color_x & 0xffu, spd);
        spectral_weight->spd =
            select(spectral_weight->spd, spd * strength_val, found);
        spectral_weight->valid = select(0u, 1u, found);
      };
    }
  }
}

[[nodiscard]] Float4 resolve_spectral_emission_weight(
    const ShaderData &shader_data, Expr<luisa::float3> weight,
    Expr<float> weight_scale,
    const SpectralClosureWeight *spectral_weight) noexcept {
  const auto &sc = *shader_data.spectral;
  if (spectral_weight == nullptr) {
    return cycles_spectrum::rgb_to_spectrum_illuminant(
        *sc.tables, weight, sc.wavelengths.lambdas);
  }
  Float4 w_spec = make_float4(0.0f);
  $if(spectral_weight->valid != 0u) {
    w_spec = spectral_weight->spd * weight_scale;
  }
  $else {
    w_spec = cycles_spectrum::rgb_to_spectrum_illuminant(
        *sc.tables, weight, sc.wavelengths.lambdas);
  };
  return w_spec;
}

void node_mix_closure(Cursor &cursor, Stack &stack) noexcept {
  const auto factor = cursor.word();
  const auto packed_offsets = cursor.word();
  const auto input_weight_offset = cursor.byte(packed_offsets, 0u);
  const auto weight1_offset = cursor.byte(packed_offsets, 1u);
  const auto weight2_offset = cursor.byte(packed_offsets, 2u);

  const Float weight = clamp(stack_load_input_float(stack, factor), 0.0f, 1.0f);
  const auto input_weight =
      stack_load_float_default(stack, input_weight_offset, 1.0f);
  if (const auto w1 = try_const_uint(weight1_offset)) {
    if (*w1 != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_float(stack, *w1, input_weight * (1.0f - weight));
    }
  } else {
    $if(weight1_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_float(stack, weight1_offset, input_weight * (1.0f - weight));
    };
  }
  if (const auto w2 = try_const_uint(weight2_offset)) {
    if (*w2 != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_float(stack, *w2, input_weight * weight);
    }
  } else {
    $if(weight2_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_float(stack, weight2_offset, input_weight * weight);
    };
  }
}

void node_closure_emission(
    const KernelGlobals &kernel_globals, Cursor &cursor, Stack &stack,
    Expr<luisa::float3> closure_weight, ShaderData &shader_data,
    const EvaluationTransition &transition,
    const SpectralClosureWeight *spectral_weight) noexcept {
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  Float3 weight = closure_weight;
  Float weight_scale = 1.0f;
  Bool active = true;

  if (const auto mwo = try_const_uint(mix_weight_offset)) {
    if (*mwo != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      const auto mix_weight = stack_load_float(stack, *mwo);
      $if(mix_weight == 0.0f) { active = false; }
      $else {
        weight *= mix_weight;
        weight_scale *= mix_weight;
      };
    }
  } else {
    $if(mix_weight_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      const auto mix_weight = stack_load_float(stack, mix_weight_offset);
      $if(mix_weight == 0.0f) { active = false; }
      $else {
        weight *= mix_weight;
        weight_scale *= mix_weight;
      };
    };
  }

  $if(active) {
    $if((shader_data.flag & shader_data_is_volume_shader_eval) != 0u) {
      if (const auto density =
              kernel_globals.object_volume_density(shader_data.object)) {
        weight *= *density;
        weight_scale *= *density;
      } else {
        $if(shader_data.object != object_none) {
          active = false;
          transition.unsupported();
        };
      }
    };
    $if(active) {
      const bool has_spectral =
          kernel_globals.is_spectral() && shader_data.spectral.has_value() &&
          shader_data.spectral_emission_background.has_value();
      Float4 w_spec = make_float4(0.0f);
      if (has_spectral) {
        w_spec = resolve_spectral_emission_weight(shader_data, weight,
                                                  weight_scale, spectral_weight);
      }
      $if((shader_data.flag & shader_data_emission) != 0u) {
        shader_data.closure_emission_background += weight;
        if (has_spectral) {
          *shader_data.spectral_emission_background += w_spec;
        }
      }
      $else {
        shader_data.flag |= shader_data_emission;
        shader_data.closure_emission_background = weight;
        if (has_spectral) {
          *shader_data.spectral_emission_background = w_spec;
        }
      };
    };
  };
}

void node_closure_holdout(Cursor &cursor, Stack &stack,
                         Expr<luisa::float3> closure_weight,
                         ShaderData &shader_data) noexcept {
  const auto mix_weight_offset = cursor.byte(cursor.word(), 0u);
  Float3 weight = closure_weight;
  Bool active = true;
  if (const auto mwo = try_const_uint(mix_weight_offset)) {
    if (*mwo != static_cast<unsigned>(SVM_STACK_INVALID)) {
      const auto mix_weight = stack_load_float(stack, *mwo);
      $if(mix_weight == 0.0f) { active = false; }
      $else { weight *= mix_weight; };
    }
  } else {
    $if(mix_weight_offset != static_cast<unsigned>(SVM_STACK_INVALID)) {
      const auto mix_weight = stack_load_float(stack, mix_weight_offset);
      $if(mix_weight == 0.0f) { active = false; }
      $else { weight *= mix_weight; };
    };
  }
  $if(active) {
    if (shader_data.closure != nullptr) {
      static_cast<void>(shader_data.closure->allocate(
          static_cast<unsigned>(CLOSURE_HOLDOUT_ID), weight));
    }
    // Native closure_alloc failure does not suppress this flag; only the
    // exactly-zero linked mix weight exits before it.
    shader_data.flag |= static_cast<unsigned>(SD_HOLDOUT);
  };
}

void node_closure_background(
    Cursor &cursor, Stack &stack, Expr<luisa::float3> closure_weight,
    ShaderData &shader_data,
    const SpectralClosureWeight *spectral_weight) noexcept {
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  Float3 weight = closure_weight;
  Float weight_scale = 1.0f;
  Bool active = true;

  if (const auto mwo = try_const_uint(mix_weight_offset)) {
    if (*mwo != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      const auto mix_weight = stack_load_float(stack, *mwo);
      $if(mix_weight == 0.0f) { active = false; }
      $else {
        weight *= mix_weight;
        weight_scale *= mix_weight;
      };
    }
  } else {
    $if(mix_weight_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      const auto mix_weight = stack_load_float(stack, mix_weight_offset);
      $if(mix_weight == 0.0f) { active = false; }
      $else {
        weight *= mix_weight;
        weight_scale *= mix_weight;
      };
    };
  }

  $if(active) {
    const bool has_spectral =
        shader_data.spectral.has_value() &&
        shader_data.spectral_emission_background.has_value();
    Float4 w_spec = make_float4(0.0f);
    if (has_spectral) {
      w_spec = resolve_spectral_emission_weight(shader_data, weight,
                                                weight_scale, spectral_weight);
    }
    $if((shader_data.flag & shader_data_emission) != 0u) {
      shader_data.closure_emission_background += weight;
      if (has_spectral) {
        *shader_data.spectral_emission_background += w_spec;
      }
    }
    $else {
      shader_data.flag |= shader_data_emission;
      shader_data.closure_emission_background = weight;
      if (has_spectral) {
        *shader_data.spectral_emission_background = w_spec;
      }
    };
  };
}

void node_closure_bsdf(const KernelGlobals &kernel_globals, Cursor &cursor,
                       Stack &stack, Expr<luisa::float3> closure_weight,
                       ShaderType shader_type, std::uint32_t kernel_features,
                       std::uint32_t node_feature_mask,
                       ShaderData &shader_data, const PathState &path_state,
                       const EvaluationTransition &transition,
                       const SpectralClosureWeight *spectral_weight) noexcept {
  const auto closure_type = cursor.word();
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  const auto const_ct = try_const_uint(closure_type);
  const auto const_mwo = try_const_uint(mix_weight_offset);
  const bool unit_mix_weight =
      const_mwo.has_value() &&
      *const_mwo == static_cast<std::uint32_t>(SVM_STACK_INVALID);
  const auto mix_weight =
      stack_load_float_default(stack, mix_weight_offset, 1.0f);

  if (shader_type != SHADER_TYPE_SURFACE) {
    if (!cursor.is_static()) {
      node_closure_bsdf_skip(cursor, closure_type);
    }
    return;
  }

  const auto usage = kernel_globals.closure_usage();
  const auto node_mask = usage.svm_bsdf_node_mask;

  const auto with_spectral_bsdf_weight = [&](auto &&emit_fn) noexcept {
    if (spectral_weight != nullptr && kernel_globals.is_spectral() &&
        shader_data.spectral.has_value() && shader_data.closure != nullptr &&
        shader_data.closure->is_spectral()) {
      auto &pool = *shader_data.closure;
      const UInt prev_count = pool.count();
      emit_fn();
      $if((spectral_weight->valid != 0u) & (pool.count() == prev_count + 1u)) {
        const auto &sc = *shader_data.spectral;
        const Float4 illum = luisa::compute::max(
            cycles_spectrum::sample_illuminant_4(*sc.tables,
                                                 sc.wavelengths.lambdas),
            make_float4(0.1f));
        const Float4 spec_refl = luisa::compute::max(
            (spectral_weight->spd / illum) * mix_weight, make_float4(0.0f));
        pool.set_spectral_weight(prev_count, spec_refl);
      };
      return;
    }
    emit_fn();
  };

  const auto emit_diffuse = [&]() noexcept {
    with_spectral_bsdf_weight([&]() noexcept {
      const auto color_x = cursor.word();
      const auto color_y = cursor.word();
      const auto color_z = cursor.word();
      const auto roughness_input = cursor.word();
      const auto normal_packed = cursor.word();
      const auto normal_offset = cursor.byte(normal_packed, 0u);
      auto normal =
          stack_load_float3_default(stack, normal_offset, shader_data.N);
      normal = native_vector_math::safe_normalize_nonzero_or(normal,
                                                             shader_data.N);

      const auto weight = closure_weight * mix_weight;
      const bool may_lambertian =
          (usage.diffuse_node_mask & ClosureUsage::diffuse_lambertian) != 0u;
      const bool may_oren_nayar =
          (usage.diffuse_node_mask & ClosureUsage::diffuse_oren_nayar) != 0u;
      if (may_lambertian && !may_oren_nayar) {
        static_cast<void>(color_x);
        static_cast<void>(color_y);
        static_cast<void>(color_z);
        static_cast<void>(roughness_input);
        diffuse_setup(shader_data, normal, weight);
      } else {
        const auto roughness = stack_load_input_float(stack, roughness_input);
        if (!may_lambertian && may_oren_nayar) {
          const auto color = clamp(
              stack_load_input_float3(stack, color_x, color_y, color_z), 0.0f,
              1.0f);
          oren_nayar_setup(shader_data, normal, weight, roughness, color);
        } else {
          $if(roughness < 1.0e-5f) {
            diffuse_setup(shader_data, normal, weight);
          }
          $else {
            const auto color = clamp(
                stack_load_input_float3(stack, color_x, color_y, color_z), 0.0f,
                1.0f);
            oren_nayar_setup(shader_data, normal, weight, roughness, color);
          };
        }
      }
    });
  };

  const auto emit_translucent = [&]() noexcept {
    with_spectral_bsdf_weight([&]() noexcept {
      const auto unused_param = cursor.word();
      const auto normal_packed = cursor.word();
      static_cast<void>(unused_param);
      const auto normal_offset = cursor.byte(normal_packed, 0u);
      auto normal =
          stack_load_float3_default(stack, normal_offset, shader_data.N);
      normal = native_vector_math::safe_normalize_nonzero_or(normal,
                                                             shader_data.N);
      const auto weight = closure_weight * mix_weight;
      translucent_setup(shader_data,
                        detail::maybe_ensure_valid_specular_reflection(
                            shader_data, normal),
                        weight);
    });
  };

  const auto emit_transparent = [&]() noexcept {
    const auto unused_param = cursor.word();
    const auto unused_normal = cursor.word();
    static_cast<void>(unused_param);
    static_cast<void>(unused_normal);
    transparent_setup(shader_data, path_state, closure_weight * mix_weight);
  };

  const auto emit_glossy = [&](Expr<std::uint32_t> type) noexcept {
    with_spectral_bsdf_weight([&]() noexcept {
      node_glossy_bsdf(kernel_globals, cursor, stack, type, mix_weight,
                       closure_weight, shader_data, path_state);
    });
  };

  const auto emit_refraction = [&](Expr<std::uint32_t> type) noexcept {
    with_spectral_bsdf_weight([&]() noexcept {
      node_refraction_bsdf(kernel_globals, cursor, stack, type, mix_weight,
                           closure_weight, shader_data, path_state);
    });
  };

  const auto emit_static_bsdf = [&](std::uint32_t ct) noexcept {
    switch (static_cast<ClosureType>(ct)) {
      case CLOSURE_BSDF_PRINCIPLED_ID:
        node_principled_bsdf(
            kernel_globals, cursor, stack, mix_weight, true,
            (kernel_features & kernel_feature_subsurface) != 0u, shader_data,
            path_state, transition);
        return;
      case CLOSURE_BSDF_DIFFUSE_ID:
        emit_diffuse();
        return;
      case CLOSURE_BSDF_TRANSLUCENT_ID:
        emit_translucent();
        return;
      case CLOSURE_BSDF_TRANSPARENT_ID:
        emit_transparent();
        return;
      case CLOSURE_BSDF_PHYSICAL_CONDUCTOR:
      case CLOSURE_BSDF_F82_CONDUCTOR:
        node_metallic_bsdf(kernel_globals, cursor, stack, ct, mix_weight,
                           shader_data, path_state);
        return;
      case CLOSURE_BSDF_RAY_PORTAL_ID:
        node_ray_portal(cursor, stack, closure_weight, mix_weight, shader_data);
        return;
      case CLOSURE_BSDF_MICROFACET_GGX_ID:
      case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
      case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
      case CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID:
        emit_glossy(ct);
        return;
      case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
      case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
        emit_refraction(ct);
        return;
      case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
      case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
      case CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID:
        node_glass_bsdf(kernel_globals, cursor, stack, ct, mix_weight,
                        shader_data, path_state);
        return;
      case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
      case CLOSURE_BSDF_SHEEN_ID:
        node_sheen(kernel_globals, cursor, stack, ct, closure_weight,
                   mix_weight, shader_data);
        return;
      case CLOSURE_BSDF_GLOSSY_TOON_ID:
      case CLOSURE_BSDF_DIFFUSE_TOON_ID:
        node_toon(kernel_globals, cursor, stack, ct, closure_weight, mix_weight,
                  shader_data, path_state);
        return;
      case CLOSURE_BSDF_HAIR_CHIANG_ID:
      case CLOSURE_BSDF_HAIR_HUANG_ID:
        if ((kernel_features & kernel_feature_hair) != 0u &&
            (kernel_features & kernel_feature_node_principled_hair) != 0u) {
          node_principled_hair(kernel_globals, cursor, stack, ct,
                               closure_weight, mix_weight, shader_data,
                               path_state);
          return;
        }
        break;
      case CLOSURE_BSDF_HAIR_REFLECTION_ID:
      case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
        if ((kernel_features & kernel_feature_hair) != 0u) {
          node_hair(cursor, stack, ct, closure_weight, mix_weight, shader_data);
          return;
        }
        break;
      case CLOSURE_BSSRDF_BURLEY_ID:
      case CLOSURE_BSSRDF_RANDOM_WALK_ID:
      case CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID:
      case CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID:
        if ((kernel_features & kernel_feature_subsurface) != 0u) {
          node_bssrdf(cursor, stack, ct, closure_weight, mix_weight,
                      shader_data, path_state);
          return;
        }
        break;
      default:
        break;
    }
    if (!cursor.is_static()) {
      node_closure_bsdf_skip(cursor, closure_type, node_mask);
    }
    transition.unsupported();
  };

  if ((node_feature_mask & kernel_feature_node_bsdf) != 0u) {
    if (const_ct.has_value()) {
      if (unit_mix_weight) {
        emit_static_bsdf(*const_ct);
      } else {
        $if(mix_weight != 0.0f) { emit_static_bsdf(*const_ct); }
        $else {
          if (!cursor.is_static()) {
            node_closure_bsdf_skip(cursor, closure_type, node_mask);
          }
        };
      }
      return;
    }

    const auto cases = [&](std::initializer_list<ClosureType> types,
                           auto &&body) noexcept {
      luisa::vector<std::uint32_t> enabled;
      enabled.reserve(types.size());
      for (const auto type : types) {
        const auto id = static_cast<std::uint32_t>(type);
        if (usage.has_bsdf_node(id)) {
          enabled.emplace_back(id);
        }
      }
      if (!enabled.empty()) {
        luisa::compute::detail::SwitchCaseStmtBuilder{
            luisa::span<const std::uint32_t>{enabled}} %
            std::forward<decltype(body)>(body);
      }
    };

    $if(mix_weight == 0.0f) {
      node_closure_bsdf_skip(cursor, closure_type, node_mask);
    }
    $else {
      // Cycles dispatches the payload type once, preserving shared case bodies.
      // A missing physical pool still permits transparent/portal/Principled
      // shader-state effects; each native setup owns its allocation guard.
      $switch(closure_type) {
        if (usage.has_bsdf_node(CLOSURE_BSDF_PRINCIPLED_ID)) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_PRINCIPLED_ID) {
            node_principled_bsdf(
                kernel_globals, cursor, stack, mix_weight, true,
                (kernel_features & kernel_feature_subsurface) != 0u,
                shader_data, path_state, transition);
          };
        }
        if (usage.has_bsdf_node(CLOSURE_BSDF_DIFFUSE_ID)) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_ID) { emit_diffuse(); };
        }
        if (usage.has_bsdf_node(CLOSURE_BSDF_TRANSLUCENT_ID)) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_TRANSLUCENT_ID) { emit_translucent(); };
        }
        if (usage.has_bsdf_node(CLOSURE_BSDF_TRANSPARENT_ID)) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_TRANSPARENT_ID) { emit_transparent(); };
        }
        cases({CLOSURE_BSDF_PHYSICAL_CONDUCTOR, CLOSURE_BSDF_F82_CONDUCTOR},
              [&] {
                node_metallic_bsdf(kernel_globals, cursor, stack, closure_type,
                                   mix_weight, shader_data, path_state);
              });
        if (usage.has_bsdf_node(CLOSURE_BSDF_RAY_PORTAL_ID)) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_RAY_PORTAL_ID) {
            node_ray_portal(cursor, stack, closure_weight, mix_weight,
                            shader_data);
          };
        }
        cases({CLOSURE_BSDF_MICROFACET_GGX_ID,
               CLOSURE_BSDF_MICROFACET_BECKMANN_ID,
               CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID,
               CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID},
              [&] { emit_glossy(closure_type); });
        cases({CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
               CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID},
              [&] { emit_refraction(closure_type); });
        cases({CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
               CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID,
               CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID},
              [&] {
                node_glass_bsdf(kernel_globals, cursor, stack, closure_type,
                                mix_weight, shader_data, path_state);
              });
        if (usage.has_bsdf_node(CLOSURE_BSDF_ASHIKHMIN_VELVET_ID)) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_ASHIKHMIN_VELVET_ID) {
            node_sheen(
                kernel_globals, cursor, stack,
                static_cast<std::uint32_t>(CLOSURE_BSDF_ASHIKHMIN_VELVET_ID),
                closure_weight, mix_weight, shader_data);
          };
        }
        if (usage.has_bsdf_node(CLOSURE_BSDF_SHEEN_ID)) {
          PSYCLES_SVM_CASE(CLOSURE_BSDF_SHEEN_ID) {
            node_sheen(kernel_globals, cursor, stack,
                       static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID),
                       closure_weight, mix_weight, shader_data);
          };
        }
        cases({CLOSURE_BSDF_GLOSSY_TOON_ID, CLOSURE_BSDF_DIFFUSE_TOON_ID}, [&] {
          node_toon(kernel_globals, cursor, stack, closure_type, closure_weight,
                    mix_weight, shader_data, path_state);
        });
        if ((kernel_features & kernel_feature_hair) != 0u) {
          if ((kernel_features & kernel_feature_node_principled_hair) != 0u) {
            cases({CLOSURE_BSDF_HAIR_CHIANG_ID, CLOSURE_BSDF_HAIR_HUANG_ID},
                  [&] {
                    node_principled_hair(kernel_globals, cursor, stack,
                                         closure_type, closure_weight,
                                         mix_weight, shader_data, path_state);
                  });
          }
          cases({CLOSURE_BSDF_HAIR_REFLECTION_ID,
                 CLOSURE_BSDF_HAIR_TRANSMISSION_ID},
                [&] {
                  node_hair(cursor, stack, closure_type, closure_weight,
                            mix_weight, shader_data);
                });
        }
        if ((kernel_features & kernel_feature_subsurface) != 0u) {
          cases({CLOSURE_BSSRDF_BURLEY_ID, CLOSURE_BSSRDF_RANDOM_WALK_ID,
                 CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID,
                 CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID},
                [&] {
                  node_bssrdf(cursor, stack, closure_type, closure_weight,
                              mix_weight, shader_data, path_state);
                });
        }
        $default {
          node_closure_bsdf_skip(cursor, closure_type, node_mask);
          transition.unsupported();
        };
      };
    };
    return;
  }

  if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
    if (const_ct.has_value()) {
      if (*const_ct != static_cast<std::uint32_t>(CLOSURE_BSDF_PRINCIPLED_ID) ||
          !usage.has_bsdf_node(CLOSURE_BSDF_PRINCIPLED_ID)) {
        if (!cursor.is_static()) {
          node_closure_bsdf_skip(cursor, closure_type, node_mask);
        }
        return;
      }
      if (unit_mix_weight) {
        node_principled_bsdf(kernel_globals, cursor, stack, mix_weight, false,
                             false, shader_data, path_state, transition);
      } else {
        $if(mix_weight != 0.0f) {
          node_principled_bsdf(kernel_globals, cursor, stack, mix_weight, false,
                               false, shader_data, path_state, transition);
        }
        $else {
          if (!cursor.is_static()) {
            node_closure_bsdf_skip(cursor, closure_type, node_mask);
          }
        };
      }
      return;
    }
    if (!usage.has_bsdf_node(CLOSURE_BSDF_PRINCIPLED_ID)) {
      node_closure_bsdf_skip(cursor, closure_type, node_mask);
      return;
    }
    $if((mix_weight == 0.0f) |
        (closure_type !=
         static_cast<std::uint32_t>(CLOSURE_BSDF_PRINCIPLED_ID))) {
      node_closure_bsdf_skip(cursor, closure_type, node_mask);
    }
    $else {
      // Cycles evaluates Principled emission with num_closure_left == 0.
      // Sheen/coat still use local BSDF values for layer attenuation, but
      // neither their evaluation nor transparency requires a closure array.
      node_principled_bsdf(kernel_globals, cursor, stack, mix_weight, false,
                           false, shader_data, path_state, transition);
    };
    return;
  }

  if (!cursor.is_static()) {
    node_closure_bsdf_skip(cursor, closure_type, node_mask);
  }
}

void node_closure_bsdf_skip(Cursor &cursor,
                            Expr<std::uint32_t> closure_type,
                            std::uint64_t svm_bsdf_node_mask) noexcept {
  UInt words = static_cast<std::uint32_t>(sizeof(SVMNodeSimpleBsdfData) /
                                          sizeof(std::uint32_t));
  const auto skip_cases = [&](std::initializer_list<ClosureType> types,
                              std::uint32_t word_count) noexcept {
    luisa::vector<std::uint32_t> enabled;
    enabled.reserve(types.size());
    for (const auto type : types) {
      const auto id = static_cast<std::uint32_t>(type);
      if (id >= 64u || (svm_bsdf_node_mask & (std::uint64_t{1u} << id)) != 0ull) {
        enabled.emplace_back(id);
      }
    }
    if (!enabled.empty()) {
      luisa::compute::detail::SwitchCaseStmtBuilder{
          luisa::span<const std::uint32_t>{enabled}} %
          [&] { words = word_count; };
    }
  };
  $switch(closure_type) {
    skip_cases({CLOSURE_BSDF_PRINCIPLED_ID},
               sizeof(SVMNodePrincipledBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_HAIR_CHIANG_ID, CLOSURE_BSDF_HAIR_HUANG_ID},
               sizeof(SVMNodePrincipledHairBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_PHYSICAL_CONDUCTOR, CLOSURE_BSDF_F82_CONDUCTOR},
               sizeof(SVMNodeMetallicBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_DIFFUSE_ID, CLOSURE_BSDF_OREN_NAYAR_ID,
                CLOSURE_BSDF_BURLEY_ID},
               sizeof(SVMNodeDiffuseBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_RAY_PORTAL_ID},
               sizeof(SVMNodeRayPortalBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_MICROFACET_GGX_ID,
                CLOSURE_BSDF_MICROFACET_BECKMANN_ID,
                CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID,
                CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID},
               sizeof(SVMNodeGlossyBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
                CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID},
               sizeof(SVMNodeRefractionBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID,
                CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID},
               sizeof(SVMNodeGlassBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_GLOSSY_TOON_ID, CLOSURE_BSDF_DIFFUSE_TOON_ID},
               sizeof(SVMNodeToonBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSDF_HAIR_REFLECTION_ID,
                CLOSURE_BSDF_HAIR_TRANSMISSION_ID},
               sizeof(SVMNodeHairBsdfData) / sizeof(std::uint32_t));
    skip_cases({CLOSURE_BSSRDF_BURLEY_ID, CLOSURE_BSSRDF_RANDOM_WALK_ID,
                CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID},
               sizeof(SVMNodeBssrdfData) / sizeof(std::uint32_t));
    $default{};
  };
  cursor.advance(words);
}

void node_closure_bsdf_shadow_only(const KernelGlobals &kernel_globals,
                                   Cursor &cursor, Stack &stack,
                                   Expr<luisa::float3> closure_weight,
                                   ShaderType shader_type,
                                   std::uint32_t kernel_features,
                                   std::uint32_t node_feature_mask,
                                   ShaderData &shader_data,
                                   const PathState &path_state,
                                   const EvaluationTransition &transition) noexcept {
  if (shader_type != SHADER_TYPE_SURFACE || !cursor.is_static()) {
    node_closure_bsdf(kernel_globals, cursor, stack, closure_weight, shader_type,
                      kernel_features, node_feature_mask, shader_data,
                      path_state, transition);
    return;
  }
  const auto const_ct = try_const_uint(cursor.word_at(0u));
  if (!const_ct) {
    node_closure_bsdf(kernel_globals, cursor, stack, closure_weight, shader_type,
                      kernel_features, node_feature_mask, shader_data,
                      path_state, transition);
    return;
  }

  static_cast<void>(cursor.word()); // closure_type
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  const auto mix_weight =
      stack_load_float_default(stack, mix_weight_offset, 1.0f);

  const auto is_const_float_at_least = [](std::uint32_t bits,
                                          float threshold) noexcept {
    if ((bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      return false;
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return !(value < threshold) && (value == value);
  };
  const auto is_const_float_at_most = [](std::uint32_t bits,
                                         float threshold) noexcept {
    if ((bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
      return false;
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value <= threshold;
  };

  const auto ct = static_cast<ClosureType>(*const_ct);
  if (ct == CLOSURE_BSDF_TRANSPARENT_ID) {
    cursor.advance(static_cast<std::uint32_t>(sizeof(SVMNodeSimpleBsdfData) /
                                              sizeof(std::uint32_t)));
    transparent_setup(shader_data, path_state, closure_weight * mix_weight);
    return;
  }
  if (ct == CLOSURE_BSDF_RAY_PORTAL_ID) {
    node_ray_portal(cursor, stack, closure_weight, mix_weight, shader_data);
    return;
  }
  if (ct == CLOSURE_BSDF_PRINCIPLED_ID) {
    constexpr std::uint32_t alpha_word = static_cast<std::uint32_t>(
        offsetof(SVMNodePrincipledBsdfData, alpha) / sizeof(std::uint32_t));
    constexpr std::uint32_t trans_word = static_cast<std::uint32_t>(
        offsetof(SVMNodePrincipledBsdfData, transmission_weight) /
        sizeof(std::uint32_t));
    constexpr std::uint32_t thin_wall_word = static_cast<std::uint32_t>(
        offsetof(SVMNodePrincipledBsdfData, thin_wall) / sizeof(std::uint32_t));
    constexpr std::uint32_t total_words = static_cast<std::uint32_t>(
        sizeof(SVMNodePrincipledBsdfData) / sizeof(std::uint32_t));

    const auto tw_val = try_const_uint(cursor.word_at(thin_wall_word));
    const auto tw_off = try_const_uint(cursor.word_at(thin_wall_word + 1u));
    const auto trans_bits = try_const_uint(cursor.word_at(trans_word));
    const bool thin_wall_inactive =
        (tw_val && tw_off &&
         (*tw_off & 0xffu) == static_cast<std::uint32_t>(SVM_STACK_INVALID) &&
         *tw_val == 0u) ||
        (trans_bits && is_const_float_at_most(*trans_bits, 0.0f));
    if (!thin_wall_inactive) {
      node_principled_bsdf(
          kernel_globals, cursor, stack, mix_weight, true,
          (kernel_features & kernel_feature_subsurface) != 0u, shader_data,
          path_state, transition);
      return;
    }

    const auto alpha_bits = try_const_uint(cursor.word_at(alpha_word));
    if (alpha_bits && is_const_float_at_least(*alpha_bits, 1.0f)) {
      cursor.advance(total_words);
      return;
    }
    const auto alpha =
        clamp(stack_load_input_float(stack, cursor.word_at(alpha_word)), 0.0f,
              1.0f);
    cursor.advance(total_words);
    $if(alpha < 1.0f) {
      transparent_setup(shader_data, path_state,
                        make_float3(mix_weight) * (1.0f - alpha));
    };
    return;
  }

  std::uint32_t skip_words = 0u;
  switch (ct) {
    case CLOSURE_BSDF_DIFFUSE_ID:
    case CLOSURE_BSDF_TRANSLUCENT_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeDiffuseBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_ASHIKHMIN_VELVET_ID:
    case CLOSURE_BSDF_SHEEN_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeSimpleBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_PHYSICAL_CONDUCTOR:
    case CLOSURE_BSDF_F82_CONDUCTOR:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeMetallicBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_ID:
    case CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID:
    case CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlossyBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeRefractionBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID:
    case CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID:
    case CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlassBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_GLOSSY_TOON_ID:
    case CLOSURE_BSDF_DIFFUSE_TOON_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeToonBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_HAIR_REFLECTION_ID:
    case CLOSURE_BSDF_HAIR_TRANSMISSION_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeHairBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSDF_HAIR_CHIANG_ID:
    case CLOSURE_BSDF_HAIR_HUANG_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodePrincipledHairBsdfData) / sizeof(std::uint32_t));
      break;
    case CLOSURE_BSSRDF_BURLEY_ID:
    case CLOSURE_BSSRDF_RANDOM_WALK_ID:
    case CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID:
    case CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID:
      skip_words = static_cast<std::uint32_t>(
          sizeof(SVMNodeBssrdfData) / sizeof(std::uint32_t));
      break;
    default:
      break;
  }
  cursor.advance(skip_words);
}

} // namespace detail

} // namespace psycles::luisa_backend::cycles_svm

#undef PSYCLES_SVM_CASE
