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

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace layout = detail::closure_layout;

static constexpr auto closure_rows = layout::ShaderClosure_sizeof / 16u;
static_assert(layout::ShaderClosure_sizeof % 16u == 0u);

ClosurePool::ClosurePool(std::size_t capacity) noexcept
    : _capacity{std::min(capacity, maximum_closure_capacity)},
      _storage{std::max(_capacity, std::size_t{1u}) * closure_rows}, _count{0u},
      _left{static_cast<std::uint32_t>(_capacity)} {}

UInt ClosurePool::load_word(Expr<std::uint32_t> slot,
                            std::uint32_t byte_offset) const noexcept {
  // Do address arithmetic in the pointer-width domain, as a native array
  // GEP does. A 32-bit multiply would introduce wraparound before indexing.
  const auto row =
      slot.cast<luisa::ulong>() * static_cast<luisa::ulong>(closure_rows) +
      static_cast<luisa::ulong>(byte_offset / 16u);
  return _storage[row][(byte_offset % 16u) / 4u];
}

void ClosurePool::store_word(Expr<std::uint32_t> slot,
                             std::uint32_t byte_offset,
                             Expr<std::uint32_t> value) noexcept {
  const auto row =
      slot.cast<luisa::ulong>() * static_cast<luisa::ulong>(closure_rows) +
      static_cast<luisa::ulong>(byte_offset / 16u);
  _storage[row][(byte_offset % 16u) / 4u] = value;
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

ClosurePool::Allocation
ClosurePool::allocate(Expr<std::uint32_t> closure_type,
                      Expr<luisa::float3> weight) noexcept {
  Allocation allocation{.index = _count, .valid = false};
  $if(_left != 0u) {
    store_word(allocation.index, layout::ShaderClosure_type, closure_type);
    store_float3(allocation.index, layout::ShaderClosure_weight, weight);
    store_float(allocation.index, layout::ShaderClosure_sample_weight, 0.0f);
    _count += 1u;
    _left -= 1u;
    allocation.valid = true;
  };
  return allocation;
}

Bool ClosurePool::allocate_extra(const Allocation &owner,
                                 Expr<std::uint32_t> slot_count,
                                 ExtraPayload payload) noexcept {
  Bool allocated = false;
  $if(owner.valid) {
    $if(slot_count <= _left) {
      _left -= slot_count;
      const auto pointer_field = payload == ExtraPayload::huang_hair
                                     ? layout::HuangHairBSDF_extra
                                     : layout::MicrofacetBsdf_fresnel;
      store_word(owner.index, pointer_field, _count + _left);
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
  store_float3(index, layout::ShaderClosure_weight, weight);
}

void ClosurePool::add_weight(Expr<std::uint32_t> index,
                             Expr<luisa::float3> weight) noexcept {
  set_weight(index, load_float3(index, layout::ShaderClosure_weight) + weight);
}

void ClosurePool::set_sample_weight(Expr<std::uint32_t> index,
                                    Expr<float> sample_weight) noexcept {
  store_float(index, layout::ShaderClosure_sample_weight, sample_weight);
}

void ClosurePool::add_sample_weight(Expr<std::uint32_t> index,
                                    Expr<float> sample_weight) noexcept {
  set_sample_weight(index,
                    load_float(index, layout::ShaderClosure_sample_weight) +
                        sample_weight);
}

void ClosurePool::set_normal(Expr<std::uint32_t> index,
                             Expr<luisa::float3> normal) noexcept {
  store_float3(index, layout::ShaderClosure_N, normal);
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
  store_float(index, layout::MicrofacetBsdf_alpha_x, param.alpha_x);
  store_float(index, layout::MicrofacetBsdf_alpha_y, param.alpha_y);
  store_float(index, layout::MicrofacetBsdf_ior, param.ior);
  store_float(index, layout::MicrofacetBsdf_energy_scale, param.energy_scale);
  store_word(index, layout::MicrofacetBsdf_fresnel_type, param.fresnel_type);
  store_float3(index, layout::MicrofacetBsdf_T, param.T);
}

void ClosurePool::set_generalized_schlick(
    Expr<std::uint32_t> index,
    const FresnelGeneralizedSchlick &fresnel) noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelGeneralizedSchlick_thin_film;
  store_float(tail, film + layout::FresnelThinFilm_thickness,
              fresnel.thin_film.thickness);
  store_float(tail, film + layout::FresnelThinFilm_ior, fresnel.thin_film.ior);
  store_float3(tail, layout::FresnelGeneralizedSchlick_reflection_tint,
               fresnel.reflection_tint);
  store_float3(tail, layout::FresnelGeneralizedSchlick_transmission_tint,
               fresnel.transmission_tint);
  store_float3(tail, layout::FresnelGeneralizedSchlick_f0, fresnel.f0);
  store_float3(tail, layout::FresnelGeneralizedSchlick_f90, fresnel.f90);
  store_float(tail, layout::FresnelGeneralizedSchlick_exponent,
              fresnel.exponent);
}

void ClosurePool::set_fresnel_conductor(
    Expr<std::uint32_t> index, const FresnelConductor &fresnel) noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelConductor_thin_film;
  store_float(tail, film + layout::FresnelThinFilm_thickness,
              fresnel.thin_film.thickness);
  store_float(tail, film + layout::FresnelThinFilm_ior, fresnel.thin_film.ior);
  store_float3(tail, layout::FresnelConductor_ior + layout::complex_Spectrum_re,
               fresnel.ior);
  store_float3(tail, layout::FresnelConductor_ior + layout::complex_Spectrum_im,
               fresnel.extinction);
}

void ClosurePool::set_fresnel_f82_tint(Expr<std::uint32_t> index,
                                       const FresnelF82Tint &fresnel) noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelF82Tint_thin_film;
  store_float(tail, film + layout::FresnelThinFilm_thickness,
              fresnel.thin_film.thickness);
  store_float(tail, film + layout::FresnelThinFilm_ior, fresnel.thin_film.ior);
  store_float3(tail, layout::FresnelF82Tint_f0, fresnel.f0);
  store_float3(tail, layout::FresnelF82Tint_b, fresnel.b);
}

void ClosurePool::set_left(Expr<std::uint32_t> left) noexcept { _left = left; }

ShaderClosureCommon
ClosurePool::common(Expr<std::uint32_t> index) const noexcept {
  return {.weight = load_float3(index, layout::ShaderClosure_weight),
          .type = load_word(index, layout::ShaderClosure_type),
          .sample_weight =
              load_float(index, layout::ShaderClosure_sample_weight),
          .N = load_float3(index, layout::ShaderClosure_N)};
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
  constexpr auto film = layout::FresnelGeneralizedSchlick_thin_film;
  return {
      .thin_film = {.thickness = load_float(
                        tail, film + layout::FresnelThinFilm_thickness),
                    .ior =
                        load_float(tail, film + layout::FresnelThinFilm_ior)},
      .reflection_tint =
          load_float3(tail, layout::FresnelGeneralizedSchlick_reflection_tint),
      .transmission_tint = load_float3(
          tail, layout::FresnelGeneralizedSchlick_transmission_tint),
      .f0 = load_float3(tail, layout::FresnelGeneralizedSchlick_f0),
      .f90 = load_float3(tail, layout::FresnelGeneralizedSchlick_f90),
      .exponent = load_float(tail, layout::FresnelGeneralizedSchlick_exponent)};
}

MicrofacetConductorClosure
ClosurePool::microfacet_conductor(Expr<std::uint32_t> index) const noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelConductor_thin_film;
  return {
      .common = common(index),
      .param = microfacet_param(index),
      .conductor = {
          .thin_film = {.thickness = load_float(
                            tail, film + layout::FresnelThinFilm_thickness),
                        .ior = load_float(tail,
                                          film + layout::FresnelThinFilm_ior)},
          .ior = load_float3(tail, layout::FresnelConductor_ior +
                                       layout::complex_Spectrum_re),
          .extinction = load_float3(tail, layout::FresnelConductor_ior +
                                              layout::complex_Spectrum_im)}};
}

MicrofacetF82TintClosure
ClosurePool::microfacet_f82_tint(Expr<std::uint32_t> index) const noexcept {
  const auto tail = load_word(index, layout::MicrofacetBsdf_fresnel);
  constexpr auto film = layout::FresnelF82Tint_thin_film;
  return {.common = common(index),
          .param = microfacet_param(index),
          .f82_tint = {
              .thin_film = {.thickness = load_float(
                                tail, film + layout::FresnelThinFilm_thickness),
                            .ior = load_float(
                                tail, film + layout::FresnelThinFilm_ior)},
              .f0 = load_float3(tail, layout::FresnelF82Tint_f0),
              .b = load_float3(tail, layout::FresnelF82Tint_b)}};
}

MicrofacetParam
ClosurePool::microfacet_param(Expr<std::uint32_t> index) const noexcept {
  return {.alpha_x = load_float(index, layout::MicrofacetBsdf_alpha_x),
          .alpha_y = load_float(index, layout::MicrofacetBsdf_alpha_y),
          .ior = load_float(index, layout::MicrofacetBsdf_ior),
          .energy_scale =
              load_float(index, layout::MicrofacetBsdf_energy_scale),
          .fresnel_type = load_word(index, layout::MicrofacetBsdf_fresnel_type),
          .T = load_float3(index, layout::MicrofacetBsdf_T)};
}

namespace detail {

void node_closure_set_weight(Cursor &cursor, Float3 &closure_weight) noexcept {
  const auto x = cursor.floating();
  const auto y = cursor.floating();
  const auto z = cursor.floating();
  closure_weight = make_float3(x, y, z);
}

void node_closure_weight(Cursor &cursor, Stack &stack,
                         Float3 &closure_weight) noexcept {
  const auto packed = cursor.word();
  closure_weight = stack_load_float3(stack, cursor.byte(packed, 0u));
}

void node_emission_weight(Cursor &cursor, Stack &stack,
                          Float3 &closure_weight) noexcept {
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto strength = cursor.word();
  closure_weight = stack_load_input_float3(stack, color_x, color_y, color_z) *
                   stack_load_input_float(stack, strength);
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
  $if(weight1_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, weight1_offset, input_weight * (1.0f - weight));
  };
  $if(weight2_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, weight2_offset, input_weight * weight);
  };
}

void node_closure_emission(const KernelGlobals &kernel_globals,
                           Cursor &cursor, Stack &stack,
                           Expr<luisa::float3> closure_weight,
                           ShaderData &shader_data,
                           Bool &supported) noexcept {
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  Float3 weight = closure_weight;
  Bool active = true;

  $if(mix_weight_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const auto mix_weight = stack_load_float(stack, mix_weight_offset);
    $if(mix_weight == 0.0f) { active = false; }
    $else { weight *= mix_weight; };
  };

  $if(active) {
    $if((shader_data.flag & shader_data_is_volume_shader_eval) != 0u) {
      if (const auto density =
              kernel_globals.object_volume_density(shader_data.object)) {
        weight *= *density;
      } else {
        $if(shader_data.object != object_none) {
          active = false;
          supported = false;
        };
      }
    };
    $if(active) {
      $if((shader_data.flag & shader_data_emission) != 0u) {
        shader_data.closure_emission_background += weight;
      }
      $else {
        shader_data.flag |= shader_data_emission;
        shader_data.closure_emission_background = weight;
      };
    };
  };
}

void node_closure_background(Cursor &cursor, Stack &stack,
                             Expr<luisa::float3> closure_weight,
                             ShaderData &shader_data) noexcept {
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  Float3 weight = closure_weight;
  Bool active = true;

  $if(mix_weight_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const auto mix_weight = stack_load_float(stack, mix_weight_offset);
    $if(mix_weight == 0.0f) { active = false; }
    $else { weight *= mix_weight; };
  };

  $if(active) {
    $if((shader_data.flag & shader_data_emission) != 0u) {
      shader_data.closure_emission_background += weight;
    }
    $else {
      shader_data.flag |= shader_data_emission;
      shader_data.closure_emission_background = weight;
    };
  };
}

void node_closure_bsdf(const KernelGlobals &kernel_globals, Cursor &cursor,
                       Stack &stack, Expr<luisa::float3> closure_weight,
                       ShaderType shader_type, std::uint32_t node_feature_mask,
                       ShaderData &shader_data, const PathState &path_state,
                       Bool &supported) noexcept {
  const auto closure_type = cursor.word();
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  const auto mix_weight =
      stack_load_float_default(stack, mix_weight_offset, 1.0f);

  if (shader_type != SHADER_TYPE_SURFACE) {
    node_closure_bsdf_skip(cursor, closure_type);
    return;
  }

  if ((node_feature_mask & kernel_feature_node_bsdf) != 0u) {
    $if(mix_weight == 0.0f) { node_closure_bsdf_skip(cursor, closure_type); }
    $else {
      // TinyStorage has no physical pool, but failed allocation does not
      // suppress transparent, portal, or Principled shader-state effects.
      {
        $if(closure_type ==
            static_cast<std::uint32_t>(CLOSURE_BSDF_PRINCIPLED_ID)) {
          node_principled_bsdf(kernel_globals, cursor, stack, mix_weight, true,
                               shader_data, path_state, supported);
        }
        $else {
          const Bool is_bssrdf =
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSSRDF_BURLEY_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_ID)) |
              (closure_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID));
          const Bool is_sheen =
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_ASHIKHMIN_VELVET_ID));
          const Bool is_toon =
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_DIFFUSE_TOON_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_GLOSSY_TOON_ID));
          const Bool is_ray_portal =
              closure_type ==
              static_cast<std::uint32_t>(CLOSURE_BSDF_RAY_PORTAL_ID);
          const Bool is_hair =
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_REFLECTION_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_TRANSMISSION_ID));
          const Bool is_principled_hair =
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_CHIANG_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_HUANG_ID));
          const Bool is_glass =
              (closure_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID)) |
              (closure_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)) |
              (closure_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID));
          const Bool is_glossy =
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID)) |
              (closure_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_BECKMANN_ID)) |
              (closure_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID));
          const Bool is_refraction =
              (closure_type == static_cast<std::uint32_t>(
                                   CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID)) |
              (closure_type ==
               static_cast<std::uint32_t>(
                   CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID));
          const Bool is_metallic =
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_PHYSICAL_CONDUCTOR)) |
              (closure_type ==
               static_cast<std::uint32_t>(CLOSURE_BSDF_F82_CONDUCTOR));
          $if(is_bssrdf) {
            node_bssrdf(cursor, stack, closure_type, closure_weight, mix_weight,
                        shader_data, path_state);
          }
          $elif(is_sheen) {
            node_sheen(kernel_globals, cursor, stack, closure_type,
                       closure_weight, mix_weight, shader_data);
          }
          $elif(is_toon) {
            node_toon(kernel_globals, cursor, stack, closure_type,
                      closure_weight, mix_weight, shader_data, path_state);
          }
          $elif(is_ray_portal) {
            node_ray_portal(cursor, stack, closure_weight, mix_weight,
                            shader_data);
          }
          $elif(is_hair) {
            node_hair(cursor, stack, closure_type, closure_weight, mix_weight,
                      shader_data);
          }
          $elif(is_principled_hair) {
            node_principled_hair(kernel_globals, cursor, stack, closure_type,
                                 closure_weight, mix_weight, shader_data,
                                 path_state);
          }
          $elif(is_glass) {
            const auto color_x = cursor.word();
            const auto color_y = cursor.word();
            const auto color_z = cursor.word();
            const auto roughness_input = cursor.word();
            const auto ior_input = cursor.word();
            const auto thin_film_thickness_input = cursor.word();
            const auto thin_film_ior_input = cursor.word();
            const auto normal_packed = cursor.word();
            const auto normal_offset = cursor.byte(normal_packed, 0u);
            auto normal =
                stack_load_float3_default(stack, normal_offset, shader_data.N);
            normal = native_vector_math::safe_normalize_nonzero_or(
                normal, shader_data.N);
            detail::glass_setup(
                kernel_globals, shader_data, path_state, closure_type,
                mix_weight, normal,
                stack_load_input_float3(stack, color_x, color_y, color_z),
                stack_load_input_float(stack, roughness_input),
                stack_load_input_float(stack, ior_input),
                stack_load_input_float(stack, thin_film_thickness_input),
                stack_load_input_float(stack, thin_film_ior_input));
          }
          $elif(is_glossy) {
            const auto color_x = cursor.word();
            const auto color_y = cursor.word();
            const auto color_z = cursor.word();
            const auto roughness_input = cursor.word();
            const auto anisotropy_input = cursor.word();
            const auto rotation_input = cursor.word();
            const auto normal_tangent_packed = cursor.word();
            const auto normal_offset = cursor.byte(normal_tangent_packed, 0u);
            const auto tangent_offset = cursor.byte(normal_tangent_packed, 1u);
            auto normal =
                stack_load_float3_default(stack, normal_offset, shader_data.N);
            normal = native_vector_math::safe_normalize_nonzero_or(
                normal, shader_data.N);
            detail::glossy_setup(
                kernel_globals, shader_data, path_state, closure_type,
                mix_weight, closure_weight, normal,
                stack_load_input_float3(stack, color_x, color_y, color_z),
                stack_load_input_float(stack, roughness_input),
                stack_load_input_float(stack, anisotropy_input),
                stack_load_input_float(stack, rotation_input),
                stack_load_float3_default(stack, tangent_offset,
                                          make_float3(0.0f)),
                tangent_offset !=
                    static_cast<std::uint32_t>(SVM_STACK_INVALID));
          }
          $elif(is_refraction) {
            const auto roughness_input = cursor.word();
            const auto ior_input = cursor.word();
            const auto normal_packed = cursor.word();
            const auto normal_offset = cursor.byte(normal_packed, 0u);
            auto normal =
                stack_load_float3_default(stack, normal_offset, shader_data.N);
            normal = native_vector_math::safe_normalize_nonzero_or(
                normal, shader_data.N);
            detail::refraction_setup(
                kernel_globals, shader_data, path_state, closure_type,
                mix_weight, closure_weight, normal,
                stack_load_input_float(stack, roughness_input),
                stack_load_input_float(stack, ior_input));
          }
          $elif(is_metallic) {
            const auto distribution = cursor.word();
            const auto base_ior_x = cursor.word();
            const auto base_ior_y = cursor.word();
            const auto base_ior_z = cursor.word();
            const auto edge_tint_k_x = cursor.word();
            const auto edge_tint_k_y = cursor.word();
            const auto edge_tint_k_z = cursor.word();
            const auto roughness_input = cursor.word();
            const auto anisotropy_input = cursor.word();
            const auto rotation_input = cursor.word();
            const auto thin_film_thickness_input = cursor.word();
            const auto thin_film_ior_input = cursor.word();
            const auto normal_tangent_packed = cursor.word();
            const auto normal_offset = cursor.byte(normal_tangent_packed, 0u);
            const auto tangent_offset = cursor.byte(normal_tangent_packed, 1u);
            auto normal =
                stack_load_float3_default(stack, normal_offset, shader_data.N);
            normal = native_vector_math::safe_normalize_nonzero_or(
                normal, shader_data.N);
            detail::metallic_setup(
                kernel_globals, shader_data, path_state, closure_type,
                distribution, mix_weight, normal,
                stack_load_input_float3(stack, base_ior_x, base_ior_y,
                                        base_ior_z),
                stack_load_input_float3(stack, edge_tint_k_x, edge_tint_k_y,
                                        edge_tint_k_z),
                stack_load_input_float(stack, roughness_input),
                stack_load_input_float(stack, anisotropy_input),
                stack_load_input_float(stack, rotation_input),
                stack_load_input_float(stack, thin_film_thickness_input),
                stack_load_input_float(stack, thin_film_ior_input),
                stack_load_float3_default(stack, tangent_offset,
                                          make_float3(0.0f)),
                tangent_offset !=
                    static_cast<std::uint32_t>(SVM_STACK_INVALID));
          }
          $else {
            $switch(closure_type) {
              PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_ID) {
                const auto color_x = cursor.word();
                const auto color_y = cursor.word();
                const auto color_z = cursor.word();
                const auto roughness_input = cursor.word();
                const auto normal_packed = cursor.word();
                const auto normal_offset = cursor.byte(normal_packed, 0u);
                auto normal = stack_load_float3_default(stack, normal_offset,
                                                        shader_data.N);
                normal = native_vector_math::safe_normalize_nonzero_or(
                    normal, shader_data.N);

                const auto weight = closure_weight * mix_weight;
                const auto roughness =
                    stack_load_input_float(stack, roughness_input);
                $if(roughness < 1.0e-5f) {
                  diffuse_setup(shader_data, normal, weight);
                }
                $else {
                  const auto color = clamp(
                      stack_load_input_float3(stack, color_x, color_y, color_z),
                      0.0f, 1.0f);
                  oren_nayar_setup(shader_data, normal, weight, roughness,
                                   color);
                };
              };
              PSYCLES_SVM_CASE(CLOSURE_BSDF_TRANSLUCENT_ID) {
                const auto unused_param = cursor.word();
                const auto normal_packed = cursor.word();
                static_cast<void>(unused_param);
                const auto normal_offset = cursor.byte(normal_packed, 0u);
                auto normal = stack_load_float3_default(stack, normal_offset,
                                                        shader_data.N);
                normal = native_vector_math::safe_normalize_nonzero_or(
                    normal, shader_data.N);
                const auto weight = closure_weight * mix_weight;
                translucent_setup(
                    shader_data,
                    detail::maybe_ensure_valid_specular_reflection(shader_data,
                                                                   normal),
                    weight);
              };
              PSYCLES_SVM_CASE(CLOSURE_BSDF_TRANSPARENT_ID) {
                const auto unused_param = cursor.word();
                const auto unused_normal = cursor.word();
                static_cast<void>(unused_param);
                static_cast<void>(unused_normal);
                transparent_setup(shader_data, path_state,
                                  closure_weight * mix_weight);
              };
              $default {
                node_closure_bsdf_skip(cursor, closure_type);
                supported = false;
              };
            };
          };
        };
      }
    };
    return;
  }

  if ((node_feature_mask & kernel_feature_node_emission) != 0u) {
    $if((mix_weight == 0.0f) |
        (closure_type !=
         static_cast<std::uint32_t>(CLOSURE_BSDF_PRINCIPLED_ID))) {
      node_closure_bsdf_skip(cursor, closure_type);
    }
    $else {
      // Cycles evaluates Principled emission with num_closure_left == 0.
      // Sheen/coat still use local BSDF values for layer attenuation, but
      // neither their evaluation nor transparency requires a closure array.
      node_principled_bsdf(kernel_globals, cursor, stack, mix_weight, false,
                           shader_data, path_state, supported);
    };
    return;
  }

  node_closure_bsdf_skip(cursor, closure_type);
}

void node_closure_bsdf_skip(Cursor &cursor,
                            Expr<std::uint32_t> closure_type) noexcept {
  UInt words = static_cast<std::uint32_t>(sizeof(SVMNodeSimpleBsdfData) /
                                          sizeof(std::uint32_t));
  $switch(closure_type) {
    PSYCLES_SVM_CASE(CLOSURE_BSDF_PRINCIPLED_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodePrincipledBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_CHIANG_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodePrincipledHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_HUANG_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodePrincipledHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_PHYSICAL_CONDUCTOR) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeMetallicBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_F82_CONDUCTOR) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeMetallicBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeDiffuseBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_OREN_NAYAR_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeDiffuseBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_BURLEY_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeDiffuseBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_RAY_PORTAL_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeRayPortalBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlossyBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeRefractionBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeRefractionBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlassBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlassBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeGlassBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_GLOSSY_TOON_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeToonBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_TOON_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeToonBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_REFLECTION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_TRANSMISSION_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeHairBsdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_BURLEY_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeBssrdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_RANDOM_WALK_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeBssrdfData) /
                                         sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID) {
      words = static_cast<std::uint32_t>(sizeof(SVMNodeBssrdfData) /
                                         sizeof(std::uint32_t));
    };
    $default{};
  };
  cursor.advance(words);
}

} // namespace detail

} // namespace psycles::luisa_backend::cycles_svm

#undef PSYCLES_SVM_CASE
