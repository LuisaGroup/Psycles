/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_closure_set_weight(Cursor &cursor,
                             Float3 &closure_weight) noexcept {
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
  closure_weight =
      stack_load_input_float3(stack, color_x, color_y, color_z) *
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
  $if (weight1_offset !=
       static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, weight1_offset, input_weight * (1.0f - weight));
  };
  $if (weight2_offset !=
       static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, weight2_offset, input_weight * weight);
  };
}

void node_closure_emission(Cursor &cursor, Stack &stack,
                           Expr<luisa::float3> closure_weight,
                           ShaderData &shader_data,
                           Bool &supported) noexcept {
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  Float3 weight = closure_weight;
  Bool active = true;

  $if (mix_weight_offset !=
       static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const auto mix_weight = stack_load_float(stack, mix_weight_offset);
    $if (mix_weight == 0.0f) {
      active = false;
    } $else {
      weight *= mix_weight;
    };
  };

  $if (active) {
    $if ((shader_data.flag & shader_data_is_volume_shader_eval) != 0u) {
      /* Cycles multiplies by object_volume_density here. The object service
       * is deliberately not guessed before that exact family is copied. */
      supported = false;
    } $else {
      $if ((shader_data.flag & shader_data_emission) != 0u) {
        shader_data.closure_emission_background += weight;
      } $else {
        shader_data.flag |= shader_data_emission;
        shader_data.closure_emission_background = weight;
      };
    };
  };
}

void node_closure_bsdf_skip(
    Cursor &cursor, Expr<std::uint32_t> closure_type) noexcept {
  UInt words =
      static_cast<std::uint32_t>(sizeof(SVMNodeSimpleBsdfData) /
                                 sizeof(std::uint32_t));
  $switch (closure_type) {
    PSYCLES_SVM_CASE(CLOSURE_BSDF_PRINCIPLED_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodePrincipledBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_CHIANG_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodePrincipledHairBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_HUANG_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodePrincipledHairBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_PHYSICAL_CONDUCTOR) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeMetallicBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_F82_CONDUCTOR) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeMetallicBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeDiffuseBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_OREN_NAYAR_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeDiffuseBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_BURLEY_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeDiffuseBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_RAY_PORTAL_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeRayPortalBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlossyBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlossyBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlossyBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlossyBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeRefractionBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeRefractionBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlassBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlassBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeGlassBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_GLOSSY_TOON_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeToonBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_DIFFUSE_TOON_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeToonBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_REFLECTION_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeHairBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSDF_HAIR_TRANSMISSION_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeHairBsdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_BURLEY_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeBssrdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_RANDOM_WALK_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeBssrdfData) / sizeof(std::uint32_t));
    };
    PSYCLES_SVM_CASE(CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID) {
      words = static_cast<std::uint32_t>(
          sizeof(SVMNodeBssrdfData) / sizeof(std::uint32_t));
    };
    $default {};
  };
  cursor.advance(words);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE
