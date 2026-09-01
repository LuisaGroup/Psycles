/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_principled.h"

#include "cycles_svm_microfacet.h"
#include "cycles_svm_simple_closure.h"

#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

inline constexpr float two_pi = 6.28318530717958647692f;

/* Named word indices are the formal projection of
 * SVMNodePrincipledBsdfData. They deliberately replace positional parsing
 * with the same typed field boundary used by Cycles' svm_node_get<T>(). */
class PrincipledDataView final {
private:
  Cursor &_cursor;

  static constexpr std::uint32_t distribution_word = 0u;
  static constexpr std::uint32_t ior_word = 1u;
  static constexpr std::uint32_t roughness_word = 2u;
  static constexpr std::uint32_t sheen_weight_word = 3u;
  static constexpr std::uint32_t coat_weight_word = 4u;
  static constexpr std::uint32_t metallic_word = 5u;
  static constexpr std::uint32_t transmission_weight_word = 6u;
  static constexpr std::uint32_t subsurface_weight_word = 7u;
  static constexpr std::uint32_t base_color_word = 8u;
  static constexpr std::uint32_t alpha_word = 11u;
  static constexpr std::uint32_t diffuse_roughness_word = 12u;
  static constexpr std::uint32_t normal_offsets_word = 13u;
  static constexpr std::uint32_t specular_tint_word = 14u;
  static constexpr std::uint32_t specular_ior_level_word = 17u;
  static constexpr std::uint32_t anisotropic_word = 18u;
  static constexpr std::uint32_t anisotropic_rotation_word = 19u;
  static constexpr std::uint32_t emission_color_word = 20u;
  static constexpr std::uint32_t emission_strength_word = 23u;
  static constexpr std::uint32_t sheen_tint_word = 24u;
  static constexpr std::uint32_t sheen_roughness_word = 27u;
  static constexpr std::uint32_t coat_tint_word = 28u;
  static constexpr std::uint32_t coat_roughness_word = 31u;
  static constexpr std::uint32_t coat_ior_word = 32u;
  static constexpr std::uint32_t subsurface_method_word = 33u;
  static constexpr std::uint32_t subsurface_radius_word = 34u;
  static constexpr std::uint32_t subsurface_scale_word = 37u;
  static constexpr std::uint32_t subsurface_ior_word = 38u;
  static constexpr std::uint32_t subsurface_anisotropy_word = 39u;
  static constexpr std::uint32_t thin_film_thickness_word = 40u;
  static constexpr std::uint32_t thin_film_ior_word = 41u;
  static constexpr std::uint32_t thin_wall_value_word = 42u;
  static constexpr std::uint32_t thin_wall_offset_word = 43u;

  [[nodiscard]] Float input_float(Stack &stack,
                                  std::uint32_t word) const noexcept {
    return stack_load_input_float(stack, _cursor.word_at(word));
  }

  [[nodiscard]] Float3 input_float3(Stack &stack,
                                    std::uint32_t word) const noexcept {
    return stack_load_input_float3(stack, _cursor.word_at(word),
                                   _cursor.word_at(word + 1u),
                                   _cursor.word_at(word + 2u));
  }

public:
  struct NormalOffsets {
    UInt normal;
    UInt tangent;
    UInt coat_normal;
  };

  explicit PrincipledDataView(Cursor &cursor) noexcept : _cursor{cursor} {}

  [[nodiscard]] UInt distribution() const noexcept {
    return _cursor.word_at(distribution_word);
  }
  [[nodiscard]] Float ior(Stack &stack) const noexcept {
    return input_float(stack, ior_word);
  }
  [[nodiscard]] Float roughness(Stack &stack) const noexcept {
    return input_float(stack, roughness_word);
  }
  [[nodiscard]] Float sheen_weight(Stack &stack) const noexcept {
    return input_float(stack, sheen_weight_word);
  }
  [[nodiscard]] Float coat_weight(Stack &stack) const noexcept {
    return input_float(stack, coat_weight_word);
  }
  [[nodiscard]] Float metallic(Stack &stack) const noexcept {
    return input_float(stack, metallic_word);
  }
  [[nodiscard]] Float transmission_weight(Stack &stack) const noexcept {
    return input_float(stack, transmission_weight_word);
  }
  [[nodiscard]] Float subsurface_weight(Stack &stack) const noexcept {
    return input_float(stack, subsurface_weight_word);
  }
  [[nodiscard]] Float3 base_color(Stack &stack) const noexcept {
    return input_float3(stack, base_color_word);
  }
  [[nodiscard]] Float alpha(Stack &stack) const noexcept {
    return input_float(stack, alpha_word);
  }
  [[nodiscard]] Float diffuse_roughness(Stack &stack) const noexcept {
    return input_float(stack, diffuse_roughness_word);
  }
  [[nodiscard]] NormalOffsets normal_offsets() const noexcept {
    const auto packed = _cursor.word_at(normal_offsets_word);
    return {.normal = _cursor.byte(packed, 0u),
            .tangent = _cursor.byte(packed, 1u),
            .coat_normal = _cursor.byte(packed, 2u)};
  }
  [[nodiscard]] Float3 specular_tint(Stack &stack) const noexcept {
    return input_float3(stack, specular_tint_word);
  }
  [[nodiscard]] Float specular_ior_level(Stack &stack) const noexcept {
    return input_float(stack, specular_ior_level_word);
  }
  [[nodiscard]] Float anisotropic(Stack &stack) const noexcept {
    return input_float(stack, anisotropic_word);
  }
  [[nodiscard]] Float anisotropic_rotation(Stack &stack) const noexcept {
    return input_float(stack, anisotropic_rotation_word);
  }
  [[nodiscard]] Float3 emission_color(Stack &stack) const noexcept {
    return input_float3(stack, emission_color_word);
  }
  [[nodiscard]] Float emission_strength(Stack &stack) const noexcept {
    return input_float(stack, emission_strength_word);
  }
  [[nodiscard]] Float3 sheen_tint(Stack &stack) const noexcept {
    return input_float3(stack, sheen_tint_word);
  }
  [[nodiscard]] Float sheen_roughness(Stack &stack) const noexcept {
    return input_float(stack, sheen_roughness_word);
  }
  [[nodiscard]] Float3 coat_tint(Stack &stack) const noexcept {
    return input_float3(stack, coat_tint_word);
  }
  [[nodiscard]] Float coat_roughness(Stack &stack) const noexcept {
    return input_float(stack, coat_roughness_word);
  }
  [[nodiscard]] Float coat_ior(Stack &stack) const noexcept {
    return input_float(stack, coat_ior_word);
  }
  [[nodiscard]] UInt subsurface_method() const noexcept {
    return _cursor.word_at(subsurface_method_word);
  }
  [[nodiscard]] Float3 subsurface_radius(Stack &stack) const noexcept {
    return input_float3(stack, subsurface_radius_word);
  }
  [[nodiscard]] Float subsurface_scale(Stack &stack) const noexcept {
    return input_float(stack, subsurface_scale_word);
  }
  [[nodiscard]] Float subsurface_ior(Stack &stack) const noexcept {
    return input_float(stack, subsurface_ior_word);
  }
  [[nodiscard]] Float subsurface_anisotropy(Stack &stack) const noexcept {
    return input_float(stack, subsurface_anisotropy_word);
  }
  [[nodiscard]] Float thin_film_thickness(Stack &stack) const noexcept {
    return input_float(stack, thin_film_thickness_word);
  }
  [[nodiscard]] Float thin_film_ior(Stack &stack) const noexcept {
    return input_float(stack, thin_film_ior_word);
  }
  [[nodiscard]] Int thin_wall(Stack &stack) const noexcept {
    const auto offset =
        _cursor.byte(_cursor.word_at(thin_wall_offset_word), 0u);
    Int value = _cursor.word_at(thin_wall_value_word).bitcast<std::int32_t>();
    $if(offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      value = stack_load_int(stack, offset);
    };
    return value;
  }

  void advance() noexcept {
    _cursor.advance(static_cast<std::uint32_t>(
        sizeof(SVMNodePrincipledBsdfData) / sizeof(std::uint32_t)));
  }
};

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] Float f0_from_ior(Expr<float> ior) noexcept {
  return square((ior - 1.0f) / (ior + 1.0f));
}

[[nodiscard]] Float ior_from_f0(Expr<float> f0) noexcept {
  const auto root = sqrt(clamp(f0, 0.0f, 0.99f));
  return (1.0f + root) / (1.0f - root);
}

[[nodiscard]] Float safe_divide(Expr<float> numerator,
                                Expr<float> denominator) noexcept {
  return select(0.0f, numerator / denominator, denominator != 0.0f);
}

[[nodiscard]] Float3
closure_layering_weight(Expr<luisa::float3> layer_albedo,
                        Expr<luisa::float3> weight) noexcept {
  const auto relative_albedo =
      make_float3(safe_divide(layer_albedo.x, weight.x),
                  safe_divide(layer_albedo.y, weight.y),
                  safe_divide(layer_albedo.z, weight.z));
  const auto yz_maximum = select(relative_albedo.z, relative_albedo.y,
                                 relative_albedo.y > relative_albedo.z);
  const auto maximum =
      select(yz_maximum, relative_albedo.x, relative_albedo.x > yz_maximum);
  return weight * clamp(1.0f - maximum, 0.0f, 1.0f);
}

void emission_setup(ShaderData &shader_data,
                    Expr<luisa::float3> weight) noexcept {
  $if((shader_data.flag & shader_data_emission) != 0u) {
    shader_data.closure_emission_background += weight;
  }
  $else {
    shader_data.flag |= shader_data_emission;
    shader_data.closure_emission_background = weight;
  };
}

} // namespace

void node_principled_bsdf(const KernelGlobals &kernel_globals, Cursor &cursor,
                          Stack &stack, Expr<float> mix_weight,
                          bool evaluate_bsdf, ShaderData &shader_data,
                          const PathState &path_state,
                          Bool &supported) noexcept {
  PrincipledDataView data{cursor};

  const auto sheen_weight = max(data.sheen_weight(stack), 0.0f);
  const auto coat_weight = max(data.coat_weight(stack), 0.0f);
  Bool subset_supported = (sheen_weight <= CLOSURE_WEIGHT_CUTOFF) &
                          (coat_weight <= CLOSURE_WEIGHT_CUTOFF);
  Float metallic = 0.0f;
  Float transmission_weight = 0.0f;
  Float subsurface_weight = 0.0f;
  UInt distribution = 0u;
  if (evaluate_bsdf) {
    metallic = clamp(data.metallic(stack), 0.0f, 1.0f);
    transmission_weight = clamp(data.transmission_weight(stack), 0.0f, 1.0f);
    subsurface_weight = clamp(data.subsurface_weight(stack), 0.0f, 1.0f);
    distribution = data.distribution();
    const Bool valid_distribution =
        (distribution ==
         static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID)) |
        (distribution == static_cast<std::uint32_t>(
                             CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID));
    subset_supported &= (metallic <= CLOSURE_WEIGHT_CUTOFF) &
                        (transmission_weight <= CLOSURE_WEIGHT_CUTOFF) &
                        (subsurface_weight <= CLOSURE_WEIGHT_CUTOFF) &
                        valid_distribution;
  }

  $if(subset_supported) {
    const auto normal_offsets = data.normal_offsets();
    auto normal =
        stack_load_float3_default(stack, normal_offsets.normal, shader_data.N);
    normal =
        native_vector_math::safe_normalize_nonzero_or(normal, shader_data.N);

    /* Before any actual shader components, apply transparency. */
    Float3 weight = make_float3(mix_weight);
    const auto alpha = clamp(data.alpha(stack), 0.0f, 1.0f);
    $if(alpha < 1.0f) {
      transparent_setup(shader_data, path_state, weight * (1.0f - alpha));
      weight *= alpha;
    };

    /* Sheen and coat are the first two layers in Cycles. Their nonzero
     * transitions are intentionally outside this proved subset for now. */

    const auto emission =
        data.emission_color(stack) * data.emission_strength(stack);
    $if(any(emission != make_float3(0.0f))) {
      emission_setup(shader_data, emission * weight);
    };

    if (evaluate_bsdf) {
      const auto base_color = max(data.base_color(stack), make_float3(0.0f));
      const auto ior = max(data.ior(stack), 1.0e-5f);
      const auto roughness = clamp(data.roughness(stack), 0.0f, 1.0f);
      const auto valid_reflection_normal =
          maybe_ensure_valid_specular_reflection(shader_data, normal);
      const auto anisotropic = clamp(data.anisotropic(stack), 0.0f, 1.0f);
      const auto specular_tint =
          max(data.specular_tint(stack), make_float3(0.0f));
      const auto thin_film_thickness = data.thin_film_thickness(stack);
      const auto thin_film_ior =
          select(0.0f, max(data.thin_film_ior(stack), 1.0e-5f),
                 thin_film_thickness > THINFILM_THICKNESS_CUTOFF);

      Float alpha_x = square(roughness);
      Float alpha_y = alpha_x;
      Float3 tangent = make_float3(0.0f);
      const auto tangent_offset = normal_offsets.tangent;
      $if((anisotropic > 0.0f) &
          (tangent_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID))) {
        tangent = stack_load_float3(stack, tangent_offset);
        const auto aspect = sqrt(1.0f - anisotropic * 0.9f);
        alpha_x /= aspect;
        alpha_y *= aspect;
        const auto rotation = data.anisotropic_rotation(stack);
        $if(rotation != 0.0f) {
          tangent = rotate_around_axis(tangent, normal, rotation * two_pi);
        };
      };

      /* Metallic and transmission are zero throughout this exact subset, so
       * Cycles leaves weight unchanged before its dielectric layer. */
      const auto specular_ior_level = max(data.specular_ior_level(stack), 0.0f);
      Float eta = ior;
      Float f0 = f0_from_ior(eta);
      $if(specular_ior_level != 0.5f) {
        f0 *= 2.0f * specular_ior_level;
        eta = ior_from_f0(f0);
        $if(ior < 1.0f) { eta = 1.0f / eta; };
      };

      const auto diffuse_visibility =
          (path_state.visibility & path_ray_visibility_diffuse) != 0u;
      const Bool reflective_caustics =
          kernel_globals.caustics_reflective() | !diffuse_visibility;
      $if(reflective_caustics &
          ((eta != 1.0f) | (thin_film_thickness > THINFILM_THICKNESS_CUTOFF))) {
        const auto layer_albedo = principled_specular_setup(
            kernel_globals, shader_data, weight, valid_reflection_normal,
            tangent, alpha_x, alpha_y, eta, f0, specular_tint,
            thin_film_thickness, thin_film_ior,
            distribution == static_cast<std::uint32_t>(
                                CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID));
        weight = closure_layering_weight(layer_albedo, weight);
      };

      const auto diffuse_roughness =
          clamp(data.diffuse_roughness(stack), 0.0f, 1.0f);
      const auto diffuse_weight =
          base_color * (1.0f - subsurface_weight) * weight;
      $if(diffuse_roughness < 1.0e-5f) {
        diffuse_setup(shader_data, normal, diffuse_weight);
      }
      $else {
        oren_nayar_setup(shader_data, normal, diffuse_weight, diffuse_roughness,
                         base_color);
      };
    }
  }
  $else { supported = false; };

  data.advance();
}

} // namespace psycles::luisa_backend::cycles_svm::detail
