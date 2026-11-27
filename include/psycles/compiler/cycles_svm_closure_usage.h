#pragma once

#include <cstdint>

namespace psycles::compiler::cycles_svm {

// Static closure and layer reachability facts for one ShaderJump entry.
// Unknown/imported word images retain all bits set so standalone device tests
// keep complete switch coverage; compiler-emitted entries start from none()
// and record only emitted nodes, closures, Fresnel tags, and Principled layers.
struct ClosureUsage {
  static constexpr std::uint32_t fresnel_none = 1u << 0u;
  static constexpr std::uint32_t fresnel_dielectric = 1u << 1u;
  static constexpr std::uint32_t fresnel_generalized_schlick = 1u << 2u;
  static constexpr std::uint32_t fresnel_conductor = 1u << 3u;
  static constexpr std::uint32_t fresnel_f82_tint = 1u << 4u;
  static constexpr std::uint32_t fresnel_positive_exponent = 1u << 8u;
  static constexpr std::uint32_t fresnel_anisotropic = 1u << 9u;
  static constexpr std::uint32_t fresnel_custom_generalized_schlick = 1u << 10u;
  static constexpr std::uint32_t fresnel_all = ~0u;

  static constexpr std::uint32_t principled_alpha = 1u << 0u;
  static constexpr std::uint32_t principled_sheen = 1u << 1u;
  static constexpr std::uint32_t principled_coat = 1u << 2u;
  static constexpr std::uint32_t principled_emission = 1u << 3u;
  static constexpr std::uint32_t principled_anisotropic = 1u << 4u;
  static constexpr std::uint32_t principled_metallic = 1u << 5u;
  static constexpr std::uint32_t principled_transmission = 1u << 6u;
  static constexpr std::uint32_t principled_thin_wall = 1u << 7u;
  static constexpr std::uint32_t principled_thin_film = 1u << 8u;
  static constexpr std::uint32_t principled_oren_nayar = 1u << 9u;
  static constexpr std::uint32_t principled_single_ggx = 1u << 10u;
  static constexpr std::uint32_t principled_multi_ggx = 1u << 11u;
  static constexpr std::uint32_t principled_coat_tint = 1u << 12u;
  static constexpr std::uint32_t principled_coat_normal = 1u << 13u;
  static constexpr std::uint32_t principled_nondefault_specular_ior = 1u << 14u;
  static constexpr std::uint32_t principled_specular_tint = 1u << 15u;
  static constexpr std::uint32_t principled_subsurface = 1u << 16u;
  static constexpr std::uint32_t principled_generalized_specular = 1u << 17u;
  static constexpr std::uint32_t principled_all = (1u << 18u) - 1u;

  static constexpr std::uint32_t diffuse_lambertian = 1u << 0u;
  static constexpr std::uint32_t diffuse_oren_nayar = 1u << 1u;
  static constexpr std::uint32_t diffuse_all = 3u;

  std::uint64_t svm_bsdf_node_mask{~0ull};
  std::uint64_t surface_closure_mask{~0ull};
  std::uint32_t microfacet_fresnel_mask{fresnel_all};
  std::uint32_t principled_mask{principled_all};
  std::uint32_t diffuse_node_mask{diffuse_all};

  [[nodiscard]] static constexpr std::uint64_t bit(std::uint32_t id) noexcept {
    return id < 64u ? (std::uint64_t{1u} << id) : 0ull;
  }

  [[nodiscard]] static constexpr ClosureUsage none() noexcept {
    return {.svm_bsdf_node_mask = 0ull,
            .surface_closure_mask = 0ull,
            .microfacet_fresnel_mask = 0u,
            .principled_mask = 0u,
            .diffuse_node_mask = 0u};
  }

  [[nodiscard]] constexpr bool has_bsdf_node(std::uint32_t id) const noexcept {
    return id >= 64u || (svm_bsdf_node_mask & (std::uint64_t{1u} << id)) != 0ull;
  }

  [[nodiscard]] constexpr bool has_fresnel(std::uint32_t flag) const noexcept {
    return (microfacet_fresnel_mask & flag) != 0u;
  }

  [[nodiscard]] constexpr bool has_principled(std::uint32_t flag) const noexcept {
    return (principled_mask & flag) != 0u;
  }

  constexpr void merge(const ClosureUsage &other) noexcept {
    svm_bsdf_node_mask |= other.svm_bsdf_node_mask;
    surface_closure_mask |= other.surface_closure_mask;
    microfacet_fresnel_mask |= other.microfacet_fresnel_mask;
    principled_mask |= other.principled_mask;
    diffuse_node_mask |= other.diffuse_node_mask;
  }

  friend constexpr bool operator==(const ClosureUsage &,
                                   const ClosureUsage &) = default;
};

} // namespace psycles::compiler::cycles_svm
