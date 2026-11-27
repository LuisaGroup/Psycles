/* SPDX-FileCopyrightText: 2011-2024 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_volume.h"
#include "cycles_svm_closure_layout.h"
#include "cycles_svm_microfacet.h"

#include <psycles/luisa/cycles_color_nodes.h>
#include <psycles/luisa/cycles_spectrum.h>
#include <psycles/luisa/cycles_volume_phase.h>

#include <cstddef>

namespace psycles::luisa_backend::cycles_svm {
using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace vl = detail::volume_closure_layout;
namespace cl = detail::closure_layout;

ShaderVolumeClosureCommon
ClosurePool::volume_common(Expr<std::uint32_t> index) const noexcept {
  return {.weight = load_float3(index, cl::ShaderClosure_weight),
          .type = load_word(index, cl::ShaderClosure_type),
          .sample_weight = load_float(index, cl::ShaderClosure_sample_weight)};
}

void ClosurePool::set_volume_henyey_greenstein(Expr<std::uint32_t> index,
                                               Expr<float> g) noexcept {
  store_float(index, vl::henyey_greenstein_g, g);
}
void ClosurePool::set_volume_draine(Expr<std::uint32_t> index, Expr<float> g,
                                    Expr<float> alpha) noexcept {
  store_float(index, vl::draine_g, g);
  store_float(index, vl::draine_alpha, alpha);
}
void ClosurePool::set_volume_fournier_forand(
    Expr<std::uint32_t> index, Expr<luisa::float3> coefficients) noexcept {
  store_float(index, vl::fournier_forand_c1, coefficients.x);
  store_float(index, vl::fournier_forand_c2, coefficients.y);
  store_float(index, vl::fournier_forand_c3, coefficients.z);
}
Float ClosurePool::volume_henyey_greenstein_g(
    Expr<std::uint32_t> index) const noexcept {
  return load_float(index, vl::henyey_greenstein_g);
}
Float2 ClosurePool::volume_draine_parameters(
    Expr<std::uint32_t> index) const noexcept {
  return make_float2(load_float(index, vl::draine_g),
                     load_float(index, vl::draine_alpha));
}
Float3 ClosurePool::volume_fournier_forand_coefficients(
    Expr<std::uint32_t> index) const noexcept {
  return make_float3(load_float(index, vl::fournier_forand_c1),
                     load_float(index, vl::fournier_forand_c2),
                     load_float(index, vl::fournier_forand_c3));
}

namespace detail {
namespace {
namespace phase = cycles_volume_phase;

Bool is_attribute_found(const AttributeDescriptor &descriptor) noexcept {
  return descriptor.offset != static_cast<int>(ATTR_STD_NOT_FOUND);
}

// Byte offsets are taken from the retained Cycles typed payloads. Reads are
// lazy, like svm_node_get<T>'s reference, so an untaken phase/attribute branch
// does not evaluate another phase's inputs. The caller advances the PC once.
class VolumeNodeView {
  Cursor &_cursor;
  Stack &_stack;

public:
  VolumeNodeView(Cursor &cursor, Stack &stack) noexcept
      : _cursor{cursor}, _stack{stack} {}
  Expr<std::uint32_t> word(std::size_t offset) const noexcept {
    return _cursor.word_at(
        static_cast<unsigned>(offset / sizeof(std::uint32_t)));
  }
  Float scalar(std::size_t offset) const noexcept {
    return stack_load_input_float(_stack, word(offset));
  }
  Float3 vector(std::size_t offset) const noexcept {
    return stack_load_input_float3(_stack, word(offset), word(offset + 4u),
                                   word(offset + 8u));
  }
  Float mix(std::size_t offset) const noexcept {
    return stack_load_float_default(
        _stack, _cursor.byte(word(offset), static_cast<std::uint32_t>(offset % 4u)), 1.0f);
  }
};

template <typename F>
void with_object_density(const KernelGlobals &kg, ShaderData &sd,
                         const EvaluationTransition &transition,
                         F &&evaluate) noexcept {
  if (const auto density = kg.object_volume_density(sd.object)) {
    evaluate(*density);
  } else {
    // A missing recording service is not permission to assume unit density
    // for an object. World evaluation needs no object table in Cycles.
    $if(sd.object == object_none) { evaluate(Float{1.0f}); }
    $else { transition.unsupported(); };
  }
}

[[nodiscard]] Float4 resolve_volume_spectral_reflectance(
    const ShaderData &sd, Expr<luisa::float3> rgb,
    const SpectralClosureWeight *spectral_weight) noexcept {
  const auto &sc = *sd.spectral;
  const Float4 fb = cycles_spectrum::rgb_to_spectrum_reflectance(
      *sc.tables, rgb, sc.wavelengths.lambdas);
  if (spectral_weight == nullptr) {
    return fb;
  }
  const Float4 illum = max(
      cycles_spectrum::sample_illuminant_4(*sc.tables, sc.wavelengths.lambdas),
      make_float4(0.1f));
  const Float4 slot_refl = max(spectral_weight->spd / illum, make_float4(0.0f));
  return select(fb, slot_refl, spectral_weight->valid != 0u);
}

void extinction_setup(ShaderData &sd, Expr<luisa::float3> weight,
                      const Float4 *spec_weight = nullptr) noexcept {
  const bool write_spec =
      spec_weight != nullptr && sd.spectral_extinction.has_value();
  $if((sd.flag & shader_data_extinction) != 0u) {
    sd.closure_transparent_extinction += weight;
    if (write_spec) {
      *sd.spectral_extinction += *spec_weight;
    }
  }
  $else {
    sd.flag |= shader_data_extinction;
    sd.closure_transparent_extinction = weight;
    if (write_spec) {
      *sd.spectral_extinction = *spec_weight;
    }
  };
}

void volume_emission_setup(ShaderData &sd,
                           Expr<luisa::float3> weight,
                           const Float4 *spec_weight = nullptr) noexcept {
  const bool write_spec =
      spec_weight != nullptr && sd.spectral_emission_background.has_value();
  $if((sd.flag & shader_data_emission) != 0u) {
    sd.closure_emission_background += weight;
    if (write_spec) {
      *sd.spectral_emission_background += *spec_weight;
    }
  }
  $else {
    sd.flag |= shader_data_emission;
    sd.closure_emission_background = weight;
    if (write_spec) {
      *sd.spectral_emission_background = *spec_weight;
    }
  };
}

template <typename G>
void henyey_greenstein_setup(ShaderData &sd, Expr<luisa::float3> weight,
                             G &&load_g,
                             const Float4 *spec_weight = nullptr) noexcept {
  if (sd.closure == nullptr) {
    return;
  }
  const auto allocation = bsdf_allocate(sd, weight);
  $if(allocation.valid) {
    sd.closure->set_type(
        allocation.index,
        static_cast<unsigned>(CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID));
    sd.closure->set_volume_henyey_greenstein(allocation.index,
                                             phase::clamp_anisotropy(load_g()));
    sd.flag |= shader_data_scatter;
    if (spec_weight != nullptr && sd.spectral_scatter.has_value()) {
      *sd.spectral_scatter += max(*spec_weight, make_float4(0.0f));
    }
  };
}

template <typename G, typename A>
void draine_setup(ShaderData &sd, Expr<luisa::float3> weight, G &&load_g,
                  A &&load_alpha,
                  const Float4 *spec_weight = nullptr) noexcept {
  if (sd.closure == nullptr) {
    return;
  }
  const auto allocation = bsdf_allocate(sd, weight);
  $if(allocation.valid) {
    sd.closure->set_type(allocation.index,
                         static_cast<unsigned>(CLOSURE_VOLUME_DRAINE_ID));
    sd.closure->set_volume_draine(
        allocation.index, phase::clamp_anisotropy(load_g()), load_alpha());
    sd.flag |= shader_data_scatter;
    if (spec_weight != nullptr && sd.spectral_scatter.has_value()) {
      *sd.spectral_scatter += max(*spec_weight, make_float4(0.0f));
    }
  };
}

Bool is_volume_scatter(Expr<std::uint32_t> type) noexcept {
  return (type >= static_cast<unsigned>(CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) &
         (type <= static_cast<unsigned>(CLOSURE_VOLUME_DRAINE_ID));
}

void allocate_volume_scatter(ShaderData &sd, Stack &stack,
                             Expr<luisa::float3> weight,
                             Expr<std::uint32_t> type,
                             Expr<std::uint32_t> param1,
                             Expr<std::uint32_t> param_extra,
                             const Float4 *spec_weight = nullptr) noexcept {
  if (sd.closure == nullptr) {
    return;
  }
  const bool write_spec =
      spec_weight != nullptr && sd.spectral_scatter.has_value();
  $switch(type) {
    $case(static_cast<unsigned>(CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) {
      henyey_greenstein_setup(
          sd, weight, [&] { return stack_load_input_float(stack, param1); },
          spec_weight);
    };
    $case(static_cast<unsigned>(CLOSURE_VOLUME_FOURNIER_FORAND_ID)) {
      const auto allocation = bsdf_allocate(sd, weight);
      $if(allocation.valid) {
        const auto ior = stack_load_input_float(stack, param1);
        const auto backscatter = stack_load_input_float(stack, param_extra);
        sd.closure->set_type(
            allocation.index,
            static_cast<unsigned>(CLOSURE_VOLUME_FOURNIER_FORAND_ID));
        sd.closure->set_volume_fournier_forand(
            allocation.index,
            phase::fournier_forand_coefficients(backscatter, ior));
        sd.flag |= shader_data_scatter;
        if (write_spec) {
          *sd.spectral_scatter += max(*spec_weight, make_float4(0.0f));
        }
      };
    };
    $case(static_cast<unsigned>(CLOSURE_VOLUME_RAYLEIGH_ID)) {
      const auto allocation = bsdf_allocate(sd, weight);
      $if(allocation.valid) {
        sd.closure->set_type(allocation.index,
                             static_cast<unsigned>(CLOSURE_VOLUME_RAYLEIGH_ID));
        sd.flag |= shader_data_scatter;
        if (write_spec) {
          *sd.spectral_scatter += max(*spec_weight, make_float4(0.0f));
        }
      };
    };
    $case(static_cast<unsigned>(CLOSURE_VOLUME_DRAINE_ID)) {
      draine_setup(
          sd, weight, [&] { return stack_load_input_float(stack, param1); },
          [&] { return stack_load_input_float(stack, param_extra); },
          spec_weight);
    };
    $case(static_cast<unsigned>(CLOSURE_VOLUME_MIE_ID)) {
      const auto p =
          phase::mie_parameters(stack_load_input_float(stack, param1));
      // Ordered independent allocations, not an atomic two-slot reservation:
      // capacity one retains HG while Draine fails, exactly as Cycles does.
      if (write_spec) {
        const Float4 hg_spec = (*spec_weight) * (1.0f - p.draine_weight);
        const Float4 draine_spec = (*spec_weight) * p.draine_weight;
        henyey_greenstein_setup(sd, weight * (1.0f - p.draine_weight),
                                [&] { return p.henyey_greenstein_g; },
                                &hg_spec);
        draine_setup(
            sd, weight * p.draine_weight, [&] { return p.draine_g; },
            [&] { return p.draine_alpha; }, &draine_spec);
      } else {
        henyey_greenstein_setup(sd, weight * (1.0f - p.draine_weight),
                                [&] { return p.henyey_greenstein_g; });
        draine_setup(
            sd, weight * p.draine_weight, [&] { return p.draine_g; },
            [&] { return p.draine_alpha; });
      }
    };
  };
}
} // namespace

void node_closure_volume(const KernelGlobals &kg, Cursor &cursor, Stack &stack,
                         Expr<luisa::float3> closure_weight, ShaderData &sd,
                         const EvaluationTransition &transition,
                         const SpectralClosureWeight *spectral_weight) noexcept {
  using Node = SVMNodeClosureVolume;
  const VolumeNodeView node{cursor, stack};
  const auto mix_weight = node.mix(offsetof(Node, mix_weight_offset));
  const bool has_spectral =
      sd.spectral.has_value() && sd.spectral_extinction.has_value() &&
      sd.spectral_scatter.has_value();
  $if(mix_weight != 0.0f) {
    with_object_density(
        kg, sd, transition, [&](Expr<float> object_density) noexcept {
          const auto density = mix_weight *
                               max(node.scalar(offsetof(Node, density)), 0.0f) *
                               object_density;
          const auto type = node.word(offsetof(Node, closure_type));
          Float3 weight = closure_weight;
          $if(type == static_cast<unsigned>(CLOSURE_VOLUME_ABSORPTION_ID)) {
            weight = make_float3(1.0f) - weight;
          };
          weight *= density;
          if (has_spectral) {
            Float4 spec_w = max(
                resolve_volume_spectral_reflectance(sd, closure_weight,
                                                    spectral_weight),
                make_float4(0.0f));
            $if(type == static_cast<unsigned>(CLOSURE_VOLUME_ABSORPTION_ID)) {
              spec_w = max(make_float4(1.0f) - spec_w, make_float4(0.0f));
            };
            spec_w *= density;
            $if(is_volume_scatter(type)) {
              allocate_volume_scatter(sd, stack, weight, type,
                                      node.word(offsetof(Node, param1)),
                                      node.word(offsetof(Node, param_extra)),
                                      &spec_w);
            };
            extinction_setup(sd, weight, &spec_w);
          } else {
            $if(is_volume_scatter(type)) {
              allocate_volume_scatter(sd, stack, weight, type,
                                      node.word(offsetof(Node, param1)),
                                      node.word(offsetof(Node, param_extra)));
            };
            extinction_setup(sd, weight);
          }
        });
  };
  cursor.advance(static_cast<unsigned>(sizeof(Node) / sizeof(std::uint32_t)));
}

void node_volume_coefficients(const KernelGlobals &kg, Cursor &cursor,
                              Stack &stack, Expr<luisa::float3> scatter_coeffs,
                              ShaderData &sd, const PathState &path,
                              const EvaluationTransition &transition,
                              const SpectralClosureWeight *spectral_weight) noexcept {
  using Node = SVMNodeVolumeCoefficients;
  const VolumeNodeView node{cursor, stack};
  const auto mix_weight = node.mix(offsetof(Node, mix_weight_offset));
  const bool has_spectral =
      sd.spectral.has_value() && sd.spectral_extinction.has_value() &&
      sd.spectral_scatter.has_value() && sd.spectral_emission_background.has_value();
  $if(mix_weight != 0.0f) {
    with_object_density(
        kg, sd, transition, [&](Expr<float> object_density) noexcept {
          const auto weight = mix_weight * object_density;
          const auto type = node.word(offsetof(Node, closure_type));
          const auto absorption =
              node.vector(offsetof(Node, absorption_coeffs));
          if (has_spectral) {
            const auto &sc = *sd.spectral;
            const Float4 scatter_spec =
                max(resolve_volume_spectral_reflectance(sd, scatter_coeffs,
                                                        spectral_weight),
                    make_float4(0.0f)) *
                weight;
            $if(any(scatter_coeffs != make_float3(0.0f)) &
                is_volume_scatter(type)) {
              allocate_volume_scatter(sd, stack, weight * scatter_coeffs, type,
                                      node.word(offsetof(Node, param1)),
                                      node.word(offsetof(Node, param_extra)),
                                      &scatter_spec);
            };
            const Float4 abs_spec =
                max(cycles_spectrum::rgb_to_spectrum_reflectance(
                        *sc.tables, absorption, sc.wavelengths.lambdas),
                    make_float4(0.0f)) *
                weight;
            const Float4 ext_spec = scatter_spec + abs_spec;
            extinction_setup(sd, weight * (scatter_coeffs + absorption),
                             &ext_spec);
          } else {
            $if(any(scatter_coeffs != make_float3(0.0f)) &
                is_volume_scatter(type)) {
              allocate_volume_scatter(sd, stack, weight * scatter_coeffs, type,
                                      node.word(offsetof(Node, param1)),
                                      node.word(offsetof(Node, param_extra)));
            };
            extinction_setup(sd, weight * (scatter_coeffs + absorption));
          }
          const auto emission = node.vector(offsetof(Node, emission_coeffs));
          $if((path.visibility & path_ray_visibility_shadow) == 0u) {
            $if(any(emission != make_float3(0.0f))) {
              if (has_spectral) {
                const auto &sc = *sd.spectral;
                Float4 em_spd = cycles_spectrum::rgb_to_spectrum_illuminant(
                    *sc.tables, emission, sc.wavelengths.lambdas);
                if (stack.is_spectral()) {
                  const auto em_x = node.word(offsetof(Node, emission_coeffs));
                  const auto try_slot = [&](Expr<std::uint32_t> off) noexcept {
                    Float4 slot_spd = make_float4(0.0f);
                    const Bool found = stack.try_load_spectral_slot(off, slot_spd);
                    em_spd = select(em_spd, slot_spd, found);
                  };
                  if (const auto ex = try_const_uint(em_x)) {
                    if ((*ex >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
                      try_slot(*ex & 0xffu);
                    }
                  } else {
                    $if((em_x >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
                      try_slot(em_x & 0xffu);
                    };
                  }
                }
                const Float4 em_spec = max(em_spd, make_float4(0.0f)) * weight;
                volume_emission_setup(sd, weight * emission, &em_spec);
              } else {
                volume_emission_setup(sd, weight * emission);
              }
            };
          };
        });
  };
  cursor.advance(static_cast<unsigned>(sizeof(Node) / sizeof(std::uint32_t)));
}

void node_principled_volume(const KernelGlobals &kg, Cursor &cursor,
                            Stack &stack, Expr<luisa::float3> closure_weight,
                            ShaderData &sd, const PathState &path,
                            const EvaluationTransition &transition,
                            const SpectralClosureWeight *spectral_weight) noexcept {
  using Node = SVMNodePrincipledVolume;
  const VolumeNodeView node{cursor, stack};
  const auto mix_weight = node.mix(offsetof(Node, mix_weight_offset));
  const bool has_spectral =
      sd.spectral.has_value() && sd.spectral_extinction.has_value() &&
      sd.spectral_scatter.has_value() && sd.spectral_emission_background.has_value();
  $if(mix_weight != 0.0f) {
    with_object_density(
        kg, sd, transition, [&](Expr<float> object_density) noexcept {
          const auto weight = mix_weight * object_density;
          Float density =
              weight * max(node.scalar(offsetof(Node, density)), 0.0f);
          $if(density > 0.0f) {
            const auto attr =
                find_attribute(kg, sd,
                               node.word(offsetof(Node, attr_density))
                                   .bitcast<int>()
                                   .cast<luisa::ulong>());
            $if(is_attribute_found(attr)) {
              density = max(density * primitive_volume_attribute_float(
                                          kg, sd, attr, true),
                            0.0f);
            };
          };
          $if(density > 0.0f) {
            Float3 color = closure_weight;
            Float3 attr_color = make_float3(1.0f);
            Bool attr_color_found = false;
            const auto attr =
                find_attribute(kg, sd,
                               node.word(offsetof(Node, attr_color))
                                   .bitcast<int>()
                                   .cast<luisa::ulong>());
            $if(is_attribute_found(attr)) {
              attr_color = primitive_volume_attribute_float3(kg, sd, attr, true);
              attr_color_found = true;
              color *= attr_color;
            };
            const auto absorption_color =
                max(sqrt(node.vector(offsetof(Node, absorption_color))),
                    make_float3(0.0f));
            const auto absorption =
                max(make_float3(1.0f) - color, make_float3(0.0f)) *
                max(make_float3(1.0f) - absorption_color, make_float3(0.0f));
            if (has_spectral) {
              const auto &sc = *sd.spectral;
              Float4 color_spec = max(
                  resolve_volume_spectral_reflectance(sd, closure_weight,
                                                      spectral_weight),
                  make_float4(0.0f));
              $if(attr_color_found) {
                color_spec *= max(
                    cycles_spectrum::rgb_to_spectrum_reflectance(
                        *sc.tables, attr_color, sc.wavelengths.lambdas),
                    make_float4(0.0f));
              };
              const Float4 scatter_spec = color_spec * density;
              henyey_greenstein_setup(
                  sd, color * density,
                  [&] { return node.scalar(offsetof(Node, anisotropy)); },
                  &scatter_spec);
              const Float4 abs_color_spec = max(
                  cycles_spectrum::rgb_to_spectrum_reflectance(
                      *sc.tables, absorption_color, sc.wavelengths.lambdas),
                  make_float4(0.0f));
              const Float4 abs_spec =
                  max(make_float4(1.0f) - color_spec, make_float4(0.0f)) *
                  max(make_float4(1.0f) - abs_color_spec, make_float4(0.0f));
              const Float4 ext_spec = (color_spec + abs_spec) * density;
              extinction_setup(sd, (color + absorption) * density, &ext_spec);
            } else {
              henyey_greenstein_setup(sd, color * density, [&] {
                return node.scalar(offsetof(Node, anisotropy));
              });
              extinction_setup(sd, (color + absorption) * density);
            }
          };
          $if((path.visibility & path_ray_visibility_shadow) == 0u) {
            const auto emission = node.scalar(offsetof(Node, emission));
            const auto blackbody = node.scalar(offsetof(Node, blackbody));
            $if(emission > 0.0f) {
              const auto emission_color =
                  node.vector(offsetof(Node, emission_color));
              if (has_spectral) {
                const auto &sc = *sd.spectral;
                Float4 em_spd = cycles_spectrum::rgb_to_spectrum_illuminant(
                    *sc.tables, emission_color, sc.wavelengths.lambdas);
                if (stack.is_spectral()) {
                  const auto em_x = node.word(offsetof(Node, emission_color));
                  const auto try_slot = [&](Expr<std::uint32_t> off) noexcept {
                    Float4 slot_spd = make_float4(0.0f);
                    const Bool found = stack.try_load_spectral_slot(off, slot_spd);
                    em_spd = select(em_spd, slot_spd, found);
                  };
                  if (const auto ex = try_const_uint(em_x)) {
                    if ((*ex >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
                      try_slot(*ex & 0xffu);
                    }
                  } else {
                    $if((em_x >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
                      try_slot(em_x & 0xffu);
                    };
                  }
                }
                const Float4 em_spec =
                    max(em_spd, make_float4(0.0f)) * (emission * weight);
                volume_emission_setup(sd, emission * emission_color * weight,
                                      &em_spec);
              } else {
                volume_emission_setup(sd, emission * emission_color * weight);
              }
            };
            $if(blackbody > 0.0f) {
              Float temperature = node.scalar(offsetof(Node, temperature));
              const auto attr =
                  find_attribute(kg, sd,
                                 node.word(offsetof(Node, attr_temperature))
                                     .bitcast<int>()
                                     .cast<luisa::ulong>());
              $if(is_attribute_found(attr)) {
                temperature *= max(
                    primitive_volume_attribute_float(kg, sd, attr, true), 0.0f);
              };
              temperature = max(temperature, 0.0f);
              const auto squared = temperature * temperature;
              const auto t4 = squared * squared;
              constexpr auto sigma = 5.670373e-8f * 1e-6f / phase::pi;
              const auto intensity = sigma * lerp(1.0f, t4, blackbody);
              $if(intensity > 0.0f) {
                const auto tint = node.vector(offsetof(Node, blackbody_tint));
                const auto bb =
                    tint * intensity *
                    rec709_to_rgb(
                        kg, cycles_color_nodes::blackbody_rec709(temperature));
                if (has_spectral) {
                  const auto &sc = *sd.spectral;
                  const Float4 bb_spec =
                      max(cycles_spectrum::rgb_to_spectrum_reflectance(
                              *sc.tables, tint, sc.wavelengths.lambdas),
                          make_float4(0.0f)) *
                      intensity *
                      cycles_spectrum::blackbody_spd(*sc.tables, temperature,
                                                     sc.wavelengths.lambdas) *
                      weight;
                  volume_emission_setup(sd, bb * weight, &bb_spec);
                } else {
                  volume_emission_setup(sd, bb * weight);
                }
              };
            };
          };
        });
  };
  cursor.advance(static_cast<unsigned>(sizeof(Node) / sizeof(std::uint32_t)));
}
} // namespace detail
} // namespace psycles::luisa_backend::cycles_svm
