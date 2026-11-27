#include "path_tracer_cycles_svm_surface.h"

#include "path_tracer_cycles_svm_kernel_globals.h"
#include "path_kernel_caustics_manifold.h"
#include "cycles_svm_bsdf.h"
#include "cycles_svm_internal.h"
#include "cycles_svm_surface_shader.h"
#include "cycles_svm_subsurface.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_svm_translator.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;
namespace svm = ::psycles::luisa_backend::cycles_svm;
namespace svm_detail = ::psycles::luisa_backend::cycles_svm::detail;
namespace closure = ::psycles::luisa_backend::cycles_closure;
namespace abi = ::psycles::compiler::cycles_svm;

[[nodiscard]] UInt runtime_flags(Expr<std::uint32_t> flags) noexcept {
  UInt result = 0u;
  const auto append = [&](std::uint32_t source,
                          std::uint32_t destination) noexcept {
    result |= select(0u, destination, (flags & source) != 0u);
  };
  append(abi::SD_BACKFACING, closure::runtime_backfacing);
  append(abi::SD_CACHE_MISS, closure::runtime_cache_miss);
  append(abi::SD_EMISSION, closure::runtime_emission);
  append(abi::SD_BSDF, closure::runtime_bsdf);
  append(abi::SD_BSDF_HAS_EVAL, closure::runtime_bsdf_has_eval);
  append(abi::SD_BSSRDF, closure::runtime_bssrdf);
  append(abi::SD_HOLDOUT, closure::runtime_holdout);
  append(abi::SD_EXTINCTION, closure::runtime_extinction);
  append(abi::SD_SCATTER, closure::runtime_scatter);
  append(abi::SD_IS_VOLUME_SHADER_EVAL,
         closure::runtime_is_volume_shader_eval);
  append(abi::SD_TRANSPARENT, closure::runtime_transparent);
  append(abi::SD_BSDF_HAS_TRANSMISSION,
         closure::runtime_bsdf_has_transmission);
  append(abi::SD_RAY_PORTAL, closure::runtime_ray_portal);
  return result;
}

[[nodiscard]] UInt events_from_label(Expr<std::uint32_t> label) noexcept {
  UInt result = static_cast<std::uint32_t>(contract::event_none);
  const auto append = [&](std::uint32_t source,
                          contract::SurfaceEvent destination) noexcept {
    result |= select(0u, static_cast<std::uint32_t>(destination),
                     (label & source) != 0u);
  };
  append(closure::label_transmit, contract::event_transmission);
  append(closure::label_reflect, contract::event_reflection);
  append(closure::label_diffuse, contract::event_diffuse);
  append(closure::label_glossy, contract::event_glossy);
  append(closure::label_singular, contract::event_singular);
  append(closure::label_transparent, contract::event_transparent);
  append(closure::label_subsurface_scatter, contract::event_subsurface);
  // LABEL_TRANSMIT_TRANSPARENT is a path-visibility hint on an ordinary
  // transmission event, not Cycles' transparent closure identity.
  return result;
}

[[nodiscard]] UInt bssrdf_method(Expr<std::uint32_t> type) noexcept {
  UInt result = static_cast<std::uint32_t>(
      SurfaceBssrdfMethod::random_walk);
  $switch(type) {
    $case(closure::type_bssrdf_burley) {
      result = static_cast<std::uint32_t>(SurfaceBssrdfMethod::burley);
    };
    $case(closure::type_bssrdf_random_walk) {
      result = static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk);
    };
    $case(closure::type_bssrdf_random_walk_legacy) {
      result = static_cast<std::uint32_t>(
          SurfaceBssrdfMethod::random_walk_legacy);
    };
    $case(closure::type_bssrdf_random_walk_skin) {
      result = static_cast<std::uint32_t>(
          SurfaceBssrdfMethod::random_walk_skin);
    };
    $default {};
  };
  return result;
}

[[nodiscard]] Bool nonzero(Expr<luisa::float3> value) noexcept {
  return (value.x != 0.0f) | (value.y != 0.0f) | (value.z != 0.0f);
}

[[nodiscard]] Bool nonzero4(Expr<luisa::float4> value) noexcept {
  return any(value != 0.0f);
}

[[nodiscard]] UInt evaluation_events(
    const svm_detail::SurfaceShaderBsdfEval &evaluation,
    const svm::ShaderData &shader_data,
    Expr<luisa::float3> outgoing) noexcept {
  const auto has_diffuse = evaluation.spectral_diffuse
                               ? nonzero4(*evaluation.spectral_diffuse)
                               : nonzero(evaluation.diffuse);
  const auto has_glossy = evaluation.spectral_glossy
                              ? nonzero4(*evaluation.spectral_glossy)
                              : nonzero(evaluation.glossy);
  const auto has_scatter = evaluation.spectral_sum
                               ? nonzero4(*evaluation.spectral_sum)
                               : nonzero(evaluation.sum);
  UInt result = 0u;
  result |= select(0u, static_cast<std::uint32_t>(contract::event_diffuse),
                   has_diffuse);
  result |= select(0u, static_cast<std::uint32_t>(contract::event_glossy),
                   has_glossy);
  const auto reflection = dot(shader_data.Ng, outgoing) >= 0.0f;
  result |= select(0u,
                   static_cast<std::uint32_t>(contract::event_reflection),
                   has_scatter & reflection);
  result |= select(0u,
                   static_cast<std::uint32_t>(contract::event_transmission),
                   has_scatter & !reflection);
  return result;
}

class CyclesSvmPopulatedSurface final : public PopulatedSurfaceShader {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    SurfacePoint _point;
    PathCyclesSvmKernelGlobals _kernel_globals;
    std::unique_ptr<svm::ClosurePool> _closures;
    std::unique_ptr<svm::ShaderData> _shader_data;
    svm_detail::ClosureTypeMask _closure_types;
    SurfacePopulationQuery _preparation_query;
    bool _include_aov_roughness{};
    bool _spectral{false};
    std::optional<Bool> _has_sms_eligible_closure{};

  private:
    [[nodiscard]] SurfacePreparation make_preparation(
        const SurfacePopulationQuery &query) const noexcept {
      auto result = SurfacePreparation::zero(_point, _spectral);
      const auto flags = _shader_data->flag;
      const auto emission_cosine = abs(dot(_shader_data->Ng,
                                           _shader_data->wi));
      const auto has_emission =
          ((flags & static_cast<std::uint32_t>(abi::SD_EMISSION)) != 0u) &
          (emission_cosine > 0.0f);
      result.emission = select(
          make_float3(0.0f),
          _shader_data->closure_emission_background,
          has_emission);
      if (_spectral && _shader_data->spectral_emission_background.has_value()) {
        result.spectral_emission = select(
            make_float4(0.0f),
            *_shader_data->spectral_emission_background,
            has_emission);
      }
      result.shading_normal = _shader_data->N;
      result.runtime_flags = select(
          0u, runtime_flags(flags), query.include_runtime_flags);
      // Cycles enters its film data-pass closure reductions only for paths
      // which can write those passes. Keep the same runtime boundary here:
      // selecting the outputs afterwards still executes every bsdf_albedo
      // branch and closure reduction on non-AOV bounces.
      result.aov.normal = _shader_data->N;
      $if(query.include_aov) {
        constexpr auto bit = [](std::uint32_t t) constexpr noexcept {
          return svm_detail::ClosureTypeMask{1u} << t;
        };
        constexpr auto glossy_or_glass_mask =
            bit(closure::type_microfacet_ggx) |
            bit(closure::type_microfacet_beckmann) |
            bit(closure::type_microfacet_multi_ggx) |
            bit(closure::type_ashikhmin_shirley) |
            bit(closure::type_ashikhmin_velvet) |
            bit(closure::type_glossy_toon) |
            bit(closure::type_hair_reflection) |
            bit(closure::type_microfacet_beckmann_glass) |
            bit(closure::type_microfacet_ggx_glass) |
            bit(closure::type_microfacet_multi_ggx_glass) |
            bit(closure::type_hair_chiang) | bit(closure::type_hair_huang);
        constexpr auto transmission_or_glass_mask =
            bit(closure::type_microfacet_beckmann_refraction) |
            bit(closure::type_microfacet_ggx_refraction) |
            bit(closure::type_thin_glass_transmission) |
            bit(closure::type_hair_transmission) |
            bit(closure::type_microfacet_beckmann_glass) |
            bit(closure::type_microfacet_ggx_glass) |
            bit(closure::type_microfacet_multi_ggx_glass);
        const auto active_glossy_mask = _closure_types & glossy_or_glass_mask;
        const auto active_transmission_mask =
            _closure_types & transmission_or_glass_mask;

        Float3 diffuse = make_float3(0.0f);
        Float3 glossy = make_float3(0.0f);
        Float3 transmission = make_float3(0.0f);
        Float3 average_normal = make_float3(0.0f);
        Float roughness = 0.0f;
        Float roughness_weight = 0.0f;
        const auto accumulate_aov_slot = [&](Expr<std::uint32_t> index) noexcept {
          const auto common = _closures->common(index);
          const auto is_bssrdf = closure::is_bssrdf(common.type);
          $if(closure::is_bsdf_diffuse(common.type) | is_bssrdf) {
            diffuse += common.weight;
          };
          if (active_glossy_mask != 0u) {
            $if(closure::is_bsdf_glossy(common.type) |
                closure::is_glass(common.type)) {
              glossy += svm_detail::bsdf_albedo(
                  _kernel_globals, *_shader_data, index, true, false,
                  active_glossy_mask);
            };
          }
          if (active_transmission_mask != 0u) {
            $if(closure::is_bsdf_transmission(common.type) |
                closure::is_glass(common.type)) {
              transmission += svm_detail::bsdf_albedo(
                  _kernel_globals, *_shader_data, index, false, true,
                  active_transmission_mask);
            };
          }
          $if(closure::is_bsdf_or_bssrdf(common.type)) {
            const auto weight = abs((common.weight.x + common.weight.y +
                                     common.weight.z) /
                                    3.0f);
            average_normal += common.N * weight;
          };
          if (_include_aov_roughness) {
            $if(closure::is_bsdf(common.type)) {
              const auto value = svm_detail::bsdf_get_roughness_pass_squared(
                  *_closures, index);
              $if(value >= 0.0f) {
                const auto weight = abs((common.weight.x + common.weight.y +
                                         common.weight.z) /
                                        3.0f);
                roughness += weight * sqrt(sqrt(value));
                roughness_weight += weight;
              };
            };
          }
        };
        if (_closures->capacity() <= 1u) {
          $if(_closures->count() != 0u) { accumulate_aov_slot(0u); };
        } else {
          UInt index = 0u;
          $while(index < _closures->count()) {
            accumulate_aov_slot(index);
            index += 1u;
          };
        }

        Float3 transparency = make_float3(0.0f);
        $if((flags & static_cast<std::uint32_t>(abi::SD_HAS_ONLY_VOLUME)) !=
            0u) {
          transparency = make_float3(1.0f);
        }
        $elif((flags & static_cast<std::uint32_t>(
                            abi::SD_TRANSPARENT | abi::SD_RAY_PORTAL)) != 0u) {
          transparency = _shader_data->closure_transparent_extinction;
        };
        const auto normal = select(
            _shader_data->N,
            svm_detail::safe_normalize_cycles(average_normal),
            nonzero(average_normal));
        result.aov.albedo = diffuse;
        result.aov.glossy_albedo = glossy;
        result.aov.transmission_albedo = transmission;
        if (_include_aov_roughness) {
          const auto average_roughness = select(
              1.0f, roughness / roughness_weight, roughness_weight > 0.0f);
          result.aov.roughness = make_float2(average_roughness);
        }
        result.aov.normal = normal;
        result.aov.transparency = transparency;
      };
      return result;
    }

    [[nodiscard]] SurfaceSampleTrace sample_nonempty_impl(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction) const noexcept {
      const auto pick = svm_detail::surface_shader_bsdf_bssrdf_pick(
          *_shader_data, make_float3(u_direction, u_lobe));
      // Cycles returns &sd->closure[sampled] directly. The caller has already
      // rejected ShaderData without SD_BSDF/SD_BSSRDF, and the picker only
      // selects from the initialized closure prefix; clamping to capacity
      // would instead make an uninitialized [count, capacity) slot readable.
      const auto closure_index = pick.index;
      const auto common = _closures->common(closure_index);
      constexpr auto bssrdf_mask =
          (svm_detail::ClosureTypeMask{1u} << closure::type_bssrdf_burley) |
          (svm_detail::ClosureTypeMask{1u} << closure::type_bssrdf_random_walk) |
          (svm_detail::ClosureTypeMask{1u} <<
           closure::type_bssrdf_random_walk_legacy) |
          (svm_detail::ClosureTypeMask{1u} <<
           closure::type_bssrdf_random_walk_skin);
      const auto has_subsurface =
          ((_scene->cycles_svm->kernel_features &
            svm::kernel_feature_subsurface) != 0u) &&
          ((_closure_types & bssrdf_mask) != 0ull);
      const auto selected =
          (has_subsurface ? closure::is_bsdf_or_bssrdf(common.type)
                          : closure::is_bsdf(common.type)) &
          (common.sample_weight > 0.0f);

      auto trace = SurfaceSampleTrace::zero(_spectral);
      auto &result = trace.sample;
      result.runtime_flags = runtime_flags(_shader_data->flag);

      const auto sample_bsdf = [&] noexcept {
        const auto sampled = svm_detail::surface_shader_bsdf_sample_closure(
            _kernel_globals, *_shader_data, pick, _closure_types);
        const auto sampled_valid =
            selected & (sampled.label != closure::label_none) &
            (sampled.evaluation.pdf > 0.0f);
        result.evaluation.f = select(
            make_float3(0.0f), sampled.evaluation.sum, sampled_valid);
        result.evaluation.pdf = select(
            0.0f, sampled.evaluation.pdf, sampled_valid);
        result.evaluation.diffuse_f = select(
            make_float3(0.0f), sampled.evaluation.diffuse, sampled_valid);
        result.evaluation.glossy_f = select(
            make_float3(0.0f), sampled.evaluation.glossy, sampled_valid);
        if (_spectral) {
          result.evaluation.spectral_f = select(
              make_float4(0.0f), *sampled.evaluation.spectral_sum,
              sampled_valid);
          result.evaluation.spectral_diffuse_f = select(
              make_float4(0.0f), *sampled.evaluation.spectral_diffuse,
              sampled_valid);
          result.evaluation.spectral_glossy_f = select(
              make_float4(0.0f), *sampled.evaluation.spectral_glossy,
              sampled_valid);
          result.evaluation.spectral_terminated_secondary =
              sampled_valid &
              sampled.spectral_terminated_secondary.value_or(Bool{false});
          if (_kernel_globals.xyz3_rough_dispersion()) {
            result.evaluation.spectral_lane_pdf = select(
                make_float4(0.0f),
                sampled.evaluation.spectral_lane_pdf.value_or(
                    make_float4(sampled.evaluation.pdf)),
                sampled_valid);
          }
        }
        const auto sampled_events = events_from_label(sampled.label);
        result.evaluation.diffuse_pdf = select(
            0.0f, sampled.evaluation.pdf,
            sampled_valid &
                ((sampled_events & static_cast<std::uint32_t>(
                                       contract::event_diffuse)) != 0u));
        result.evaluation.average_roughness_squared = select(
            0.0f, sampled.evaluation.average_roughness_squared,
            sampled_valid);
        result.evaluation.events = select(0u, sampled_events, sampled_valid);
        result.wi = select(make_float3(0.0f, 0.0f, 1.0f), sampled.wo,
                           sampled_valid);
        result.eta = select(1.0f, sampled.eta, sampled_valid);
        result.roughness = select(make_float2(0.0f),
                                  sampled.sampled_roughness, sampled_valid);
        result.valid = sampled_valid;
      };

      if (has_subsurface) {
        const auto selected_bssrdf =
            selected & closure::is_bssrdf(common.type);
        // Direct projection of Cycles' __SUBSURFACE__ branch: an ordinary
        // BSDF sample is not evaluated for a selected BSSRDF closure.
        $if(selected_bssrdf) {
          const auto bssrdf = _closures->bssrdf(closure_index);
          const auto bssrdf_weight =
              svm_detail::surface_shader_bssrdf_sample_weight(
                  *_shader_data, closure_index);
          result.evaluation.f = bssrdf_weight;
          if (_spectral && _shader_data->spectral.has_value()) {
            const auto &sc = *_shader_data->spectral;
            result.evaluation.spectral_f =
                cycles_spectrum::rgb_to_spectrum_reflectance(
                    *sc.tables, bssrdf_weight, sc.wavelengths.lambdas);
            if (_kernel_globals.xyz3_rough_dispersion()) {
              result.evaluation.spectral_lane_pdf = make_float4(1.0f);
            }
          }
          result.evaluation.pdf = 1.0f;
          result.evaluation.events =
              static_cast<std::uint32_t>(contract::event_subsurface);
          result.wi = common.N;
          result.bssrdf_method = bssrdf_method(common.type);
          result.bssrdf_radius = bssrdf.param.radius;
          result.bssrdf_albedo = bssrdf.param.albedo;
          result.bssrdf_normal = common.N;
          result.bssrdf_ior = bssrdf.param.ior;
          result.bssrdf_roughness = bssrdf.param.alpha;
          result.bssrdf_anisotropy = bssrdf.param.anisotropy;
          result.valid = true;
        }
        $else { sample_bsdf(); };
      } else {
        sample_bsdf();
      }

      trace.closure_index = select(0u, closure_index, selected);
      trace.closure_type = select(0u, common.type, selected);
      trace.closure_sample_weight = select(0.0f, common.sample_weight,
                                           selected);
      trace.selection_rescaled = select(0.0f, pick.random.z, selected);
      trace.closure_weight = select(make_float3(0.0f), common.weight,
                                    selected);
      trace.closure_normal = select(make_float3(0.0f, 0.0f, 1.0f), common.N,
                                    selected);
      trace.closure_valid = selected;
      return trace;
    }

    [[nodiscard]] SurfaceSampleTrace sample_impl(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction) const noexcept {
      auto result = SurfaceSampleTrace::zero(_spectral);
      /* Cycles 5.2.1 integrate_surface_bsdf_bssrdf_bounce() rejects a
       * surface without SD_BSDF/SD_BSSRDF before calling the closure picker.
       * Keep that precondition explicit in the pointer-free projection: an
       * empty ClosurePool has no initialized closure[0] to inspect. */
      constexpr auto scatter_flags =
          svm::shader_data_bsdf | svm::shader_data_bssrdf;
      $if((_shader_data->flag & scatter_flags) != 0u) {
        result = sample_nonempty_impl(u_lobe, u_direction);
      };
      return result;
    }

  public:
    [[nodiscard]] std::optional<CyclesSvmSurfaceState>
    native_surface_state() const noexcept override {
      return CyclesSvmSurfaceState{_kernel_globals, *_shader_data, _closure_types};
    }

    CyclesSvmPopulatedSurface(
        std::shared_ptr<LuisaSceneData> scene,
        const SurfacePopulationContext &context,
        const CyclesSvmSurfaceBucketSpecialization &bucket) noexcept
        : _scene{std::move(scene)},
          _point{context.point},
          _kernel_globals{
              _scene, context,
              std::span<const std::uint8_t>{bucket.active_shader_mask},
              false, context.vt_miss_page, context.spectral.has_value(),
              context.spectral ? context.spectral->mode
                               : contract::SpectralMode::none},
          _closures{[&]() noexcept {
            const auto &cu = _kernel_globals.closure_usage();
            constexpr std::uint64_t wide_closure_mask =
                abi::ClosureUsage::bit(closure::type_sheen) |
                abi::ClosureUsage::bit(closure::type_hair_chiang) |
                abi::ClosureUsage::bit(closure::type_hair_huang) |
                abi::ClosureUsage::bit(closure::type_bssrdf_burley) |
                abi::ClosureUsage::bit(closure::type_bssrdf_random_walk) |
                abi::ClosureUsage::bit(closure::type_bssrdf_random_walk_legacy) |
                abi::ClosureUsage::bit(closure::type_bssrdf_random_walk_skin);
            const bool compact_64b =
                !cu.has_fresnel(abi::ClosureUsage::fresnel_anisotropic) &&
                (cu.surface_closure_mask & wide_closure_mask) == 0ull;
            return std::make_unique<svm::ClosurePool>(
                std::clamp<std::size_t>(
                    bucket.max_surface_closures != 0u
                        ? bucket.max_surface_closures
                        : (_scene->cycles_svm
                               ? _scene->cycles_svm->compilation
                                     .max_surface_closures
                               : _scene->volume_metadata
                                     .closure_allocation_budget),
                    1u, svm::maximum_closure_capacity),
                svm::ClosurePool::Options{
                    .may_have_thin_film = _kernel_globals.may_have_thin_film(),
                    .may_have_anisotropic_microfacet = cu.has_fresnel(
                        abi::ClosureUsage::fresnel_anisotropic),
                    .may_have_custom_generalized_schlick = cu.has_fresnel(
                        abi::ClosureUsage::fresnel_custom_generalized_schlick),
                    .may_have_positive_exponent = cu.has_fresnel(
                        abi::ClosureUsage::fresnel_positive_exponent),
                    .may_have_extra_fresnel = cu.has_fresnel(
                        abi::ClosureUsage::fresnel_conductor |
                        abi::ClosureUsage::fresnel_f82_tint |
                        abi::ClosureUsage::fresnel_generalized_schlick),
                    .outline_transparent_merge =
                        _scene->device.backend_name() == "simd" &&
                        _scene->device.compute_warp_size() > 1u,
                    .spectral = context.spectral.has_value(),
                    .compact_64b_slots = compact_64b});
          }()},
          _closure_types{
              svm_detail::closure_types_for_kernel_features(
                  _scene->cycles_svm->kernel_features) &
              (_kernel_globals.closure_usage().surface_closure_mask |
               (((_scene->cycles_svm->kernel_features &
                  svm::kernel_feature_subsurface) != 0u)
                    ? abi::ClosureUsage::bit(closure::type_diffuse)
                    : 0ull))},
          _preparation_query{context.query},
          _include_aov_roughness{false},
          _spectral{context.spectral.has_value()} {
      UInt flags;
      if (context.cycles_shader_flags) {
        flags = *context.cycles_shader_flags;
      } else {
        const Expr<Buffer<abi::KernelShader>> shaders{
            *_scene->cycles_svm->kernel_shader_buffer};
        const auto shader = shaders->read(
            context.cycles_surface_shader & svm::shader_mask);
        flags = shader.flags.cast<std::uint32_t>() |
                select(0u, static_cast<std::uint32_t>(abi::SD_BACKFACING),
                       context.point.back_facing);
      }
      const UInt object_flags =
          context.cycles_object_flags
              ? UInt{*context.cycles_object_flags}
              : _scene->cycles_svm->objects->object_flag_buffer->read(
                    context.cycles_object_index);
      const auto lcg_state = cycles_noise::hash_uint3(
          context.rng_hash ^ 0xb4bc3953u,
          context.rng_offset,
          context.sample_index);
      const auto identity = make_float4x4(1.0f);
      _shader_data = std::make_unique<svm::ShaderData>(
          context.point.position,
          context.point.shading_normal,
          context.point.geometric_normal,
          context.point.incoming,
          context.primitive_type,
          context.cycles_surface_shader,
          flags,
          object_flags,
          context.cycles_primitive_index,
          context.point.barycentric.x,
          context.point.barycentric.y,
          context.cycles_object_index,
          context.point.time,
          context.point.ray_length,
          context.ray_position_differential,
          context.ray_direction_differential,
          context.point.barycentric_dx.x,
          context.point.barycentric_dy.x,
          context.point.barycentric_dx.y,
          context.point.barycentric_dy.y,
          context.point.dpdu,
          context.point.dpdv,
          identity,
          identity,
          lcg_state,
          _closures.get());
      /* KernelShader::flags contains only Cycles SD_SHADER_FLAGS; runtime
       * closure bits are produced by SVM closure setup below. State that
       * scene-table ABI invariant explicitly so XIR can relate the later
       * SD_BSDF/SD_BSSRDF test to a successful closure allocation without a
       * renderer-specific lifetime marker or a redundant device branch. */
      assume((_shader_data->flag &
              static_cast<std::uint32_t>(abi::SD_BSDF |
                                         abi::SD_BSSRDF)) == 0u);
      _shader_data->ray_P = context.ray_origin;
      _shader_data->spectral = context.spectral;
      if (_spectral) {
        _shader_data->spectral_emission_background.emplace(make_float4(0.0f));
      }

      const auto unpack_tfm = [](Var<abi::PackedTransform> t) noexcept {
        return svm_detail::transform_from_rows(
            make_float4(t.x.x, t.x.y, t.x.z, t.x.w),
            make_float4(t.y.x, t.y.y, t.y.z, t.y.w),
            make_float4(t.z.x, t.z.y, t.z.z, t.z.w));
      };
      const auto scene_ref = _scene;
      const auto obj_idx = context.cycles_object_index;
      svm::TransformState transform_state{
          context.parameters.camera_transform,
          context.parameters.camera_inverse_transform,
          [scene_ref, obj_idx, unpack_tfm]() noexcept {
            return unpack_tfm(
                scene_ref->cycles_svm->objects->object_buffer
                    ->read(obj_idx)
                    .tfm);
          },
          [scene_ref, obj_idx, unpack_tfm]() noexcept {
            return unpack_tfm(
                scene_ref->cycles_svm->objects->object_buffer
                    ->read(obj_idx)
                    .itfm);
          }};
      if (scene_ref && scene_ref->cycles_svm && scene_ref->cycles_svm->objects) {
        transform_state.lazy_object_buffer =
            &scene_ref->cycles_svm->objects->object_buffer;
      }
      const svm::PathState path_state{
          cycles_path_state::from_contract_shader_visibility(
              context.point.ray_visibility),
          context.path_flags,
          context.point.ray_depth,
          context.point.transparent_depth,
          context.point.diffuse_depth,
          context.point.glossy_depth,
          context.point.transmission_depth,
          (_scene->cycles_svm->kernel_features & svm::kernel_feature_node_portal)
              ? UInt{context.portal_depth} : UInt{0u}};
      const Expr<Buffer<luisa::uint>> words{
          *_scene->cycles_svm->word_buffer};
      const auto evaluate_material = [&] {
        const auto usage =
            bucket.entry_usage.has_value()
                ? *bucket.entry_usage
                : cycles_svm_domain_entry_usage(
                      *_scene, PathCyclesSvmShaderDomain::surface_only);
        const auto default_mask =
            bucket.active_shader_mask.empty()
                ? cycles_svm_domain_shader_mask(
                      *_scene, PathCyclesSvmShaderDomain::surface_only)
                : std::vector<std::uint8_t>{};
        const auto active_mask =
            bucket.active_shader_mask.empty()
                ? std::span<const std::uint8_t>{default_mask}
                : std::span<const std::uint8_t>{bucket.active_shader_mask};
        const auto backend_name = _scene->device.backend_name();
        const bool is_simd = backend_name == "simd";
        const bool is_simd_wide =
            is_simd && _scene->device.compute_warp_size() > 1u;
        const bool cpu_backend = backend_name == "fallback" || is_simd;
        const bool outline_svm_chunks = backend_name == "fallback" || is_simd_wide;
        svm_detail::surface_shader_initialize_closures(
            *_closures, path_state.visibility, path_state.flag);
        const auto simd_chunk_groups = []() noexcept -> std::size_t {
          if (const char *raw = std::getenv("PSYCLES_SIMD_SVM_CHUNK_GROUPS");
              raw != nullptr && *raw != '\0') {
            if (const auto parsed = std::strtoull(raw, nullptr, 10);
                parsed > 0ull) {
              return static_cast<std::size_t>(parsed);
            }
          }
          return std::size_t{2u};
        }();
        const auto simd_chunk_weight = []() noexcept -> std::size_t {
          if (const char *raw = std::getenv("PSYCLES_SIMD_SVM_CHUNK_WEIGHT");
              raw != nullptr && *raw != '\0') {
            if (const auto parsed = std::strtoull(raw, nullptr, 10);
                parsed > 0ull) {
              return static_cast<std::size_t>(parsed);
            }
          }
          return std::size_t{180u};
        }();
        svm::translate_svm_nodes_assume_valid(
            svm::SvmTranslationRequest{
                .kernel_globals = _kernel_globals,
                .table = _scene->cycles_svm->compilation.table,
                .words = words,
                .shader_type = abi::SHADER_TYPE_SURFACE,
                .kernel_features = _scene->cycles_svm->kernel_features,
                .node_feature_mask = svm::kernel_feature_node_mask_surface,
                .node_types_used = usage.node_types_used,
                .transform_state = transform_state,
                .shader_data = *_shader_data,
                .path_state = path_state,
                .stack_size =
                    std::max<std::size_t>(1u, usage.peak_stack_usage),
                .noise_usage = usage.noise_usage,
                .active_shader_mask = active_mask,
                .max_groups_per_outlined_chunk =
                    (cpu_backend && outline_svm_chunks)
                        ? (is_simd_wide ? simd_chunk_groups : std::size_t{16u})
                        : std::size_t{0u},
                .max_chunk_jit_weight =
                    is_simd_wide ? simd_chunk_weight : std::size_t{900u},
                .outline_chunk_dispatcher = true},
            cpu_backend ? svm::SvmBackendStrategy::direct_jit
                        : svm::SvmBackendStrategy::auto_select);
      };
      if (context.record_sms_eligibility) {
        _has_sms_eligible_closure.emplace(Bool{false});
      }
      const bool may_filter_glossy = context.may_filter_glossy;
      const auto prepare_closures = [&] {
        if (context.record_sms_eligibility) {
          Bool eligible = false;
          UInt ci = 0u;
          $while(ci < _closures->count()) {
            const auto common = _closures->common(ci);
            const auto is_sms_cl =
                cycles_closure::is_glass(common.type) |
                cycles_closure::is_reflection_microfacet(common.type) |
                cycles_closure::is_refraction_microfacet(common.type);
            $if(is_sms_cl) {
              const auto m = _closures->microfacet(ci);
              const auto alpha2 = m.param.alpha_x * m.param.alpha_y;
              eligible |= (alpha2 <= kMaxSmsAlphaSquared);
            };
            ci += 1u;
          };
          *_has_sms_eligible_closure = eligible;
        }
        if (may_filter_glossy) {
          svm_detail::surface_shader_prepare_closures(
              *_shader_data, context.query.glossy_filter_roughness,
              _closure_types);
        }
      };
      // Cycles integrate_surface: SSS exits without bump never evaluate the
      // material. Bumped exits evaluate it once for the BSSRDF normal, then
      // replace the retained closures before any ordinary BSDF consumer.
      if ((_scene->cycles_svm->kernel_features &
           svm::kernel_feature_subsurface) != 0u) {
        const auto subsurface_exit =
            (path_state.flag & cycles_path_state::flag_subsurface) != 0u;
        const auto material_evaluated = svm_detail::surface_shader_material_eval_required(
            subsurface_exit, _shader_data->flag);
        $if(material_evaluated) { evaluate_material(); };
        $if(subsurface_exit) {
          svm_detail::subsurface_shader_data_setup(*_shader_data);
        }
        $else { prepare_closures(); };
      } else {
        evaluate_material();
        prepare_closures();
      }
    }

    [[nodiscard]] Expr<std::uint32_t>
    native_shader_data_flags() const noexcept override {
      return _shader_data->flag;
    }

    [[nodiscard]] Bool has_sms_eligible_closure() const noexcept override {
      return _has_sms_eligible_closure.value_or(Bool{false});
    }

    [[nodiscard]] Expr<std::uint32_t>
    closure_count() const noexcept override {
      return _closures->count();
    }

    [[nodiscard]] SurfacePreparation preparation() const noexcept override {
      // Native holdout runs after closure filtering and before emission/pass
      // consumers. Observe the current retained state, never an eager cache
      // captured before that integrator mutation.
      return make_preparation(_preparation_query);
    }

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept override {
      const bool outline_bsdf =
          _scene->device.backend_name() == "simd" &&
          _scene->device.compute_warp_size() > 1u &&
          ([]() noexcept {
            const char *raw = std::getenv("PSYCLES_SIMD_OUTLINE_BSDF");
            return raw == nullptr || std::string_view{raw} != "0";
          })();
      Float3 eval_sum = make_float3(0.0f);
      Float3 eval_diffuse = make_float3(0.0f);
      Float3 eval_glossy = make_float3(0.0f);
      Float eval_pdf = 0.0f;
      Float eval_avg_r2 = 0.0f;
      UInt eval_events = 0u;
      std::optional<Float4> eval_spec_sum;
      std::optional<Float4> eval_spec_diffuse;
      std::optional<Float4> eval_spec_glossy;
      std::optional<Float4> eval_spec_lane_pdf;
      if (_spectral) {
        eval_spec_sum.emplace(make_float4(0.0f));
        eval_spec_diffuse.emplace(make_float4(0.0f));
        eval_spec_glossy.emplace(make_float4(0.0f));
        if (_kernel_globals.xyz3_rough_dispersion()) {
          eval_spec_lane_pdf.emplace(make_float4(0.0f));
        }
      }
      $outline_when_with_name(outline_bsdf, "simd_svm_bsdf_eval") {
        const auto evaluation = svm_detail::surface_shader_bsdf_eval(
            _kernel_globals, *_shader_data, outgoing, query.shader_flags,
            _closure_types);
        eval_sum = evaluation.sum;
        eval_diffuse = evaluation.diffuse;
        eval_glossy = evaluation.glossy;
        eval_pdf = evaluation.pdf;
        eval_avg_r2 = evaluation.average_roughness_squared;
        if (_spectral) {
          *eval_spec_sum = *evaluation.spectral_sum;
          *eval_spec_diffuse = *evaluation.spectral_diffuse;
          *eval_spec_glossy = *evaluation.spectral_glossy;
          if (eval_spec_lane_pdf.has_value()) {
            *eval_spec_lane_pdf = evaluation.spectral_lane_pdf.value_or(
                make_float4(evaluation.pdf));
          }
        }
        eval_events = evaluation_events(evaluation, *_shader_data, outgoing);
      };
      auto result = SurfaceEvaluation{
          .f = eval_sum,
          .pdf = eval_pdf,
          .diffuse_f = eval_diffuse,
          .glossy_f = eval_glossy,
          .diffuse_pdf = select(
              0.0f, eval_pdf,
              _spectral ? nonzero4(*eval_spec_diffuse) : nonzero(eval_diffuse)),
          .average_roughness_squared = eval_avg_r2,
          .events = eval_events};
      if (_spectral) {
        result.spectral_f = *eval_spec_sum;
        result.spectral_diffuse_f = *eval_spec_diffuse;
        result.spectral_glossy_f = *eval_spec_glossy;
        if (eval_spec_lane_pdf.has_value()) {
          result.spectral_lane_pdf = *eval_spec_lane_pdf;
        }
      }
      return result;
    }

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        Expr<std::uint32_t> requested_index,
        const SurfaceQuery &) const noexcept override {
      const auto valid = requested_index < _closures->count();
      auto result = SurfaceClosureTrace{
          .count = _closures->count(),
          .runtime_flags = runtime_flags(_shader_data->flag),
          .index = requested_index,
          .type = 0u,
          .sample_weight = 0.0f,
          .weight = make_float3(0.0f),
          .normal = make_float3(0.0f, 0.0f, 1.0f),
          .valid = false};
      $if(valid) {
        const auto common = _closures->common(requested_index);
        result.type = common.type;
        result.weight = common.weight;
        // closure_alloc defines only type/weight. Holdout and invalidated
        // records must not expose an undefined sample weight or normal just
        // for diagnostics. Portal belongs to the native BSDF interval;
        // volume scatter defines sample_weight but has phase payload, not N.
        $if(closure::is_bsdf_or_bssrdf(common.type)) {
          const auto scattering = _closures->common(requested_index);
          result.sample_weight = scattering.sample_weight;
          result.normal = scattering.N;
        }
        $elif((common.type >= unsigned(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) &
              (common.type <= unsigned(abi::CLOSURE_VOLUME_DRAINE_ID))) {
          result.sample_weight = _closures->volume_common(requested_index).sample_weight;
        };
        result.valid = true;
      };
      return result;
    }

    [[nodiscard]] SurfaceSample sample(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &) const noexcept override {
      return sample_impl(u_lobe, u_direction).sample;
    }

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &) const noexcept override {
      return sample_impl(u_lobe, u_direction);
    }
};

class CyclesSvmSurfacePopulationComponent final
    : public SurfacePopulationComponent {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    CyclesSvmSurfaceBucketSpecialization _bucket;

  public:
    CyclesSvmSurfacePopulationComponent(
        std::shared_ptr<LuisaSceneData> scene,
        CyclesSvmSurfaceBucketSpecialization bucket) noexcept
        : _scene{std::move(scene)}, _bucket{std::move(bucket)} {}

    [[nodiscard]] std::shared_ptr<PopulatedSurfaceShader> populate(
        const SurfacePopulationContext &context) const noexcept override {
      return std::make_shared<CyclesSvmPopulatedSurface>(
          _scene, context, _bucket);
    }
};

} // namespace

std::shared_ptr<const SurfacePopulationComponent>
make_cycles_svm_surface_population_component(
    const std::shared_ptr<LuisaSceneData> &scene,
    CyclesSvmSurfaceBucketSpecialization bucket) noexcept {
  return std::make_shared<CyclesSvmSurfacePopulationComponent>(
      scene, std::move(bucket));
}

SurfaceCallables make_surface_callables(
    const std::shared_ptr<LuisaSceneData> &scene,
    bool /*spectral_rendering*/,
    contract::SpectralMode /*spectral_mode*/) noexcept {
  return {make_cycles_svm_surface_population_component(scene)};
}

} // namespace psycles::luisa_backend::detail
