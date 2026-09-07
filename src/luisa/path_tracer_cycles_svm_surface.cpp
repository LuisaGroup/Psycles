#include "path_tracer_cycles_svm_surface.h"

#include "path_tracer_cycles_svm_kernel_globals.h"
#include "cycles_svm_bsdf.h"
#include "cycles_svm_internal.h"
#include "cycles_svm_surface_shader.h"
#include "cycles_svm_subsurface.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_noise.h>
#include <psycles/luisa/cycles_path_state.h>

#include <algorithm>
#include <memory>
#include <utility>

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

[[nodiscard]] UInt evaluation_events(
    const svm_detail::SurfaceShaderBsdfEval &evaluation,
    const svm::ShaderData &shader_data,
    Expr<luisa::float3> outgoing) noexcept {
  const auto has_diffuse = nonzero(evaluation.diffuse);
  const auto has_glossy = nonzero(evaluation.glossy);
  const auto has_scatter = nonzero(evaluation.sum);
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
    SurfacePreparation _preparation;
    Bool _material_evaluated{true};

  private:
    [[nodiscard]] SurfacePreparation make_preparation(
        const SurfacePopulationContext &context) noexcept {
      auto result = SurfacePreparation::zero(_point);
      const auto flags = _shader_data->flag;
      const auto emission_cosine = abs(dot(_shader_data->Ng,
                                           _shader_data->wi));
      result.emission = select(
          make_float3(0.0f),
          _shader_data->closure_emission_background,
          ((flags & static_cast<std::uint32_t>(abi::SD_EMISSION)) != 0u) &
              (emission_cosine > 0.0f));
      result.shading_normal = _shader_data->N;
      result.runtime_flags = select(
          0u, runtime_flags(flags), context.query.include_runtime_flags);
      // Cycles enters its film data-pass closure reductions only for paths
      // which can write those passes. Keep the same runtime boundary here:
      // selecting the outputs afterwards still executes every bsdf_albedo
      // branch and closure reduction on non-AOV bounces.
      result.aov.normal = _shader_data->N;
      $if(context.query.include_aov) {
        Float3 diffuse = make_float3(0.0f);
        Float3 glossy = make_float3(0.0f);
        Float3 transmission = make_float3(0.0f);
        Float3 average_normal = make_float3(0.0f);
        Float roughness = 0.0f;
        Float roughness_weight = 0.0f;
        UInt index = 0u;
        $while(index < _closures->count()) {
          const auto common = _closures->common(index);
          const auto is_bssrdf = closure::is_bssrdf(common.type);
          $if(closure::is_bsdf_diffuse(common.type) | is_bssrdf) {
            diffuse += svm_detail::bsdf_albedo(
                _kernel_globals, *_shader_data, index, true, true,
                _closure_types);
          };
          $if(closure::is_bsdf_glossy(common.type) |
              closure::is_glass(common.type)) {
            glossy += svm_detail::bsdf_albedo(
                _kernel_globals, *_shader_data, index, true, false,
                _closure_types);
          };
          $if(closure::is_bsdf_transmission(common.type) |
              closure::is_glass(common.type)) {
            transmission += svm_detail::bsdf_albedo(
                _kernel_globals, *_shader_data, index, false, true,
                _closure_types);
          };
          $if(closure::is_bsdf_or_bssrdf(common.type)) {
            const auto weight = abs((common.weight.x + common.weight.y +
                                     common.weight.z) /
                                    3.0f);
            average_normal += common.N * weight;
          };
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
          index += 1u;
        };

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
        const auto average_roughness = select(
            1.0f, roughness / roughness_weight, roughness_weight > 0.0f);
        result.aov.albedo = diffuse;
        result.aov.glossy_albedo = glossy;
        result.aov.transmission_albedo = transmission;
        result.aov.roughness = make_float2(average_roughness);
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
      const auto has_subsurface =
          (_scene->cycles_svm->kernel_features &
           svm::kernel_feature_subsurface) != 0u;
      const auto selected =
          (has_subsurface ? closure::is_bsdf_or_bssrdf(common.type)
                          : closure::is_bsdf(common.type)) &
          (common.sample_weight > 0.0f);

      auto trace = SurfaceSampleTrace::zero();
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
          result.evaluation.f =
              svm_detail::surface_shader_bssrdf_sample_weight(
                  *_shader_data, closure_index);
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
      auto result = SurfaceSampleTrace::zero();
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
    CyclesSvmPopulatedSurface(
        std::shared_ptr<LuisaSceneData> scene,
        const SurfacePopulationContext &context) noexcept
        : _scene{std::move(scene)},
          _point{context.point},
          _kernel_globals{_scene, context},
          _closures{std::make_unique<svm::ClosurePool>(
              std::clamp<std::size_t>(
                  _scene->volume_metadata.closure_allocation_budget,
                  1u, svm::maximum_closure_capacity))},
          _closure_types{svm_detail::closure_types_for_kernel_features(
              _scene->cycles_svm->kernel_features)},
          _preparation{SurfacePreparation::zero(_point)} {
      const Expr<Buffer<abi::KernelShader>> shaders{
          *_scene->cycles_svm->kernel_shader_buffer};
      const auto shader = shaders->read(
          context.cycles_surface_shader & svm::shader_mask);
      const auto object_flags =
          _scene->cycles_svm->objects->object_flag_buffer->read(
              context.cycles_object_index);
      auto flags = shader.flags.cast<std::uint32_t>();
      flags |= select(0u, static_cast<std::uint32_t>(abi::SD_BACKFACING),
                      context.point.back_facing);
      const auto lcg_state = cycles_noise::hash_uint3(
          context.rng_hash ^ 0xb4bc3953u,
          context.rng_offset,
          context.sample_index);
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
          context.object_to_world,
          context.world_to_object,
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

      const svm::TransformState transform_state{
          context.parameters.camera_transform,
          context.parameters.camera_inverse_transform,
          context.object_to_world,
          context.world_to_object};
      const svm::PathState path_state{
          cycles_path_state::from_contract_shader_visibility(
              context.point.ray_visibility),
          context.path_flags,
          context.point.ray_depth,
          context.point.transparent_depth,
          context.point.diffuse_depth,
          context.point.glossy_depth,
          context.point.transmission_depth,
          0u};
      const Expr<Buffer<luisa::uint>> words{
          *_scene->cycles_svm->word_buffer};
      const auto evaluate_material = [&] {
        svm_detail::surface_shader_initialize_closures(
            *_closures, path_state.visibility, path_state.flag);
        svm::eval_nodes_assume_valid(
          _kernel_globals,
          words,
          abi::SHADER_TYPE_SURFACE,
          _scene->cycles_svm->kernel_features,
          svm::kernel_feature_node_mask_surface,
          _scene->cycles_svm->compilation.table.node_types_used,
          transform_state,
          *_shader_data,
          path_state,
          std::max<std::size_t>(
              1u,
              _scene->cycles_svm->compilation.table.peak_stack_usage));
      };
      const auto prepare_closures = [&] {
        svm_detail::surface_shader_prepare_closures(
            *_shader_data, context.query.glossy_filter_roughness,
            _closure_types);
      };
      // Cycles integrate_surface: SSS exits without bump never evaluate the
      // material. Bumped exits evaluate it once for the BSSRDF normal, then
      // replace the retained closures before any ordinary BSDF consumer.
      if ((_scene->cycles_svm->kernel_features &
           svm::kernel_feature_subsurface) != 0u) {
        const auto subsurface_exit =
            (path_state.flag & cycles_path_state::flag_subsurface) != 0u;
        _material_evaluated = svm_detail::surface_shader_material_eval_required(
            subsurface_exit, _shader_data->flag);
        $if(_material_evaluated) { evaluate_material(); };
        $if(subsurface_exit) {
          svm_detail::subsurface_shader_data_setup(*_shader_data);
        }
        $else { prepare_closures(); };
      } else {
        evaluate_material();
        prepare_closures();
      }
      _preparation = make_preparation(context);
    }

    [[nodiscard]] Expr<bool> material_evaluated() const noexcept override {
      return _material_evaluated;
    }

    [[nodiscard]] Expr<std::uint32_t>
    native_shader_data_flags() const noexcept override {
      return _shader_data->flag;
    }

    [[nodiscard]] Expr<std::uint32_t>
    closure_count() const noexcept override {
      return _closures->count();
    }

    [[nodiscard]] SurfacePreparation preparation() const noexcept override {
      return _preparation;
    }

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept override {
      const auto evaluation = svm_detail::surface_shader_bsdf_eval(
          _kernel_globals, *_shader_data, outgoing, query.shader_flags,
          _closure_types);
      auto result = SurfaceEvaluation{
          .f = evaluation.sum,
          .pdf = evaluation.pdf,
          .diffuse_f = evaluation.diffuse,
          .glossy_f = evaluation.glossy,
          .diffuse_pdf = select(0.0f, evaluation.pdf,
                                nonzero(evaluation.diffuse)),
          .average_roughness_squared =
              evaluation.average_roughness_squared,
          .events = evaluation_events(evaluation, *_shader_data, outgoing)};
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
        result.sample_weight = common.sample_weight;
        result.weight = common.weight;
        result.normal = common.N;
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

  public:
    explicit CyclesSvmSurfacePopulationComponent(
        std::shared_ptr<LuisaSceneData> scene) noexcept
        : _scene{std::move(scene)} {}

    [[nodiscard]] std::shared_ptr<PopulatedSurfaceShader> populate(
        const SurfacePopulationContext &context) const noexcept override {
      return std::make_shared<CyclesSvmPopulatedSurface>(_scene, context);
    }
};

} // namespace

std::shared_ptr<const SurfacePopulationComponent>
make_cycles_svm_surface_population_component(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
  return std::make_shared<CyclesSvmSurfacePopulationComponent>(scene);
}

} // namespace psycles::luisa_backend::detail
