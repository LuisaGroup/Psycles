#include "path_tracer_surfaces.h"

#include "path_tracer_cycles_svm_surface.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_ambient_occlusion.h"
#include "path_tracer_surface_closure_evaluation.h"
#include "path_tracer_surface_closure_sampling.h"
#include "path_tracer_surface_closure_setup.h"
#include "subsurface_exit_closure_component.h"

#include <psycles/luisa/surface_closure_operations.h>
#include <psycles/luisa/surface_closure_population.h>
#include <psycles/luisa/surface_closure_evaluator.h>
#include <psycles/luisa/surface_closure_set.h>

#include <utility>
#include <vector>

namespace psycles::luisa_backend::detail {
namespace {

using SceneSurfaceShaderServices = BufferShaderServices<
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>,
    BindlessArray, BindlessArray>;

class ExpandedSurfacePopulationProgram final
    : public SurfacePopulationProgram {

  private:
    std::shared_ptr<LuisaSceneData> _scene;

  public:
    explicit ExpandedSurfacePopulationProgram(
        std::shared_ptr<LuisaSceneData> scene) noexcept
        : _scene{std::move(scene)} {}

    [[nodiscard]] SurfacePopulation populate(
        Expr<std::uint32_t> surface_tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector,
        const PathSurfaceAmbientOcclusionContext *) const noexcept override {
        return _scene->surfaces.populate(
            surface_tag, services, point, query, collector);
    }
};

class PopulatedSurfaceShaderImpl final
    : public PopulatedSurfaceShader {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<const SurfacePopulationProgram> _program;
    SurfacePoint _point;
    SurfaceClosureSetupCallables _closure_setup;
    Texture2DSamplingCallables _texture_sampling;
    SurfaceAttributeLookupCallable _attribute_lookup;
    CallableSurfaceClosureSetupProvider _setup_provider;
    CallableTexture2DSamplingProvider _texture_provider;
    CallableSurfaceAttributeLookupProvider _attribute_provider;
    std::shared_ptr<PathSurfaceAmbientOcclusionProvider>
        _ambient_occlusion_provider;
    SceneSurfaceShaderServices _services;
    SurfaceClosurePopulationCollector _population;
    SurfacePreparation _preparation;
    Float3 _shading_normal{make_float3(0.0f, 0.0f, 1.0f)};
    std::unique_ptr<SurfaceClosureEvaluator> _evaluator;

  public:
    PopulatedSurfaceShaderImpl(
        std::shared_ptr<LuisaSceneData> scene,
        std::shared_ptr<const SurfacePopulationProgram> program,
        const SurfaceClosureSetupCallables &closure_setup,
        const Texture2DSamplingCallables &texture_sampling,
        const SurfaceAttributeLookupCallable &attribute_lookup,
        Expr<std::uint32_t> surface_tag,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        const SurfaceClosureIdentityCallable &identity,
        const SurfaceClosureAovCallable &aov_operation,
        std::shared_ptr<const SceneTraversalComponent>
            ambient_occlusion_traversal,
        const PathSurfaceAmbientOcclusionContext
            *ambient_occlusion) noexcept
        : _scene{std::move(scene)},
          _program{std::move(program)},
          _point{point},
          _closure_setup{closure_setup},
          _texture_sampling{texture_sampling},
          _attribute_lookup{attribute_lookup},
          _setup_provider{
              _scene->cycles_bsdf_table_buffer,
              _closure_setup},
          _texture_provider{
              _scene->texture_heap,
              _texture_sampling},
          _attribute_provider{
              _scene->heap,
              _attribute_lookup},
          _ambient_occlusion_provider{
              ambient_occlusion != nullptr
                  ? std::make_shared<PathSurfaceAmbientOcclusionProvider>(
                        _scene,
                        std::move(ambient_occlusion_traversal),
                        *ambient_occlusion)
                  : nullptr},
          _services{
              _scene->scalar_parameter_buffer,
              _scene->vector_parameter_buffer,
              _scene->cycles_bsdf_table_buffer,
              _scene->texture_heap,
              _scene->heap,
              _scene->attribute_binding_slot,
              _scene->attribute_range_slot,
              _scene->nishita_texture_bindings,
              _scene->shader_color_space,
              &_setup_provider,
              &_texture_provider,
              &_attribute_provider,
              _ambient_occlusion_provider.get()},
          _population{
              _point,
              _scene->volume_metadata.closure_allocation_budget,
              query,
              identity,
              aov_operation},
          _preparation{SurfacePreparation::zero(_point)} {
        const auto population = _program->populate(
            surface_tag, _services, _point, query, _population,
            ambient_occlusion);
        _preparation = _population.preparation(population.emission);
        _shading_normal = population.shading_normal;
        const auto reachability =
            _scene->surface_values
                ? _scene->surface_values->physical_closure_reachability
                : all_surface_closure_reachability;
        _evaluator = std::make_unique<SurfaceClosureEvaluator>(
            _point,
            _population.closures(),
            _shading_normal,
            _population.runtime_state(),
            reachability);
    }

    [[nodiscard]] Expr<std::uint32_t>
    closure_count() const noexcept override {
        return _population.closures().count();
    }

    [[nodiscard]] SurfacePreparation preparation()
        const noexcept override {
        return _preparation;
    }

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept override {
        auto result = _evaluator->evaluate_light(
            _services, outgoing, query);
        $if(query.surface.subsurface_exit) {
            result = SubsurfaceExitClosureComponent{}.evaluate_light(
                _point, outgoing, query.surface, query.shader_flags);
        };
        return result;
    }

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        Expr<std::uint32_t> requested_index,
        const SurfaceQuery &query) const noexcept override {
        auto result = _evaluator->closure_trace(
            UInt{requested_index});
        $if(query.subsurface_exit) {
            result = SubsurfaceExitClosureComponent{}.trace(
                _point, query, requested_index);
        };
        return result;
    }

    [[nodiscard]] SurfaceSample sample(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept override {
        auto result = _evaluator->sample(
            _services, u_lobe, u_direction, query);
        $if(query.subsurface_exit) {
            result = SubsurfaceExitClosureComponent{}.sample(
                _point, u_direction, query);
        };
        return result;
    }

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept override {
        auto result = _evaluator->sample_trace(
            _services, u_lobe, u_direction, query);
        $if(query.subsurface_exit) {
            result = SubsurfaceExitClosureComponent{}.sample_trace(
                _point, u_direction, query);
        };
        return result;
    }
};

class SurfacePopulationComponentImpl final
    : public SurfacePopulationComponent {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<const SurfacePopulationProgram> _program;
    SurfaceClosureIdentityCallable _identity;
    SurfaceClosureAovCallable _aov_operation;
    SurfaceClosureSetupCallables _closure_setup;
    Texture2DSamplingCallables _texture_sampling;
    SurfaceAttributeLookupCallable _attribute_lookup;
    std::shared_ptr<const SceneTraversalComponent>
        _ambient_occlusion_traversal;

  public:
    explicit SurfacePopulationComponentImpl(
        std::shared_ptr<LuisaSceneData> scene,
        std::shared_ptr<const SurfacePopulationProgram> program,
        const SurfaceClosureIdentityCallable &identity,
        const SurfaceClosureAovCallable &aov_operation,
        const SurfaceClosureSetupCallables &closure_setup,
        const Texture2DSamplingCallables &texture_sampling,
        const SurfaceAttributeLookupCallable &attribute_lookup) noexcept
        : _scene{std::move(scene)},
          _program{std::move(program)},
          _identity{identity},
          _aov_operation{aov_operation},
          _closure_setup{closure_setup},
          _texture_sampling{texture_sampling},
          _attribute_lookup{attribute_lookup},
          _ambient_occlusion_traversal{
              _scene->has_ambient_occlusion
                  ? make_scene_traversal_component(
                        make_scene_traversal_stage_plan(
                            _scene->geometries.size(),
                            _scene->curve_geometries.size()))
                  : nullptr} {}

    [[nodiscard]] std::shared_ptr<PopulatedSurfaceShader> populate(
        const SurfacePopulationContext &context) const noexcept override {
        const auto *ambient_occlusion = context.ambient_occlusion;
        if (_scene->has_ambient_occlusion &&
            ambient_occlusion == nullptr) {
            std::abort();
        }
        return std::make_shared<PopulatedSurfaceShaderImpl>(
            _scene,
            _program,
            _closure_setup,
            _texture_sampling,
            _attribute_lookup,
            context.surface_tag,
            context.point,
            context.query,
            _identity,
            _aov_operation,
            _ambient_occlusion_traversal,
            ambient_occlusion);
    }
};

using SurfacePreparationImplementationCallable =
    Callable<SurfacePreparationCall(
        Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
        BindlessArray, SurfacePointCall, SurfacePreparationQueryCall)>;

using SurfaceEvaluateLightImplementationCallable =
    Callable<SurfaceEvaluationCall(
        Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
        BindlessArray, SurfacePointCall, luisa::float3, luisa::uint,
        luisa::uint, float, bool, bool, luisa::uint)>;

using SurfaceSampleImplementationCallable = Callable<SurfaceSampleCall(
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
    BindlessArray, SurfacePointCall, float, luisa::float2, luisa::uint,
    luisa::uint, float, bool, bool)>;

// For every valid runtime tag t this records exactly
// `result = implementations[t](args...)`. Unlike placing the material graph
// directly in each switch arm, the graph's temporaries are owned by the typed
// leaf callable and cannot become live at the dispatch merge. The single-tag
// case deliberately matches Polymorphic<T>: its only implementation is used
// without inspecting the tag.
template<typename Result, typename Callable, typename Invoke,
         typename EmptyResult>
[[nodiscard]] Var<Result> dispatch_surface_implementation(
    UInt tag, const std::vector<Callable> &implementations, Invoke &&invoke,
    EmptyResult &&empty_result) noexcept {
    if (implementations.empty()) {
        return empty_result();
    }
    if (implementations.size() == 1u) {
        return invoke(implementations.front());
    }
    Var<Result> result;
    luisa::compute::detail::SwitchStmtBuilder{tag} % [&] {
        for (auto index = std::size_t{0u}; index < implementations.size();
             index++) {
            luisa::compute::detail::SwitchCaseStmtBuilder{
                static_cast<luisa::uint>(index)} %
                [&, index] { result = invoke(implementations[index]); };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
            luisa::compute::dsl::unreachable("invalid surface topology tag");
        };
    };
    return result;
}

[[nodiscard]] SurfacePreparationCallable
make_expanded_surface_preparation_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SurfaceClosureSetupCallables &closure_setup,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup) noexcept {
    std::vector<SurfacePreparationImplementationCallable>
        implementations;
    implementations.reserve(scene->surfaces.size());
    for (auto surface_index = std::size_t{0u};
         surface_index < scene->surfaces.size(); ++surface_index) {
        const auto *surface =
            scene->surfaces.implementation(surface_index);
        SurfacePreparationImplementationCallable implementation =
            [scene, surface, closure_setup, texture_sampling, attribute_lookup](
                BufferFloat scalar_parameters,
                BufferFloat3 vector_parameters,
                BufferFloat cycles_bsdf_tables,
                BindlessVar textures,
                BindlessVar geometry_heap,
                Var<SurfacePointCall> packed_point,
                Var<SurfacePreparationQueryCall> packed_query) noexcept {
                CallableSurfaceClosureSetupProvider setup_provider{
                    cycles_bsdf_tables, closure_setup};
                CallableTexture2DSamplingProvider texture_provider{
                    textures, texture_sampling};
                CallableSurfaceAttributeLookupProvider attribute_provider{
                    geometry_heap, attribute_lookup};
                BufferShaderServices services{
                    scalar_parameters,
                    vector_parameters,
                    cycles_bsdf_tables,
                    textures,
                    geometry_heap,
                    scene->attribute_binding_slot,
                    scene->attribute_range_slot,
                    scene->nishita_texture_bindings,
                    scene->shader_color_space,
                    &setup_provider,
                    &texture_provider,
                    &attribute_provider};
                return pack_surface_preparation(
                    surface->prepare(
                        services,
                        unpack_surface_point(packed_point),
                        unpack_surface_preparation_query(packed_query)));
            };
        implementation.set_name(
            luisa::format(
                "surface_prepare_topology_{}", surface_index));
        implementations.emplace_back(std::move(implementation));
    }
    SurfacePreparationCallable result =
        [implementations = std::move(implementations)](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Var<SurfacePreparationQueryCall> packed_query) noexcept {
            const auto point = unpack_surface_point(packed_point);
            return dispatch_surface_implementation<
                SurfacePreparationCall>(
                surface_tag,
                implementations,
                [&](const auto &implementation) noexcept {
                    return implementation(
                        scalar_parameters,
                        vector_parameters,
                        cycles_bsdf_tables,
                        textures,
                        geometry_heap,
                        packed_point,
                        packed_query);
                },
                [&]() noexcept {
                    return pack_surface_preparation(
                        SurfacePreparation::zero(point));
                });
        };
    result.set_name("surface_prepare");
    return result;
}

[[nodiscard]] SurfaceBssrdfNormalCallable
make_expanded_surface_bssrdf_normal_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SurfaceClosureSetupCallables &closure_setup,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup) noexcept {
    SurfaceBssrdfNormalCallable result =
        [scene, closure_setup, texture_sampling, attribute_lookup](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point, Bool has_bssrdf_bump,
            Bool reflective_caustics, Bool refractive_caustics) noexcept {
            CallableSurfaceClosureSetupProvider setup_provider{cycles_bsdf_tables,
                                                               closure_setup};
            CallableTexture2DSamplingProvider texture_provider{textures,
                                                               texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{scalar_parameters,
                                          vector_parameters,
                                          cycles_bsdf_tables,
                                          textures,
                                          geometry_heap,
                                          scene->attribute_binding_slot,
                                          scene->attribute_range_slot,
                                          scene->nishita_texture_bindings,
                                          scene->shader_color_space,
                                          &setup_provider,
                                          &texture_provider,
                                          &attribute_provider};
            const auto point = unpack_surface_point(packed_point);
            SurfaceBssrdfNormalVisitor visitor{
                scene->volume_metadata.closure_allocation_budget};
            $if (has_bssrdf_bump) {
                static_cast<void>(scene->surfaces.collect_bssrdf_bump_closures(
                    surface_tag, scene->surface_bssrdf_bump_tags, services, point,
                    reflective_caustics, refractive_caustics, visitor));
            }
            $else {
                visitor.begin(point.shading_normal);
                visitor.finish();
            };
            return Float3{visitor.result()};
        };
    result.set_name("surface_bssrdf_normal");
    return result;
}

[[nodiscard]] SurfaceEmissionCallable make_expanded_surface_emission_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup) noexcept {
  SurfaceEmissionCallable result =
      [scene, texture_sampling, attribute_lookup](
          BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
          BufferFloat cycles_bsdf_tables, BindlessVar textures,
          BindlessVar geometry_heap, UInt surface_tag,
          Var<SurfacePointCall> packed_point, Float3 outgoing,
          Bool reflective_caustics) noexcept {
        CallableTexture2DSamplingProvider texture_provider{textures,
                                                           texture_sampling};
        CallableSurfaceAttributeLookupProvider attribute_provider{
            geometry_heap, attribute_lookup};
        BufferShaderServices services{scalar_parameters,
                                      vector_parameters,
                                      cycles_bsdf_tables,
                                      textures,
                                      geometry_heap,
                                      scene->attribute_binding_slot,
                                      scene->attribute_range_slot,
                                      scene->nishita_texture_bindings,
                                      scene->shader_color_space,
                                      nullptr,
                                      &texture_provider,
                                      &attribute_provider};
        return scene->surfaces.emission(surface_tag, services,
                                        unpack_surface_point(packed_point),
                                        outgoing, reflective_caustics);
      };
  result.set_name("surface_emission");
  return result;
}

[[nodiscard]] SurfaceConstantEmissionCallable
make_surface_constant_emission_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
  SurfaceConstantEmissionCallable result =
      [scene](BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
              UInt surface_tag, UInt parameter_block) noexcept {
        BufferSurfaceParameterServices services{scalar_parameters,
                                                vector_parameters};
        return scene->surfaces.constant_emission(surface_tag, services,
                                                 parameter_block);
      };
  result.set_name("surface_constant_emission");
  return result;
}

struct SurfaceReplayCallables {
  SurfacePreparationCallable preparation;
  SurfaceEvaluateLightCallable evaluate_light;
  SurfaceSampleCallable sample;
  SurfaceClosureTraceCallable closure_trace;
  SurfaceSampleTraceCallable sample_trace;
};

// These roots make the invariant executable: once the host/JIT selected a
// populated surface, no downstream shader consumer may replay the graph. The
// roots remain in SurfaceCallables only to keep the diagnostic ABI uniform;
// they contain neither a topology dispatcher nor material-node callables.
[[nodiscard]] SurfaceReplayCallables
make_unreachable_surface_replay_callables() noexcept {
  SurfacePreparationCallable preparation =
      [](BufferFloat, BufferFloat3, BufferFloat, BindlessVar, BindlessVar, UInt,
         Var<SurfacePointCall> packed_point,
         Var<SurfacePreparationQueryCall>) noexcept {
        luisa::compute::dsl::unreachable(
            "surface preparation replay after population");
        return pack_surface_preparation(
            SurfacePreparation::zero(unpack_surface_point(packed_point)));
      };
  preparation.set_name("surface_prepare_after_population_unreachable");

  SurfaceEvaluateLightCallable evaluate_light =
      [](BufferFloat, BufferFloat3, BufferFloat, BindlessVar, BindlessVar, UInt,
         Var<SurfacePointCall>, Float3, UInt, UInt, Float, Bool, Bool,
         UInt) noexcept {
        luisa::compute::dsl::unreachable(
            "surface evaluation replay after population");
        return pack_surface_evaluation(SurfaceEvaluation::zero());
      };
  evaluate_light.set_name(
      "surface_evaluate_light_after_population_unreachable");

  SurfaceSampleCallable sample = [](BufferFloat, BufferFloat3, BufferFloat,
                                    BindlessVar, BindlessVar, UInt,
                                    Var<SurfacePointCall>, Float, Float2, UInt,
                                    UInt, Float, Bool, Bool) noexcept {
    luisa::compute::dsl::unreachable(
        "surface sampling replay after population");
    return pack_surface_sample(SurfaceSample::zero());
  };
  sample.set_name("surface_sample_after_population_unreachable");

  SurfaceClosureTraceCallable closure_trace =
      [](BufferFloat, BufferFloat3, BufferFloat, BindlessVar, BindlessVar, UInt,
         Var<SurfacePointCall>, UInt requested_index, Bool, Bool) noexcept {
        luisa::compute::dsl::unreachable(
            "surface closure trace replay after population");
        return pack_surface_closure_trace(
            SurfaceClosureTrace::zero(requested_index));
      };
  closure_trace.set_name("surface_closure_trace_after_population_unreachable");

  SurfaceSampleTraceCallable sample_trace =
      [](BufferFloat, BufferFloat3, BufferFloat, BindlessVar, BindlessVar, UInt,
         Var<SurfacePointCall>, Float, Float2, UInt, UInt, Float, Bool,
         Bool) noexcept {
        luisa::compute::dsl::unreachable(
            "surface sample trace replay after population");
        return pack_surface_sample_trace(SurfaceSampleTrace::zero());
      };
  sample_trace.set_name("surface_sample_trace_after_population_unreachable");

  return {std::move(preparation), std::move(evaluate_light), std::move(sample),
          std::move(closure_trace), std::move(sample_trace)};
}

}// namespace

SurfaceCallables
make_surface_callables(const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    const auto closure_identity = make_surface_closure_identity_callable();
    const auto closure_setup = make_surface_closure_setup_callables();
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup = make_surface_attribute_lookup_callable(
        scene->attribute_binding_slot, scene->attribute_range_slot);
    std::shared_ptr<const SurfacePopulationComponent> population;
    if (scene->native_cycles_svm_surface) {
        population = make_cycles_svm_surface_population_component(scene);
    } else if (scene->populate_surface_once) {
        const auto closure_aov = make_surface_closure_aov_callable();
        std::shared_ptr<const SurfacePopulationProgram> program;
        if (scene->surface_values) {
            program = make_compact_surface_population_program(scene);
        } else {
            program = std::make_shared<
                ExpandedSurfacePopulationProgram>(scene);
        }
        population = std::make_shared<SurfacePopulationComponentImpl>(
            scene,
            std::move(program),
            closure_identity,
            closure_aov,
            closure_setup,
            texture_sampling,
            attribute_lookup);
    }
  auto emission = scene->surface_values
                      ? make_compact_surface_emission_callable(scene)
                      : make_expanded_surface_emission_callable(
                            scene, texture_sampling, attribute_lookup);
  auto constant_emission = make_surface_constant_emission_callable(scene);
  auto bssrdf_normal =
      scene->surface_values
          ? make_compact_surface_bssrdf_normal_callable(scene)
          : make_expanded_surface_bssrdf_normal_callable(
                scene, closure_setup, texture_sampling, attribute_lookup);

  // This is a host/JIT branch, not a device scheduler predicate. In the
  // Cycles-style route all surface consumers are emitted against the one
  // PopulatedSurfaceShader object, so constructing topology-expanded replay
  // callables would add dead scene-sized ASTs and weaken that invariant.
  if (population) {
    auto replay = make_unreachable_surface_replay_callables();
    return {std::move(population),
            std::move(replay.preparation),
            std::move(replay.evaluate_light),
            std::move(constant_emission),
            std::move(emission),
            std::move(replay.sample),
            std::move(replay.closure_trace),
            std::move(replay.sample_trace),
            std::move(bssrdf_normal)};
  }

    const auto closure_evaluation =
        make_surface_closure_evaluation_callable(scene);
    const auto closure_sampling = make_surface_closure_sampling_callables(scene);
    auto preparation = scene->surface_values
                           ? make_compact_surface_preparation_callable(scene)
                           : make_expanded_surface_preparation_callable(
                                 scene,
                                 closure_setup,
                                 texture_sampling,
                                 attribute_lookup);
    std::vector<SurfaceEvaluateLightImplementationCallable>
        evaluate_light_implementations;
    evaluate_light_implementations.reserve(scene->surfaces.size());
    for (auto surface_index = std::size_t{0u};
         surface_index < scene->surfaces.size(); surface_index++) {
        const auto *surface = scene->surfaces.implementation(surface_index);
        SurfaceEvaluateLightImplementationCallable implementation =
            [scene, surface, closure_evaluation, closure_setup, texture_sampling,
             attribute_lookup](
                BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
                BufferFloat cycles_bsdf_tables, BindlessVar textures,
                BindlessVar geometry_heap, Var<SurfacePointCall> packed_point,
                Float3 outgoing, UInt lobe_mask, UInt transport_mode,
                Float glossy_filter_roughness, Bool reflective_caustics,
                Bool refractive_caustics, UInt shader_flags) noexcept {
                CallableSurfaceClosureSetupProvider setup_provider{cycles_bsdf_tables,
                                                                   closure_setup};
                CallableTexture2DSamplingProvider texture_provider{textures,
                                                                   texture_sampling};
                CallableSurfaceAttributeLookupProvider attribute_provider{
                    geometry_heap, attribute_lookup};
                BufferShaderServices services{scalar_parameters,
                                              vector_parameters,
                                              cycles_bsdf_tables,
                                              textures,
                                              geometry_heap,
                                              scene->attribute_binding_slot,
                                              scene->attribute_range_slot,
                                              scene->nishita_texture_bindings,
                                              scene->shader_color_space,
                                              &setup_provider,
                                              &texture_provider,
                                              &attribute_provider};
                auto query = SurfaceLightQuery{
                    .surface = {.lobe_mask = lobe_mask,
                                .transport_mode = transport_mode,
                                .glossy_filter_roughness = glossy_filter_roughness,
                                .reflective_caustics = reflective_caustics,
                                .refractive_caustics = refractive_caustics},
                    .shader_flags = shader_flags};
                const auto point = unpack_surface_point(packed_point);
                const SurfaceClosurePoint closure_point{point};
                const auto packed_closure_point =
                    pack_surface_closure_point(closure_point);
                const auto policy = make_surface_closure_evaluation_policy(
                    true, Expr<std::uint32_t>{shader_flags.expression()});
                CallableSurfaceClosureEvaluationOperation operation{
                    scalar_parameters, vector_parameters, cycles_bsdf_tables,
                    textures, geometry_heap, packed_closure_point,
                    closure_point, query.surface, policy,
                    closure_evaluation};
                operation.set_outgoing(Expr<luisa::float3>{outgoing.expression()});
                SurfaceClosureEvaluationVisitor visitor{
                    scene->volume_metadata.closure_allocation_budget, operation,
                    Expr<bool>{policy.preserve_pdf.expression()}};
                static_cast<void>(
                    surface->collect_closures(services, point, reflective_caustics,
                                              refractive_caustics, visitor));
                return pack_surface_evaluation(visitor.result());
            };
        implementation.set_name(
            luisa::format("surface_evaluate_light_topology_{}", surface_index));
        evaluate_light_implementations.emplace_back(std::move(implementation));
    }
    SurfaceEvaluateLightCallable evaluate_light =
        [evaluate_light_implementations =
             std::move(evaluate_light_implementations)](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point, Float3 outgoing, UInt lobe_mask,
            UInt transport_mode, Float glossy_filter_roughness,
            Bool reflective_caustics, Bool refractive_caustics,
            UInt shader_flags) noexcept {
            return dispatch_surface_implementation<SurfaceEvaluationCall>(
                surface_tag, evaluate_light_implementations,
                [&](const auto &implementation) noexcept {
                    return implementation(
                        scalar_parameters, vector_parameters, cycles_bsdf_tables,
                        textures, geometry_heap, packed_point, outgoing, lobe_mask,
                        transport_mode, glossy_filter_roughness, reflective_caustics,
                        refractive_caustics, shader_flags);
                },
                []() noexcept {
                    return pack_surface_evaluation(SurfaceEvaluation::zero());
                });
        };
    evaluate_light.set_name("surface_evaluate_light");
    std::vector<SurfaceSampleImplementationCallable> sample_implementations;
    sample_implementations.reserve(scene->surfaces.size());
    for (auto surface_index = std::size_t{0u};
         surface_index < scene->surfaces.size(); surface_index++) {
        const auto *surface = scene->surfaces.implementation(surface_index);
        SurfaceSampleImplementationCallable implementation =
            [scene, surface, closure_sampling, closure_evaluation, closure_setup,
             texture_sampling, attribute_lookup](
                BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
                BufferFloat cycles_bsdf_tables, BindlessVar textures,
                BindlessVar geometry_heap, Var<SurfacePointCall> packed_point,
                Float u_lobe, Float2 u_direction, UInt lobe_mask,
                UInt transport_mode, Float glossy_filter_roughness,
                Bool reflective_caustics, Bool refractive_caustics) noexcept {
                CallableSurfaceClosureSetupProvider setup_provider{cycles_bsdf_tables,
                                                                   closure_setup};
                CallableTexture2DSamplingProvider texture_provider{textures,
                                                                   texture_sampling};
                CallableSurfaceAttributeLookupProvider attribute_provider{
                    geometry_heap, attribute_lookup};
                BufferShaderServices services{scalar_parameters,
                                              vector_parameters,
                                              cycles_bsdf_tables,
                                              textures,
                                              geometry_heap,
                                              scene->attribute_binding_slot,
                                              scene->attribute_range_slot,
                                              scene->nishita_texture_bindings,
                                              scene->shader_color_space,
                                              &setup_provider,
                                              &texture_provider,
                                              &attribute_provider};
                auto query =
                    SurfaceQuery{.lobe_mask = lobe_mask,
                                 .transport_mode = transport_mode,
                                 .glossy_filter_roughness = glossy_filter_roughness,
                                 .reflective_caustics = reflective_caustics,
                                 .refractive_caustics = refractive_caustics};
                const auto point = unpack_surface_point(packed_point);
                return pack_surface_sample(
                    sample_surface_closures_for_surface(
                        *scene, *surface, closure_sampling, closure_evaluation,
                        scalar_parameters, vector_parameters, cycles_bsdf_tables,
                        textures, geometry_heap, services, point,
                        Expr<float>{u_lobe.expression()},
                        Expr<luisa::float2>{u_direction.expression()}, query, false)
                        .sample);
            };
        implementation.set_name(
            luisa::format("surface_sample_topology_{}", surface_index));
        sample_implementations.emplace_back(std::move(implementation));
    }
    SurfaceSampleCallable sample =
        [sample_implementations = std::move(sample_implementations)](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point, Float u_lobe, Float2 u_direction,
            UInt lobe_mask, UInt transport_mode, Float glossy_filter_roughness,
            Bool reflective_caustics, Bool refractive_caustics) noexcept {
            return dispatch_surface_implementation<SurfaceSampleCall>(
                surface_tag, sample_implementations,
                [&](const auto &implementation) noexcept {
                    return implementation(
                        scalar_parameters, vector_parameters, cycles_bsdf_tables,
                        textures, geometry_heap, packed_point, u_lobe, u_direction,
                        lobe_mask, transport_mode, glossy_filter_roughness,
                        reflective_caustics, refractive_caustics);
                },
                []() noexcept {
                    return pack_surface_sample(SurfaceSample::zero());
                });
        };
    sample.set_name("surface_sample");
    SurfaceClosureTraceCallable closure_trace =
        [scene, closure_identity, closure_setup, texture_sampling,
         attribute_lookup](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point, UInt requested_index,
            Bool reflective_caustics, Bool refractive_caustics) noexcept {
            CallableSurfaceClosureSetupProvider setup_provider{cycles_bsdf_tables,
                                                               closure_setup};
            CallableTexture2DSamplingProvider texture_provider{textures,
                                                               texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{scalar_parameters,
                                          vector_parameters,
                                          cycles_bsdf_tables,
                                          textures,
                                          geometry_heap,
                                          scene->attribute_binding_slot,
                                          scene->attribute_range_slot,
                                          scene->nishita_texture_bindings,
                                          scene->shader_color_space,
                                          &setup_provider,
                                          &texture_provider,
                                          &attribute_provider};
            const auto point = unpack_surface_point(packed_point);
            SurfaceClosureTraceVisitor visitor{
                point, requested_index,
                scene->volume_metadata.closure_allocation_budget, closure_identity};
            static_cast<void>(scene->surfaces.collect_closures(
                surface_tag, services, point, reflective_caustics,
                refractive_caustics, visitor));
            return pack_surface_closure_trace(visitor.result());
        };
    closure_trace.set_name("surface_closure_trace");
    SurfaceSampleTraceCallable sample_trace =
        [scene, closure_sampling, closure_evaluation, closure_setup,
         texture_sampling, attribute_lookup](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point, Float u_lobe, Float2 u_direction,
            UInt lobe_mask, UInt transport_mode, Float glossy_filter_roughness,
            Bool reflective_caustics, Bool refractive_caustics) noexcept {
            CallableSurfaceClosureSetupProvider setup_provider{cycles_bsdf_tables,
                                                               closure_setup};
            CallableTexture2DSamplingProvider texture_provider{textures,
                                                               texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{scalar_parameters,
                                          vector_parameters,
                                          cycles_bsdf_tables,
                                          textures,
                                          geometry_heap,
                                          scene->attribute_binding_slot,
                                          scene->attribute_range_slot,
                                          scene->nishita_texture_bindings,
                                          scene->shader_color_space,
                                          &setup_provider,
                                          &texture_provider,
                                          &attribute_provider};
            auto query =
                SurfaceQuery{.lobe_mask = lobe_mask,
                             .transport_mode = transport_mode,
                             .glossy_filter_roughness = glossy_filter_roughness,
                             .reflective_caustics = reflective_caustics,
                             .refractive_caustics = refractive_caustics};
            const auto point = unpack_surface_point(packed_point);
            return pack_surface_sample_trace(sample_surface_closures(
                *scene, closure_sampling, closure_evaluation, scalar_parameters,
                vector_parameters, cycles_bsdf_tables, textures, geometry_heap,
                Expr<std::uint32_t>{surface_tag.expression()}, services, point,
                Expr<float>{u_lobe.expression()},
                Expr<luisa::float2>{u_direction.expression()}, query, true));
        };
    sample_trace.set_name("surface_sample_trace");
    return {std::move(population),
            std::move(preparation),
            std::move(evaluate_light),
            std::move(constant_emission),
            std::move(emission),
            std::move(sample),
            std::move(closure_trace),
            std::move(sample_trace),
            std::move(bssrdf_normal)};
}

}// namespace psycles::luisa_backend::detail
