#include "path_tracer_surface_svm.h"

#include "path_tracer_attribute_lookup.h"
#include "path_tracer_surface_closure_setup.h"
#include "path_tracer_texture_sampling.h"
#include "principled_layer_component.h"
#include "surface_preparation_accumulator.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_operations.h>

#include <cstdlib>
#include <memory>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceSvmExecutionResources surface_svm_resources(
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables,
    Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap) noexcept {
    return SurfaceSvmExecutionResources{
        .scalar_parameters = std::move(scalar_parameters),
        .vector_parameters = std::move(vector_parameters),
        .cycles_bsdf_tables = std::move(cycles_bsdf_tables),
        .textures = std::move(textures),
        .geometry_heap = std::move(geometry_heap)};
}

void accumulate_surface_svm_emission(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedClosure &raw,
    Expr<std::uint32_t> endpoints_expression,
    Expr<bool> reflective_caustics,
    Float3 &emission) noexcept {
    const auto endpoints = UInt{endpoints_expression};
    const auto emission_endpoint =
        (endpoints & compiler::surface_closure_endpoint_bit(
                         compiler::SurfaceClosureEndpoint::emission)) != 0u;
    if (raw.operation == compiler::ClosureOperation::emission) {
        $if(emission_endpoint) { emission += raw.weight; };
        return;
    }
    if (raw.operation == compiler::ClosureOperation::principled &&
        (raw.principled_features &
         compiler::principled_closure_feature_bit(
             compiler::PrincipledClosureFeature::emission)) != 0u) {
        const auto contribution =
            PrincipledLayerComponent{services, point}
                .evaluate_emission(
                    raw, raw.principled_features, reflective_caustics)
                .radiance;
        $if(emission_endpoint) { emission += contribution; };
    }
}

template<typename Sink>
void visit_surface_svm_physical_leaf(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedClosure &raw,
    Expr<std::uint32_t> endpoints_expression,
    Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics,
    Sink &&sink) noexcept {
    const auto endpoints = UInt{endpoints_expression};
    const auto physical_endpoint =
        (endpoints & compiler::surface_closure_endpoint_bit(
                         compiler::SurfaceClosureEndpoint::physical)) != 0u;
    $if(physical_endpoint) {
        expand_physical_surface_closure(
            services, point, raw, reflective_caustics, refractive_caustics,
            std::forward<Sink>(sink));
    };
}

[[nodiscard]] SurfaceClosureRecord make_merged_transparent_closure(
    Expr<luisa::float3> shading_normal,
    Float3 weight,
    Float sample_weight) noexcept {
    auto closure = TracedClosure{
        .operation = compiler::ClosureOperation::transparent,
        .weight = weight,
        .allocation_weight = sample_weight,
        .sample_weight = sample_weight,
        .setup_valid = true,
        .albedo = weight,
        .color = make_float3(1.0f),
        .normal = Float3{shading_normal},
        .roughness = 0.0f,
        .ior = 1.0f,
        .evaluation_scale = make_float3(1.0f)};
    set_cycles_closure_identity_after_setup(
        closure, cycles_closure::type_transparent);
    return canonical_surface_closure(closure);
}

class SurfaceSvmEmissionConsumer final : public SurfaceSvmClosureConsumer {

  private:
    const ShaderServices &_services;
    Bool _reflective_caustics;
    Float3 _emission{make_float3(0.0f)};

  public:
    SurfaceSvmEmissionConsumer(
        const ShaderServices &services,
        Expr<bool> reflective_caustics) noexcept
        : _services{services},
          _reflective_caustics{reflective_caustics} {}

    void set_shading_normal(Expr<luisa::float3>) noexcept override {}

    void visit(
        const SurfacePoint &point,
        const TracedClosure &closure,
        Expr<std::uint32_t> endpoints,
        Expr<std::uint32_t>) noexcept override {
        accumulate_surface_svm_emission(
            _services, point, closure, endpoints, _reflective_caustics,
            _emission);
    }

    [[nodiscard]] Float3 emission() const noexcept {
        return Float3{_emission.expression()};
    }
};

class SurfaceSvmFinalizingPopulationConsumer final
    : public SurfaceSvmClosureConsumer {

  private:
    const ShaderServices &_services;
    const SurfacePopulationQuery &_query;
    SurfaceClosureCollector &_collector;
    Float3 _emission{make_float3(0.0f)};
    Float4 _transparent_sum{make_float4(0.0f)};

  public:
    SurfaceSvmFinalizingPopulationConsumer(
        const ShaderServices &services,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) noexcept
        : _services{services}, _query{query}, _collector{collector} {}

    void set_shading_normal(
        Expr<luisa::float3> shading_normal) noexcept override {
        _collector.begin(shading_normal);
    }

    void visit(
        const SurfacePoint &point,
        const TracedClosure &closure,
        Expr<std::uint32_t> endpoints,
        Expr<std::uint32_t>) noexcept override {
        accumulate_surface_svm_emission(
            _services, point, closure, endpoints,
            _query.emission_reflective_caustics, _emission);
        visit_surface_svm_physical_leaf(
            _services, point, closure, endpoints,
            _query.reflective_caustics, _query.refractive_caustics,
            [&](const TracedClosure &physical) noexcept {
                if (physical.operation ==
                    compiler::ClosureOperation::transparent) {
                    const auto allocated =
                        physical.sample_weight >=
                        cycles_closure::closure_weight_cutoff;
                    $if(allocated) {
                        $if(_transparent_sum.w == 0.0f) {
                            _collector.begin_transparent_closure(
                                canonical_surface_closure(physical));
                        };
                        _transparent_sum += make_float4(
                            physical.weight, physical.sample_weight);
                    };
                    return;
                }
                _collector.add(canonical_surface_closure(physical));
            });
    }

    void finish() noexcept {
        $if(_transparent_sum.w > 0.0f) {
            _collector.finalize_transparent_closure(
                _transparent_sum.xyz(), _transparent_sum.w);
        };
    }

    [[nodiscard]] Float3 emission() const noexcept {
        return Float3{_emission.expression()};
    }
};

class SurfaceSvmFirstPopulationPass final
    : public SurfaceSvmClosureConsumer {

  private:
    const ShaderServices &_services;
    const SurfacePopulationQuery &_query;
    SurfaceClosureCollector &_collector;
    Float3 _emission{make_float3(0.0f)};
    Float3 _transparent_weight{make_float3(0.0f)};
    Float _transparent_sample_weight{0.0f};
    Bool _transparent_pending{false};
    Float3 _shading_normal;

  public:
    SurfaceSvmFirstPopulationPass(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) noexcept
        : _services{services},
          _query{query},
          _collector{collector},
          _shading_normal{point.shading_normal} {}

    void set_shading_normal(
        Expr<luisa::float3> shading_normal) noexcept override {
        _shading_normal = shading_normal;
        _collector.begin(shading_normal);
    }

    void visit(
        const SurfacePoint &point,
        const TracedClosure &closure,
        Expr<std::uint32_t> endpoints,
        Expr<std::uint32_t>) noexcept override {
        accumulate_surface_svm_emission(
            _services, point, closure, endpoints,
            _query.emission_reflective_caustics, _emission);
        visit_surface_svm_physical_leaf(
            _services, point, closure, endpoints,
            _query.reflective_caustics, _query.refractive_caustics,
            [&](const TracedClosure &physical) noexcept {
                if (physical.operation ==
                    compiler::ClosureOperation::transparent) {
                    const auto allocated =
                        physical.sample_weight >=
                        cycles_closure::closure_weight_cutoff;
                    $if(allocated) {
                        _transparent_weight += physical.weight;
                        _transparent_sample_weight += physical.sample_weight;
                        _transparent_pending = true;
                    };
                    return;
                }
                $if(!_transparent_pending) {
                    _collector.add(canonical_surface_closure(physical));
                };
            });
    }

    [[nodiscard]] Bool transparent_pending() const noexcept {
        return Bool{_transparent_pending.expression()};
    }

    [[nodiscard]] SurfaceClosureRecord merged_transparent() const noexcept {
        return make_merged_transparent_closure(
            _shading_normal, _transparent_weight,
            _transparent_sample_weight);
    }

    [[nodiscard]] Float3 emission() const noexcept {
        return Float3{_emission.expression()};
    }
};

class SurfaceSvmPopulationReplay final : public SurfaceSvmClosureConsumer {

  private:
    const ShaderServices &_services;
    const SurfacePopulationQuery &_query;
    SurfaceClosureCollector &_collector;
    Bool _reached_first_transparent{false};

  public:
    SurfaceSvmPopulationReplay(
        const ShaderServices &services,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) noexcept
        : _services{services}, _query{query}, _collector{collector} {}

    // begin() is a transaction hook, not a replay hook. The first pass has
    // already established the final shading normal and emitted the prefix.
    void set_shading_normal(Expr<luisa::float3>) noexcept override {}

    void visit(
        const SurfacePoint &point,
        const TracedClosure &closure,
        Expr<std::uint32_t> endpoints,
        Expr<std::uint32_t>) noexcept override {
        visit_surface_svm_physical_leaf(
            _services, point, closure, endpoints,
            _query.reflective_caustics, _query.refractive_caustics,
            [&](const TracedClosure &physical) noexcept {
                if (physical.operation ==
                    compiler::ClosureOperation::transparent) {
                    _reached_first_transparent |=
                        physical.sample_weight >=
                        cycles_closure::closure_weight_cutoff;
                    return;
                }
                $if(_reached_first_transparent) {
                    _collector.add(canonical_surface_closure(physical));
                };
            });
    }
};

class SurfaceSvmPreparationConsumer final
    : public SurfaceSvmClosureConsumer {

  private:
    const ShaderServices &_services;
    const SurfacePreparationQuery &_query;
    SurfacePreparationAccumulator &_accumulator;
    Float3 _emission{make_float3(0.0f)};
    Float3 _transparent_weight{make_float3(0.0f)};
    Bool _transparent_pending{false};

  public:
    SurfaceSvmPreparationConsumer(
        const ShaderServices &services,
        const SurfacePreparationQuery &query,
        SurfacePreparationAccumulator &accumulator) noexcept
        : _services{services}, _query{query}, _accumulator{accumulator} {}

    void set_shading_normal(
        Expr<luisa::float3> shading_normal) noexcept override {
        _accumulator.set_shading_normal(shading_normal);
    }

    void visit(
        const SurfacePoint &point,
        const TracedClosure &closure,
        Expr<std::uint32_t> endpoints,
        Expr<std::uint32_t>) noexcept override {
        accumulate_surface_svm_emission(
            _services, point, closure, endpoints,
            _query.emission_reflective_caustics, _emission);
        visit_surface_svm_physical_leaf(
            _services, point, closure, endpoints,
            _query.reflective_caustics, _query.refractive_caustics,
            [&](const TracedClosure &physical) noexcept {
                if (physical.operation ==
                    compiler::ClosureOperation::transparent) {
                    const auto allocated =
                        physical.sample_weight >=
                        cycles_closure::closure_weight_cutoff;
                    $if(allocated) {
                        $if(!_transparent_pending) {
                            auto placeholder =
                                canonical_surface_closure(physical);
                            placeholder.weight = make_float3(0.0f);
                            _accumulator.add(placeholder);
                        };
                        _transparent_weight += physical.weight;
                        _transparent_pending = true;
                    };
                    return;
                }
                _accumulator.add(canonical_surface_closure(physical));
            });
    }

    void finish() noexcept {
        $if(_transparent_pending) {
            _accumulator.finalize_transparent_setup(_transparent_weight);
        };
        _accumulator.finish();
    }

    [[nodiscard]] Float3 emission() const noexcept {
        return Float3{_emission.expression()};
    }
};

class SurfaceSvmBssrdfNormalConsumer final
    : public SurfaceSvmClosureConsumer {

  private:
    const ShaderServices &_services;
    Bool _reflective_caustics;
    Bool _refractive_caustics;
    SurfaceBssrdfNormalAccumulator &_accumulator;
    Bool _transparent_pending{false};

  public:
    SurfaceSvmBssrdfNormalConsumer(
        const ShaderServices &services,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        SurfaceBssrdfNormalAccumulator &accumulator) noexcept
        : _services{services},
          _reflective_caustics{reflective_caustics},
          _refractive_caustics{refractive_caustics},
          _accumulator{accumulator} {}

    void set_shading_normal(
        Expr<luisa::float3> shading_normal) noexcept override {
        _accumulator.set_shading_normal(shading_normal);
    }

    void visit(
        const SurfacePoint &point,
        const TracedClosure &closure,
        Expr<std::uint32_t> endpoints,
        Expr<std::uint32_t>) noexcept override {
        visit_surface_svm_physical_leaf(
            _services, point, closure, endpoints,
            _reflective_caustics, _refractive_caustics,
            [&](const TracedClosure &physical) noexcept {
                if (physical.operation ==
                    compiler::ClosureOperation::transparent) {
                    const auto allocated =
                        physical.sample_weight >=
                        cycles_closure::closure_weight_cutoff;
                    $if(allocated & !_transparent_pending) {
                        const auto canonical =
                            canonical_surface_closure(physical);
                        _accumulator.add(
                            canonical.closure_type,
                            canonical.weight,
                            canonical.allocation_weight,
                            canonical.normal);
                    };
                    _transparent_pending |= allocated;
                    return;
                }
                const auto canonical = canonical_surface_closure(physical);
                _accumulator.add(
                    canonical.closure_type,
                    canonical.weight,
                    canonical.allocation_weight,
                    canonical.normal);
            });
    }
};

class SurfaceSvmPopulationProgramImpl final
    : public SurfacePopulationProgram {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    SurfaceSvmInterpreter _interpreter;

  public:
    SurfaceSvmPopulationProgramImpl(
        std::shared_ptr<LuisaSceneData> scene,
        SurfaceSvmInterpreter interpreter) noexcept
        : _scene{std::move(scene)},
          _interpreter{std::move(interpreter)} {}

    [[nodiscard]] SurfacePopulation populate(
        Expr<std::uint32_t> surface_tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector,
        const PathSurfaceAmbientOcclusionContext
            *ambient_occlusion) const noexcept override {
        collector.begin(point.shading_normal);
        const auto resources = surface_svm_resources(
            Expr<Buffer<float>>{_scene->scalar_parameter_buffer},
            Expr<Buffer<luisa::float3>>{_scene->vector_parameter_buffer},
            Expr<Buffer<float>>{_scene->cycles_bsdf_table_buffer},
            Expr<BindlessArray>{_scene->texture_heap},
            Expr<BindlessArray>{_scene->heap});
        Float3 emission = make_float3(0.0f);
        Float3 shading_normal = point.shading_normal;
        if (collector.supports_transparent_closure_finalization()) {
            SurfaceSvmFinalizingPopulationConsumer consumer{
                services, query, collector};
            shading_normal = _interpreter.execute(
                resources, services, UInt{surface_tag}, point, consumer,
                ambient_occlusion);
            consumer.finish();
            emission = consumer.emission();
        } else {
            SurfaceSvmFirstPopulationPass first{
                services, point, query, collector};
            shading_normal = _interpreter.execute(
                resources, services, UInt{surface_tag}, point, first,
                ambient_occlusion);
            emission = first.emission();
            $if(first.transparent_pending()) {
                collector.add(first.merged_transparent());
                SurfaceSvmPopulationReplay replay{
                    services, query, collector};
                static_cast<void>(_interpreter.execute(
                    resources, services, UInt{surface_tag}, point, replay,
                    ambient_occlusion));
            };
        }
        collector.finish();
        return {.emission = std::move(emission),
                .shading_normal = std::move(shading_normal)};
    }
};

} // namespace

std::shared_ptr<const SurfacePopulationProgram>
make_compact_surface_population_program(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene || !scene->surface_values ||
        scene->surface_values->topologies.size() != scene->surfaces.size()) {
        std::abort();
    }
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup = make_surface_attribute_lookup_callable(
        scene->attribute_binding_slot, scene->attribute_range_slot);
    auto interpreter = make_surface_svm_interpreter(
        scene, texture_sampling, attribute_lookup,
        SurfaceValueProgramDomain::preparation, true);
    return std::make_shared<SurfaceSvmPopulationProgramImpl>(
        scene, std::move(interpreter));
}

SurfaceEmissionCallable make_compact_surface_emission_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene || !scene->surface_values ||
        scene->surface_values->topologies.size() != scene->surfaces.size()) {
        std::abort();
    }
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup = make_surface_attribute_lookup_callable(
        scene->attribute_binding_slot, scene->attribute_range_slot);
    auto interpreter = make_surface_svm_interpreter(
        scene, texture_sampling, attribute_lookup,
        SurfaceValueProgramDomain::emission);

    SurfaceEmissionCallable callable =
        [scene, interpreter = std::move(interpreter), texture_sampling,
         attribute_lookup](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float3,
            Bool reflective_caustics) noexcept {
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
                nullptr,
                &texture_provider,
                &attribute_provider};
            const auto point = unpack_surface_point(packed_point);
            SurfaceSvmEmissionConsumer consumer{
                services, reflective_caustics};
            static_cast<void>(interpreter.execute(
                surface_svm_resources(
                    scalar_parameters, vector_parameters,
                    cycles_bsdf_tables, textures, geometry_heap),
                services, surface_tag, point, consumer));
            return consumer.emission();
        };
    callable.set_name("surface_emission_unified_svm");
    return callable;
}

SurfacePreparationCallable make_compact_surface_preparation_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene || !scene->surface_values ||
        scene->surface_values->topologies.size() != scene->surfaces.size()) {
        std::abort();
    }
    const auto closure_setup = make_surface_closure_setup_callables();
    const auto closure_identity = make_surface_closure_identity_callable();
    const auto closure_aov = make_surface_closure_aov_callable();
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup = make_surface_attribute_lookup_callable(
        scene->attribute_binding_slot, scene->attribute_range_slot);
    auto interpreter = make_surface_svm_interpreter(
        scene, texture_sampling, attribute_lookup,
        SurfaceValueProgramDomain::preparation);

    SurfacePreparationCallable callable =
        [scene, interpreter = std::move(interpreter), closure_setup,
         closure_identity, closure_aov, texture_sampling, attribute_lookup](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
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
            const auto point = unpack_surface_point(packed_point);
            const auto query =
                unpack_surface_preparation_query(packed_query);
            SurfacePreparationAccumulator accumulator{
                point,
                maximum_surface_closure_capacity,
                query.glossy_filter_roughness,
                query.include_runtime_flags,
                query.include_aov,
                closure_identity,
                closure_aov};
            SurfaceSvmPreparationConsumer consumer{
                services, query, accumulator};
            static_cast<void>(interpreter.execute(
                surface_svm_resources(
                    scalar_parameters, vector_parameters,
                    cycles_bsdf_tables, textures, geometry_heap),
                services, surface_tag, point, consumer));
            consumer.finish();
            return pack_surface_preparation(
                accumulator.preparation(consumer.emission()));
        };
    callable.set_name("surface_prepare_unified_svm");
    return callable;
}

SurfaceBssrdfNormalCallable make_compact_surface_bssrdf_normal_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene || !scene->surface_values ||
        scene->surface_values->topologies.size() != scene->surfaces.size()) {
        std::abort();
    }
    if (scene->surface_bssrdf_bump_tags.empty()) {
        SurfaceBssrdfNormalCallable identity = [](
            BufferFloat,
            BufferFloat3,
            BufferFloat,
            BindlessVar,
            BindlessVar,
            UInt,
            Var<SurfacePointCall> packed_point,
            Bool,
            Bool,
            Bool) noexcept {
            return unpack_surface_point(packed_point).shading_normal;
        };
        identity.set_name("surface_bssrdf_normal_scene_identity");
        return identity;
    }
    const auto closure_setup = make_surface_closure_setup_callables();
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup = make_surface_attribute_lookup_callable(
        scene->attribute_binding_slot, scene->attribute_range_slot);
    auto interpreter = make_surface_svm_interpreter(
        scene, texture_sampling, attribute_lookup,
        SurfaceValueProgramDomain::bssrdf);

    SurfaceBssrdfNormalCallable callable =
        [scene, interpreter = std::move(interpreter), closure_setup,
         texture_sampling, attribute_lookup](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Bool has_bssrdf_bump,
            Bool reflective_caustics,
            Bool refractive_caustics) noexcept {
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
            const auto point = unpack_surface_point(packed_point);
            Float3 result = point.shading_normal;
            $if(has_bssrdf_bump) {
                SurfaceBssrdfNormalAccumulator accumulator{
                    point.shading_normal,
                    scene->volume_metadata.closure_allocation_budget};
                SurfaceSvmBssrdfNormalConsumer consumer{
                    services, reflective_caustics, refractive_caustics,
                    accumulator};
                static_cast<void>(interpreter.execute(
                    surface_svm_resources(
                        scalar_parameters, vector_parameters,
                        cycles_bsdf_tables, textures, geometry_heap),
                    services, surface_tag, point, consumer));
                result = accumulator.result();
            };
            return result;
        };
    callable.set_name("surface_bssrdf_normal_unified_svm");
    return callable;
}

} // namespace psycles::luisa_backend::detail
