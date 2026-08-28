#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_evaluator.h>
#include <psycles/luisa/surface_closure_blocks.h>
#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_identity.h>
#include <psycles/luisa/surface_closure_operations.h>
#include <psycles/luisa/surface_closure_physical_blocks.h>
#include <psycles/luisa/surface_closure_sampling.h>
#include <psycles/luisa/surface_closure_set.h>

#include "luisa_surface_test_support.h"
#include "luisa_surface_closure_collection_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::merged_surface_closure_plan;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::RequestedClosureCollector;

constexpr auto closure_slots = 8u;
constexpr auto records_per_slot = 7u;
constexpr auto storage_records_per_slot = 19u;
constexpr auto evaluator_records_per_slot = 10u;
constexpr auto scattering_records_per_slot = 6u;
constexpr auto sampling_records_per_slot = 8u;
constexpr auto categorical_mask_count = 16u;
constexpr auto categorical_random_count = 4u;
constexpr auto categorical_invocation_count =
    categorical_mask_count * categorical_random_count;
constexpr auto categorical_records_per_invocation = 2u;

// Test-only operation which exercises the same OOP visitor and canonical
// contribution algebra without the path tracer's scene-resource callable.
// The separate block-ABI regression below covers callable transport itself;
// production kernels combine both pieces.
class InlineSurfaceClosureEvaluationOperation final
    : public SurfaceClosureEvaluationOperation {

private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;
    const SurfaceQuery &_query;
    const SurfaceClosureEvaluationPolicy &_policy;
    Float3 _incoming{make_float3(0.0f)};
    Float3 _outgoing{make_float3(0.0f)};

public:
    InlineSurfaceClosureEvaluationOperation(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfaceQuery &query,
        const SurfaceClosureEvaluationPolicy &policy) noexcept
        : _services{services},
          _point{point},
          _query{query},
          _policy{policy} {}

    void set_outgoing(
        Expr<luisa::float3> outgoing) noexcept override {
        const auto directions =
            make_surface_closure_evaluation_directions(
                _point, outgoing);
        _incoming = directions.incoming;
        _outgoing = directions.outgoing;
    }

    [[nodiscard]] luisa::compute::Var<
        SurfaceClosureEvaluationContributionCall>
    evaluate(
        Expr<luisa::float3> shading_normal,
        const SurfaceClosureExpression &closure,
        Expr<bool> selected_sample) const noexcept override {
        return surface_closure_evaluation_contribution(
            _services,
            _point,
            shading_normal,
            closure.reference(),
            Expr<luisa::float3>{_incoming.expression()},
            Expr<luisa::float3>{_outgoing.expression()},
            _query,
            _policy,
            selected_sample);
    }
};

class InlineSurfaceClosureSamplingOperation final
    : public SurfaceClosureSamplingOperation {

private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;
    const SurfaceQuery &_query;
    Float3 _incoming{make_float3(0.0f)};
    std::size_t *_selection_recordings{};
    std::size_t *_conditional_sample_recordings{};

public:
    InlineSurfaceClosureSamplingOperation(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfaceQuery &query,
        std::size_t *selection_recordings = nullptr,
        std::size_t *conditional_sample_recordings = nullptr) noexcept
        : _services{services},
          _point{point},
          _query{query},
          _incoming{make_surface_closure_sampling_incoming(
              _point)},
          _selection_recordings{selection_recordings},
          _conditional_sample_recordings{
              conditional_sample_recordings} {}

    [[nodiscard]] luisa::compute::Var<
        SurfaceClosureSelectionCall>
    selection(
        const SurfaceClosureExpression &closure)
        const noexcept override {
        if (_selection_recordings != nullptr) {
            ++*_selection_recordings;
        }
        return surface_closure_selection(
            make_surface_closure_selection_context(_query),
            static_cast<SurfaceClosurePhysicalRecord>(
                closure.reference()));
    }

    [[nodiscard]] luisa::compute::Var<
        SurfaceClosureConditionalSampleCall>
    conditional_sample(
        Expr<luisa::float3> shading_normal,
        const SurfaceClosureExpression &closure,
        Expr<luisa::float3> glossy_normal,
        Expr<luisa::float2> random_direction,
        Expr<float> rescaled_lobe) const noexcept override {
        if (_conditional_sample_recordings != nullptr) {
            ++*_conditional_sample_recordings;
        }
        return surface_closure_conditional_sample(
            _services,
            _point,
            shading_normal,
            closure.reference(),
            Expr<luisa::float3>{_incoming.expression()},
            glossy_normal,
            random_direction,
            rescaled_lobe,
            _query);
    }
};

[[nodiscard]] ShaderGraph make_layered_graph() {
    ShaderGraph graph;
    const auto principled = graph.add_node(node_type::principled_bsdf,
                                           "All physical Principled lobes");
    const auto configured =
        graph.set_input(principled, "BaseColor",
                        SocketValue::color({0.38f, 0.21f, 0.09f})) &&
        graph.set_input(principled, "Metallic", SocketValue::floating(0.22f)) &&
        graph.set_input(principled, "Roughness", SocketValue::floating(0.31f)) &&
        graph.set_input(principled, "DiffuseRoughness",
                        SocketValue::floating(0.17f)) &&
        graph.set_input(principled, "TransmissionWeight",
                        SocketValue::floating(0.27f)) &&
        graph.set_input(principled, "IOR", SocketValue::floating(1.46f)) &&
        graph.set_input(principled, "SpecularIORLevel",
                        SocketValue::floating(0.63f)) &&
        graph.set_input(principled, "SpecularTint",
                        SocketValue::color({0.71f, 0.84f, 0.93f})) &&
        graph.set_input(principled, "Alpha", SocketValue::floating(0.68f)) &&
        graph.set_input(principled, "SheenWeight",
                        SocketValue::floating(0.19f)) &&
        graph.set_input(principled, "SheenRoughness",
                        SocketValue::floating(0.42f)) &&
        graph.set_input(principled, "SheenTint",
                        SocketValue::color({0.92f, 0.37f, 0.16f})) &&
        graph.set_input(principled, "CoatWeight", SocketValue::floating(0.16f)) &&
        graph.set_input(principled, "CoatRoughness",
                        SocketValue::floating(0.13f)) &&
        graph.set_input(principled, "CoatIOR", SocketValue::floating(1.58f)) &&
        graph.set_property(principled, "Distribution",
                           SocketValue::string("MULTI_GGX"));
    if (!configured) {
        throw std::runtime_error{"failed to configure layered collection graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_beckmann_glass_graph() {
    ShaderGraph graph;
    const auto glass =
        graph.add_node(node_type::glass_bsdf, "Beckmann glass collection record");
    const auto configured =
        graph.set_input(glass, "Color",
                        SocketValue::color({0.73f, 0.86f, 0.94f})) &&
        graph.set_input(glass, "Roughness", SocketValue::floating(0.24f)) &&
        graph.set_input(glass, "IOR", SocketValue::floating(1.37f)) &&
        graph.set_input(glass, "ThinFilmThickness",
                        SocketValue::floating(425.0f)) &&
        graph.set_input(glass, "ThinFilmIOR",
                        SocketValue::floating(1.32f)) &&
        graph.set_property(glass, "Distribution",
                           SocketValue::string("BECKMANN"));
    if (!configured) {
        throw std::runtime_error{"failed to configure Beckmann collection graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = glass, .socket = "Closure"});
    return graph;
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto compile = [&](ShaderGraph graph) {
        auto shader = compiler.compile(graph);
        if (!shader.ok()) {
            throw std::runtime_error{"failed to compile collection shader graph"};
        }
        auto program = compile_surface_program(*shader.program);
        if (!program.ok()) {
            throw std::runtime_error{"failed to lower collection surface program"};
        }
        return program.program;
    };
    const auto layered = compile(make_layered_graph());
    const auto glass = compile(make_beckmann_glass_graph());

    auto parameters = parameter_data(*layered);
    const auto glass_parameter_base =
        static_cast<std::uint32_t>(parameters.size());
    const auto glass_parameters = parameter_data(*glass);
    SurfaceDispatch surfaces;
    const auto layered_tag = surfaces.create<GraphSurface>(
        layered,
        merged_surface_closure_plan(*layered, parameters));
    const auto glass_tag = surfaces.create<GraphSurface>(
        glass,
        merged_surface_closure_plan(*glass, glass_parameters));
    parameters.insert(parameters.end(), glass_parameters.begin(),
                      glass_parameters.end());
    const auto closure_identity =
        make_surface_closure_identity_callable();
    const auto closure_aov =
        make_surface_closure_aov_callable();
    Callable<luisa::float4x4(luisa::float4x4)>
        closure_block_passthrough = [](
                                        Float4x4 block) noexcept {
            return block;
        };

    Kernel1D collect = [&](BufferFloat4 parameter_buffer,
                           BufferFloat4 output) noexcept {
        const auto invocation = dispatch_x();
        const auto material = invocation / closure_slots;
        const auto requested = invocation % closure_slots;
        auto point = make_surface_point();
        point.parameter_block = select(0u, glass_parameter_base, material != 0u);
        ParameterShaderServices services{parameter_buffer};
        RequestedClosureCollector collector{requested};
        const auto tag = select(UInt{layered_tag}, UInt{glass_tag}, material != 0u);
        const auto collection =
            surfaces.collect_closures(tag, services, point, true, true, collector);
        const auto trace = collector.result();
        static_cast<void>(collection);
        const auto &closure = trace.closure;
        const auto base = invocation * records_per_slot;
        output.write(base, make_float4(cast<float>(trace.count),
                                       cast<float>(closure.closure_type),
                                       cast<float>(closure.microfacet_fresnel),
                                       select(0.0f, 1.0f, trace.valid)));
        output.write(base + 1u, make_float4(closure.weight, closure.sample_weight));
        output.write(base + 2u,
                     make_float4(closure.normal, closure.allocation_weight));
        output.write(base + 3u, make_float4(closure.albedo, closure.roughness));
        const auto flags = select(0u, 1u, closure.setup_valid) |
                           select(0u, 2u, closure.preserve_ggx_energy) |
                           select(0u, 4u, closure.beckmann);
        output.write(base + 4u,
                     make_float4(closure.reflection_albedo, cast<float>(flags)));
        output.write(base + 5u,
                     make_float4(trace.shading_normal, closure.ior));
        output.write(base + 6u,
                     make_float4(closure.thin_film_thickness,
                                 closure.thin_film_ior,
                                 0.0f,
                                 0.0f));
    };

    Kernel1D legacy = [&](BufferFloat4 parameter_buffer,
                          BufferFloat4 output) noexcept {
        const auto invocation = dispatch_x();
        const auto material = invocation / closure_slots;
        const auto requested = invocation % closure_slots;
        auto point = make_surface_point();
        point.parameter_block = select(0u, glass_parameter_base, material != 0u);
        ParameterShaderServices services{parameter_buffer};
        const auto tag = select(UInt{layered_tag}, UInt{glass_tag}, material != 0u);
        const auto trace = surfaces.closure_trace(tag, services, point, requested);
        const auto base = invocation * 3u;
        output.write(base, make_float4(cast<float>(trace.count),
                                       cast<float>(trace.type), trace.sample_weight,
                                       select(0.0f, 1.0f, trace.valid)));
        output.write(base + 1u, make_float4(trace.weight, 0.0f));
        output.write(base + 2u, make_float4(trace.normal, 0.0f));
    };

    Kernel1D retain = [&](BufferFloat4 parameter_buffer,
                          BufferFloat4 output) noexcept {
        const auto invocation = dispatch_x();
        const auto material = invocation / closure_slots;
        const auto requested = invocation % closure_slots;
        auto point = make_surface_point();
        point.parameter_block = select(
            0u, glass_parameter_base, material != 0u);
        ParameterShaderServices services{parameter_buffer};
        const auto tag = select(
            UInt{layered_tag}, UInt{glass_tag}, material != 0u);
        SurfaceClosureSet closures{3u};
        static_cast<void>(surfaces.collect_closures(
            tag,
            services,
            point,
            true,
            true,
            closures));
        const auto closure = closures.entry(requested);
        const auto valid = requested < closures.count();
        const auto base = invocation * 3u;
        output.write(base,
            make_float4(
                cast<float>(closures.count()),
                cast<float>(closure.closure_type),
                cast<float>(closure.microfacet_fresnel),
                select(0.0f, 1.0f, valid)));
        output.write(base + 1u,
            make_float4(
                closure.weight,
                closure.sample_weight));
        output.write(base + 2u,
            make_float4(
                closure.normal,
                closure.allocation_weight));
    };

    Kernel1D storage = [&](BufferFloat4 output) noexcept {
        SurfaceClosureSet closures{2u};

        auto ignored = SurfaceClosureRecord::zero();
        closures.add(ignored);

        auto below_cutoff = SurfaceClosureRecord::zero();
        below_cutoff.closure_type = cycles_closure::type_diffuse;
        below_cutoff.allocation_weight =
            0.5f * cycles_closure::closure_weight_cutoff;
        closures.add(below_cutoff);

        auto invalid_setup = SurfaceClosureRecord::zero();
        invalid_setup.weight = make_float3(0.1f, 0.2f, 0.3f);
        invalid_setup.allocation_weight = 0.2f;
        invalid_setup.sample_weight = 0.0f;
        invalid_setup.albedo = make_float3(0.4f, 0.5f, 0.6f);
        invalid_setup.normal = make_float3(0.0f, 1.0f, 0.0f);
        invalid_setup.ior = 1.2f;
        invalid_setup.setup_valid = false;
        closures.add(invalid_setup);
        SurfaceClosureSet invalid_setup_only{1u};
        invalid_setup_only.add(invalid_setup);

        auto glass_record = SurfaceClosureRecord::zero();
        glass_record.weight = make_float3(1.0f, 2.0f, 3.0f);
        glass_record.allocation_weight = 0.4f;
        glass_record.sample_weight = 0.5f;
        glass_record.setup_valid = true;
        glass_record.albedo = make_float3(4.0f, 5.0f, 6.0f);
        glass_record.reflection_albedo =
            make_float3(7.0f, 8.0f, 9.0f);
        glass_record.transmission_albedo =
            make_float3(10.0f, 11.0f, 12.0f);
        glass_record.color = make_float3(13.0f, 14.0f, 15.0f);
        glass_record.normal = make_float3(16.0f, 17.0f, 18.0f);
        glass_record.roughness = 0.19f;
        glass_record.microfacet_tangent =
            make_float3(53.0f, 54.0f, 55.0f);
        glass_record.microfacet_alpha_x = 0.56f;
        glass_record.microfacet_alpha_y = 0.57f;
        glass_record.diffuse_roughness = 0.20f;
        glass_record.metallic = 0.21f;
        glass_record.ior = 1.37f;
        glass_record.thin_film_thickness = 456.0f;
        glass_record.thin_film_ior = 1.29f;
        glass_record.specular_ior_level = 0.22f;
        glass_record.specular_tint =
            make_float3(23.0f, 24.0f, 25.0f);
        glass_record.sheen_transform_a = 0.26f;
        glass_record.sheen_transform_b = 0.27f;
        glass_record.evaluation_scale =
            make_float3(28.0f, 29.0f, 30.0f);
        glass_record.fresnel_f0 =
            make_float3(31.0f, 32.0f, 33.0f);
        glass_record.fresnel_f90 =
            make_float3(34.0f, 35.0f, 36.0f);
        glass_record.reflection_tint =
            make_float3(37.0f, 38.0f, 39.0f);
        glass_record.transmission_tint =
            make_float3(40.0f, 41.0f, 42.0f);
        glass_record.preserve_ggx_energy = true;
        // MULTI_GGX is an authoring choice, not an independent retained flag:
        // Cycles lowers it through GGX setup with energy preservation. A
        // simultaneous Beckmann+multi state is not representable by Cycles.
        glass_record.beckmann = false;
        glass_record.bssrdf_method = static_cast<std::uint32_t>(
            SurfaceBssrdfMethod::random_walk_skin);
        glass_record.bssrdf_radius =
            make_float3(43.0f, 44.0f, 45.0f);
        glass_record.bssrdf_albedo =
            make_float3(46.0f, 47.0f, 48.0f);
        glass_record.bssrdf_ior = 1.49f;
        glass_record.bssrdf_roughness = 0.51f;
        glass_record.bssrdf_anisotropy = 0.52f;
        psycles::luisa_backend::detail::finalize_cycles_closure_identity(
            glass_record,
            SurfaceClosureKind::glass,
            SurfaceClosureLobe::transmission);
        closures.add(glass_record);
        SurfaceClosureSet physical_closures{
            1u, SurfaceClosureStorageProfile::physical};
        UInt physical_fold_mask = 0u;
        physical_closures.append(glass_record, [&] {
            physical_fold_mask |= 1u;
        });

        auto overflow = SurfaceClosureRecord::zero();
        overflow.weight = make_float3(99.0f);
        overflow.allocation_weight = 0.9f;
        overflow.setup_valid = true;
        psycles::luisa_backend::detail::finalize_cycles_closure_identity(
            overflow, SurfaceClosureKind::diffuse);
        closures.add(overflow);
        physical_closures.append(overflow, [&] {
            physical_fold_mask |= 2u;
        });

        const auto requested = dispatch_x();
        const auto write_closure = [&output](
                                               UInt base,
                                               UInt count,
                                               Bool valid,
                                               const SurfaceClosureRecord
                                                   &closure) noexcept {
            const auto flags =
                select(0u, 1u, closure.setup_valid) |
                select(0u, 2u, closure.preserve_ggx_energy) |
                select(0u, 4u, closure.beckmann);
            output.write(base,
                make_float4(
                    cast<float>(count),
                    cast<float>(closure.closure_type),
                    cast<float>(closure.microfacet_fresnel),
                    select(0.0f, 1.0f, valid)));
            output.write(base + 1u,
                make_float4(
                    closure.weight,
                    closure.allocation_weight));
            output.write(base + 2u,
                make_float4(
                    closure.albedo,
                    closure.sample_weight));
            output.write(base + 3u,
                make_float4(
                    closure.reflection_albedo,
                    closure.roughness));
            output.write(base + 4u,
                make_float4(
                    closure.transmission_albedo,
                    closure.diffuse_roughness));
            output.write(base + 5u,
                make_float4(
                    closure.color,
                    closure.metallic));
            output.write(base + 6u,
                make_float4(
                    closure.normal,
                    closure.ior));
            output.write(base + 7u,
                make_float4(
                    closure.specular_tint,
                    closure.specular_ior_level));
            output.write(base + 8u,
                make_float4(
                    closure.evaluation_scale,
                    closure.sheen_transform_a));
            output.write(base + 9u,
                make_float4(
                    closure.fresnel_f0,
                    closure.sheen_transform_b));
            output.write(base + 10u,
                make_float4(
                    closure.fresnel_f90,
                    cast<float>(flags)));
            output.write(base + 11u,
                make_float4(
                    closure.reflection_tint,
                    closure.microfacet_alpha_x));
            output.write(base + 12u,
                make_float4(
                    closure.transmission_tint,
                    closure.microfacet_alpha_y));
            output.write(base + 13u,
                make_float4(closure.microfacet_tangent, 0.0f));
            output.write(base + 15u,
                make_float4(
                    closure.bssrdf_radius,
                    closure.bssrdf_anisotropy));
            output.write(base + 16u,
                make_float4(
                    closure.bssrdf_albedo,
                    closure.bssrdf_roughness));
            output.write(base + 17u,
                make_float4(
                    cast<float>(closure.bssrdf_method),
                    closure.bssrdf_ior,
                    0.0f,
                    0.0f));
            output.write(base + 18u,
                make_float4(
                    closure.thin_film_thickness,
                    closure.thin_film_ior,
                    0.0f,
                    0.0f));
        };
        const auto closure = closures.entry(requested);
        const auto valid = requested < closures.count();
        const auto base = requested * storage_records_per_slot;
        write_closure(base, closures.count(), valid, closure);
        auto point = make_surface_point();
        const SurfaceClosureEvaluator invalid_evaluator{
            point,
            invalid_setup_only,
            point.shading_normal};
        const auto invalid_trace =
            invalid_evaluator.closure_trace(requested);
        output.write(base + 14u,
            make_float4(
                cast<float>(invalid_trace.count),
                cast<float>(invalid_trace.type),
                select(0.0f, 1.0f, invalid_trace.valid),
                cast<float>(invalid_trace.runtime_flags)));
        $if(requested == 0u) {
            const auto blocks = pack_surface_closure(glass_record);
            const auto round_trip = unpack_surface_closure(
                Expr<luisa::float4x4>{
                    closure_block_passthrough(blocks.block_0)
                        .expression()},
                Expr<luisa::float4x4>{
                    closure_block_passthrough(blocks.block_1)
                        .expression()},
                Expr<luisa::float4x4>{
                    closure_block_passthrough(blocks.block_2)
                        .expression()},
                Expr<luisa::float4x4>{
                    closure_block_passthrough(blocks.block_3)
                        .expression()});
            constexpr auto round_trip_base =
                3u * storage_records_per_slot;
            write_closure(
                round_trip_base,
                2u,
                true,
                round_trip);
            output.write(
                round_trip_base + 14u,
                make_float4(0.0f));
            constexpr auto physical_base =
                4u * storage_records_per_slot;
            for (auto row = 0u; row < storage_records_per_slot; ++row) {
                output.write(physical_base + row, make_float4(0.0f));
            }
            const auto physical_access =
                physical_closures.physical_access(0u);
            const auto physical_common =
                physical_closures.physical_common_entry(physical_access);
            const auto physical_dielectric =
                unpack_surface_closure_physical_dielectric(
                    physical_common,
                    Expr<luisa::float4x4>{
                        physical_closures
                            .physical_payload_block(physical_access)
                            .expression()});
            output.write(
                physical_base,
                make_float4(
                    cast<float>(physical_closures.count()),
                    cast<float>(physical_common.closure_type),
                    cast<float>(physical_common.microfacet_fresnel),
                    select(0.0f, 1.0f, physical_access.valid())));
            output.write(
                physical_base + 1u,
                make_float4(
                    physical_common.weight,
                    physical_common.sample_weight));
            output.write(
                physical_base + 2u,
                make_float4(
                    physical_common.color_or_evaluation_scale,
                    physical_common.roughness));
            output.write(
                physical_base + 3u,
                make_float4(physical_common.normal, 0.0f));
            output.write(
                physical_base + 4u,
                make_float4(
                    physical_dielectric.payload.fresnel_f0,
                    physical_dielectric.payload.ior));
            output.write(
                physical_base + 5u,
                make_float4(
                    physical_dielectric.payload.fresnel_f90,
                    physical_dielectric.payload.thin_film_thickness));
            output.write(
                physical_base + 6u,
                make_float4(
                    physical_dielectric.payload.reflection_tint,
                    physical_dielectric.payload.thin_film_ior));
            output.write(
                physical_base + 7u,
                make_float4(
                    physical_dielectric.payload.transmission_tint,
                    0.0f));
            output.write(
                physical_base + 14u,
                make_float4(
                    cast<float>(physical_fold_mask),
                    cast<float>(physical_closures.count()),
                    0.0f,
                    0.0f));
        };
    };

    const auto write_evaluator_result = [](
        const BufferFloat4 &output,
        UInt base,
        const SurfaceClosureTrace &trace,
        UInt filtered_runtime_flags,
        const SurfaceAov &aov) noexcept {
        output.write(base,
            make_float4(
                cast<float>(trace.count),
                cast<float>(trace.runtime_flags),
                cast<float>(trace.index),
                cast<float>(trace.type)));
        output.write(base + 1u,
            make_float4(
                trace.sample_weight,
                select(0.0f, 1.0f, trace.valid),
                cast<float>(filtered_runtime_flags),
                0.0f));
        output.write(base + 2u,
            make_float4(trace.weight, 0.0f));
        output.write(base + 3u,
            make_float4(trace.normal, 0.0f));
        output.write(base + 4u,
            make_float4(aov.albedo, aov.roughness.x));
        output.write(base + 5u,
            make_float4(
                aov.glossy_albedo,
                aov.roughness.y));
        output.write(base + 6u,
            make_float4(
                aov.transmission_albedo,
                0.0f));
        output.write(base + 7u,
            make_float4(aov.normal, 0.0f));
        output.write(base + 8u,
            make_float4(aov.transparency, 0.0f));
        output.write(base + 9u,
            make_float4(
                aov.albedo + aov.glossy_albedo +
                    aov.transmission_albedo,
                    0.0f));
    };

    const auto write_scattering_result = [](
        const BufferFloat4 &output,
        UInt base,
        const SurfaceEvaluation &regular,
        const SurfaceEvaluation &light) noexcept {
        output.write(base,
            make_float4(regular.f, regular.pdf));
        output.write(base + 1u,
            make_float4(
                regular.diffuse_f, regular.diffuse_pdf));
        output.write(base + 2u,
            make_float4(
                regular.glossy_f,
                cast<float>(regular.events)));
        output.write(base + 3u,
            make_float4(light.f, light.pdf));
        output.write(base + 4u,
            make_float4(
                light.diffuse_f, light.diffuse_pdf));
        output.write(base + 5u,
            make_float4(
                light.glossy_f,
                cast<float>(light.events)));
    };

    const auto write_sampling_result = [](
        const BufferFloat4 &output,
        UInt base,
        const SurfaceSampleTrace &trace) noexcept {
        const auto &sample = trace.sample;
        output.write(base,
            make_float4(
                sample.evaluation.f,
                sample.evaluation.pdf));
        output.write(base + 1u,
            make_float4(
                sample.evaluation.diffuse_f,
                sample.evaluation.diffuse_pdf));
        output.write(base + 2u,
            make_float4(
                sample.evaluation.glossy_f,
                cast<float>(sample.evaluation.events)));
        output.write(base + 3u,
            make_float4(sample.wi, sample.eta));
        output.write(base + 4u,
            make_float4(
                sample.roughness,
                cast<float>(sample.runtime_flags),
                select(0.0f, 1.0f, sample.valid)));
        output.write(base + 5u,
            make_float4(
                cast<float>(trace.closure_index),
                cast<float>(trace.closure_type),
                trace.closure_sample_weight,
                trace.selection_rescaled));
        output.write(base + 6u,
            make_float4(
                trace.closure_weight,
                select(0.0f, 1.0f, trace.closure_valid)));
        output.write(base + 7u,
            make_float4(trace.closure_normal, 0.0f));
    };

    const auto scattering_inputs = [](
        UInt scenario) noexcept {
        constexpr auto all_lobes =
            static_cast<std::uint32_t>(
                event_diffuse | event_glossy |
                event_transmission | event_transparent);
        UInt lobe_mask = all_lobes;
        lobe_mask = select(lobe_mask,
            static_cast<std::uint32_t>(
                event_diffuse | event_transmission),
            scenario == 1u);
        lobe_mask = select(lobe_mask,
            static_cast<std::uint32_t>(event_glossy),
            scenario == 2u);
        lobe_mask = select(lobe_mask,
            static_cast<std::uint32_t>(
                event_glossy | event_transmission),
            scenario == 3u);
        lobe_mask = select(lobe_mask,
            static_cast<std::uint32_t>(event_transparent),
            scenario == 4u);
        lobe_mask = select(lobe_mask,
            static_cast<std::uint32_t>(event_diffuse),
            scenario == 6u);
        const auto outgoing = select(
            normalize(make_float3(0.31f, -0.17f, 0.936f)),
            normalize(make_float3(0.02f, 0.01f, -0.99975f)),
            (scenario == 1u) | (scenario == 3u) |
                (scenario == 6u));
        UInt shader_flags = cycles_abi::shader_use_mis;
        shader_flags |= select(0u,
            cycles_abi::shader_exclude_diffuse,
            (scenario == 1u) | (scenario == 7u));
        shader_flags |= select(0u,
            cycles_abi::shader_exclude_glossy,
            (scenario == 2u) | (scenario == 4u) |
                (scenario == 7u));
        shader_flags |= select(0u,
            cycles_abi::shader_exclude_transmit,
            (scenario == 3u) | (scenario == 4u) |
                (scenario == 6u));
        shader_flags = select(
            shader_flags, 0u, scenario == 5u);
        return std::tuple{
            outgoing, lobe_mask, shader_flags};
    };

    Kernel1D evaluate_shared =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 output) noexcept {
            const auto invocation = dispatch_x();
            const auto material = invocation / closure_slots;
            const auto requested = invocation % closure_slots;
            auto point = make_surface_point();
            point.parameter_block = select(
                0u, glass_parameter_base, material != 0u);
            point.back_facing = requested == 7u;
            ParameterShaderServices services{parameter_buffer};
            const auto tag = select(
                UInt{layered_tag},
                UInt{glass_tag},
                material != 0u);
            // Cycles reserves twelve slots for the reachable Principled
            // graph, which is also the scene maximum for this fixture. Each
            // production callable records only its formal field projection.
            SurfaceRuntimeFlagsVisitor runtime_visitor{
                point,
                0.04f,
                12u,
                closure_identity};
            static_cast<void>(surfaces.collect_closures(
                tag,
                services,
                point,
                true,
                true,
                runtime_visitor));
            const auto runtime_flags = runtime_visitor.result();

            SurfaceClosureTraceVisitor trace_visitor{
                point,
                requested,
                12u,
                closure_identity};
            static_cast<void>(surfaces.collect_closures(
                tag,
                services,
                point,
                true,
                true,
                trace_visitor));
            const auto &trace = trace_visitor.result();

            SurfaceAovVisitor aov_visitor{
                point,
                12u,
                closure_aov};
            static_cast<void>(surfaces.collect_closures(
                tag,
                services,
                point,
                true,
                true,
                aov_visitor));
            const auto preparation = surfaces.prepare(
                tag,
                services,
                point,
                {.outgoing = point.incoming,
                 .glossy_filter_roughness = 0.04f,
                 .emission_reflective_caustics = true,
                 .reflective_caustics = true,
                 .refractive_caustics = true,
                 .include_runtime_flags = true,
                 .include_aov = true});
            const auto split_emission = surfaces.emission(
                tag,
                services,
                point,
                point.incoming,
                true);
            const auto omitted = surfaces.prepare(
                tag,
                services,
                point,
                {.outgoing = point.incoming,
                 .glossy_filter_roughness = 0.04f,
                 .emission_reflective_caustics = true,
                 .reflective_caustics = true,
                 .refractive_caustics = true,
                 .include_runtime_flags = false,
                 .include_aov = false});
            const auto &split_aov = aov_visitor.result();
            device_assert(
                preparation.runtime_flags == runtime_flags);
            device_assert(all(abs(
                preparation.emission - split_emission) <= 1.0e-6f));
            device_assert(all(abs(
                preparation.aov.albedo - split_aov.albedo) <= 1.0e-6f));
            device_assert(all(abs(
                preparation.aov.glossy_albedo -
                split_aov.glossy_albedo) <= 1.0e-6f));
            device_assert(all(abs(
                preparation.aov.transmission_albedo -
                split_aov.transmission_albedo) <= 1.0e-6f));
            device_assert(all(abs(
                preparation.aov.roughness - split_aov.roughness) <=
                1.0e-6f));
            device_assert(all(abs(
                preparation.aov.normal - split_aov.normal) <= 1.0e-6f));
            device_assert(all(abs(
                preparation.aov.transparency -
                split_aov.transparency) <= 1.0e-6f));
            device_assert(all(abs(
                omitted.emission - split_emission) <= 1.0e-6f));
            device_assert(omitted.runtime_flags == 0u);
            device_assert(all(omitted.aov.albedo == 0.0f));
            device_assert(all(omitted.aov.glossy_albedo == 0.0f));
            device_assert(all(
                omitted.aov.transmission_albedo == 0.0f));
            device_assert(all(omitted.aov.roughness == 0.0f));
            device_assert(all(
                omitted.aov.normal == point.shading_normal));
            device_assert(all(omitted.aov.transparency == 0.0f));
            const auto base =
                invocation * evaluator_records_per_slot;
            write_evaluator_result(
                output,
                base,
                trace,
                preparation.runtime_flags,
                preparation.aov);
        };

    Kernel1D evaluate_legacy =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 output) noexcept {
            const auto invocation = dispatch_x();
            const auto material = invocation / closure_slots;
            const auto requested = invocation % closure_slots;
            auto point = make_surface_point();
            point.parameter_block = select(
                0u, glass_parameter_base, material != 0u);
            point.back_facing = requested == 7u;
            ParameterShaderServices services{parameter_buffer};
            const auto tag = select(
                UInt{layered_tag},
                UInt{glass_tag},
                material != 0u);
            const auto preparation = surfaces.prepare(
                tag,
                services,
                point,
                {.outgoing = point.incoming,
                 .glossy_filter_roughness = 0.04f,
                 .emission_reflective_caustics = true,
                 .reflective_caustics = true,
                 .refractive_caustics = true,
                 .include_runtime_flags = true,
                 .include_aov = true});
            const auto base =
                invocation * evaluator_records_per_slot;
            write_evaluator_result(
                output,
                base,
                surfaces.closure_trace(
                    tag,
                    services,
                    point,
                    requested,
                    true,
                    true),
                preparation.runtime_flags,
                preparation.aov);
        };

    Kernel1D scatter_shared =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 output) noexcept {
            const auto invocation = dispatch_x();
            const auto material = invocation / closure_slots;
            const auto scenario = invocation % closure_slots;
            auto point = make_surface_point();
            point.parameter_block = select(
                0u, glass_parameter_base, material != 0u);
            point.back_facing = scenario == 7u;
            ParameterShaderServices services{parameter_buffer};
            const auto tag = select(
                UInt{layered_tag},
                UInt{glass_tag},
                material != 0u);
            const auto [outgoing, lobe_mask, shader_flags] =
                scattering_inputs(scenario);
            const auto query = SurfaceQuery{
                .lobe_mask = lobe_mask,
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = select(
                    0.0f, 0.08f, scenario == 7u),
                .reflective_caustics = scenario != 6u,
                .refractive_caustics = scenario != 2u};
            const auto regular_policy =
                make_surface_closure_evaluation_policy(
                    false, Expr<std::uint32_t>{0u});
            InlineSurfaceClosureEvaluationOperation regular_operation{
                services,
                point,
                query,
                regular_policy};
            regular_operation.set_outgoing(
                Expr<luisa::float3>{outgoing.expression()});
            SurfaceClosureEvaluationVisitor regular_visitor{
                12u,
                regular_operation,
                Expr<bool>{regular_policy.preserve_pdf.expression()}};
            static_cast<void>(surfaces.collect_closures(
                tag,
                services,
                point,
                query.reflective_caustics,
                query.refractive_caustics,
                regular_visitor));
            const auto light_policy =
                make_surface_closure_evaluation_policy(
                    true,
                    Expr<std::uint32_t>{
                        shader_flags.expression()});
            InlineSurfaceClosureEvaluationOperation light_operation{
                services,
                point,
                query,
                light_policy};
            light_operation.set_outgoing(
                Expr<luisa::float3>{outgoing.expression()});
            SurfaceClosureEvaluationVisitor light_visitor{
                12u,
                light_operation,
                Expr<bool>{light_policy.preserve_pdf.expression()}};
            static_cast<void>(surfaces.collect_closures(
                tag,
                services,
                point,
                query.reflective_caustics,
                query.refractive_caustics,
                light_visitor));
            write_scattering_result(
                output,
                invocation * scattering_records_per_slot,
                regular_visitor.result(),
                light_visitor.result());
        };

    Kernel1D scatter_legacy =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 output) noexcept {
            const auto invocation = dispatch_x();
            const auto material = invocation / closure_slots;
            const auto scenario = invocation % closure_slots;
            auto point = make_surface_point();
            point.parameter_block = select(
                0u, glass_parameter_base, material != 0u);
            point.back_facing = scenario == 7u;
            ParameterShaderServices services{parameter_buffer};
            const auto tag = select(
                UInt{layered_tag},
                UInt{glass_tag},
                material != 0u);
            const auto [outgoing, lobe_mask, shader_flags] =
                scattering_inputs(scenario);
            const auto query = SurfaceQuery{
                .lobe_mask = lobe_mask,
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = select(
                    0.0f, 0.08f, scenario == 7u),
                .reflective_caustics = scenario != 6u,
                .refractive_caustics = scenario != 2u};
            write_scattering_result(
                output,
                invocation * scattering_records_per_slot,
                surfaces.evaluate(
                    tag, services, point, outgoing, query),
                surfaces.evaluate_light(tag,
                    services,
                    point,
                    outgoing,
                    SurfaceLightQuery{
                        .surface = query,
                        .shader_flags = shader_flags}));
        };

    auto selection_recordings = std::size_t{};
    auto conditional_sample_recordings = std::size_t{};
    Kernel1D sample_shared =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 output) noexcept {
            const auto invocation = dispatch_x();
            const auto material = invocation / closure_slots;
            const auto scenario = invocation % closure_slots;
            auto point = make_surface_point();
            point.parameter_block = select(
                0u, glass_parameter_base, material != 0u);
            point.back_facing = scenario == 7u;
            ParameterShaderServices services{parameter_buffer};
            const auto tag = select(
                UInt{layered_tag},
                UInt{glass_tag},
                material != 0u);
            const auto [unused_outgoing, lobe_mask, unused_flags] =
                scattering_inputs(scenario);
            static_cast<void>(unused_outgoing);
            static_cast<void>(unused_flags);
            const auto query = SurfaceQuery{
                .lobe_mask = lobe_mask,
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = select(
                    0.0f, 0.08f, scenario == 7u),
                .reflective_caustics = scenario != 6u,
                .refractive_caustics = scenario != 2u};
            const auto scenario_float = cast<float>(scenario);
            const auto u_lobe = min(
                0.02f + 0.137f * scenario_float,
                0.99999994f);
            const auto u_direction = make_float2(
                0.07f + 0.11f * scenario_float,
                0.91f - 0.09f * scenario_float);
            const auto policy =
                make_surface_closure_evaluation_policy(
                    false, Expr<std::uint32_t>{0u});
            InlineSurfaceClosureEvaluationOperation
                evaluation_operation{
                    services,
                    point,
                    query,
                    policy};
            InlineSurfaceClosureSamplingOperation
                sampling_operation{
                    services,
                    point,
                    query,
                    &selection_recordings,
                    &conditional_sample_recordings};
            const SurfaceClosurePoint closure_point{point};
            SurfaceClosureSamplingVisitor visitor{
                12u,
                closure_point,
                sampling_operation,
                evaluation_operation,
                Expr<float>{u_lobe.expression()},
                Expr<luisa::float2>{u_direction.expression()},
                true};
            static_cast<void>(surfaces.collect_closures(
                tag,
                services,
                point,
                query.reflective_caustics,
                query.refractive_caustics,
                visitor));
            write_sampling_result(
                output,
                invocation * sampling_records_per_slot,
                visitor.result());
        };

    Kernel1D sample_legacy =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 output) noexcept {
            const auto invocation = dispatch_x();
            const auto material = invocation / closure_slots;
            const auto scenario = invocation % closure_slots;
            auto point = make_surface_point();
            point.parameter_block = select(
                0u, glass_parameter_base, material != 0u);
            point.back_facing = scenario == 7u;
            ParameterShaderServices services{parameter_buffer};
            const auto tag = select(
                UInt{layered_tag},
                UInt{glass_tag},
                material != 0u);
            const auto [unused_outgoing, lobe_mask, unused_flags] =
                scattering_inputs(scenario);
            static_cast<void>(unused_outgoing);
            static_cast<void>(unused_flags);
            const auto query = SurfaceQuery{
                .lobe_mask = lobe_mask,
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = select(
                    0.0f, 0.08f, scenario == 7u),
                .reflective_caustics = scenario != 6u,
                .refractive_caustics = scenario != 2u};
            const auto scenario_float = cast<float>(scenario);
            const auto u_lobe = min(
                0.02f + 0.137f * scenario_float,
                0.99999994f);
            const auto u_direction = make_float2(
                0.07f + 0.11f * scenario_float,
                0.91f - 0.09f * scenario_float);
            write_sampling_result(
                output,
                invocation * sampling_records_per_slot,
                surfaces.sample_trace(tag,
                    services,
                    point,
                    u_lobe,
                    u_direction,
                    query));
        };

    // Every subset of four authored closures is compared below with the
    // compact retained sequence. In particular, inactive entries carry
    // nonzero weights/flags and one retained entry has zero mass. This pins
    // the formal identity between predication and sequence compaction rather
    // than merely exercising one material-specific closure layout.
    Kernel1D predicated_categorical =
        [&](BufferFloat4 output) noexcept {
            const auto invocation = dispatch_x();
            const auto retained_mask =
                invocation % categorical_mask_count;
            const auto random_bucket =
                invocation / categorical_mask_count;
            const auto random_lobe =
                (cast<float>(random_bucket) + 0.5f) /
                static_cast<float>(categorical_random_count);
            const auto make_selection = [](
                                            float weight,
                                            std::uint32_t runtime_flags,
                                            std::uint32_t closure_type) noexcept {
                luisa::compute::Var<
                    SurfaceClosureSelectionCall> selection;
                selection.weight = weight;
                selection.glossy_normal =
                    make_float3(0.0f, 0.0f, 1.0f);
                selection.runtime_flags = runtime_flags;
                selection.closure_type = closure_type;
                selection.closure_sample_weight = weight;
                return selection;
            };
            const auto first = make_selection(0.2f, 1u, 11u);
            const auto zero_mass = make_selection(0.0f, 4u, 12u);
            const auto third = make_selection(0.5f, 16u, 13u);
            const auto fourth = make_selection(0.3f, 64u, 14u);
            const auto retained_first =
                (retained_mask & 1u) != 0u;
            const auto retained_zero_mass =
                (retained_mask & 2u) != 0u;
            const auto retained_third =
                (retained_mask & 4u) != 0u;
            const auto retained_fourth =
                (retained_mask & 8u) != 0u;

            SurfaceClosureSelectionMeasure measure{false};
            measure.add(first, retained_first);
            measure.add(zero_mass, retained_zero_mass);
            measure.add(third, retained_third);
            measure.add(fourth, retained_fourth);

            SurfaceClosureCategoricalInversion inversion{
                Expr<float>{random_lobe.expression()}, measure};
            Float selected_index = -1.0f;
            Float selected_rescaled = 0.0f;
            Bool selected = false;
            const auto consider = [&]<typename Retained>(
                                      const auto &selection,
                                      Retained retained,
                                      std::uint32_t index) noexcept {
                const auto choice = inversion.consider(
                    selection, Expr<bool>{retained.expression()});
                selected_index = select(
                    selected_index,
                    static_cast<float>(index),
                    choice.choose);
                selected_rescaled = select(
                    selected_rescaled,
                    choice.rescaled,
                    choice.choose);
                selected |= choice.choose;
            };
            consider(first, retained_first, 0u);
            consider(zero_mass, retained_zero_mass, 1u);
            consider(third, retained_third, 2u);
            consider(fourth, retained_fourth, 3u);

            const auto base = invocation *
                              categorical_records_per_invocation;
            output.write(
                base,
                make_float4(
                    measure.total_weight(),
                    cast<float>(measure.retained_count()),
                    cast<float>(measure.runtime_flags()),
                    select(0.0f, 1.0f, selected)));
            output.write(
                base + 1u,
                make_float4(
                    selected_index,
                    selected_rescaled,
                    random_lobe,
                    cast<float>(retained_mask)));
        };

    // Population owns every Cycles closure-setup transform. Selection is a
    // pure observer of the resulting ShaderClosure base and must therefore
    // preserve its stored normal even for values that a second bump-normal
    // correction would change.
    Kernel1D selection_observes_populated_normal =
        [](BufferFloat4 output) noexcept {
            constexpr auto stored_normal =
                luisa::float3{0.75f, -0.5f, 0.125f};
            const auto context = SurfaceClosureSelectionContext{
                .lobe_mask = ~std::uint32_t{0u},
                .glossy_filter_roughness = 0.0f};
            const auto closure = SurfaceClosurePhysicalCommonRecord{
                .closure_type = cycles_closure::type_diffuse,
                .microfacet_fresnel = static_cast<std::uint32_t>(
                    cycles_closure::MicrofacetFresnel::none),
                .weight = make_float3(1.0f),
                .sample_weight = 0.7f,
                .color_or_evaluation_scale = make_float3(1.0f),
                .normal = stored_normal,
                .roughness = 0.25f};
            const auto selection = surface_closure_selection(
                context, closure);
            output.write(
                0u,
                make_float4(
                    selection.glossy_normal,
                    selection.weight));
        };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    constexpr auto invocation_count = 2u * closure_slots;
    auto collected_buffer =
        device.create_buffer<luisa::float4>(invocation_count * records_per_slot);
    auto legacy_buffer =
        device.create_buffer<luisa::float4>(invocation_count * 3u);
    auto retained_buffer =
        device.create_buffer<luisa::float4>(invocation_count * 3u);
    auto storage_buffer = device.create_buffer<luisa::float4>(
        5u * storage_records_per_slot);
    auto shared_evaluator_buffer =
        device.create_buffer<luisa::float4>(
            invocation_count *
            evaluator_records_per_slot);
    auto legacy_evaluator_buffer =
        device.create_buffer<luisa::float4>(
            invocation_count *
            evaluator_records_per_slot);
    auto shared_scattering_buffer =
        device.create_buffer<luisa::float4>(
            invocation_count *
            scattering_records_per_slot);
    auto legacy_scattering_buffer =
        device.create_buffer<luisa::float4>(
            invocation_count *
            scattering_records_per_slot);
    auto shared_sampling_buffer =
        device.create_buffer<luisa::float4>(
            invocation_count * sampling_records_per_slot);
    auto legacy_sampling_buffer =
        device.create_buffer<luisa::float4>(
            invocation_count * sampling_records_per_slot);
    auto categorical_buffer =
        device.create_buffer<luisa::float4>(
            categorical_invocation_count *
            categorical_records_per_invocation);
    auto selection_normal_buffer =
        device.create_buffer<luisa::float4>(1u);
    auto collect_kernel = device.compile(collect);
    auto legacy_kernel = device.compile(legacy);
    auto retain_kernel = device.compile(retain);
    auto storage_kernel = device.compile(storage);
    auto shared_evaluator_kernel =
        device.compile(evaluate_shared);
    auto legacy_evaluator_kernel =
        device.compile(evaluate_legacy);
    auto shared_scattering_kernel =
        device.compile(scatter_shared);
    auto legacy_scattering_kernel =
        device.compile(scatter_legacy);
    auto shared_sampling_kernel =
        device.compile(sample_shared);
    if (selection_recordings == 0u ||
        selection_recordings !=
            conditional_sample_recordings) {
        std::cerr
            << "surface closure selection was not scheduled exactly once "
               "per conditional sampler: selection="
            << selection_recordings
            << ", conditional="
            << conditional_sample_recordings << '\n';
        return EXIT_FAILURE;
    }
    auto legacy_sampling_kernel =
        device.compile(sample_legacy);
    auto predicated_categorical_kernel =
        device.compile(predicated_categorical);
    auto selection_normal_kernel =
        device.compile(selection_observes_populated_normal);
    std::array<luisa::float4, invocation_count * records_per_slot> collected{};
    std::array<luisa::float4, invocation_count * 3u> old{};
    std::array<luisa::float4, invocation_count * 3u> retained{};
    std::array<luisa::float4,
        5u * storage_records_per_slot>
        stored{};
    std::array<luisa::float4,
        invocation_count * evaluator_records_per_slot>
        shared_evaluator{};
    std::array<luisa::float4,
        invocation_count * evaluator_records_per_slot>
        legacy_evaluator{};
    std::array<luisa::float4,
        invocation_count * scattering_records_per_slot>
        shared_scattering{};
    std::array<luisa::float4,
        invocation_count * scattering_records_per_slot>
        legacy_scattering{};
    std::array<luisa::float4,
        invocation_count * sampling_records_per_slot>
        shared_sampling{};
    std::array<luisa::float4,
        invocation_count * sampling_records_per_slot>
        legacy_sampling{};
    std::array<luisa::float4,
        categorical_invocation_count *
            categorical_records_per_invocation>
        categorical{};
    std::array<luisa::float4, 1u> selection_normal{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << collect_kernel(parameter_buffer, collected_buffer)
                  .dispatch(invocation_count)
           << collected_buffer.copy_to(luisa::span{collected})
           << legacy_kernel(parameter_buffer, legacy_buffer)
                  .dispatch(invocation_count)
           << legacy_buffer.copy_to(luisa::span{old})
           << retain_kernel(parameter_buffer, retained_buffer)
                  .dispatch(invocation_count)
           << retained_buffer.copy_to(luisa::span{retained})
           << storage_kernel(storage_buffer).dispatch(3u)
           << storage_buffer.copy_to(luisa::span{stored})
           << shared_evaluator_kernel(
                  parameter_buffer,
                  shared_evaluator_buffer)
                  .dispatch(invocation_count)
           << shared_evaluator_buffer.copy_to(
                  luisa::span{shared_evaluator})
           << legacy_evaluator_kernel(
                  parameter_buffer,
                  legacy_evaluator_buffer)
                  .dispatch(invocation_count)
           << legacy_evaluator_buffer.copy_to(
                  luisa::span{legacy_evaluator})
           << shared_scattering_kernel(
                  parameter_buffer,
                  shared_scattering_buffer)
                  .dispatch(invocation_count)
           << shared_scattering_buffer.copy_to(
                  luisa::span{shared_scattering})
           << legacy_scattering_kernel(
                  parameter_buffer,
                  legacy_scattering_buffer)
                  .dispatch(invocation_count)
           << legacy_scattering_buffer.copy_to(
                  luisa::span{legacy_scattering})
           << shared_sampling_kernel(
                  parameter_buffer,
                  shared_sampling_buffer)
                  .dispatch(invocation_count)
           << shared_sampling_buffer.copy_to(
                  luisa::span{shared_sampling})
           << legacy_sampling_kernel(
                  parameter_buffer,
                  legacy_sampling_buffer)
                  .dispatch(invocation_count)
           << legacy_sampling_buffer.copy_to(
                  luisa::span{legacy_sampling})
           << predicated_categorical_kernel(
                  categorical_buffer)
                  .dispatch(categorical_invocation_count)
           << categorical_buffer.copy_to(
                  luisa::span{categorical})
           << selection_normal_kernel(
                  selection_normal_buffer)
                  .dispatch(1u)
           << selection_normal_buffer.copy_to(
                  luisa::span{selection_normal})
           << synchronize();

    if (!approximately_equal(
            selection_normal[0u],
            luisa::float4{0.75f, -0.5f, 0.125f, 0.7f})) {
        const auto observed = selection_normal[0u];
        std::cerr
            << "surface closure selection changed the populated normal on "
            << backend << ": {" << observed.x << ", "
            << observed.y << ", " << observed.z << ", "
            << observed.w << "}\n";
        return EXIT_FAILURE;
    }

    constexpr std::array categorical_weights{
        0.2f, 0.0f, 0.5f, 0.3f};
    constexpr std::array<std::uint32_t, 4u>
        categorical_flags{1u, 4u, 16u, 64u};
    for (auto invocation = 0u;
         invocation < categorical_invocation_count;
         ++invocation) {
        const auto retained_mask =
            invocation % categorical_mask_count;
        const auto random_bucket =
            invocation / categorical_mask_count;
        const auto random_lobe =
            (static_cast<float>(random_bucket) + 0.5f) /
            static_cast<float>(categorical_random_count);
        auto total_weight = 0.0f;
        auto retained_count = 0u;
        auto runtime_flags = 0u;
        for (auto index = 0u;
             index < categorical_weights.size();
             ++index) {
            if ((retained_mask & (1u << index)) == 0u) {
                continue;
            }
            total_weight += categorical_weights[index];
            runtime_flags |= categorical_flags[index];
            ++retained_count;
        }
        auto selected_index = -1.0f;
        auto selected_rescaled = 0.0f;
        auto accumulated = 0.0f;
        const auto target = random_lobe * total_weight;
        for (auto index = 0u;
             index < categorical_weights.size();
             ++index) {
            if ((retained_mask & (1u << index)) == 0u) {
                continue;
            }
            const auto weight = categorical_weights[index];
            const auto next = accumulated + weight;
            if (selected_index < 0.0f &&
                weight > 0.0f && target < next) {
                selected_index = static_cast<float>(index);
                selected_rescaled =
                    retained_count > 1u
                        ? (target - accumulated) / weight
                        : random_lobe;
            }
            accumulated = next;
        }
        const auto base = invocation *
                          categorical_records_per_invocation;
        const auto actual_measure = categorical[base];
        const auto actual_choice = categorical[base + 1u];
        const auto expected_selected =
            selected_index >= 0.0f ? 1.0f : 0.0f;
        if (!approximately_equal(
                actual_measure,
                luisa::float4{
                    total_weight,
                    static_cast<float>(retained_count),
                    static_cast<float>(runtime_flags),
                    expected_selected}) ||
            !approximately_equal(
                actual_choice,
                luisa::float4{
                    selected_index,
                    selected_rescaled,
                    random_lobe,
                    static_cast<float>(retained_mask)})) {
            std::cerr
                << "predicated categorical compaction failed on "
                << backend << " at mask " << retained_mask
                << ", random bucket " << random_bucket << '\n';
            return EXIT_FAILURE;
        }
    }

    constexpr std::array layered_types{
        cycles_closure::type_transparent,
        cycles_closure::type_sheen,
        cycles_closure::type_microfacet_ggx,
        cycles_closure::type_microfacet_ggx,
        cycles_closure::type_microfacet_ggx_glass,
        cycles_closure::type_microfacet_ggx,
        cycles_closure::type_oren_nayar};
    constexpr std::array layered_fresnels{
        cycles_closure::MicrofacetFresnel::none,
        cycles_closure::MicrofacetFresnel::none,
        cycles_closure::MicrofacetFresnel::dielectric,
        cycles_closure::MicrofacetFresnel::f82_tint,
        cycles_closure::MicrofacetFresnel::generalized_schlick,
        cycles_closure::MicrofacetFresnel::generalized_schlick,
        cycles_closure::MicrofacetFresnel::none};

    for (auto invocation = 0u; invocation < invocation_count; ++invocation) {
        const auto material = invocation / closure_slots;
        const auto requested = invocation % closure_slots;
        const auto collected_base = invocation * records_per_slot;
        const auto legacy_base = invocation * 3u;
        const auto meta = collected[collected_base];
        const auto legacy_meta = old[legacy_base];
        const auto expected_count = material == 0u ? 7u : 1u;
        const auto expected_valid = requested < expected_count;
        const auto expected_type =
            material == 0u && expected_valid
                ? layered_types[requested]
            : material != 0u && expected_valid
                ? cycles_closure::type_microfacet_beckmann_glass
                : cycles_closure::type_none;
        const auto expected_fresnel =
            material == 0u && expected_valid
                ? layered_fresnels[requested]
            : material != 0u && expected_valid
                ? cycles_closure::MicrofacetFresnel::generalized_schlick
                : cycles_closure::MicrofacetFresnel::none;
        const auto retained_count = material == 0u ? 3u : 1u;
        const auto retained_valid = requested < retained_count;
        const auto retained_meta = retained[legacy_base];
        const auto retained_equal =
            approximately_equal(retained_meta.x,
                static_cast<float>(retained_count)) &&
            approximately_equal(retained_meta.y,
                static_cast<float>(retained_valid
                                       ? expected_type
                                       : cycles_closure::type_none)) &&
            approximately_equal(retained_meta.z,
                static_cast<float>(retained_valid
                                       ? expected_fresnel
                                       : cycles_closure::MicrofacetFresnel::none)) &&
            approximately_equal(retained_meta.w,
                retained_valid ? 1.0f : 0.0f) &&
            approximately_equal(
                retained[legacy_base + 1u],
                retained_valid
                    ? collected[collected_base + 1u]
                    : luisa::float4{}) &&
            approximately_equal(
                retained[legacy_base + 2u],
                retained_valid
                    ? collected[collected_base + 2u]
                    : luisa::float4{0.0f, 0.0f, 1.0f, 0.0f});
        const auto core_equal =
            approximately_equal(meta.x, static_cast<float>(expected_count)) &&
            approximately_equal(meta.y, static_cast<float>(expected_type)) &&
            approximately_equal(
                meta.z, static_cast<float>(expected_fresnel)) &&
            approximately_equal(meta.w, expected_valid ? 1.0f : 0.0f) &&
            approximately_equal(legacy_meta.x,
                                static_cast<float>(expected_count)) &&
            approximately_equal(legacy_meta.y, static_cast<float>(expected_type)) &&
            approximately_equal(legacy_meta.w, expected_valid ? 1.0f : 0.0f) &&
            approximately_equal(
                collected[collected_base + 1u],
                luisa::float4{old[legacy_base + 1u].x, old[legacy_base + 1u].y,
                              old[legacy_base + 1u].z, legacy_meta.z}) &&
            approximately_equal(collected[collected_base + 2u].x,
                                old[legacy_base + 2u].x) &&
            approximately_equal(collected[collected_base + 2u].y,
                                old[legacy_base + 2u].y) &&
            approximately_equal(collected[collected_base + 2u].z,
                                old[legacy_base + 2u].z);
        auto all_finite = true;
        for (auto record = 0u; record < records_per_slot; ++record) {
            all_finite &= finite(collected[collected_base + record]);
        }
        const auto flags = collected[collected_base + 4u].w;
        const auto film = collected[collected_base + 6u];
        const auto expected_flags =
            !expected_valid
                ? 0.0f
            : material != 0u
                ? 5.0f
            : expected_type >= cycles_closure::type_microfacet_beckmann_glass &&
                      expected_type <=
                          cycles_closure::type_microfacet_ggx_glass
                ? 3.0f
                : 1.0f +
                      (requested == 2u || requested == 3u ||
                              requested == 5u
                          ? 2.0f
                          : 0.0f);
        if (!core_equal || !retained_equal || !all_finite ||
            !approximately_equal(flags, expected_flags) ||
            !approximately_equal(collected[collected_base + 5u].x, 0.0f) ||
            !approximately_equal(collected[collected_base + 5u].y, 0.0f) ||
            !approximately_equal(collected[collected_base + 5u].z, 1.0f) ||
            !approximately_equal(
                film,
                material != 0u && expected_valid
                    ? luisa::float4{425.0f, 1.32f, 0.0f, 0.0f}
                    : luisa::float4{0.0f})) {
            std::cerr << "surface closure collection failed on " << backend
                      << " at material " << material << ", closure " << requested
                      << ": collected {" << meta.x << ", " << meta.y << ", " << meta.z
                      << ", " << meta.w << "}, legacy {" << legacy_meta.x << ", "
                      << legacy_meta.y << ", " << legacy_meta.z << ", "
                      << legacy_meta.w << "}, flags " << flags << '\n';
            return EXIT_FAILURE;
        }
    }

    constexpr std::array glass_storage_expected{
        luisa::float4{2.0f,
            static_cast<float>(
                cycles_closure::type_microfacet_ggx_glass),
            static_cast<float>(cycles_closure::MicrofacetFresnel::
                                   generalized_schlick),
            1.0f},
        luisa::float4{1.0f, 2.0f, 3.0f, 0.4f},
        luisa::float4{4.0f, 5.0f, 6.0f, 0.5f},
        luisa::float4{7.0f, 8.0f, 9.0f, 0.19f},
        luisa::float4{10.0f, 11.0f, 12.0f, 0.0f},
        luisa::float4{13.0f, 14.0f, 15.0f, 0.0f},
        luisa::float4{16.0f, 17.0f, 18.0f, 1.37f},
        luisa::float4{23.0f, 24.0f, 25.0f, 0.22f},
        luisa::float4{28.0f, 29.0f, 30.0f, 0.26f},
        luisa::float4{31.0f, 32.0f, 33.0f, 0.27f},
        luisa::float4{34.0f, 35.0f, 36.0f, 3.0f},
        luisa::float4{37.0f, 38.0f, 39.0f, 0.56f},
        luisa::float4{40.0f, 41.0f, 42.0f, 0.57f},
        luisa::float4{53.0f, 54.0f, 55.0f, 0.0f}};
    const auto invalid_setup_retained =
        approximately_equal(stored[0u],
            luisa::float4{2.0f,
                static_cast<float>(cycles_closure::type_none),
                static_cast<float>(
                    cycles_closure::MicrofacetFresnel::none),
                1.0f}) &&
        approximately_equal(stored[1u],
            luisa::float4{0.1f, 0.2f, 0.3f, 0.2f}) &&
        approximately_equal(stored[2u],
            luisa::float4{0.4f, 0.5f, 0.6f, 0.0f}) &&
        approximately_equal(stored[6u],
            luisa::float4{0.0f, 1.0f, 0.0f, 1.2f}) &&
        approximately_equal(stored[10u].w, 0.0f) &&
        approximately_equal(stored[14u],
            luisa::float4{1.0f,
                static_cast<float>(cycles_closure::type_none),
                1.0f,
                0.0f});
    auto glass_round_trip = true;
    auto callable_round_trip = true;
    for (auto record = 0u;
         record < glass_storage_expected.size();
         ++record) {
        glass_round_trip &= approximately_equal(
            stored[storage_records_per_slot + record],
            glass_storage_expected[record]);
        callable_round_trip &= approximately_equal(
            stored[3u * storage_records_per_slot + record],
            glass_storage_expected[record]);
    }
    constexpr std::array bssrdf_storage_expected{
        luisa::float4{43.0f, 44.0f, 45.0f, 0.52f},
        luisa::float4{46.0f, 47.0f, 48.0f, 0.51f},
        luisa::float4{
            static_cast<float>(SurfaceBssrdfMethod::random_walk_skin),
            1.49f,
            0.0f,
            0.0f}};
    for (auto record = 0u;
         record < bssrdf_storage_expected.size();
         ++record) {
        const auto offset = 15u + record;
        glass_round_trip &= approximately_equal(
            stored[storage_records_per_slot + offset],
            bssrdf_storage_expected[record]);
        callable_round_trip &= approximately_equal(
            stored[3u * storage_records_per_slot + offset],
            bssrdf_storage_expected[record]);
    }
    constexpr std::array physical_storage_expected{
        luisa::float4{1.0f,
            static_cast<float>(
                cycles_closure::type_microfacet_ggx_glass),
            static_cast<float>(cycles_closure::MicrofacetFresnel::
                                   generalized_schlick),
            1.0f},
        luisa::float4{1.0f, 2.0f, 3.0f, 0.5f},
        luisa::float4{28.0f, 29.0f, 30.0f, 0.19f},
        luisa::float4{16.0f, 17.0f, 18.0f, 0.0f},
        luisa::float4{31.0f, 32.0f, 33.0f, 1.37f},
        luisa::float4{34.0f, 35.0f, 36.0f, 456.0f},
        luisa::float4{37.0f, 38.0f, 39.0f, 1.29f},
        luisa::float4{40.0f, 41.0f, 42.0f, 0.0f}};
    auto physical_round_trip = true;
    for (auto record = 0u;
         record < physical_storage_expected.size();
         ++record) {
        physical_round_trip &= approximately_equal(
            stored[4u * storage_records_per_slot + record],
            physical_storage_expected[record]);
    }
    physical_round_trip &= approximately_equal(
        stored[4u * storage_records_per_slot + 14u],
        luisa::float4{1.0f, 1.0f, 0.0f, 0.0f});
    const auto thin_film_round_trip =
        approximately_equal(
            stored[storage_records_per_slot + 18u],
            luisa::float4{456.0f, 1.29f, 0.0f, 0.0f}) &&
        approximately_equal(
            stored[3u * storage_records_per_slot + 18u],
            luisa::float4{456.0f, 1.29f, 0.0f, 0.0f});
    const auto overflow_truncated =
        approximately_equal(
            stored[2u * storage_records_per_slot],
            luisa::float4{2.0f,
                static_cast<float>(cycles_closure::type_none),
                static_cast<float>(
                    cycles_closure::MicrofacetFresnel::none),
                0.0f}) &&
        approximately_equal(
            stored[2u * storage_records_per_slot + 6u],
            luisa::float4{0.0f, 0.0f, 1.0f, 1.0f}) &&
        approximately_equal(
            stored[2u * storage_records_per_slot + 8u],
            luisa::float4{1.0f, 1.0f, 1.0f, 0.0f});
    if (!invalid_setup_retained ||
        !glass_round_trip ||
        !callable_round_trip ||
        !physical_round_trip ||
        !thin_film_round_trip ||
        !overflow_truncated) {
        std::cerr
            << "surface closure Local storage failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    for (auto record = std::size_t{0u};
         record < shared_evaluator.size();
         ++record) {
        if (!approximately_equal(
                shared_evaluator[record],
                legacy_evaluator[record])) {
            const auto actual = shared_evaluator[record];
            const auto expected = legacy_evaluator[record];
            std::cerr
                << "shared closure evaluator failed on "
                << backend << " at record " << record
                << ": shared {" << actual.x << ", "
                << actual.y << ", " << actual.z << ", "
                << actual.w << "}, legacy {" << expected.x
                << ", " << expected.y << ", " << expected.z
                << ", " << expected.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    for (auto record = std::size_t{0u};
         record < shared_scattering.size();
         ++record) {
        if (!finite(shared_scattering[record]) ||
            !approximately_equal(
                shared_scattering[record],
                legacy_scattering[record],
                1.0e-5f)) {
            const auto actual = shared_scattering[record];
            const auto expected = legacy_scattering[record];
            std::cerr
                << "shared closure scattering failed on "
                << backend << " at record " << record
                << ": shared {" << actual.x << ", "
                << actual.y << ", " << actual.z << ", "
                << actual.w << "}, legacy {" << expected.x
                << ", " << expected.y << ", " << expected.z
                << ", " << expected.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    for (auto record = std::size_t{0u};
         record < shared_sampling.size();
         ++record) {
        if (!finite(shared_sampling[record]) ||
            !approximately_equal(
                shared_sampling[record],
                legacy_sampling[record],
                1.0e-4f)) {
            const auto actual = shared_sampling[record];
            const auto expected = legacy_sampling[record];
            std::cerr
                << "shared closure sampling failed on "
                << backend << " at record " << record
                << ": shared {" << actual.x << ", "
                << actual.y << ", " << actual.z << ", "
                << actual.w << "}, legacy {" << expected.x
                << ", " << expected.y << ", " << expected.z
                << ", " << expected.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
