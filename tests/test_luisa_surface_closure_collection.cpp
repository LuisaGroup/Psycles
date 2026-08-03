#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
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
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;

constexpr auto closure_slots = 8u;
constexpr auto records_per_slot = 6u;

struct CollectedClosureTrace {
    UInt count;
    SurfaceClosureRecord closure;
    Bool valid;
};

// Diagnostic consumer of the new host-stage collection boundary. Selection
// is runtime-indexed so this exercises the same dynamic record identity that
// the shared production closure set will consume.
class RequestedClosureCollector final : public SurfaceClosureCollector {

private:
    UInt _requested;
    UInt _count{0u};
    SurfaceClosureRecord _selected{SurfaceClosureRecord::zero()};
    Bool _valid{false};

public:
    explicit RequestedClosureCollector(UInt requested) noexcept
        : _requested{requested} {}

    void add(const SurfaceClosureRecord &closure) noexcept override {
        const auto scattering =
            closure.kind != static_cast<std::uint32_t>(SurfaceClosureKind::none);
        const auto allocated = scattering & (closure.allocation_weight >=
                                             cycles_closure::closure_weight_cutoff);
        const auto match = allocated & (_count == _requested);
        _selected.kind = select(_selected.kind, closure.kind, match);
        _selected.lobe = select(_selected.lobe, closure.lobe, match);
        _selected.weight = select(_selected.weight, closure.weight, match);
        _selected.allocation_weight =
            select(_selected.allocation_weight, closure.allocation_weight, match);
        _selected.sample_weight =
            select(_selected.sample_weight, closure.sample_weight, match);
        _selected.setup_valid =
            select(_selected.setup_valid, closure.setup_valid, match);
        _selected.albedo = select(_selected.albedo, closure.albedo, match);
        _selected.reflection_albedo =
            select(_selected.reflection_albedo, closure.reflection_albedo, match);
        _selected.transmission_albedo = select(_selected.transmission_albedo,
                                               closure.transmission_albedo, match);
        _selected.color = select(_selected.color, closure.color, match);
        _selected.normal = select(_selected.normal, closure.normal, match);
        _selected.roughness = select(_selected.roughness, closure.roughness, match);
        _selected.diffuse_roughness =
            select(_selected.diffuse_roughness, closure.diffuse_roughness, match);
        _selected.metallic = select(_selected.metallic, closure.metallic, match);
        _selected.ior = select(_selected.ior, closure.ior, match);
        _selected.specular_ior_level =
            select(_selected.specular_ior_level, closure.specular_ior_level, match);
        _selected.specular_tint =
            select(_selected.specular_tint, closure.specular_tint, match);
        _selected.sheen_transform_a =
            select(_selected.sheen_transform_a, closure.sheen_transform_a, match);
        _selected.sheen_transform_b =
            select(_selected.sheen_transform_b, closure.sheen_transform_b, match);
        _selected.evaluation_scale =
            select(_selected.evaluation_scale, closure.evaluation_scale, match);
        _selected.fresnel_f0 =
            select(_selected.fresnel_f0, closure.fresnel_f0, match);
        _selected.fresnel_f90 =
            select(_selected.fresnel_f90, closure.fresnel_f90, match);
        _selected.reflection_tint =
            select(_selected.reflection_tint, closure.reflection_tint, match);
        _selected.transmission_tint =
            select(_selected.transmission_tint, closure.transmission_tint, match);
        _selected.preserve_ggx_energy = select(_selected.preserve_ggx_energy,
                                               closure.preserve_ggx_energy, match);
        _selected.beckmann = select(_selected.beckmann, closure.beckmann, match);
        _valid |= match;
        _count += select(0u, 1u, allocated);
    }

    [[nodiscard]] CollectedClosureTrace result() const noexcept {
        return {.count = _count, .closure = _selected, .valid = _valid};
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

    SurfaceDispatch surfaces;
    const auto layered_tag = surfaces.create<GraphSurface>(layered);
    const auto glass_tag = surfaces.create<GraphSurface>(glass);
    auto parameters = parameter_data(*layered);
    const auto glass_parameter_base =
        static_cast<std::uint32_t>(parameters.size());
    const auto glass_parameters = parameter_data(*glass);
    parameters.insert(parameters.end(), glass_parameters.begin(),
                      glass_parameters.end());

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
        const auto &closure = trace.closure;
        const auto base = invocation * records_per_slot;
        output.write(base, make_float4(cast<float>(trace.count),
                                       cast<float>(closure.kind),
                                       cast<float>(closure.lobe),
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
                     make_float4(collection.shading_normal, closure.ior));
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
    auto collect_kernel = device.compile(collect);
    auto legacy_kernel = device.compile(legacy);
    std::array<luisa::float4, invocation_count * records_per_slot> collected{};
    std::array<luisa::float4, invocation_count * 3u> old{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << collect_kernel(parameter_buffer, collected_buffer)
                  .dispatch(invocation_count)
           << collected_buffer.copy_to(luisa::span{collected})
           << legacy_kernel(parameter_buffer, legacy_buffer)
                  .dispatch(invocation_count)
           << legacy_buffer.copy_to(luisa::span{old}) << synchronize();

    constexpr std::array layered_kinds{
        SurfaceClosureKind::transparent, SurfaceClosureKind::principled,
        SurfaceClosureKind::principled, SurfaceClosureKind::principled,
        SurfaceClosureKind::glass, SurfaceClosureKind::principled,
        SurfaceClosureKind::diffuse};
    constexpr std::array layered_lobes{
        SurfaceClosureLobe::none, SurfaceClosureLobe::sheen,
        SurfaceClosureLobe::coat, SurfaceClosureLobe::metallic,
        SurfaceClosureLobe::transmission, SurfaceClosureLobe::dielectric,
        SurfaceClosureLobe::none};
    constexpr std::array layered_types{
        cycles_closure::type_transparent,
        cycles_closure::type_sheen,
        cycles_closure::type_microfacet_ggx,
        cycles_closure::type_microfacet_ggx,
        cycles_closure::type_microfacet_multi_ggx_glass,
        cycles_closure::type_microfacet_ggx,
        cycles_closure::type_oren_nayar};

    for (auto invocation = 0u; invocation < invocation_count; ++invocation) {
        const auto material = invocation / closure_slots;
        const auto requested = invocation % closure_slots;
        const auto collected_base = invocation * records_per_slot;
        const auto legacy_base = invocation * 3u;
        const auto meta = collected[collected_base];
        const auto legacy_meta = old[legacy_base];
        const auto expected_count = material == 0u ? 7u : 1u;
        const auto expected_valid = requested < expected_count;
        const auto expected_kind =
            material == 0u && expected_valid
                ? layered_kinds[requested]
            : material != 0u && expected_valid
                ? SurfaceClosureKind::glass
                : SurfaceClosureKind::none;
        const auto expected_lobe =
            material == 0u && expected_valid
                ? layered_lobes[requested]
                : SurfaceClosureLobe::none;
        const auto expected_type =
            material == 0u && expected_valid
                ? layered_types[requested]
            : material != 0u && expected_valid
                ? cycles_closure::type_microfacet_beckmann_glass
                : cycles_closure::type_none;
        const auto core_equal =
            approximately_equal(meta.x, static_cast<float>(expected_count)) &&
            approximately_equal(meta.y, static_cast<float>(expected_kind)) &&
            approximately_equal(meta.z, static_cast<float>(expected_lobe)) &&
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
        const auto expected_flags =
            !expected_valid
                ? 0.0f
            : material != 0u
                ? 5.0f
            : expected_kind == SurfaceClosureKind::glass
                ? 3.0f
                : 1.0f +
                      (requested == 2u || requested == 3u ||
                              requested == 5u
                          ? 2.0f
                          : 0.0f);
        if (!core_equal || !all_finite ||
            !approximately_equal(flags, expected_flags) ||
            !approximately_equal(collected[collected_base + 5u].x, 0.0f) ||
            !approximately_equal(collected[collected_base + 5u].y, 0.0f) ||
            !approximately_equal(collected[collected_base + 5u].z, 1.0f)) {
            std::cerr << "surface closure collection failed on " << backend
                      << " at material " << material << ", closure " << requested
                      << ": collected {" << meta.x << ", " << meta.y << ", " << meta.z
                      << ", " << meta.w << "}, legacy {" << legacy_meta.x << ", "
                      << legacy_meta.y << ", " << legacy_meta.z << ", "
                      << legacy_meta.w << "}, flags " << flags << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
