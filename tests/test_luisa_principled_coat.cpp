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
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;

constexpr std::uint32_t coat_record_count = 14u;
constexpr std::uint32_t coat_case_count = 9u;
constexpr std::uint32_t aggregate_record_count = 7u;

struct CoatCase {
    float weight;
    float roughness;
    float ior;
    luisa::float3 tint;
    float base_ior;
};

constexpr std::array coat_cases{
    CoatCase{1.0e-5f, 0.2f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{2.0e-5f, 0.2f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.2f, 1.0f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.0f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.01f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{0.7f, 0.02f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.0f},
    CoatCase{1.3f, 0.31f, 1.45f, {0.45f, 0.8f, 0.2f}, 1.0f},
    CoatCase{0.7f, 0.2f, 1.5f, {0.2f, 0.55f, 0.9f}, 1.0f},
    CoatCase{0.0f, 0.2f, 1.5f, {1.0f, 1.0f, 1.0f}, 1.5f}};

[[nodiscard]] ShaderGraph make_principled_coat_graph() {
    ShaderGraph graph;
    const auto principled = graph.add_node(
        node_type::principled_bsdf, "Principled Coat regression");
    const auto configured =
        graph.set_input(principled,
            "BaseColor",
            SocketValue::color({0.38f, 0.16f, 0.07f})) &&
        graph.set_input(
            principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Roughness", SocketValue::floating(0.4f)) &&
        graph.set_input(principled,
            "DiffuseRoughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "IOR", SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(principled,
            "SpecularTint",
            SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SheenWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "CoatWeight",
            SocketValue::floating(0.7f)) &&
        graph.set_input(principled,
            "CoatRoughness",
            SocketValue::floating(0.2f)) &&
        graph.set_input(principled,
            "CoatIOR",
            SocketValue::floating(1.5f)) &&
        graph.set_input(principled,
            "CoatTint",
            SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(principled,
            "SubsurfaceWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_property(principled,
            "Distribution",
            SocketValue::string("GGX"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure Principled Coat regression graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_glass_diffuse_graph() {
    ShaderGraph graph;
    const auto glass =
        graph.add_node(node_type::glass_bsdf, "Delta glass");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Competing diffuse");
    const auto add =
        graph.add_node(node_type::add_closure, "Glass plus diffuse");
    const auto configured =
        graph.set_input(glass,
            "Color",
            SocketValue::color({0.72f, 0.86f, 0.94f})) &&
        graph.set_input(
            glass, "Roughness", SocketValue::floating(0.0f)) &&
        graph.set_input(
            glass, "IOR", SocketValue::floating(1.45f)) &&
        graph.set_property(
            glass, "Distribution", SocketValue::string("GGX")) &&
        graph.set_input(diffuse,
            "Color",
            SocketValue::color({0.31f, 0.19f, 0.11f})) &&
        graph.set_input(
            diffuse, "Roughness", SocketValue::floating(0.0f)) &&
        graph.connect(
            {.node = glass, .socket = "Closure"}, add, "A") &&
        graph.connect(
            {.node = diffuse, .socket = "Closure"}, add, "B");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure delta-glass MIS graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = add, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_transparent_translucent_graph() {
    ShaderGraph graph;
    const auto transparent = graph.add_node(
        node_type::transparent_bsdf, "Delta transparent");
    const auto translucent = graph.add_node(
        node_type::translucent_bsdf, "Competing translucent");
    const auto add = graph.add_node(
        node_type::add_closure, "Transparent plus translucent");
    const auto configured =
        graph.set_input(transparent,
            "Color",
            SocketValue::color({0.42f, 0.36f, 0.24f})) &&
        graph.set_input(translucent,
            "Color",
            SocketValue::color({0.27f, 0.51f, 0.79f})) &&
        graph.connect(
            {.node = transparent, .socket = "Closure"}, add, "A") &&
        graph.connect(
            {.node = translucent, .socket = "Closure"}, add, "B");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure transparent MIS graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = add, .socket = "Closure"});
    return graph;
}

void set_parameter(std::vector<luisa::float4> &values,
    const SurfaceProgram &program,
    std::string_view socket,
    luisa::float4 value) {
    const auto &parameters = program.parameters();
    for (std::size_t index = 0u; index < parameters.size(); ++index) {
        if (parameters[index].socket == socket) {
            values[index] = value;
            return;
        }
    }
    throw std::runtime_error{"missing surface parameter: " +
                             std::string{socket}};
}

[[nodiscard]] std::vector<luisa::float4> coat_parameter_data(
    const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    const auto defaults = parameter_data(program);
    result.reserve(defaults.size() * coat_cases.size());
    for (const auto &test_case : coat_cases) {
        auto values = defaults;
        set_parameter(values,
            program,
            "CoatWeight",
            {test_case.weight, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "CoatRoughness",
            {test_case.roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "CoatIOR",
            {test_case.ior, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "CoatTint",
            {test_case.tint.x,
                test_case.tint.y,
                test_case.tint.z,
                0.0f});
        set_parameter(values,
            program,
            "IOR",
            {test_case.base_ior, 0.0f, 0.0f, 0.0f});
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

[[nodiscard]] bool rgb_equal(luisa::float4 lhs,
    luisa::float4 rhs,
    float tolerance = 3.0e-6f) noexcept {
    return approximately_equal(lhs.x, rhs.x, tolerance) &&
           approximately_equal(lhs.y, rhs.y, tolerance) &&
           approximately_equal(lhs.z, rhs.z, tolerance);
}

[[nodiscard]] bool rgb_positive(luisa::float4 value) noexcept {
    return value.x > 0.0f && value.y > 0.0f && value.z > 0.0f;
}

[[nodiscard]] bool finite_rgb(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};

    auto coat_shader = compiler.compile(make_principled_coat_graph());
    auto glass_shader = compiler.compile(make_glass_diffuse_graph());
    auto transparent_shader =
        compiler.compile(make_transparent_translucent_graph());
    if (!coat_shader.ok() || !glass_shader.ok() ||
        !transparent_shader.ok()) {
        std::cerr << "failed to compile Coat/MIS regression graphs\n";
        return EXIT_FAILURE;
    }
    auto coat_program =
        compile_surface_program(*coat_shader.program);
    auto glass_program =
        compile_surface_program(*glass_shader.program);
    auto transparent_program =
        compile_surface_program(*transparent_shader.program);
    if (!coat_program.ok() || !glass_program.ok() ||
        !transparent_program.ok()) {
        std::cerr << "failed to lower Coat/MIS surface programs\n";
        return EXIT_FAILURE;
    }

    const auto coat_parameters =
        coat_parameter_data(*coat_program.program);
    const auto glass_parameters =
        parameter_data(*glass_program.program);
    const auto transparent_parameters =
        parameter_data(*transparent_program.program);
    const auto coat_parameter_stride = static_cast<std::uint32_t>(
        coat_program.program->parameters().size());

    SurfaceDispatch surfaces;
    const auto coat_tag =
        surfaces.create<GraphSurface>(coat_program.program);
    const auto glass_tag =
        surfaces.create<GraphSurface>(glass_program.program);
    const auto transparent_tag =
        surfaces.create<GraphSurface>(transparent_program.program);

    Kernel1D evaluate_coat =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto case_index = dispatch_x();
            auto point = make_surface_point();
            point.parameter_block =
                case_index * coat_parameter_stride;
            const auto first = surfaces.closure_trace(
                UInt{coat_tag}, services, point, 0u, true);
            const auto second = surfaces.closure_trace(
                UInt{coat_tag}, services, point, 1u, true);
            const auto without_reflection = surfaces.closure_trace(
                UInt{coat_tag}, services, point, 0u, false);
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f,
                .reflective_caustics = true};
            const auto evaluation = surfaces.evaluate(
                UInt{coat_tag},
                services,
                point,
                make_float3(0.0f, 0.0f, 1.0f),
                query);
            const auto sample = surfaces.sample_trace(
                UInt{coat_tag},
                services,
                point,
                0.0f,
                make_float2(0.37f, 0.73f),
                query);
            const auto base = case_index * coat_record_count;
            output.write(base,
                make_float4(cast<float>(first.count),
                    cast<float>(first.type),
                    first.sample_weight,
                    select(0.0f, 1.0f, first.valid)));
            output.write(base + 1u,
                make_float4(first.weight,
                    cast<float>(first.runtime_flags)));
            output.write(base + 2u,
                make_float4(cast<float>(second.count),
                    cast<float>(second.type),
                    second.sample_weight,
                    select(0.0f, 1.0f, second.valid)));
            output.write(base + 3u,
                make_float4(second.weight, 0.0f));
            output.write(base + 4u,
                make_float4(cast<float>(without_reflection.count),
                    cast<float>(without_reflection.type),
                    without_reflection.sample_weight,
                    select(0.0f, 1.0f, without_reflection.valid)));
            output.write(base + 5u,
                make_float4(without_reflection.weight,
                    cast<float>(without_reflection.runtime_flags)));
            output.write(base + 6u,
                make_float4(evaluation.f, evaluation.pdf));
            output.write(base + 7u,
                make_float4(
                    evaluation.diffuse_f, evaluation.diffuse_pdf));
            output.write(base + 8u,
                make_float4(evaluation.glossy_f,
                    cast<float>(evaluation.events)));
            output.write(base + 9u,
                make_float4(sample.sample.evaluation.f,
                    sample.sample.evaluation.pdf));
            output.write(base + 10u,
                make_float4(sample.sample.evaluation.diffuse_f,
                    sample.sample.evaluation.diffuse_pdf));
            output.write(base + 11u,
                make_float4(sample.sample.evaluation.glossy_f,
                    cast<float>(sample.sample.evaluation.events)));
            output.write(base + 12u,
                make_float4(sample.sample.wi,
                    select(0.0f, 1.0f, sample.sample.valid)));
            output.write(base + 13u,
                make_float4(sample.sample.roughness,
                    sample.sample.eta,
                    sample.closure_sample_weight));
        };

    Kernel1D evaluate_aggregate =
        [&](BufferFloat4 glass_parameter_buffer,
            BufferFloat4 transparent_parameter_buffer,
            BufferFloat4 output) noexcept {
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f,
                .reflective_caustics = true};
            const auto point = make_surface_point();
            ParameterShaderServices glass_services{
                glass_parameter_buffer};
            const auto glass_evaluation = surfaces.evaluate(
                UInt{glass_tag},
                glass_services,
                point,
                make_float3(0.0f, 0.0f, 1.0f),
                query);
            const auto glass_sample = surfaces.sample_trace(
                UInt{glass_tag},
                glass_services,
                point,
                0.0f,
                make_float2(0.29f, 0.61f),
                query);
            ParameterShaderServices transparent_services{
                transparent_parameter_buffer};
            const auto transparent_evaluation = surfaces.evaluate(
                UInt{transparent_tag},
                transparent_services,
                point,
                make_float3(0.0f, 0.0f, -1.0f),
                query);
            const auto transparent_sample = surfaces.sample_trace(
                UInt{transparent_tag},
                transparent_services,
                point,
                0.0f,
                make_float2(0.29f, 0.61f),
                query);
            const auto write_result = [&](std::uint32_t base,
                                          const SurfaceEvaluation &evaluation,
                                          const SurfaceSampleTrace &sample) noexcept {
                output.write(base,
                    make_float4(evaluation.f, evaluation.pdf));
                output.write(base + 1u,
                    make_float4(
                        evaluation.diffuse_f, evaluation.diffuse_pdf));
                output.write(base + 2u,
                    make_float4(evaluation.glossy_f,
                        cast<float>(evaluation.events)));
                output.write(base + 3u,
                    make_float4(sample.sample.evaluation.f,
                        sample.sample.evaluation.pdf));
                output.write(base + 4u,
                    make_float4(sample.sample.evaluation.diffuse_f,
                        sample.sample.evaluation.diffuse_pdf));
                output.write(base + 5u,
                    make_float4(sample.sample.evaluation.glossy_f,
                        cast<float>(sample.sample.evaluation.events)));
                output.write(base + 6u,
                    make_float4(sample.sample.wi,
                        select(0.0f, 1.0f, sample.sample.valid)));
            };
            write_result(0u, glass_evaluation, glass_sample);
            write_result(aggregate_record_count,
                transparent_evaluation,
                transparent_sample);
        };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto coat_parameter_buffer =
        device.create_buffer<luisa::float4>(coat_parameters.size());
    auto glass_parameter_buffer =
        device.create_buffer<luisa::float4>(glass_parameters.size());
    auto transparent_parameter_buffer =
        device.create_buffer<luisa::float4>(
            transparent_parameters.size());
    auto coat_output = device.create_buffer<luisa::float4>(
        coat_record_count * coat_case_count);
    auto aggregate_output = device.create_buffer<luisa::float4>(
        aggregate_record_count * 2u);
    auto coat_kernel = device.compile(evaluate_coat);
    auto aggregate_kernel = device.compile(evaluate_aggregate);
    std::array<luisa::float4,
        coat_record_count * coat_case_count>
        coat_actual{};
    std::array<luisa::float4,
        aggregate_record_count * 2u>
        aggregate_actual{};
    stream << coat_parameter_buffer.copy_from(
                  luisa::span{coat_parameters})
           << glass_parameter_buffer.copy_from(
                  luisa::span{glass_parameters})
           << transparent_parameter_buffer.copy_from(
                  luisa::span{transparent_parameters})
           << coat_kernel(coat_parameter_buffer, coat_output)
                  .dispatch(coat_case_count)
           << coat_output.copy_to(luisa::span{coat_actual})
           << aggregate_kernel(glass_parameter_buffer,
                  transparent_parameter_buffer,
                  aggregate_output)
                  .dispatch(1u)
           << aggregate_output.copy_to(luisa::span{aggregate_actual})
           << synchronize();

    if (std::getenv("PSYCLES_DUMP_COAT_REGRESSION") != nullptr) {
        for (std::size_t index = 0u; index < coat_actual.size(); ++index) {
            const auto value = coat_actual[index];
            std::cerr << "coat " << index << ": {" << value.x << ", "
                      << value.y << ", " << value.z << ", " << value.w
                      << "}\n";
        }
        for (std::size_t index = 0u;
             index < aggregate_actual.size();
             ++index) {
            const auto value = aggregate_actual[index];
            std::cerr << "aggregate " << index << ": {" << value.x
                      << ", " << value.y << ", " << value.z << ", "
                      << value.w << "}\n";
        }
    }

    const auto coat = [&](std::uint32_t case_index,
                          std::uint32_t record) noexcept
        -> const luisa::float4 & {
        return coat_actual[case_index * coat_record_count + record];
    };
    const auto closure_meta_matches = [&](std::uint32_t case_index,
                                          std::uint32_t record,
                                          float count,
                                          std::uint32_t type,
                                          bool valid) noexcept {
        const auto value = coat(case_index, record);
        return approximately_equal(value.x, count) &&
               approximately_equal(value.y, static_cast<float>(type)) &&
               approximately_equal(value.w, valid ? 1.0f : 0.0f);
    };

    if (!closure_meta_matches(0u,
            0u,
            1.0f,
            cycles_closure::type_diffuse,
            true) ||
        !closure_meta_matches(
            0u, 2u, 1.0f, cycles_closure::type_none, false) ||
        !closure_meta_matches(1u,
            0u,
            2.0f,
            cycles_closure::type_microfacet_ggx,
            true) ||
        !closure_meta_matches(1u,
            2u,
            2.0f,
            cycles_closure::type_diffuse,
            true)) {
        std::cerr << "Cycles Coat cutoff/order regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!closure_meta_matches(2u,
            0u,
            2.0f,
            cycles_closure::type_microfacet_ggx,
            true) ||
        !approximately_equal(coat(2u, 0u).z, 0.0f) ||
        !closure_meta_matches(2u,
            2u,
            2.0f,
            cycles_closure::type_diffuse,
            true)) {
        std::cerr << "Cycles IOR-one Coat allocation regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto singular_evaluation = coat(3u, 6u);
    const auto singular_evaluation_diffuse = coat(3u, 7u);
    const auto singular_sample = coat(3u, 9u);
    const auto singular_sample_diffuse = coat(3u, 10u);
    const auto singular_sample_glossy = coat(3u, 11u);
    const auto singular_direction = coat(3u, 12u);
    const auto singular_roughness = coat(3u, 13u);
    constexpr auto singular_reflection_events =
        static_cast<float>(event_singular | event_reflection);
    if (!rgb_positive(singular_evaluation_diffuse) ||
        !rgb_equal(singular_sample_diffuse,
            singular_evaluation_diffuse) ||
        !rgb_positive(singular_sample_glossy) ||
        !approximately_equal(singular_sample.x,
            singular_sample_diffuse.x + singular_sample_glossy.x) ||
        !approximately_equal(singular_sample.y,
            singular_sample_diffuse.y + singular_sample_glossy.y) ||
        !approximately_equal(singular_sample.z,
            singular_sample_diffuse.z + singular_sample_glossy.z) ||
        !(singular_sample.w > singular_evaluation.w &&
            singular_sample.w < 1.0e6f) ||
        !approximately_equal(
            singular_sample_glossy.w, singular_reflection_events) ||
        !approximately_equal(singular_direction,
            luisa::float4{0.0f, 0.0f, 1.0f, 1.0f}) ||
        !approximately_equal(singular_roughness.x, 0.0f)) {
        std::cerr << "Cycles singular Coat MIS regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto narrow_glossy = coat(4u, 8u);
    const auto wider_glossy = coat(5u, 8u);
    if (!finite_rgb(narrow_glossy) || !finite_rgb(wider_glossy) ||
        !(narrow_glossy.x > 0.0f && wider_glossy.x > 0.0f) ||
        !approximately_equal(
            narrow_glossy.x / wider_glossy.x, 16.0f, 2.0e-5f)) {
        std::cerr << "Cycles stable GGX peak regression failed on "
                  << backend << ": ratio "
                  << narrow_glossy.x / wider_glossy.x << '\n';
        return EXIT_FAILURE;
    }

    const auto overdriven_diffuse = coat(6u, 3u);
    if (!(overdriven_diffuse.x > 0.0f &&
          overdriven_diffuse.y > 0.0f) ||
        !approximately_equal(overdriven_diffuse.z, 0.0f)) {
        std::cerr << "Cycles bsdf_alloc channel clamp regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto tinted_without_reflection_meta = coat(7u, 4u);
    const auto tinted_without_reflection = coat(7u, 5u);
    const auto dielectric_without_reflection_meta = coat(8u, 4u);
    const auto dielectric_without_reflection = coat(8u, 5u);
    if (!approximately_equal(tinted_without_reflection_meta.x, 1.0f) ||
        !approximately_equal(tinted_without_reflection_meta.y,
            static_cast<float>(cycles_closure::type_diffuse)) ||
        !(tinted_without_reflection.x < 0.38f &&
            tinted_without_reflection.y < 0.16f &&
            tinted_without_reflection.z < 0.07f) ||
        !approximately_equal(
            dielectric_without_reflection_meta.x, 1.0f) ||
        !approximately_equal(dielectric_without_reflection_meta.y,
            static_cast<float>(cycles_closure::type_diffuse)) ||
        !rgb_equal(dielectric_without_reflection,
            luisa::float4{0.38f, 0.16f, 0.07f, 0.0f})) {
        std::cerr << "Cycles reflective-caustics layer regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto check_reflective_delta =
        [&](std::uint32_t base, std::string_view label) {
            const auto evaluation = aggregate_actual[base];
            const auto evaluation_diffuse = aggregate_actual[base + 1u];
            const auto sample = aggregate_actual[base + 3u];
            const auto sample_diffuse = aggregate_actual[base + 4u];
            const auto sample_glossy = aggregate_actual[base + 5u];
            const auto direction = aggregate_actual[base + 6u];
            const auto ok = rgb_positive(evaluation_diffuse) &&
                            rgb_equal(sample_diffuse,
                                evaluation_diffuse) &&
                            rgb_positive(sample_glossy) &&
                            approximately_equal(sample.x,
                                sample_diffuse.x + sample_glossy.x) &&
                            approximately_equal(sample.y,
                                sample_diffuse.y + sample_glossy.y) &&
                            approximately_equal(sample.z,
                                sample_diffuse.z + sample_glossy.z) &&
                            sample.w > evaluation.w &&
                            sample.w < 1.0e6f &&
                            approximately_equal(sample_glossy.w,
                                singular_reflection_events) &&
                            approximately_equal(direction,
                                luisa::float4{
                                    0.0f, 0.0f, 1.0f, 1.0f});
            if (!ok) {
                std::cerr << "Cycles " << label
                          << " delta MIS regression failed on "
                          << backend << '\n';
            }
            return ok;
        };
    if (!check_reflective_delta(0u, "glass")) {
        return EXIT_FAILURE;
    }

    const auto transparent_base = aggregate_record_count;
    const auto transparent_evaluation =
        aggregate_actual[transparent_base];
    const auto transparent_evaluation_diffuse =
        aggregate_actual[transparent_base + 1u];
    const auto transparent_sample =
        aggregate_actual[transparent_base + 3u];
    const auto transparent_sample_diffuse =
        aggregate_actual[transparent_base + 4u];
    const auto transparent_sample_glossy =
        aggregate_actual[transparent_base + 5u];
    const auto transparent_direction =
        aggregate_actual[transparent_base + 6u];
    constexpr auto transparent_events =
        static_cast<float>(event_transmission | event_transparent);
    if (!rgb_positive(transparent_evaluation_diffuse) ||
        !rgb_equal(transparent_sample_diffuse,
            transparent_evaluation_diffuse) ||
        !(transparent_sample.x > transparent_sample_diffuse.x &&
            transparent_sample.y > transparent_sample_diffuse.y &&
            transparent_sample.z > transparent_sample_diffuse.z) ||
        !(transparent_sample.w > transparent_evaluation.w &&
            transparent_sample.w < 1.0e6f) ||
        !approximately_equal(transparent_sample_glossy,
            luisa::float4{0.0f, 0.0f, 0.0f, transparent_events}) ||
        !approximately_equal(transparent_direction,
            luisa::float4{0.0f, 0.0f, -1.0f, 1.0f})) {
        std::cerr << "Cycles transparent delta MIS regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
