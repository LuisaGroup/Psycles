#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/cycles_abi.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
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

constexpr std::uint32_t case_count = 7u;
constexpr std::uint32_t record_count = 51u;
constexpr luisa::float3 base_color{0.25f, 0.64f, 1.0f};
constexpr luisa::float3 specular_tint{0.5f, 1.0f, 0.25f};

[[nodiscard]] ShaderGraph make_thin_wall_graph() {
    ShaderGraph graph;
    const auto thin_wall_value = graph.add_node(
        node_type::constant_float,
        "Linked Cycles FLOAT to Boolean");
    const auto thin_wall_conversion = graph.add_node(
        node_type::scalar_to_boolean,
        "Cycles FLOAT to INT Boolean conversion");
    const auto normal_color = graph.add_node(
        node_type::constant_color,
        "Linked closure normal");
    const auto normal_vector = graph.add_node(
        node_type::color_to_vector,
        "Closure normal color to vector");
    const auto normal = graph.add_node(
        node_type::vector_to_normal,
        "Closure normal vector to normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Raw dynamic Thin Wall regression");
    const auto configured =
        graph.set_input(thin_wall_value,
            "Value",
            SocketValue::floating(0.0f)) &&
        graph.connect(
            OutputRef{.node = thin_wall_value, .socket = "Value"},
            thin_wall_conversion,
            "Value") &&
        graph.connect(
            OutputRef{
                .node = thin_wall_conversion,
                .socket = "Boolean"},
            principled,
            "ThinWall") &&
        graph.connect(
            OutputRef{.node = normal_color, .socket = "Color"},
            normal_vector,
            "Color") &&
        graph.connect(
            OutputRef{.node = normal_vector, .socket = "Vector"},
            normal,
            "Vector") &&
        graph.connect(
            OutputRef{.node = normal, .socket = "Normal"},
            principled,
            "Normal") &&
        graph.set_input(principled,
            "BaseColor",
            SocketValue::color(
                {base_color.x, base_color.y, base_color.z})) &&
        graph.set_input(principled,
            "Metallic",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "Roughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "DiffuseRoughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "SubsurfaceWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "SubsurfaceRadius",
            SocketValue::vector({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(principled,
            "SubsurfaceScale",
            SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SubsurfaceAnisotropy",
            SocketValue::floating(0.2f)) &&
        graph.set_input(principled,
            "TransmissionWeight",
            SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "IOR",
            SocketValue::floating(1.5f)) &&
        graph.set_input(principled,
            "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(principled,
            "SpecularTint",
            SocketValue::color(
                {specular_tint.x,
                    specular_tint.y,
                    specular_tint.z})) &&
        graph.set_input(principled,
            "Alpha",
            SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SheenWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "CoatWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_property(principled,
            "Distribution",
            SocketValue::string("GGX")) &&
        graph.set_property(principled,
            "SubsurfaceMethod",
            SocketValue::string("RANDOM_WALK"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure dynamic Thin Wall graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

void set_parameter(
    std::vector<luisa::float4> &values,
    const SurfaceProgram &program,
    std::string_view socket,
    luisa::float4 value) {
    for (std::size_t index = 0u;
         index < program.parameters().size();
         ++index) {
        if (program.parameters()[index].socket == socket) {
            values[index] = value;
            return;
        }
    }
    throw std::runtime_error{
        "missing Thin Wall parameter: " + std::string{socket}};
}

[[nodiscard]] std::vector<luisa::float4> make_parameters(
    const SurfaceProgram &program) {
    const auto defaults = parameter_data(program);
    std::vector<luisa::float4> result;
    result.reserve(defaults.size() * case_count);
    const auto append = [&](float thin_wall,
                            float transmission,
                            float roughness,
                            float subsurface,
                            float diffuse_roughness,
                            float ior,
                            float alpha,
                            luisa::float3 normal) {
        auto values = defaults;
        set_parameter(values,
            program,
            "Value",
            {thin_wall, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "TransmissionWeight",
            {transmission, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "Roughness",
            {roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "SubsurfaceWeight",
            {subsurface, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "DiffuseRoughness",
            {diffuse_roughness, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "IOR",
            {ior, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "Alpha",
            {alpha, 0.0f, 0.0f, 0.0f});
        set_parameter(values,
            program,
            "Color",
            {normal.x, normal.y, normal.z, 0.0f});
        result.insert(result.end(), values.begin(), values.end());
    };

    // Cycles converts FLOAT to INT by truncation before treating the target
    // Boolean socket as true/false. The non-integral values pin both sides of
    // that boundary and prevent replacement with Blender's generic > 0 cast.
    constexpr luisa::float3 geometric_normal{0.0f, 0.0f, 1.0f};
    append(0.75f, 1.0f, 0.0f, 0.0f, 0.0f, 1.5f, 1.0f,
        geometric_normal);
    append(1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.5f, 0.8f,
        geometric_normal);
    append(1.25f, 1.0f, 0.35f, 0.0f, 0.0f, 1.5f, 1.0f,
        geometric_normal);
    append(-1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        geometric_normal);
    append(1.0f, 0.0f, 0.0f, 1.0f, 0.4f, 1.0f, 1.0f,
        geometric_normal);
    append(-0.75f, 0.0f, 0.0f, 1.0f, 0.4f, 1.0f, 1.0f,
        geometric_normal);
    append(1.0f, 0.0f, 0.0f, 1.0f, 0.4f, 1.0f, 1.0f,
        {0.6f, 0.0f, 0.8f});
    return result;
}

[[nodiscard]] bool rgb_equal(
    luisa::float4 actual,
    luisa::float3 expected,
    float tolerance = 6.0e-5f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
}

[[nodiscard]] constexpr luisa::float3 scale(
    luisa::float3 value,
    float factor) noexcept {
    return {value.x * factor, value.y * factor, value.z * factor};
}

[[nodiscard]] constexpr luisa::float3 add(
    luisa::float3 lhs,
    luisa::float3 rhs) noexcept {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

[[nodiscard]] constexpr float average(
    luisa::float3 value) noexcept {
    return (value.x + value.y + value.z) / 3.0f;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(make_thin_wall_graph());
    if (!shader.ok()) {
        std::cerr << "failed to compile Thin Wall graph\n";
        return EXIT_FAILURE;
    }
    const auto lowered = compile_surface_program(*shader.program);
    if (!lowered.ok()) {
        std::cerr << "failed to lower Thin Wall surface program\n";
        return EXIT_FAILURE;
    }
    const auto &program = *lowered.program;
    const auto &closure = program.closure_instructions().front();
    bool raw_thin_wall_source = false;
    for (const auto &parameter : program.parameters()) {
        raw_thin_wall_source |= parameter.socket == "Value";
    }
    bool exact_float_to_boolean = false;
    for (const auto &instruction : program.value_instructions()) {
        exact_float_to_boolean |=
            instruction.operation ==
            ValueOperation::scalar_to_boolean;
    }
    if (!raw_thin_wall_source || !exact_float_to_boolean ||
        !closure.thin_wall.valid()) {
        std::cerr << "raw Thin Wall expression was lost before Luisa setup\n";
        return EXIT_FAILURE;
    }

    const auto parameters = make_parameters(program);
    const auto parameter_stride = static_cast<std::uint32_t>(
        program.parameters().size());
    SurfaceDispatch surfaces;
    const auto surface_tag = surfaces.create<GraphSurface>(lowered.program);

    Kernel1D evaluate = [&](BufferFloat4 parameter_buffer,
                            BufferFloat4 output) noexcept {
        // A unit table value isolates the closure algebra from GGX energy
        // table interpolation. Production scene regressions use the exact
        // versioned Cycles tables.
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto make_point = [&](std::uint32_t parameter_case) noexcept {
            auto point = make_surface_point();
            point.parameter_block = parameter_case * parameter_stride;
            return point;
        };
        const auto write_trace = [&](std::uint32_t record,
                                     const SurfaceClosureTrace &trace) noexcept {
            output.write(record,
                make_float4(cast<float>(trace.count),
                    cast<float>(trace.type),
                    trace.sample_weight,
                    select(0.0f, 1.0f, trace.valid)));
            output.write(record + 1u,
                make_float4(trace.weight, 0.0f));
        };
        constexpr auto all_lobes = static_cast<std::uint32_t>(
            event_diffuse | event_glossy | event_transmission |
            event_transparent);
        const auto query = SurfaceQuery{
            .lobe_mask = all_lobes,
            .transport_mode = static_cast<std::uint32_t>(
                TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto trace = [&](const SurfacePoint &point,
                               std::uint32_t index) noexcept {
            return surfaces.closure_trace(
                UInt{surface_tag},
                services,
                point,
                index,
                true,
                true);
        };
        const auto light = [&](const SurfacePoint &point,
                               Float3 outgoing,
                               std::uint32_t flags) noexcept {
            return surfaces.evaluate_light(
                UInt{surface_tag},
                services,
                point,
                outgoing,
                SurfaceLightQuery{
                    .surface = query,
                    .shader_flags =
                        flags | cycles_abi::shader_use_mis});
        };

        const auto thick_glass = make_point(0u);
        write_trace(0u, trace(thick_glass, 0u));

        const auto smooth_thin = make_point(1u);
        write_trace(2u, trace(smooth_thin, 0u));
        write_trace(4u, trace(smooth_thin, 1u));
        write_trace(6u, trace(smooth_thin, 2u));
        write_trace(8u, trace(smooth_thin, 3u));

        auto noncamera_thin = smooth_thin;
        noncamera_thin.ray_visibility = 1u << 1u;
        write_trace(10u, trace(noncamera_thin, 0u));
        write_trace(12u, trace(noncamera_thin, 1u));
        output.write(14u,
            make_float4(surfaces.transparent_extinction(
                            UInt{surface_tag},
                            services,
                            noncamera_thin),
                0.0f));

        const auto smooth_sample = surfaces.sample_trace(
            UInt{surface_tag},
            services,
            smooth_thin,
            0.99f,
            make_float2(0.37f, 0.73f),
            query);
        output.write(15u,
            make_float4(smooth_sample.sample.evaluation.f,
                smooth_sample.sample.evaluation.pdf));
        output.write(16u,
            make_float4(smooth_sample.sample.wi,
                select(0.0f, 1.0f, smooth_sample.sample.valid)));
        output.write(17u,
            make_float4(smooth_sample.sample.roughness.x,
                smooth_sample.sample.eta,
                cast<float>(smooth_sample.sample.evaluation.events),
                cast<float>(smooth_sample.closure_type)));

        const auto smooth_aov = surfaces.aov(
            UInt{surface_tag}, services, smooth_thin);
        output.write(18u,
            make_float4(smooth_aov.glossy_albedo,
                smooth_aov.roughness.x));
        output.write(19u,
            make_float4(smooth_aov.transmission_albedo,
                smooth_aov.roughness.y));
        output.write(20u,
            make_float4(smooth_aov.transparency, 0.0f));

        const auto rough_thin = make_point(2u);
        write_trace(21u, trace(rough_thin, 0u));
        write_trace(23u, trace(rough_thin, 1u));
        const auto rough_transmission = surfaces.evaluate(
            UInt{surface_tag},
            services,
            rough_thin,
            make_float3(0.0f, 0.0f, -1.0f),
            query);
        output.write(25u,
            make_float4(
                rough_transmission.f, rough_transmission.pdf));
        const auto rough_exclude_glossy = light(
            rough_thin,
            make_float3(0.0f, 0.0f, -1.0f),
            cycles_abi::shader_exclude_glossy);
        output.write(26u,
            make_float4(
                rough_exclude_glossy.f,
                rough_exclude_glossy.pdf));
        const auto rough_exclude_transmit = light(
            rough_thin,
            make_float3(0.0f, 0.0f, -1.0f),
            cycles_abi::shader_exclude_transmit);
        output.write(27u,
            make_float4(
                rough_exclude_transmit.f,
                rough_exclude_transmit.pdf));
        const auto rough_reflection = surfaces.evaluate(
            UInt{surface_tag},
            services,
            rough_thin,
            make_float3(0.0f, 0.0f, 1.0f),
            query);
        output.write(28u,
            make_float4(rough_reflection.f, rough_reflection.pdf));

        const auto smooth_subsurface = make_point(3u);
        write_trace(29u, trace(smooth_subsurface, 0u));
        write_trace(31u, trace(smooth_subsurface, 1u));

        const auto rough_subsurface = make_point(4u);
        write_trace(33u, trace(rough_subsurface, 0u));
        write_trace(35u, trace(rough_subsurface, 1u));
        const auto rough_subsurface_value = surfaces.evaluate(
            UInt{surface_tag},
            services,
            rough_subsurface,
            make_float3(0.0f, 0.0f, -1.0f),
            query);
        output.write(37u,
            make_float4(rough_subsurface_value.f,
                rough_subsurface_value.pdf));
        const auto subsurface_exclude_transmit = light(
            rough_subsurface,
            make_float3(0.0f, 0.0f, -1.0f),
            cycles_abi::shader_exclude_transmit);
        output.write(38u,
            make_float4(subsurface_exclude_transmit.f,
                subsurface_exclude_transmit.pdf));
        const auto subsurface_exclude_diffuse = light(
            rough_subsurface,
            make_float3(0.0f, 0.0f, -1.0f),
            cycles_abi::shader_exclude_diffuse);
        output.write(39u,
            make_float4(subsurface_exclude_diffuse.f,
                subsurface_exclude_diffuse.pdf));

        const auto rough_subsurface_sample = surfaces.sample_trace(
            UInt{surface_tag},
            services,
            rough_subsurface,
            0.9f,
            make_float2(0.23f, 0.79f),
            query);
        output.write(40u,
            make_float4(
                rough_subsurface_sample.sample.evaluation.f,
                rough_subsurface_sample.sample.evaluation.pdf));
        output.write(41u,
            make_float4(rough_subsurface_sample.sample.wi,
                select(0.0f,
                    1.0f,
                    rough_subsurface_sample.sample.valid)));
        output.write(42u,
            make_float4(
                rough_subsurface_sample.sample.roughness.x,
                rough_subsurface_sample.sample.eta,
                cast<float>(
                    rough_subsurface_sample.sample.evaluation.events),
                cast<float>(rough_subsurface_sample.closure_type)));

        const auto thick_subsurface = make_point(5u);
        write_trace(43u, trace(thick_subsurface, 0u));
        const auto subsurface_aov = surfaces.aov(
            UInt{surface_tag}, services, rough_subsurface);
        output.write(45u,
            make_float4(
                subsurface_aov.albedo, subsurface_aov.roughness.x));
        output.write(46u,
            make_float4(subsurface_aov.normal,
                subsurface_aov.transmission_albedo.x));

        // Cycles applies bump shadowing to an ordinary Rough Translucent
        // evaluation, but not to the selected bsdf_sample() contribution
        // because its event label contains LABEL_TRANSMIT. A tilted closure
        // normal makes the two contracts observably different.
        const auto tilted_subsurface = make_point(6u);
        auto uncorrected_tilted_subsurface = tilted_subsurface;
        uncorrected_tilted_subsurface.use_bump_map_correction = false;
        const auto tilted_sample = surfaces.sample_trace(
            UInt{surface_tag},
            services,
            tilted_subsurface,
            0.9f,
            make_float2(0.23f, 0.79f),
            query);
        const auto uncorrected_tilted_sample = surfaces.sample_trace(
            UInt{surface_tag},
            services,
            uncorrected_tilted_subsurface,
            0.9f,
            make_float2(0.23f, 0.79f),
            query);
        const auto tilted_evaluation = surfaces.evaluate(
            UInt{surface_tag},
            services,
            tilted_subsurface,
            tilted_sample.sample.wi,
            query);
        const auto uncorrected_tilted_evaluation = surfaces.evaluate(
            UInt{surface_tag},
            services,
            uncorrected_tilted_subsurface,
            tilted_sample.sample.wi,
            query);
        output.write(47u,
            make_float4(tilted_sample.sample.evaluation.f,
                tilted_sample.sample.evaluation.pdf));
        output.write(48u,
            make_float4(
                uncorrected_tilted_sample.sample.evaluation.f,
                uncorrected_tilted_sample.sample.evaluation.pdf));
        output.write(49u,
            make_float4(
                tilted_evaluation.f, tilted_evaluation.pdf));
        output.write(50u,
            make_float4(uncorrected_tilted_evaluation.f,
                uncorrected_tilted_evaluation.pdf));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto parameter_buffer =
        device.create_buffer<luisa::float4>(parameters.size());
    auto output_buffer =
        device.create_buffer<luisa::float4>(record_count);
    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, record_count> actual{};
    stream << parameter_buffer.copy_from(luisa::span{parameters})
           << kernel(parameter_buffer, output_buffer).dispatch(1u)
           << output_buffer.copy_to(luisa::span{actual})
           << synchronize();

    if (std::getenv("PSYCLES_DUMP_THIN_WALL_REGRESSION") != nullptr) {
        for (std::size_t index = 0u; index < actual.size(); ++index) {
            const auto value = actual[index];
            std::cerr << index << ": {" << value.x << ", "
                      << value.y << ", " << value.z << ", "
                      << value.w << "}\n";
        }
    }

    const auto meta = [&](std::uint32_t record,
                          float count,
                          std::uint32_t type,
                          bool valid) noexcept {
        return approximately_equal(actual[record].x, count) &&
               approximately_equal(
                   actual[record].y, static_cast<float>(type)) &&
               approximately_equal(
                   actual[record].w, valid ? 1.0f : 0.0f);
    };
    constexpr luisa::float3 reflectance{
        0.021200530013f,
        0.055109396494f,
        0.019801980198f};
    constexpr luisa::float3 transmittance{
        0.240106002650f,
        0.590210800550f,
        0.980198019802f};
    constexpr auto alpha = 0.8f;
    const auto thin_reflection = scale(reflectance, alpha);
    const auto thin_transmission = scale(transmittance, alpha);
    constexpr luisa::float3 alpha_transparency{0.2f, 0.2f, 0.2f};
    const auto merged_transparency = add(
        alpha_transparency, thin_transmission);

    if (!meta(0u,
            1.0f,
            cycles_closure::type_microfacet_ggx_glass,
            true) ||
        !meta(2u, 3.0f, cycles_closure::type_transparent, true) ||
        !rgb_equal(actual[3u], alpha_transparency) ||
        !meta(4u, 3.0f, cycles_closure::type_microfacet_ggx, true) ||
        !rgb_equal(actual[5u], thin_reflection) ||
        !meta(6u,
            3.0f,
            cycles_closure::type_thin_glass_transmission,
            true) ||
        !rgb_equal(actual[7u], thin_transmission) ||
        !meta(8u, 3.0f, cycles_closure::type_none, false)) {
        std::cerr << "dynamic Thin Wall camera topology regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!meta(10u, 2.0f, cycles_closure::type_transparent, true) ||
        !rgb_equal(actual[11u], merged_transparency) ||
        !meta(12u,
            2.0f,
            cycles_closure::type_microfacet_ggx,
            true) ||
        !rgb_equal(actual[13u], thin_reflection) ||
        !rgb_equal(actual[14u], merged_transparency)) {
        std::cerr << "smooth non-camera transparent merge regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto total_selection_weight =
        average(alpha_transparency) +
        average(thin_reflection) +
        average(thin_transmission);
    const auto transmission_probability =
        average(thin_transmission) / total_selection_weight;
    constexpr auto singular_transmission = static_cast<float>(
        event_singular | event_transmission);
    if (!rgb_equal(actual[15u], scale(thin_transmission, 1.0e6f)) ||
        !approximately_equal(
            actual[15u].w,
            transmission_probability * 1.0e6f,
            8.0e-5f) ||
        !approximately_equal(
            actual[16u], {0.0f, 0.0f, -1.0f, 1.0f}) ||
        !approximately_equal(actual[17u].x, 0.0f) ||
        !approximately_equal(actual[17u].y, 1.0f) ||
        !approximately_equal(actual[17u].z, singular_transmission) ||
        !approximately_equal(
            actual[17u].w,
            static_cast<float>(
                cycles_closure::type_thin_glass_transmission))) {
        std::cerr << "thin-glass singular sampling regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!rgb_equal(actual[18u], thin_reflection) ||
        !rgb_equal(actual[19u], thin_transmission) ||
        !rgb_equal(actual[20u], alpha_transparency)) {
        std::cerr << "thin-glass split AOV regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!meta(21u,
            2.0f,
            cycles_closure::type_microfacet_ggx,
            true) ||
        !meta(23u,
            2.0f,
            cycles_closure::type_thin_glass_transmission,
            true) ||
        !finite(actual[25u]) ||
        !finite(actual[28u]) ||
        !(actual[25u].w > 0.0f) ||
        !(actual[28u].w > 0.0f) ||
        !approximately_equal(actual[26u], actual[25u], 8.0e-5f) ||
        !rgb_equal(actual[27u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(
            actual[27u].w, actual[25u].w, 8.0e-5f)) {
        std::cerr << "rough thin-glass eval/type classifier regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto subsurface_reflection = scale(base_color, 0.4f);
    const auto subsurface_transmission = scale(base_color, 0.6f);
    if (!meta(29u, 2.0f, cycles_closure::type_diffuse, true) ||
        !rgb_equal(actual[30u], subsurface_reflection) ||
        !meta(31u, 2.0f, cycles_closure::type_translucent, true) ||
        !rgb_equal(actual[32u], subsurface_transmission) ||
        !meta(33u, 2.0f, cycles_closure::type_oren_nayar, true) ||
        !rgb_equal(actual[34u], subsurface_reflection) ||
        !meta(35u,
            2.0f,
            cycles_closure::type_rough_translucent,
            true) ||
        !rgb_equal(actual[36u], subsurface_transmission)) {
        std::cerr << "thin-subsurface topology/anisotropy regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    // Rough Translucent is in Cycles' diffuse closure interval even though
    // its sample label transmits. EXCLUDE_TRANSMIT retains it;
    // EXCLUDE_DIFFUSE removes its value but preserves the mixture PDF.
    if (!finite(actual[37u]) || !(actual[37u].w > 0.0f) ||
        !approximately_equal(actual[38u], actual[37u], 8.0e-5f) ||
        !rgb_equal(actual[39u], {0.0f, 0.0f, 0.0f}) ||
        !approximately_equal(
            actual[39u].w, actual[37u].w, 8.0e-5f) ||
        !finite(actual[40u]) || !(actual[40u].w > 0.0f) ||
        !(actual[41u].z < 0.0f) ||
        !approximately_equal(actual[41u].w, 1.0f) ||
        !approximately_equal(
            actual[42u].z,
            static_cast<float>(event_diffuse | event_transmission)) ||
        !approximately_equal(
            actual[42u].w,
            static_cast<float>(
                cycles_closure::type_rough_translucent))) {
        std::cerr << "rough-translucent eval/sample classifier regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    if (!meta(43u,
            1.0f,
            cycles_closure::type_bssrdf_random_walk,
            true) ||
        !rgb_equal(actual[45u], base_color) ||
        !approximately_equal(actual[45u].w, 1.0f) ||
        !approximately_equal(
            actual[46u], {0.0f, 0.0f, -1.0f, 0.0f})) {
        std::cerr << "thick/thin subsurface dispatch and AOV regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto corrected_sample = actual[47u];
    const auto uncorrected_sample = actual[48u];
    const auto corrected_evaluation = actual[49u];
    const auto uncorrected_evaluation = actual[50u];
    if (!approximately_equal(
            corrected_sample, uncorrected_sample, 8.0e-5f) ||
        !approximately_equal(
            corrected_sample, uncorrected_evaluation, 8.0e-5f) ||
        !(corrected_evaluation.x <
            uncorrected_evaluation.x - 1.0e-5f) ||
        !approximately_equal(corrected_evaluation.w,
            uncorrected_evaluation.w,
            8.0e-5f)) {
        std::cerr << "Cycles transmissive sample bump-shadowing regression failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
