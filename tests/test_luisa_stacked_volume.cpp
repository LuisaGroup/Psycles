#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/homogeneous_volume_segment.h>
#include <psycles/luisa/stacked_volume.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;

inline constexpr std::size_t record_count = 22u;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

struct GraphFixture {
    ShaderGraph graph;
    NodeId primary_scatter;
};

[[nodiscard]] GraphFixture make_graph() {
    ShaderGraph graph;
    const auto surface =
        graph.add_node(
            node_type::diffuse_bsdf,
            "Surface");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = surface,
            .socket = "Closure"});

    const auto primary =
        graph.add_node(
            node_type::volume_scatter,
            "Primary HG");
    require(
        graph.set_input(
            primary,
            "Color",
            SocketValue::color(
                {0.2f, 0.4f, 0.6f})) &&
            graph.set_input(
                primary,
                "Density",
                SocketValue::floating(1.0f)) &&
            graph.set_input(
                primary,
                "Anisotropy",
                SocketValue::floating(0.42f)) &&
            graph.set_property(
                primary,
                "Phase",
                SocketValue::string(
                    "HENYEY_GREENSTEIN")),
        "failed to configure primary volume scatter");

    const auto secondary =
        graph.add_node(
            node_type::volume_scatter,
            "Secondary HG");
    require(
        graph.set_input(
            secondary,
            "Color",
            SocketValue::color(
                {0.3f, 0.1f, 0.2f})) &&
            graph.set_input(
                secondary,
                "Density",
                SocketValue::floating(0.5f)) &&
            graph.set_input(
                secondary,
                "Anisotropy",
                SocketValue::floating(0.42f)) &&
            graph.set_property(
                secondary,
                "Phase",
                SocketValue::string(
                    "HENYEY_GREENSTEIN")),
        "failed to configure secondary volume scatter");

    const auto absorption =
        graph.add_node(
            node_type::volume_absorption,
            "Absorption");
    require(
        graph.set_input(
            absorption,
            "Color",
            SocketValue::color(
                {0.5f, 0.25f, 0.0f})) &&
            graph.set_input(
                absorption,
                "Density",
                SocketValue::floating(2.0f)),
        "failed to configure volume absorption");

    const auto coefficients =
        graph.add_node(
            node_type::volume_coefficients,
            "Emission");
    require(
        graph.set_input(
            coefficients,
            "ScatterCoefficients",
            SocketValue::vector(
                {0.0f, 0.0f, 0.0f})) &&
            graph.set_input(
                coefficients,
                "AbsorptionCoefficients",
                SocketValue::vector(
                    {0.0f, 0.0f, 0.0f})) &&
            graph.set_input(
                coefficients,
                "EmissionCoefficients",
                SocketValue::vector(
                    {0.1f, 0.2f, 0.3f})),
        "failed to configure volume emission");

    const auto add_scatter =
        graph.add_node(
            node_type::add_volume,
            "Add Scatter");
    const auto add_absorption =
        graph.add_node(
            node_type::add_volume,
            "Add Absorption");
    const auto add_emission =
        graph.add_node(
            node_type::add_volume,
            "Add Emission");
    require(
        graph.connect(
            {.node = primary,
             .socket = "Volume"},
            add_scatter,
            "A") &&
            graph.connect(
                {.node = secondary,
                 .socket = "Volume"},
                add_scatter,
                "B") &&
            graph.connect(
                {.node = add_scatter,
                 .socket = "Volume"},
                add_absorption,
                "A") &&
            graph.connect(
                {.node = absorption,
                 .socket = "Volume"},
                add_absorption,
                "B") &&
            graph.connect(
                {.node = add_absorption,
                 .socket = "Volume"},
                add_emission,
                "A") &&
            graph.connect(
                {.node = coefficients,
                 .socket = "Volume"},
                add_emission,
                "B"),
        "failed to connect stacked-volume graph");
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{
            .node = add_emission,
            .socket = "Volume"});
    return {
        .graph = std::move(graph),
        .primary_scatter = primary};
}

[[nodiscard]] std::vector<luisa::float4>
parameter_data(const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    result.reserve(program.parameters().size());
    for (const auto &parameter :
         program.parameters()) {
        const auto &value =
            parameter.default_value;
        if (const auto *scalar =
                std::get_if<float>(
                    &value.value)) {
            result.emplace_back(
                *scalar,
                0.0f,
                0.0f,
                0.0f);
        } else if (
            const auto *vector =
                std::get_if<Vec3f>(
                    &value.value)) {
            result.emplace_back(
                vector->x,
                vector->y,
                vector->z,
                0.0f);
        } else {
            throw std::runtime_error{
                "stacked-volume fixture has an unsupported parameter type"};
        }
    }
    return result;
}

class FixtureShaderServices final
    : public ShaderServices {

  private:
    const BufferFloat4 &_parameters;

  public:
    explicit FixtureShaderServices(
        const BufferFloat4 &parameters) noexcept
        : _parameters{parameters} {}

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<std::uint64_t>,
        const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot)
        const noexcept override {
        return _parameters
            .read(block + slot)
            .x;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot)
        const noexcept override {
        return _parameters
            .read(block + slot)
            .xyz();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>)
        const noexcept override {
        return 1.0f;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value)
        const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value)
        const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>)
        const noexcept override {
        return make_float3(0.0f);
    }
};

class FixturePointProvider final
    : public VolumeStackEntryPointProvider {

  public:
    VolumeStackEntryShading emit(
        const VolumeStackEntry &entry,
        const VolumeShadingState &state)
        const noexcept override {
        const auto object_entry =
            entry.instance_id !=
            invalid_volume_identity;
        const auto object_density =
            select(1.0f, 2.0f, object_entry);
        SurfacePoint point{};
        point.position = state.position;
        point.object_position =
            state.position /
            object_density;
        point.object_location =
            select(
                make_float3(0.0f),
                make_float3(1.0f, 2.0f, 3.0f),
                object_entry);
        point.generated =
            point.object_position;
        point.geometric_normal =
            state.incoming;
        point.shading_normal =
            state.incoming;
        point.incoming = state.incoming;
        point.geometry_index =
            invalid_volume_identity;
        point.instance_id =
            entry.instance_id;
        point.primitive_id =
            invalid_volume_identity;
        point.parameter_block =
            entry.parameter_block;
        point.ray_visibility =
            state.ray_visibility;
        point.ray_events =
            state.ray_events;
        point.ray_depth =
            state.ray_depth;
        point.diffuse_depth =
            state.diffuse_depth;
        point.glossy_depth =
            state.glossy_depth;
        point.transparent_depth =
            state.transparent_depth;
        point.transmission_depth =
            state.transmission_depth;
        point.ray_length =
            state.ray_length;
        point.time = state.time;
        return {
            .point = std::move(point),
            .object_density =
                std::move(object_density)};
    }
};

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 5.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

[[nodiscard]] bool approximately_equal(
    luisa::float4 actual,
    luisa::float4 expected,
    float tolerance = 5.0e-6f) noexcept {
    return
        approximately_equal(
            actual.x, expected.x, tolerance) &&
        approximately_equal(
            actual.y, expected.y, tolerance) &&
        approximately_equal(
            actual.z, expected.z, tolerance) &&
        approximately_equal(
            actual.w, expected.w, tolerance);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{
            argc > 1 ? argv[1] : "fallback"};
    auto fixture = make_graph();
    ShaderCompiler compiler{
        make_core_node_registry()};
    const auto shader =
        compiler.compile(fixture.graph);
    if (!shader.ok()) {
        std::cerr
            << "failed to compile stacked-volume graph\n";
        return EXIT_FAILURE;
    }
    const auto program =
        compile_surface_program(*shader.program);
    if (!program.ok()) {
        std::cerr
            << "failed to lower stacked-volume graph\n";
        return EXIT_FAILURE;
    }

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(
            program.program);
    if (!surfaces
             .capabilities(surface_tag)
             .may_have_volume) {
        std::cerr
            << "stacked-volume capability was lost\n";
        return EXIT_FAILURE;
    }

    auto base_parameters =
        parameter_data(*program.program);
    auto host_parameters = base_parameters;
    host_parameters.insert(
        host_parameters.end(),
        base_parameters.begin(),
        base_parameters.end());
    auto primary_density_found = false;
    for (const auto &parameter :
         program.program->parameters()) {
        if (parameter.node ==
                fixture.primary_scatter &&
            parameter.socket == "Density") {
            host_parameters[
                base_parameters.size() +
                parameter.id.value]
                .x = 2.0f;
            primary_density_found = true;
        }
    }
    if (!primary_density_found) {
        std::cerr
            << "primary density parameter was not retained\n";
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device =
        context.create_device(backend);
    auto stream = device.create_stream();
    auto parameters =
        device.create_buffer<luisa::float4>(
            host_parameters.size());
    auto output =
        device.create_buffer<luisa::float4>(
            record_count);

    auto point_provider =
        std::make_shared<
            FixturePointProvider>();
    StackedVolumeEvaluator evaluator{
        surfaces, *point_provider};
    auto segment =
        make_homogeneous_volume_segment_component(
            surfaces,
            point_provider,
            64u);
    Kernel1D evaluate =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 records) noexcept {
            FixtureShaderServices services{
                parameter_buffer};
            const VolumeShadingState state{
                .position =
                    make_float3(
                        2.0f, 4.0f, 6.0f),
                .incoming =
                    -normalize(
                        make_float3(
                            0.2f,
                            -0.3f,
                            0.9327379f)),
                .ray_visibility = 17u,
                .ray_events =
                    static_cast<std::uint32_t>(
                        contract::event_transmission),
                .ray_depth = 3u,
                .diffuse_depth = 1u,
                .glossy_depth = 2u,
                .transparent_depth = 4u,
                .transmission_depth = 2u,
                .ray_length = 5.0f,
                .time = 0.25f};
            const auto make_entry =
                [&](std::uint32_t object,
                    std::uint32_t parameter_block,
                    std::uint32_t instance) {
                    return VolumeStackEntry{
                        .object = object,
                        .shader = object + 100u,
                        .surface_tag =
                            surface_tag,
                        .parameter_block =
                            parameter_block,
                        .instance_id = instance,
                        .valid = true};
                };
            const auto world_entry =
                make_entry(
                    10u,
                    0u,
                    invalid_volume_identity);
            const auto object_entry =
                make_entry(
                    20u,
                    static_cast<std::uint32_t>(
                        base_parameters.size()),
                    7u);

            VolumeStack single{3u};
            single.initialize_background(
                world_entry, true);
            VolumePhaseSet single_phases{8u};
            const auto single_coefficients =
                evaluator.evaluate(
                    single,
                    services,
                    state,
                    true,
                    &single_phases);
            records.write(
                0u,
                make_float4(
                    single_coefficients.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        single_coefficients
                            .has_extinction)));
            records.write(
                1u,
                make_float4(
                    single_coefficients.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        single_coefficients
                            .has_scatter)));
            records.write(
                2u,
                make_float4(
                    single_coefficients.emission,
                    select(
                        0.0f,
                        1.0f,
                        single_coefficients
                            .has_emission)));
            const auto single_phase_0 =
                single_phases.entry(0u);
            const auto single_phase_1 =
                single_phases.entry(1u);
            records.write(
                3u,
                make_float4(
                    cast<float>(
                        single_phases.count()),
                    cast<float>(single.count()),
                    cast<float>(
                        single_phase_0.type),
                    cast<float>(
                        single_phase_1.type)));
            records.write(
                4u,
                make_float4(
                    single_phase_0.weight,
                    single_phase_0
                        .sample_weight));
            records.write(
                5u,
                make_float4(
                    single_phase_1.weight,
                    single_phase_1
                        .sample_weight));
            records.write(
                6u,
                make_float4(
                    single_phase_0
                        .parameters.x,
                    single_phase_1
                        .parameters.x,
                    select(
                        0.0f,
                        1.0f,
                        single_phase_0.valid),
                    select(
                        0.0f,
                        1.0f,
                        single_phase_1.valid)));

            VolumeStack stacked{3u};
            stacked.initialize_background(
                world_entry, true);
            stacked.cross_boundary(
                object_entry,
                false,
                true,
                true);
            VolumePhaseSet stacked_phases{8u};
            const auto stacked_coefficients =
                evaluator.evaluate(
                    stacked,
                    services,
                    state,
                    true,
                    &stacked_phases);
            records.write(
                7u,
                make_float4(
                    stacked_coefficients.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        stacked_coefficients
                            .has_extinction)));
            records.write(
                8u,
                make_float4(
                    stacked_coefficients.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        stacked_coefficients
                            .has_scatter)));
            records.write(
                9u,
                make_float4(
                    stacked_coefficients.emission,
                    select(
                        0.0f,
                        1.0f,
                        stacked_coefficients
                            .has_emission)));
            const auto stacked_phase =
                stacked_phases.entry(0u);
            records.write(
                10u,
                make_float4(
                    cast<float>(
                        stacked_phases.count()),
                    cast<float>(
                        stacked.count()),
                    cast<float>(
                        stacked_phase.type),
                    stacked_phase
                        .parameters.x));
            records.write(
                11u,
                make_float4(
                    stacked_phase.weight,
                    stacked_phase
                        .sample_weight));

            const auto extinction_only =
                evaluator.evaluate(
                    stacked,
                    services,
                    state,
                    false);
            records.write(
                12u,
                make_float4(
                    extinction_only.emission,
                    select(
                        0.0f,
                        1.0f,
                        extinction_only
                            .has_emission)));

            VolumeStack empty{3u};
            VolumePhaseSet empty_phases{8u};
            const auto empty_coefficients =
                evaluator.evaluate(
                    empty,
                    services,
                    state,
                    true,
                    &empty_phases);
            records.write(
                13u,
                make_float4(
                    empty_coefficients
                        .sigma_t.x,
                    empty_coefficients
                        .sigma_s.x,
                    empty_coefficients
                        .emission.x,
                    cast<float>(
                        empty_phases.count())));
            records.write(
                14u,
                make_float4(
                    select(
                        0.0f,
                        1.0f,
                        empty_coefficients
                            .has_extinction),
                    select(
                        0.0f,
                        1.0f,
                        empty_coefficients
                            .has_scatter),
                    select(
                        0.0f,
                        1.0f,
                        empty_coefficients
                            .has_emission),
                    cast<float>(empty.count())));

            const auto collision =
                segment->emit(
                    single,
                    services,
                    state,
                    0.5f,
                    make_float3(1.0f),
                    0.2f,
                    0.2f,
                    make_float2(
                        0.034f, 0.83f),
                    false,
                    {.scattered_radiance =
                         make_float3(0.0f),
                     .transmitted_radiance =
                         make_float3(0.0f),
                     .majorant_optical_depth =
                         0.0f,
                     .enabled = false},
                    false,
                    make_float3(0.0f));
            records.write(
                15u,
                make_float4(
                    collision.transport
                        .throughput,
                    select(
                        0.0f,
                        1.0f,
                        collision.scattered)));
            records.write(
                16u,
                make_float4(
                    collision.transport.emission,
                    collision.transport.distance));
            records.write(
                17u,
                make_float4(
                    collision.transport
                        .transmittance,
                    collision.transport
                        .event_pdf));
            records.write(
                18u,
                make_float4(
                    collision.phase.direction,
                    collision.phase.pdf));
            records.write(
                19u,
                make_float4(
                    collision.phase
                        .sampled_roughness,
                    collision.phase
                        .selection_rescaled,
                    cast<float>(
                        collision.phase
                            .closure_index),
                    cast<float>(
                        collision.phase
                            .closure_type)));
            records.write(
                20u,
                make_float4(
                    collision.transport
                        .scatter_random,
                    cast<float>(
                        collision.transport.channel),
                    select(
                        0.0f,
                        1.0f,
                        collision.transport.active),
                    select(
                        0.0f,
                        1.0f,
                        collision.phase_failed)));

            const auto empty_segment =
                segment->emit(
                    empty,
                    services,
                    state,
                    0.5f,
                    make_float3(1.0f),
                    0.2f,
                    0.2f,
                    make_float2(
                        0.034f, 0.83f),
                    false,
                    {.scattered_radiance =
                         make_float3(0.0f),
                     .transmitted_radiance =
                         make_float3(0.0f),
                     .majorant_optical_depth =
                         0.0f,
                     .enabled = false},
                    false,
                    make_float3(0.0f));
            records.write(
                21u,
                make_float4(
                    empty_segment.transport
                        .throughput,
                    select(
                        0.0f,
                        1.0f,
                        empty_segment.scattered)));
        };

    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, record_count>
        actual{};
    stream
        << parameters.copy_from(
               luisa::span{host_parameters})
        << kernel(parameters, output)
               .dispatch(1u)
        << output.copy_to(
               luisa::span{actual})
        << synchronize();

    // Pinned to official Cycles main b82c3f0 volume_shader.h. Entry zero
    // retains the two raw equal HG closures. The second stack entry triggers
    // volume_shader_merge_closures(), so all four closures become one after
    // their independently parameterized coefficients and object-density
    // scales have been evaluated.
    constexpr std::array expected{
        luisa::float4{
            1.35f, 1.95f, 2.7f, 1.0f},
        luisa::float4{
            0.35f, 0.45f, 0.7f, 1.0f},
        luisa::float4{
            0.1f, 0.2f, 0.3f, 1.0f},
        luisa::float4{
            2.0f, 1.0f, 0.0f, 0.0f},
        luisa::float4{
            0.2f, 0.4f, 0.6f, 0.4f},
        luisa::float4{
            0.15f, 0.05f, 0.1f, 0.1f},
        luisa::float4{
            0.42f, 0.42f, 1.0f, 1.0f},
        luisa::float4{
            4.45f, 6.65f, 9.3f, 1.0f},
        luisa::float4{
            1.45f, 2.15f, 3.3f, 1.0f},
        luisa::float4{
            0.3f, 0.6f, 0.9f, 1.0f},
        luisa::float4{
            1.0f, 2.0f, 0.0f, 0.42f},
        luisa::float4{
            1.45f, 2.15f, 3.3f, 2.3f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f},
        // Full raw-graph -> homogeneous estimator -> phase continuation.
        // Values are pinned to official Cycles main b82c3f0. The phase input
        // deliberately reservoir-selects closure one and rescales 0.034 to
        // 0.17, reproducing the HG oracle's first direction sample.
        luisa::float4{
            0.204449043f,
            0.238044649f,
            0.327118456f,
            1.0f},
        luisa::float4{
            0.0363587849f,
            0.0638777092f,
            0.0823066384f,
            0.165291518f},
        luisa::float4{
            0.509156406f,
            0.377192348f,
            0.25924027f,
            1.36953449f},
        luisa::float4{
            -0.117953472f,
            -0.899912298f,
            -0.419815481f,
            0.044300843f},
        luisa::float4{
            0.58f, 0.17f, 1.0f, 0.0f},
        luisa::float4{
            0.407461792f,
            0.0f,
            1.0f,
            0.0f},
        luisa::float4{
            1.0f, 1.0f, 1.0f, 0.0f}};
    for (std::size_t index = 0u;
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "Cycles stacked-volume fixture failed on "
                << backend << " at record "
                << index << ": got {"
                << actual[index].x << ", "
                << actual[index].y << ", "
                << actual[index].z << ", "
                << actual[index].w
                << "}, expected {"
                << expected[index].x << ", "
                << expected[index].y << ", "
                << expected[index].z << ", "
                << expected[index].w
                << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
