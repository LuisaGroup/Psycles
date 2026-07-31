#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/volume_phase_set.h>

#include <algorithm>
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
namespace phase =
    psycles::luisa_backend::cycles_volume_phase;

inline constexpr std::size_t record_count = 37u;

class VolumeShaderServices final : public ShaderServices {

private:
    const BufferFloat4 &_parameters;
    bool _density_found{};
    float _density{};
    bool _temperature_found{};
    float _temperature{};

public:
    VolumeShaderServices(
        const BufferFloat4 &parameters,
        bool density_found,
        float density,
        bool temperature_found,
        float temperature) noexcept
        : _parameters{parameters},
          _density_found{density_found},
          _density{density},
          _temperature_found{temperature_found},
          _temperature{temperature} {}

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
        Expr<std::uint64_t> id,
        const SurfacePoint &) const noexcept override {
        auto result = ShaderAttribute::missing();
        if (_density_found) {
            const auto match =
                id == contract::attribute_id("density");
            result.value = select(
                result.value,
                make_float4(_density),
                match);
            result.found = result.found | match;
        }
        if (_temperature_found) {
            const auto match =
                id == contract::attribute_id("temperature");
            result.value = select(
                result.value,
                make_float4(_temperature),
                match);
            result.found = result.found | match;
        }
        return result;
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).x;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).xyz();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        return 1.0f;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] ShaderGraph make_volume_graph() {
    ShaderGraph graph;
    const auto surface =
        graph.add_node(node_type::diffuse_bsdf, "Surface");
    require(
        graph.set_input(
            surface,
            "Color",
            SocketValue::color({0.0f, 0.0f, 0.0f})),
        "failed to configure surface");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = surface, .socket = "Closure"});

    const auto absorption =
        graph.add_node(node_type::volume_absorption, "Absorption");
    require(
        graph.set_input(
            absorption,
            "Color",
            SocketValue::color({0.2f, 0.4f, 0.8f})) &&
            graph.set_input(
                absorption,
                "Density",
                SocketValue::floating(2.0f)),
        "failed to configure absorption volume");

    const auto scatter =
        graph.add_node(node_type::volume_scatter, "Scatter");
    require(
        graph.set_input(
            scatter,
            "Color",
            SocketValue::color({0.6f, -0.2f, 0.3f})) &&
            graph.set_input(
                scatter,
                "Density",
                SocketValue::floating(1.5f)) &&
            graph.set_property(
                scatter,
                "Phase",
                SocketValue::string("DRAINE")),
        "failed to configure scatter volume");

    const auto mixed =
        graph.add_node(node_type::mix_volume, "Mixed");
    require(
        graph.set_input(
            mixed,
            "Factor",
            SocketValue::floating(0.25f)) &&
            graph.connect(
                {.node = absorption, .socket = "Volume"},
                mixed,
                "A") &&
            graph.connect(
                {.node = scatter, .socket = "Volume"},
                mixed,
                "B"),
        "failed to configure mixed volume");

    const auto coefficients =
        graph.add_node(
            node_type::volume_coefficients,
            "Coefficients");
    require(
        graph.set_input(
            coefficients,
            "ScatterCoefficients",
            SocketValue::vector({0.1f, 0.2f, -0.3f})) &&
            graph.set_input(
                coefficients,
                "AbsorptionCoefficients",
                SocketValue::vector({0.05f, 0.1f, 0.2f})) &&
            graph.set_input(
                coefficients,
                "EmissionCoefficients",
                SocketValue::vector({1.0f, 2.0f, 3.0f})) &&
            graph.set_property(
                coefficients,
                "Phase",
                SocketValue::string("MIE")),
        "failed to configure coefficient volume");

    const auto principled =
        graph.add_node(node_type::principled_volume, "Principled");
    require(
        graph.set_input(
            principled,
            "Color",
            SocketValue::color({0.4f, 0.5f, 0.8f})) &&
            graph.set_input(
                principled,
                "Density",
                SocketValue::floating(0.5f)) &&
            graph.set_input(
                principled,
                "AbsorptionColor",
                SocketValue::color({0.25f, 0.36f, 0.81f})) &&
            graph.set_input(
                principled,
                "EmissionStrength",
                SocketValue::floating(2.0f)) &&
            graph.set_input(
                principled,
                "EmissionColor",
                SocketValue::color({0.2f, 0.3f, 0.4f})) &&
            graph.set_input(
                principled,
                "BlackbodyIntensity",
                SocketValue::floating(0.0f)),
        "failed to configure principled volume");

    const auto add_left =
        graph.add_node(node_type::add_volume, "AddLeft");
    const auto add_root =
        graph.add_node(node_type::add_volume, "AddRoot");
    require(
        graph.connect(
            {.node = mixed, .socket = "Volume"},
            add_left,
            "A") &&
            graph.connect(
                {.node = coefficients, .socket = "Volume"},
                add_left,
                "B") &&
            graph.connect(
                {.node = add_left, .socket = "Volume"},
                add_root,
                "A") &&
            graph.connect(
                {.node = principled, .socket = "Volume"},
                add_root,
                "B"),
        "failed to configure additive volume tree");
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{.node = add_root, .socket = "Volume"});
    return graph;
}

[[nodiscard]] std::vector<luisa::float4> parameter_data(
    const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    result.reserve(program.parameters().size());
    for (const auto &parameter : program.parameters()) {
        const auto &value = parameter.default_value;
        if (const auto *scalar =
                std::get_if<float>(&value.value)) {
            result.emplace_back(
                *scalar, 0.0f, 0.0f, 0.0f);
        } else if (
            const auto *vector =
                std::get_if<Vec3f>(&value.value)) {
            result.emplace_back(
                vector->x,
                vector->y,
                vector->z,
                0.0f);
        } else {
            throw std::runtime_error{
                "volume fixture contains an unsupported parameter type"};
        }
    }
    return result;
}

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 4.0e-6f) noexcept {
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
    float tolerance = 4.0e-6f) noexcept {
    return approximately_equal(
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
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(make_volume_graph());
    if (!shader.ok()) {
        std::cerr << "failed to compile volume fixture graph\n";
        return EXIT_FAILURE;
    }
    const auto program =
        compile_surface_program(*shader.program);
    if (!program.ok()) {
        std::cerr << "failed to lower volume fixture program\n";
        return EXIT_FAILURE;
    }

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(program.program);
    if (!surfaces.capabilities(surface_tag).may_have_volume) {
        std::cerr << "volume capability was not retained\n";
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    const auto host_parameters =
        parameter_data(*program.program);
    auto parameters = device.create_buffer<luisa::float4>(
        host_parameters.size());
    auto output =
        device.create_buffer<luisa::float4>(
            record_count);

    Kernel1D evaluate =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 records) noexcept {
            SurfacePoint point{};
            point.parameter_block = 0u;
            point.geometry_index = ~0u;
            const auto query = VolumeQuery{
                .object_density = 2.0f,
                .evaluate_emission = true};
            VolumeShaderServices attributes{
                parameter_buffer,
                true,
                0.5f,
                true,
                2.0f};
            const auto full =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    attributes,
                    point,
                    query);
            records.write(
                0u,
                make_float4(
                    full.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        full.has_extinction)));
            records.write(
                1u,
                make_float4(
                    full.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        full.has_scatter)));
            records.write(
                2u,
                make_float4(
                    full.emission,
                    select(
                        0.0f,
                        1.0f,
                        full.has_emission)));

            const auto extinction_only =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    attributes,
                    point,
                    VolumeQuery{
                        .object_density = 2.0f,
                        .evaluate_emission = false});
            records.write(
                3u,
                make_float4(
                    extinction_only.emission,
                    select(
                        0.0f,
                        1.0f,
                        extinction_only.has_emission)));

            VolumeShaderServices zero_density{
                parameter_buffer,
                true,
                0.0f,
                false,
                0.0f};
            const auto zero_attribute =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    zero_density,
                    point,
                    query);
            records.write(
                4u,
                make_float4(
                    zero_attribute.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        zero_attribute.has_extinction)));
            records.write(
                5u,
                make_float4(
                    zero_attribute.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        zero_attribute.has_scatter)));

            VolumeShaderServices missing_density{
                parameter_buffer,
                false,
                0.0f,
                false,
                0.0f};
            const auto missing_attribute =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    missing_density,
                    point,
                    query);
            records.write(
                6u,
                make_float4(
                    missing_attribute.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        missing_attribute.has_extinction)));
            records.write(
                7u,
                make_float4(
                    missing_attribute.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        missing_attribute.has_scatter)));
            records.write(
                8u,
                make_float4(
                    zero_attribute.emission,
                    select(
                        0.0f,
                        1.0f,
                        zero_attribute.has_emission)));

            VolumePhaseSet phases{8u};
            const auto combined =
                surfaces.evaluate_volume(
                UInt{surface_tag},
                attributes,
                point,
                query,
                &phases);
            records.write(
                34u,
                make_float4(
                    combined.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        combined.has_extinction)));
            records.write(
                35u,
                make_float4(
                    combined.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        combined.has_scatter)));
            records.write(
                36u,
                make_float4(
                    combined.emission,
                    select(
                        0.0f,
                        1.0f,
                        combined.has_emission)));
            phases.merge_equal();
            phases.truncate();
            const auto phase_axis =
                normalize(
                    make_float3(
                        0.2f,
                        -0.3f,
                        0.9327379f));
            const auto phase_evaluation =
                phases.evaluate(
                    make_float3(0.0f, 0.0f, 1.0f),
                    make_float3(
                        0.7599342f,
                        0.0f,
                        0.65f));
            records.write(
                9u,
                make_float4(
                    cast<float>(phases.count()),
                    phase_evaluation.value,
                    phase_evaluation.pdf,
                    phase_evaluation.sample_weight));
            for (auto index = 0u;
                 index < 4u;
                 ++index) {
                const auto entry =
                    phases.entry(UInt{index});
                records.write(
                    10u + index,
                    make_float4(
                        entry.weight,
                        entry.sample_weight));
                records.write(
                    14u + index,
                    make_float4(
                        entry.parameters,
                        cast<float>(entry.type)));
            }
            const auto invalid_entry =
                phases.entry(4u);
            records.write(
                18u,
                make_float4(
                    select(
                        0.0f,
                        1.0f,
                        phases.entry(0u).valid),
                    select(
                        0.0f,
                        1.0f,
                        phases.entry(1u).valid),
                    select(
                        0.0f,
                        1.0f,
                        phases.entry(3u).valid),
                    select(
                        0.0f,
                        1.0f,
                        invalid_entry.valid)));

            const std::array sample_randoms{
                0.17f, 0.37f, 0.9f};
            for (auto index = 0u;
                 index < sample_randoms.size();
                 ++index) {
                const auto sample =
                    phases.sample(
                        phase_axis,
                        make_float2(
                            sample_randoms[index],
                            0.72f));
                records.write(
                    19u + index * 2u,
                    make_float4(
                        sample.direction,
                        sample.pdf));
                records.write(
                    20u + index * 2u,
                    make_float4(
                        sample.sampled_roughness,
                        sample.selection_rescaled,
                        cast<float>(
                            sample.closure_index),
                        cast<float>(
                            sample.closure_type)));
            }

            VolumePhaseSet merged{6u};
            merged.add(
                phase::henyey_greenstein(0.2f),
                make_float3(1.0f, 2.0f, 3.0f));
            merged.add(
                phase::draine(-0.1f, 2.0f),
                make_float3(0.3f, 0.6f, 0.9f));
            merged.add(
                phase::henyey_greenstein(0.2f),
                make_float3(-1.0f, 0.5f, 1.0f));
            merged.add(
                phase::rayleigh(),
                make_float3(0.9f, 0.3f, 0.0f));
            const auto pre_merge_count =
                merged.count();
            merged.merge_equal();
            records.write(
                25u,
                make_float4(
                    cast<float>(pre_merge_count),
                    cast<float>(merged.count()),
                    0.0f,
                    0.0f));
            for (auto index = 0u;
                 index < 3u;
                 ++index) {
                const auto entry =
                    merged.entry(UInt{index});
                records.write(
                    26u + index,
                    make_float4(
                        entry.weight,
                        entry.sample_weight));
            }
            records.write(
                29u,
                make_float4(
                    cast<float>(
                        merged.entry(0u).type),
                    cast<float>(
                        merged.entry(1u).type),
                    cast<float>(
                        merged.entry(2u).type),
                    merged.entry(0u)
                        .parameters.x));

            VolumePhaseSet truncated{10u};
            for (auto index = 0u;
                 index < 10u;
                 ++index) {
                truncated.add(
                    phase::henyey_greenstein(
                        static_cast<float>(index) *
                        0.05f),
                    make_float3(1.0f));
            }
            const auto pre_truncate_count =
                truncated.count();
            truncated.truncate();
            records.write(
                30u,
                make_float4(
                    cast<float>(
                        pre_truncate_count),
                    cast<float>(truncated.count()),
                    truncated.entry(7u)
                        .parameters.x,
                    select(
                        0.0f,
                        1.0f,
                        truncated.entry(8u).valid)));

            VolumePhaseSet capacity_limited{2u};
            capacity_limited.add(
                phase::henyey_greenstein(0.1f),
                make_float3(1.0f));
            capacity_limited.add(
                phase::draine(0.2f, 0.5f),
                make_float3(1.0f));
            capacity_limited.add(
                phase::rayleigh(),
                make_float3(1.0f));
            records.write(
                31u,
                make_float4(
                    cast<float>(
                        capacity_limited.count()),
                    cast<float>(
                        capacity_limited.entry(0u)
                            .type),
                    cast<float>(
                        capacity_limited.entry(1u)
                            .type),
                    select(
                        0.0f,
                        1.0f,
                        capacity_limited.entry(2u)
                            .valid)));

            VolumePhaseSet empty{1u};
            const auto empty_evaluation =
                empty.evaluate(
                    make_float3(0.0f, 0.0f, 1.0f),
                    make_float3(0.0f, 0.0f, 1.0f));
            records.write(
                32u,
                make_float4(
                    empty_evaluation.value,
                    empty_evaluation.pdf,
                    empty_evaluation.sample_weight,
                    select(
                        0.0f,
                        1.0f,
                        empty_evaluation.valid)));
            const auto empty_sample =
                empty.sample(
                    make_float3(0.0f, 0.0f, 1.0f),
                    make_float2(0.3f, 0.7f));
            records.write(
                33u,
                make_float4(
                    empty_sample.pdf,
                    select(
                        0.0f,
                        1.0f,
                        empty_sample.valid),
                    cast<float>(
                        empty_sample.closure_index),
                    cast<float>(
                        empty_sample.closure_type)));
        };

    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, record_count>
        actual{};
    stream << parameters.copy_from(
                  luisa::span{host_parameters})
           << kernel(parameters, output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    // Pinned to Cycles main 2bad74a8. These records exercise SVM's raw
    // absorption/scatter extinction, volume-only negative-weight clamping,
    // Add/Mix weighting, Principled sqrt(absorption color), implicit density
    // attribute semantics, object density scaling, and emission suppression.
    constexpr std::array expected{
        luisa::float4{3.5f, 2.6f, 1.035f, 1.0f},
        luisa::float4{0.85f, 0.65f, 0.625f, 1.0f},
        luisa::float4{2.8f, 5.2f, 7.6f, 1.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{3.15f, 2.25f, 0.625f, 1.0f},
        luisa::float4{0.65f, 0.4f, 0.225f, 1.0f},
        luisa::float4{3.85f, 2.95f, 1.445f, 1.0f},
        luisa::float4{1.05f, 0.9f, 1.025f, 1.0f},
        luisa::float4{2.8f, 5.2f, 7.6f, 1.0f}};
    for (std::size_t index = 0u;
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "Cycles volume coefficient fixture failed on "
                << backend << " at record " << index
                << ": got {" << actual[index].x
                << ", " << actual[index].y
                << ", " << actual[index].z
                << ", " << actual[index].w
                << "}\n";
            return EXIT_FAILURE;
        }
    }

    // Generated from official Cycles main b82c3f0. The graph must preserve
    // four raw closures in source order: Draine, fitted-Mie HG, fitted-Mie
    // Draine, and Principled HG. No closure is pre-baked by Blender/Cycles.
    constexpr std::array phase_expected{
        luisa::float4{
            4.0f,
            0.0707130209f,
            0.0707130209f,
            0.708333373f},
        luisa::float4{
            0.45f, 0.0f, 0.225f, 0.225f},
        luisa::float4{
            0.100368105f,
            0.20073621f,
            0.0f,
            0.100368105f},
        luisa::float4{
            0.0996318981f,
            0.199263796f,
            0.0f,
            0.0996318981f},
        luisa::float4{
            0.2f,
            0.25f,
            0.4f,
            0.283333331f},
        luisa::float4{
            0.0f, 0.5f, 0.0f, 2.0f},
        luisa::float4{
            0.994610012f,
            0.0f,
            0.0f,
            0.0f},
        luisa::float4{
            0.59379518f,
            27.1138496f,
            0.0f,
            2.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            1.0f, 1.0f, 1.0f, 0.0f},
        luisa::float4{
            0.049110055f,
            -0.44522649f,
            0.894070327f,
            0.0892113373f},
        luisa::float4{
            0.00538998842f,
            0.0227289386f,
            1.0f,
            0.0f},
        luisa::float4{
            -0.484209776f,
            -0.0472326875f,
            -0.87367624f,
            0.0795774683f},
        luisa::float4{
            1.0f,
            0.948791504f,
            3.0f,
            0.0f},
        luisa::float4{
            -0.543359995f,
            -0.777569592f,
            0.316457808f,
            0.0742187649f},
        luisa::float4{
            1.0f,
            0.685185194f,
            0.0f,
            2.0f}};
    for (auto index = std::size_t{0u};
         index < phase_expected.size();
         ++index) {
        const auto record = 9u + index;
        if (!approximately_equal(
                actual[record],
                phase_expected[index],
                2.0e-5f)) {
            std::cerr
                << "Cycles raw volume phase fixture failed on "
                << backend << " at record " << record
                << ": got {" << actual[record].x
                << ", " << actual[record].y
                << ", " << actual[record].z
                << ", " << actual[record].w
                << "}\n";
            return EXIT_FAILURE;
        }
    }

    constexpr std::array set_expected{
        luisa::float4{
            4.0f, 3.0f, 0.0f, 0.0f},
        luisa::float4{
            1.0f, 2.5f, 4.0f, 2.5f},
        luisa::float4{
            0.3f, 0.6f, 0.9f, 0.6f},
        luisa::float4{
            0.9f, 0.3f, 0.0f, 0.4f},
        luisa::float4{
            0.0f, 2.0f, 3.0f, 0.2f},
        luisa::float4{
            10.0f, 8.0f, 0.35f, 0.0f},
        luisa::float4{
            2.0f, 0.0f, 2.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f}};
    for (auto index = std::size_t{0u};
         index < set_expected.size();
         ++index) {
        const auto record = 25u + index;
        if (!approximately_equal(
                actual[record],
                set_expected[index])) {
            std::cerr
                << "volume phase set invariant failed on "
                << backend << " at record " << record
                << ": got {" << actual[record].x
                << ", " << actual[record].y
                << ", " << actual[record].z
                << ", " << actual[record].w
                << "}\n";
            return EXIT_FAILURE;
        }
    }
    for (auto index = std::size_t{0u};
         index < 3u;
         ++index) {
        if (!approximately_equal(
                actual[34u + index],
                expected[index])) {
            std::cerr
                << "combined volume evaluation failed on "
                << backend << " at record "
                << 34u + index << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
