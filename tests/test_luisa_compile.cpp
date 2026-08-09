#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/graph_surface.h>

#include "../src/luisa/path_tracer_shader_services.h"

#include <luisa/xir/instructions/if.h>
#include <luisa/xir/translators/ast2xir.h>

#include <cstdint>
#include <stdexcept>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using namespace luisa::compute;

class ConstantShaderServices final : public ShaderServices {

public:
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
        Expr<luisa::ulong>,
        const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t>,
        Expr<std::uint32_t>) const noexcept override {
        return 0.0f;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t>,
        Expr<std::uint32_t>) const noexcept override {
        return make_float3(0.8f, 0.4f, 0.2f);
    }

    [[nodiscard]] ULong
    parameter_uint64(
        Expr<std::uint32_t>,
        Expr<std::uint32_t>) const noexcept override {
        return 0u;
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        return 1.0f;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> xyz_expression)
        const noexcept override {
        Float3 xyz{xyz_expression};
        return make_float3(
            3.2404542f * xyz.x - 1.5371385f * xyz.y -
                0.4985314f * xyz.z,
            -0.9692660f * xyz.x + 1.8760108f * xyz.y +
                0.0415560f * xyz.z,
            0.0556434f * xyz.x - 0.2040259f * xyz.y +
                1.0572252f * xyz.z);
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> rec709) const noexcept override {
        return Float3{rec709};
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

class CompileProbeSurface final : public Surface {

private:
    bool _transparent{};
    std::size_t *_transparent_recordings{};
    std::size_t *_subsurface_recordings{};

public:
    CompileProbeSurface(
        bool transparent,
        std::size_t *transparent_recordings,
        std::size_t *subsurface_recordings) noexcept
        : _transparent{transparent},
          _transparent_recordings{transparent_recordings},
          _subsurface_recordings{subsurface_recordings} {}

    [[nodiscard]] SurfaceCapabilities capabilities()
        const noexcept override {
        return {.may_be_transparent = _transparent};
    }

    [[nodiscard]] SurfaceClosureCollection collect_closures(
        const ShaderServices &,
        const SurfacePoint &point,
        Expr<bool>,
        Expr<bool>,
        SurfaceClosureCollector &collector) const noexcept override {
        if (_subsurface_recordings != nullptr) {
            ++*_subsurface_recordings;
        }
        collector.begin(point.shading_normal);
        collector.finish();
        return {.shading_normal = point.shading_normal};
    }

    [[nodiscard]] SurfaceEvaluation evaluate(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<luisa::float3>,
        const SurfaceQuery &) const noexcept override {
        return SurfaceEvaluation::zero();
    }

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<luisa::float3>,
        const SurfaceLightQuery &) const noexcept override {
        return SurfaceEvaluation::zero();
    }

    [[nodiscard]] SurfaceSample sample(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<float>,
        Expr<luisa::float2>,
        const SurfaceQuery &) const noexcept override {
        return SurfaceSample::zero();
    }

    [[nodiscard]] Float3 transparent_extinction(
        const ShaderServices &,
        const SurfacePoint &) const noexcept override {
        if (_transparent_recordings != nullptr) {
            ++*_transparent_recordings;
        }
        return make_float3(_transparent ? 0.5f : 0.0f);
    }
};

class CompileProbeCollector final : public SurfaceClosureCollector {

public:
    void add(const SurfaceClosureRecord &) noexcept override {}
};

[[nodiscard]] ShaderGraph make_graph() {
    ShaderGraph graph;
    const auto geometry =
        graph.add_node(node_type::geometry, "Geometry");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    if (!graph.connect(
            {.node = geometry, .socket = "Normal"},
            diffuse,
            "Normal")) {
        throw std::runtime_error{"failed to connect test normal"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] bool attribute_lookup_cfg_is_bounded() {
    constexpr auto logical_binding_count = 512u;
    std::vector<
        psycles::luisa_backend::detail::
            NishitaTextureBinding>
        nishita_textures;
    contract::ShaderColorSpace shader_color_space;
    Kernel1D kernel =
        [nishita_textures =
             std::move(nishita_textures),
         shader_color_space,
         logical_binding_count](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap) noexcept {
            psycles::luisa_backend::detail::
                BufferShaderServices services{
                    scalar_parameters,
                    vector_parameters,
                    cycles_bsdf_tables,
                    textures,
                    geometry_heap,
                    0u,
                    1u,
                    nishita_textures,
                    shader_color_space};
            SurfacePoint point{};
            point.geometry_index = dispatch_x();
            point.primitive_id = 0u;
            point.barycentric = make_float2(0.25f);
            auto value = services.attribute(
                def<luisa::ulong>(
                    logical_binding_count),
                point);
            device_assert(all(value.value >= 0.0f));
        };
    auto module = luisa::compute::xir::
        ast_to_xir_translate(
            kernel.function()->function(), {});
    std::size_t structured_if_count = 0u;
    for (auto *function : module->function_list()) {
        if (auto *definition = function->definition()) {
            definition->traverse_instructions(
                [&](const luisa::compute::xir::
                        Instruction *instruction) noexcept {
                    structured_if_count +=
                        instruction->isa<
                            luisa::compute::xir::IfInst>()
                            ? 1u :
                            0u;
                });
        }
    }
    // Scene metadata belongs in device data. Its cardinality must not become
    // one host-recorded branch per attribute in every material callable.
    return structured_if_count <= 8u;
}

}// namespace

int main() {
    ShaderCompiler shader_compiler{make_core_node_registry()};
    auto shader = shader_compiler.compile(make_graph());
    if (!shader.ok()) {
        return 1;
    }
    auto surface_program =
        compile_surface_program(*shader.program);
    if (!surface_program.ok()) {
        return 2;
    }

    SurfaceDispatch surfaces;
    const auto surface_tag = surfaces.create<GraphSurface>(
        surface_program.program);
    std::size_t opaque_transparency_recordings = 0u;
    std::size_t transparent_transparency_recordings = 0u;
    SurfaceDispatch transparency_surfaces;
    static_cast<void>(
        transparency_surfaces.create<CompileProbeSurface>(
            false, &opaque_transparency_recordings, nullptr));
    static_cast<void>(
        transparency_surfaces.create<CompileProbeSurface>(
            true, &transparent_transparency_recordings, nullptr));
    std::size_t opaque_subsurface_recordings = 0u;
    std::size_t subsurface_recordings = 0u;
    SurfaceDispatch subsurface_surfaces;
    static_cast<void>(
        subsurface_surfaces.create<CompileProbeSurface>(
            false, nullptr, &opaque_subsurface_recordings));
    static_cast<void>(
        subsurface_surfaces.create<CompileProbeSurface>(
            false, nullptr, &subsurface_recordings));
    const luisa::vector<luisa::uint> surface_bssrdf_bump_tags{1u};

    Kernel1D kernel = [&]() noexcept {
        ConstantShaderServices services;
        auto point = SurfacePoint{
            .position = make_float3(0.0f),
            .object_position = make_float3(0.0f),
            .object_location = make_float3(0.0f),
            .generated = make_float3(0.5f),
            .geometric_normal = make_float3(0.0f, 0.0f, 1.0f),
            .shading_normal = make_float3(0.0f, 0.0f, 1.0f),
            .object_shading_normal =
                make_float3(0.0f, 0.0f, 1.0f),
            .object_tangent =
                make_float3(1.0f, 0.0f, 0.0f),
            .tangent_sign = 1.0f,
            .normal_to_world_x =
                make_float3(1.0f, 0.0f, 0.0f),
            .normal_to_world_y =
                make_float3(0.0f, 1.0f, 0.0f),
            .normal_to_world_z =
                make_float3(0.0f, 0.0f, 1.0f),
            .dpdu = make_float3(1.0f, 0.0f, 0.0f),
            .dpdv = make_float3(0.0f, 1.0f, 0.0f),
            .dPdx = make_float3(0.0f),
            .dPdy = make_float3(0.0f),
            .object_dPdx = make_float3(0.0f),
            .object_dPdy = make_float3(0.0f),
            .generated_dx = make_float3(0.0f),
            .generated_dy = make_float3(0.0f),
            .incoming = make_float3(0.0f, 0.0f, 1.0f),
            .uv = make_float2(0.0f),
            .uv_dx = make_float2(0.0f),
            .uv_dy = make_float2(0.0f),
            .geometry_index = 0u,
            .barycentric = make_float2(0.0f),
            .barycentric_dx = make_float2(0.0f),
            .barycentric_dy = make_float2(0.0f),
            .instance_id = 0u,
            .primitive_id = 0u,
            .parameter_block = 0u,
            .object_random = 0.0f,
            .particle_index = 0u,
            .random_per_island = 0.0f,
            .is_curve = false,
            .curve_intercept = 0.0f,
            .curve_length = 0.0f,
            .curve_thickness = 0.0f,
            .curve_tangent_normal = make_float3(0.0f),
            .curve_random = 0.0f,
            .ray_visibility = 1u,
            .ray_events = 0u,
            .ray_depth = 0u,
            .diffuse_depth = 0u,
            .glossy_depth = 0u,
            .transparent_depth = 0u,
            .transmission_depth = 0u,
            .ray_length = 0.0f,
            .time = 0.0f,
            .use_bump_map_correction = true,
            .back_facing = false};
        auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode =
                static_cast<std::uint32_t>(
                    TransportMode::radiance),
            .glossy_filter_roughness = 0.0f};
        auto evaluation = surfaces.evaluate(
            UInt{surface_tag},
            services,
            point,
            make_float3(0.0f, 0.0f, 1.0f),
            query);
        auto transparent =
            transparency_surfaces.transparent_extinction(
                dispatch_x() % 2u,
                services,
                point);
        CompileProbeCollector subsurface_collector;
        const auto subsurface =
            subsurface_surfaces.collect_bssrdf_bump_closures(
                dispatch_x() % 2u,
                surface_bssrdf_bump_tags,
                services,
                point,
                true,
                true,
                subsurface_collector);
        device_assert(evaluation.pdf >= 0.0f);
        device_assert(all(transparent >= 0.0f));
        device_assert(all(subsurface.shading_normal == point.shading_normal));
    };
    if (!kernel.function() ||
        opaque_transparency_recordings != 0u ||
        transparent_transparency_recordings != 1u ||
        opaque_subsurface_recordings != 0u ||
        subsurface_recordings != 1u) {
        return 3;
    }
    if (!attribute_lookup_cfg_is_bounded()) {
        return 4;
    }

    Kernel1D sampler_kernel = [](
                                  BufferFloat4 table,
                                  UInt sequence_size,
                                  BufferUInt indices,
                                  BufferFloat4 samples) noexcept {
        const auto rng_hash =
            cycles_sampler::pixel_hash(17u, 29u, 0u);
        const auto dimension = cycles_sampler::path_dimension(
            0u,
            sampling::tabulated_sobol::light_dimension);
        indices.write(
            0u,
            cycles_sampler::shuffled_sample_index(
                63u,
                dimension,
                rng_hash,
                sequence_size));
        samples.write(
            0u,
            cycles_sampler::sample_4d(
                table,
                sequence_size,
                63u,
                rng_hash,
                dimension));
    };
    return sampler_kernel.function() ? 0 : 5;
}
