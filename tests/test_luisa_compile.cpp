#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>

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
        Expr<luisa::float2>) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] Float4 attribute(
        Expr<std::uint64_t>,
        const SurfacePoint &) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] Float parameter_float(
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return 0.0f;
    }

    [[nodiscard]] Float3 parameter_float3(
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float3(0.8f, 0.4f, 0.2f);
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        return 1.0f;
    }
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
        surface_program.program, 0u);

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
            .object_random = 0.0f,
            .particle_index = 0u,
            .random_per_island = 0.0f,
            .ray_visibility = 1u,
            .ray_events = 0u,
            .ray_depth = 0u,
            .diffuse_depth = 0u,
            .glossy_depth = 0u,
            .transparent_depth = 0u,
            .transmission_depth = 0u,
            .ray_length = 0.0f,
            .time = 0.0f,
            .back_facing = false};
        auto query = SurfaceQuery{
            .lobe_mask = ~std::uint32_t{0u},
            .transport_mode =
                static_cast<std::uint32_t>(
                    TransportMode::radiance)};
        auto evaluation = surfaces.evaluate(
            UInt{surface_tag},
            services,
            point,
            make_float3(0.0f, 0.0f, 1.0f),
            query);
        device_assert(evaluation.pdf >= 0.0f);
    };
    return kernel.function() ? 0 : 3;
}
