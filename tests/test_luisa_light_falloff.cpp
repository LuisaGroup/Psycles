#include "surface_light_falloff.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_trace_schema.h>
#include <psycles/luisa/path_tracer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/passes/dom_tree.h>
#include <luisa/xir/translators/ast2xir.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend::detail;

inline constexpr std::array<std::uint16_t, 3u> falloff_domain{
    static_cast<std::uint16_t>(LightFalloffType::quadratic),
    static_cast<std::uint16_t>(LightFalloffType::linear),
    static_cast<std::uint16_t>(LightFalloffType::constant)};

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 2.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               (1.0f + std::max(std::abs(actual), std::abs(expected)));
}

[[nodiscard]] Kernel1D<Buffer<float>> make_svm_kernel() {
    return [](BufferFloat output) noexcept {
        const UInt type = dispatch_x();
        output.write(
            type * 3u,
            evaluate_surface_light_falloff_svm(
                type, falloff_domain, 2.0f, 0.0f, 4.0f));
        output.write(
            type * 3u + 1u,
            evaluate_surface_light_falloff_svm(
                type, falloff_domain, 3.0f, 4.0f, 2.0f));
        output.write(
            type * 3u + 2u,
            evaluate_surface_light_falloff_svm(
                type,
                falloff_domain,
                2.0f,
                4.0f,
                std::numeric_limits<float>::max()));
    };
}

[[nodiscard]] Kernel1D<Buffer<float>> make_static_kernel() {
    return [](BufferFloat output) noexcept {
        const UInt type = dispatch_x();
        Float finite = 0.0f;
        Float smoothed = 0.0f;
        Float distant = 0.0f;
        luisa::compute::detail::SwitchStmtBuilder{type} % [&] {
            for (auto index = std::size_t{0u}; index < falloff_domain.size();
                 ++index) {
                luisa::compute::detail::SwitchCaseStmtBuilder{
                    static_cast<luisa::uint>(index)} %
                    [&, index] {
                        const auto output_type =
                            static_cast<LightFalloffType>(index);
                        finite = evaluate_surface_light_falloff(
                            output_type, 2.0f, 0.0f, 4.0f);
                        smoothed = evaluate_surface_light_falloff(
                            output_type, 3.0f, 4.0f, 2.0f);
                        distant = evaluate_surface_light_falloff(
                            output_type,
                            2.0f,
                            4.0f,
                            std::numeric_limits<float>::max());
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid static Light Falloff output");
            };
        };
        output.write(type * 3u, finite);
        output.write(type * 3u + 1u, smoothed);
        output.write(type * 3u + 2u, distant);
    };
}

[[nodiscard]] bool short_circuit_is_structural() {
    Kernel1D shape = [](BufferFloat input, BufferFloat output) noexcept {
        output.write(
            0u,
            evaluate_surface_light_falloff(
                LightFalloffType::constant,
                input.read(0u),
                input.read(1u),
                input.read(2u)));
    };
    auto module = luisa::compute::xir::ast_to_xir_translate(
        shape.function()->function(), {});
    auto arithmetic_count = std::size_t{0u};
    auto guarded_arithmetic_count = std::size_t{0u};
    for (auto *function : module->function_list()) {
        auto *definition = function->definition();
        if (definition == nullptr) {
            continue;
        }
        std::vector<luisa::compute::xir::IfInst *> branches;
        std::vector<luisa::compute::xir::ArithmeticInst *> arithmetic;
        definition->traverse_instructions(
            [&](luisa::compute::xir::Instruction *instruction) noexcept {
                if (instruction->isa<luisa::compute::xir::IfInst>()) {
                    branches.emplace_back(
                        static_cast<luisa::compute::xir::IfInst *>(instruction));
                    return;
                }
                if (!instruction->isa<
                        luisa::compute::xir::ArithmeticInst>()) {
                    return;
                }
                auto *candidate = static_cast<
                    luisa::compute::xir::ArithmeticInst *>(instruction);
                if (candidate->type() == Type::of<float>() &&
                    (candidate->op() ==
                         luisa::compute::xir::ArithmeticOp::BINARY_MUL ||
                     candidate->op() ==
                         luisa::compute::xir::ArithmeticOp::BINARY_DIV)) {
                    arithmetic.emplace_back(candidate);
                }
            });
        const auto dominance =
            luisa::compute::xir::compute_dom_tree(function);
        arithmetic_count += arithmetic.size();
        for (auto *candidate : arithmetic) {
            const auto guarded = std::any_of(
                branches.begin(), branches.end(), [&](auto *branch) {
                    return dominance.dominates(
                        branch->true_block(), candidate->parent_block());
                });
            guarded_arithmetic_count += guarded ? 1u : 0u;
        }
    }
    return arithmetic_count >= 4u &&
           guarded_arithmetic_count == arithmetic_count;
}

[[nodiscard]] Mat4f translated(float x, float y, float z) noexcept {
    Mat4f result;
    result.elements[12u] = x;
    result.elements[13u] = y;
    result.elements[14u] = z;
    return result;
}

[[nodiscard]] ShaderGraph receiver_shader() {
    ShaderGraph graph;
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf, "Distant-light receiver");
    const auto configured =
        graph.set_input(
            diffuse, "Color", SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(
            diffuse, "Roughness", SocketValue::floating(0.0f));
    if (!configured) {
        throw std::runtime_error{"failed to configure receiver shader"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph distant_light_shader() {
    ShaderGraph graph;
    const auto falloff = graph.add_node(
        node_type::light_falloff, "Cycles Light Falloff");
    const auto emission = graph.add_node(
        node_type::emission, "Cycles distant emission");
    const auto configured =
        graph.set_input(
            falloff, "Strength", SocketValue::floating(2.0f)) &&
        graph.set_input(
            falloff, "Smooth", SocketValue::floating(0.0f)) &&
        graph.set_input(
            emission,
            "Color",
            SocketValue::color({1.0f, 0.72074449f, 0.52323443f})) &&
        graph.connect(
            {.node = falloff, .socket = "Quadratic"},
            emission,
            "Strength");
    if (!configured) {
        throw std::runtime_error{"failed to configure distant-light shader"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] SceneSnapshot make_distant_light_scene() {
    constexpr MaterialId receiver_material{1u};
    constexpr MaterialId light_material{2u};
    constexpr GeometryId geometry_id{3u};
    constexpr InstanceId instance_id{4u};
    constexpr CameraId camera_id{5u};
    constexpr LightId light_id{6u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        receiver_material,
        MaterialDesc{.name = "Distant-light receiver",
                     .shader = receiver_shader(),
                     .cycles_shader_index = 0u});
    scene.materials.emplace(
        light_material,
        MaterialDesc{.name = "Cycles Light Falloff distant shader",
                     .shader = distant_light_shader(),
                     .cycles_shader_index = 1u});

    TriangleMeshDesc mesh;
    mesh.name = "Distant-light receiver";
    mesh.positions = {
        {-4.0f, -4.0f, 0.0f},
        {4.0f, -4.0f, 0.0f},
        {0.0f, 4.0f, 0.0f}};
    mesh.normals.values.assign(
        mesh.positions.size(), Vec3f{0.0f, 0.0f, 1.0f});
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.material_slots = {receiver_material};
    mesh.triangle_material_slots = {0u};
    mesh.triangle_smooth = {1u};
    mesh.triangle_random_per_island = {0.0f};
    scene.geometries.emplace(geometry_id, std::move(mesh));
    scene.instances.emplace(
        instance_id,
        InstanceDesc{.name = "Distant-light receiver",
                     .geometry = geometry_id,
                     .transform = {},
                     .cycles_object_index = 0u});
    scene.cameras.emplace(
        camera_id,
        CameraDesc{.name = "Distant-light camera",
                   .projection = CameraProjection::orthographic,
                   .transform = translated(0.0f, 0.0f, 3.0f),
                   .orthographic_scale = 1.0f,
                   .near_clip = 0.1f,
                   .far_clip = 100.0f});
    scene.active_camera = camera_id;
    scene.lights.emplace(
        light_id,
        LightDesc{.name = "Cycles Light Falloff sun",
                  .type = LightType::distant,
                  .transform = {},
                  .color = {1.0f, 1.0f, 1.0f},
                  .power = 1.0f,
                  .angle = 0.1f,
                  .normalize = false,
                  .shader = light_material,
                  .use_mis = true,
                  .cast_shadow = true,
                  .visibility_mask = all_ray_visibility,
                  .cycles_shader_index = 1u,
                  .cycles_object_index = 1u});
    scene.world_sampling = WorldSampling::none;
    return scene;
}

[[nodiscard]] RenderSettings make_settings() {
    return {
        .full_extent = {.width = 1u, .height = 1u},
        .window = {},
        .seed = 0u,
        .transparent_background = false,
        .pixel_filter = PixelFilter::box,
        .filter_width = 1.0f,
        .pass_alpha_threshold = 0.5f,
        .integrator = {
            .max_bounces = 1u,
            .min_bounces = 0u,
            .diffuse_bounces = 0u,
            .glossy_bounces = 0u,
            .transmission_bounces = 0u,
            .volume_bounces = 0u,
            .transparent_min_bounces = 0u,
            .transparent_max_bounces = 0u,
            .sample_clamp_direct = 0.0f,
            .sample_clamp_indirect = 0.0f,
            .filter_glossy = 0.0f,
            .film_exposure = 1.0f,
            // This is a one-sample transport semantic regression, not a
            // stochastic-termination test. Disable Cycles' light-sample
            // roulette so a valid distant proposal cannot be discarded by
            // the chosen sample's random variate.
            .light_sampling_threshold = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::multiple_importance_sampling},
        .passes = {{.kind = PassKind::combined,
                    .name = "Combined",
                    .light_group = {},
                    .channels = 4u}}};
}

class CapturingPathTraceSink final
    : public psycles::luisa_backend::LuisaPathTraceSink {

  private:
    std::optional<psycles::luisa_backend::LuisaPathTrace> _trace;

  public:
    void write(const psycles::luisa_backend::LuisaPathTrace &trace) override {
        _trace = trace;
    }

    [[nodiscard]] const auto &trace() const noexcept { return _trace; }
};

[[nodiscard]] bool validate_distant_scene(
    luisa::compute::Context &context,
    std::string_view backend) {
    auto device = context.create_device(backend);
    auto trace_sink = std::make_shared<CapturingPathTraceSink>();
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true,
         .max_samples_per_dispatch = 1u,
         .path_trace = psycles::luisa_backend::LuisaPathTraceRequest{
             .pixel_x = 0u,
             .pixel_y = 0u,
             .sample = 0u,
             .sink = trace_sink}}};
    const auto compilation = renderer.compile_scene(make_distant_light_scene());
    if (!compilation.ok()) {
        for (const auto &diagnostic : compilation.diagnostics) {
            std::cerr << diagnostic.message << '\n';
        }
        return false;
    }
    auto session = renderer.create_session(*compilation.scene, make_settings());
    if (!session) {
        std::cerr << "could not create distant-light session on "
                  << backend << '\n';
        return false;
    }
    psycles::io::MemoryOutputSink output;
    if (!session->render_samples(
            {.first = 0u, .count = 1u, .offset = 0u, .total = 1u},
            output)) {
        std::cerr << "distant Light Falloff render failed on "
                  << backend << '\n';
        return false;
    }
    if (!trace_sink->trace()) {
        std::cerr << "distant Light Falloff trace was not delivered on "
                  << backend << '\n';
        return false;
    }

    using psycles::luisa_backend::path_trace_schema::EventSlot;
    using psycles::luisa_backend::path_trace_schema::index;
    const auto &trace = *trace_sink->trace();
    const auto &slot = [&](EventSlot field) -> const auto & {
        return trace.slots[index(0u, field)];
    };
    const auto &evaluation = slot(EventSlot::light_eval);
    const auto &light_shader = slot(EventSlot::nee_light_shader);
    const auto &unshadowed = slot(EventSlot::nee_unshadowed);
    constexpr std::array expected_shader{
        2.0f, 1.44148898f, 1.04646886f};
    auto valid = evaluation[3u] == 1.0f &&
                 evaluation[0u] == std::numeric_limits<float>::max() &&
                 light_shader[3u] == 1.0f && unshadowed[3u] == 1.0f;
    for (auto channel = std::size_t{0u}; channel < 3u; ++channel) {
        valid = valid && std::isfinite(light_shader[channel]) &&
                approximately_equal(
                    light_shader[channel], expected_shader[channel]) &&
                std::isfinite(unshadowed[channel]) &&
                unshadowed[channel] > 0.0f;
    }
    const auto *combined = output.find(PassKind::combined);
    valid = valid && combined != nullptr && combined->pixels.size() == 4u &&
            std::isfinite(combined->pixels[0u]) &&
            std::isfinite(combined->pixels[1u]) &&
            std::isfinite(combined->pixels[2u]) &&
            combined->pixels[0u] > 0.0f && combined->pixels[1u] > 0.0f &&
            combined->pixels[2u] > 0.0f;
    if (!valid) {
        std::cerr << "distant Light Falloff sentinel/NEE regression on "
                  << backend << '\n'
                  << "  light_eval = (" << evaluation[0u] << ", "
                  << evaluation[1u] << ", " << evaluation[2u] << ", "
                  << evaluation[3u] << ")\n"
                  << "  nee_light_shader = (" << light_shader[0u] << ", "
                  << light_shader[1u] << ", " << light_shader[2u] << ", "
                  << light_shader[3u] << ")\n"
                  << "  nee_unshadowed = (" << unshadowed[0u] << ", "
                  << unshadowed[1u] << ", " << unshadowed[2u] << ", "
                  << unshadowed[3u] << ")\n";
        if (combined == nullptr) {
            std::cerr << "  Combined = missing\n";
        } else if (combined->pixels.size() != 4u) {
            std::cerr << "  Combined channels = "
                      << combined->pixels.size() << '\n';
        } else {
            std::cerr << "  Combined = (" << combined->pixels[0u] << ", "
                      << combined->pixels[1u] << ", "
                      << combined->pixels[2u] << ", "
                      << combined->pixels[3u] << ")\n";
        }
    }
    return valid;
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    if (!short_circuit_is_structural()) {
        std::cerr << "Light Falloff distance arithmetic escaped the finite-ray "
                     "control region\n";
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    {
        // Vulkan currently permits one live Device per process. End the
        // low-level SVM/static parity fixture before the renderer creates its
        // own Device; the two phases share no device-side state.
        auto device = context.create_device(backend);
        auto stream = device.create_stream();
        const ShaderOption uncached{.enable_cache = false};
        auto svm_shader = device.compile(make_svm_kernel(), uncached);
        auto static_shader = device.compile(make_static_kernel(), uncached);
        auto svm_buffer = device.create_buffer<float>(9u);
        auto static_buffer = device.create_buffer<float>(9u);
        std::array<float, 9u> svm{};
        std::array<float, 9u> statically_selected{};
        stream << svm_shader(svm_buffer).dispatch(3u)
               << static_shader(static_buffer).dispatch(3u)
               << svm_buffer.copy_to(luisa::span{svm})
               << static_buffer.copy_to(luisa::span{statically_selected})
               << synchronize();
        constexpr std::array expected{
            2.0f, 1.5f, 2.0f,
            8.0f, 3.0f, 2.0f,
            32.0f, 6.0f, 2.0f};
        for (auto index = std::size_t{0u}; index < expected.size(); ++index) {
            if (!approximately_equal(svm[index], expected[index]) ||
                !approximately_equal(
                    statically_selected[index], expected[index]) ||
                !std::isfinite(svm[index]) ||
                !std::isfinite(statically_selected[index])) {
                std::cerr << "Light Falloff semantic mismatch on " << backend
                          << " at " << index << ": svm=" << svm[index]
                          << ", static=" << statically_selected[index]
                          << ", expected=" << expected[index] << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    return validate_distant_scene(context, backend)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
