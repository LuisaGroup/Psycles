#include <psycles/compiler/core_nodes.h>
#include <psycles/contract/scene.h>
#include <psycles/io/image.h>
#include <psycles/luisa/path_tracer.h>

#include "../src/luisa/path_kernel_emissive_triangle.h"
#include "../src/luisa/path_kernel_environment_light.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <luisa/runtime/context.h>

namespace {

namespace proposal_contract {

using namespace psycles::luisa_backend::detail;

template<typename T>
concept CarriesRadiance = requires(T value) {
    value.radiance;
};

template<typename T>
concept CanSampleTexture = requires(
    const T &services,
    UInt handle,
    Float2 uv) {
    services.texture_2d(
        handle,
        uv,
        uv,
        uv,
        0u,
        0u);
};

using EmissiveProposalFunction =
    EmissiveTriangleLightProposal (
        EmissiveTriangleComponent::*)(
        const std::shared_ptr<LuisaSceneData> &,
        UInt,
        Float3,
        Float2,
        Float) const noexcept;
using EmissiveEvaluationFunction =
    Float3 (EmissiveTriangleComponent::*)(
        PathSampleContext &,
        const EmissiveTriangleLightProposal &)
        const noexcept;
using EmissiveConstantEvaluationFunction =
    Float3 (EmissiveTriangleComponent::*)(
        PathSampleContext &,
        const EmissiveTriangleLightProposal &)
        const noexcept;
using SurfaceConstantEvaluationFunction =
    Float3 (psycles::luisa_backend::Surface::*)(
        const psycles::luisa_backend::
            SurfaceParameterServices &,
        Expr<std::uint32_t>) const noexcept;
using EnvironmentProposalFunction =
    EnvironmentLightProposal (
        EnvironmentLightComponent::*)(
        const std::shared_ptr<LuisaSceneData> &,
        Float3,
        Float2,
        Float,
        UInt,
        UInt) const noexcept;
using EnvironmentEvaluationFunction =
    Float3 (EnvironmentLightComponent::*)(
        PathSampleContext &,
        Float3,
        const psycles::luisa_backend::
            cycles_path_state::
                ShaderEvaluationState &)
        const noexcept;
using EnvironmentConstantEvaluationFunction =
    Float3 (EnvironmentLightComponent::*)(
        PathSampleContext &) const noexcept;

static_assert(
    !CarriesRadiance<
        EmissiveTriangleLightProposal>);
static_assert(
    !CarriesRadiance<
        EnvironmentLightProposal>);
static_assert(
    !CanSampleTexture<
        psycles::luisa_backend::
            SurfaceParameterServices>);
static_assert(
    std::same_as<
        decltype(
            &EmissiveTriangleComponent::
                from_position),
        EmissiveProposalFunction>);
static_assert(
    std::same_as<
        decltype(
            &EmissiveTriangleComponent::
                evaluate_emission),
        EmissiveEvaluationFunction>);
static_assert(
    std::same_as<
        decltype(
            &EmissiveTriangleComponent::
                evaluate_constant_emission),
        EmissiveConstantEvaluationFunction>);
static_assert(
    std::same_as<
        decltype(
            &psycles::luisa_backend::Surface::
                constant_emission),
        SurfaceConstantEvaluationFunction>);
static_assert(
    std::same_as<
        decltype(
            &EnvironmentLightComponent::
                from_position),
        EnvironmentProposalFunction>);
static_assert(
    std::same_as<
        decltype(
            &EnvironmentLightComponent::
                evaluate_emission),
        EnvironmentEvaluationFunction>);
static_assert(
    std::same_as<
        decltype(
            &EnvironmentLightComponent::
                evaluate_constant_emission),
        EnvironmentConstantEvaluationFunction>);

}// namespace proposal_contract

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;

constexpr auto pi = 3.14159265358979323846f;
constexpr std::uint32_t width = 32u;
constexpr std::uint32_t height = 32u;
constexpr std::string_view oracle_color_attribute{
    "Cycles rectangle oracle color"};
constexpr std::string_view point_identity_attribute{
    "Cycles rectangle point identity"};
constexpr std::string_view face_identity_attribute{
    "Cycles rectangle face identity"};
constexpr std::string_view zero_emission_attribute{
    "Cycles zero emission attribute"};

[[nodiscard]] Mat4f translated(
    float x,
    float y,
    float z) noexcept {
    auto result = Mat4f{};
    result.elements[12u] = x;
    result.elements[13u] = y;
    result.elements[14u] = z;
    return result;
}

[[nodiscard]] ShaderGraph diffuse_shader() {
    ShaderGraph graph;
    const auto oracle_color =
        graph.add_node(node_type::vertex_color, "Corner Color");
    const auto point_identity =
        graph.add_node(node_type::vertex_color, "Point Identity");
    const auto face_identity =
        graph.add_node(node_type::vertex_color, "Face Identity");
    const auto multiply_point =
        graph.add_node(node_type::multiply_color, "Multiply Point");
    const auto multiply_face =
        graph.add_node(node_type::multiply_color, "Multiply Face");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    const auto bind_attribute =
        [&](NodeId node, std::string_view name) {
            static_cast<void>(graph.set_property(
                node,
                "AttributeId",
                SocketValue::unsigned_integer(
                    attribute_id(name))));
        };
    bind_attribute(oracle_color, oracle_color_attribute);
    bind_attribute(point_identity, point_identity_attribute);
    bind_attribute(face_identity, face_identity_attribute);
    static_cast<void>(graph.connect(
        {.node = oracle_color, .socket = "Color"},
        multiply_point,
        "A"));
    static_cast<void>(graph.connect(
        {.node = point_identity, .socket = "Color"},
        multiply_point,
        "B"));
    static_cast<void>(graph.connect(
        {.node = multiply_point, .socket = "Color"},
        multiply_face,
        "A"));
    static_cast<void>(graph.connect(
        {.node = face_identity, .socket = "Color"},
        multiply_face,
        "B"));
    static_cast<void>(graph.connect(
        {.node = multiply_face, .socket = "Color"},
        diffuse,
        "Color"));
    static_cast<void>(graph.set_input(
        diffuse,
        "Roughness",
        SocketValue::floating(0.0f)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph transparent_shader() {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(
            node_type::transparent_bsdf,
            "Cycles non-evaluable transparent closure");
    static_cast<void>(graph.set_input(
        transparent,
        "Color",
        SocketValue::color({1.0f, 1.0f, 1.0f})));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = transparent,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph emission_shader() {
    ShaderGraph graph;
    const auto zero = graph.add_node(
        node_type::vertex_color,
        "Device zero emission attribute");
    const auto emission =
        graph.add_node(
            node_type::emission,
            "Raw triangle emission closure");
    static_cast<void>(graph.set_property(
        zero,
        "AttributeId",
        SocketValue::unsigned_integer(
            attribute_id(zero_emission_attribute))));
    static_cast<void>(graph.set_input(
        emission,
        "Strength",
        SocketValue::floating(1.0f)));
    static_cast<void>(graph.connect(
        {.node = zero, .socket = "Color"},
        emission,
        "Color"));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = emission,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph spatial_world_shader() {
    ShaderGraph graph;
    const auto coordinates =
        graph.add_node(
            node_type::texture_coordinate,
            "World Texture Coordinate");
    const auto gradient =
        graph.add_node(
            node_type::gradient_texture,
            "Linear world gradient");
    const auto gray =
        graph.add_node(
            node_type::scalar_to_color,
            "Gradient to color");
    const auto background =
        graph.add_node(
            node_type::emission,
            "Raw world emission closure");
    static_cast<void>(graph.set_property(
        gradient,
        "GradientType",
        SocketValue::string("LINEAR")));
    static_cast<void>(graph.set_input(
        background,
        "Strength",
        SocketValue::floating(0.5f)));
    static_cast<void>(graph.connect(
        {.node = coordinates,
         .socket = "Generated"},
        gradient,
        "Vector"));
    static_cast<void>(graph.connect(
        {.node = gradient,
         .socket = "Factor"},
        gray,
        "Value"));
    static_cast<void>(graph.connect(
        {.node = gray,
         .socket = "Color"},
        background,
        "Color"));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = background,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] SceneSnapshot make_scene(
    bool narrow_spread_ellipse = false) {
    constexpr MaterialId material_id{1u};
    constexpr GeometryId geometry_id{2u};
    constexpr InstanceId instance_id{3u};
    constexpr CameraId camera_id{4u};
    constexpr LightId light_id{5u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        material_id,
        MaterialDesc{
            .name = "Cycles rectangle oracle diffuse",
            .shader = diffuse_shader(),
            .cycles_shader_index = 6u});

    TriangleMeshDesc mesh;
    mesh.name = "Cycles rectangle oracle plane";
    mesh.positions = {
        {-4.0f, -4.0f, 0.0f},
        {4.0f, -4.0f, 0.0f},
        {4.0f, 4.0f, 0.0f},
        {-4.0f, 4.0f, 0.0f}};
    mesh.normals.values.assign(
        mesh.positions.size(),
        Vec3f{0.0f, 0.0f, 1.0f});
    mesh.uv.values = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f}};
    mesh.uv.domain = MeshAttributeDomain::corner;
    auto &oracle_color =
        mesh.color_attributes[
            std::string{oracle_color_attribute}];
    oracle_color.domain = MeshAttributeDomain::corner;
    oracle_color.values.assign(
        mesh.uv.values.size(),
        Vec4f{
            0.6200000047683716f,
            0.4099999964237213f,
            0.23000000417232513f,
            1.0f});
    auto &point_identity =
        mesh.color_attributes[
            std::string{point_identity_attribute}];
    point_identity.domain = MeshAttributeDomain::point;
    point_identity.values.assign(
        mesh.positions.size(),
        Vec4f{1.0f, 1.0f, 1.0f, 1.0f});
    auto &face_identity =
        mesh.color_attributes[
            std::string{face_identity_attribute}];
    face_identity.domain = MeshAttributeDomain::face;
    face_identity.values.assign(
        2u,
        Vec4f{1.0f, 1.0f, 1.0f, 1.0f});
    mesh.triangles = {
        {0u, 1u, 2u},
        {0u, 2u, 3u}};
    mesh.material_slots = {material_id};
    mesh.triangle_material_slots = {0u, 0u};
    mesh.triangle_smooth = {0u, 0u};
    mesh.triangle_random_per_island = {
        0.874585747718811f,
        0.874585747718811f};
    scene.geometries.emplace(
        geometry_id, std::move(mesh));
    scene.instances.emplace(
        instance_id,
        InstanceDesc{
            .name = "Cycles rectangle oracle plane",
            .geometry = geometry_id,
            .transform = {},
            .cycles_object_index = 0u});

    scene.cameras.emplace(
        camera_id,
        CameraDesc{
            .name = "Cycles rectangle oracle camera",
            .projection = CameraProjection::orthographic,
            .transform = translated(0.0f, 0.0f, 3.0f),
            .field_of_view = 0.4710899591445923f,
            .horizontal_field_of_view =
                0.6911112070083618f,
            .sensor_fit = CameraSensorFit::automatic,
            .orthographic_scale = 2.200000047683716f,
            .near_clip = 0.10000000149011612f,
            .far_clip = 1000.0f});
    scene.active_camera = camera_id;

    scene.lights.emplace(
        light_id,
        LightDesc{
            .name =
                narrow_spread_ellipse
                    ? "Cycles narrow ellipse oracle light"
                    : "Cycles rectangle oracle light",
            .type = LightType::area,
            .transform = translated(
                0.3700000047683716f,
                -0.20999999344348907f,
                1.399999976158142f),
            .color = {
                0.36000001430511475f,
                0.7200000286102295f,
                1.0f},
            .power = 37.0f,
            .size = 0.800000011920929f,
            .size_y = 0.5f,
            .spread =
                narrow_spread_ellipse
                    ? 1.2f
                    : pi,
            .normalize = true,
            .ellipse =
                narrow_spread_ellipse,
            .is_sphere = false,
            .use_mis = true,
            .cast_shadow = true,
            .visibility_mask =
                all_ray_visibility &
                ~visibility_bit(RayVisibility::camera),
            .is_shadow_catcher = true,
            .cycles_shader_index = 5u,
            .cycles_object_index = 1u});
    scene.world_sampling = WorldSampling::automatic;
    return scene;
}

[[nodiscard]] SceneSnapshot make_mesh_light_scene() {
    constexpr MaterialId emitter_material{6u};
    constexpr GeometryId emitter_geometry{7u};
    constexpr InstanceId emitter_instance{8u};
    auto scene = make_scene();
    scene.revision = 2u;
    scene.lights.clear();
    scene.materials.emplace(
        emitter_material,
        MaterialDesc{
            .name = "Cycles raw triangle emitter",
            .shader = emission_shader(),
            .emission_sampling = EmissionSampling::front,
            .cycles_shader_index = 11u});
    TriangleMeshDesc mesh;
    mesh.name = "Cycles direct-light trace triangle";
    mesh.positions = {
        {1.4f, -0.6f, 1.4f},
        {2.0f, 0.6f, 1.4f},
        {2.6f, -0.6f, 1.4f}};
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.material_slots = {emitter_material};
    mesh.triangle_material_slots = {0u};
    mesh.triangle_smooth = {0u};
    mesh.cycles_primitive_offset = 41u;
    auto &zero_emission =
        mesh.color_attributes[
            std::string{zero_emission_attribute}];
    zero_emission.domain = MeshAttributeDomain::point;
    zero_emission.values.assign(
        mesh.positions.size(),
        Vec4f{0.0f, 0.0f, 0.0f, 1.0f});
    scene.geometries.emplace(
        emitter_geometry,
        std::move(mesh));
    scene.instances.emplace(
        emitter_instance,
        InstanceDesc{
            .name = "Cycles direct-light trace triangle",
            .geometry = emitter_geometry,
            .transform = {},
            .cycles_object_index = 9u,
            .cycles_light_group = 6,
            .is_shadow_catcher = true});
    return scene;
}

[[nodiscard]] SceneSnapshot make_environment_light_scene() {
    constexpr MaterialId world_material{6u};
    auto scene = make_scene();
    scene.revision = 3u;
    scene.lights.clear();
    scene.materials.emplace(
        world_material,
        MaterialDesc{
            .name = "Cycles spatial world emitter",
            .shader = spatial_world_shader(),
            .cycles_shader_index = 12u});
    scene.world_shader = world_material;
    scene.world_sampling = WorldSampling::manual;
    scene.world_sample_map_resolution = 4u;
    scene.cycles_background_object_index = 10u;
    scene.cycles_background_light_group = 4;
    return scene;
}

[[nodiscard]] SceneSnapshot make_no_direct_light_scene() {
    auto scene = make_scene();
    scene.revision = 4u;
    scene.lights.clear();
    scene.world_sampling = WorldSampling::none;
    return scene;
}

[[nodiscard]] SceneSnapshot make_no_evaluable_bsdf_scene() {
    constexpr MaterialId material_id{1u};
    auto scene = make_scene();
    scene.revision = 5u;
    scene.materials.at(material_id).shader =
        transparent_shader();
    return scene;
}

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 2.0e-6f) noexcept {
    // Keep a fixed zero-energy bound while scaling the float32 comparison
    // for narrow-spread radiance above one. This is one symmetric norm for
    // every fixture/backend, not a per-pixel exception.
    return std::abs(actual - expected) <=
           tolerance *
               (1.0f +
                std::max(
                    std::abs(actual),
                    std::abs(expected)));
}

class CapturingPathTraceSink final
    : public psycles::luisa_backend::LuisaPathTraceSink {

  private:
    std::optional<psycles::luisa_backend::LuisaPathTrace>
        _trace;

  public:
    void write(
        const psycles::luisa_backend::LuisaPathTrace &trace)
        override {
        _trace = trace;
    }

    void reset() noexcept { _trace.reset(); }

    [[nodiscard]] const auto &trace() const noexcept {
        return _trace;
    }
};

struct DirectLightTraceOracle {
    float type;
    float emitter;
    float primitive;
    float object;
    float light_group;
    float shader_low;
    float shader_high;
    bool environment;
};

[[nodiscard]] bool validate_direct_light_trace(
    const CapturingPathTraceSink &sink,
    std::string_view backend,
    std::string_view label,
    const DirectLightTraceOracle &oracle) {
    using psycles::luisa_backend::path_trace_schema::EventSlot;
    using psycles::luisa_backend::path_trace_schema::index;
    if (!sink.trace()) {
        std::cerr << label
                  << " path trace was not delivered on "
                  << backend << '\n';
        return false;
    }
    const auto &trace = *sink.trace();
    const auto &slot = [&](EventSlot field) -> const auto & {
        return trace.slots[index(0u, field)];
    };
    auto passed = true;
    const auto require = [&](bool condition,
                             std::string_view field) {
        if (!condition) {
            std::cerr << "Cycles " << label
                      << " direct-light trace regression on "
                      << backend << " field " << field
                      << '\n';
            passed = false;
        }
    };
    const auto &meta = slot(EventSlot::light_meta);
    const auto &identity = slot(EventSlot::light_id);
    const auto &shader = slot(EventSlot::light_shader);
    const auto &pdf = slot(EventSlot::light_pdf);
    const auto &direction = slot(EventSlot::light_d);
    const auto &position = slot(EventSlot::light_p);
    const auto &normal = slot(EventSlot::light_ng);
    const auto &evaluation = slot(EventSlot::light_eval);
    require(meta[3u] == 1.0f &&
                meta[0u] == oracle.type &&
                meta[1u] == oracle.emitter &&
                meta[2u] == oracle.primitive,
            "light_meta");
    require(identity[3u] == 1.0f &&
                identity[0u] == oracle.object &&
                identity[1u] == oracle.light_group,
            "light_id");
    require(shader[3u] == 1.0f &&
                shader[0u] == oracle.shader_low &&
                shader[1u] == oracle.shader_high,
            "light_shader");
    require(pdf[3u] == 1.0f &&
                pdf[0u] > 0.0f &&
                approximately_equal(pdf[1u], 1.0f) &&
                pdf[2u] > 0.0f,
            "light_pdf");
    const auto direction_length =
        std::sqrt(direction[0u] * direction[0u] +
                  direction[1u] * direction[1u] +
                  direction[2u] * direction[2u]);
    require(direction[3u] == 1.0f &&
                approximately_equal(direction_length, 1.0f),
            "light_d");
    require(position[3u] == 1.0f &&
                normal[3u] == 1.0f,
            "light_position_normal_presence");
    require(evaluation[3u] == 1.0f &&
                evaluation[1u] >= 0.0f &&
                evaluation[2u] > 0.0f &&
                evaluation[2u] <= 1.0f,
            "light_eval");
    if (oracle.environment) {
        require(evaluation[0u] >= 1.0e30f &&
                    approximately_equal(pdf[2u], 1.0f),
                "environment_distance_eval_factor");
        for (std::size_t channel = 0u;
             channel < 3u;
             ++channel) {
            require(
                approximately_equal(
                    position[channel],
                    -direction[channel]) &&
                    approximately_equal(
                        normal[channel],
                        -direction[channel]),
                "environment_P_Ng");
        }
    } else {
        const auto &surface = slot(EventSlot::surface_p);
        const auto dx = position[0u] - surface[0u];
        const auto dy = position[1u] - surface[1u];
        const auto dz = position[2u] - surface[2u];
        const auto distance =
            std::sqrt(dx * dx + dy * dy + dz * dz);
        require(surface[3u] == 1.0f &&
                    evaluation[1u] > 0.0f &&
                    approximately_equal(
                        evaluation[0u], distance),
                "finite_light_distance");
        require(
            approximately_equal(
                direction[0u], dx / distance) &&
                approximately_equal(
                    direction[1u], dy / distance) &&
                approximately_equal(
                    direction[2u], dz / distance),
            "finite_light_direction");
    }
    return passed;
}

[[nodiscard]] bool validate_direct_light_not_attempted(
    const CapturingPathTraceSink &sink,
    std::string_view backend,
    std::string_view label) {
    using psycles::luisa_backend::path_trace_schema::EventSlot;
    using psycles::luisa_backend::path_trace_schema::index;
    if (!sink.trace()) {
        std::cerr << label
                  << " path trace was not delivered on "
                  << backend << '\n';
        return false;
    }
    const auto &trace = *sink.trace();
    const auto &slot = [&](EventSlot field) -> const auto & {
        return trace.slots[index(0u, field)];
    };
    if (slot(EventSlot::surface_p)[3u] != 1.0f) {
        std::cerr << "Cycles " << label
                  << " did not reach surface shading on "
                  << backend << '\n';
        return false;
    }
    struct TraceField {
        EventSlot slot;
        std::string_view name;
    };
    constexpr std::array fields{
        TraceField{EventSlot::light_meta, "light_meta"},
        TraceField{EventSlot::light_id, "light_id"},
        TraceField{EventSlot::light_shader, "light_shader"},
        TraceField{EventSlot::light_pdf, "light_pdf"},
        TraceField{EventSlot::light_d, "light_d"},
        TraceField{EventSlot::light_p, "light_p"},
        TraceField{EventSlot::light_ng, "light_ng"},
        TraceField{EventSlot::light_eval, "light_eval"}};
    auto passed = true;
    for (const auto &field : fields) {
        if (slot(field.slot)[3u] != 0.0f) {
            std::cerr << "Cycles " << label
                      << " unexpectedly attempted direct lighting on "
                      << backend << " field "
                      << field.name << '\n';
            passed = false;
        }
    }
    return passed;
}

[[nodiscard]] RenderSettings make_settings() {
    return {
        .full_extent = {
            .width = width,
            .height = height},
        .window = {},
        .seed = 20903u,
        .transparent_background = false,
        .pixel_filter =
            PixelFilter::blackman_harris,
        .filter_width = 1.5f,
        .pass_alpha_threshold = 0.5f,
        .integrator = {
            .max_bounces = 1u,
            .min_bounces = 0u,
            .diffuse_bounces = 0u,
            .glossy_bounces = 0u,
            .transmission_bounces = 0u,
            .volume_bounces = 0u,
            .transparent_min_bounces = 0u,
            .transparent_max_bounces = 8u,
            .sample_clamp_direct = 0.0f,
            .sample_clamp_indirect = 10.0f,
            .filter_glossy = 1.0f,
            .film_exposure = 1.0f,
            .light_sampling_threshold =
                0.009999999776482582f,
            .reflective_caustics = true,
            .refractive_caustics = true,
            .use_light_tree = false,
            .direct_light_sampling =
                DirectLightSampling::
                    multiple_importance_sampling},
        .passes = {
            {.kind = PassKind::combined,
             .name = "Combined",
             .light_group = {},
             .channels = 4u}}};
}

[[nodiscard]] bool render_fixture(
    psycles::luisa_backend::
        LuisaPathTracerBackend &renderer,
    SceneSnapshot scene,
    const RenderSettings &settings,
    std::string_view label,
    psycles::io::MemoryOutputSink
        &sink) {
    auto compilation =
        renderer.compile_scene(scene);
    if (!compilation.ok()) {
        for (const auto &diagnostic :
             compilation.diagnostics) {
            std::cerr << diagnostic.message
                      << '\n';
        }
        return false;
    }
    auto session =
        renderer.create_session(
            *compilation.scene,
            settings);
    if (!session) {
        std::cerr
            << "could not create " << label
            << " render session\n";
        return false;
    }
    if (!session->render_samples(
            {.first = 0u,
             .count = 1u,
             .offset = 0u,
             .total = 1u},
            sink)) {
        std::cerr << label
                  << " render failed\n";
        return false;
    }
    return true;
}

struct PixelOracle {
    std::uint32_t x;
    std::uint32_t y;
    std::array<float, 4u> value;
};

template<std::size_t N>
[[nodiscard]] bool validate_fixture(
    const psycles::io::MemoryOutputSink
        &sink,
    std::string_view backend,
    std::string_view label,
    const std::array<double, 3u>
        &expected_means,
    const std::array<PixelOracle, N>
        &pixel_oracles) {
    const auto *combined =
        sink.find(PassKind::combined);
    if (combined == nullptr ||
        combined->channels != 4u ||
        combined->pixels.size() !=
            static_cast<std::size_t>(
                width) *
                height *
                combined->channels) {
        std::cerr << label
                  << " Combined pass has an invalid shape\n";
        return false;
    }

    std::array<double, 3u> means{};
    auto passed = true;
    for (std::size_t pixel = 0u;
         pixel <
         static_cast<std::size_t>(
             width) *
             height;
         ++pixel) {
        const auto base =
            pixel * combined->channels;
        for (std::size_t channel = 0u;
             channel < means.size();
             ++channel) {
            means[channel] +=
                combined->pixels[
                    base + channel];
        }
        if (!approximately_equal(
                combined->pixels[
                    base + 3u],
                1.0f)) {
            std::cerr << label
                      << " alpha regression failed on "
                      << backend << " pixel "
                      << pixel << '\n';
            passed = false;
        }
    }
    for (auto &mean : means) {
        mean /=
            static_cast<double>(
                width * height);
    }
    for (std::size_t channel = 0u;
         channel < means.size();
         ++channel) {
        if (std::abs(
                means[channel] -
                expected_means[channel]) >
            2.0e-6) {
            std::cerr
                << "Cycles " << label
                << " full-frame mean regression failed on "
                << backend << " channel "
                << channel << ": got "
                << means[channel]
                << ", expected "
                << expected_means[channel]
                << '\n';
            passed = false;
        }
    }
    for (const auto &oracle :
         pixel_oracles) {
        const auto base =
            (static_cast<std::size_t>(
                 oracle.y) *
                 width +
             oracle.x) *
            combined->channels;
        for (std::size_t channel = 0u;
             channel <
             oracle.value.size();
             ++channel) {
            if (!approximately_equal(
                    combined->pixels[
                        base + channel],
                    oracle.value[
                        channel])) {
                std::cerr
                    << std::setprecision(10)
                    << "Cycles " << label
                    << " pixel regression failed on "
                    << backend << " at ("
                    << oracle.x << ", "
                    << oracle.y << ") channel "
                    << channel << ": got "
                    << combined->pixels[
                           base + channel]
                    << ", expected "
                    << oracle.value[channel]
                    << '\n';
                passed = false;
            }
        }
    }
    return passed;
}

[[nodiscard]] bool validate_exact_black_fixture(
    const psycles::io::MemoryOutputSink &sink,
    std::string_view backend,
    std::string_view label) {
    const auto *combined =
        sink.find(PassKind::combined);
    if (combined == nullptr ||
        combined->channels != 4u ||
        combined->pixels.size() !=
            static_cast<std::size_t>(width) * height * 4u) {
        std::cerr << label
                  << " Combined pass has an invalid shape on "
                  << backend << '\n';
        return false;
    }
    for (std::size_t pixel = 0u;
         pixel < static_cast<std::size_t>(width) * height;
         ++pixel) {
        const auto base = pixel * 4u;
        if (combined->pixels[base] != 0.0f ||
            combined->pixels[base + 1u] != 0.0f ||
            combined->pixels[base + 2u] != 0.0f ||
            combined->pixels[base + 3u] != 1.0f) {
            std::cerr << label
                      << " expected exact opaque black on "
                      << backend << " pixel "
                      << pixel << '\n';
            return false;
        }
    }
    return true;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(backend);
    auto trace_sink =
        std::make_shared<CapturingPathTraceSink>();
    psycles::luisa_backend::LuisaPathTracerBackend renderer{
        std::move(device),
        {.next_event_estimation = true,
         .max_samples_per_dispatch = 1u,
         .path_trace =
             psycles::luisa_backend::LuisaPathTraceRequest{
                 .pixel_x = 20u,
                 .pixel_y = 14u,
                 .sample = 0u,
                 .sink = trace_sink}}};
    const auto settings =
        make_settings();

    psycles::io::MemoryOutputSink
        rectangle_sink;
    if (!render_fixture(
            renderer,
            make_scene(),
            settings,
            "full-spread rectangle",
            rectangle_sink)) {
        return EXIT_FAILURE;
    }
    constexpr std::array
        rectangle_means{
            0.21808528017572826,
            0.2884353661502246,
            0.2247294618464366};
    constexpr std::array
        rectangle_pixels{
            PixelOracle{
                .x = 20u,
                .y = 14u,
                .value = {
                    0.37258365750312805f,
                    0.4927719235420227f,
                    0.38393476605415344f,
                    1.0f}}};
    if (!validate_fixture(
            rectangle_sink,
            backend,
            "full-spread rectangle",
            rectangle_means,
            rectangle_pixels)) {
        return EXIT_FAILURE;
    }
    if (!validate_direct_light_trace(
            *trace_sink,
            backend,
            "full-spread rectangle",
            {.type = 3.0f,
             .emitter = 0.0f,
             .primitive = 0.0f,
             .object = 1.0f,
             .light_group = -1.0f,
             .shader_low = 5.0f,
             .shader_high = 20736.0f,
             .environment = false})) {
        return EXIT_FAILURE;
    }

    // The same one-emitter fixture must be bit-equivalent through the real
    // scene-built Light Tree. This covers session admission, host upload,
    // surface selection, and the NEE selection PDF in the fused kernel.
    auto light_tree_settings = settings;
    light_tree_settings.integrator.use_light_tree = true;
    psycles::io::MemoryOutputSink light_tree_rectangle_sink;
    trace_sink->reset();
    if (!render_fixture(
            renderer,
            make_scene(),
            light_tree_settings,
            "light-tree full-spread rectangle",
            light_tree_rectangle_sink) ||
        !validate_fixture(
            light_tree_rectangle_sink,
            backend,
            "light-tree full-spread rectangle",
            rectangle_means,
            rectangle_pixels) ||
        !validate_direct_light_trace(
            *trace_sink,
            backend,
            "light-tree full-spread rectangle",
            {.type = 3.0f,
             .emitter = 0.0f,
             .primitive = 0.0f,
             .object = 1.0f,
             .light_group = -1.0f,
             .shader_low = 5.0f,
             .shader_high = 20736.0f,
             .environment = false})) {
        return EXIT_FAILURE;
    }

    psycles::io::MemoryOutputSink
        narrow_ellipse_sink;
    trace_sink->reset();
    if (!render_fixture(
            renderer,
            make_scene(true),
            settings,
            "narrow-spread ellipse",
            narrow_ellipse_sink)) {
        return EXIT_FAILURE;
    }
    // Official Blender 5.3.0 Alpha/Cycles main b82c3f0d, one Tabulated
    // Sobol sample. These records exercise the shared spread-clamped
    // rectangle/circle/ellipse construction from production surface NEE
    // and its BSDF-forward complement.
    constexpr std::array
        narrow_ellipse_means{
            0.5357027209195167,
            0.7085100397669066,
            0.552023389617716};
    constexpr std::array
        narrow_ellipse_pixels{
            PixelOracle{
                .x = 0u,
                .y = 0u,
                .value = {
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f}},
            PixelOracle{
                .x = 12u,
                .y = 10u,
                .value = {
                    0.07275009155273438f,
                    0.09621785581111908f,
                    0.07496648281812668f,
                    1.0f}},
            PixelOracle{
                .x = 17u,
                .y = 16u,
                .value = {
                    0.7915834188461304f,
                    1.0469329357147217f,
                    0.8156996369361877f,
                    1.0f}},
            PixelOracle{
                .x = 20u,
                .y = 14u,
                .value = {
                    2.6395480632781982f,
                    3.4910149574279785f,
                    2.719963788986206f,
                    1.0f}},
            PixelOracle{
                .x = 24u,
                .y = 22u,
                .value = {
                    1.5788248777389526f,
                    2.088123083114624f,
                    1.6269252300262451f,
                    1.0f}}};
    if (!validate_fixture(
            narrow_ellipse_sink,
            backend,
            "narrow-spread ellipse",
            narrow_ellipse_means,
            narrow_ellipse_pixels)) {
        return EXIT_FAILURE;
    }

    psycles::io::MemoryOutputSink
        mesh_light_sink;
    trace_sink->reset();
    if (!render_fixture(
            renderer,
            make_mesh_light_scene(),
            settings,
            "zero-energy raw triangle emitter",
            mesh_light_sink) ||
        !validate_direct_light_trace(
            *trace_sink,
            backend,
            "zero-energy raw triangle emitter",
            {.type = 5.0f,
             .emitter = 0.0f,
             .primitive = 41.0f,
             .object = 9.0f,
             .light_group = 6.0f,
             .shader_low = 11.0f,
             .shader_high = 20480.0f,
             .environment = false})) {
        return EXIT_FAILURE;
    }

    psycles::io::MemoryOutputSink
        environment_light_sink;
    trace_sink->reset();
    if (!render_fixture(
            renderer,
            make_environment_light_scene(),
            settings,
            "spatial environment emitter",
            environment_light_sink) ||
        !validate_direct_light_trace(
            *trace_sink,
            backend,
            "spatial environment emitter",
            {.type = 2.0f,
             .emitter = 0.0f,
             .primitive = 0.0f,
             .object = 10.0f,
             .light_group = 4.0f,
             .shader_low = 12.0f,
             .shader_high = 20480.0f,
             .environment = true})) {
        return EXIT_FAILURE;
    }

    psycles::io::MemoryOutputSink
        no_direct_light_sink;
    trace_sink->reset();
    if (!render_fixture(
            renderer,
            make_no_direct_light_scene(),
            settings,
            "zero-weight direct-light distribution",
            no_direct_light_sink) ||
        !validate_exact_black_fixture(
            no_direct_light_sink,
            backend,
            "zero-weight direct-light distribution") ||
        !validate_direct_light_not_attempted(
            *trace_sink,
            backend,
            "zero-weight direct-light distribution")) {
        return EXIT_FAILURE;
    }

    psycles::io::MemoryOutputSink
        no_evaluable_bsdf_sink;
    trace_sink->reset();
    if (!render_fixture(
            renderer,
            make_no_evaluable_bsdf_scene(),
            settings,
            "non-evaluable surface closure",
            no_evaluable_bsdf_sink) ||
        !validate_exact_black_fixture(
            no_evaluable_bsdf_sink,
            backend,
            "non-evaluable surface closure") ||
        !validate_direct_light_not_attempted(
            *trace_sink,
            backend,
            "non-evaluable surface closure")) {
        return EXIT_FAILURE;
    }

#if defined(PSYCLES_WITH_OPENIMAGEIO)
    if (argc > 2) {
        std::string error;
        if (!psycles::io::write_multilayer_exr(
                rectangle_sink.images(),
                argv[2],
                "ViewLayer",
                &error)) {
            std::cerr
                << "could not write rectangle "
                   "diagnostic EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
    if (argc > 3) {
        std::string error;
        if (!psycles::io::write_multilayer_exr(
                narrow_ellipse_sink.images(),
                argv[3],
                "ViewLayer",
                &error)) {
            std::cerr
                << "could not write narrow ellipse "
                   "diagnostic EXR: "
                << error << '\n';
            return EXIT_FAILURE;
        }
    }
#endif
    return EXIT_SUCCESS;
}
