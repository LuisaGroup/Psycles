#include "../src/luisa/path_tracer_analytic_light_scene.h"
#include "../src/luisa/path_tracer_light_tree.h"
#include "../src/luisa/path_tracer_light_tree_scene.h"
#include "../src/luisa/path_tracer_light_sampling_scene.h"
#include "../src/luisa/path_tracer_mesh_light_scene.h"
#include "../src/luisa/path_tracer_scene_geometry.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/sampling/light_distribution.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <luisa/runtime/context.h>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend::detail;
using namespace psycles::sampling;
using namespace luisa;
using namespace luisa::compute;

constexpr std::uint32_t emitter_count = 4u;
constexpr std::uint32_t sample_count = 8192u;

[[nodiscard]] bool close(float a, float b, float tolerance);

[[nodiscard]] ShaderGraph emission_graph(Vec3f color, float strength) {
    ShaderGraph graph;
    const auto emission =
        graph.add_node(node_type::emission, "Light-tree emission");
    static_cast<void>(graph.set_input(
        emission, "Color", SocketValue::color(color)));
    static_cast<void>(graph.set_input(
        emission, "Strength", SocketValue::floating(strength)));
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] bool verify_scene_upload() {
    constexpr MaterialId base_material{1u};
    constexpr MaterialId override_material{2u};
    constexpr GeometryId geometry_id{3u};
    constexpr InstanceId instance_id{4u};
    constexpr LightId light_id{5u};

    SceneSnapshot snapshot;
    snapshot.revision = 1u;
    snapshot.materials.emplace(
        base_material,
        MaterialDesc{
            .name = "base emitter",
            .shader = emission_graph({1.0f, 1.0f, 1.0f}, 0.25f),
            .emission_sampling = EmissionSampling::front});
    snapshot.materials.emplace(
        override_material,
        MaterialDesc{
            .name = "instance override emitter",
            .shader = emission_graph({0.25f, 0.5f, 1.0f}, 4.0f),
            .emission_sampling = EmissionSampling::front});
    TriangleMeshDesc mesh;
    mesh.name = "source support";
    mesh.positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}};
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.material_slots = {base_material};
    mesh.triangle_material_slots = {0u};
    snapshot.geometries.emplace(geometry_id, std::move(mesh));
    Mat4f transform;
    transform.elements[12u] = 10.0f;
    snapshot.instances.emplace(
        instance_id,
        InstanceDesc{
            .name = "overridden displaced emitter",
            .geometry = geometry_id,
            .transform = transform,
            .material_overrides = {override_material},
            .cycles_object_index = 9u});
    snapshot.lights.emplace(
        light_id,
        LightDesc{.name = "point emitter", .type = LightType::point});

    LuisaSceneData scene;
    ShaderCompiler compiler{make_core_node_registry()};
    const auto material_update = scene.materials.update(snapshot, compiler);
    if (!material_update.committed) {
        std::cerr << "light-tree scene material compilation failed\n";
        return false;
    }

    std::array<GeometryUpload, 1u> uploads;
    // These are the final displaced vertices. Source positions deliberately
    // differ so this regression catches construction before displacement.
    uploads[0u].positions = {
        make_float3(0.0f, 0.0f, 3.0f),
        make_float3(2.0f, 0.0f, 3.0f),
        make_float3(0.0f, 2.0f, 3.0f)};
    const std::array lights{
        LightGpu{
            .type = static_cast<std::uint32_t>(LightType::point),
            .position = make_float3(20.0f, 0.0f, 2.0f),
            .color = make_float3(1.0f),
            .power = 1.0f,
            .flags = light_flag_normalize}};
    constexpr std::array light_emission_estimates{
        Vec3f{1.0f, 1.0f, 1.0f}};
    const std::array triangles{
        EmissiveTriangleGpu{
            .instance_index = 0u,
            .geometry_index = 0u,
            .primitive_index = 0u,
            .emission_sampling =
                static_cast<std::uint32_t>(EmissionSampling::front),
            .cycles_primitive_index = 17u,
            .cycles_object_index = 9u}};
    constexpr std::array triangle_areas{2.0f};
    const auto result = build_light_sampling_scene_upload(
        snapshot,
        scene,
        uploads,
        lights,
        light_emission_estimates,
        triangles,
        triangle_areas,
        false);
    if (!result.ok() || !result.tree_usable() ||
        result.distribution_count != 2u ||
        result.tree_emitters.size() != 3u ||
        result.tree_emitter_mappings.size() != 2u ||
        result.tree_triangle_emitter_mappings.size() != 1u ||
        result.tree_mesh_triangles !=
            luisa::vector<luisa::uint>{0u} ||
        result.tree_triangle_lookup.size() != 1u ||
        !close(result.distribution[0u].selection_pdf, 0.5f, 1.0e-6f) ||
        !close(result.distribution[1u].selection_pdf, 0.5f, 1.0e-6f)) {
        std::cerr << "light-tree scene upload contract failed: "
                  << result.diagnostic << '\n';
        return false;
    }
    const auto kind = [](const LightTreeEmitterGpu &emitter) noexcept {
        return static_cast<LightTreeEmitterKind>(
            (emitter.identity.y & light_tree_emitter_kind_mask) >>
            light_tree_emitter_kind_shift);
    };
    const auto source = [](const LightTreeEmitterGpu &emitter) noexcept {
        return static_cast<LightTreeEmitterSource>(
            (emitter.identity.y & light_tree_emitter_source_mask) >>
            light_tree_emitter_source_shift);
    };
    const auto triangle = std::find_if(
        result.tree_emitters.begin(),
        result.tree_emitters.end(),
        [kind](const LightTreeEmitterGpu &emitter) noexcept {
            return kind(emitter) == LightTreeEmitterKind::mesh_triangle;
        });
    const auto proxy = std::find_if(
        result.tree_emitters.begin(),
        result.tree_emitters.end(),
        [kind](const LightTreeEmitterGpu &emitter) noexcept {
            return kind(emitter) == LightTreeEmitterKind::mesh_instance;
        });
    const auto direct = std::find_if(
        result.tree_emitters.begin(),
        result.tree_emitters.end(),
        [kind](const LightTreeEmitterGpu &emitter) noexcept {
            return kind(emitter) == LightTreeEmitterKind::direct;
        });
    const auto &lookup = result.tree_triangle_lookup.front();
    const auto expected_energy = 2.0f * (1.0f + 2.0f + 4.0f) / 3.0f;
    if (triangle == result.tree_emitters.end() ||
        proxy == result.tree_emitters.end() ||
        direct == result.tree_emitters.end() ||
        source(*triangle) != LightTreeEmitterSource::triangle ||
        triangle->identity.z != 0u ||
        source(*proxy) != LightTreeEmitterSource::measure ||
        source(*direct) != LightTreeEmitterSource::analytic_light ||
        direct->identity.z != 0u ||
        !close(triangle->bounds_min_energy.x, 0.0f, 1.0e-6f) ||
        !close(triangle->bounds_min_energy.z, 3.0f, 1.0e-6f) ||
        !close(triangle->bounds_max_theta_o.x, 2.0f, 1.0e-6f) ||
        !close(triangle->bounds_max_theta_o.z, 3.0f, 1.0e-6f) ||
        !close(
            triangle->bounds_min_energy.w, expected_energy, 1.0e-5f) ||
        !close(proxy->bounds_min_energy.x, 10.0f, 1.0e-6f) ||
        !close(proxy->bounds_max_theta_o.x, 12.0f, 1.0e-6f) ||
        !close(proxy->bounds_min_energy.w, expected_energy, 1.0e-5f) ||
        proxy->identity.x != 0u ||
        proxy->identity.z >= result.tree_nodes.size() ||
        proxy->identity.w != 0u ||
        result.tree_emitter_mappings[0u].x !=
            static_cast<std::uint32_t>(proxy - result.tree_emitters.begin()) ||
        result.tree_triangle_emitter_mappings[0u].x !=
            static_cast<std::uint32_t>(triangle - result.tree_emitters.begin()) ||
        lookup.x != 9u || lookup.y != 17u || lookup.z != 0u) {
        std::cerr << "final-support/material-override light-tree regression failed\n";
        return false;
    }
    return true;
}

[[nodiscard]] bool verify_analytic_light_population() {
    constexpr MaterialId dark_shader{1u};
    SceneSnapshot snapshot;
    snapshot.materials.emplace(
        dark_shader,
        MaterialDesc{
            .name = "proven dark light shader",
            .shader = emission_graph({1.0f, 1.0f, 1.0f}, 0.0f)});

    LightDesc regular{
        .name = "regular",
        .type = LightType::point,
        .cycles_object_index = 11u};
    LightDesc portal{
        .name = "portal",
        .type = LightType::area,
        .size = 2.0f,
        .size_y = 3.0f,
        .is_portal = true,
        .cycles_object_index = 12u};
    LightDesc zero_strength{
        .name = "zero strength",
        .type = LightType::point,
        .power = 0.0f};
    LightDesc degenerate_area{
        .name = "degenerate area",
        .type = LightType::area,
        .size = 0.0f};
    LightDesc zero_shader{
        .name = "zero shader",
        .type = LightType::point,
        .shader = dark_shader};
    LightDesc background{
        .name = "background",
        .type = LightType::background,
        .color = {0.25f, 0.5f, 1.0f},
        .power = 2.0f};
    snapshot.lights.emplace(LightId{1u}, regular);
    snapshot.lights.emplace(LightId{2u}, portal);
    snapshot.lights.emplace(LightId{3u}, zero_strength);
    snapshot.lights.emplace(LightId{4u}, degenerate_area);
    snapshot.lights.emplace(LightId{5u}, zero_shader);
    snapshot.lights.emplace(LightId{6u}, background);

    LuisaSceneData scene;
    ShaderCompiler compiler{make_core_node_registry()};
    const auto material_update = scene.materials.update(snapshot, compiler);
    if (!material_update.committed) {
        std::cerr << "analytic-light material compilation failed\n";
        return false;
    }
    const auto upload =
        AnalyticLightSceneComponent{}.build(snapshot, scene);
    if (!upload.ok() || upload.regular_count != 1u ||
        upload.portal_count != 1u || upload.device_lights.size() != 2u ||
        upload.regular_shader_emission_estimates.size() != 1u ||
        upload.device_lights[0u].cycles_object_index != 11u ||
        upload.device_lights[1u].cycles_object_index != 12u ||
        upload.background != Vec3f{0.5f, 1.0f, 2.0f}) {
        std::cerr << "analytic-light L/P/B partition failed: "
                  << upload.diagnostic << '\n';
        return false;
    }
    if (classify_analytic_light(portal, Vec3f{}) !=
            AnalyticLightRole::portal ||
        classify_analytic_light(zero_strength, Vec3f{1.0f}) !=
            AnalyticLightRole::disabled ||
        classify_analytic_light(degenerate_area, Vec3f{1.0f}) !=
            AnalyticLightRole::disabled ||
        classify_analytic_light(regular, Vec3f{}) !=
            AnalyticLightRole::disabled ||
        classify_analytic_light(background, Vec3f{}) !=
            AnalyticLightRole::background) {
        std::cerr << "analytic-light contribution classification failed\n";
        return false;
    }

    const auto sampling = build_light_sampling_scene_upload(
        snapshot,
        scene,
        std::span<const GeometryUpload>{},
        upload.regular_lights(),
        upload.regular_shader_emission_estimates,
        std::span<const EmissiveTriangleGpu>{},
        std::span<const float>{},
        false);
    if (!sampling.ok() || sampling.distribution_count != 1u ||
        sampling.tree_emitters.size() != 1u ||
        sampling.distribution.front().index != 0u) {
        std::cerr << "portal entered the direct-light population: "
                  << sampling.diagnostic << '\n';
        return false;
    }
    return true;
}

[[nodiscard]] bool verify_mesh_scene_quotient() {
    constexpr MaterialId base_material{1u};
    constexpr MaterialId override_material{2u};
    constexpr GeometryId geometry_id{3u};
    SceneSnapshot snapshot;
    snapshot.materials.emplace(
        base_material,
        MaterialDesc{
            .name = "shared mesh emission",
            .shader = emission_graph({1.0f, 1.0f, 1.0f}, 1.0f),
            .emission_sampling = EmissionSampling::front});
    snapshot.materials.emplace(
        override_material,
        MaterialDesc{
            .name = "non-shared override emission",
            .shader = emission_graph({2.0f, 4.0f, 6.0f}, 1.0f),
            .emission_sampling = EmissionSampling::front});
    TriangleMeshDesc mesh;
    mesh.positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}};
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.material_slots = {base_material};
    mesh.triangle_material_slots = {0u};
    snapshot.geometries.emplace(geometry_id, std::move(mesh));

    Mat4f identity;
    Mat4f uniform;
    uniform.elements[0u] = 2.0f;
    uniform.elements[5u] = 2.0f;
    uniform.elements[10u] = 2.0f;
    uniform.elements[12u] = 3.0f;
    Mat4f nonuniform;
    nonuniform.elements[0u] = 2.0f;
    nonuniform.elements[12u] = 6.0f;
    snapshot.instances.emplace(
        InstanceId{1u},
        InstanceDesc{.geometry = geometry_id, .transform = identity});
    snapshot.instances.emplace(
        InstanceId{2u},
        InstanceDesc{.geometry = geometry_id, .transform = uniform});
    snapshot.instances.emplace(
        InstanceId{3u},
        InstanceDesc{.geometry = geometry_id,
                     .transform = nonuniform,
                     .material_overrides = {override_material}});

    LuisaSceneData scene;
    ShaderCompiler compiler{make_core_node_registry()};
    if (!scene.materials.update(snapshot, compiler).committed) {
        std::cerr << "mesh-light quotient material compilation failed\n";
        return false;
    }
    std::array<GeometryUpload, 1u> uploads;
    uploads[0u].positions = {
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    std::array<EmissiveTriangleGpu, 3u> triangles;
    for (std::uint32_t instance = 0u; instance < triangles.size(); ++instance) {
        triangles[instance] = {
            .instance_index = instance,
            .geometry_index = 0u,
            .primitive_index = 0u,
            .emission_sampling =
                static_cast<std::uint32_t>(EmissionSampling::front)};
    }
    const auto quotient = MeshLightTreeSceneComponent{}.build(
        snapshot, scene, uploads, triangles);
    if (!quotient.ok() || quotient.subtrees.size() != 2u ||
        quotient.mesh_emitters.size() != 3u ||
        quotient.mesh_emitters[0u].subtree !=
            quotient.mesh_emitters[1u].subtree ||
        quotient.mesh_emitters[2u].subtree ==
            quotient.mesh_emitters[0u].subtree ||
        quotient.mesh_emitters[0u].triangle_emitters !=
            std::vector<std::uint32_t>{0u} ||
        quotient.mesh_emitters[1u].triangle_emitters !=
            std::vector<std::uint32_t>{1u} ||
        quotient.mesh_emitters[2u].triangle_emitters !=
            std::vector<std::uint32_t>{2u} ||
        !close(quotient.mesh_emitters[0u].emitter.measure.energy,
               0.5f, 1.0e-6f) ||
        !close(quotient.mesh_emitters[1u].emitter.measure.energy,
               2.0f, 1.0e-6f) ||
        !close(quotient.mesh_emitters[2u].emitter.measure.energy,
               4.0f, 1.0e-6f)) {
        std::cerr << "mesh-light semantic quotient failed: "
                  << quotient.diagnostic << '\n';
        return false;
    }
    return true;
}

[[nodiscard]] LightTreeEmitter local_emitter(
    std::uint32_t id,
    float x,
    float energy) {
    return {
        .measure = {
            .bounds = {
                .minimum = {x - 0.2f, -0.2f, 1.8f},
                .maximum = {x + 0.2f, 0.2f, 2.2f},
                .empty = false},
            .orientation = {
                .axis = {0.0f, 0.0f, -1.0f},
                .theta_o = 3.14159265358979323846f,
                .theta_e = 1.57079632679489661923f,
                .empty = false},
            .energy = energy},
        .centroid = {x, 0.0f, 2.0f},
        .emitter_id = id,
        .distant = false};
}

[[nodiscard]] bool close(float a, float b, float tolerance) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= tolerance;
}

}// namespace

int main(int argc, char **argv) {
    if (!verify_scene_upload() || !verify_analytic_light_population() ||
        !verify_mesh_scene_quotient()) {
        return 1;
    }
    const std::string backend = argc > 1 ? argv[1] : "fallback";
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream(StreamTag::COMPUTE);

    std::array<LightTreeEmitter, emitter_count> emitters{
        local_emitter(0u, -1.5f, 1.0f),
        local_emitter(1u, 0.0f, 1.0f),
        local_emitter(2u, 1.5f, 1.0f),
        LightTreeEmitter{
            .measure = {
                .orientation = {
                    .axis = {0.0f, 0.0f, 1.0f},
                    .theta_o = 0.0f,
                    .theta_e = 0.1f,
                    .empty = false},
                .energy = 0.75f},
            .centroid = {0.0f, 0.0f, 1.0f},
            .emitter_id = 3u,
            .distant = true}};
    LightTreeHierarchyInput hierarchy;
    hierarchy.distribution_emitter_count = emitter_count;
    hierarchy.triangle_emitter_count = 3u;
    hierarchy.subtrees.emplace_back(LightTreeSubtreeInput{
        .emitters = {local_emitter(0u, 0.0f, 1.0f)}});
    for (std::uint32_t instance = 0u; instance < 3u; ++instance) {
        hierarchy.top_emitters.emplace_back(LightTreeTopEmitterInput{
            .emitter = emitters[instance],
            .kind = LightTreeEmitterKind::mesh_instance,
            .payload = instance,
            .subtree = 0u,
            .triangle_emitters = {instance}});
    }
    hierarchy.top_emitters.emplace_back(LightTreeTopEmitterInput{
        .emitter = emitters[3u],
        .kind = LightTreeEmitterKind::direct,
        .payload = 3u});
    auto upload = make_light_tree_hierarchy_scene_upload(hierarchy);
    if (!upload.usable()) {
        std::cerr << "light-tree upload failed: " << upload.diagnostic << '\n';
        return 1;
    }
    const auto same_mapping = [](luisa::uint2 a, luisa::uint2 b) noexcept {
        return a.x == b.x && a.y == b.y;
    };
    if (upload.emitters.size() != 5u ||
        upload.emitter_mappings.size() != emitter_count ||
        upload.triangle_emitter_mappings.size() != 3u ||
        upload.mesh_triangles != luisa::vector<luisa::uint>{0u, 1u, 2u} ||
        !same_mapping(upload.triangle_emitter_mappings[0u],
                      upload.triangle_emitter_mappings[1u]) ||
        !same_mapping(upload.triangle_emitter_mappings[1u],
                      upload.triangle_emitter_mappings[2u]) ||
        same_mapping(upload.emitter_mappings[0u],
                     upload.emitter_mappings[1u]) ||
        same_mapping(upload.emitter_mappings[1u],
                     upload.emitter_mappings[2u])) {
        std::cerr << "mesh-instance light-tree quotient contract failed\n";
        return 1;
    }

    std::array<LightDistributionGpu, emitter_count> distribution{};
    for (std::uint32_t id = 0u; id < emitter_count; ++id) {
        distribution[id].kind = static_cast<std::uint32_t>(
            id < 3u ? LightDistributionEmitterKind::emissive_triangle
                    : LightDistributionEmitterKind::analytic_light);
        distribution[id].index = id;
        distribution[id].emitter_id = id;
    }

    auto scene = std::make_shared<LuisaSceneData>();
    scene->device = Device{device.impl_shared()};
    scene->light_tree_node_count =
        static_cast<std::uint32_t>(upload.nodes.size());
    scene->light_tree_emitter_count =
        static_cast<std::uint32_t>(upload.emitters.size());
    scene->light_tree_root = upload.root;
    scene->light_distribution_count = emitter_count;
    scene->emissive_triangle_count = 3u;
    scene->light_tree_mesh_triangle_count =
        static_cast<std::uint32_t>(upload.mesh_triangles.size());
    scene->light_tree_node_buffer =
        device.create_buffer<LightTreeNodeGpu>(upload.nodes.size());
    scene->light_tree_emitter_buffer =
        device.create_buffer<LightTreeEmitterGpu>(upload.emitters.size());
    scene->light_tree_emitter_mapping_buffer =
        device.create_buffer<uint2>(upload.emitter_mappings.size());
    scene->light_tree_triangle_emitter_mapping_buffer =
        device.create_buffer<uint2>(upload.triangle_emitter_mappings.size());
    scene->light_tree_mesh_triangle_buffer =
        device.create_buffer<uint>(upload.mesh_triangles.size());
    scene->light_distribution_buffer =
        device.create_buffer<LightDistributionGpu>(distribution.size());

    std::array<Mat4f, 3u> transforms;
    transforms[0u].elements[12u] = -1.5f;
    transforms[2u].elements[12u] = 1.5f;
    std::array<InstanceGpu, 3u> instances;
    for (std::size_t index = 0u; index < instances.size(); ++index) {
        instances[index].cycles_world_to_object =
            to_luisa(cycles_inverse_transform(transforms[index]));
    }
    scene->instance_buffer =
        device.create_buffer<InstanceGpu>(instances.size());
    constexpr std::array accel_vertices{
        luisa::float3{-1.0f, -1.0f, 0.0f},
        luisa::float3{1.0f, -1.0f, 0.0f},
        luisa::float3{0.0f, 1.0f, 0.0f}};
    constexpr std::array accel_triangles{Triangle{0u, 1u, 2u}};
    auto accel_vertex_buffer =
        device.create_buffer<luisa::float3>(accel_vertices.size());
    auto accel_triangle_buffer =
        device.create_buffer<Triangle>(accel_triangles.size());
    auto accel_mesh =
        device.create_mesh(accel_vertex_buffer, accel_triangle_buffer);
    scene->accel = device.create_accel();
    for (std::size_t index = 0u; index < transforms.size(); ++index) {
        scene->accel.emplace_back(
            accel_mesh,
            to_luisa(transforms[index]),
            0xffu,
            false,
            static_cast<std::uint32_t>(index));
    }
    stream << scene->light_tree_node_buffer.copy_from(span{upload.nodes})
           << scene->light_tree_emitter_buffer.copy_from(span{upload.emitters})
           << scene->light_tree_emitter_mapping_buffer.copy_from(
                  span{upload.emitter_mappings})
           << scene->light_tree_triangle_emitter_mapping_buffer.copy_from(
                  span{upload.triangle_emitter_mappings})
           << scene->light_tree_mesh_triangle_buffer.copy_from(
                  span{upload.mesh_triangles})
           << scene->light_distribution_buffer.copy_from(span{distribution});
    stream << scene->instance_buffer.copy_from(span{instances})
           << accel_vertex_buffer.copy_from(accel_vertices.data())
           << accel_triangle_buffer.copy_from(accel_triangles.data())
           << accel_mesh.build()
           << scene->accel.build();

    auto callables = make_light_tree_callables(scene);
    auto samples = device.create_buffer<float4>(sample_count * 2u);
    auto probabilities = device.create_buffer<float>(emitter_count + 5u);
    Kernel1D sample_kernel = [=](BufferFloat4 output) noexcept {
        const auto index = dispatch_x();
        const auto random =
            (cast<float>(index % sample_count) + 0.5f) /
            static_cast<float>(sample_count);
        const auto volume = index >= sample_count;
        Var<LightDistributionGpu> selected;
        $if (volume) {
            selected = callables.volume_sample(
                random,
                make_float3(0.0f, 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                3.0f,
                false);
        }
        $else {
            selected = callables.surface_sample(
                random,
                make_float3(0.0f, 0.0f, 0.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                0.0f,
                false);
        };
        Float reverse_pdf = 0.0f;
        $if (selected.emitter_id < emitter_count) {
            reverse_pdf = select(
                callables.surface_pdf(
                    selected.emitter_id,
                    make_float3(0.0f, 0.0f, 0.0f),
                    make_float3(0.0f, 0.0f, 1.0f),
                    0.0f,
                    false),
                callables.volume_pdf(
                    selected.emitter_id,
                    make_float3(0.0f, 0.0f, 0.0f),
                    make_float3(0.0f, 0.0f, 1.0f),
                    3.0f,
                    false),
                volume);
        };
        output.write(
            index,
            make_float4(
                cast<float>(selected.emitter_id),
                selected.selection_pdf,
                reverse_pdf,
                cast<float>(selected.kind)));
    };
    Kernel1D probability_kernel = [=](BufferFloat output) noexcept {
        const auto index = dispatch_x();
        $if (index < emitter_count) {
            output.write(
                index,
                callables.surface_pdf(
                    index,
                    make_float3(0.0f, 0.0f, 0.0f),
                    make_float3(0.0f, 0.0f, 1.0f),
                    0.0f,
                    false));
        }
        $elif (index == emitter_count) {
            const auto behind = callables.surface_sample(
                0.25f,
                make_float3(0.0f, 0.0f, 3.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                0.0f,
                false);
            output.write(index, cast<float>(
                behind.emitter_id == invalid_light_tree_index));
        }
        $elif (index == emitter_count + 1u) {
            const auto transmitted = callables.surface_sample(
                0.25f,
                make_float3(0.0f, 0.0f, 3.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                0.0f,
                true);
            output.write(index, cast<float>(
                transmitted.emitter_id < emitter_count));
        }
        $elif (index == emitter_count + 2u) {
            const auto expected = callables.surface_pdf(
                2u,
                make_float3(0.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                0.0f,
                false);
            const auto actual = callables.forward_pdf(
                2u,
                make_float3(0.0f),
                make_float3(0.0f, 0.0f, 1.0f),
                9.0f,
                0u,
                0u);
            output.write(index, abs(actual - expected));
        }
        $elif (index == emitter_count + 3u) {
            const auto origin = make_float3(0.0f, 0.0f, 0.25f);
            const Float3 traveled = make_float3(0.0f, 0.0f, 0.75f);
            const auto segment_length = 3.0f;
            const auto expected = callables.volume_pdf(
                2u,
                origin,
                make_float3(0.0f, 0.0f, 1.0f),
                segment_length,
                false);
            const auto actual = callables.forward_pdf(
                2u,
                origin + traveled,
                traveled,
                segment_length,
                psycles::luisa_backend::cycles_path_state::
                    visibility_volume_scatter,
                0u);
            output.write(index, actual);
        }
        $elif (index == emitter_count + 4u) {
            output.write(
                index,
                callables.volume_pdf(
                    2u,
                    make_float3(0.0f, 0.0f, 0.25f),
                    make_float3(0.0f, 0.0f, 1.0f),
                    3.0f,
                    false));
        };
    };
    const ShaderOption test_shader_options{
        .enable_cache = false,
        .enable_fast_math = true};
    auto sample_shader = device.compile(
        sample_kernel, test_shader_options);
    auto probability_shader = device.compile(
        probability_kernel, test_shader_options);
    std::vector<float4> sample_results(sample_count * 2u);
    std::array<float, emitter_count + 5u> pdf_results{};
    stream << sample_shader(samples).dispatch(sample_count * 2u)
           << probability_shader(probabilities).dispatch(emitter_count + 5u)
           << samples.copy_to(sample_results.data())
           << probabilities.copy_to(pdf_results.data())
           << synchronize();

    std::array<std::uint32_t, emitter_count> frequency{};
    for (std::size_t i = 0u; i < sample_results.size(); ++i) {
        const auto &sample = sample_results[i];
        const auto id = static_cast<std::uint32_t>(sample.x);
        if (id >= emitter_count || !close(sample.y, sample.z, 2.0e-5f) ||
            !(sample.y > 0.0f)) {
            std::cerr << "sample/PDF reciprocity failed on " << backend
                      << " at " << i << ": id=" << id
                      << " sample=" << sample.y << " reverse=" << sample.z
                      << '\n';
            return 1;
        }
        if (i < sample_count) {
            ++frequency[id];
        }
    }

    float pdf_sum = 0.0f;
    for (std::uint32_t id = 0u; id < emitter_count; ++id) {
        pdf_sum += pdf_results[id];
        const auto observed = static_cast<float>(frequency[id]) /
                              static_cast<float>(sample_count);
        if (!close(observed, pdf_results[id], 0.015f)) {
            std::cerr << "sampling frequency mismatch on " << backend
                      << " emitter " << id << ": observed=" << observed
                      << " expected=" << pdf_results[id] << '\n';
            return 1;
        }
    }
    if (!close(pdf_sum, 1.0f, 2.0e-5f) ||
        !close(pdf_results[emitter_count], 1.0f, 0.0f) ||
        !close(pdf_results[emitter_count + 1u], 1.0f, 0.0f) ||
        !close(pdf_results[emitter_count + 2u], 0.0f, 2.0e-6f) ||
        !close(
            pdf_results[emitter_count + 3u],
            pdf_results[emitter_count + 4u],
            2.0e-6f)) {
        std::cerr << "normalization/visibility contract failed on " << backend
                  << ": sum=" << pdf_sum
                  << " behind=" << pdf_results[emitter_count]
                  << " transmitted=" << pdf_results[emitter_count + 1u]
                  << " forward_surface="
                  << pdf_results[emitter_count + 2u]
                  << " forward_volume_actual="
                  << pdf_results[emitter_count + 3u]
                  << " forward_volume_expected="
                  << pdf_results[emitter_count + 4u]
                  << '\n';
        return 1;
    }
    return 0;
}
