#include "cycles_shader_identity.h"
#include "path_kernel_heterogeneous_volume.h"
#include "path_kernel_volume_majorant_provider.h"
#include "path_tracer_generated_coordinates.h"
#include "path_tracer_volume_capabilities.h"
#include "path_tracer_volume_majorant_scene.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;

void expect(
    bool condition,
    const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] bool close(
    float actual,
    float expected,
    float tolerance = 3.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

[[nodiscard]] Mat4f transform(
    Vec3f translation,
    Vec3f scale = {1.0f, 1.0f, 1.0f})
    noexcept {
    Mat4f result;
    result.elements = {
        scale.x, 0.0f, 0.0f, 0.0f,
        0.0f, scale.y, 0.0f, 0.0f,
        0.0f, 0.0f, scale.z, 0.0f,
        translation.x,
        translation.y,
        translation.z,
        1.0f};
    return result;
}

[[nodiscard]] TriangleMeshDesc bounds_geometry(
    std::string name,
    Vec3f minimum,
    Vec3f maximum,
    std::vector<MaterialId> materials) {
    return {
        .name = std::move(name),
        .positions = {minimum, maximum},
        .material_slots = std::move(materials)};
}

void add_instance(
    SceneSnapshot &scene,
    std::uint64_t id,
    GeometryId geometry,
    Mat4f matrix,
    std::vector<MaterialId> overrides = {},
    std::optional<std::uint32_t> cycles_object =
        std::nullopt) {
    scene.instances.emplace(
        InstanceId{id},
        InstanceDesc{
            .name =
                "majorant-instance-" +
                std::to_string(id),
            .geometry = geometry,
            .transform = std::move(matrix),
            .material_overrides =
                std::move(overrides),
            .cycles_object_index =
                cycles_object});
}

void test_scene_plan() {
    constexpr MaterialId heterogeneous_a{1u};
    constexpr MaterialId heterogeneous_b{2u};
    constexpr MaterialId homogeneous{3u};
    constexpr MaterialId surface_only{4u};
    constexpr MaterialId world{5u};
    constexpr GeometryId ordinary_geometry{10u};
    constexpr GeometryId partial_geometry{11u};
    constexpr GeometryId collapsed_geometry{12u};

    const std::map<MaterialId, VolumeMajorantSceneMaterial>
        materials{
            {heterogeneous_a,
             {.surface_tag = 101u,
              .parameter_block = 201u,
              .shader =
                  17u |
                  cycles_shader_identity::
                      smooth_normal |
                  cycles_shader_identity::
                      cast_shadow,
              .has_volume = true,
              .heterogeneous = true}},
            {heterogeneous_b,
             {.surface_tag = 102u,
              .parameter_block = 202u,
              .shader = 18u,
              .has_volume = true,
              .heterogeneous = true}},
            {homogeneous,
             {.surface_tag = 103u,
              .parameter_block = 203u,
              .shader = 19u,
              .has_volume = true,
              .heterogeneous = false}},
            {surface_only,
             {.surface_tag = 104u,
              .parameter_block = 204u,
              .shader = 20u,
              .has_volume = false,
              .heterogeneous = false}},
            {world,
             {.surface_tag = 105u,
              .parameter_block = 205u,
              .shader = 21u,
              .has_volume = true,
              .heterogeneous = true}}};

    SceneSnapshot scene;
    scene.geometries.emplace(
        ordinary_geometry,
        bounds_geometry(
            "ordinary bounds",
            {-2.0f, -1.0f, 0.0f},
            {4.0f, 5.0f, 6.0f},
            {heterogeneous_a,
             homogeneous,
             heterogeneous_b}));
    scene.geometries.emplace(
        partial_geometry,
        bounds_geometry(
            "partially degenerate bounds",
            {2.0f, -3.0f, -4.0f},
            {2.0f, 7.0f, 8.0f},
            {heterogeneous_a}));
    scene.geometries.emplace(
        collapsed_geometry,
        bounds_geometry(
            "fully collapsed bounds",
            {3.0f, 3.0f, 3.0f},
            {3.0f, 3.0f, 3.0f},
            {heterogeneous_a}));

    Mat4f sheared;
    sheared.elements = {
        2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 3.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 4.0f, 0.0f,
        7.0f, -8.0f, 9.0f, 1.0f};
    add_instance(
        scene,
        10u,
        ordinary_geometry,
        sheared,
        {},
        77u);
    add_instance(
        scene,
        20u,
        ordinary_geometry,
        transform({1.0f, 2.0f, 3.0f}),
        {surface_only, heterogeneous_b},
        88u);
    add_instance(
        scene,
        30u,
        partial_geometry,
        transform({4.0f, 5.0f, 6.0f}));
    add_instance(
        scene,
        40u,
        collapsed_geometry,
        transform({7.0f, 8.0f, 9.0f}));
    scene.world_shader = world;
    scene.cycles_background_object_index = 99u;

    const VolumeMajorantSceneComponent component;
    const auto plan =
        component.plan(scene, materials);
    expect(plan.ok(), plan.diagnostic);
    expect(
        plan.ranges.size() == 5u &&
            plan.world_range == 4u,
        "scene plan did not reserve one range per "
        "instance plus World");
    constexpr std::array expected_ranges{
        VolumeMajorantRootRangeGpu{0u, 3u},
        VolumeMajorantRootRangeGpu{3u, 1u},
        VolumeMajorantRootRangeGpu{4u, 1u},
        VolumeMajorantRootRangeGpu{5u, 0u},
        VolumeMajorantRootRangeGpu{5u, 1u}};
    for (auto index = std::size_t{0u};
         index < expected_ranges.size();
         ++index) {
        expect(
            plan.ranges[index].offset ==
                    expected_ranges[index].offset &&
                plan.ranges[index].count ==
                    expected_ranges[index].count,
            "scene plan changed the ordered root "
            "partition at range " +
                std::to_string(index));
    }
    expect(
        plan.roots.size() == 6u,
        "scene plan did not select every effective "
        "volume material");

    const auto &first = plan.roots[0u];
    expect(
        first.material == heterogeneous_a &&
            first.range_index == 0u &&
            first.object == 77u &&
            first.instance_id == 0u &&
            first.shader == 17u &&
            first.surface_tag == 101u &&
            first.parameter_block == 201u &&
            first.heterogeneous,
        "first object root lost Cycles object/shader "
        "identity");
    expect(
        first.bounds.minimum.x == -2.0f &&
            first.bounds.minimum.y == -1.0f &&
            first.bounds.minimum.z == 0.0f &&
            first.bounds.maximum.x == 4.0f &&
            first.bounds.maximum.y == 5.0f &&
            first.bounds.maximum.z == 6.0f,
        "object root did not retain mesh-space bounds");
    expect(
        close(
            first.volume_scale,
            std::sqrt(50.0f / 3.0f)),
        "object root did not use Cycles' diagonal "
        "transform-direction scale");
    expect(
        close(first.object_to_world[3u].x, 7.0f) &&
            close(
                first.object_to_world[3u].y,
                -8.0f) &&
            close(first.object_to_world[3u].z, 9.0f),
        "object root transform was not retained");

    expect(
        plan.roots[1u].material ==
                heterogeneous_b &&
            plan.roots[1u].heterogeneous &&
            plan.roots[2u].material ==
                homogeneous &&
            !plan.roots[2u].heterogeneous,
        "ordinary geometry lost its second "
        "heterogeneous shader or homogeneous root");
    expect(
        plan.roots[3u].material ==
                heterogeneous_b &&
            plan.roots[3u].range_index == 1u &&
            plan.roots[3u].object == 88u,
        "instance override did not replace the "
        "geometry material slots");
    expect(
        plan.roots[4u].range_index == 2u &&
            plan.roots[4u].bounds.minimum.x ==
                plan.roots[4u].bounds.maximum.x,
        "partially degenerate Cycles bounds were "
        "discarded");

    const auto &world_root = plan.roots[5u];
    expect(
        world_root.material == world &&
            world_root.range_index == 4u &&
            world_root.object == 99u &&
            world_root.instance_id ==
                invalid_volume_identity &&
            world_root.bounds.minimum.x ==
                -10000.0f &&
            world_root.bounds.maximum.z ==
                10000.0f,
        "World root did not match Cycles' finite "
        "majorant domain");

    SceneSnapshot invalid_transform;
    invalid_transform.geometries.emplace(
        ordinary_geometry,
        bounds_geometry(
            "invalid transform bounds",
            {-1.0f, -1.0f, -1.0f},
            {1.0f, 1.0f, 1.0f},
            {heterogeneous_a}));
    auto nonfinite = Mat4f{};
    nonfinite.elements[12u] =
        std::numeric_limits<float>::infinity();
    add_instance(
        invalid_transform,
        1u,
        ordinary_geometry,
        nonfinite);
    expect(
        !component
             .plan(invalid_transform, materials)
             .ok(),
        "non-finite scene transform entered the "
        "majorant prepass");
}

[[nodiscard]] VolumeMajorantHierarchy leaf_hierarchy(
    float minimum,
    float maximum) {
    VolumeMajorantHierarchy result;
    result.root = {
        .scale = {1.0f, 2.0f, 3.0f},
        .node = 0u,
        .translation = {4.0f, 5.0f, 6.0f},
        .shader = 999u};
    result.nodes.emplace_back(
        VolumeMajorantNodeGpu{
            .parent = -1,
            .first_child = -1,
            .sigma_minimum = minimum,
            .sigma_maximum = maximum});
    return result;
}

[[nodiscard]] VolumeMajorantHierarchy
one_level_hierarchy() {
    auto result = leaf_hierarchy(0.1f, 2.0f);
    result.nodes[0u].first_child = 1;
    result.nodes.resize(9u);
    for (auto child = std::size_t{1u};
         child < result.nodes.size();
         ++child) {
        result.nodes[child] = {
            .parent = 0,
            .first_child = -1,
            .sigma_minimum =
                0.1f *
                static_cast<float>(child),
            .sigma_maximum =
                0.1f *
                    static_cast<float>(child) +
                0.25f};
    }
    return result;
}

[[nodiscard]] VolumeMajorantScenePlan
flatten_plan() {
    VolumeMajorantScenePlan plan;
    plan.world_range = 1u;
    plan.ranges = {
        {0u, 1u},
        {1u, 1u}};
    plan.roots = {
        {.material = MaterialId{31u},
         .range_index = 0u,
         .shader = 41u},
        {.material = MaterialId{32u},
         .range_index = 1u,
         .shader = 42u}};
    return plan;
}

void test_flatten_contract() {
    const VolumeMajorantSceneComponent component;
    const auto plan = flatten_plan();
    const std::array hierarchies{
        leaf_hierarchy(0.25f, 0.5f),
        one_level_hierarchy()};
    const auto flattened =
        component.flatten(plan, hierarchies);
    expect(flattened.ok(), flattened.diagnostic);
    expect(
        flattened.nodes.size() == 10u &&
            flattened.roots.size() == 2u &&
            flattened.ranges.size() == 2u,
        "flattened scene resource counts changed");
    expect(
        flattened.roots[0u].node == 0u &&
            flattened.roots[0u].shader == 41u &&
            flattened.roots[1u].node == 1u &&
            flattened.roots[1u].shader == 42u,
        "flattening did not relocate roots or replace "
        "their shader identities");
    expect(
        flattened.nodes[1u].parent == -1 &&
            flattened.nodes[1u].first_child == 2,
        "flattening did not relocate the second "
        "hierarchy root");
    for (auto child = std::size_t{2u};
         child < flattened.nodes.size();
         ++child) {
        expect(
            flattened.nodes[child].parent == 1,
            "flattening did not relocate a child "
            "parent index");
    }

    auto partition_gap = plan;
    partition_gap.ranges[1u].offset = 2u;
    expect(
        !component
             .flatten(
                 partition_gap, hierarchies)
             .ok(),
        "a gapped root partition was accepted");

    auto wrong_identity = plan;
    wrong_identity.roots[1u].range_index = 0u;
    expect(
        !component
             .flatten(
                 wrong_identity, hierarchies)
             .ok(),
        "a root assigned to the wrong range was "
        "accepted");

    auto incomplete = one_level_hierarchy();
    incomplete.nodes.pop_back();
    std::array malformed{
        hierarchies[0u], incomplete};
    expect(
        !component.flatten(plan, malformed).ok(),
        "an incomplete eight-child block was accepted");

    auto wrong_parent = one_level_hierarchy();
    wrong_parent.nodes[8u].parent = 7;
    malformed = {hierarchies[0u], wrong_parent};
    expect(
        !component.flatten(plan, malformed).ok(),
        "a child with the wrong parent was accepted");

    auto unreachable = hierarchies[0u];
    unreachable.nodes.emplace_back(
        VolumeMajorantNodeGpu{
            .parent = -1,
            .first_child = -1,
            .sigma_minimum = 0.0f,
            .sigma_maximum = 0.0f});
    malformed = {unreachable, hierarchies[1u]};
    expect(
        !component.flatten(plan, malformed).ok(),
        "an unreachable local node was accepted");

    auto invalid_extrema = hierarchies[0u];
    invalid_extrema.nodes[0u].sigma_minimum =
        -1.0f;
    malformed = {
        invalid_extrema, hierarchies[1u]};
    expect(
        !component.flatten(plan, malformed).ok(),
        "negative local majorant extrema were "
        "accepted");
}

enum class VolumeGraphKind {
    spatial,
    homogeneous_light_path,
    spatial_light_path,
    surface_light_path,
};

[[nodiscard]] ShaderGraph
make_volume_graph(VolumeGraphKind kind) {
    ShaderGraph graph;
    const auto surface =
        graph.add_node(
            node_type::diffuse_bsdf,
            "Zero-contribution surface");
    expect(
        graph.set_input(
            surface,
            "Color",
            SocketValue::color(
                {0.0f, 0.0f, 0.0f})),
        "failed to construct companion surface root");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = surface,
            .socket = "Closure"});
    if (kind ==
        VolumeGraphKind::surface_light_path) {
        const auto surface_path =
            graph.add_node(
                node_type::light_path,
                "Surface-only Light Path");
        expect(
            graph.connect(
                {.node = surface_path,
                 .socket = "RayDepth"},
                surface,
                "Roughness"),
            "failed to construct surface-only Light "
            "Path dependency");
    }

    if (kind ==
        VolumeGraphKind::homogeneous_light_path) {
        const auto absorption =
            graph.add_node(
                node_type::volume_absorption,
                "Light Path volume absorption");
        const auto path =
            graph.add_node(
                node_type::light_path,
                "Runtime majorant Light Path");
        expect(
            graph.set_input(
                absorption,
                "Color",
                SocketValue::color(
                    {0.0f, 0.0f, 0.0f})) &&
                graph.connect(
                    {.node = path,
                     .socket = "IsCameraRay"},
                    absorption,
                    "Density"),
            "failed to construct Light Path volume graph");
        graph.set_root(
            ShaderDomain::volume,
            OutputRef{
                .node = absorption,
                .socket = "Volume"});
        return graph;
    }

    if (kind ==
        VolumeGraphKind::spatial_light_path) {
        const auto absorption =
            graph.add_node(
                node_type::volume_absorption,
                "Spatial Light Path volume absorption");
        const auto coordinates =
            graph.add_node(
                node_type::texture_coordinate,
                "Spatial runtime coordinates");
        const auto point_to_vector =
            graph.add_node(
                node_type::point_to_vector,
                "Spatial runtime point to vector");
        const auto scalar =
            graph.add_node(
                node_type::vector_to_scalar,
                "Generated coordinate average");
        const auto path =
            graph.add_node(
                node_type::light_path,
                "Runtime majorant Light Path");
        const auto density =
            graph.add_node(
                node_type::multiply_float,
                "Spatial runtime density");
        expect(
            graph.set_input(
                absorption,
                "Color",
                SocketValue::color(
                    {0.0f, 0.0f, 0.0f})) &&
                graph.connect(
                    {.node = coordinates,
                     .socket = "Generated"},
                    point_to_vector,
                    "Point") &&
                graph.connect(
                    {.node = point_to_vector,
                     .socket = "Vector"},
                    scalar,
                    "Vector") &&
                graph.connect(
                    {.node = scalar,
                     .socket = "Value"},
                    density,
                    "A") &&
                graph.connect(
                    {.node = path,
                     .socket = "RayDepth"},
                    density,
                    "B") &&
                graph.connect(
                    {.node = density,
                     .socket = "Value"},
                    absorption,
                    "Density"),
            "failed to construct spatial Light Path "
            "volume graph");
        graph.set_root(
            ShaderDomain::volume,
            OutputRef{
                .node = absorption,
                .socket = "Volume"});
        return graph;
    }

    const auto coefficients =
        graph.add_node(
            node_type::volume_coefficients,
            "Raw volume coefficients");
    expect(
        graph.set_input(
            coefficients,
            "ScatterCoefficients",
            SocketValue::vector(
                {0.0f, 0.0f, 0.0f})) &&
            graph.set_input(
                coefficients,
                "AbsorptionCoefficients",
                SocketValue::vector(
                    {0.0f, 0.0f, 0.0f})),
        "failed to construct raw volume graph");
    if (kind == VolumeGraphKind::spatial) {
        const auto coordinates =
            graph.add_node(
                node_type::texture_coordinate,
                "Raw volume coordinates");
        const auto point_to_vector =
            graph.add_node(
                node_type::point_to_vector,
                "Raw volume point to vector");
        expect(
            graph.connect(
                {.node = coordinates,
                 .socket = "Generated"},
                point_to_vector,
                "Point") &&
                graph.connect(
                    {.node = point_to_vector,
                     .socket = "Vector"},
                    coefficients,
                    "EmissionCoefficients"),
            "failed to construct raw spatial volume graph");
    } else {
        expect(
            graph.set_input(
                coefficients,
                "EmissionCoefficients",
                SocketValue::vector(
                    {0.35f, 0.2f, 0.1f})),
            "failed to construct raw homogeneous "
            "volume graph");
    }
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{
            .node = coefficients,
            .socket = "Volume"});
    return graph;
}

struct TypedParameterData {
    std::vector<float> scalars;
    std::vector<luisa::float3> vectors;
};

[[nodiscard]] TypedParameterData
parameter_data(const SurfaceProgram &program) {
    TypedParameterData result;
    result.scalars.reserve(program.parameters().size());
    result.vectors.reserve(program.parameters().size());
    for (const auto &parameter :
         program.parameters()) {
        const auto &value =
            parameter.default_value;
        if (const auto *scalar =
                std::get_if<float>(
                    &value.value)) {
            result.scalars.emplace_back(*scalar);
            result.vectors.emplace_back(
                luisa::make_float3(0.0f));
        } else if (
            const auto *vector =
                std::get_if<Vec3f>(
                    &value.value)) {
            result.scalars.emplace_back(0.0f);
            result.vectors.emplace_back(
                vector->x,
                vector->y,
                vector->z);
        } else {
            throw std::runtime_error{
                "scene-majorant fixture has an "
                "unsupported parameter type"};
        }
    }
    if (result.scalars.empty()) {
        result.scalars.emplace_back(0.0f);
        result.vectors.emplace_back(
            luisa::make_float3(0.0f));
    }
    return result;
}

void run_scene_build(
    std::string_view backend,
    const char *program) {
    constexpr MaterialId spatial_material{61u};
    constexpr MaterialId homogeneous_material{62u};
    constexpr GeometryId geometry_id{71u};

    auto spatial_graph =
        make_volume_graph(
            VolumeGraphKind::spatial);
    auto homogeneous_graph =
        make_volume_graph(
            VolumeGraphKind::
                homogeneous_light_path);
    auto spatial_light_path_graph =
        make_volume_graph(
            VolumeGraphKind::spatial_light_path);
    auto surface_light_path_graph =
        make_volume_graph(
            VolumeGraphKind::surface_light_path);
    ShaderCompiler compiler{
        make_core_node_registry()};
    const auto spatial_shader_program =
        compiler.compile(spatial_graph);
    expect(
        spatial_shader_program.ok(),
        "failed to compile raw spatial volume graph");
    const auto spatial_lowered =
        compile_surface_program(
            *spatial_shader_program.program);
    expect(
        spatial_lowered.ok(),
        "failed to lower raw spatial volume graph");
    const auto homogeneous_shader_program =
        compiler.compile(homogeneous_graph);
    expect(
        homogeneous_shader_program.ok(),
        "failed to compile raw homogeneous volume graph");
    const auto homogeneous_lowered =
        compile_surface_program(
            *homogeneous_shader_program.program);
    expect(
        homogeneous_lowered.ok(),
        "failed to lower raw homogeneous volume graph");
    const auto spatial_light_path_shader_program =
        compiler.compile(
            spatial_light_path_graph);
    expect(
        spatial_light_path_shader_program.ok(),
        "failed to compile raw spatial Light Path "
        "volume graph");
    const auto spatial_light_path_lowered =
        compile_surface_program(
            *spatial_light_path_shader_program
                 .program);
    expect(
        spatial_light_path_lowered.ok(),
        "failed to lower raw spatial Light Path "
        "volume graph");
    const auto surface_light_path_shader_program =
        compiler.compile(
            surface_light_path_graph);
    expect(
        surface_light_path_shader_program.ok(),
        "failed to compile surface-only Light Path "
        "graph");
    const auto surface_light_path_lowered =
        compile_surface_program(
            *surface_light_path_shader_program
                 .program);
    expect(
        surface_light_path_lowered.ok(),
        "failed to lower surface-only Light Path "
        "graph");
    const VolumeProgramCapabilityComponent
        capabilities;
    const auto spatial_capabilities =
        capabilities.analyze(
            *spatial_lowered.program);
    const auto homogeneous_capabilities =
        capabilities.analyze(
            *homogeneous_lowered.program);
    const auto spatial_light_path_capabilities =
        capabilities.analyze(
            *spatial_light_path_lowered.program);
    const auto surface_light_path_capabilities =
        capabilities.analyze(
            *surface_light_path_lowered.program);
    const auto program_has_surface_light_path =
        std::any_of(
            surface_light_path_lowered.program
                ->value_instructions()
                .begin(),
            surface_light_path_lowered.program
                ->value_instructions()
                .end(),
            [](const auto &instruction) {
                return instruction.operation ==
                       ValueOperation::path_ray_depth;
            });
    expect(
        program_has_surface_light_path &&
            !spatial_capabilities.homogeneous &&
            !spatial_capabilities.has_light_path &&
            homogeneous_capabilities.homogeneous &&
            homogeneous_capabilities.has_light_path &&
            !spatial_light_path_capabilities
                 .homogeneous &&
            spatial_light_path_capabilities
                .has_light_path &&
            surface_light_path_capabilities
                .homogeneous &&
            surface_light_path_capabilities
                .has_light_path,
        "volume-root capability analysis changed");

    SceneSnapshot snapshot;
    TriangleMeshDesc geometry{
        .name = "scene-majorant triangle",
        .positions = {
            {-1.0f, -1.0f, -1.0f},
            {1.0f, -1.0f, -1.0f},
            {0.0f, 1.0f, 1.0f}},
        .triangles = {{0u, 1u, 2u}},
        .material_slots = {
            spatial_material,
            homogeneous_material}};
    snapshot.geometries.emplace(
        geometry_id, geometry);
    add_instance(
        snapshot,
        81u,
        geometry_id,
        transform(
            {2.0f, -3.0f, 5.0f},
            {0.5f, 0.5f, 0.5f}),
        {},
        91u);

    Context context{program};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto scene =
        std::make_shared<LuisaSceneData>();
    scene->device =
        Device{device.impl_shared()};

    const auto spatial_surface_tag =
        scene->surfaces.create<GraphSurface>(
            spatial_lowered.program);
    const auto homogeneous_surface_tag =
        scene->surfaces.create<GraphSurface>(
            homogeneous_lowered.program);
    const auto spatial_light_path_surface_tag =
        scene->surfaces.create<GraphSurface>(
            spatial_light_path_lowered.program);
    auto host_parameters =
        parameter_data(*spatial_lowered.program);
    const auto homogeneous_parameter_block =
        static_cast<std::uint32_t>(
            host_parameters.scalars.size());
    const auto homogeneous_parameters =
        parameter_data(
            *homogeneous_lowered.program);
    host_parameters.scalars.insert(
        host_parameters.scalars.end(),
        homogeneous_parameters.scalars.begin(),
        homogeneous_parameters.scalars.end());
    host_parameters.vectors.insert(
        host_parameters.vectors.end(),
        homogeneous_parameters.vectors.begin(),
        homogeneous_parameters.vectors.end());
    const auto spatial_light_path_parameter_block =
        static_cast<std::uint32_t>(
            host_parameters.scalars.size());
    const auto spatial_light_path_parameters =
        parameter_data(
            *spatial_light_path_lowered.program);
    host_parameters.scalars.insert(
        host_parameters.scalars.end(),
        spatial_light_path_parameters.scalars.begin(),
        spatial_light_path_parameters.scalars.end());
    host_parameters.vectors.insert(
        host_parameters.vectors.end(),
        spatial_light_path_parameters.vectors.begin(),
        spatial_light_path_parameters.vectors.end());
    scene->scalar_parameter_buffer =
        device.create_buffer<float>(
            host_parameters.scalars.size());
    scene->vector_parameter_buffer =
        device.create_buffer<luisa::float3>(
            host_parameters.vectors.size());
    scene->cycles_bsdf_table_buffer =
        device.create_buffer<float>(1u);
    std::vector<std::uint32_t>
        volume_surface_flags;
    capabilities.merge_surface_flags(
        volume_surface_flags,
        spatial_surface_tag,
        *spatial_lowered.program);
    capabilities.merge_surface_flags(
        volume_surface_flags,
        homogeneous_surface_tag,
        *homogeneous_lowered.program);
    capabilities.merge_surface_flags(
        volume_surface_flags,
        spatial_light_path_surface_tag,
        *spatial_light_path_lowered.program);
    expect(
        volume_surface_flags.size() ==
                spatial_light_path_surface_tag +
                    1u &&
            volume_surface_flags[
                spatial_surface_tag] ==
                volume_surface_flag_heterogeneous &&
            volume_surface_flags[
                homogeneous_surface_tag] ==
                volume_surface_flag_light_path &&
            volume_surface_flags[
                spatial_light_path_surface_tag] ==
                (volume_surface_flag_heterogeneous |
                 volume_surface_flag_light_path),
        "volume surface capability flag table changed");
    scene->volume_surface_flag_count =
        static_cast<std::uint32_t>(
            volume_surface_flags.size());
    scene->volume_surface_flag_buffer =
        device.create_buffer<luisa::uint>(
            volume_surface_flags.size());

    const auto generated_mapping =
        make_generated_coordinate_mapping(
            geometry);
    const std::array geometry_records{
        GeometryGpu{
            .generated_transform =
                to_luisa(
                    generated_mapping
                        .object_to_generated)}};
    constexpr std::array instance_records{
        InstanceGpu{
            .geometry_index = 0u,
            .cycles_object_index = 91u}};
    scene->geometry_buffer =
        device.create_buffer<GeometryGpu>(1u);
    scene->instance_buffer =
        device.create_buffer<InstanceGpu>(1u);

    constexpr std::array positions{
        luisa::float3{-1.0f, -1.0f, -1.0f},
        luisa::float3{1.0f, -1.0f, -1.0f},
        luisa::float3{0.0f, 1.0f, 1.0f}};
    constexpr std::array triangles{
        Triangle{0u, 1u, 2u}};
    auto position_buffer =
        device.create_buffer<luisa::float3>(
            positions.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(
            triangles.size());
    auto mesh = device.create_mesh(
        position_buffer, triangle_buffer);
    scene->accel = device.create_accel();
    scene->accel.emplace_back(
        mesh,
        to_luisa(
            snapshot.instances.begin()
                ->second.transform),
        0xffu,
        false,
        0u);

    scene->attribute_binding_slot = 0u;
    scene->attribute_range_slot = 1u;
    scene->attribute_binding_buffer =
        device.create_buffer<
            AttributeBindingGpu>(1u);
    scene->attribute_range_buffer =
        device.create_buffer<
            AttributeRangeGpu>(1u);
    scene->heap =
        device.create_bindless_array(2u);
    scene->heap.emplace_on_update(
        scene->attribute_binding_slot,
        scene->attribute_binding_buffer);
    scene->heap.emplace_on_update(
        scene->attribute_range_slot,
        scene->attribute_range_buffer);

    scene->texture_heap =
        device.create_bindless_array(1u);
    scene->images.emplace_back(
        device.create_image<float>(
            PixelStorage::BYTE4, 1u, 1u));
    scene->texture_heap.emplace_on_update(
        0u,
        scene->images.back(),
        Sampler::linear_point_repeat());

    constexpr std::array cycles_table{1.0f};
    constexpr std::array attribute_bindings{
        AttributeBindingGpu{}};
    constexpr std::array attribute_ranges{
        AttributeRangeGpu{}};
    constexpr std::array dummy_pixel{
        std::byte{255u},
        std::byte{0u},
        std::byte{255u},
        std::byte{255u}};
    stream
        << scene->scalar_parameter_buffer.copy_from(
               luisa::span{host_parameters.scalars})
        << scene->vector_parameter_buffer.copy_from(
               luisa::span{host_parameters.vectors})
        << scene->cycles_bsdf_table_buffer
               .copy_from(
                   luisa::span{cycles_table})
        << scene->volume_surface_flag_buffer
               .copy_from(
                   luisa::span{
                       volume_surface_flags})
        << scene->geometry_buffer.copy_from(
               luisa::span{geometry_records})
        << scene->instance_buffer.copy_from(
               luisa::span{instance_records})
        << scene->attribute_binding_buffer
               .copy_from(
                   luisa::span{
                       attribute_bindings})
        << scene->attribute_range_buffer
               .copy_from(
                   luisa::span{attribute_ranges})
        << position_buffer.copy_from(
               luisa::span{positions})
        << triangle_buffer.copy_from(
               luisa::span{triangles})
        << scene->images.back().copy_from(
               luisa::span{dummy_pixel})
        << mesh.build()
        << scene->texture_heap.update()
        << scene->heap.update()
        << scene->accel.build()
        << synchronize();

    const std::map<MaterialId, VolumeMajorantSceneMaterial>
        materials{
            {spatial_material,
             {.surface_tag = spatial_surface_tag,
              .parameter_block = 0u,
              .shader =
                  37u |
                  cycles_shader_identity::
                      cast_shadow,
              .has_volume = true,
              .heterogeneous = true}},
            {homogeneous_material,
             {.surface_tag =
                  homogeneous_surface_tag,
              .parameter_block =
                  homogeneous_parameter_block,
              .shader =
                  38u |
                  cycles_shader_identity::
                      smooth_normal,
              .has_volume = true,
              .heterogeneous = false}}};
    const VolumeMajorantSceneComponent component;
    const auto plan =
        component.plan(snapshot, materials);
    expect(plan.ok(), plan.diagnostic);
    expect(
        plan.roots.size() == 2u &&
            plan.roots[0u].heterogeneous &&
            !plan.roots[1u].heterogeneous,
        "backend scene fixture did not plan one root "
        "per Cycles volume class");

    const auto built =
        component.build(scene, stream, plan);
    expect(
        built.ok(),
        "scene majorant build failed on " +
            std::string{backend} + ": " +
            built.diagnostic);
    expect(
        built.root_count == 2u &&
            built.range_count == 2u &&
            built.node_count == 10u,
        "scene majorant resource counts changed on " +
            std::string{backend});

    std::vector<VolumeMajorantRootGpu> roots(
        built.root_count);
    std::vector<VolumeMajorantNodeGpu> nodes(
        built.node_count);
    std::vector<VolumeMajorantRootRangeGpu> ranges(
        built.range_count);
    stream
        << scene->volume_majorant_root_buffer
               .copy_to(luisa::span{roots})
        << scene->volume_majorant_node_buffer
               .copy_to(luisa::span{nodes})
        << scene->volume_majorant_range_buffer
               .copy_to(luisa::span{ranges})
        << synchronize();

    expect(
        roots[0u].shader == 37u &&
            roots[0u].node == 0u &&
            close(roots[0u].scale.x, 0.5f) &&
            close(roots[0u].scale.y, 0.5f) &&
            close(roots[0u].scale.z, 0.5f) &&
            close(
                roots[0u].translation.x,
                1.5f) &&
            close(
                roots[0u].translation.y,
                1.5f) &&
            close(
                roots[0u].translation.z,
                1.5f),
        "uploaded root transform/identity changed on " +
            std::string{backend});
    expect(
        roots[1u].shader == 38u &&
            roots[1u].node == 9u &&
            close(roots[1u].scale.x, 0.5f) &&
            close(
                roots[1u].translation.x,
                1.5f),
        "uploaded homogeneous root transform/identity "
        "changed on " +
            std::string{backend});
    expect(
        ranges[0u].offset == 0u &&
            ranges[0u].count == 2u &&
            ranges[1u].offset == 2u &&
            ranges[1u].count == 0u &&
            scene->volume_majorant_world_range == 1u,
        "uploaded instance/World range partition "
        "changed on " +
            std::string{backend});
    expect(
        nodes[0u].parent == -1 &&
            nodes[0u].first_child == 1 &&
            nodes[0u].sigma_minimum >= 0.0f &&
            nodes[0u].sigma_minimum < 0.02f &&
            nodes[0u].sigma_maximum > 0.99f &&
            nodes[0u].sigma_maximum < 1.02f,
        "uploaded root extrema/topology changed on " +
            std::string{backend});
    for (auto child = std::size_t{1u};
         child < 9u;
         ++child) {
        expect(
            nodes[child].parent == 0 &&
                nodes[child].first_child == -1 &&
                nodes[child].sigma_minimum >= 0.0f &&
                nodes[child].sigma_maximum >=
                    nodes[child].sigma_minimum,
            "uploaded child topology/extrema changed "
            "on " +
                std::string{backend});
    }
    expect(
        nodes[9u].parent == -1 &&
            nodes[9u].first_child == -1 &&
            close(nodes[9u].sigma_minimum, 1.0f) &&
            close(nodes[9u].sigma_maximum, 1.0f),
        "Cycles 1^3 homogeneous root was not evaluated "
        "as a single raw-graph cell on " +
            std::string{backend});

    auto points =
        make_scene_volume_stack_entry_point_provider(
            scene);
    auto provider_output =
        device.create_buffer<luisa::float4>(7u);
    Kernel1D evaluate_provider =
        [scene,
         points,
         spatial_surface_tag,
         homogeneous_surface_tag,
         homogeneous_parameter_block,
         spatial_light_path_surface_tag,
         spatial_light_path_parameter_block](
            BufferVar<luisa::float4> output)
            noexcept {
            BufferShaderServices services{
                scene->scalar_parameter_buffer,
                scene->vector_parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            const auto world_origin =
                make_float3(
                    2.5f, -2.0f, 6.5f);
            const auto world_direction =
                make_float3(
                    1.0f, 0.0f, 0.0f);
            const VolumeShadingState
                camera_state{
                    .position = world_origin,
                    .incoming =
                        -world_direction,
                    .ray_visibility =
                        camera_visibility,
                    .ray_events = 0u,
                    .ray_depth = 0u,
                    .diffuse_depth = 0u,
                    .glossy_depth = 0u,
                    .transparent_depth = 0u,
                    .transmission_depth = 0u,
                    .ray_length = 0.0f,
                    .time = 0.0f};
            const VolumeShadingState
                indirect_state{
                    .position = world_origin,
                    .incoming =
                        -world_direction,
                    .ray_visibility =
                        diffuse_visibility,
                    .ray_events = 0u,
                    .ray_depth = 1u,
                    .diffuse_depth = 1u,
                    .glossy_depth = 0u,
                    .transparent_depth = 0u,
                    .transmission_depth = 0u,
                    .ray_length = 0.0f,
                    .time = 0.0f};
            auto camera_provider =
                make_scene_volume_majorant_entry_provider(
                    scene,
                    points,
                    services,
                    camera_state,
                    true);
            auto indirect_provider =
                make_scene_volume_majorant_entry_provider(
                    scene,
                    points,
                    services,
                    indirect_state,
                    true);
            const VolumeStackEntry
                spatial_entry{
                    .object = 91u,
                    .shader = 37u,
                    .surface_tag =
                        spatial_surface_tag,
                    .parameter_block = 0u,
                    .instance_id = 0u,
                    .sample_method =
                        volume_sample_distance,
                    .valid = true};
            const VolumeStackEntry
                homogeneous_entry{
                    .object = 91u,
                    .shader = 38u,
                    .surface_tag =
                        homogeneous_surface_tag,
                    .parameter_block =
                        homogeneous_parameter_block,
                    .instance_id = 0u,
                    .sample_method =
                        volume_sample_distance,
                    .valid = true};
            const VolumeStackEntry
                spatial_light_path_entry{
                    .object = 91u,
                    .shader = 39u,
                    .surface_tag =
                        spatial_light_path_surface_tag,
                    .parameter_block =
                        spatial_light_path_parameter_block,
                    .instance_id = 0u,
                    .sample_method =
                        volume_sample_distance,
                    .valid = true};
            auto world_entry =
                homogeneous_entry;
            world_entry.object =
                invalid_volume_identity;
            world_entry.instance_id =
                invalid_volume_identity;
            const VolumeMajorantLeaf leaf{
                .minimum = 0.0f,
                .maximum = 2.0f,
                .sigma_minimum = 0.25f,
                .sigma_maximum = 0.75f,
                .node = 0u,
                .valid = true};
            const auto object_space =
                indirect_provider->entry_space(
                    spatial_entry,
                    world_origin,
                    world_direction);
            const auto world_space =
                indirect_provider->entry_space(
                    world_entry,
                    world_origin,
                    world_direction);
            const auto spatial_extrema =
                indirect_provider->extrema(
                    spatial_entry,
                    leaf,
                    object_space.object_density,
                    0.25f,
                    world_origin,
                    world_direction);
            const auto camera_extrema =
                camera_provider->extrema(
                    homogeneous_entry,
                    leaf,
                    object_space.object_density,
                    0.25f,
                    world_origin,
                    world_direction);
            const auto runtime_extrema =
                indirect_provider->extrema(
                    homogeneous_entry,
                    leaf,
                    object_space.object_density,
                    0.25f,
                    world_origin,
                    world_direction);
            const auto spatial_runtime_extrema =
                indirect_provider->extrema(
                    spatial_light_path_entry,
                    leaf,
                    object_space.object_density,
                    0.25f,
                    world_origin,
                    world_direction);
            const auto next_spatial_runtime_extrema =
                indirect_provider->extrema(
                    spatial_light_path_entry,
                    leaf,
                    object_space.object_density,
                    0.75f,
                    world_origin,
                    world_direction);
            output.write(
                0u,
                make_float4(
                    object_space.ray_origin,
                    object_space.object_density));
            output.write(
                1u,
                make_float4(
                    object_space.ray_direction,
                    spatial_extrema.minimum));
            output.write(
                2u,
                make_float4(
                    spatial_extrema.maximum,
                    camera_extrema.minimum,
                    camera_extrema.maximum,
                    runtime_extrema.minimum));
            output.write(
                3u,
                make_float4(
                    runtime_extrema.maximum,
                    world_space.ray_origin.x,
                    world_space.ray_origin.y,
                    world_space.ray_origin.z));
            output.write(
                4u,
                make_float4(
                    world_space.ray_direction,
                    world_space.object_density));
            output.write(
                5u,
                make_float4(
                    spatial_runtime_extrema.minimum,
                    spatial_runtime_extrema.maximum,
                    0.0f,
                    0.0f));
            output.write(
                6u,
                make_float4(
                    next_spatial_runtime_extrema
                        .minimum,
                    next_spatial_runtime_extrema
                        .maximum,
                    0.0f,
                    0.0f));
        };
    ShaderOption provider_options;
    provider_options.enable_cache = true;
    provider_options.enable_fast_math = false;
    auto provider_shader =
        device.compile(
            evaluate_provider,
            provider_options);
    std::array<luisa::float4, 7u>
        provider_values{};
    stream
        << provider_shader(provider_output)
               .dispatch(1u)
        << provider_output.copy_to(
               luisa::span{provider_values})
        << synchronize();
    expect(
        close(provider_values[0u].x, 1.0f) &&
            close(provider_values[0u].y, 2.0f) &&
            close(provider_values[0u].z, 3.0f) &&
            close(provider_values[0u].w, 1.0f) &&
            close(provider_values[1u].x, 2.0f) &&
            close(provider_values[1u].y, 0.0f) &&
            close(provider_values[1u].z, 0.0f),
        "scene majorant provider did not apply the "
        "inverse TLAS transform on " +
            std::string{backend});
    expect(
        close(provider_values[1u].w, 0.25f) &&
            close(provider_values[2u].x, 0.75f) &&
            close(provider_values[2u].y, 0.25f) &&
            close(provider_values[2u].z, 0.75f) &&
            close(provider_values[2u].w, 0.0f) &&
            close(provider_values[3u].x, 0.0f),
        "scene majorant provider did not preserve baked "
        "extrema or re-evaluate Light Path on " +
            std::string{backend});
    expect(
        close(provider_values[3u].y, 2.5f) &&
            close(provider_values[3u].z, -2.0f) &&
            close(provider_values[3u].w, 6.5f) &&
            close(provider_values[4u].x, 1.0f) &&
            close(provider_values[4u].y, 0.0f) &&
            close(provider_values[4u].z, 0.0f) &&
            close(provider_values[4u].w, 1.0f),
        "scene majorant provider transformed the World "
        "entry on " +
            std::string{backend});
    expect(
        close(
            provider_values[5u].x,
            1.54166667f) &&
            close(
                provider_values[5u].y,
                3.0625f),
        "scene majorant provider did not preserve "
        "Cycles heterogeneous four-sample offset and "
        "1.5x safety bound on " +
            std::string{backend});
    expect(
        close(
            provider_values[6u].x,
            1.625f) &&
            close(
                provider_values[6u].y,
                3.1875f),
        "scene majorant provider did not re-evaluate "
        "the next Cycles tracking shade offset on " +
            std::string{backend});

    const auto generated_sobol =
        sampling::tabulated_sobol::
            generate_table(256u);
    std::vector<luisa::float4> sobol_values;
    sobol_values.reserve(
        generated_sobol.size());
    for (const auto value : generated_sobol) {
        sobol_values.emplace_back(
            value.x,
            value.y,
            value.z,
            value.w);
    }
    auto sobol_buffer =
        device.create_buffer<luisa::float4>(
            sobol_values.size());
    auto transport_output =
        device.create_buffer<luisa::float4>(
            6u);
    auto path_segment =
        make_path_heterogeneous_volume_component(
            scene,
            points,
            8u);
    Kernel1D evaluate_transport =
        [scene,
         points,
         path_segment =
             path_segment.get(),
         spatial_surface_tag](
            BufferFloat4 sobol,
            BufferFloat4 output) noexcept {
            BufferShaderServices services{
                scene->scalar_parameter_buffer,
                scene->vector_parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            const auto ray_origin =
                make_float3(
                    2.0f, -3.0f, 5.0f);
            const auto ray_direction =
                make_float3(
                    1.0f, 0.0f, 0.0f);
            const VolumeShadingState state{
                .position = ray_origin,
                .incoming =
                    -ray_direction,
                .ray_visibility =
                    camera_visibility,
                .ray_events = 0u,
                .ray_depth = 0u,
                .diffuse_depth = 0u,
                .glossy_depth = 0u,
                .transparent_depth = 0u,
                .transmission_depth = 0u,
                .ray_length = 0.0f,
                .time = 0.0f};
            const VolumeStackEntry entry{
                .object = 91u,
                .shader =
                    37u |
                    cycles_shader_identity::
                        cast_shadow,
                .surface_tag =
                    spatial_surface_tag,
                .parameter_block = 0u,
                .instance_id = 0u,
                .sample_method =
                    volume_sample_distance,
                .valid = true};
            VolumeStack stack{2u};
            stack.initialize_background(
                entry, true);
            constexpr auto sequence_size =
                std::uint32_t{256u};
            constexpr auto sample_index =
                std::uint32_t{63u};
            constexpr auto path_rng_offset =
                std::uint32_t{16u};
            const auto rng_hash =
                cycles_sampler::pixel_hash(
                    17u, 29u, 0u);
            const auto phase_random =
                cycles_sampler::sample_2d(
                    sobol,
                    sequence_size,
                    sample_index,
                    rng_hash,
                    cycles_sampler::
                        path_state_dimension(
                            path_rng_offset,
                            sampling::
                                tabulated_sobol::
                                    volume_phase_dimension));
            const auto heterogeneous =
                path_segment
                    ->stack_is_heterogeneous(
                        stack);
            const auto result =
                path_segment->emit(
                    {.stack = stack,
                     .services = services,
                     .state = state,
                     .sobol_table = sobol,
                     .sobol_sequence_size =
                         sequence_size,
                     .sample_index =
                         sample_index,
                     .rng_hash = rng_hash,
                     .path_rng_offset =
                         path_rng_offset,
                     .ray_origin =
                         ray_origin,
                     .ray_direction =
                         ray_direction,
                     .ray_minimum = 0.0f,
                     .ray_maximum = 0.5f,
                     .throughput =
                         make_float3(1.0f),
                     .reservoir_random =
                         0.01f,
                     .phase_random =
                         phase_random,
                     .guiding =
                         {.scattered_radiance =
                              make_float3(0.0f),
                          .transmitted_radiance =
                              make_float3(0.0f),
                          .majorant_optical_depth =
                              std::numeric_limits<
                                  float>::max(),
                          .enabled = false},
                     .direct =
                         {.requested_method =
                              volume_sample_none,
                          .light_position =
                              make_float3(0.0f),
                          .interval =
                              {.minimum = 0.0f,
                               .maximum = 0.0f},
                          .enabled = false},
                     .direct_light = nullptr,
                     .terminate = false});
            output.write(
                0u,
                make_float4(
                    result.transport
                        .throughput,
                    result.transport
                        .distance));
            output.write(
                1u,
                make_float4(
                    result.transport
                        .emission,
                    result.transport
                        .null_transmittance));
            output.write(
                2u,
                make_float4(
                    result.transport
                        .reservoir_random,
                    result.transport
                        .optical_depth,
                    cast<float>(
                        result.transport
                            .steps),
                    as<float>(
                        result.transport
                            .next_tracking_rng_offset)));
            output.write(
                3u,
                make_float4(
                    select(
                        0.0f,
                        1.0f,
                        heterogeneous),
                    select(
                        0.0f,
                        1.0f,
                        result.transport
                            .selected_scatter),
                    select(
                        0.0f,
                        1.0f,
                        result.transport
                            .traversal_exhausted),
                    select(
                        0.0f,
                        1.0f,
                        result.transport
                            .active)));
            output.write(
                4u,
                make_float4(
                    result.phase.direction,
                    result.phase.pdf));
            output.write(
                5u,
                make_float4(
                    phase_random,
                    select(
                        0.0f,
                        1.0f,
                        result.scattered),
                    select(
                        0.0f,
                        1.0f,
                        result.transport
                            .majorant_exceeded)));
        };
    ShaderOption transport_options;
    transport_options.enable_cache = true;
    transport_options.enable_fast_math = false;
    auto transport_shader =
        device.compile(
            evaluate_transport,
            transport_options);
    std::array<luisa::float4, 6u>
        transport_values{};
    stream
        << sobol_buffer.copy_from(
               luisa::span{sobol_values})
        << transport_shader(
               sobol_buffer,
               transport_output)
               .dispatch(1u)
        << transport_output.copy_to(
               luisa::span{
                   transport_values})
        << synchronize();
    for (const auto &value : transport_values) {
        expect(
            std::isfinite(value.x) &&
                std::isfinite(value.y) &&
                std::isfinite(value.z) &&
                std::isfinite(value.w),
            "production heterogeneous segment "
            "produced a non-finite value on " +
                std::string{backend});
    }
    // Pinned to current Cycles' absolute-ray octree interval, copied and
    // scrambled tracking RNG, weighted null recursion, and pure-emission
    // estimator. The original spatial GraphSurface is evaluated at the
    // candidate; the hierarchy contributes only its sampled majorant.
    expect(
        close(transport_values[0u].x, 1.0f) &&
            close(
                transport_values[0u].y,
                1.0f) &&
            close(
                transport_values[0u].z,
                1.0f) &&
            close(
                transport_values[0u].w,
                0.5f) &&
            close(
                transport_values[1u].x,
                0.5614846945f) &&
            close(
                transport_values[1u].y,
                0.4992200434f) &&
            close(
                transport_values[1u].z,
                0.4992200434f) &&
            close(
                transport_values[1u].w,
                1.0f) &&
            close(
                transport_values[2u].x,
                0.01f) &&
            close(
                transport_values[2u].y,
                0.5007811785f) &&
            close(
                transport_values[2u].z,
                2.0f) &&
            std::bit_cast<std::uint32_t>(
                transport_values[2u].w) ==
                362294328u &&
            close(
                transport_values[3u].x,
                1.0f) &&
            close(
                transport_values[3u].y,
                0.0f) &&
            close(
                transport_values[3u].z,
                1.0f) &&
            close(
                transport_values[3u].w,
                1.0f) &&
            close(
                transport_values[4u].x,
                1.0f) &&
            close(
                transport_values[4u].y,
                0.0f) &&
            close(
                transport_values[4u].z,
                0.0f) &&
            close(
                transport_values[4u].w,
                0.0f) &&
            close(
                transport_values[5u].x,
                0.099866651f) &&
            close(
                transport_values[5u].y,
                0.01901064254f) &&
            close(
                transport_values[5u].z,
                0.0f) &&
            close(
                transport_values[5u].w,
                0.0f),
        "production heterogeneous raw-graph segment "
        "changed on " +
            std::string{backend});
}

void test_last_value_operand_participates_in_volume_analysis() {
    // Isolate the IR edge contract from graph lowering: a spatial source is
    // reachable only through the final operand slot. An analysis that
    // hand-spells an a..j prefix will misclassify this volume as homogeneous.
    const SurfaceProgram program{
        0u,
        {},
        {ValueInstruction{
             .operation = ValueOperation::surface_position,
             .result_type = SocketType::point},
         ValueInstruction{
             .operation = ValueOperation::brick_factor,
             .result_type = SocketType::floating,
             .operands = [] {
                 std::vector<ValueExpressionId> operands(
                     value_operand::brick::count);
                 operands[value_operand::brick::squash_frequency] =
                     ValueExpressionId{0u};
                 return operands;
             }()}},
        {},
        {},
        {VolumeInstruction{
            .operation = VolumeOperation::scatter,
            .density = ValueExpressionId{1u}}},
        VolumeExpressionId{0u}};
    const auto capabilities =
        VolumeProgramCapabilityComponent{}.analyze(program);
    expect(
        !capabilities.homogeneous,
        "the final value operand was omitted from volume dependency analysis");
}

void test_displacement_spatial_metadata() {
    for (const auto operation :
         std::array{ValueOperation::displacement,
                    ValueOperation::vector_displacement}) {
        const auto input =
            operation == ValueOperation::displacement
                ? value_operand::displacement::height
                : value_operand::vector_displacement::vector;
        const auto make_program = [operation, input](
                                      bool spatial_dependency) {
            std::vector<ValueInstruction> values;
            if (spatial_dependency) {
                values.emplace_back(ValueInstruction{
                    .operation = ValueOperation::surface_position,
                    .result_type = SocketType::point});
            }
            std::vector<ValueExpressionId> operands(
                value_operation_operand_count(operation));
            operands[input] =
                spatial_dependency ? ValueExpressionId{0u}
                                   : ValueExpressionId{};
            const auto displacement = ValueExpressionId{
                static_cast<std::uint32_t>(values.size())};
            values.emplace_back(ValueInstruction{
                .operation = operation,
                .result_type = SocketType::vector,
                .operands = std::move(operands)});
            return SurfaceProgram{
                0u,
                {},
                std::move(values),
                {},
                {},
                {VolumeInstruction{
                    .operation = VolumeOperation::coefficients,
                    .emission_coefficients = displacement}},
                VolumeExpressionId{0u}};
        };
        const auto constant =
            VolumeProgramCapabilityComponent{}.analyze(
                make_program(false));
        const auto spatial =
            VolumeProgramCapabilityComponent{}.analyze(
                make_program(true));
        expect(
            constant.homogeneous &&
                !constant.has_spatial_values &&
                !spatial.homogeneous &&
                spatial.has_spatial_values,
            "Displacement intrinsic spatial metadata diverged from Cycles");
    }
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto backend =
            std::string_view{
                argc > 1
                    ? argv[1]
                    : "fallback"};
        test_scene_plan();
        test_flatten_contract();
        test_last_value_operand_participates_in_volume_analysis();
        test_displacement_spatial_metadata();
        run_scene_build(backend, argv[0]);
        std::cout
            << "All current-Cycles volume-majorant "
               "scene resource fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Volume-majorant scene fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
