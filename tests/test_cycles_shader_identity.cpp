#include "../src/luisa/cycles_shader_identity.h"
#include "../src/luisa/path_tracer_displacement_plan.h"
#include "../src/luisa/path_tracer_scene_geometry.h"

#include <array>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <vector>

namespace {

namespace identity =
    psycles::luisa_backend::detail::cycles_shader_identity;
using psycles::contract::RayVisibility;
using psycles::contract::all_ray_visibility;
using psycles::contract::visibility_bit;
using psycles::luisa_backend::detail::
    CyclesPrimitiveIntervalError;
using psycles::luisa_backend::detail::
    CyclesPrimitiveIntervalResolver;
using psycles::luisa_backend::detail::
    world_triangle_area;
using psycles::luisa_backend::detail::
    build_cycles_instance_intersection_plan;
using psycles::luisa_backend::detail::
    cycles_inverse_transform;
using psycles::luisa_backend::detail::
    cycles_transform_point;
using psycles::luisa_backend::detail::
    make_cycles_mesh_displacement_plan;

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_surface_shader_composition() {
    require(
        identity::surface(6u, false) == 0x40000006u,
        "flat surface shader identity changed");
    require(
        identity::surface(6u, true) == 0xc0000006u,
        "smooth surface shader identity changed");
    require(
        (identity::surface(0x1234u, false) &
         identity::shader_mask) == 0x1234u,
        "surface flags changed the shader-vector index");
}

void test_light_shader_composition() {
    using psycles::contract::LightType;
    require(
        identity::light_type(LightType::point) == 0u &&
            identity::light_type(LightType::distant) == 1u &&
            identity::light_type(LightType::background) == 2u &&
            identity::light_type(LightType::area) == 3u &&
            identity::light_type(LightType::spot) == 4u,
        "scene light types no longer map to Cycles' kernel ABI");

    const auto point_visibility =
        all_ray_visibility &
        ~visibility_bit(RayVisibility::camera);
    require(
        identity::analytic_light(
            5u,
            true,
            point_visibility,
            true,
            false) == 0x41000005u,
        "delta point-light shader identity changed");
    require(
        identity::analytic_light_flags(
            true,
            point_visibility,
            true,
            false) == 0x41000000u,
        "point-light flags depend on shader-vector identity");
    require(
        identity::analytic_light(
            5u,
            true,
            point_visibility,
            true,
            true) == 0x51000005u,
        "finite point-light MIS flag changed");
    require(
        identity::analytic_light(
            5u,
            false,
            all_ray_visibility,
            false,
            false) ==
            (identity::exclude_shadow_catcher | 5u),
        "cast-shadow or shadow-catcher composition changed");

    const auto visibility =
        all_ray_visibility &
        ~visibility_bit(RayVisibility::diffuse) &
        ~visibility_bit(RayVisibility::glossy) &
        ~visibility_bit(RayVisibility::transmission) &
        ~visibility_bit(RayVisibility::volume_scatter);
    require(
        identity::object_visibility(visibility, true) ==
            (identity::exclude_diffuse |
             identity::exclude_glossy |
             identity::exclude_transmit |
             identity::exclude_scatter),
        "object ray visibility did not map to Cycles shader flags");

    require(
        identity::emissive_triangle(
            9u,
            true,
            visibility,
            false) ==
            (identity::surface(9u, true) |
             identity::use_mis |
             identity::exclude_diffuse |
             identity::exclude_glossy |
             identity::exclude_transmit |
             identity::exclude_scatter |
             identity::exclude_shadow_catcher),
        "triangle-light shader identity changed");
    require(
        identity::emissive_triangle_flags(
            true, visibility, false) ==
            (identity::emissive_triangle(
                 9u, true, visibility, false) &
             ~identity::shader_mask),
        "triangle-light flags diverged from its full identity");

    require(
        identity::background_light(
            3u,
            false,
            visibility &
                ~visibility_bit(RayVisibility::camera)) ==
            (3u |
             identity::use_mis |
             identity::exclude_diffuse |
             identity::exclude_glossy |
             identity::exclude_transmit |
             identity::exclude_scatter),
        "background-light shader identity changed");
    require(
        identity::background_light_flags(
            false,
            visibility &
                ~visibility_bit(RayVisibility::camera)) ==
            (identity::background_light(
                 3u,
                 false,
                 visibility &
                     ~visibility_bit(RayVisibility::camera)) &
             ~identity::shader_mask),
        "background-light flags diverged from its full identity");
}

void test_geometry_identity() {
    CyclesPrimitiveIntervalResolver resolver;
    const auto first = resolver.resolve(3u);
    const auto explicit_gap = resolver.resolve(2u, 5u);
    const auto implicit_after_gap = resolver.resolve(4u);
    require(
        first.offset == 0u &&
            explicit_gap.offset == 5u &&
            implicit_after_gap.offset == 7u &&
            resolver.end() == 11u,
        "implicit and explicit Cycles primitive prefixes diverged");

    const auto overlap = resolver.resolve(1u, 10u);
    require(
        !overlap.offset &&
            overlap.error ==
                CyclesPrimitiveIntervalError::overlap &&
            resolver.end() == 11u,
        "overlapping primitive interval was accepted or mutated state");

    CyclesPrimitiveIntervalResolver boundary;
    const auto last = boundary.resolve(
        1u,
        std::numeric_limits<std::uint32_t>::max());
    const auto overflow = boundary.resolve(1u);
    require(
        last.offset ==
                std::numeric_limits<std::uint32_t>::max() &&
            !overflow.offset &&
            overflow.error ==
                CyclesPrimitiveIntervalError::out_of_range,
        "Cycles primitive address-space boundary changed");

    psycles::Mat4f transform;
    transform.elements[0u] = 2.0f;
    transform.elements[5u] = 3.0f;
    require(
        world_triangle_area(
            transform,
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}) == 3.0f,
        "emissive triangle area ignored instance transform");
}

void test_cycles_intersection_representation_plan() {
    using namespace psycles::contract;
    using psycles::Mat4f;
    using psycles::Vec3f;
    SceneSnapshot scene;
    constexpr MaterialId ordinary_material{1u};
    constexpr MaterialId bssrdf_material{2u};
    constexpr MaterialId displaced_material{3u};
    scene.materials.emplace(
        ordinary_material,
        MaterialDesc{.name = "ordinary"});
    scene.materials.emplace(
        bssrdf_material,
        MaterialDesc{.name = "bssrdf"});
    scene.materials.emplace(
        displaced_material,
        MaterialDesc{
            .name = "displaced",
            .displacement_method =
                DisplacementMethod::displacement});

    const auto support = [](MaterialId material) {
        return TriangleMeshDesc{
            .name = "support",
            .positions = {
                {0.0f, 0.0f, 0.0f},
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f}},
            .triangles = {{0u, 1u, 2u}},
            .material_slots = {material}};
    };
    scene.geometries.emplace(GeometryId{1u}, support(ordinary_material));
    auto differently_shaded = support(ordinary_material);
    differently_shaded.name = "same support, different shading";
    differently_shaded.normals = {
        .domain = MeshAttributeDomain::point,
        .values = {
            {0.0f, 0.0f, -1.0f},
            {0.0f, 0.0f, -1.0f},
            {0.0f, 0.0f, -1.0f}}};
    differently_shaded.uv = {
        .domain = MeshAttributeDomain::corner,
        .values = {
            {0.25f, 0.25f},
            {0.75f, 0.25f},
            {0.25f, 0.75f}}};
    scene.geometries.emplace(
        GeometryId{2u}, std::move(differently_shaded));
    auto changed_support = support(bssrdf_material);
    changed_support.positions[0u].x = std::nextafter(0.0f, 1.0f);
    scene.geometries.emplace(GeometryId{3u}, std::move(changed_support));
    scene.geometries.emplace(
        GeometryId{4u}, support(displaced_material));
    auto adaptive = support(ordinary_material);
    adaptive.uses_adaptive_subdivision = true;
    scene.geometries.emplace(GeometryId{5u}, std::move(adaptive));

    Mat4f transform;
    transform.elements[0u] = 0.75f;
    transform.elements[5u] = 1.25f;
    transform.elements[10u] = 2.0f;
    transform.elements[12u] = 3.0f;
    transform.elements[13u] = -2.0f;
    transform.elements[14u] = 5.0f;
    auto changed_transform = transform;
    changed_transform.elements[12u] =
        std::nextafter(changed_transform.elements[12u], 4.0f);
    auto displaced_transform = transform;
    displaced_transform.elements[13u] += 1.0f;
    auto adaptive_transform = transform;
    adaptive_transform.elements[14u] += 2.0f;
    scene.instances.emplace(
        InstanceId{1u},
        InstanceDesc{
            .name = "unique depsgraph instance",
            .geometry = GeometryId{1u},
            .transform = transform,
            .is_blender_instance = true});
    scene.instances.emplace(
        InstanceId{2u},
        InstanceDesc{
            .name = "shared support A",
            .geometry = GeometryId{2u},
            .transform = transform});
    scene.instances.emplace(
        InstanceId{3u},
        InstanceDesc{
            .name = "shared support B",
            .geometry = GeometryId{2u},
            .transform = changed_transform});
    scene.instances.emplace(
        InstanceId{4u},
        InstanceDesc{
            .name = "one-bit support change",
            .geometry = GeometryId{3u},
            .transform = transform});
    scene.instances.emplace(
        InstanceId{5u},
        InstanceDesc{
            .name = "true displacement",
            .geometry = GeometryId{4u},
            .transform = displaced_transform});
    scene.instances.emplace(
        InstanceId{6u},
        InstanceDesc{
            .name = "adaptive subdivision",
            .geometry = GeometryId{5u},
            .transform = adaptive_transform});

    const auto plan = build_cycles_instance_intersection_plan(
        scene, std::set<MaterialId>{bssrdf_material});
    require(plan.size() == 6u, "instance intersection plan changed size");
    require(
        plan[0u].coincident_count == 2u &&
            plan[0u].coincident_next == 1u &&
            plan[1u].coincident_count == 2u &&
            plan[1u].coincident_next == 0u,
        "exact support ignored or included shading attributes");
    require(
        plan[2u].coincident_count == 1u &&
            plan[3u].coincident_count == 1u,
        "one-bit transform or support changes were grouped");
    require(
        plan[0u].transform_applied &&
            !plan[1u].transform_applied &&
            !plan[2u].transform_applied,
        "Cycles geometry-user transform relation diverged");
    require(
        !plan[3u].transform_applied &&
            !plan[4u].transform_applied &&
            !plan[5u].transform_applied,
        "BSSRDF, displacement, or subdivision transform gate diverged");

    const auto point = Vec3f{0.125f, -0.5f, 2.0f};
    const auto world = cycles_transform_point(transform, point);
    const auto round_trip = cycles_transform_point(
        cycles_inverse_transform(transform), world);
    require(
        std::abs(round_trip.x - point.x) < 1.0e-6f &&
            std::abs(round_trip.y - point.y) < 1.0e-6f &&
            std::abs(round_trip.z - point.z) < 1.0e-6f,
        "Cycles affine inverse does not round-trip a point");
}

void test_cycles_displacement_vertex_ownership() {
    using namespace psycles::contract;
    const auto displacement_graph = [] {
        ShaderGraph graph;
        const auto output = graph.add_node("test_displacement");
        graph.set_root(
            ShaderDomain::displacement,
            OutputRef{output, "Vector"});
        return graph;
    };
    std::map<MaterialId, MaterialDesc> materials;
    materials.emplace(
        MaterialId{1u},
        MaterialDesc{
            .name = "bump",
            .shader = displacement_graph(),
            .displacement_method = DisplacementMethod::bump});
    materials.emplace(
        MaterialId{2u},
        MaterialDesc{
            .name = "both",
            .shader = displacement_graph(),
            .displacement_method = DisplacementMethod::both});
    materials.emplace(
        MaterialId{3u},
        MaterialDesc{
            .name = "true",
            .shader = displacement_graph(),
            .displacement_method = DisplacementMethod::displacement});
    materials.emplace(
        MaterialId{4u},
        MaterialDesc{
            .name = "unconnected true",
            .displacement_method = DisplacementMethod::displacement});

    TriangleMeshDesc geometry{
        .name = "shared displacement support",
        .positions = {
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f},
            {1.0f, 1.0f, 0.0f},
            {2.0f, 1.0f, 0.0f}},
        .triangles = {
            {0u, 1u, 2u},
            {2u, 1u, 3u},
            {3u, 2u, 4u},
            {0u, 4u, 1u}},
        .material_slots = {
            MaterialId{1u},
            MaterialId{2u},
            MaterialId{3u},
            MaterialId{4u}},
        // The last out-of-range slot must clamp to the last material exactly
        // like the render-time triangle-material lookup.
        .triangle_material_slots = {0u, 1u, 2u, 99u}};
    const auto plan = make_cycles_mesh_displacement_plan(
        geometry, materials);
    require(
        plan.evaluations.size() == 4u,
        "Cycles displacement did not evaluate each eligible vertex once");
    const auto expected = std::array{
        std::array{2u, 1u, 0u, 2u},
        std::array{1u, 1u, 1u, 2u},
        std::array{3u, 1u, 2u, 2u},
        std::array{4u, 2u, 2u, 3u}};
    for (std::size_t index = 0u; index < expected.size(); ++index) {
        const auto &actual = plan.evaluations[index];
        require(
            actual.vertex_index == expected[index][0u] &&
                actual.primitive_index == expected[index][1u] &&
                actual.corner_index == expected[index][2u] &&
                actual.material.value == expected[index][3u],
            "Cycles first-eligible-triangle displacement order diverged");
    }
    require(
        plan.true_displacement_triangles ==
            std::vector<bool>{false, false, true, false},
        "BOTH incorrectly requested post-displacement vertex normals");

    const std::array overrides{MaterialId{3u}};
    const auto overridden = make_cycles_mesh_displacement_plan(
        geometry, materials, overrides);
    require(
        overridden.evaluations.size() == 5u &&
            overridden.evaluations.front().primitive_index == 0u &&
            overridden.evaluations.front().material == MaterialId{3u} &&
            overridden.true_displacement_triangles ==
                std::vector<bool>{true, false, true, false},
        "first Cycles object material override did not own displacement");
}

}// namespace

int main() {
    test_surface_shader_composition();
    test_light_shader_composition();
    test_geometry_identity();
    test_cycles_intersection_representation_plan();
    test_cycles_displacement_vertex_ownership();
    return EXIT_SUCCESS;
}
