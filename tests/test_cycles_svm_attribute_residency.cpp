#include "../src/luisa/path_tracer_attribute_residency.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace psycles::contract;
using namespace psycles::luisa_backend::detail;
namespace abi = psycles::compiler::cycles_svm;

void require(bool condition, const char *message) {
  if (!condition) { throw std::runtime_error{message}; }
}
} // namespace

int main() {
  try {
    // Inputs at the native compiler/geometry boundary, not an expected SVM
    // stream or a material evaluator. Native compiler request collection and
    // map packing have their separate original-Cycles oracle regressions.
    abi::CompiledShaderTable table;
    table.named_attributes = {{"map.tangent_sign", 100u},
                              {"map.undisplaced_tangent", 101u},
                              {"paint", 102u}, {"unused", 103u},
                              {"map.tangent", 104u}, {"map", 105u}};
    table.shader_attribute_ids_used = {
        {}, {abi::ATTR_STD_POINTINESS, 100u, 101u}, {102u}, {104u}, {105u}};
    const std::map<MaterialId, std::uint32_t> shaders{
        {MaterialId{1u}, 0u}, {MaterialId{2u}, 1u}, {MaterialId{3u}, 2u},
        {MaterialId{4u}, 3u}, {MaterialId{5u}, 4u}};
    constexpr GeometryId mesh_id{1u};
    SceneSnapshot snapshot;
    auto &mesh = snapshot.geometries[mesh_id];
    mesh.positions.resize(3u);
    mesh.triangles = {{0u, 1u, 2u}};
    mesh.triangle_material_slots = {0u};
    // Slot 1 is never selected by a primitive, but Cycles used_shaders still
    // requires its attributes. Primitive reachability would prune it wrongly.
    mesh.material_slots = {MaterialId{1u}, MaterialId{2u}};
    mesh.uv_layers["map"].values.resize(3u);
    mesh.uv_tangent_layers["map"].values.resize(3u);
    mesh.color_attributes["paint"].values.resize(3u);
    mesh.color_attributes["unused"].values.resize(3u);
    mesh.pointiness_source.emplace();
    snapshot.instances.emplace(InstanceId{1u}, InstanceDesc{
        .geometry = mesh_id, .material_overrides = {MaterialId{3u}}});
    auto plan = build_scene_attribute_residency_plan(snapshot, table, shaders);
    const auto &demand = plan.geometry(mesh_id).demand;
    require(!demand.all, "native finite attribute requests became conservative-all");
    require(demand.contains(cycles_pointiness_attribute_id),
            "unused shader slot lost its native Pointiness request");
    require(demand.contains(attribute_id("paint")), "override lost its color request");
    require(demand.contains(uv_attribute_id("map")) &&
                demand.contains(uv_tangent_attribute_id("map")) &&
                demand.contains(uv_undisplaced_tangent_attribute_id("map")),
            "native named tangent/sign request lost its MikkTSpace source");
    require(!demand.contains(attribute_id("unused")), "dead attribute was retained");
    require(plan.resident_binding_count == 5u && plan.source_binding_count == 6u,
            "native resident/source accounting is wrong");

    // Actual named color takes precedence over a tangent-looking suffix,
    // just as in the native geometry source packer.
    snapshot.instances.clear();
    mesh.material_slots = {MaterialId{4u}};
    mesh.color_attributes["map.tangent"].values.resize(3u);
    plan = build_scene_attribute_residency_plan(snapshot, table, shaders);
    require(plan.geometry(mesh_id).contains(attribute_id("map.tangent")) &&
                !plan.geometry(mesh_id).contains(uv_tangent_attribute_id("map")),
            "real color name was reinterpreted as a generated tangent");

    mesh.color_attributes.erase("map.tangent");
    mesh.cycles_byte_color_attributes["map.tangent"].values.resize(3u);
    plan = build_scene_attribute_residency_plan(snapshot, table, shaders);
    require(!plan.geometry(mesh_id).contains(uv_tangent_attribute_id("map")),
            "native byte-color name was reinterpreted as a generated tangent");

    constexpr GeometryId curve_id{2u};
    auto &curve = snapshot.curve_geometries[curve_id];
    curve.material_slots = {MaterialId{3u}, MaterialId{5u}};
    curve.uv_layers["map"].resize(2u);
    curve.color_attributes["paint"].resize(2u);
    curve.color_attributes["unused"].resize(2u);
    plan = build_scene_attribute_residency_plan(snapshot, table, shaders);
    require(plan.geometry(curve_id).contains(uv_attribute_id("map")) &&
                plan.geometry(curve_id).contains(attribute_id("paint")) &&
                !plan.geometry(curve_id).contains(attribute_id("unused")),
            "native curve attribute projection is wrong");
    std::cout << "Native Cycles attribute residency passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
