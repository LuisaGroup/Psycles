#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] std::shared_ptr<const ShaderProgram>
compile_frontend(const ShaderGraph &graph) {
  const ShaderCompiler compiler{make_core_node_registry()};
  auto result = compiler.compile(graph);
  if (!result.ok()) {
    for (const auto &diagnostic : result.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(result.ok(), "attribute-request graph failed frontend validation");
  return std::move(result.program);
}

[[nodiscard]] ShaderImage compile_svm(const ShaderGraph &graph,
                                      AttributeIDMap &attribute_ids,
                                      ShaderCompileContext context = {}) {
  const auto program = compile_frontend(graph);
  auto image = compile_shader(*program, attribute_ids, context);
  require(image.valid, image.diagnostic);
  return image;
}

void add_surface_anchor(ShaderGraph &graph) {
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Surface");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});
}

void require_requests(const ShaderImage &image,
                      std::span<const AttributeStandard> expected_standards,
                      std::span<const std::string_view> expected_names,
                      std::string_view label) {
  std::vector<AttributeStandard> standards;
  std::vector<std::string> names;
  for (const auto &request : image.attribute_requests) {
    if (request.standard != ATTR_STD_NONE) {
      standards.emplace_back(request.standard);
    } else {
      names.emplace_back(request.name);
    }
  }
  std::ranges::sort(standards);
  std::ranges::sort(names);
  auto expected_standard_values = std::vector<AttributeStandard>{
      expected_standards.begin(), expected_standards.end()};
  auto expected_name_values =
      std::vector<std::string>{expected_names.begin(), expected_names.end()};
  std::ranges::sort(expected_standard_values);
  std::ranges::sort(expected_name_values);
  require(standards == expected_standard_values &&
              names == expected_name_values,
          label);
}

void test_standard_name_domain() {
  struct StandardNameCase {
    AttributeStandard standard;
    std::string_view name;
    AttributeStandard inverse;
  };
  // Direct image of Cycles 5.2.1 Attribute::standard_name. Vertex and corner
  // normals deliberately share "N"; name_standard returns the first enum.
  static constexpr auto cases = std::array{
      StandardNameCase{ATTR_STD_NONE, "", ATTR_STD_NONE},
      StandardNameCase{ATTR_STD_POSITION, "P", ATTR_STD_POSITION},
      StandardNameCase{ATTR_STD_RADIUS, "radius", ATTR_STD_RADIUS},
      StandardNameCase{ATTR_STD_VERTEX_NORMAL, "N", ATTR_STD_VERTEX_NORMAL},
      StandardNameCase{ATTR_STD_CORNER_NORMAL, "N", ATTR_STD_VERTEX_NORMAL},
      StandardNameCase{ATTR_STD_UV, "uv", ATTR_STD_UV},
      StandardNameCase{ATTR_STD_UV_TANGENT, "tangent", ATTR_STD_UV_TANGENT},
      StandardNameCase{ATTR_STD_UV_TANGENT_SIGN, "tangent_sign",
                       ATTR_STD_UV_TANGENT_SIGN},
      StandardNameCase{ATTR_STD_UV_TANGENT_UNDISPLACED, "undisplaced_tangent",
                       ATTR_STD_UV_TANGENT_UNDISPLACED},
      StandardNameCase{ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED,
                       "undisplaced_tangent_sign",
                       ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED},
      StandardNameCase{ATTR_STD_VERTEX_COLOR, "vertex_color",
                       ATTR_STD_VERTEX_COLOR},
      StandardNameCase{ATTR_STD_GENERATED, "generated", ATTR_STD_GENERATED},
      StandardNameCase{ATTR_STD_GENERATED_TRANSFORM, "generated_transform",
                       ATTR_STD_GENERATED_TRANSFORM},
      StandardNameCase{ATTR_STD_POSITION_UNDEFORMED, "undeformed",
                       ATTR_STD_POSITION_UNDEFORMED},
      StandardNameCase{ATTR_STD_POSITION_UNDISPLACED, "undisplaced",
                       ATTR_STD_POSITION_UNDISPLACED},
      StandardNameCase{ATTR_STD_NORMAL_UNDISPLACED, "undisplaced_N",
                       ATTR_STD_NORMAL_UNDISPLACED},
      StandardNameCase{ATTR_STD_PARTICLE, "particle", ATTR_STD_PARTICLE},
      StandardNameCase{ATTR_STD_CURVE_INTERCEPT, "curve_intercept",
                       ATTR_STD_CURVE_INTERCEPT},
      StandardNameCase{ATTR_STD_CURVE_LENGTH, "curve_length",
                       ATTR_STD_CURVE_LENGTH},
      StandardNameCase{ATTR_STD_CURVE_RANDOM, "curve_random",
                       ATTR_STD_CURVE_RANDOM},
      StandardNameCase{ATTR_STD_POINT_RANDOM, "point_random",
                       ATTR_STD_POINT_RANDOM},
      StandardNameCase{ATTR_STD_PTEX_FACE_ID, "ptex_face_id",
                       ATTR_STD_PTEX_FACE_ID},
      StandardNameCase{ATTR_STD_PTEX_UV, "ptex_uv", ATTR_STD_PTEX_UV},
      StandardNameCase{ATTR_STD_VOLUME_DENSITY, "density",
                       ATTR_STD_VOLUME_DENSITY},
      StandardNameCase{ATTR_STD_VOLUME_COLOR, "color", ATTR_STD_VOLUME_COLOR},
      StandardNameCase{ATTR_STD_VOLUME_FLAME, "flame", ATTR_STD_VOLUME_FLAME},
      StandardNameCase{ATTR_STD_VOLUME_HEAT, "heat", ATTR_STD_VOLUME_HEAT},
      StandardNameCase{ATTR_STD_VOLUME_TEMPERATURE, "temperature",
                       ATTR_STD_VOLUME_TEMPERATURE},
      StandardNameCase{ATTR_STD_VOLUME_VELOCITY, "velocity",
                       ATTR_STD_VOLUME_VELOCITY},
      StandardNameCase{ATTR_STD_VOLUME_VELOCITY_X, "velocity_x",
                       ATTR_STD_VOLUME_VELOCITY_X},
      StandardNameCase{ATTR_STD_VOLUME_VELOCITY_Y, "velocity_y",
                       ATTR_STD_VOLUME_VELOCITY_Y},
      StandardNameCase{ATTR_STD_VOLUME_VELOCITY_Z, "velocity_z",
                       ATTR_STD_VOLUME_VELOCITY_Z},
      StandardNameCase{ATTR_STD_POINTINESS, "pointiness", ATTR_STD_POINTINESS},
      StandardNameCase{ATTR_STD_RANDOM_PER_ISLAND, "random_per_island",
                       ATTR_STD_RANDOM_PER_ISLAND},
      StandardNameCase{ATTR_STD_SHADOW_TRANSPARENCY, "shadow_transparency",
                       ATTR_STD_SHADOW_TRANSPARENCY},
      StandardNameCase{ATTR_STD_NUM, "", ATTR_STD_NONE},
      StandardNameCase{ATTR_STD_NOT_FOUND, "", ATTR_STD_NONE},
  };
  for (const auto &item : cases) {
    require(attribute_standard_name(item.standard) == item.name,
            "Cycles standard attribute name table differs");
    require(attribute_standard_from_name(item.name) == item.inverse,
            "Cycles standard attribute inverse mapping differs");
  }
  require(attribute_standard_from_name("not_a_standard_attribute") ==
              ATTR_STD_NONE,
          "unknown attribute name was standardized");
}

void test_request_set_semantics() {
  AttributeRequestSet requests;
  requests.add_standard("");
  requests.add_standard("uv");
  requests.add(ATTR_STD_UV);
  requests.add("custom");
  requests.add("custom");
  requests.add(ATTR_STD_POSITION);

  const auto insertion_order = requests.requests();
  require(insertion_order.size() == 3u &&
              insertion_order[0u] ==
                  AttributeRequest{.standard = ATTR_STD_UV, .name = {}} &&
              insertion_order[1u] == AttributeRequest{.standard = ATTR_STD_NONE,
                                                      .name = "custom"} &&
              insertion_order[2u] ==
                  AttributeRequest{.standard = ATTR_STD_POSITION, .name = {}},
          "AttributeRequestSet insertion or duplicate semantics differ");
}

void test_pre_finalize_base_requests() {
  ShaderGraph graph;
  static_cast<void>(graph.add_node(node_type::noise_texture, "Dead Noise"));
  static_cast<void>(graph.add_node(node_type::image_texture, "Dead Image"));
  add_surface_anchor(graph);

  AttributeIDMap ids;
  const auto image = compile_svm(graph, ids);
  constexpr std::array standards{ATTR_STD_UV, ATTR_STD_GENERATED};
  require_requests(image, standards, {},
                   "base ShaderNode attribute transfer differs from Cycles");
  require(!image.node_types_used[NODE_TEX_NOISE] &&
              !image.node_types_used[NODE_TEX_IMAGE],
          "disconnected request sources leaked into SVM bytecode");
}

[[nodiscard]] ShaderImage
compile_anisotropic_dead_node(std::string_view node_type_name,
                              float anisotropy) {
  ShaderGraph graph;
  const auto node = graph.add_node(std::string{node_type_name}, "Dead BSDF");
  require(
      graph.set_input(node, "Anisotropy", SocketValue::floating(anisotropy)),
      "failed to configure anisotropic request probe");
  add_surface_anchor(graph);
  AttributeIDMap ids;
  return compile_svm(graph, ids);
}

void test_closure_requests() {
  constexpr std::array generated{ATTR_STD_GENERATED};
  require_requests(compile_anisotropic_dead_node(node_type::glossy_bsdf, 0.0f),
                   {}, {}, "isotropic Glossy requested a tangent basis");
  require_requests(compile_anisotropic_dead_node(node_type::glossy_bsdf, 0.2f),
                   generated, {},
                   "anisotropic Glossy lost its generated tangent basis");
  require_requests(
      compile_anisotropic_dead_node(node_type::metallic_bsdf, 0.2f), generated,
      {}, "anisotropic Metallic lost its generated tangent basis");

  ShaderGraph graph;
  static_cast<void>(
      graph.add_node(node_type::principled_bsdf, "Dead Principled"));
  add_surface_anchor(graph);
  AttributeIDMap ids;
  require_requests(compile_svm(graph, ids), generated, {},
                   "Principled lost Cycles' default tangent dependency");
}

[[nodiscard]] ShaderGraph normal_map_graph() {
  ShaderGraph graph;
  const auto normal_map = graph.add_node(node_type::normal_map, "Normal Map");
  require(graph.set_property(normal_map, "Attribute",
                             SocketValue::string("MappedUV")) &&
              graph.set_property(normal_map, "Base",
                                 SocketValue::string("ORIGINAL")),
          "failed to configure Normal Map request probe");
  const auto to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({normal_map, "Normal"}, to_vector, "Normal") &&
              graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
              graph.connect({to_color, "Color"}, emission, "Color"),
          "failed to connect Normal Map request probe");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

void test_hidden_names_do_not_renumber_svm() {
  const auto graph = normal_map_graph();
  const auto program = compile_frontend(graph);
  AttributeIDMap ids;
  const auto image = compile_shader(*program, ids, {});
  require(image.valid, image.diagnostic);
  constexpr std::array standards{ATTR_STD_NORMAL_UNDISPLACED};
  constexpr std::array names{
      std::string_view{"MappedUV"},
      std::string_view{"MappedUV.undisplaced_tangent"},
      std::string_view{"MappedUV.undisplaced_tangent_sign"},
  };
  require_requests(image, standards, names,
                   "Normal Map symbolic requests differ from Cycles");

  const auto bytecode_names = ids.bindings();
  require(bytecode_names.size() == 2u &&
              bytecode_names[0u].first == "MappedUV.undisplaced_tangent" &&
              bytecode_names[0u].second == ATTR_STD_NUM &&
              bytecode_names[1u].first == "MappedUV.undisplaced_tangent_sign" &&
              bytecode_names[1u].second == ATTR_STD_NUM + 1u,
          "hidden Normal Map base attribute renumbered SVM bytecode");

  const std::array units{
      ShaderTableCompileUnit{.shader_index = 0u, .shader = program.get()}};
  const auto table = compile_shader_table(units);
  require(table.table.valid, table.table.diagnostic);
  require(table.named_attributes.size() == 3u &&
              table.named_attributes[0u].first ==
                  "MappedUV.undisplaced_tangent" &&
              table.named_attributes[0u].second == ATTR_STD_NUM &&
              table.named_attributes[1u].first ==
                  "MappedUV.undisplaced_tangent_sign" &&
              table.named_attributes[1u].second == ATTR_STD_NUM + 1u &&
              table.named_attributes[2u].first == "MappedUV" &&
              table.named_attributes[2u].second == ATTR_STD_NUM + 2u,
          "scene attribute resolution did not defer hidden names");
}

void test_derived_named_attribute_request() {
  ShaderGraph graph;
  const auto attribute = graph.add_node(node_type::attribute, "Attribute");
  require(
      graph.set_property(attribute, "Attribute",
                         SocketValue::string("MappedUV.undisplaced_tangent")),
      "failed to configure derived Attribute request probe");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({attribute, "Fac"}, emission, "Strength"),
          "failed to connect derived Attribute request probe");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  AttributeIDMap ids;
  const auto image = compile_svm(graph, ids);
  constexpr std::array names{
      std::string_view{"MappedUV"},
      std::string_view{"MappedUV.undisplaced_tangent"},
  };
  require_requests(image, {}, names,
                   "derived Attribute did not request its source UV map");
}

[[nodiscard]] ShaderImage compile_coordinate_output(std::string_view socket) {
  ShaderGraph graph;
  const auto coordinate =
      graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
  const auto to_vector =
      graph.add_node(node_type::point_to_vector, "Point to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.connect({coordinate, std::string{socket}}, to_vector, "Point") &&
          graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
          graph.connect({to_color, "Color"}, emission, "Color"),
      "failed to connect Texture Coordinate request probe");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  AttributeIDMap ids;
  return compile_svm(graph, ids);
}

void test_coordinate_requests() {
  constexpr std::array generated{ATTR_STD_GENERATED};
  constexpr std::array uv{ATTR_STD_UV};
  require_requests(compile_coordinate_output("Generated"), generated, {},
                   "Texture Coordinate Generated request differs from Cycles");
  require_requests(compile_coordinate_output("UV"), uv, {},
                   "Texture Coordinate UV request differs from Cycles");

  ShaderGraph graph;
  const auto uv_map = graph.add_node(node_type::uv_map, "UV Map");
  require(
      graph.set_property(uv_map, "Attribute", SocketValue::string("DetailUV")),
      "failed to configure UV Map request probe");
  const auto to_vector =
      graph.add_node(node_type::point_to_vector, "Point to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({uv_map, "UV"}, to_vector, "Point") &&
              graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
              graph.connect({to_color, "Color"}, emission, "Color"),
          "failed to connect UV Map request probe");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  AttributeIDMap ids;
  constexpr std::array names{std::string_view{"DetailUV"}};
  require_requests(compile_svm(graph, ids), {}, names,
                   "named UV Map request differs from Cycles");
}

[[nodiscard]] ShaderImage compile_geometry_output(std::string_view socket) {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  if (socket == "Tangent") {
    const auto to_color =
        graph.add_node(node_type::vector_to_color, "Vector to Color");
    require(graph.connect({geometry, "Tangent"}, to_color, "Vector") &&
                graph.connect({to_color, "Color"}, emission, "Color"),
            "failed to connect Geometry Tangent request probe");
  } else {
    require(
        graph.connect({geometry, std::string{socket}}, emission, "Strength"),
        "failed to connect Geometry scalar request probe");
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  AttributeIDMap ids;
  return compile_svm(graph, ids);
}

void test_geometry_requests() {
  constexpr std::array generated{ATTR_STD_GENERATED};
  constexpr std::array pointiness{ATTR_STD_POINTINESS};
  constexpr std::array random_per_island{ATTR_STD_RANDOM_PER_ISLAND};
  require_requests(compile_geometry_output("Tangent"), generated, {},
                   "Geometry Tangent request differs from Cycles");
  require_requests(compile_geometry_output("Pointiness"), pointiness, {},
                   "Geometry Pointiness request differs from Cycles");
  require_requests(compile_geometry_output("RandomPerIsland"),
                   random_per_island, {},
                   "Geometry Random Per Island request differs from Cycles");
}

void test_vertex_color_standardization() {
  ShaderGraph graph;
  const auto color = graph.add_node(node_type::vertex_color, "Vertex Color");
  require(graph.set_property(color, "Layer Name", SocketValue::string("uv")),
          "failed to configure Vertex Color request probe");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({color, "Color"}, emission, "Color"),
          "failed to connect Vertex Color request probe");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  AttributeIDMap ids;
  constexpr std::array uv{ATTR_STD_UV};
  require_requests(compile_svm(graph, ids), uv, {},
                   "Vertex Color add_standard semantics differ from Cycles");
}

void test_volume_requests() {
  ShaderGraph graph;
  static_cast<void>(graph.add_node(node_type::attribute, "Dead Attribute"));
  static_cast<void>(
      graph.add_node(node_type::principled_volume, "Dead Principled Volume"));
  const auto volume =
      graph.add_node(node_type::volume_absorption, "Volume Anchor");
  graph.set_root(ShaderDomain::volume,
                 OutputRef{.node = volume, .socket = "Volume"});
  AttributeIDMap ids;
  constexpr std::array standards{ATTR_STD_GENERATED_TRANSFORM,
                                 ATTR_STD_VOLUME_DENSITY};
  require_requests(compile_svm(graph, ids), standards, {},
                   "volume attribute requests differ from Cycles");
}

void test_displacement_both_requests() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  require(graph.connect({geometry, "Normal"}, to_vector, "Normal"),
          "failed to connect displacement request probe");
  graph.set_root(ShaderDomain::displacement,
                 OutputRef{.node = to_vector, .socket = "Vector"});

  AttributeIDMap ids;
  const auto image = compile_svm(
      graph, ids,
      ShaderCompileContext{.background = false,
                           .displacement_method = DisplacementMethod::both});
  constexpr std::array standards{ATTR_STD_POSITION_UNDISPLACED,
                                 ATTR_STD_NORMAL_UNDISPLACED};
  require_requests(image, standards, {},
                   "BOTH displacement lost undisplaced geometry requests");
}

} // namespace

int main() {
  test_standard_name_domain();
  test_request_set_semantics();
  test_pre_finalize_base_requests();
  test_closure_requests();
  test_hidden_names_do_not_renumber_svm();
  test_derived_named_attribute_request();
  test_coordinate_requests();
  test_geometry_requests();
  test_vertex_color_standardization();
  test_volume_requests();
  test_displacement_both_requests();
  std::cout << "Cycles SVM attribute-request tests passed\n";
  return 0;
}
