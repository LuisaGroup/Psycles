#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

class TemporaryDirectory {
private:
  std::filesystem::path _path;

public:
  TemporaryDirectory() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
            ("psycles-curve-family-import-" + std::to_string(nonce));
    std::filesystem::create_directories(_path);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return _path;
  }
};

void write_scene_bundle(const std::filesystem::path &directory) {
  {
    std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  std::ofstream scene{directory / "scene.json"};
  scene << std::setprecision(9);
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],
  "materials":[{
    "name":"Curve Family Material",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Curve Family Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Coordinates","from_socket":"Generated",
         "to_node":"Vector Curves","to_socket":"Vector"},
        {"from_node":"Vector Curves","from_socket":"Vector",
         "to_node":"Separate XYZ","to_socket":"Vector"},
        {"from_node":"Separate XYZ","from_socket":"X",
         "to_node":"Float Curve","to_socket":"Value"},
        {"from_node":"Float Curve","from_socket":"Value",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Coordinates","type":"TEX_COORD","mute":false,
          "internal_links":[],"inputs":[],
          "outputs":[{"identifier":"Generated","name":"Generated",
            "type":"NodeSocketVector","linked":true,
            "default":[0,0,0]}],
          "properties":{"from_instancer":false},"special":{}
        },
        {
          "name":"Vector Curves","type":"CURVE_VEC","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Fac","name":"Factor",
             "type":"NodeSocketFloatFactor","linked":false,
             "default":0.37},
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,
             "default":[0,0,0]}],
          "outputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,
            "default":[0,0,0]}],
          "properties":{},
          "special":{"curve_mapping":{
            "min_x":-0.25,"max_x":1.3,"extrapolate":false,
            "samples":[
)JSON";
  for (auto index = std::uint32_t{}; index <= 256u; ++index) {
    const auto t = static_cast<float>(index) / 256.0f;
    if (index != 0u) {
      scene << ",\n";
    }
    scene << '[' << t << ',' << (0.25f + 0.5f * t) << ',' << (1.0f - t) << ']';
  }
  scene << R"JSON(
            ],"curves":[]}}
        },
        {
          "name":"Separate XYZ","type":"SEPXYZ","mute":false,
          "internal_links":[],
          "inputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,
            "default":[0,0,0]}],
          "outputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":true,"default":0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":false,"default":0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":0}],
          "properties":{},"special":{}
        },
        {
          "name":"Float Curve","type":"CURVE_FLOAT","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Factor","name":"Factor",
             "type":"NodeSocketFloatFactor","linked":false,
             "default":0.61},
            {"identifier":"Value","name":"Value",
             "type":"NodeSocketFloat","linked":true,"default":1}],
          "outputs":[{"identifier":"Value","name":"Value",
            "type":"NodeSocketFloat","linked":true,"default":0}],
          "properties":{},
          "special":{"curve_mapping":{
            "min_x":-0.35,"max_x":1.25,"extrapolate":true,
            "samples":[
)JSON";
  for (auto index = std::uint32_t{}; index <= 256u; ++index) {
    const auto t = static_cast<float>(index) / 256.0f;
    if (index != 0u) {
      scene << ",\n";
    }
    scene << 0.1f + 0.7f * t;
  }
  scene << R"JSON(
            ],"curves":[]}}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1}],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}],
          "properties":{},"special":{}
        }
      ]
    }
  }],
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

[[nodiscard]] const ShaderNode &find_semantic_node(const MaterialDesc &material,
                                                   std::string_view type) {
  const ShaderNode *result = nullptr;
  for (const auto &node : material.shader.nodes()) {
    if (node.type == type) {
      require(result == nullptr, "curve-family import duplicated a node");
      result = &node;
    }
  }
  require(result != nullptr, "curve-family import lost a semantic node");
  return *result;
}

void require_table_metadata(const ShaderNode &node, float expected_min,
                            float expected_max, bool expected_extrapolate) {
  const auto sampled = node.properties.find("Sampled");
  const auto min_x = node.properties.find("MinX");
  const auto max_x = node.properties.find("MaxX");
  const auto extrapolate = node.properties.find("Extrapolate");
  const auto table = node.properties.find("Table");
  require(sampled != node.properties.end() &&
              std::get<bool>(sampled->second.value) &&
              min_x != node.properties.end() &&
              std::get<float>(min_x->second.value) == expected_min &&
              max_x != node.properties.end() &&
              std::get<float>(max_x->second.value) == expected_max &&
              extrapolate != node.properties.end() &&
              std::get<bool>(extrapolate->second.value) == expected_extrapolate,
          "curve-family import lost sampled/domain/extend metadata");
  require(table != node.properties.end() &&
              std::count(std::get<std::string>(table->second.value).begin(),
                         std::get<std::string>(table->second.value).end(),
                         ';') == 256,
          "curve-family import did not preserve all 257 samples");
}

void test_curve_family_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "curve-family scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Curve Family Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "curve-family material is missing");
  const auto &vector = find_semantic_node(*material, node_type::vector_curve);
  const auto &scalar = find_semantic_node(*material, node_type::float_curve);
  require_table_metadata(vector, -0.25f, 1.3f, false);
  require_table_metadata(scalar, -0.35f, 1.25f, true);
  require(vector.inputs.at("Factor").value.has_value() &&
              std::get<float>(vector.inputs.at("Factor").value->value) ==
                  0.37f &&
              vector.inputs.at("Vector").source.has_value(),
          "Vector Curves raw Fac or Vector link was lowered incorrectly");
  require(scalar.inputs.at("Factor").value.has_value() &&
              std::get<float>(scalar.inputs.at("Factor").value->value) ==
                  0.61f &&
              scalar.inputs.at("Value").source.has_value(),
          "Float Curve raw Factor or Value link was lowered incorrectly");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "imported curve-family graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);

  const auto vector_opcode = std::find(image.words.begin(), image.words.end(),
                                       static_cast<std::uint32_t>(NODE_CURVES));
  const auto float_opcode =
      std::find(image.words.begin(), image.words.end(),
                static_cast<std::uint32_t>(NODE_FLOAT_CURVE));
  require(vector_opcode != image.words.end() &&
              float_opcode != image.words.end(),
          "imported curve family lost a Cycles opcode");
  const auto vector_begin =
      static_cast<std::size_t>(vector_opcode - image.words.begin());
  const auto float_begin =
      static_cast<std::size_t>(float_opcode - image.words.begin());
  require(vector_begin + 9u + 257u * 4u <= image.words.size() &&
              image.words[vector_begin + 4u] ==
                  std::bit_cast<std::uint32_t>(0.37f) &&
              image.words[vector_begin + 5u] ==
                  std::bit_cast<std::uint32_t>(-0.25f) &&
              image.words[vector_begin + 6u] ==
                  std::bit_cast<std::uint32_t>(1.3f) &&
              image.words[vector_begin + 7u] == 257u &&
              (image.words[vector_begin + 8u] & 0xffu) == 0u,
          "imported Vector Curves payload differs from Cycles");
  require(float_begin + 7u + 257u <= image.words.size() &&
              image.words[float_begin + 1u] ==
                  std::bit_cast<std::uint32_t>(0.61f) &&
              image.words[float_begin + 3u] ==
                  std::bit_cast<std::uint32_t>(-0.35f) &&
              image.words[float_begin + 4u] ==
                  std::bit_cast<std::uint32_t>(1.25f) &&
              image.words[float_begin + 5u] == 257u &&
              (image.words[float_begin + 6u] & 0xffu) == 1u,
          "imported Float Curve payload differs from Cycles");

  const auto vector_word = [&](std::uint32_t sample, std::uint32_t component) {
    return image.words[vector_begin + 9u + sample * 4u + component];
  };
  require(vector_word(0u, 0u) == std::bit_cast<std::uint32_t>(0.0f) &&
              vector_word(128u, 1u) == std::bit_cast<std::uint32_t>(0.5f) &&
              vector_word(256u, 2u) == std::bit_cast<std::uint32_t>(0.0f) &&
              vector_word(0u, 3u) == std::bit_cast<std::uint32_t>(1.0f),
          "imported Vector Curves table or float4 padding changed");
  require(image.words[float_begin + 7u] == std::bit_cast<std::uint32_t>(0.1f) &&
              image.words[float_begin + 7u + 256u] ==
                  std::bit_cast<std::uint32_t>(0.8f),
          "imported Float Curve scalar table changed");
}

} // namespace

int main() {
  try {
    test_curve_family_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
