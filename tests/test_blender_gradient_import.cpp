#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error{message};
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
            ("psycles-gradient-import-" + std::to_string(nonce));
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
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],
  "materials":[{
    "name":"Gradient Color Material",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Gradient Color Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[{
        "from_node":"Gradient","from_socket":"Color",
        "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Gradient","type":"TEX_GRADIENT","mute":false,
          "internal_links":[],
          "inputs":[{
            "identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":false,
            "default":[0.0,0.0,0.0]}],
          "outputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[0.8,0.8,0.8,1.0]},
            {"identifier":"Fac","name":"Factor",
             "type":"NodeSocketFloat","linked":false,"default":0.0}],
          "properties":{"gradient_type":"SPHERICAL"},"special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1.0,1.0,1.0,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1.0}],
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
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

void write_mapped_scene_bundle(const std::filesystem::path &directory) {
  {
    std::ofstream geometry{directory / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO1\0", 8);
  }
  std::ofstream scene{directory / "scene.json"};
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],
  "materials":[{
    "name":"Mapped Gradient Material",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Mapped Gradient Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Combine","from_socket":"Vector",
         "to_node":"Mapping","to_socket":"Vector"},
        {"from_node":"Mapping","from_socket":"Vector",
         "to_node":"Gradient","to_socket":"Vector"},
        {"from_node":"Gradient","from_socket":"Fac",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Combine","type":"COMBXYZ","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":false,"default":0.2},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":false,"default":-0.4},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":0.3}],
          "outputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,"default":[0,0,0]}],
          "properties":{},"special":{}
        },
        {
          "name":"Mapping","type":"MAPPING","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":true,"default":[0,0,0]},
            {"identifier":"Location","name":"Location",
             "type":"NodeSocketVectorTranslation","linked":false,
             "default":[0.1,0.2,-0.1]},
            {"identifier":"Rotation","name":"Rotation",
             "type":"NodeSocketVectorEuler","linked":false,
             "default":[0.17,-0.11,0.3]},
            {"identifier":"Scale","name":"Scale",
             "type":"NodeSocketVectorXYZ","linked":false,
             "default":[2.0,0.5,1.3]}],
          "outputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,"default":[0,0,0]}],
          "properties":{"vector_type":"POINT"},"special":{}
        },
        {
          "name":"Gradient","type":"TEX_GRADIENT","mute":false,
          "internal_links":[],
          "inputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,"default":[0,0,0]}],
          "outputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.8,0.8,0.8,1.0]},
            {"identifier":"Fac","name":"Factor",
             "type":"NodeSocketFloat","linked":true,"default":0.0}],
          "properties":{"gradient_type":"DIAGONAL"},"special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1.0}],
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
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

void test_gradient_color_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Gradient Color scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Gradient Color Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Gradient Color material is missing");

  const ShaderNode *gradient = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.type == node_type::gradient_texture) {
      require(gradient == nullptr,
              "Gradient Color import duplicated the semantic node");
      gradient = &node;
    }
  }
  require(gradient != nullptr, "Gradient Color import lost the semantic node");
  const auto type = gradient->properties.find("GradientType");
  require(type != gradient->properties.end() &&
              std::get<std::string>(type->second.value) == "SPHERICAL",
          "Gradient Color import lost the exact enum property");

  ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "Gradient Color graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());

  // Complete shader-local stream from the external Cycles 5.2.1 Color cell
  // of gradient_opcode_matrix. fac_offset is invalid and color_offset is 3.
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x00000015u,
      0x0000000bu, 0x00000000u, 0x00000000u, 0x00000040u, 0x00000006u,
      0x0003ff00u, 0x00000007u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  require(image.words.size() == expected.size(),
          "Gradient Color imported stream has the wrong size");
  require(std::equal(image.words.begin(), image.words.end(), expected.begin()),
          "Gradient Color imported stream differs from Cycles 5.2.1");
}

void test_mapping_fold_does_not_fold_gradient() {
  TemporaryDirectory temporary;
  write_mapped_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "mapped Gradient scene did not import");

  const auto material =
      std::find_if(imported.scene->materials.begin(),
                   imported.scene->materials.end(), [](const auto &item) {
                     return item.second.name == "Mapped Gradient Material";
                   });
  require(material != imported.scene->materials.end(),
          "mapped Gradient material is missing");

  ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->second.shader);
  require(shader.ok(), "mapped Gradient graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());

  // External Cycles 5.2.1 gradient_mapping_constant_fold oracle. Mapping is
  // folded to a vector value by Cycles, while Gradient remains an opcode.
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u, 0x00000013u,
      0x00000000u, 0x3f0553fbu, 0x3d605b04u, 0x3e95acd8u, 0x00000040u,
      0x00000003u, 0x00ff0300u, 0x0000000du, 0x00000000u, 0x00000003u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};
  require(image.words.size() == expected.size(),
          "mapped Gradient stream has the wrong size");
  require(std::equal(image.words.begin(), image.words.end(), expected.begin()),
          "mapped Gradient stream differs from Cycles 5.2.1");
}

} // namespace

int main() {
  try {
    test_gradient_color_import();
    test_mapping_fold_does_not_fold_gradient();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
