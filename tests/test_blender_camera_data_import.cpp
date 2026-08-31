#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

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
            ("psycles-camera-data-import-" + std::to_string(nonce));
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
    "name":"Camera Data Material",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Camera Data Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Camera Data","from_socket":"View Vector",
         "to_node":"Separate View Vector","to_socket":"Vector"},
        {"from_node":"Separate View Vector","from_socket":"X",
         "to_node":"Camera Data RGB","to_socket":"X"},
        {"from_node":"Camera Data","from_socket":"View Z Depth",
         "to_node":"Camera Data RGB","to_socket":"Y"},
        {"from_node":"Camera Data","from_socket":"View Distance",
         "to_node":"Camera Data RGB","to_socket":"Z"},
        {"from_node":"Camera Data RGB","from_socket":"Vector",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Camera Data","type":"CAMERA","mute":false,
          "internal_links":[],"inputs":[],"properties":{},"special":{},
          "outputs":[
            {"identifier":"View Vector","name":"View Vector",
             "type":"NodeSocketVector","linked":true,"default":[0,0,0]},
            {"identifier":"View Z Depth","name":"View Z Depth",
             "type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"View Distance","name":"View Distance",
             "type":"NodeSocketFloat","linked":true,"default":0}]
        },
        {
          "name":"Separate View Vector","type":"SEPXYZ","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,"default":[0,0,0]}],
          "outputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":true,"default":0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":false,"default":0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":0}]
        },
        {
          "name":"Camera Data RGB","type":"COMBXYZ","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":true,"default":0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":true,"default":0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":true,"default":0}],
          "outputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,"default":[0,0,0]}]
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color","type":"NodeSocketColor",
             "linked":true,"default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength","type":"NodeSocketFloat",
             "linked":false,"default":1}],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}]
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

void test_camera_data_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Camera Data scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Camera Data Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Camera Data material is missing");

  const ShaderNode *camera = nullptr;
  auto camera_count = std::size_t{};
  for (const auto &node : material->shader.nodes()) {
    if (node.type == node_type::camera_data) {
      camera = &node;
      ++camera_count;
    }
  }
  require(camera_count == 1u && camera != nullptr && camera->inputs.empty(),
          "three raw Camera Data outputs did not share one semantic node");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "imported Camera Data graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);

  const auto opcode = std::find(image.words.begin(), image.words.end(),
                                static_cast<std::uint32_t>(NODE_CAMERA));
  require(opcode != image.words.end() && image.words.end() - opcode >= 2 &&
              opcode[1u] == 0x00040300u,
          "imported Camera Data payload differs from external Cycles 5.2.1");
  require(std::count(image.words.begin(), image.words.end(),
                     static_cast<std::uint32_t>(NODE_CAMERA)) == 1,
          "imported Camera Data emitted more than one opcode");
}

} // namespace

int main() {
  try {
    test_camera_data_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
