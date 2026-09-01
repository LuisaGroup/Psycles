#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
            ("psycles-light-path-import-" + std::to_string(nonce));
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
    "name":"Light Path Material",
    "cycles_sync":{"shader_index":7},
    "node_tree":{
      "name":"Light Path Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Light Path","from_socket":"Portal Depth",
         "to_node":"Combine","to_socket":"X"},
        {"from_node":"Light Path","from_socket":"Ray Length",
         "to_node":"Combine","to_socket":"Y"},
        {"from_node":"Light Path","from_socket":"Is Camera Ray",
         "to_node":"Combine","to_socket":"Z"},
        {"from_node":"Combine","from_socket":"Vector",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Light Path","type":"LIGHT_PATH","mute":false,
          "internal_links":[],"properties":{},"special":{},"inputs":[],
          "outputs":[
            {"identifier":"Is Camera Ray","name":"Is Camera Ray","type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"Is Shadow Ray","name":"Is Shadow Ray","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Is Diffuse Ray","name":"Is Diffuse Ray","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Is Glossy Ray","name":"Is Glossy Ray","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Is Singular Ray","name":"Is Singular Ray","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Is Reflection Ray","name":"Is Reflection Ray","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Is Transmission Ray","name":"Is Transmission Ray","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Is Volume Scatter Ray","name":"Is Volume Scatter Ray","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Ray Length","name":"Ray Length","type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"Ray Depth","name":"Ray Depth","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Diffuse Depth","name":"Diffuse Depth","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Glossy Depth","name":"Glossy Depth","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Transparent Depth","name":"Transparent Depth","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Transmission Depth","name":"Transmission Depth","type":"NodeSocketFloat","linked":false,"default":0},
            {"identifier":"Portal Depth","name":"Portal Depth","type":"NodeSocketFloat","linked":true,"default":0}]
        },
        {
          "name":"Combine","type":"COMBXYZ","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat","linked":true,"default":0}],
          "outputs":[{"identifier":"Vector","name":"Vector","type":"NodeSocketVector","linked":true,"default":[0,0,0]}]
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color","type":"NodeSocketColor","linked":true,"default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength","type":"NodeSocketFloat","linked":false,"default":1}],
          "outputs":[{"identifier":"Emission","name":"Emission","type":"NodeSocketShader","linked":true}]
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

[[nodiscard]] std::vector<NodeLightPath>
light_path_records(const ShaderImage &image) {
  std::vector<NodeLightPath> result;
  for (auto index = std::size_t{}; index + 2u < image.words.size(); ++index) {
    if (image.words[index] != static_cast<std::uint32_t>(NODE_LIGHT_PATH)) {
      continue;
    }
    const auto path_type = image.words[index + 1u];
    const auto packed_output = image.words[index + 2u];
    require(path_type <= static_cast<std::uint32_t>(NODE_LP_ray_portal) &&
                (packed_output & 0xffffff00u) == 0u,
            "imported Light Path payload is malformed");
    result.emplace_back(static_cast<NodeLightPath>(path_type));
    index += 2u;
  }
  return result;
}

void test_light_path_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Light Path scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Light Path Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Light Path material is missing");

  auto semantic_count = std::size_t{};
  for (const auto &node : material->shader.nodes()) {
    semantic_count += node.type == node_type::light_path ? 1u : 0u;
  }
  require(semantic_count == 1u,
          "raw Light Path outputs did not share one semantic source node");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "imported Light Path graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);
  const auto records = light_path_records(image);
  static constexpr std::array expected{NODE_LP_camera, NODE_LP_ray_length,
                                       NODE_LP_ray_portal};
  require(std::ranges::equal(records, expected),
          "imported Light Path emitted the wrong ordered SVM records");
}

} // namespace

int main() {
  try {
    test_light_path_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
