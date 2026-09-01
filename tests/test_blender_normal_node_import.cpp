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
#include <variant>

namespace {

using psycles::Vec3f;
using psycles::adapter::load_blender_scene_bundle;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

constexpr auto cycles_record = std::array<std::uint32_t, 8u>{
    0x00000048u, 0xc0800000u, 0x40a00000u, 0x3f800000u,
    0x00000300u, 0x3f800000u, 0xc0000000u, 0x40400000u};

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
            ("psycles-normal-import-" + std::to_string(nonce));
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
    "name":"Normal Node",
    "cycles_sync":{"shader_index":11},
    "node_tree":{
      "name":"Normal Node",
      "surface_root":{"node":"Add","socket":"Shader"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Normal","from_socket":"Normal",
         "to_node":"Normal Emission","to_socket":"Color"},
        {"from_node":"Normal","from_socket":"Dot",
         "to_node":"Dot Emission","to_socket":"Strength"},
        {"from_node":"Normal Emission","from_socket":"Emission",
         "to_node":"Add","to_socket":"Shader"},
        {"from_node":"Dot Emission","from_socket":"Emission",
         "to_node":"Add","to_socket":"Shader_001"}],
      "nodes":[
        {
          "name":"Normal","type":"NORMAL","bl_idname":"ShaderNodeNormal",
          "mute":false,"internal_links":[],"properties":{},"special":{},
          "inputs":[{"identifier":"Normal","name":"Normal",
            "type":"NodeSocketVectorDirection","linked":false,
            "default":[-4,5,1]}],
          "outputs":[
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVectorDirection","linked":true,
             "default":[1,-2,3]},
            {"identifier":"Dot","name":"Dot",
             "type":"NodeSocketFloat","linked":true,"default":0}]
        },
        {
          "name":"Normal Emission","type":"EMISSION","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color","type":"NodeSocketColor",
             "linked":true,"default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength","type":"NodeSocketFloat",
             "linked":false,"default":1}],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}]
        },
        {
          "name":"Dot Emission","type":"EMISSION","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color","type":"NodeSocketColor",
             "linked":false,"default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength","type":"NodeSocketFloat",
             "linked":true,"default":1}],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}]
        },
        {
          "name":"Add","type":"ADD_SHADER","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Shader","name":"Shader","type":"NodeSocketShader",
             "linked":true},
            {"identifier":"Shader_001","name":"Shader",
             "type":"NodeSocketShader","linked":true}],
          "outputs":[{"identifier":"Shader","name":"Shader",
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

[[nodiscard]] const Vec3f &vector_value(const SocketValue &value) {
  const auto *vector = std::get_if<Vec3f>(&value.value);
  require(vector != nullptr, "Normal value is not a typed float3");
  return *vector;
}

void test_normal_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Normal-node scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Normal Node") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Normal-node material is missing");

  const ShaderNode *normal = nullptr;
  std::size_t vector_math_count{};
  for (const auto &node : material->shader.nodes()) {
    if (node.type == node_type::normal) {
      require(normal == nullptr,
              "Normal outputs did not share one semantic node");
      normal = &node;
    }
    vector_math_count += node.type == node_type::vector_math ? 1u : 0u;
  }
  require(normal != nullptr && vector_math_count == 0u,
          "raw Normal was decomposed into an alternate Vector Math DAG");
  require(vector_value(normal->properties.at("Direction")) ==
                  Vec3f{1.0f, -2.0f, 3.0f} &&
              normal->inputs.at("Normal").value.has_value() &&
              vector_value(*normal->inputs.at("Normal").value) ==
                  Vec3f{-4.0f, 5.0f, 1.0f},
          "Normal direction/input data changed during import");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "imported Normal graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);

  const auto record = std::search(image.words.begin(), image.words.end(),
                                  cycles_record.begin(), cycles_record.end());
  require(record != image.words.end(),
          "imported Normal payload differs from Cycles 5.2.1");
}

} // namespace

int main() {
  try {
    test_normal_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
