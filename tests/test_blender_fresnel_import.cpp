#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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
            ("psycles-fresnel-import-" + std::to_string(nonce));
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
    "name":"Fresnel Family",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Fresnel Family",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Layer Weight","from_socket":"Fresnel",
         "to_node":"Combine","to_socket":"X"},
        {"from_node":"Layer Weight","from_socket":"Facing",
         "to_node":"Combine","to_socket":"Y"},
        {"from_node":"Fresnel","from_socket":"Fac",
         "to_node":"Combine","to_socket":"Z"},
        {"from_node":"Combine","from_socket":"Vector",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Layer Weight","type":"LAYER_WEIGHT","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Blend","name":"Blend","type":"NodeSocketFloat",
             "linked":false,"default":0.25},
            {"identifier":"Normal","name":"Normal","type":"NodeSocketVector",
             "linked":false,"default":[0,0,0]}],
          "outputs":[
            {"identifier":"Fresnel","name":"Fresnel","type":"NodeSocketFloat",
             "linked":true,"default":0},
            {"identifier":"Facing","name":"Facing","type":"NodeSocketFloat",
             "linked":true,"default":0}]
        },
        {
          "name":"Fresnel","type":"FRESNEL","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"IOR","name":"IOR","type":"NodeSocketFloat",
             "linked":false,"default":1.5},
            {"identifier":"Normal","name":"Normal","type":"NodeSocketVector",
             "linked":false,"default":[0,0,0]}],
          "outputs":[{"identifier":"Fac","name":"Fac",
             "type":"NodeSocketFloat","linked":true,"default":0}]
        },
        {
          "name":"Combine","type":"COMBXYZ","mute":false,
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

void test_fresnel_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Fresnel-family scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Fresnel Family") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Fresnel-family material is missing");

  std::size_t fresnel_count{};
  std::size_t layer_count{};
  for (const auto &node : material->shader.nodes()) {
    fresnel_count += node.type == node_type::fresnel ? 1u : 0u;
    layer_count += node.type == node_type::layer_weight ? 1u : 0u;
  }
  require(fresnel_count == 1u && layer_count == 1u,
          "Layer Weight outputs did not share one semantic source node");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "imported Fresnel-family graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);

  std::size_t svm_fresnel_count{};
  std::size_t svm_layer_count{};
  for (auto index = std::size_t{}; index < image.words.size(); ++index) {
    if (image.words[index] == static_cast<std::uint32_t>(NODE_FRESNEL)) {
      ++svm_fresnel_count;
      require(index + 2u < image.words.size() &&
                  (image.words[index + 2u] & 0xffu) == SVM_STACK_INVALID,
              "unlinked imported Fresnel Normal is not SVM_STACK_INVALID");
    }
    if (image.words[index] == static_cast<std::uint32_t>(NODE_LAYER_WEIGHT)) {
      ++svm_layer_count;
      require(index + 3u < image.words.size() &&
                  (image.words[index + 3u] & 0xffu) == SVM_STACK_INVALID,
              "unlinked imported Layer Weight Normal is not invalid");
    }
  }
  require(svm_fresnel_count == 1u && svm_layer_count == 2u,
          "imported Fresnel family emitted the wrong opcode multiplicity");
}

} // namespace

int main() {
  try {
    test_fresnel_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
