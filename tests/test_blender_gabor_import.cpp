#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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
            ("psycles-gabor-import-" + std::to_string(nonce));
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
    "name":"Gabor Multi Output",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Gabor Multi Output",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Gabor","from_socket":"Phase",
         "to_node":"Emission","to_socket":"Color"},
        {"from_node":"Gabor","from_socket":"Value",
         "to_node":"Emission","to_socket":"Strength"}],
      "nodes":[
        {
          "name":"Gabor","type":"TEX_GABOR","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":false,
             "default":[0,0,0]},
            {"identifier":"Scale","name":"Scale",
             "type":"NodeSocketFloat","linked":false,"default":3.25},
            {"identifier":"Frequency","name":"Frequency",
             "type":"NodeSocketFloat","linked":false,"default":1.75},
            {"identifier":"Anisotropy","name":"Anisotropy",
             "type":"NodeSocketFloat","linked":false,"default":0.625},
            {"identifier":"Orientation 2D","name":"Orientation 2D",
             "type":"NodeSocketFloat","linked":false,"default":-0.35},
            {"identifier":"Orientation 3D","name":"Orientation 3D",
             "type":"NodeSocketVector","linked":false,
             "default":[0.2,-0.4,0.7]}],
          "outputs":[
            {"identifier":"Value","name":"Value",
             "type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"Phase","name":"Phase",
             "type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"Intensity","name":"Intensity",
             "type":"NodeSocketFloat","linked":false,"default":0}],
          "properties":{"gabor_type":"3D"},
          "special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":true,"default":1}],
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

void test_multi_output_gabor_is_shared() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Gabor multi-output scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Gabor Multi Output") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Gabor multi-output material is missing");

  std::size_t gabor_count{};
  for (const auto &node : material->shader.nodes()) {
    gabor_count += node.type == node_type::gabor_texture ? 1u : 0u;
  }
  require(gabor_count == 1u,
          "Gabor outputs duplicated the semantic graph node");

  ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "Gabor multi-output graph did not validate");

  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid && image.node_types_used[NODE_TEX_GABOR],
          "Gabor multi-output graph did not compile to Cycles SVM");

  std::size_t record_count{};
  for (auto index = std::size_t{};
       index + 1u + sizeof(SVMNodeTexGabor) / sizeof(std::uint32_t) <=
       image.words.size();
       ++index) {
    if (image.words[index] != NODE_TEX_GABOR) {
      continue;
    }
    SVMNodeTexGabor payload{};
    std::memcpy(&payload, image.words.data() + index + 1u, sizeof(payload));
    if (payload.gabor_type != NODE_GABOR_TYPE_3D) {
      continue;
    }
    ++record_count;
    require(payload.value_offset != SVM_STACK_INVALID &&
                payload.phase_offset != SVM_STACK_INVALID &&
                payload.intensity_offset == SVM_STACK_INVALID,
            "Gabor SVM record did not preserve exact live-output offsets");
  }
  require(record_count == 1u,
          "multi-output Gabor did not emit exactly one Cycles SVM record");
}

void test_exported_gabor_bundle(const std::filesystem::path &directory) {
  const auto imported = load_blender_scene_bundle(directory);
  require(imported.ok(), "exported Gabor matrix did not import");

  ShaderCompiler frontend{make_core_node_registry()};
  std::size_t material_count{};
  for (const auto &[id, material] : imported.scene->materials) {
    static_cast<void>(id);
    if (!material.name.starts_with("SVM Gabor ")) {
      continue;
    }
    ++material_count;

    std::size_t graph_gabor_count{};
    for (const auto &node : material.shader.nodes()) {
      graph_gabor_count += node.type == node_type::gabor_texture ? 1u : 0u;
    }
    if (graph_gabor_count != 1u) {
      throw std::runtime_error{"exported material '" + material.name +
                               "' did not retain one Gabor graph node"};
    }

    const auto shader = frontend.compile(material.shader);
    require(shader.ok(), "exported Gabor material did not validate");
    AttributeIDMap attributes;
    ImageIDMap images;
    const auto image =
        compile_shader(*shader.program, attributes, images,
                       ShaderCompileContext{.background = false});
    require(image.valid && image.node_types_used[NODE_VECTOR_MATH] &&
                image.node_types_used[NODE_TEX_GABOR],
            "exported Gabor material lost its Cycles SVM topology");

    std::size_t record_count{};
    for (auto index = std::size_t{};
         index + 1u + sizeof(SVMNodeTexGabor) / sizeof(std::uint32_t) <=
         image.words.size();
         ++index) {
      if (image.words[index] != NODE_TEX_GABOR) {
        continue;
      }
      SVMNodeTexGabor payload{};
      std::memcpy(&payload, image.words.data() + index + 1u, sizeof(payload));
      if (payload.gabor_type != NODE_GABOR_TYPE_2D &&
          payload.gabor_type != NODE_GABOR_TYPE_3D) {
        continue;
      }
      ++record_count;
      const auto live_outputs =
          (payload.value_offset != SVM_STACK_INVALID ? 1u : 0u) +
          (payload.phase_offset != SVM_STACK_INVALID ? 1u : 0u) +
          (payload.intensity_offset != SVM_STACK_INVALID ? 1u : 0u);
      require(live_outputs == 1u,
              "exported Gabor material changed its single live output");
    }
    require(record_count == 1u,
            "exported material did not emit one Gabor SVM record");
  }
  require(material_count == 12u,
          "exported Gabor oracle no longer contains all twelve materials");
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2) {
    test_exported_gabor_bundle(argv[1]);
    return 0;
  }
  test_multi_output_gabor_is_shared();
  return 0;
}
