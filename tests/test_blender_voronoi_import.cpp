#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/surface_program.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
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
            ("psycles-voronoi-import-" + std::to_string(nonce));
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
    "name":"Voronoi Multi Output",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Voronoi Multi Output",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Voronoi","from_socket":"Color",
         "to_node":"Emission","to_socket":"Color"},
        {"from_node":"Voronoi","from_socket":"Distance",
         "to_node":"Emission","to_socket":"Strength"}],
      "nodes":[
        {
          "name":"Voronoi","type":"TEX_VORONOI","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":false,
             "default":[0,0,0]},
            {"identifier":"W","name":"W","type":"NodeSocketFloat",
             "linked":false,"default":-0.375},
            {"identifier":"Scale","name":"Scale","type":"NodeSocketFloat",
             "linked":false,"default":2.5},
            {"identifier":"Detail","name":"Detail","type":"NodeSocketFloat",
             "linked":false,"default":1.75},
            {"identifier":"Roughness","name":"Roughness",
             "type":"NodeSocketFloat","linked":false,"default":0.63},
            {"identifier":"Lacunarity","name":"Lacunarity",
             "type":"NodeSocketFloat","linked":false,"default":2.17},
            {"identifier":"Smoothness","name":"Smoothness",
             "type":"NodeSocketFloat","linked":false,"default":0.4},
            {"identifier":"Exponent","name":"Exponent",
             "type":"NodeSocketFloat","linked":false,"default":1.5},
            {"identifier":"Randomness","name":"Randomness",
             "type":"NodeSocketFloat","linked":false,"default":0.83}],
          "outputs":[
            {"identifier":"Distance","name":"Distance",
             "type":"NodeSocketFloat","linked":true,"default":0},
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[0,0,0,1]},
            {"identifier":"Position","name":"Position",
             "type":"NodeSocketVector","linked":false,"default":[0,0,0]},
            {"identifier":"W","name":"W","type":"NodeSocketFloat",
             "linked":false,"default":0},
            {"identifier":"Radius","name":"Radius",
             "type":"NodeSocketFloat","linked":false,"default":0}],
          "properties":{"voronoi_dimensions":"3D","feature":"F1",
                        "distance":"EUCLIDEAN","normalize":false},
          "special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color","type":"NodeSocketColor",
             "linked":true,"default":[1,1,1,1]},
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

void test_multi_output_voronoi_is_shared() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Voronoi multi-output scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Voronoi Multi Output") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Voronoi multi-output material is missing");

  std::size_t voronoi_count{};
  for (const auto &node : material->shader.nodes()) {
    voronoi_count += node.type == node_type::voronoi_texture ? 1u : 0u;
  }
  require(voronoi_count == 1u,
          "Voronoi outputs duplicated the semantic graph node");

  ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "Voronoi multi-output graph did not validate");

  const auto legacy = compile_surface_program(*shader.program);
  require(legacy.ok(), "Voronoi multi-output graph broke transitional lowering");
  std::size_t distance_count{};
  std::size_t color_count{};
  for (const auto &instruction : legacy.program->value_instructions()) {
    distance_count +=
        instruction.operation == ValueOperation::voronoi_distance ? 1u : 0u;
    color_count +=
        instruction.operation == ValueOperation::voronoi_color ? 1u : 0u;
  }
  require(distance_count == 1u && color_count == 1u,
          "Voronoi live outputs were lost in transitional lowering");

  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid && image.node_types_used[NODE_TEX_VORONOI],
          "Voronoi multi-output graph did not compile to Cycles SVM");

  std::size_t record_count{};
  for (auto index = std::size_t{};
       index + 1u + sizeof(SVMNodeTexVoronoi) / sizeof(std::uint32_t) <=
       image.words.size(); ++index) {
    if (image.words[index] != NODE_TEX_VORONOI) {
      continue;
    }
    SVMNodeTexVoronoi payload{};
    std::memcpy(&payload, image.words.data() + index + 1u, sizeof(payload));
    if (payload.dimensions == 3u && payload.feature == NODE_VORONOI_F1 &&
        payload.metric == NODE_VORONOI_EUCLIDEAN) {
      ++record_count;
      require(payload.distance_offset != SVM_STACK_INVALID &&
                  payload.color_offset != SVM_STACK_INVALID &&
                  payload.position_offset == SVM_STACK_INVALID &&
                  payload.w_out_offset == SVM_STACK_INVALID &&
                  payload.radius_offset == SVM_STACK_INVALID,
              "Voronoi SVM record did not preserve exact live-output offsets");
    }
  }
  require(record_count == 1u,
          "multi-output Voronoi did not emit exactly one Cycles SVM record");
}

} // namespace

int main() {
  test_multi_output_voronoi_is_shared();
  return EXIT_SUCCESS;
}
