#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

constexpr std::string_view ies_content = R"IES(IESNA:LM-63-2002
[TEST] PSYCLES TYPE C
TILT=NONE
1 1000 1 5 5 1 1 0 0 0 1 1 100
0 20 75 120 180
0 45 130 250 360
1 2 5 9 12
2 4 8 13 17
4 7 11 16 22
3 6 10 15 20
1 2 5 9 12
)IES";

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
            ("psycles-ies-import-" + std::to_string(nonce));
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
    "name":"IES Material","cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"IES Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Light Path","from_socket":"Ray Length",
         "to_node":"IES Light","to_socket":"Strength"},
        {"from_node":"IES Light","from_socket":"Fac",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"IES Light","type":"TEX_IES",
          "bl_idname":"ShaderNodeTexIES","mute":false,
          "internal_links":[],"properties":{},
          "special":{"ies":{"mode":"INTERNAL",
            "source":"IES Profile.ies","available":true,
            "content_bytes":[)JSON";
  for (auto index = std::size_t{}; index < ies_content.size(); ++index) {
    if (index != 0u) {
      scene << ',';
    }
    scene << static_cast<unsigned int>(
        static_cast<unsigned char>(ies_content[index]));
  }
  scene << R"JSON(]}},
          "inputs":[
            {"identifier":"Vector","name":"Vector",
             "type":"NodeSocketVector","linked":false,
             "default":[0,0,0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":true,"default":0.4}],
          "outputs":[
            {"identifier":"Fac","name":"Factor",
             "type":"NodeSocketFloat","linked":true,"default":0}]
        },
        {
          "name":"Light Path","type":"LIGHT_PATH","mute":false,
          "internal_links":[],"properties":{},"special":{},"inputs":[],
          "outputs":[
            {"identifier":"Ray Length","name":"Ray Length",
             "type":"NodeSocketFloat","linked":true,"default":0}]
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1}],
          "outputs":[
            {"identifier":"Emission","name":"Emission",
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

template <std::size_t N>
[[nodiscard]] std::size_t record_index(
    const ShaderImage &image,
    const std::array<std::uint32_t, N> &record) {
  const auto iter = std::search(image.words.begin(), image.words.end(),
                                record.begin(), record.end());
  require(iter != image.words.end(), "expected Cycles SVM record is absent");
  return static_cast<std::size_t>(iter - image.words.begin());
}

void test_ies_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "IES scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "IES Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "IES material is missing");

  const ShaderNode *ies = nullptr;
  auto ies_count = std::size_t{};
  auto baked_node_count = std::size_t{};
  for (const auto &node : material->shader.nodes()) {
    if (node.type == node_type::ies_light) {
      ies = &node;
      ++ies_count;
    }
    baked_node_count += node.label.find("IES Lookup") != std::string::npos ||
                         node.label.find("IES Bake") != std::string::npos;
  }
  require(ies_count == 1u && ies != nullptr && baked_node_count == 0u,
          "IES source was not preserved as one semantic node");
  const auto property = ies->properties.find("IES");
  require(property != ies->properties.end() &&
              property->second.type == SocketType::string,
          "IES semantic node lost its raw source property");
  const auto *content = std::get_if<std::string>(&property->second.value);
  require(content != nullptr && *content == ies_content,
          "IES raw bytes changed during Blender import");
  const auto vector_input = ies->inputs.find("Vector");
  require(vector_input == ies->inputs.end() ||
              vector_input->second.source == std::nullopt,
          "IES importer baked the implicit Incoming coordinate edge");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "imported IES graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  IESIDMap profiles;
  const auto image = compile_shader(
      *shader.program, attributes, images, profiles,
      ShaderCompileContext{.background = false});
  require(image.valid && profiles.slot_count() == 1u,
          "imported IES graph did not compile to one profile slot");

  constexpr auto record = std::array<std::uint32_t, 4u>{
      0x0000004au, 0x7fc00000u, 0x00000000u, 0x00000103u};
  require(std::search(image.words.begin(), image.words.end(), record.begin(),
                      record.end()) != image.words.end(),
          "imported NODE_IES record differs from Cycles 5.2.1");
  // Match complete records rather than individual opcode words: a payload may
  // legitimately have the same integer value as an opcode. These records are
  // the exact Cycles 5.2.1 oracle sequence for the implicit texture Incoming
  // coordinate, its WORLD-to-OBJECT normal transform, and stack Strength.
  constexpr auto geometry_record = std::array<std::uint32_t, 3u>{
      static_cast<std::uint32_t>(NODE_GEOMETRY),
      static_cast<std::uint32_t>(NODE_GEOM_I), 0x00000000u};
  constexpr auto transform_record = std::array<std::uint32_t, 8u>{
      static_cast<std::uint32_t>(NODE_VECTOR_TRANSFORM),
      static_cast<std::uint32_t>(NODE_VECTOR_TRANSFORM_TYPE_NORMAL),
      static_cast<std::uint32_t>(NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD),
      static_cast<std::uint32_t>(NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT),
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x00000003u};
  constexpr auto light_path_record = std::array<std::uint32_t, 3u>{
      static_cast<std::uint32_t>(NODE_LIGHT_PATH),
      static_cast<std::uint32_t>(NODE_LP_ray_length), 0x00000000u};
  const auto geometry = record_index(image, geometry_record);
  const auto transform = record_index(image, transform_record);
  const auto light_path = record_index(image, light_path_record);
  const auto ies_opcode = record_index(image, record);
  require(geometry < transform && transform < light_path &&
              light_path < ies_opcode,
          "implicit IES Incoming topology differs from Cycles 5.2.1");
}

} // namespace

int main() {
  try {
    test_ies_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
