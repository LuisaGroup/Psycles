#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
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
            ("psycles-hair-svm-import-" + std::to_string(nonce));
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
    "name":"Standalone Hair Transmission SVM Oracle",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Standalone Hair Transmission SVM Oracle",
      "surface_root":{"node":"Hair","socket":"BSDF"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Tangent","from_socket":"Vector",
         "to_node":"Hair","to_socket":"Tangent"}],
      "nodes":[
        {
          "name":"Tangent","type":"COMBXYZ","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":false,"default":0.3},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":false,"default":0.4},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":0.0}],
          "outputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,"default":[0,0,0]}]
        },
        {
          "name":"Hair","type":"BSDF_HAIR","mute":false,
          "internal_links":[],"special":{},
          "properties":{"component":"Transmission"},
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.83,0.17,0.52,1]},
            {"identifier":"Offset","name":"Offset",
             "type":"NodeSocketFloatAngle","linked":false,"default":0.27},
            {"identifier":"RoughnessU","name":"RoughnessU",
             "type":"NodeSocketFloatFactor","linked":false,
             "default":0.0002},
            {"identifier":"RoughnessV","name":"RoughnessV",
             "type":"NodeSocketFloatFactor","linked":false,"default":1.4},
            {"identifier":"Tangent","name":"Tangent",
             "type":"NodeSocketVector","linked":true,"default":[0,0,0]},
            {"identifier":"Weight","name":"Weight",
             "type":"NodeSocketFloat","linked":false,"default":0}],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
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

void require_words(std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected) {
  if (actual.size() != expected.size()) {
    std::cerr << "imported Hair has " << actual.size() << " words, expected "
              << expected.size() << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << "  [" << std::dec << index << "] = 0x" << std::hex
                << actual[index] << '\n';
    }
    throw std::runtime_error{
        "imported Hair word count differs from Cycles 5.2.1"};
  }
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    require(actual[index] == expected[index],
            "imported Hair word stream differs from Cycles 5.2.1");
  }
}

void test_hair_svm_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Hair SVM scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Standalone Hair Transmission SVM Oracle") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Hair SVM material is missing");

  const ShaderNode *hair = nullptr;
  const ShaderNode *tangent = nullptr;
  for (const auto &node : material->shader.nodes()) {
    hair = node.type == node_type::hair_bsdf ? &node : hair;
    tangent = node.type == node_type::combine_xyz ? &node : tangent;
  }
  require(hair != nullptr && tangent != nullptr,
          "raw Hair or Tangent node was pre-baked");
  require(hair->properties.at("Component") ==
              SocketValue::string("Transmission"),
          "raw Hair component was altered during import");
  require(hair->inputs.at("Color").value ==
              SocketValue::color({0.83f, 0.17f, 0.52f}),
          "raw Hair Color was altered during import");
  require(hair->inputs.at("Offset").value == SocketValue::floating(0.27f),
          "raw Hair Offset was altered during import");
  require(hair->inputs.at("RoughnessU").value == SocketValue::floating(0.0002f),
          "raw Hair RoughnessU was altered during import");
  require(hair->inputs.at("RoughnessV").value == SocketValue::floating(1.4f),
          "raw Hair RoughnessV was altered during import");
  require(hair->inputs.at("Tangent").source ==
              OutputRef{.node = tangent->id, .socket = "Vector"},
          "raw Hair Tangent link was altered during import");
  require(!hair->inputs.contains("Weight"),
          "Blender-internal Hair Weight leaked into the graph");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "imported Hair graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);

  static constexpr std::array<std::uint32_t, 23u> expected{
      0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u, 0x00000013u,
      0x00000000u, 0x3e99999au, 0x3ecccccdu, 0x00000000u, 0x00000005u,
      0x3f547ae1u, 0x3e2e147bu, 0x3f051eb8u, 0x00000002u, 0x00000017u,
      0x000000ffu, 0x3951b717u, 0x3fb33333u, 0x3e8a3d71u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected);
}

} // namespace

int main() {
  try {
    test_hair_svm_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
