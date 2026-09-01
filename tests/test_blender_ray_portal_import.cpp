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
            ("psycles-ray-portal-import-" + std::to_string(nonce));
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
    "name":"Standalone Authored Ray Portal SVM Oracle",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"Standalone Authored Ray Portal SVM Oracle",
      "surface_root":{"node":"Ray Portal","socket":"BSDF"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Position","from_socket":"Vector",
         "to_node":"Ray Portal","to_socket":"Position"}],
      "nodes":[
        {
          "name":"Position","type":"COMBXYZ","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"X","name":"X","type":"NodeSocketFloat",
             "linked":false,"default":1.25},
            {"identifier":"Y","name":"Y","type":"NodeSocketFloat",
             "linked":false,"default":-0.75},
            {"identifier":"Z","name":"Z","type":"NodeSocketFloat",
             "linked":false,"default":2.5}],
          "outputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":true,
            "default":[0,0,0]}]
        },
        {
          "name":"Ray Portal","type":"BSDF_RAY_PORTAL","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.83,0.17,0.52,1]},
            {"identifier":"Position","name":"Position",
             "type":"NodeSocketVector","linked":true,
             "default":[0,0,0]},
            {"identifier":"Direction","name":"Direction",
             "type":"NodeSocketVector","linked":false,
             "default":[0.3,-0.4,1.2]},
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
  require(actual.size() == expected.size(),
          "imported Ray Portal word count differs from Cycles 5.2.1");
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    require(actual[index] == expected[index],
            "imported Ray Portal word stream differs from Cycles 5.2.1");
  }
}

void test_ray_portal_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Ray Portal scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Standalone Authored Ray Portal SVM Oracle") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "Ray Portal material is missing");

  const ShaderNode *portal = nullptr;
  const ShaderNode *position = nullptr;
  for (const auto &node : material->shader.nodes()) {
    portal = node.type == node_type::ray_portal_bsdf ? &node : portal;
    position = node.type == node_type::combine_xyz ? &node : position;
  }
  require(portal != nullptr && position != nullptr,
          "raw Ray Portal or Position node was pre-baked");
  require(portal->inputs.contains("Color") &&
              portal->inputs.contains("Position") &&
              portal->inputs.contains("Direction") &&
              !portal->inputs.contains("Weight") &&
              portal->inputs.at("Position").source ==
                  OutputRef{.node = position->id, .socket = "Vector"} &&
              portal->inputs.at("Color").value ==
                  SocketValue::color({0.83f, 0.17f, 0.52f}) &&
              portal->inputs.at("Direction").value ==
                  SocketValue::vector({0.3f, -0.4f, 1.2f}),
          "raw Ray Portal sockets were altered during import");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "imported Ray Portal graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);

  static constexpr std::array<std::uint32_t, 23u> expected{
      0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u, 0x00000013u,
      0x00000000u, 0x3fa00000u, 0xbf400000u, 0x40200000u, 0x00000005u,
      0x3f547ae1u, 0x3e2e147bu, 0x3f051eb8u, 0x00000002u, 0x0000001du,
      0x000000ffu, 0x3e99999au, 0xbecccccdu, 0x3f99999au, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected);
}

} // namespace

int main() {
  try {
    test_ray_portal_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
