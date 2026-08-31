#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
            ("psycles-rgb-ramp-import-" + std::to_string(nonce));
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
  scene << std::setprecision(9);
  scene << R"JSON({
  "schema":"psycles.blender-scene.v1",
  "images":[],"node_groups":[],
  "materials":[{
    "name":"RGB Ramp Material",
    "cycles_sync":{"shader_index":5},
    "node_tree":{
      "name":"RGB Ramp Material",
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Gradient","from_socket":"Fac",
         "to_node":"Ramp","to_socket":"Fac"},
        {"from_node":"Ramp","from_socket":"Color",
         "to_node":"Emission","to_socket":"Color"}],
      "nodes":[
        {
          "name":"Gradient","type":"TEX_GRADIENT","mute":false,
          "internal_links":[],
          "inputs":[{"identifier":"Vector","name":"Vector",
            "type":"NodeSocketVector","linked":false,
            "default":[0,0,0]}],
          "outputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.8,0.8,0.8,1]},
            {"identifier":"Fac","name":"Factor",
             "type":"NodeSocketFloat","linked":true,"default":0}],
          "properties":{"gradient_type":"LINEAR"},"special":{}
        },
        {
          "name":"Ramp","type":"VALTORGB","mute":false,
          "internal_links":[],
          "inputs":[{"identifier":"Fac","name":"Factor",
            "type":"NodeSocketFloatFactor","linked":true,
            "default":0.5}],
          "outputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[0.8,0.8,0.8,1]},
            {"identifier":"Alpha","name":"Alpha",
             "type":"NodeSocketFloat","linked":false,"default":0}],
          "properties":{},
          "special":{"color_ramp":{
            "color_mode":"HSV","interpolation":"CARDINAL",
            "hue_interpolation":"FAR",
            "samples":[
)JSON";
  for (auto index = std::uint32_t{}; index <= 256u; ++index) {
    const auto r = static_cast<float>(index) / 256.0f;
    if (index != 0u) {
      scene << ",\n";
    }
    scene << '[' << r << ",0.25," << (1.0f - r) << ",0.5]";
  }
  scene << R"JSON(
            ],
            "elements":[
              {"position":0,"color":[1,0,0,1]},
              {"position":1,"color":[0,1,0,1]}]
          }}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":true,
             "default":[1,1,1,1]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":1}],
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
    "clip_start":0.01,"clip_end":100},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

void test_sampled_table_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "RGB Ramp scene did not import");

  const MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "RGB Ramp Material") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "RGB Ramp material is missing");

  const ShaderNode *ramp = nullptr;
  for (const auto &node : material->shader.nodes()) {
    if (node.type == node_type::color_ramp) {
      require(ramp == nullptr, "RGB Ramp import duplicated the semantic node");
      ramp = &node;
    }
  }
  require(ramp != nullptr, "RGB Ramp import lost the semantic node");
  const auto sampled = ramp->properties.find("Sampled");
  const auto interpolation = ramp->properties.find("Interpolation");
  const auto table = ramp->properties.find("Table");
  require(sampled != ramp->properties.end() &&
              std::get<bool>(sampled->second.value),
          "RGB Ramp import did not mark the Cycles table as sampled");
  require(interpolation != ramp->properties.end() &&
              std::get<std::string>(interpolation->second.value) == "CARDINAL",
          "RGB Ramp import lost the original interpolation enum");
  require(table != ramp->properties.end() &&
              std::count(std::get<std::string>(table->second.value).begin(),
                         std::get<std::string>(table->second.value).end(),
                         ';') == 256,
          "RGB Ramp import did not preserve all 257 Cycles samples");

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(material->shader);
  require(shader.ok(), "RGB Ramp graph did not validate");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());
  const auto opcode = std::find(image.words.begin(), image.words.end(),
                                static_cast<std::uint32_t>(NODE_RGB_RAMP));
  require(opcode != image.words.end(), "imported RGB Ramp opcode is missing");
  const auto begin = static_cast<std::size_t>(opcode - image.words.begin());
  require(begin + 4u + 257u * 4u <= image.words.size(),
          "imported RGB Ramp table is truncated");
  require(image.words[begin + 1u] == 257u,
          "imported RGB Ramp table size differs from Cycles");
  const auto packed = image.words[begin + 3u];
  require((packed & 0xffu) == 1u &&
              ((packed >> 8u) & 0xffu) != SVM_STACK_INVALID &&
              ((packed >> 16u) & 0xffu) == SVM_STACK_INVALID,
          "imported RGB Ramp header lost interpolation or output liveness");

  const auto sample_word = [&](std::uint32_t sample, std::uint32_t component) {
    return image.words[begin + 4u + sample * 4u + component];
  };
  require(sample_word(0u, 0u) == std::bit_cast<std::uint32_t>(0.0f) &&
              sample_word(0u, 1u) == std::bit_cast<std::uint32_t>(0.25f) &&
              sample_word(0u, 2u) == std::bit_cast<std::uint32_t>(1.0f) &&
              sample_word(128u, 0u) == std::bit_cast<std::uint32_t>(0.5f) &&
              sample_word(128u, 2u) == std::bit_cast<std::uint32_t>(0.5f) &&
              sample_word(256u, 0u) == std::bit_cast<std::uint32_t>(1.0f) &&
              sample_word(256u, 2u) == std::bit_cast<std::uint32_t>(0.0f) &&
              sample_word(256u, 3u) == std::bit_cast<std::uint32_t>(0.5f),
          "imported RGB Ramp payload used control points or changed samples");
}

} // namespace

int main() {
  try {
    test_sampled_table_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
