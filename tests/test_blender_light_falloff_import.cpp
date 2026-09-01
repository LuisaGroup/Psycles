#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using psycles::adapter::load_blender_scene_bundle;
using psycles::compiler::SurfaceSvmValueOpcode;
using psycles::compiler::ValueOperation;

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
            ("psycles-light-falloff-import-" + std::to_string(nonce));
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
  "images":[],"node_groups":[],"materials":[],
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],
  "lights":[{
    "name":"Cycles Light Falloff Sun","type":"SUN",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "color":[1,1,1],"temperature_color":[1,1,1],
    "energy":1,"exposure":0,"angle":0.1,"normalize":false,
    "use_mis":true,"cast_shadow":true,
    "cycles_sync":{"shader_index":1,"object_index":1,"light_group":-1},
    "node_tree":{
      "name":"Cycles Light Falloff Sun",
      "surface_root":{"node":"Add 2","socket":"Shader"},
      "volume_root":null,"displacement_root":null,
      "links":[
        {"from_node":"Light Falloff","from_socket":"Quadratic",
         "to_node":"Quadratic Emission","to_socket":"Strength"},
        {"from_node":"Light Falloff","from_socket":"Linear",
         "to_node":"Linear Emission","to_socket":"Strength"},
        {"from_node":"Light Falloff","from_socket":"Constant",
         "to_node":"Constant Emission","to_socket":"Strength"},
        {"from_node":"Quadratic Emission","from_socket":"Emission",
         "to_node":"Add 1","to_socket":"Shader"},
        {"from_node":"Linear Emission","from_socket":"Emission",
         "to_node":"Add 1","to_socket":"Shader_001"},
        {"from_node":"Add 1","from_socket":"Shader",
         "to_node":"Add 2","to_socket":"Shader"},
        {"from_node":"Constant Emission","from_socket":"Emission",
         "to_node":"Add 2","to_socket":"Shader_001"}],
      "nodes":[
        {
          "name":"Light Falloff","type":"LIGHT_FALLOFF","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":6.0},
            {"identifier":"Smooth","name":"Smooth",
             "type":"NodeSocketFloat","linked":false,"default":1.5}],
          "outputs":[
            {"identifier":"Quadratic","name":"Quadratic",
             "type":"NodeSocketFloat","linked":true},
            {"identifier":"Linear","name":"Linear",
             "type":"NodeSocketFloat","linked":true},
            {"identifier":"Constant","name":"Constant",
             "type":"NodeSocketFloat","linked":true}],
          "properties":{},"special":{}
        },
        {
          "name":"Quadratic Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[1.0,0.72074449,0.52323443,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":true,"default":1.0}],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}],
          "properties":{},"special":{}
        },
        {
          "name":"Linear Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[1.0,0.72074449,0.52323443,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":true,"default":1.0}],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}],
          "properties":{},"special":{}
        },
        {
          "name":"Constant Emission","type":"EMISSION","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[1.0,0.72074449,0.52323443,1.0]},
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":true,"default":1.0}],
          "outputs":[{"identifier":"Emission","name":"Emission",
            "type":"NodeSocketShader","linked":true}],
          "properties":{},"special":{}
        },
        {
          "name":"Add 1","type":"ADD_SHADER","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Shader","name":"Shader",
             "type":"NodeSocketShader","linked":true},
            {"identifier":"Shader_001","name":"Shader",
             "type":"NodeSocketShader","linked":true}],
          "outputs":[{"identifier":"Shader","name":"Shader",
            "type":"NodeSocketShader","linked":true}]
        },
        {
          "name":"Add 2","type":"ADD_SHADER","mute":false,
          "internal_links":[],"properties":{},"special":{},
          "inputs":[
            {"identifier":"Shader","name":"Shader",
             "type":"NodeSocketShader","linked":true},
            {"identifier":"Shader_001","name":"Shader",
             "type":"NodeSocketShader","linked":true}],
          "outputs":[{"identifier":"Shader","name":"Shader",
            "type":"NodeSocketShader","linked":true}]
        }
      ]
    }
  }],
  "world":null,"world_environment":null
})JSON";
}

void test_light_falloff_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());
  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "Light Falloff SUN scene did not import");
  require(imported.scene->lights.size() == 1u,
          "imported SUN light is missing");
  const auto &light = imported.scene->lights.begin()->second;
  require(light.type == psycles::contract::LightType::distant &&
              light.shader.has_value(),
          "SUN light did not retain its authored shader graph");
  const auto &material = imported.scene->materials.at(*light.shader);

  const psycles::contract::ShaderNode *falloff = nullptr;
  auto falloff_count = 0u;
  auto light_path_count = 0u;
  auto ad_hoc_algebra_count = 0u;
  for (const auto &node : material.shader.nodes()) {
    if (node.type == psycles::compiler::node_type::light_falloff) {
      falloff = &node;
      ++falloff_count;
    }
    light_path_count +=
        node.type == psycles::compiler::node_type::light_path ? 1u : 0u;
    ad_hoc_algebra_count +=
        node.label.find("Squared Distance") != std::string::npos ||
        node.label.find("Smooth Factor") != std::string::npos ||
        node.label.find("Distance Falloff") != std::string::npos;
  }
  require(falloff_count == 1u && falloff != nullptr,
          "Light Falloff outputs did not share one semantic graph node");
  require(light_path_count == 0u,
          "Light Falloff import injected an artificial Light Path node");
  require(ad_hoc_algebra_count == 0u,
          "Light Falloff was decomposed into ad hoc graph algebra");

  auto registry = psycles::compiler::make_core_node_registry();
  const auto *schema =
      registry.find(psycles::compiler::node_type::light_falloff);
  require(schema != nullptr && schema->inputs.size() == 2u &&
              schema->inputs[0].name == "Strength" &&
              schema->inputs[1].name == "Smooth" &&
              schema->outputs.size() == 3u &&
              schema->outputs[0].name == "Quadratic" &&
              schema->outputs[1].name == "Linear" &&
              schema->outputs[2].name == "Constant" &&
              schema->outputs[0].type ==
                  psycles::contract::SocketType::floating &&
              schema->outputs[1].type ==
                  psycles::contract::SocketType::floating &&
              schema->outputs[2].type ==
                  psycles::contract::SocketType::floating,
          "Light Falloff schema lost one of its typed semantic outputs");
  psycles::compiler::ShaderCompiler compiler{std::move(registry)};
  const auto shader = compiler.compile(material.shader);
  require(shader.ok(), "Light Falloff graph did not validate");

  psycles::compiler::cycles_svm::AttributeIDMap attributes;
  psycles::compiler::cycles_svm::ImageIDMap images;
  const auto svm = psycles::compiler::cycles_svm::compile_shader(
      *shader.program, attributes, images,
      psycles::compiler::cycles_svm::ShaderCompileContext{
          .background = false});
  require(svm.valid, "Light Falloff graph did not compile to Cycles SVM");
  using Record = std::array<std::uint32_t, 5u>;
  constexpr auto expected_records = std::array{
      Record{0x00000049u, 0x00000000u, 0x40c00000u, 0x3fc00000u,
             0x00000000u},
      Record{0x00000049u, 0x00000001u, 0x40c00000u, 0x3fc00000u,
             0x00000001u},
      Record{0x00000049u, 0x00000002u, 0x40c00000u, 0x3fc00000u,
             0x00000002u}};
  std::array<Record, expected_records.size()> actual_records{};
  auto record_count = std::size_t{};
  for (auto index = std::size_t{}; index + 4u < svm.words.size(); ++index) {
    if (svm.words[index] != static_cast<std::uint32_t>(
                                psycles::compiler::cycles_svm::
                                    NODE_LIGHT_FALLOFF)) {
      continue;
    }
    require(record_count < actual_records.size(),
            "Light Falloff emitted too many SVM records");
    std::copy_n(svm.words.begin() + index, 5u,
                actual_records[record_count].begin());
    ++record_count;
  }
  require(record_count == expected_records.size() &&
              actual_records == expected_records,
          "imported Light Falloff payload differs from Cycles 5.2.1");

  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  require(surface.ok(), "Light Falloff graph did not lower");

  std::array<bool, 3u> falloff_types{};
  auto instruction_count = 0u;
  for (const auto &instruction : surface.program->value_instructions()) {
    if (instruction.operation != ValueOperation::light_falloff) {
      continue;
    }
    ++instruction_count;
    require(instruction.static_u0 < falloff_types.size(),
            "Light Falloff output selector is invalid");
    falloff_types[instruction.static_u0] = true;
    require(instruction.result_type ==
                    psycles::contract::SocketType::floating &&
                instruction.operands.size() ==
                    psycles::compiler::value_operand::light_falloff::count,
            "Light Falloff lost its named operand ABI");
    const auto immediate =
        psycles::compiler::make_surface_value_svm_immediate(
            instruction.operation, instruction.static_u0,
            instruction.static_u1);
    require(immediate == instruction.static_u0 &&
                psycles::compiler::surface_svm_value_opcode(
                    instruction.operation, immediate) ==
                    SurfaceSvmValueOpcode::light_falloff,
            "Light Falloff did not project to its exact SVM family");
  }
  require(instruction_count == 3u &&
              std::ranges::all_of(falloff_types,
                                  [](bool present) { return present; }),
          "Light Falloff reachability lost a connected Cycles output");
}

} // namespace

int main() {
  try {
    test_light_falloff_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
