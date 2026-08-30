#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>

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
using psycles::compiler::LightFalloffType;
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
      "surface_root":{"node":"Emission","socket":"Emission"},
      "volume_root":null,"displacement_root":null,
      "links":[{
        "from_node":"Light Falloff","from_socket":"Quadratic",
        "to_node":"Emission","to_socket":"Strength"}],
      "nodes":[
        {
          "name":"Light Falloff","type":"LIGHT_FALLOFF","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Strength","name":"Strength",
             "type":"NodeSocketFloat","linked":false,"default":2.0},
            {"identifier":"Smooth","name":"Smooth",
             "type":"NodeSocketFloat","linked":false,"default":0.0}],
          "outputs":[
            {"identifier":"Quadratic","name":"Quadratic",
             "type":"NodeSocketFloat","linked":true},
            {"identifier":"Linear","name":"Linear",
             "type":"NodeSocketFloat","linked":false},
            {"identifier":"Constant","name":"Constant",
             "type":"NodeSocketFloat","linked":false}],
          "properties":{},"special":{}
        },
        {
          "name":"Emission","type":"EMISSION","mute":false,
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
  auto ad_hoc_algebra_count = 0u;
  for (const auto &node : material.shader.nodes()) {
    if (node.type == psycles::compiler::node_type::light_falloff) {
      falloff = &node;
      ++falloff_count;
    }
    ad_hoc_algebra_count +=
        node.label.find("Squared Distance") != std::string::npos ||
        node.label.find("Smooth Factor") != std::string::npos ||
        node.label.find("Distance Falloff") != std::string::npos;
  }
  require(falloff_count == 1u && falloff != nullptr,
          "Light Falloff was not preserved as one semantic graph node");
  require(ad_hoc_algebra_count == 0u,
          "Light Falloff was decomposed into ad hoc graph algebra");

  auto registry = psycles::compiler::make_core_node_registry();
  const auto *schema =
      registry.find(psycles::compiler::node_type::light_falloff);
  require(schema != nullptr && schema->outputs.size() == 3u &&
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
  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  require(surface.ok(), "Light Falloff graph did not lower");

  const psycles::compiler::ValueInstruction *quadratic = nullptr;
  auto instruction_count = 0u;
  for (const auto &instruction : surface.program->value_instructions()) {
    if (instruction.operation != ValueOperation::light_falloff) {
      continue;
    }
    ++instruction_count;
    if (instruction.static_u0 ==
        static_cast<std::uint64_t>(LightFalloffType::quadratic)) {
      quadratic = &instruction;
    }
  }
  // Only Quadratic is reachable from the authored surface root. The compact
  // program must preserve that exact semantic output while proving the two
  // unconnected siblings dead; retaining all three would inflate the SVM.
  require(instruction_count == 1u && quadratic != nullptr,
          "Light Falloff reachability did not retain exactly the connected output");
  require(quadratic->result_type ==
                  psycles::contract::SocketType::floating &&
              quadratic->operands.size() ==
                  psycles::compiler::value_operand::light_falloff::count,
          "Light Falloff lost its named operand ABI");
  const auto immediate = psycles::compiler::make_surface_value_svm_immediate(
      quadratic->operation, quadratic->static_u0, quadratic->static_u1);
  require(immediate ==
                  static_cast<std::uint32_t>(LightFalloffType::quadratic) &&
              psycles::compiler::surface_svm_value_opcode(
                  quadratic->operation, immediate) ==
                  SurfaceSvmValueOpcode::light_falloff,
          "Light Falloff did not project to its exact SVM family");
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
