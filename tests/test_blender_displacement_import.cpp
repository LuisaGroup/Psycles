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
            ("psycles-displacement-import-" + std::to_string(nonce));
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
  "images":[],
  "node_groups":[],
  "materials":[{
    "name":"Object Displacement",
    "displacement_method":"BUMP",
    "cycles_sync":{"shader_index":4},
    "node_tree":{
      "name":"Object Displacement",
      "surface_root":{"node":"Diffuse","socket":"BSDF"},
      "volume_root":null,
      "displacement_root":{"node":"Displacement","socket":"Displacement"},
      "links":[],
      "nodes":[
        {
          "name":"Diffuse","type":"BSDF_DIFFUSE","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Color","name":"Color",
             "type":"NodeSocketColor","linked":false,
             "default":[0.8,0.8,0.8,1.0]},
            {"identifier":"Roughness","name":"Roughness",
             "type":"NodeSocketFloat","linked":false,"default":0.0},
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[{"identifier":"BSDF","name":"BSDF",
            "type":"NodeSocketShader","linked":true}],
          "properties":{},"special":{}
        },
        {
          "name":"Displacement","type":"DISPLACEMENT","mute":false,
          "internal_links":[],
          "inputs":[
            {"identifier":"Height","name":"Height",
             "type":"NodeSocketFloat","linked":false,"default":0.75},
            {"identifier":"Midlevel","name":"Midlevel",
             "type":"NodeSocketFloat","linked":false,"default":0.5},
            {"identifier":"Scale","name":"Scale",
             "type":"NodeSocketFloat","linked":false,"default":0.1},
            {"identifier":"Normal","name":"Normal",
             "type":"NodeSocketVector","linked":false,
             "default":[0.0,0.0,0.0]}
          ],
          "outputs":[{"identifier":"Displacement","name":"Displacement",
            "type":"NodeSocketVector","linked":true,
            "default":[0.0,0.0,0.0]}],
          "properties":{"space":"OBJECT"},"special":{}
        }
      ]
    }
  }],
  "render":{"width":16,"height":16,"percentage":100,"cycles":{}},
  "camera":{"name":"Camera","type":"PERSP",
    "transform":[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
    "clip_start":0.01,"clip_end":100.0},
  "geometries":[],"curve_geometries":[],"instances":[],"lights":[],
  "world":null,"world_environment":null
})JSON";
}

void test_object_displacement_import() {
  TemporaryDirectory temporary;
  write_scene_bundle(temporary.path());

  const auto imported = load_blender_scene_bundle(temporary.path());
  require(imported.ok(), "OBJECT Displacement scene did not import");
  const psycles::contract::MaterialDesc *material = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    static_cast<void>(id);
    if (candidate.name == "Object Displacement") {
      material = &candidate;
      break;
    }
  }
  require(material != nullptr, "OBJECT Displacement material is missing");
  require(material->shader
                  .root(psycles::contract::ShaderDomain::surface_normal)
                  .has_value() &&
              !material->shader
                   .root(psycles::contract::ShaderDomain::displacement)
                   .has_value(),
          "BUMP displacement did not become a surface-normal program");

  const psycles::contract::ShaderNode *displacement = nullptr;
  auto displacement_count = 0u;
  auto ad_hoc_algebra_count = 0u;
  for (const auto &node : material->shader.nodes()) {
    if (node.type == psycles::compiler::node_type::displacement) {
      displacement = &node;
      ++displacement_count;
    }
    ad_hoc_algebra_count +=
        (node.type == psycles::compiler::node_type::subtract_float &&
         node.label == "Displacement / Height Offset") ||
        (node.type == psycles::compiler::node_type::multiply_float &&
         node.label == "Displacement / Scaled Height") ||
        (node.type == psycles::compiler::node_type::vector_math &&
         node.label == "Displacement");
  }
  require(displacement_count == 1u && displacement != nullptr,
          "scalar Displacement was not preserved as one semantic node");
  require(ad_hoc_algebra_count == 0u,
          "scalar Displacement was decomposed into ad hoc graph algebra");
  require(std::get<std::string>(
              displacement->properties.at("Space").value) == "OBJECT" &&
              !std::get<bool>(
                  displacement->properties.at("NormalLinked").value),
          "OBJECT space or unlinked-normal semantics were erased");

  psycles::compiler::ShaderCompiler compiler{
      psycles::compiler::make_core_node_registry()};
  const auto shader = compiler.compile(material->shader);
  require(shader.ok(), "OBJECT Displacement graph did not validate");
  const auto surface =
      psycles::compiler::compile_surface_program(*shader.program);
  require(surface.ok(), "OBJECT Displacement graph did not lower");

  const psycles::compiler::ValueInstruction *instruction = nullptr;
  auto instruction_count = 0u;
  for (const auto &candidate : surface.program->value_instructions()) {
    if (candidate.operation == ValueOperation::displacement) {
      instruction = &candidate;
      ++instruction_count;
    }
  }
  require(instruction_count == 1u && instruction != nullptr,
          "OBJECT Displacement did not survive as one value instruction");
  require(instruction->result_type == psycles::contract::SocketType::vector &&
              instruction->operands.size() ==
                  psycles::compiler::value_operand::displacement::count &&
              instruction->static_u0 ==
                  psycles::compiler::displacement_object_space,
          "OBJECT Displacement lost its typed operands or configuration");
  const auto immediate = psycles::compiler::make_surface_value_svm_immediate(
      instruction->operation, instruction->static_u0, instruction->static_u1);
  require(immediate == psycles::compiler::displacement_object_space &&
              psycles::compiler::surface_svm_value_opcode(
                  instruction->operation, immediate) ==
                  SurfaceSvmValueOpcode::displacement,
          "OBJECT Displacement did not project to its exact SVM family");
}

} // namespace

int main() {
  try {
    test_object_displacement_import();
  } catch (const std::exception &exception) {
    std::cerr << "test failure: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
