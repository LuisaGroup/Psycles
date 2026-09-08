#include "cycles_svm_imported_word_fixture.h"

#include <psycles/compiler/surface_program.h>

namespace {
void check_import_boundary_and_private_prepass() {
  using namespace psycles;
  using test_support::require;
  test_support::Bundle bundle{"cycles_math_expand"};
  const auto imported = adapter::load_blender_scene_bundle(bundle.path);
  require(imported.ok(), "Math fixture import");
  compiler::ShaderCompiler compiler{compiler::make_core_node_registry()};
  for (const auto clamp : {false, true}) {
    const auto name = std::string{"minimal-clamp-"} + (clamp ? "1" : "0");
    const auto material = std::ranges::find_if(imported.scene->materials,
        [&](const auto &entry) { return entry.second.name == name; });
    require(material != imported.scene->materials.end(), "minimal original graph");
    auto clamped_math = 0u;
    for (const auto &node : material->second.shader.nodes()) {
      require(node.type != compiler::node_type::clamp_float,
              "import must retain Math.use_clamp until Cycles expansion");
      if (node.type == compiler::node_type::math) {
        clamped_math += std::get<bool>(node.properties.at("Clamp").value);
      }
    }
    require(clamped_math == unsigned(clamp), "Math clamp property was lost");
    const auto shader = compiler.compile(material->second.shader);
    require(shader.ok(), "minimal graph validation");
    const auto prepass = compiler::compile_surface_program(*shader.program);
    require(prepass.ok(), "private displacement representation");
    const auto instructions = prepass.program->value_instructions();
    const auto count = std::ranges::count_if(instructions, [](const auto &instruction) {
      return instruction.operation == compiler::ValueOperation::clamp01;
    });
    require(count == int(clamp), "private prepass discarded Math.use_clamp");
  }
}
} // namespace

int main() {
  try {
    psycles::test_support::check_imported_words("cycles_math_expand", 18);
    check_import_boundary_and_private_prepass();
    std::cout << "18 Math expansion/scheduling graphs match original Cycles\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
