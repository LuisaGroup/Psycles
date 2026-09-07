#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

struct Bundle {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("psycles-analytic-sky-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  explicit Bundle(const char *scene) {
    require(std::filesystem::create_directory(path), "cannot create bundle");
    std::filesystem::copy_file(scene, path / "scene.json");
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
    require(geometry.good(), "cannot write geometry header");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

void run(const char *scene, const char *words) {
  Bundle bundle{scene};
  const auto imported =
      psycles::adapter::load_blender_scene_bundle(bundle.path);
  require(imported.ok() && imported.scene->world_shader,
          "external sky world did not import");
  const auto &material =
      imported.scene->materials.at(*imported.scene->world_shader);
  const ShaderCompiler compiler{make_core_node_registry()};
  const auto shader = compiler.compile(material.shader);
  require(shader.ok(), "external world graph is invalid");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*shader.program, attributes, images,
                                    ShaderCompileContext{.background = true});
  require(image.valid, image.diagnostic);
  std::ifstream oracle{words};
  unsigned count{};
  oracle >> count >> std::hex;
  require(bool(oracle) && count == 52u && image.words.size() == count,
          "analytic sky shader length differs from the external Cycles image");
  for (auto i = 0u; i < count; ++i) {
    unsigned expected{};
    oracle >> expected;
    require(bool(oracle), "truncated external sky words");
    const auto actual = image.words[i];
    // Only the typed 32-float coefficient payload admits 1 ULP. Opcodes,
    // offsets, node order, and the rest of the complete image stay exact.
    const auto ulps = actual > expected ? actual - expected : expected - actual;
    if (actual != expected && !(i >= 10u && i < 42u && ulps <= 1u)) {
      std::cerr << "word=" << i << std::hex << " actual=" << actual
                << " Cycles=" << expected << std::dec << '\n';
      throw std::runtime_error{"analytic sky stream differs from Cycles 5.2.1"};
    }
  }
  require(image.peak_stack_usage == 6u && images.bindings().empty() &&
              image.node_types_used[NODE_GEOMETRY] &&
              image.node_types_used[NODE_TEX_SKY] &&
              image.node_types_used[NODE_CLOSURE_BACKGROUND],
          "analytic sky changed stack, resources, or opcode reachability");
}
} // namespace

int main() {
  try {
    run(PSYCLES_HOSEK_SCENE, PSYCLES_HOSEK_WORDS);
    run(PSYCLES_PREETHAM_SCENE, PSYCLES_PREETHAM_WORDS);
    std::cout
        << "Imported analytic sky matches the external Cycles word image\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
