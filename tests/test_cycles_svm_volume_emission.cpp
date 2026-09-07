#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

struct Bundle {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("psycles-volume-emission-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  Bundle() {
    require(std::filesystem::create_directory(path), "cannot create test bundle");
    std::filesystem::copy_file(PSYCLES_VOLUME_EMISSION_SCENE, path / "scene.json");
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
    require(geometry.good(), "cannot write empty geometry header");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

void run() {
  Bundle bundle;
  auto imported = psycles::adapter::load_blender_scene_bundle(bundle.path);
  for (const auto &diagnostic : imported.diagnostics) {
    std::cerr << diagnostic.message << '\n';
  }
  require(imported.ok(), "external volume emission material did not import");
  const MaterialDesc *source = nullptr;
  for (const auto &[id, candidate] : imported.scene->materials) {
    if (candidate.name == "Volume Emission") { source = &candidate; }
  }
  require(source != nullptr, "external volume emission material is missing");
  const auto &material = *source;
  require(material.name == "Volume Emission" &&
              material.shader.root(ShaderDomain::volume).has_value(),
          "import changed the original output domains");
  const ShaderCompiler compiler{make_core_node_registry()};
  auto shader = compiler.compile(material.shader);
  require(shader.ok(), "volume source validation failed");
  AttributeIDMap attributes;
  auto image = compile_shader(*shader.program, attributes, ShaderCompileContext{});
  require(image.valid, image.diagnostic);

  // Exact external Cycles 5.2.1 shader tail; only the global jump-table
  // relocation is normalized. Neither this compiler nor its runtime produced
  // the expected words. The dynamic Light Path input prevents constant folding
  // from removing the domain-dependent Emission evaluation.
  std::ifstream oracle{PSYCLES_VOLUME_EMISSION_WORDS};
  std::size_t count{};
  oracle >> count >> std::hex;
  std::vector<std::uint32_t> expected(count);
  for (auto &word : expected) { oracle >> word; }
  require(!oracle.fail() && count == 23u, "invalid external word fixture");
  if (image.words != expected) {
    for (auto i = 0u; i < image.words.size(); ++i) {
      std::cerr << i << ": " << std::hex << image.words[i] << std::dec << '\n';
    }
    throw std::runtime_error{"volume emission stream differs from Cycles 5.2.1"};
  }
  require(image.node_types_used[NODE_CLOSURE_EMISSION] &&
              image.node_types_used[NODE_LIGHT_PATH] &&
              image.metadata.has_volume && !image.metadata.has_surface,
          "volume emission lost native opcode or domain metadata");
}
} // namespace

int main() {
  try {
    run();
    std::cout << "Native volume Emission matches the external Cycles word image\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
