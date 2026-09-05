#include "path_tracer_cycles_svm_scene.h"

#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace luisa::compute;
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::luisa_backend::detail;
namespace abi = psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error{std::string{message}};
  }
}

class TemporaryBundle final {
  std::filesystem::path _path;

public:
  TemporaryBundle() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = std::filesystem::temp_directory_path() /
            ("psycles-native-shader-input-" + std::to_string(nonce));
    require(std::filesystem::create_directory(_path),
            "cannot create unique shader input fixture directory");
    std::filesystem::copy_file(PSYCLES_CAMERA_SCENE, _path / "scene.json");
    std::ofstream geometry{_path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
    require(geometry.good(), "cannot write empty geometry fixture header");
  }
  ~TemporaryBundle() {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }
  const auto &path() const noexcept { return _path; }
};

MaterialId camera_material(const SceneSnapshot &snapshot) {
  for (const auto &[id, material] : snapshot.materials) {
    if (material.name == "Camera Data") {
      return id;
    }
  }
  throw std::runtime_error{"imported scene has no Camera Data material"};
}

ShaderGraph constant_shader() {
  ShaderGraph graph;
  const auto emission = graph.add_node(node_type::emission, "Stale emission");
  require(graph.set_input(emission, "Color",
                          SocketValue::color({8.0f, 0.0f, 0.0f})),
          "cannot construct stale shader fixture");
  graph.set_root(ShaderDomain::surface, OutputRef{emission, "Closure"});
  return graph;
}

std::shared_ptr<LuisaSceneData> empty_scene(Device &device) {
  auto scene = std::make_shared<LuisaSceneData>();
  scene->device = Device{device.impl_shared()};
  return scene;
}

void retain_stale_library(LuisaSceneData &scene,
                          const SceneSnapshot &snapshot) {
  auto stale = snapshot;
  stale.materials.at(camera_material(stale)).shader = constant_shader();
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto update = scene.materials.update(stale, frontend);
  require(update.committed, "stale legacy material fixture failed to compile");
}

std::vector<unsigned> oracle_words() {
  // Entire external Cycles 5.2.1 svm_nodes image, not a Psycles-generated
  // expected stream. Only the shader-tail relocation is normalized below.
  std::ifstream file{PSYCLES_CAMERA_WORDS};
  unsigned count{};
  file >> count >> std::hex;
  require(file.good() && count == 143u,
          "missing or malformed Cycles word oracle");
  std::vector<unsigned> words(count);
  for (auto &word : words) {
    file >> word;
  }
  require(!file.fail(), "truncated Cycles word oracle");
  std::string trailing;
  require(!(file >> trailing), "trailing Cycles word oracle data");
  return words;
}

void verify_camera_image(const CyclesSvmRuntime &runtime,
                         const SceneSnapshot &snapshot) {
  const auto &table = runtime.compilation.table;
  const auto expected = oracle_words();
  constexpr auto index = 5u;
  require(runtime.material_shader_indices.at(camera_material(snapshot)) ==
                  index &&
              table.shader_count == 6u,
          "native scene compilation changed the source shader identity");
  const auto actual_begin = table.words.at(index * 4u + 1u);
  const auto expected_begin = expected.at(index * 4u + 1u);
  require(table.words.size() - actual_begin == expected.size() - expected_begin,
          "Camera Data shader tail size differs from Cycles");
  for (auto entry = 1u; entry < 4u; ++entry) {
    require(
        table.words.at(index * 4u + entry) - actual_begin ==
            expected.at(index * 4u + entry) - expected_begin,
        "Camera Data surface/volume/displacement entry differs from Cycles");
  }
  bool same_words = true;
  for (auto offset = 0u; offset < table.words.size() - actual_begin; ++offset) {
    const auto actual_word = table.words[actual_begin + offset];
    const auto expected_word = expected[expected_begin + offset];
    if (actual_word != expected_word) {
      std::cerr << "Camera tail word " << offset << ": got 0x" << std::hex
                << actual_word << ", expected 0x" << expected_word << std::dec
                << '\n';
      same_words = false;
    }
  }
  require(same_words,
          "imported Camera Data word stream differs from Cycles 5.2.1");
  require(table.node_types_used[abi::NODE_CAMERA],
          "native shader omitted NODE_CAMERA");
}

void verify_device_upload(Device &device, CyclesSvmRuntime &runtime) {
  auto stream = device.create_stream();
  upload_cycles_svm_runtime(stream, runtime);
  // Use actual shader reads as well as host-image checks: this covers the
  // production word-buffer upload and native backend execution boundary.
  Kernel1D inspect = [](BufferUInt words, BufferUInt output) noexcept {
    const auto index = dispatch_x();
    output.write(index, words.read(index));
  };
  auto shader = device.compile(inspect);
  const auto &expected = runtime.compilation.table.words;
  auto output = device.create_buffer<unsigned>(expected.size());
  std::vector<unsigned> actual(expected.size());
  stream << shader(*runtime.word_buffer, output).dispatch(expected.size())
         << output.copy_to(actual.data()) << synchronize();
  require(actual == expected,
          "device shader image differs from the linked native stream");
}

bool run(Device &device, const SceneSnapshot &snapshot) {
  bool passed = true;
  const auto check = [&](std::string_view name, auto &&test) {
    try {
      test();
      std::cout << name << ": passed\n";
    } catch (const std::exception &error) {
      std::cerr << name << ": " << error.what() << '\n';
      passed = false;
    }
  };
  check("no legacy lowering prerequisite", [&] {
    auto scene = empty_scene(device);
    std::string diagnostic;
    auto runtime = build_cycles_svm_runtime(scene, snapshot, diagnostic);
    require(runtime != nullptr, diagnostic);
    require(scene->materials.materials().empty(),
            "native compilation mutated legacy materials");
    verify_camera_image(*runtime, snapshot);
    verify_device_upload(device, *runtime);
  });
  check("snapshot overrides stale legacy programs", [&] {
    auto scene = empty_scene(device);
    retain_stale_library(*scene, snapshot);
    std::string diagnostic;
    auto runtime = build_cycles_svm_runtime(scene, snapshot, diagnostic);
    require(runtime != nullptr, diagnostic);
    verify_camera_image(*runtime, snapshot);
  });
  check("stale cache cannot hide invalid source", [&] {
    auto scene = empty_scene(device);
    retain_stale_library(*scene, snapshot);
    auto invalid = snapshot;
    auto &graph = invalid.materials.at(camera_material(invalid)).shader;
    graph = ShaderGraph{};
    const auto unknown =
        graph.add_node("psycles.invalid_test_node", "Invalid native source");
    graph.set_root(ShaderDomain::surface, OutputRef{unknown, "Closure"});
    std::string diagnostic;
    require(!build_cycles_svm_runtime(scene, invalid, diagnostic) &&
                !diagnostic.empty(),
            "native compilation trusted stale legacy shader instead of "
            "validating source");
  });
  check("unreachable invalid material stays outside used-shader domain", [&] {
    auto scene = empty_scene(device);
    auto unused = snapshot;
    const MaterialId id{999u};
    auto material = unused.materials.at(camera_material(unused));
    material.name = "Unused invalid material";
    material.shader = ShaderGraph{};
    const auto unknown =
        material.shader.add_node("psycles.invalid_test_node", "Unused");
    material.shader.set_root(ShaderDomain::surface,
                             OutputRef{unknown, "Closure"});
    unused.materials.emplace(id, std::move(material));
    std::string diagnostic;
    auto runtime = build_cycles_svm_runtime(scene, unused, diagnostic);
    require(runtime != nullptr, diagnostic);
    require(!runtime->material_shader_indices.contains(id),
            "native compilation expanded the used-shader domain");
    verify_camera_image(*runtime, unused);
  });
  check("missing used material is diagnosed", [&] {
    auto scene = empty_scene(device);
    auto missing = snapshot;
    missing.materials.erase(camera_material(missing));
    std::string diagnostic;
    require(!build_cycles_svm_runtime(scene, missing, diagnostic) &&
                !diagnostic.empty(),
            "native compilation accepted an unavailable used material");
  });
  return passed;
}
} // namespace

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "fallback");
    TemporaryBundle fixture;
    // An optional original Blender export exercises the identical production
    // entry without replacing or modifying the source scene on disk.
    auto imported = psycles::adapter::load_blender_scene_bundle(
        argc > 2 ? std::filesystem::path{argv[2]} : fixture.path());
    for (const auto &diagnostic : imported.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    require(imported.ok(), "Camera Data export failed to import");
    if (imported.scene->geometries.empty()) {
      TriangleMeshDesc mesh;
      mesh.material_slots.emplace_back(camera_material(*imported.scene));
      imported.scene->geometries.emplace(GeometryId{1u}, std::move(mesh));
    }
    return run(device, *imported.scene) ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
