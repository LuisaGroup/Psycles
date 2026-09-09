#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

struct Bundle {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("psycles-holdout-volume-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  Bundle() {
    require(std::filesystem::create_directory(path), "cannot create original volume bundle");
    const auto data = std::filesystem::path{PSYCLES_HOLDOUT_VOLUME_FIXTURES};
    std::filesystem::copy_file(data / "scene.json", path / "scene.json");
    std::ifstream encoded{data / "geometry.txt"};
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    std::size_t expected{}, count{};
    encoded >> expected;
    require(bool(encoded) && expected > 0u, "missing original volume geometry length");
    unsigned byte{};
    while (encoded >> std::hex >> byte) {
      require(byte <= 255u, "invalid original geometry byte");
      geometry.put(static_cast<char>(byte));
      ++count;
    }
    require(encoded.eof() && count == expected && geometry.good(), "truncated original volume geometry");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  Bundle(const Bundle &) = delete;
  Bundle &operator=(const Bundle &) = delete;
};
} // namespace

int main() {
  try {
    Bundle bundle;
    const auto imported = psycles::adapter::load_blender_scene_bundle(bundle.path);
    for (const auto &diagnostic : imported.diagnostics) { std::cerr << diagnostic.message << '\n'; }
    require(imported.ok(), "original Holdout volume scene import failed");
    const ShaderCompiler frontend{make_core_node_registry()};
    std::ifstream oracle{std::filesystem::path{PSYCLES_HOLDOUT_VOLUME_FIXTURES} / "words.txt"};
    unsigned count{};
    oracle >> count;
    require(bool(oracle) && count == 3u, "original Holdout volume image count changed");
    auto passed = true;
    for (auto probe = 0u; probe < count; ++probe) {
      std::string name;
      unsigned shader_index{}, word_count{};
      oracle >> std::quoted(name) >> std::dec >> shader_index >> word_count;
      require(bool(oracle) && word_count >= 7u && word_count < 1024u, "invalid original volume image");
      std::vector<std::uint32_t> expected(word_count);
      for (auto &word : expected) { oracle >> std::hex >> word; }
      require(bool(oracle), "truncated original Holdout volume image");
      try {
        const auto material = std::ranges::find_if(imported.scene->materials, [&](const auto &entry) {
          return entry.second.cycles_shader_index == shader_index;
        });
        require(material != imported.scene->materials.end() && material->second.name == name,
                "original Holdout volume shader identity changed");
        const auto normalized = frontend.compile(material->second.shader);
        for (const auto &diagnostic : normalized.diagnostics) { std::cerr << diagnostic.message << '\n'; }
        require(normalized.ok(), "Holdout volume normalization failed");
        AttributeIDMap attributes;
        ImageIDMap images;
        const auto image = compile_shader(*normalized.program, attributes, images,
            ShaderCompileContext{.background = false,
                                 .displacement_method = material->second.displacement_method});
        require(image.valid, image.diagnostic);
        auto mismatches = std::size_t{image.words.size() != expected.size()};
        for (auto i = std::size_t{}; i < std::min(image.words.size(), expected.size()); ++i) {
          if (image.words[i] != expected[i]) {
            if (mismatches < 12u) {
              std::cerr << name << " word " << i << " actual=" << std::hex << image.words[i]
                        << " original=" << expected[i] << std::dec << '\n';
            }
            ++mismatches;
          }
        }
        std::cout << name << " words=" << image.words.size() << " original=" << expected.size()
                  << " mismatches=" << mismatches << '\n';
        require(mismatches == 0u, "Holdout volume words differ from original Cycles");

        const auto volume = name != "holdout-only-volume";
        const auto surface = name == "holdout-shared-domains";
        // Native optimize_volume_output/clean prune the pure Holdout volume.
        // The valid mixed/shared graph contains ONE Holdout closure node:
        // both contract-domain views must project to that same native node.
        require(image.metadata.has_volume_connected && image.metadata.has_volume == volume &&
                    image.metadata.has_surface == surface,
                "Holdout volume cleanup or domain metadata differs from Cycles");
        require(image.metadata.num_closures == static_cast<unsigned>(volume),
                "Holdout canonical projection changed native closure budget");
        require(image.entry_usage.has_value(), "missing static entry usage");
        require(image.node_types_used[NODE_CLOSURE_HOLDOUT] == volume &&
                    image.usage_for(SHADER_TYPE_VOLUME).node_types_used[NODE_CLOSURE_HOLDOUT] == volume &&
                    image.usage_for(SHADER_TYPE_SURFACE).node_types_used[NODE_CLOSURE_HOLDOUT] == surface &&
                    !image.usage_for(SHADER_TYPE_DISPLACEMENT).node_types_used[NODE_CLOSURE_HOLDOUT],
                "Holdout per-domain static node pruning changed");
      } catch (const std::exception &error) {
        std::cerr << name << ": " << error.what() << '\n';
        passed = false;
      }
    }
    std::string extra;
    require(!(oracle >> extra) && oracle.eof(), "extra original Holdout volume words");
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
