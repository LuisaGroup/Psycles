#pragma once

#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace psycles::test_support {
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;

void require(bool value, std::string_view message) {
  if (!value) {
    throw std::runtime_error{std::string{message}};
  }
}

struct Bundle {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("psycles-hidden-socket-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  explicit Bundle(std::string_view stem) {
    require(std::filesystem::create_directory(path), "cannot create bundle");
    std::filesystem::copy_file(std::filesystem::path{PSYCLES_SVM_IMPORT_FIXTURE_DIR} /
                                   (std::string{stem} + "_scene.json"),
                               path / "scene.json");
    std::ofstream geometry{path / "geometry.bin", std::ios::binary};
    geometry.write("PSYGEO2\0", 8);
    require(geometry.good(), "cannot write empty geometry header");
  }
  ~Bundle() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

void check_imported_words(std::string_view stem, unsigned expected_count,
                          bool allow_constant_emission_roundoff = false) {
  Bundle bundle{stem};
  const auto imported = psycles::adapter::load_blender_scene_bundle(bundle.path);
  require(imported.ok(), "hidden socket fixture import failed");
  ShaderCompiler compiler{make_core_node_registry()};
  std::ifstream oracle{std::filesystem::path{PSYCLES_SVM_IMPORT_FIXTURE_DIR} /
                       (std::string{stem} + "_words.txt")};
  unsigned count{};
  oracle >> count;
  require(bool(oracle) && count == expected_count, "invalid external fixture count");
  unsigned failures = 0;
  unsigned constant_literal_fields = 0;
  for (unsigned test = 0; test < count; ++test) {
    unsigned shader_index{}, word_count{};
    std::string name;
    oracle >> std::dec >> shader_index >> std::quoted(name) >> word_count;
    require(bool(oracle) && word_count < 1024, "invalid external shader header");
    std::vector<std::uint32_t> expected(word_count);
    for (auto &word : expected) {
      oracle >> std::hex >> word;
    }
    require(bool(oracle), "truncated external word image");
    const auto material =
        std::ranges::find_if(imported.scene->materials, [&](const auto &entry) {
          return entry.second.cycles_shader_index == shader_index &&
                 entry.second.name == name;
        });
    require(material != imported.scene->materials.end(), "missing source shader");
    const auto shader = compiler.compile(material->second.shader);
    require(shader.ok(), "invalid normalized graph");
    AttributeIDMap attributes;
    const auto image = compile_shader(*shader.program, attributes, {});
    require(image.valid, image.diagnostic);
    auto comparable = image.words;
    // Only the three typed NODE_CLOSURE_SET_WEIGHT float literals in constant
    // emission probes admit ordinary host math roundoff. Every opcode, PC,
    // stack address, and dynamic-input record stays exact.
    if (allow_constant_emission_roundoff && name.starts_with("constant-") &&
        comparable.size() == 13 && expected.size() == 13 &&
        comparable[4] == NODE_CLOSURE_SET_WEIGHT &&
        expected[4] == NODE_CLOSURE_SET_WEIGHT) {
      for (unsigned i = 5; i < 8; ++i) {
        ++constant_literal_fields;
        const float actual = std::bit_cast<float>(comparable[i]);
        const float reference = std::bit_cast<float>(expected[i]);
        if (std::isnan(actual) && std::isnan(reference)) {
          comparable[i] = expected[i]; // NaN payload bits are not a topology.
        } else if (std::isfinite(actual) && std::isfinite(reference) &&
                   std::abs(actual - reference) <=
                       2e-6f * std::max(1.0f, std::abs(reference))) {
          comparable[i] = expected[i];
        }
      }
    }
    if (comparable != expected) {
      ++failures;
      const auto mismatch = std::mismatch(image.words.begin(), image.words.end(),
                                          expected.begin(), expected.end());
      std::cerr << name << ": Psycles=" << image.words.size()
                << " Cycles=" << expected.size() << " words, first difference="
                << std::distance(image.words.begin(), mismatch.first) << '\n';
    }
  }
  oracle >> std::ws;
  require(oracle.eof(), "trailing oracle fixture data");
  require(!allow_constant_emission_roundoff || constant_literal_fields >= 90,
          "constant-emission tolerance did not inspect the typed literal fields");
  require(failures == 0, "imported graph differs from original Cycles 5.2.1");
}
} // namespace psycles::test_support
