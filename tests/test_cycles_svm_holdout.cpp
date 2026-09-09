#include "cycles_holdout_fixture.h"

#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace f = psycles::test_support::holdout;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;

void check_entry_pruning(const ShaderImage &image, const std::vector<unsigned> &original,
                         std::string_view name) {
  // The immutable authoring captures contain a Holdout node in all twelve
  // target materials, but only these four connect it to the surface root.
  // Object holdout is object state, not a reason to generate a Holdout case.
  constexpr std::array connected{
      std::string_view{"node-holdout"}, std::string_view{"node-mixed-emission"},
      std::string_view{"node-mixed-transparent"}, std::string_view{"node-dynamic-holdout"}};
  const bool surface_holdout = std::find(connected.begin(), connected.end(), name) != connected.end();
  f::require(image.entry_usage.has_value(), "missing emitted per-entry specialization proof");
  f::require(image.node_types_used[NODE_CLOSURE_HOLDOUT] == surface_holdout,
             "whole-image Holdout mask includes a disconnected node or omits a connected node");
  f::require(image.usage_for(SHADER_TYPE_SURFACE).node_types_used[NODE_CLOSURE_HOLDOUT] == surface_holdout,
             "surface Holdout mask includes a disconnected node or omits a connected node");

  // Every captured volume/displacement entry contains only NODE_END. The
  // entry still executes the common ShaderJump, so exactly these two cases
  // belong in its JIT mask; no surface closure/value node may leak across.
  std::array<bool, NODE_NUM> inert_nodes{};
  inert_nodes[NODE_SHADER_JUMP] = true;
  inert_nodes[NODE_END] = true;
  for (const auto type : {SHADER_TYPE_VOLUME, SHADER_TYPE_DISPLACEMENT}) {
    const auto target = original.at(1u + static_cast<unsigned>(type));
    f::require(target < original.size() && original[target] == NODE_END,
               "original fixture no longer has an inert volume/displacement entry");
    f::require(image.usage_for(type).node_types_used == inert_nodes,
               "inert entry retains unreachable surface node cases");
  }

  std::array<bool, NODE_NUM> entry_union{};
  for (const auto &entry : *image.entry_usage) {
    for (std::size_t node = 0; node < entry_union.size(); ++node) {
      entry_union[node] = entry_union[node] || entry.node_types_used[node];
    }
  }
  f::require(entry_union == image.node_types_used, "entry masks do not cover the emitted whole-image mask");
  // PSYSVM52 v1 and these captures' cycles_sync metadata do not record
  // get_num_closures/max_closures. Do not invent a captured closure-budget
  // expectation from words; allocation state has its own original-GPU test.
}

int main() {
  bool passed = true;
  for (const auto name : f::cases) {
    try {
      f::Bundle bundle{name};
      const auto imported = psycles::adapter::load_blender_scene_bundle(bundle.path);
      for (const auto &d : imported.diagnostics) { std::cerr << d.message << '\n'; }
      f::require(imported.ok(), "original holdout scene import failed");
      const psycles::contract::MaterialDesc *material = nullptr;
      for (const auto &[id, candidate] : imported.scene->materials) {
        if (candidate.name == name) { material = &candidate; }
      }
      f::require(material != nullptr, "original holdout material missing");
      const ShaderCompiler frontend{make_core_node_registry()};
      const auto normalized = frontend.compile(material->shader);
      for (const auto &d : normalized.diagnostics) { std::cerr << d.message << '\n'; }
      f::require(normalized.ok(), "holdout graph normalization failed");
      AttributeIDMap attributes;
      ImageIDMap images;
      const auto image = compile_shader(*normalized.program, attributes, images,
          ShaderCompileContext{.background = false, .displacement_method = material->displacement_method});
      f::require(image.valid, image.diagnostic);
      std::ifstream oracle{f::fixture(name, "-words.txt")};
      std::size_t count{};
      oracle >> count;
      f::require(bool(oracle) && count >= 4u, "invalid original word count");
      std::vector<unsigned> expected(count);
      for (auto &word : expected) { oracle >> std::hex >> word; }
      f::require(bool(oracle), "truncated original word stream");
      std::string extra;
      f::require(!(oracle >> extra) && oracle.eof(), "extra original words");
      check_entry_pruning(image, expected, name);
      std::size_t mismatches = image.words.size() != expected.size();
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
                << " mismatches=" << mismatches
                << " surface_holdout=" << image.usage_for(SHADER_TYPE_SURFACE).node_types_used[NODE_CLOSURE_HOLDOUT]
                << " inert_entries_pruned=2\n";
      passed = passed && mismatches == 0u;
    } catch (const std::exception &error) {
      std::cerr << name << ": " << error.what() << '\n';
      passed = false;
    }
  }
  return passed ? 0 : 1;
}
