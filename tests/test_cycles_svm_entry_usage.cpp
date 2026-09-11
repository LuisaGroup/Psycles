#include "cycles_svm_shared_closure_test_support.h"

#include <psycles/compiler/cycles_svm_scene.h>

#include <algorithm>
#include <iostream>

namespace {
using namespace psycles::test_support::shared_closure;
namespace abi = psycles::compiler::cycles_svm;

abi::ShaderEntryUsage empty_entry() {
  abi::ShaderEntryUsage result;
  result.node_types_used[abi::NODE_SHADER_JUMP] = true;
  result.node_types_used[abi::NODE_END] = true;
  result.noise_usage = abi::NoiseUsage::none();
  return result;
}

void verify_entry_union(const abi::ShaderImage &image) {
  abi::ShaderEntryUsage combined;
  for (const auto type : {abi::SHADER_TYPE_SURFACE, abi::SHADER_TYPE_VOLUME,
                          abi::SHADER_TYPE_DISPLACEMENT}) {
    const auto usage = image.usage_for(type);
    combined.peak_stack_usage =
        std::max(combined.peak_stack_usage, usage.peak_stack_usage);
    for (auto node = 0u; node < abi::NODE_NUM; ++node) {
      combined.node_types_used[node] |= usage.node_types_used[node];
    }
  }
  require(combined.node_types_used == image.node_types_used &&
              combined.peak_stack_usage == image.peak_stack_usage,
          "entry union lost emitted opcodes or stack allocations");
}
} // namespace

int main() {
  try {
    const auto oracle = read_words(PSYCLES_SHARED_CLOSURE_WORDS_ORACLE);
    std::array<abi::ShaderImage, 4u> images;
    for (auto i = 0u; i < images.size(); ++i) {
      auto &image = images[i] = compile(i);
      require(image.words == oracle[i], "entry analysis changed original Cycles words");
      verify_entry_union(image);
      const auto active = i < 2u ? abi::SHADER_TYPE_SURFACE : abi::SHADER_TYPE_VOLUME;
      const auto inactive = i < 2u ? abi::SHADER_TYPE_VOLUME : abi::SHADER_TYPE_SURFACE;
      require(image.usage_for(inactive) == empty_entry() &&
                  image.usage_for(abi::SHADER_TYPE_DISPLACEMENT) == empty_entry(),
              std::string{names[i]} + ": an END-only entry retained another entry's usage");
      abi::ShaderEntryUsage expected_active{
          image.node_types_used, image.peak_stack_usage};
      expected_active.noise_usage = abi::NoiseUsage::none();
      require(image.usage_for(active) == expected_active,
              "active entry lost shader emission facts");
      if ((i & 1u) != 0u) {
        require(image.usage_for(active).node_types_used[abi::NODE_JUMP_IF_ZERO] &&
                    image.usage_for(active).node_types_used[abi::NODE_JUMP_IF_ONE],
                "static analysis must include both data-dependent closure branches");
      }
    }
    const auto table = abi::link_shader_table(images);
    require(table.valid, table.diagnostic);
    require(table.usage_for(abi::SHADER_TYPE_DISPLACEMENT) == empty_entry(),
            "linker widened empty displacement entry");
    for (const auto type : {abi::SHADER_TYPE_SURFACE, abi::SHADER_TYPE_VOLUME}) {
      const auto usage = table.usage_for(type);
      require(usage.node_types_used[abi::NODE_CLOSURE_BSDF] ==
                  (type == abi::SHADER_TYPE_SURFACE) &&
                  usage.node_types_used[abi::NODE_CLOSURE_VOLUME] ==
                  (type == abi::SHADER_TYPE_VOLUME),
              "linker mixed surface/volume-only closure opcodes");
    }

    // No metadata must mean a conservative bound, not an empty program.
    auto external = images;
    for (auto &image : external) { image.entry_usage.reset(); }
    const auto unproven = abi::link_shader_table(external);
    require(unproven.valid && unproven.words == table.words,
            "external-image fallback changed the linked stream");
    abi::ShaderEntryUsage whole{table.node_types_used, table.peak_stack_usage};
    // This fixture contains no Noise opcode, so the conservative fallback has
    // no shape domain to retain.
    whole.noise_usage = abi::NoiseUsage::none();
    for (const auto type : {abi::SHADER_TYPE_SURFACE, abi::SHADER_TYPE_VOLUME,
                            abi::SHADER_TYPE_DISPLACEMENT}) {
      require(unproven.usage_for(type) == whole,
              "unproven external image must retain the global bound");
    }
    auto invalid = images;
    require(invalid[0].entry_usage.has_value(), "compiler omitted emission proof");
    (*invalid[0].entry_usage)[0].peak_stack_usage = invalid[0].peak_stack_usage + 1u;
    require(!abi::link_shader_table(invalid).valid,
            "linker accepted an entry bound exceeding its whole image");
    invalid = images;
    (*invalid[0].entry_usage)[0].node_types_used[abi::NODE_CLOSURE_BSDF] = false;
    require(!abi::link_shader_table(invalid).valid,
            "linker accepted an entry union omitting an emitted node");
    std::cout << "Original Cycles entry usage and conservative linking passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
