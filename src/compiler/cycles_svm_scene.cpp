#include <psycles/compiler/cycles_svm_scene.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace psycles::compiler::cycles_svm {
namespace {

constexpr auto jump_node_word_count =
    1u + sizeof(SVMNodeShaderJump) / sizeof(std::uint32_t);
static_assert(jump_node_word_count == 4u);

[[nodiscard]] ShaderTableImage reject(std::string diagnostic) {
  ShaderTableImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] bool valid_local_offset(std::uint32_t word,
                                      std::size_t word_count) noexcept {
  const auto offset = std::bit_cast<std::int32_t>(word);
  return offset >= static_cast<std::int32_t>(jump_node_word_count) &&
         static_cast<std::size_t>(offset) < word_count;
}

} // namespace

ShaderTableImage link_shader_table(std::span<const ShaderImage> shaders) {
  constexpr auto maximum_int_offset =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  constexpr auto maximum_shader_count =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());

  if (shaders.size() > maximum_shader_count ||
      shaders.size() > maximum_int_offset / jump_node_word_count) {
    return reject("Cycles SVM shader count overflows the global jump table");
  }

  auto global_word_count = shaders.size() * jump_node_word_count;
  auto peak_stack_usage = std::uint32_t{};
  std::array<bool, NODE_NUM> node_types_used{};
  node_types_used[NODE_SHADER_JUMP] = !shaders.empty();

  for (auto shader_index = std::size_t{}; shader_index < shaders.size();
       ++shader_index) {
    const auto &shader = shaders[shader_index];
    const auto prefix =
        "Cycles SVM shader " + std::to_string(shader_index) + ": ";
    if (!shader.valid) {
      return reject(prefix + (shader.diagnostic.empty()
                                  ? "local image is invalid"
                                  : shader.diagnostic));
    }
    if (shader.words.size() <= jump_node_word_count) {
      return reject(prefix + "local image has no shader tail");
    }
    if (shader.words.front() != static_cast<std::uint32_t>(NODE_SHADER_JUMP)) {
      return reject(prefix + "local image does not begin with ShaderJump");
    }
    if (!shader.node_types_used[NODE_SHADER_JUMP]) {
      return reject(prefix + "node usage omits its ShaderJump opcode");
    }
    for (auto entry = std::size_t{1u}; entry < jump_node_word_count; ++entry) {
      if (!valid_local_offset(shader.words[entry], shader.words.size())) {
        return reject(prefix + "local ShaderJump entry is outside its tail");
      }
    }
    if (shader.peak_stack_usage > SVM_STACK_SIZE) {
      return reject(prefix + "peak stack usage exceeds Cycles SVM capacity");
    }
    const auto tail_word_count = shader.words.size() - jump_node_word_count;
    if (tail_word_count > maximum_int_offset - global_word_count) {
      return reject("Cycles SVM global word offsets overflow int32");
    }
    global_word_count += tail_word_count;
    peak_stack_usage = std::max(peak_stack_usage, shader.peak_stack_usage);
    for (auto node = std::size_t{}; node < node_types_used.size(); ++node) {
      node_types_used[node] =
          node_types_used[node] || shader.node_types_used[node];
    }
  }

  ShaderTableImage result;
  result.valid = true;
  result.node_types_used = node_types_used;
  result.peak_stack_usage = peak_stack_usage;
  result.shader_count = static_cast<std::uint32_t>(shaders.size());
  result.words.resize(global_word_count);

  auto tail_base = shaders.size() * jump_node_word_count;
  for (auto shader_index = std::size_t{}; shader_index < shaders.size();
       ++shader_index) {
    const auto &shader = shaders[shader_index];
    const auto jump_base = shader_index * jump_node_word_count;
    result.words[jump_base] = static_cast<std::uint32_t>(NODE_SHADER_JUMP);
    for (auto entry = std::size_t{1u}; entry < jump_node_word_count; ++entry) {
      const auto local = static_cast<std::size_t>(
          std::bit_cast<std::int32_t>(shader.words[entry]));
      const auto global = tail_base + local - jump_node_word_count;
      result.words[jump_base + entry] = static_cast<std::uint32_t>(global);
    }
    const auto tail = std::span{shader.words}.subspan(jump_node_word_count);
    std::copy(tail.begin(), tail.end(), result.words.data() + tail_base);
    tail_base += tail.size();
  }
  return result;
}

} // namespace psycles::compiler::cycles_svm
