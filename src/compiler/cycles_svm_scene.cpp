#include <psycles/compiler/cycles_svm_scene.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
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

[[nodiscard]] ShaderImage inert_shader() {
  ShaderImage image;
  image.valid = true;
  image.words = {NODE_SHADER_JUMP, 4u, 5u, 6u, NODE_END, NODE_END, NODE_END};
  image.node_types_used[NODE_SHADER_JUMP] = true;
  image.node_types_used[NODE_END] = true;
  return image;
}

[[nodiscard]] bool same_local_shader(const ShaderImage &lhs,
                                     const ShaderImage &rhs) noexcept {
  return lhs.words == rhs.words && lhs.node_types_used == rhs.node_types_used &&
         lhs.attribute_requests == rhs.attribute_requests &&
         lhs.peak_stack_usage == rhs.peak_stack_usage;
}

[[nodiscard]] std::uint64_t
resolve_attribute_request(const AttributeRequest &request,
                          AttributeIDMap &attribute_ids) {
  return request.standard != ATTR_STD_NONE
             ? AttributeIDMap::get_attribute_id(request.standard)
             : attribute_ids.get_attribute_id(request.name);
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

CompiledShaderTable
compile_shader_table(std::span<const ShaderTableCompileUnit> shaders) {
  CompiledShaderTable result;
  if (shaders.empty()) {
    result.table = link_shader_table({});
    return result;
  }

  auto maximum_index = std::uint32_t{};
  for (const auto &unit : shaders) {
    maximum_index = std::max(maximum_index, unit.shader_index);
    if (unit.shader == nullptr) {
      result.table =
          reject("Cycles SVM shader " + std::to_string(unit.shader_index) +
                 ": source ShaderProgram is absent");
      return result;
    }
  }
  constexpr auto maximum_size = std::numeric_limits<std::size_t>::max();
  if (static_cast<std::uint64_t>(maximum_index) + 1u > maximum_size) {
    result.table = reject("Cycles SVM shader index overflows host size_t");
    return result;
  }

  AttributeIDMap attribute_ids;
  ImageIDMap image_ids;
  IESIDMap ies_ids;
  std::vector<std::optional<ShaderImage>> occupied(
      static_cast<std::size_t>(maximum_index) + 1u);
  for (const auto &unit : shaders) {
    auto image = compile_shader(*unit.shader, attribute_ids, image_ids, ies_ids,
                                unit.context);
    if (!image.valid) {
      result.table =
          reject("Cycles SVM shader " + std::to_string(unit.shader_index) +
                 ": " + image.diagnostic);
      return result;
    }
    auto &slot = occupied[unit.shader_index];
    if (slot && !same_local_shader(*slot, image)) {
      result.table = reject("Cycles SVM shader index " +
                            std::to_string(unit.shader_index) +
                            " names distinct local images");
      return result;
    }
    if (!slot) {
      slot.emplace(std::move(image));
    }
  }

  std::vector<ShaderImage> local;
  local.reserve(occupied.size());
  for (auto &slot : occupied) {
    local.emplace_back(slot ? std::move(*slot) : inert_shader());
  }
  result.shader_node_types_used.reserve(local.size());
  result.shader_attribute_ids_used.reserve(local.size());
  for (const auto &shader : local) {
    result.shader_node_types_used.emplace_back(shader.node_types_used);
    auto &ids = result.shader_attribute_ids_used.emplace_back();
    ids.reserve(shader.attribute_requests.size());
    for (const auto &request : shader.attribute_requests) {
      ids.emplace_back(resolve_attribute_request(request, attribute_ids));
    }
    std::ranges::sort(ids);
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }
  result.table = link_shader_table(local);
  if (result.table.valid) {
    result.named_attributes = attribute_ids.bindings();
    result.images = image_ids.bindings();
    result.ies = ies_ids.packed_data();
  }
  return result;
}

} // namespace psycles::compiler::cycles_svm
