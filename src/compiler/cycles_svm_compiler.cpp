#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

class Stack {
private:
  std::array<int, SVM_STACK_SIZE> _users{};
  std::uint32_t _peak{};
  bool _failed{};

public:
  void clear() noexcept {
    _users.fill(0);
  }

  [[nodiscard]] SVMStackOffset assign(std::uint32_t size) noexcept {
    auto offset = -1;
    for (auto index = std::uint32_t{}, unused = std::uint32_t{};
         index < SVM_STACK_SIZE; ++index) {
      if (_users[index] != 0) {
        unused = 0u;
      } else {
        ++unused;
      }
      if (unused == size) {
        offset = static_cast<int>(index + 1u - size);
        _peak = std::max(_peak, index + 1u);
        while (static_cast<int>(index) >= offset) {
          _users[index] = 1;
          if (index == 0u) {
            break;
          }
          --index;
        }
        return static_cast<SVMStackOffset>(offset);
      }
    }
    _failed = true;
    return 0u;
  }

  void release(SVMStackOffset offset, std::uint32_t size) noexcept {
    const auto begin = static_cast<std::uint32_t>(offset);
    if (begin + size > SVM_STACK_SIZE) {
      std::abort();
    }
    for (auto index = std::uint32_t{}; index < size; ++index) {
      auto &users = _users[begin + index];
      if (users <= 0) {
        std::abort();
      }
      --users;
    }
  }

  [[nodiscard]] bool failed() const noexcept { return _failed; }
  [[nodiscard]] std::uint32_t peak() const noexcept { return _peak; }
};

[[nodiscard]] const contract::InputBinding *input_binding(
    const contract::ShaderNode &node, std::string_view name) noexcept {
  const auto iter = node.inputs.find(name);
  return iter == node.inputs.end() ? nullptr : &iter->second;
}

template<typename T>
[[nodiscard]] std::optional<T> literal(const contract::InputBinding *binding,
                                       contract::SocketType type) noexcept {
  if (binding == nullptr || binding->source || !binding->value ||
      binding->value->type != type) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<T>(&binding->value->value)) {
    return *value;
  }
  return std::nullopt;
}

class Compiler {
private:
  const ShaderProgram &_shader;
  BytecodeBuilder _stream;
  Stack _stack;
  std::string _diagnostic;

private:
  [[nodiscard]] bool reject(std::string diagnostic) {
    if (_diagnostic.empty()) {
      _diagnostic = std::move(diagnostic);
    }
    return false;
  }

  [[nodiscard]] const contract::ShaderNode *find(contract::NodeId id) const noexcept {
    return _shader.graph().find(id);
  }

  [[nodiscard]] bool compile_default_normal(SVMStackOffset &offset) {
    offset = _stack.assign(3u);
    if (_stack.failed()) {
      return reject("Shader graph: out of SVM stack space");
    }
    static_cast<void>(_stream.add_node(
        NODE_GEOMETRY,
        SVMNodeGeometry{.geom_type = NODE_GEOM_N,
                        .bump_offset = NODE_BUMP_OFFSET_CENTER,
                        .store_derivatives = 0u,
                        .out_offset = offset,
                        .bump_filter_width = 0.0f}));
    return true;
  }

  [[nodiscard]] bool compile_diffuse(const contract::ShaderNode &node) {
    const auto *color_binding = input_binding(node, "Color");
    const auto *roughness_binding = input_binding(node, "Roughness");
    const auto *normal_binding = input_binding(node, "Normal");
    if (color_binding == nullptr || roughness_binding == nullptr ||
        normal_binding == nullptr) {
      return reject("Cycles Diffuse BSDF inputs are incomplete");
    }
    if (color_binding->source || roughness_binding->source ||
        normal_binding->source) {
      return reject("linked Diffuse BSDF inputs are not migrated yet");
    }
    const auto color = literal<Vec3f>(color_binding, contract::SocketType::color);
    const auto roughness =
        literal<float>(roughness_binding, contract::SocketType::floating);
    if (!color || !roughness) {
      return reject("Cycles Diffuse BSDF input literals are ill typed");
    }

    // ShaderGraph::default_inputs connects one shared Geometry::Normal output
    // to every unlinked LINK_NORMAL socket before SVM compilation.
    SVMStackOffset normal_offset = SVM_STACK_INVALID;
    if (!compile_default_normal(normal_offset)) {
      return false;
    }

    static_cast<void>(_stream.add_node(
        NODE_CLOSURE_SET_WEIGHT,
        SVMNodeClosureSetWeight{.rgb = packed_float3{
                                    color->x, color->y, color->z}}));
    static_cast<void>(_stream.add_node(
        NODE_CLOSURE_BSDF,
        SVMNodeClosureBsdf{.closure_type = CLOSURE_BSDF_DIFFUSE_ID,
                           .mix_weight_offset = SVM_STACK_INVALID,
                           ._pad = {0u, 0u, 0u}}));
    _stream.add_node_data(SVMNodeDiffuseBsdfData{
        .color = input_float3(color->x, color->y, color->z),
        .roughness = input_float(*roughness),
        .normal_offset = normal_offset,
        ._pad = {0u, 0u, 0u}});
    _stack.release(normal_offset, 3u);
    return true;
  }

  [[nodiscard]] bool compile_surface() {
    const auto &root = _shader.graph().root(contract::ShaderDomain::surface);
    if (root) {
      const auto *node = find(root->node);
      if (node == nullptr) {
        return reject("surface root refers to an absent shader node");
      }
      if (node->type == node_type::diffuse_bsdf) {
        if (root->socket != "Closure") {
          return reject("Diffuse BSDF root does not select Closure");
        }
        if (!compile_diffuse(*node)) {
          return false;
        }
      } else {
        return reject("surface Cycles SVM node family is not migrated: " +
                      node->type);
      }
    }
    static_cast<void>(_stream.add_node(NODE_END));
    return true;
  }

  [[nodiscard]] bool compile_empty_domain(contract::ShaderDomain domain,
                                          std::string_view name) {
    if (_shader.graph().root(domain)) {
      return reject(std::string{name} +
                    " Cycles SVM node family is not migrated");
    }
    static_cast<void>(_stream.add_node(NODE_END));
    return true;
  }

public:
  explicit Compiler(const ShaderProgram &shader) noexcept : _shader{shader} {}

  [[nodiscard]] ShaderImage compile() {
    const auto jump_index = _stream.add_node(
        NODE_SHADER_JUMP, SVMNodeShaderJump{0, 0, 0});
    if (jump_index != 0u) {
      std::abort();
    }

    const auto surface_offset = _stream.size();
    _stack.clear();
    if (!compile_surface()) {
      return finish(false);
    }

    const auto volume_offset = _stream.size();
    _stack.clear();
    if (!compile_empty_domain(contract::ShaderDomain::volume, "volume")) {
      return finish(false);
    }

    const auto displacement_offset = _stream.size();
    _stack.clear();
    if (!compile_empty_domain(contract::ShaderDomain::displacement,
                              "displacement")) {
      return finish(false);
    }

    if (surface_offset > static_cast<std::size_t>(
                             std::numeric_limits<std::int32_t>::max()) ||
        volume_offset > static_cast<std::size_t>(
                            std::numeric_limits<std::int32_t>::max()) ||
        displacement_offset > static_cast<std::size_t>(
                                  std::numeric_limits<std::int32_t>::max())) {
      return reject_and_finish("Cycles SVM shader offsets overflow int32");
    }
    _stream.set_word(1u, static_cast<std::uint32_t>(surface_offset));
    _stream.set_word(2u, static_cast<std::uint32_t>(volume_offset));
    _stream.set_word(3u, static_cast<std::uint32_t>(displacement_offset));
    return finish(true);
  }

private:
  [[nodiscard]] ShaderImage reject_and_finish(std::string diagnostic) {
    static_cast<void>(reject(std::move(diagnostic)));
    return finish(false);
  }

  [[nodiscard]] ShaderImage finish(bool valid) {
    ShaderImage result;
    result.valid = valid && _diagnostic.empty();
    result.diagnostic = std::move(_diagnostic);
    result.words.assign(_stream.words().begin(), _stream.words().end());
    result.node_types_used = _stream.node_types_used();
    result.peak_stack_usage = _stack.peak();
    return result;
  }
};

} // namespace

ShaderImage compile_shader(const ShaderProgram &shader) {
  return Compiler{shader}.compile();
}

} // namespace psycles::compiler::cycles_svm
