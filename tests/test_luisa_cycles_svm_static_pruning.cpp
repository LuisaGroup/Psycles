#include "luisa_cycles_svm_test_kernel_globals.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
using Usage = std::array<bool, NODE_NUM>;

constexpr std::array<std::string_view, NODE_NUM> node_names{
#define SHADER_NODE_TYPE(name) #name,
#define SHADER_NODE_TYPE_DERIVATIVE(name) #name, #name "_DERIVATIVE",
#include <psycles/compiler/cycles_svm_node_types_template.h>
};

// Cycles 5.2.1 kernel/svm/svm.h guards every derivative dispatch body with
// IF_NOT_KERNEL_NODES_FEATURE(VOLUME). The list comes from its node catalog,
// independently of Psycles' dispatch implementation.
constexpr std::array derivative_nodes{
#define SHADER_NODE_TYPE(name)
#define SHADER_NODE_TYPE_DERIVATIVE(name) name##_DERIVATIVE,
#include <psycles/compiler/cycles_svm_node_types_template.h>
};
static_assert(derivative_nodes.size() == 16u);

// These are implementation gaps, not successful static pruning. Keep them
// explicit so this test cannot silently count a missing handler as a win.
constexpr bool implemented(ShaderNodeType node) noexcept {
  switch (node) {
  case NODE_CLOSURE_HOLDOUT:
  case NODE_CLOSURE_VOLUME:
  case NODE_VOLUME_COEFFICIENTS:
  case NODE_PRINCIPLED_VOLUME:
  case NODE_RADIAL_TILING:
  case NODE_BEVEL:
  case NODE_AMBIENT_OCCLUSION:
  case NODE_RAYCAST:
  case NODE_AOV_START:
  case NODE_AOV_COLOR:
  case NODE_AOV_VALUE:
  case NODE_SCENE_TIME:
  case NODE_NONE:
  case NODE_PAD1:
    return false;
  default:
    return true;
  }
}

void require(bool condition, std::string_view message) {
  if (!condition) { throw std::runtime_error{std::string{message}}; }
}

struct Dispatch {
  Usage cases{};
  std::array<std::size_t, NODE_NUM> body_statements{};
  unsigned count{};
};

// Inspect the recorded AST, before XIR/LLVM/SPIR-V optimizations. Never enter
// a case body: its nested switches and loops are not the opcode dispatcher.
void find_dispatch(const ScopeStmt *scope, unsigned loop_depth, Dispatch &out) {
  for (const auto *stmt : scope->statements()) {
    if (stmt->tag() == Statement::Tag::SCOPE) {
      find_dispatch(static_cast<const ScopeStmt *>(stmt), loop_depth, out);
    } else if (stmt->tag() == Statement::Tag::LOOP) {
      find_dispatch(static_cast<const LoopStmt *>(stmt)->body(), loop_depth + 1u, out);
    } else if (stmt->tag() == Statement::Tag::SWITCH && loop_depth == 1u) {
      ++out.count;
      const auto *dispatch = static_cast<const SwitchStmt *>(stmt);
      for (const auto *entry : dispatch->body()->statements()) {
        if (entry->tag() != Statement::Tag::SWITCH_CASE) { continue; }
        const auto *branch = static_cast<const SwitchCaseStmt *>(entry);
        require(branch->expression()->tag() == Expression::Tag::LITERAL,
                "opcode case must be a literal");
        const auto node = luisa::get<std::uint32_t>(
            static_cast<const LiteralExpr *>(branch->expression())->value().to_variant());
        require(node < NODE_NUM && !out.cases[node], "invalid or duplicate opcode case");
        out.cases[node] = true;
        // The DSL appends a break even to an empty case. A disabled Cycles
        // feature must retain that no-op case but record no payload reads,
        // stores, device ifs, or helper calls inside it.
        out.body_statements[node] = std::ranges::count_if(
            branch->body()->statements(), [](const Statement *body) {
              return body->tag() != Statement::Tag::BREAK &&
                     body->tag() != Statement::Tag::COMMENT;
            });
      }
    }
  }
}

Dispatch record(Usage used, ShaderType domain, std::uint32_t node_features) {
  const Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>, Buffer<luisa::uint4>>
      kernel = [=](BufferUInt words, BufferFloat4 values, BufferUInt4 status) {
        const auto identity = make_float4x4(1.0f);
        const svm::TransformState transforms{identity, identity, identity, identity};
        const psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
        svm::ClosurePool pool{4u};
        svm::ShaderData sd{
            make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, 1.0f),
            svm::primitive_triangle, 0u, 0u, 0u, 0u, 0.2f, 0.3f, 0u, 0.5f, 1.0f,
            0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f,
            make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f),
            identity, identity, 0u, &pool};
        const svm::PathState path{svm::path_ray_visibility_camera, 0u};
        svm::EvaluationResult result;
        // All scene features are enabled to isolate opcode-usage and node
        // feature-mask pruning from geometry/SSS service specialization.
        svm::eval_nodes(kg, words, domain, ~0u, node_features, used,
                        transforms, sd, path, result);
        values.write(0u, make_float4(sd.closure_emission_background, 1.0f));
        status.write(0u, make_uint4(result.status, result.final_offset, sd.flag, 0u));
      };
  Dispatch result;
  find_dispatch(kernel.function()->function().body(), 0u, result);
  require(result.count == 1u, "SVM must retain exactly one PC-loop opcode dispatcher");
  return result;
}

void test_every_opcode_usage_bit() {
  constexpr auto mask = svm::kernel_feature_node_mask_surface;
  const auto empty = record({}, SHADER_TYPE_SURFACE, mask);
  require(std::ranges::none_of(empty.cases, [](bool value) { return value; }),
          "an unused opcode still generated a device case");
  auto supported = 0u;
  for (auto id = 0u; id < NODE_NUM; ++id) {
    const auto node = static_cast<ShaderNodeType>(id);
    Usage used{};
    used[id] = true;
    const auto actual = record(used, SHADER_TYPE_SURFACE, mask);
    auto expected = used;
    expected[id] = implemented(node);
    require(actual.cases == expected,
            std::string{node_names[id]} + ": one-hot usage did not produce the exact case set");
    if (expected[id]) {
      ++supported;
      require(actual.body_statements[id] != 0u,
              std::string{node_names[id]} + ": enabled opcode body is missing");
    }
  }
  Usage all;
  all.fill(true);
  const auto full = record(all, SHADER_TYPE_SURFACE, mask);
  for (auto id = 0u; id < NODE_NUM; ++id) {
    require(full.cases[id] == implemented(static_cast<ShaderNodeType>(id)),
            std::string{node_names[id]} + ": all-enabled dispatch lost a handler");
  }
  std::cout << "Opcode usage: " << supported << " implemented handlers / " << NODE_NUM
            << " enum entries; every one-hot and empty mask verified before optimization\n";
}

void test_cycles_feature_guards() {
  struct Gate { ShaderNodeType node; std::uint32_t feature; bool inverse; };
  // The remaining whole-body guards in Cycles 5.2.1 kernel/svm/svm.h.
  constexpr std::array gates{
      Gate{NODE_CLOSURE_EMISSION, svm::kernel_feature_node_emission, false},
      Gate{NODE_CLOSURE_BACKGROUND, svm::kernel_feature_node_emission, false},
      Gate{NODE_EMISSION_WEIGHT, svm::kernel_feature_node_emission, false},
      Gate{NODE_CLOSURE_SET_NORMAL, svm::kernel_feature_node_bump, false},
      Gate{NODE_ENTER_BUMP_EVAL, svm::kernel_feature_node_bump_state, false},
      Gate{NODE_LEAVE_BUMP_EVAL, svm::kernel_feature_node_bump_state, false}};
  auto failures = 0u;
  const auto check = [&](Gate gate) {
    for (const auto domain : {SHADER_TYPE_SURFACE, SHADER_TYPE_VOLUME,
                              SHADER_TYPE_DISPLACEMENT}) {
      Usage used{};
      used[gate.node] = true;
      for (const auto enabled : {false, true}) {
        const auto mask = enabled != gate.inverse ? gate.feature : 0u;
        const auto actual = record(used, domain, mask);
        require(actual.cases == used, "feature guard must preserve the Cycles no-op case");
        const auto has_body = actual.body_statements[gate.node] != 0u;
        if (has_body != enabled) {
          ++failures;
          std::cerr << node_names[gate.node] << " domain=" << unsigned(domain)
                    << " mask=0x" << std::hex << mask << std::dec
                    << " body statements=" << actual.body_statements[gate.node]
                    << ", expected " << (enabled ? "nonempty" : "zero") << '\n';
        }
      }
    }
  };
  for (const auto gate : gates) { check(gate); }
  for (const auto node : derivative_nodes) {
    check({node, svm::kernel_feature_node_volume, true});
  }
  require(failures == 0u, "Cycles feature-disabled node bodies leaked into the recorded AST");
  std::cout << "Node feature guards: 22 families x 3 shader domains x on/off verified\n";
}
} // namespace

int main() {
  try {
    test_every_opcode_usage_bit();
    test_cycles_feature_guards();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
