#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/luisa/cycles_svm.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;

[[nodiscard]] ShaderImage compile_dynamic_math() {
  ShaderGraph graph;
  const auto geometry_a = graph.add_node(node_type::geometry, "Geometry A");
  const auto geometry_b = graph.add_node(node_type::geometry, "Geometry B");
  const auto math = graph.add_node(node_type::math, "Dynamic Add");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.set_property(math, "Operation",
                         contract::SocketValue::string("ADD")) &&
      graph.connect({geometry_a, "Backfacing"}, math, "A") &&
      graph.connect({geometry_b, "Backfacing"}, math, "B") &&
      graph.set_input(emission, "Color",
                      contract::SocketValue::color(
                          {0.21f, 0.47f, 0.83f})) &&
      graph.connect({math, "Value"}, emission, "Strength");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic Math SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic Math graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_mix() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto transparent =
      graph.add_node(node_type::transparent_bsdf, "Transparent BSDF");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto mix = graph.add_node(node_type::mix_closure, "Dynamic Mix");
  const auto configured =
      graph.set_input(transparent, "Color",
                      contract::SocketValue::color(
                          {0.75f, 0.9f, 0.6f})) &&
      graph.set_input(emission, "Color",
                      contract::SocketValue::color(
                          {0.85f, 0.08f, 0.03f})) &&
      graph.set_input(emission, "Strength",
                      contract::SocketValue::floating(1.2f)) &&
      graph.connect({geometry, "Backfacing"}, mix, "Factor") &&
      graph.connect({transparent, "Closure"}, mix, "A") &&
      graph.connect({emission, "Closure"}, mix, "B");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic Mix SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic Mix graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] auto make_interpreter_kernel(
    std::array<bool, NODE_NUM> node_types_used) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::uint4>>{
      [node_types_used](BufferUInt words, BufferFloat4 floating_output,
                        BufferUInt4 integer_output) noexcept {
        const UInt index = dispatch_x();
        const UInt shader_flags =
            select(0u, device_svm::shader_data_backfacing, index != 0u);
        device_svm::ShaderData shader_data{
            make_float3(1.0f, 2.0f, 3.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, 1.0f),
            make_float3(0.0f, 0.0f, -1.0f),
            0u,
            shader_flags,
            0.2f,
            0.3f,
            4.0f};
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            words, SHADER_TYPE_SURFACE,
            device_svm::kernel_feature_node_emission |
                device_svm::kernel_feature_node_light_path,
            node_types_used, shader_data, path_state, result);
        floating_output.write(
            index,
            make_float4(shader_data.closure_emission_background,
                        result.closure_weight.x));
        integer_output.write(
            index,
            make_uint4(result.status, result.final_offset, shader_data.flag,
                       0u));
      }};
}

struct InterpreterShape final : StmtVisitor {
  std::uint32_t loop_depth{};
  std::uint32_t switch_depth{};
  std::uint32_t loops{};
  std::uint32_t primary_switches{};

  void visit(const BreakStmt *) override {}
  void visit(const ContinueStmt *) override {}
  void visit(const ReturnStmt *) override {}
  void visit(const ScopeStmt *stmt) override {
    for (const auto *statement : stmt->statements()) {
      statement->accept(*this);
    }
  }
  void visit(const IfStmt *stmt) override {
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
  }
  void visit(const LoopStmt *stmt) override {
    ++loops;
    ++loop_depth;
    stmt->body()->accept(*this);
    --loop_depth;
  }
  void visit(const ExprStmt *) override {}
  void visit(const SwitchStmt *stmt) override {
    if (loop_depth == 1u && switch_depth == 0u) {
      ++primary_switches;
    }
    ++switch_depth;
    stmt->body()->accept(*this);
    --switch_depth;
  }
  void visit(const SwitchCaseStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const SwitchDefaultStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const AssignStmt *) override {}
  void visit(const ForStmt *stmt) override { stmt->body()->accept(*this); }
  void visit(const CommentStmt *) override {}
  void visit(const RayQueryStmt *stmt) override {
    stmt->on_triangle_candidate()->accept(*this);
    stmt->on_procedural_candidate()->accept(*this);
  }
  void visit(const SuspendStmt *) override {}
  void visit(const AutoDiffStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const PrintStmt *) override {}
  void visit(const DebugBreakStmt *) override {}
};

[[nodiscard]] bool approximately_equal(float actual, float expected) {
  return std::abs(actual - expected) <= 2.0e-6f;
}

[[nodiscard]] bool require_float3(const luisa::float4 &actual,
                                  const luisa::float3 &expected,
                                  std::string_view label) {
  if (approximately_equal(actual.x, expected.x) &&
      approximately_equal(actual.y, expected.y) &&
      approximately_equal(actual.z, expected.z)) {
    return true;
  }
  std::cerr << label << " mismatch: (" << actual.x << ", " << actual.y
            << ", " << actual.z << ") != (" << expected.x << ", "
            << expected.y << ", " << expected.z << ")\n";
  return false;
}

void run_image(Device &device, Stream &stream, const ShaderImage &image,
               std::array<luisa::float4, 2u> &floating,
               std::array<luisa::uint4, 2u> &integer) {
  const auto kernel = make_interpreter_kernel(image.node_types_used);
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto word_buffer = device.create_buffer<std::uint32_t>(image.words.size());
  auto floating_buffer = device.create_buffer<luisa::float4>(floating.size());
  auto integer_buffer = device.create_buffer<luisa::uint4>(integer.size());
  stream << word_buffer.copy_from(luisa::span{image.words})
         << shader(word_buffer, floating_buffer, integer_buffer).dispatch(2u)
         << floating_buffer.copy_to(luisa::span{floating})
         << integer_buffer.copy_to(luisa::span{integer}) << synchronize();
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto math_image = compile_dynamic_math();
  const auto mix_image = compile_dynamic_mix();

  const auto shape_kernel = make_interpreter_kernel(math_image.node_types_used);
  InterpreterShape shape;
  shape_kernel.function()->function().body()->accept(shape);
  if (shape.loops != 1u || shape.primary_switches != 1u) {
    std::cerr << "Cycles SVM AST must contain exactly one PC loop and one "
                 "primary opcode switch; loops="
              << shape.loops << ", primary switches=" << shape.primary_switches
              << '\n';
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();

  std::array<luisa::float4, 2u> floating{};
  std::array<luisa::uint4, 2u> integer{};
  run_image(device, stream, math_image, floating, integer);
  const auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  if (!require_float3(floating[0], {0.0f, 0.0f, 0.0f},
                      "front-facing Math emission") ||
      !require_float3(floating[1], {0.42f, 0.94f, 1.66f},
                      "back-facing Math emission") ||
      integer[0].x != ended || integer[1].x != ended ||
      integer[0].y != 21u || integer[1].y != 21u ||
      integer[0].z != device_svm::shader_data_emission ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic Math Cycles SVM state mismatch on " << backend
              << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run_image(device, stream, mix_image, floating, integer);
  if (!require_float3(floating[0], {0.0f, 0.0f, 0.0f},
                      "front-facing Mix emission") ||
      !require_float3(floating[1], {1.02f, 0.096f, 0.036f},
                      "back-facing Mix emission") ||
      !approximately_equal(floating[0].w, 0.75f) ||
      !approximately_equal(floating[1].w, 1.02f) ||
      integer[0].x != ended || integer[1].x != ended ||
      integer[0].y != 32u || integer[1].y != 32u ||
      integer[0].z != 0u ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic closure jump Cycles SVM state mismatch on "
              << backend << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
