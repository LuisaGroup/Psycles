#include "graph_surface_internal.h"

#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/luisa/cycles_voronoi.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] cycles_voronoi::Configuration
decode_svm_configuration(compiler::ValueOperation operation,
                         std::uint32_t immediate) noexcept {
  auto output = compiler::VoronoiOutput::distance;
  switch (operation) {
  case compiler::ValueOperation::voronoi_color:
    output = compiler::VoronoiOutput::color;
    break;
  case compiler::ValueOperation::voronoi_position:
    output = compiler::VoronoiOutput::position;
    break;
  case compiler::ValueOperation::voronoi_w:
    output = compiler::VoronoiOutput::w;
    break;
  case compiler::ValueOperation::voronoi_radius:
    output = compiler::VoronoiOutput::radius;
    break;
  case compiler::ValueOperation::voronoi_distance:
    break;
  default:
    std::abort();
  }
  return {.dimensions =
              compiler::decode_surface_value_voronoi_dimensions(immediate),
          .feature = compiler::decode_surface_value_voronoi_feature(immediate),
          .metric = compiler::decode_surface_value_voronoi_metric(immediate),
          .output = output,
          .normalize =
              compiler::decode_surface_value_voronoi_normalize(immediate)};
}

class VoronoiValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression
    evaluate(ValueEvaluationContext &context) const noexcept override {
        const auto &instruction = this->instruction();
        const auto vector_value = vector(
            instruction.operand(operand::voronoi::vector), context.result);
        const auto w =
            scalar(instruction.operand(operand::voronoi::w), context.result);
        const auto scale = scalar(instruction.operand(operand::voronoi::scale),
                                  context.result);
        const auto detail = scalar(
            instruction.operand(operand::voronoi::detail), context.result);
        const auto roughness = scalar(
            instruction.operand(operand::voronoi::roughness), context.result);
        const auto lacunarity = scalar(
            instruction.operand(operand::voronoi::lacunarity), context.result);
        const auto smoothness = scalar(
            instruction.operand(operand::voronoi::smoothness), context.result);
        const auto exponent = scalar(
            instruction.operand(operand::voronoi::exponent), context.result);
        const auto randomness = scalar(
            instruction.operand(operand::voronoi::randomness), context.result);
        Float4 value = make_float4(0.0f);
        if (context.svm_immediate_override != nullptr) {
          std::vector<std::uint16_t> active;
          active.reserve(context.svm_immediate_domain.size());
          for (const auto encoded : context.svm_immediate_domain) {
            if ((encoded &
                 ~compiler::surface_value_voronoi_configuration_mask) != 0u) {
              std::abort();
            }
            if (std::find(active.begin(), active.end(), encoded) ==
                active.end()) {
              active.emplace_back(encoded);
            }
            cycles_voronoi::prepare(
                decode_svm_configuration(instruction.operation, encoded));
          }
          luisa::compute::detail::SwitchStmtBuilder{
              *context.svm_immediate_override} %
              [&] {
                for (const auto encoded : active) {
                  luisa::compute::detail::SwitchCaseStmtBuilder{encoded} %
                      [&, encoded] {
                        value = cycles_voronoi::evaluate(
                            decode_svm_configuration(instruction.operation,
                                                     encoded),
                            vector_value, w, scale, detail, roughness,
                            lacunarity, smoothness, exponent, randomness);
                      };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                  luisa::compute::dsl::unreachable(
                      "invalid compact surface Voronoi immediate");
                };
              };
        } else {
          value = cycles_voronoi::evaluate(
              cycles_voronoi::decode_configuration(instruction), vector_value,
              w, scale, detail, roughness, lacunarity, smoothness, exponent,
              randomness);
        }
        return project_surface_value(instruction.result_type, value);
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_voronoi_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (!cycles_voronoi::is_operation(instruction.operation)) {
        return nullptr;
    }
    return std::make_unique<VoronoiValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail
