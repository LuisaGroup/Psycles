/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_constant_fold.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <utility>
#include <vector>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] contract::SocketType
constant_type(const GraphInput *input) noexcept {
  if (input->value) {
    return input->value->type;
  }
  switch (input->type) {
    case GraphSocketType::floating:
      return contract::SocketType::floating;
    case GraphSocketType::integer:
      return contract::SocketType::integer;
    case GraphSocketType::color:
      return contract::SocketType::color;
    case GraphSocketType::vector:
      return contract::SocketType::vector;
    case GraphSocketType::normal:
      return contract::SocketType::normal;
    case GraphSocketType::point:
      return contract::SocketType::point;
    case GraphSocketType::closure:
      break;
  }
  std::abort();
}

void set_constant(GraphInput *input, float value) {
  input->value.emplace(constant_type(input),
                       contract::SocketLiteral{value});
  input->constant_folded_in = true;
}

void set_constant(GraphInput *input, Vec3f value) {
  input->value.emplace(constant_type(input),
                       contract::SocketLiteral{value});
  input->constant_folded_in = true;
}

void set_constant(GraphInput *input, std::int32_t value) {
  input->value.emplace(
      constant_type(input),
      contract::SocketLiteral{static_cast<std::int64_t>(value)});
  input->constant_folded_in = true;
}

[[nodiscard]] std::optional<float> float_value(
    const GraphInput *input) noexcept {
  if (input == nullptr || input->link != nullptr || !input->value ||
      input->value->type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Vec3f> float3_value(
    const GraphInput *input) noexcept {
  if (input == nullptr || input->link != nullptr || !input->value) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<Vec3f>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] float safe_divide(float a, float b) noexcept {
  return b != 0.0f ? a / b : 0.0f;
}

[[nodiscard]] float cycles_max(float a, float b) noexcept {
  return a > b ? a : b;
}

[[nodiscard]] float cycles_min(float a, float b) noexcept {
  return a < b ? a : b;
}

[[nodiscard]] float cycles_clamp(float value, float minimum,
                                 float maximum) noexcept {
  return cycles_min(cycles_max(value, minimum), maximum);
}

[[nodiscard]] int cycles_float_to_int(float value) noexcept {
  // Direct host spelling of Cycles util/math_base.h::float_to_int. Math-node
  // defaults originate from the same finite Blender float socket domain.
  return static_cast<int>(value);
}

[[nodiscard]] float safe_pow(float a, float b) noexcept {
  if (a < 0.0f && b != static_cast<float>(cycles_float_to_int(b))) {
    return 0.0f;
  }
  if (b == 0.0f) {
    return 1.0f;
  }
  if (a == 0.0f) {
    return 0.0f;
  }
  return std::pow(a, b);
}

[[nodiscard]] float safe_log(float a, float b) noexcept {
  return a <= 0.0f || b <= 0.0f
             ? 0.0f
             : safe_divide(std::log(a), std::log(b));
}

[[nodiscard]] float wrap(float value, float maximum,
                         float minimum) noexcept {
  const auto range = maximum - minimum;
  return range != 0.0f
             ? value - range * std::floor((value - minimum) / range)
             : minimum;
}

[[nodiscard]] float pingpong(float a, float b) noexcept {
  if (b == 0.0f) {
    return 0.0f;
  }
  const auto x = (a - b) / (b * 2.0f);
  const auto fraction = x - std::floor(x);
  return std::fabs(fraction * b * 2.0f - b);
}

[[nodiscard]] float smooth_min(float a, float b, float k) noexcept {
  if (k != 0.0f) {
    const auto h = std::fmax(k - std::fabs(a - b), 0.0f) / k;
    return std::fmin(a, b) - h * h * h * k * (1.0f / 6.0f);
  }
  return std::fmin(a, b);
}

} // namespace

ConstantFolder::ConstantFolder(CyclesGraph *graph_value, GraphNode *node_value,
                               GraphOutput *output_value) noexcept
    : graph{graph_value}, node{node_value}, output{output_value} {}

bool ConstantFolder::all_inputs_constant() const noexcept {
  return std::none_of(node->inputs.begin(), node->inputs.end(),
                      [](const auto &input) { return input.link != nullptr; });
}

void ConstantFolder::make_constant(float value) const {
  for (auto *socket : output->links) {
    set_constant(socket, value);
  }
  graph->disconnect(output);
}

void ConstantFolder::make_constant(Vec3f value) const {
  for (auto *socket : output->links) {
    set_constant(socket, value);
  }
  graph->disconnect(output);
}

void ConstantFolder::make_constant(std::int32_t value) const {
  for (auto *socket : output->links) {
    set_constant(socket, value);
  }
  graph->disconnect(output);
}

void ConstantFolder::make_constant_clamp(float value, bool clamp) const {
  make_constant(clamp ? cycles_clamp(value, 0.0f, 1.0f) : value);
}

void ConstantFolder::make_constant_clamp(Vec3f value, bool clamp) const {
  if (clamp) {
    value.x = cycles_clamp(value.x, 0.0f, 1.0f);
    value.y = cycles_clamp(value.y, 0.0f, 1.0f);
    value.z = cycles_clamp(value.z, 0.0f, 1.0f);
  }
  make_constant(value);
}

void ConstantFolder::make_zero() const {
  if (output->type == GraphSocketType::floating) {
    make_constant(0.0f);
  } else if (output->type == GraphSocketType::color ||
             output->type == GraphSocketType::vector ||
             output->type == GraphSocketType::normal ||
             output->type == GraphSocketType::point) {
    make_constant(Vec3f{});
  } else {
    std::abort();
  }
}

void ConstantFolder::make_one() const {
  if (output->type == GraphSocketType::floating) {
    make_constant(1.0f);
  } else if (output->type == GraphSocketType::color ||
             output->type == GraphSocketType::vector ||
             output->type == GraphSocketType::normal ||
             output->type == GraphSocketType::point) {
    make_constant(Vec3f{1.0f, 1.0f, 1.0f});
  } else {
    std::abort();
  }
}

void ConstantFolder::bypass(GraphOutput *new_output) const {
  if (new_output == nullptr) {
    std::abort();
  }
  const auto links = output->links;
  graph->disconnect(output);
  for (auto *socket : links) {
    if (!graph->connect(new_output, socket)) {
      std::abort();
    }
  }
}

void ConstantFolder::discard() const {
  if (output->type != GraphSocketType::closure) {
    std::abort();
  }
  graph->disconnect(output);
}

void ConstantFolder::bypass_or_discard(GraphInput *input) const {
  if (input == nullptr || input->type != GraphSocketType::closure) {
    std::abort();
  }
  if (input->link != nullptr) {
    bypass(input->link);
  } else {
    discard();
  }
}

bool ConstantFolder::try_bypass_or_make_constant(GraphInput *input,
                                                  bool clamp) const {
  if (input == nullptr || input->type != output->type) {
    return false;
  }
  if (input->link == nullptr) {
    if (input->type == GraphSocketType::floating) {
      if (const auto value = float_value(input)) {
        make_constant_clamp(*value, clamp);
        return true;
      }
    } else if (const auto value = float3_value(input)) {
      make_constant_clamp(*value, clamp);
      return true;
    }
  } else if (!clamp) {
    bypass(input->link);
    return true;
  } else {
    for (auto &other : node->inputs) {
      if (&other != input && other.link != nullptr) {
        graph->disconnect(&other);
      }
    }
  }
  return false;
}

bool ConstantFolder::is_zero(GraphInput *input) const noexcept {
  if (const auto value = float_value(input)) {
    return *value == 0.0f;
  }
  if (const auto value = float3_value(input)) {
    return *value == Vec3f{};
  }
  return false;
}

bool ConstantFolder::is_one(GraphInput *input) const noexcept {
  if (const auto value = float_value(input)) {
    return *value == 1.0f;
  }
  if (const auto value = float3_value(input)) {
    return *value == Vec3f{1.0f, 1.0f, 1.0f};
  }
  return false;
}

void ConstantFolder::fold_math(NodeMathType type) const {
  auto *value1 = node->input("Value1");
  auto *value2 = node->input("Value2");
  switch (type) {
    case NODE_MATH_ADD:
      if (is_zero(value1)) {
        static_cast<void>(try_bypass_or_make_constant(value2));
      } else if (is_zero(value2)) {
        static_cast<void>(try_bypass_or_make_constant(value1));
      }
      break;
    case NODE_MATH_SUBTRACT:
      if (is_zero(value2)) {
        static_cast<void>(try_bypass_or_make_constant(value1));
      }
      break;
    case NODE_MATH_MULTIPLY:
      if (is_one(value1)) {
        static_cast<void>(try_bypass_or_make_constant(value2));
      } else if (is_one(value2)) {
        static_cast<void>(try_bypass_or_make_constant(value1));
      } else if (is_zero(value1) || is_zero(value2)) {
        make_zero();
      }
      break;
    case NODE_MATH_DIVIDE:
      if (is_one(value2)) {
        static_cast<void>(try_bypass_or_make_constant(value1));
      } else if (is_zero(value1)) {
        make_zero();
      }
      break;
    case NODE_MATH_POWER:
      if (is_one(value1) || is_zero(value2)) {
        make_one();
      } else if (is_one(value2)) {
        static_cast<void>(try_bypass_or_make_constant(value1));
      }
      break;
    default:
      break;
  }
}

float svm_math(NodeMathType type, float a, float b, float c) noexcept {
  switch (type) {
    case NODE_MATH_ADD:
      return a + b;
    case NODE_MATH_SUBTRACT:
      return a - b;
    case NODE_MATH_MULTIPLY:
      return a * b;
    case NODE_MATH_DIVIDE:
      return safe_divide(a, b);
    case NODE_MATH_POWER:
      return safe_pow(a, b);
    case NODE_MATH_LOGARITHM:
      return safe_log(a, b);
    case NODE_MATH_SQRT:
      return std::sqrt(std::fmax(a, 0.0f));
    case NODE_MATH_INV_SQRT:
      return a > 0.0f ? 1.0f / std::sqrt(a) : 0.0f;
    case NODE_MATH_ABSOLUTE:
      return std::fabs(a);
    case NODE_MATH_RADIANS:
      return a * (std::numbers::pi_v<float> / 180.0f);
    case NODE_MATH_DEGREES:
      return a * (180.0f / std::numbers::pi_v<float>);
    case NODE_MATH_MINIMUM:
      return std::fmin(a, b);
    case NODE_MATH_MAXIMUM:
      return std::fmax(a, b);
    case NODE_MATH_LESS_THAN:
      return a < b ? 1.0f : 0.0f;
    case NODE_MATH_GREATER_THAN:
      return a > b ? 1.0f : 0.0f;
    case NODE_MATH_ROUND:
      return std::floor(a + 0.5f);
    case NODE_MATH_FLOOR:
      return std::floor(a);
    case NODE_MATH_CEIL:
      return std::ceil(a);
    case NODE_MATH_FRACTION:
      return a - std::floor(a);
    case NODE_MATH_MODULO:
      return b != 0.0f ? std::fmod(a, b) : 0.0f;
    case NODE_MATH_FLOORED_MODULO:
      return b != 0.0f ? a - std::floor(a / b) * b : 0.0f;
    case NODE_MATH_TRUNC:
      return a >= 0.0f ? std::floor(a) : std::ceil(a);
    case NODE_MATH_SNAP:
      return std::floor(safe_divide(a, b)) * b;
    case NODE_MATH_WRAP:
      return wrap(a, b, c);
    case NODE_MATH_PINGPONG:
      return pingpong(a, b);
    case NODE_MATH_SINE:
      return std::sin(a);
    case NODE_MATH_COSINE:
      return std::cos(a);
    case NODE_MATH_TANGENT:
      return std::tan(a);
    case NODE_MATH_SINH:
      return std::sinh(a);
    case NODE_MATH_COSH:
      return std::cosh(a);
    case NODE_MATH_TANH:
      return std::tanh(a);
    case NODE_MATH_ARCSINE:
      return std::asin(cycles_clamp(a, -1.0f, 1.0f));
    case NODE_MATH_ARCCOSINE:
      return std::acos(cycles_clamp(a, -1.0f, 1.0f));
    case NODE_MATH_ARCTANGENT:
      return std::atan(a);
    case NODE_MATH_ARCTAN2:
      return a == 0.0f && b == 0.0f ? 0.0f : std::atan2(a, b);
    case NODE_MATH_SIGN:
      return a == 0.0f ? 0.0f : (a < 0.0f ? -1.0f : 1.0f);
    case NODE_MATH_EXPONENT:
      return std::exp(a);
    case NODE_MATH_COMPARE:
      return a == b || std::fabs(a - b) <= std::fmax(c, FLT_EPSILON)
                 ? 1.0f
                 : 0.0f;
    case NODE_MATH_MULTIPLY_ADD:
      return a * b + c;
    case NODE_MATH_SMOOTH_MIN:
      return smooth_min(a, b, c);
    case NODE_MATH_SMOOTH_MAX:
      return -smooth_min(-a, -b, c);
  }
  return 0.0f;
}

} // namespace psycles::compiler::cycles_svm
