#include "path_tracer_surface_value_procedural_family.h"

#include <psycles/luisa/cycles_noise.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

constexpr auto noise_shape_mask =
    compiler::surface_value_noise_dimensions_mask |
    compiler::surface_value_noise_type_mask;
constexpr auto noise_semantic_mask =
    noise_shape_mask | compiler::surface_value_noise_normalize_immediate_bit;

// The bytecode enum and the Luisa semantic core are independent types. This
// establishes the complete order-preserving isomorphism used by the decoder.
static_assert(static_cast<std::uint32_t>(compiler::NoiseType::multifractal) ==
              static_cast<std::uint32_t>(cycles_noise::Type::multifractal));
static_assert(static_cast<std::uint32_t>(compiler::NoiseType::fbm) ==
              static_cast<std::uint32_t>(cycles_noise::Type::fbm));
static_assert(
    static_cast<std::uint32_t>(compiler::NoiseType::hybrid_multifractal) ==
    static_cast<std::uint32_t>(cycles_noise::Type::hybrid_multifractal));
static_assert(
    static_cast<std::uint32_t>(compiler::NoiseType::ridged_multifractal) ==
    static_cast<std::uint32_t>(cycles_noise::Type::ridged_multifractal));
static_assert(static_cast<std::uint32_t>(compiler::NoiseType::hetero_terrain) ==
              static_cast<std::uint32_t>(cycles_noise::Type::hetero_terrain));

struct NoiseShape {
  std::uint16_t encoded{};
  std::uint32_t dimensions{};
  cycles_noise::Type type{};
};

[[nodiscard]] NoiseShape decode_noise_shape(std::uint16_t immediate) noexcept {
  if ((immediate & ~noise_semantic_mask) != 0u) {
    std::abort();
  }
  const auto dimensions =
      (immediate & compiler::surface_value_noise_dimensions_mask) >>
      compiler::surface_value_noise_dimensions_shift;
  const auto type = (immediate & compiler::surface_value_noise_type_mask) >>
                    compiler::surface_value_noise_type_shift;
  if (dimensions < 1u || dimensions > 4u ||
      type > static_cast<std::uint32_t>(cycles_noise::Type::hetero_terrain)) {
    std::abort();
  }
  return {.encoded = static_cast<std::uint16_t>(immediate & noise_shape_mask),
          .dimensions = dimensions,
          .type = static_cast<cycles_noise::Type>(type)};
}

[[nodiscard]] std::vector<NoiseShape> active_noise_shapes(
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  std::vector<NoiseShape> shapes;
  shapes.reserve(variant.svm_immediates.size());
  for (const auto immediate : variant.svm_immediates) {
    const auto shape = decode_noise_shape(immediate);
    if (std::none_of(shapes.begin(), shapes.end(),
                     [encoded = shape.encoded](const auto &candidate) {
                       return candidate.encoded == encoded;
                     })) {
      shapes.emplace_back(shape);
    }
  }
  if (shapes.empty()) {
    std::abort();
  }
  return shapes;
}

[[nodiscard]] compiler::SurfaceValueBank
result_bank(const compiler::SurfaceValueStaticVariant &variant) noexcept {
  auto bank = compiler::SurfaceValueBank::scalar;
  if (!compiler::classify_surface_value_type(variant.instruction.result_type,
                                             bank)) {
    std::abort();
  }
  return bank;
}

void emit_noise_family(const SurfaceValueLocalsView &locals,
                       Var<luisa::uint4> instruction,
                       const compiler::SurfaceValueStaticVariant &variant,
                       SurfaceValueOperandReader &operands) noexcept {
  const auto operation = variant.instruction.operation;
  const auto color_needed = operation == compiler::ValueOperation::noise_color;
  if ((!color_needed && operation != compiler::ValueOperation::noise_factor) ||
      result_bank(variant) != (color_needed
                                   ? compiler::SurfaceValueBank::vector
                                   : compiler::SurfaceValueBank::scalar)) {
    std::abort();
  }

  // Read the bytecode ABI exactly once and in ascending semantic endpoint
  // order. Cycles' evaluation order is recovered only after all operands have
  // been projected into their strong scalar/vector types.
  const auto vector = operands.vector(operand::noise::vector);
  const auto scale = operands.scalar(operand::noise::scale);
  const auto detail = operands.scalar(operand::noise::detail);
  const auto roughness = operands.scalar(operand::noise::roughness);
  const auto lacunarity = operands.scalar(operand::noise::lacunarity);
  const auto distortion = operands.scalar(operand::noise::distortion);
  const auto w = operands.scalar(operand::noise::w);
  const auto offset = operands.scalar(operand::noise::offset);
  const auto gain = operands.scalar(operand::noise::gain);

  const auto scaled_vector = vector * scale;
  const auto scaled_w = w * scale;
  const auto immediate = surface_value_runtime_immediate(instruction);
  const auto normalize =
      (immediate & compiler::surface_value_noise_normalize_immediate_bit) != 0u;
  const auto shapes = active_noise_shapes(variant);
  const auto evaluate = [&](const NoiseShape &shape) noexcept {
    return cycles_noise::evaluate_texture_shared(
        shape.dimensions, shape.type, normalize, color_needed, scaled_vector,
        scaled_w, detail, roughness, lacunarity, offset, gain, distortion);
  };

  Float4 result = make_float4(0.0f);
  if (shapes.size() == 1u) {
    result = evaluate(shapes.front());
  } else {
    const auto shape = immediate & noise_shape_mask;
    luisa::compute::detail::SwitchStmtBuilder{shape} % [&] {
      for (const auto &active : shapes) {
        luisa::compute::detail::SwitchCaseStmtBuilder{active.encoded} %
            [&, active] { result = evaluate(active); };
      }
      luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
        luisa::compute::dsl::unreachable("invalid compact surface Noise shape");
      };
    };
  }
  if (color_needed) {
    write_surface_value_vector(locals, instruction, result.xyz());
  } else {
    write_surface_value_scalar(locals, instruction, result.x);
  }
}

} // namespace

void emit_direct_surface_procedural_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
  switch (family) {
  case compiler::SurfaceSvmValueOpcode::noise:
    emit_noise_family(locals, instruction, variant, operands);
    return;
  default:
    std::abort();
  }
}

} // namespace psycles::luisa_backend::detail
