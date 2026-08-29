#include "path_tracer_surface_value_color_family.h"

#include "surface_color_nodes.h"
#include "surface_shader_table_evaluation.h"

#include <cstdlib>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] compiler::SurfaceValueBank
result_bank(const compiler::SurfaceValueStaticVariant &variant) noexcept {
  auto bank = compiler::SurfaceValueBank::scalar;
  if (!compiler::classify_surface_value_type(variant.instruction.result_type,
                                             bank)) {
    std::abort();
  }
  return bank;
}

void require_zero_immediate(
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  if (variant.svm_immediates.empty()) {
    std::abort();
  }
  for (const auto immediate : variant.svm_immediates) {
    if (immediate != 0u) {
      std::abort();
    }
  }
}

void emit_hsv(const ShaderServices &services,
              const SurfaceValueLocalsView &locals,
              Var<luisa::uint4> instruction,
              const compiler::SurfaceValueStaticVariant &variant,
              SurfaceValueOperandReader &operands) noexcept {
  if (variant.instruction.operation !=
          compiler::ValueOperation::hue_saturation ||
      result_bank(variant) != compiler::SurfaceValueBank::vector) {
    std::abort();
  }
  require_zero_immediate(variant);
  const auto color = operands.vector(operand::hue_saturation::color);
  const auto hue = operands.scalar(operand::hue_saturation::hue);
  const auto saturation = operands.scalar(operand::hue_saturation::saturation);
  const auto value = operands.scalar(operand::hue_saturation::value);
  const auto factor = operands.scalar(operand::hue_saturation::factor);
  write_surface_value_vector(
      locals, instruction,
      evaluate_surface_hsv(services, color, hue, saturation, value, factor));
}

void emit_invert(const SurfaceValueLocalsView &locals,
                 Var<luisa::uint4> instruction,
                 const compiler::SurfaceValueStaticVariant &variant,
                 SurfaceValueOperandReader &operands) noexcept {
  if (variant.instruction.operation != compiler::ValueOperation::invert ||
      result_bank(variant) != compiler::SurfaceValueBank::vector) {
    std::abort();
  }
  require_zero_immediate(variant);
  const auto color = operands.vector(operand::color_factor::color);
  const auto factor = operands.scalar(operand::color_factor::factor);
  write_surface_value_vector(locals, instruction,
                             evaluate_surface_invert(color, factor));
}

void emit_gamma(const SurfaceValueLocalsView &locals,
                Var<luisa::uint4> instruction,
                const compiler::SurfaceValueStaticVariant &variant,
                SurfaceValueOperandReader &operands) noexcept {
  if (variant.instruction.operation != compiler::ValueOperation::gamma ||
      result_bank(variant) != compiler::SurfaceValueBank::vector) {
    std::abort();
  }
  require_zero_immediate(variant);
  const auto color = operands.vector(operand::gamma::color);
  const auto exponent = operands.scalar(operand::gamma::exponent);
  write_surface_value_vector(locals, instruction,
                             evaluate_surface_gamma(color, exponent));
}

void emit_brightness_contrast(
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
  if (variant.instruction.operation !=
          compiler::ValueOperation::brightness_contrast ||
      result_bank(variant) != compiler::SurfaceValueBank::vector) {
    std::abort();
  }
  require_zero_immediate(variant);
  const auto color = operands.vector(operand::brightness_contrast::color);
  const auto brightness =
      operands.scalar(operand::brightness_contrast::brightness);
  const auto contrast = operands.scalar(operand::brightness_contrast::contrast);
  write_surface_value_vector(
      locals, instruction,
      evaluate_surface_brightness_contrast(color, brightness, contrast));
}

void emit_spectral_color(compiler::SurfaceSvmValueOpcode family,
                         const ShaderServices &services,
                         const SurfaceValueLocalsView &locals,
                         Var<luisa::uint4> instruction,
                         const compiler::SurfaceValueStaticVariant &variant,
                         SurfaceValueOperandReader &operands) noexcept {
  if (result_bank(variant) != compiler::SurfaceValueBank::vector) {
    std::abort();
  }
  require_zero_immediate(variant);
  if (family == compiler::SurfaceSvmValueOpcode::blackbody &&
      variant.instruction.operation == compiler::ValueOperation::blackbody) {
    const auto temperature = operands.scalar(operand::blackbody::temperature);
    write_surface_value_vector(
        locals, instruction, evaluate_surface_blackbody(services, temperature));
    return;
  }
  if (family == compiler::SurfaceSvmValueOpcode::wavelength &&
      variant.instruction.operation == compiler::ValueOperation::wavelength) {
    const auto nanometers = operands.scalar(operand::wavelength::nanometers);
    write_surface_value_vector(
        locals, instruction, evaluate_surface_wavelength(services, nanometers));
    return;
  }
  std::abort();
}

void emit_rgb_curve(const SurfaceValueRuntime &runtime,
                    SurfaceValueBytecodeSlots bytecode_slots,
                    const ShaderServices &services, const SurfacePoint &point,
                    const SurfaceValueLocalsView &locals,
                    Var<luisa::uint4> instruction,
                    const compiler::SurfaceValueStaticVariant &variant,
                    SurfaceValueOperandReader &operands) noexcept {
  if (variant.instruction.operation != compiler::ValueOperation::rgb_curve ||
      result_bank(variant) != compiler::SurfaceValueBank::vector) {
    std::abort();
  }
  const auto parameter = surface_value_runtime_buffer<luisa::uint>(
                             runtime, bytecode_slots.metadata_parameter)
                             .read(instruction.w);
  const auto table = surface_shader_table_view(
      services, point, Expr<std::uint32_t>{parameter.expression()});
  const auto color = operands.vector(operand::rgb_curve::color);
  const auto factor = operands.scalar(operand::rgb_curve::factor);
  const auto min_x = operands.scalar(operand::rgb_curve::min_x);
  const auto max_x = operands.scalar(operand::rgb_curve::max_x);
  const auto extrapolate = operands.scalar(operand::rgb_curve::extrapolate);
  write_surface_value_vector(locals, instruction,
                             evaluate_surface_rgb_curve_svm(
                                 services,
                                 surface_value_runtime_immediate(instruction),
                                 variant.svm_immediates, table, color, factor,
                                 min_x, max_x, extrapolate));
}

void emit_separate_color(const ShaderServices &services,
                         const SurfaceValueLocalsView &locals,
                         Var<luisa::uint4> instruction,
                         const compiler::SurfaceValueStaticVariant &variant,
                         SurfaceValueOperandReader &operands) noexcept {
  const auto operation = variant.instruction.operation;
  if ((operation != compiler::ValueOperation::separate_r &&
       operation != compiler::ValueOperation::separate_g &&
       operation != compiler::ValueOperation::separate_b) ||
      result_bank(variant) != compiler::SurfaceValueBank::scalar) {
    std::abort();
  }
  const auto color = operands.vector(operand::separate_color::color);
  const auto channels =
      separate_color_svm(services, surface_value_runtime_immediate(instruction),
                         variant.svm_immediates, color);
  const auto result =
      operation == compiler::ValueOperation::separate_r   ? channels.x
      : operation == compiler::ValueOperation::separate_g ? channels.y
                                                          : channels.z;
  write_surface_value_scalar(locals, instruction, result);
}

void emit_combine_color(const ShaderServices &services,
                        const SurfaceValueLocalsView &locals,
                        Var<luisa::uint4> instruction,
                        const compiler::SurfaceValueStaticVariant &variant,
                        SurfaceValueOperandReader &operands) noexcept {
  if (variant.instruction.operation !=
          compiler::ValueOperation::combine_color ||
      result_bank(variant) != compiler::SurfaceValueBank::vector) {
    std::abort();
  }
  const auto r = operands.scalar(operand::combine_color::r);
  const auto g = operands.scalar(operand::combine_color::g);
  const auto b = operands.scalar(operand::combine_color::b);
  write_surface_value_vector(
      locals, instruction,
      combine_color_svm(services, surface_value_runtime_immediate(instruction),
                        variant.svm_immediates, make_float3(r, g, b)));
}

} // namespace

void emit_direct_surface_color_family(
    compiler::SurfaceSvmValueOpcode family, const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
  switch (family) {
  case compiler::SurfaceSvmValueOpcode::hsv:
    emit_hsv(services, locals, instruction, variant, operands);
    return;
  case compiler::SurfaceSvmValueOpcode::invert:
    emit_invert(locals, instruction, variant, operands);
    return;
  case compiler::SurfaceSvmValueOpcode::gamma:
    emit_gamma(locals, instruction, variant, operands);
    return;
  case compiler::SurfaceSvmValueOpcode::brightness_contrast:
    emit_brightness_contrast(locals, instruction, variant, operands);
    return;
  case compiler::SurfaceSvmValueOpcode::blackbody:
  case compiler::SurfaceSvmValueOpcode::wavelength:
    emit_spectral_color(family, services, locals, instruction, variant,
                        operands);
    return;
  case compiler::SurfaceSvmValueOpcode::rgb_curve:
    emit_rgb_curve(runtime, bytecode_slots, services, point, locals,
                   instruction, variant, operands);
    return;
  case compiler::SurfaceSvmValueOpcode::separate_color:
    emit_separate_color(services, locals, instruction, variant, operands);
    return;
  case compiler::SurfaceSvmValueOpcode::combine_color:
    emit_combine_color(services, locals, instruction, variant, operands);
    return;
  default:
    std::abort();
  }
}

} // namespace psycles::luisa_backend::detail
