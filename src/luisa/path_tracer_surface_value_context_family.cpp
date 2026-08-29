#include "path_tracer_surface_value_context_family.h"

#include "surface_fresnel.h"
#include "surface_geometry_context.h"

#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_noise.h>

#include <cstddef>
#include <cstdlib>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

void require_operand_count(const compiler::SurfaceValueStaticVariant &variant,
                           std::size_t expected) noexcept {
  if (variant.operand_types.size() != expected ||
      variant.operand_routes.size() != expected) {
    std::abort();
  }
}

void require_zero_immediate_domain(
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  require_operand_count(variant, 0u);
  if (variant.svm_immediates.size() != 1u ||
      variant.svm_immediates.front() != 0u) {
    std::abort();
  }
}

void require_result_bank(const compiler::SurfaceValueStaticVariant &variant,
                         compiler::SurfaceValueBank expected) noexcept {
  auto actual = compiler::SurfaceValueBank::scalar;
  if (!compiler::classify_surface_value_type(variant.instruction.result_type,
                                             actual) ||
      actual != expected) {
    std::abort();
  }
}

[[nodiscard]] Float predicate(Bool value) noexcept {
  return select(0.0f, 1.0f, value);
}

[[nodiscard]] Float3
context_normal(const SurfacePoint &point, Var<luisa::uint4> instruction,
               const compiler::SurfaceValueStaticVariant &variant,
               SurfaceValueOperandReader &operands,
               std::size_t normal_operand) noexcept {
  if (variant.svm_immediates.empty()) {
    std::abort();
  }
  auto has_unlinked = false;
  auto has_linked = false;
  for (const auto immediate : variant.svm_immediates) {
    if ((immediate & ~std::uint16_t{1u}) != 0u) {
      std::abort();
    }
    if (immediate == 0u) {
      if (has_unlinked) {
        std::abort();
      }
      has_unlinked = true;
    } else {
      if (has_linked) {
        std::abort();
      }
      has_linked = true;
    }
  }
  if (!has_linked) {
    return point.shading_normal;
  }
  if (!has_unlinked) {
    return operands.vector(normal_operand);
  }
  Float3 normal = point.shading_normal;
  $if((surface_value_runtime_immediate(instruction) & 1u) != 0u) {
    normal = operands.vector(normal_operand);
  };
  return normal;
}

void emit_tangent_family(
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  require_zero_immediate_domain(variant);
  require_result_bank(variant, compiler::SurfaceValueBank::vector);
  if (variant.instruction.operation != compiler::ValueOperation::tangent) {
    std::abort();
  }
  write_surface_value_vector(locals, instruction,
                             surface_geometry_tangent(point));
}

void emit_object_info_family(
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  require_zero_immediate_domain(variant);
  switch (variant.instruction.operation) {
  case compiler::ValueOperation::object_location:
    require_result_bank(variant, compiler::SurfaceValueBank::vector);
    write_surface_value_vector(locals, instruction, point.object_location);
    return;
  case compiler::ValueOperation::object_random:
    require_result_bank(variant, compiler::SurfaceValueBank::scalar);
    write_surface_value_scalar(locals, instruction, point.object_random);
    return;
  case compiler::ValueOperation::random_per_island:
    require_result_bank(variant, compiler::SurfaceValueBank::scalar);
    write_surface_value_scalar(locals, instruction, point.random_per_island);
    return;
  default:
    std::abort();
  }
}

void emit_particle_info_family(
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  require_zero_immediate_domain(variant);
  require_result_bank(variant, compiler::SurfaceValueBank::scalar);
  switch (variant.instruction.operation) {
  case compiler::ValueOperation::particle_index:
    write_surface_value_scalar(locals, instruction,
                               cast<float>(point.particle_index));
    return;
  case compiler::ValueOperation::particle_random:
    write_surface_value_scalar(
        locals, instruction,
        cycles_noise::uint_to_float_inclusive(
            cycles_noise::hash_uint2(point.particle_index, 0u)));
    return;
  default:
    std::abort();
  }
}

void emit_hair_info_family(
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  require_zero_immediate_domain(variant);
  switch (variant.instruction.operation) {
  case compiler::ValueOperation::curve_is_strand:
    require_result_bank(variant, compiler::SurfaceValueBank::scalar);
    write_surface_value_scalar(locals, instruction, predicate(point.is_curve));
    return;
  case compiler::ValueOperation::curve_intercept:
    require_result_bank(variant, compiler::SurfaceValueBank::scalar);
    write_surface_value_scalar(locals, instruction, point.curve_intercept);
    return;
  case compiler::ValueOperation::curve_length:
    require_result_bank(variant, compiler::SurfaceValueBank::scalar);
    write_surface_value_scalar(locals, instruction, point.curve_length);
    return;
  case compiler::ValueOperation::curve_thickness:
    require_result_bank(variant, compiler::SurfaceValueBank::scalar);
    write_surface_value_scalar(locals, instruction, point.curve_thickness);
    return;
  case compiler::ValueOperation::curve_tangent_normal:
    require_result_bank(variant, compiler::SurfaceValueBank::vector);
    write_surface_value_vector(locals, instruction, point.curve_tangent_normal);
    return;
  case compiler::ValueOperation::curve_random:
    require_result_bank(variant, compiler::SurfaceValueBank::scalar);
    write_surface_value_scalar(locals, instruction, point.curve_random);
    return;
  default:
    std::abort();
  }
}

void emit_light_path_family(
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
  require_zero_immediate_domain(variant);
  require_result_bank(variant, compiler::SurfaceValueBank::scalar);
  Float value = 0.0f;
  switch (variant.instruction.operation) {
  case compiler::ValueOperation::back_facing:
    value = predicate(point.back_facing);
    break;
  case compiler::ValueOperation::path_is_camera:
    value = predicate(
        (point.ray_visibility &
         contract::visibility_bit(contract::RayVisibility::camera)) != 0u);
    break;
  case compiler::ValueOperation::path_is_shadow:
    value = predicate(
        (point.ray_visibility &
         contract::visibility_bit(contract::RayVisibility::shadow)) != 0u);
    break;
  case compiler::ValueOperation::path_is_diffuse:
    value = predicate(
        (point.ray_visibility &
         contract::visibility_bit(contract::RayVisibility::diffuse)) != 0u);
    break;
  case compiler::ValueOperation::path_is_glossy:
    value = predicate(
        (point.ray_visibility &
         contract::visibility_bit(contract::RayVisibility::glossy)) != 0u);
    break;
  case compiler::ValueOperation::path_is_singular:
    value = predicate((point.ray_events & static_cast<std::uint32_t>(
                                              contract::event_singular)) != 0u);
    break;
  case compiler::ValueOperation::path_is_reflection:
    value =
        predicate((point.ray_events & static_cast<std::uint32_t>(
                                          contract::event_reflection)) != 0u);
    break;
  case compiler::ValueOperation::path_is_transmission:
    value = predicate((point.ray_visibility &
                       contract::visibility_bit(
                           contract::RayVisibility::transmission)) != 0u);
    break;
  case compiler::ValueOperation::path_is_volume_scatter:
    value = predicate((point.ray_visibility &
                       contract::visibility_bit(
                           contract::RayVisibility::volume_scatter)) != 0u);
    break;
  case compiler::ValueOperation::path_ray_length:
    value = point.ray_length;
    break;
  case compiler::ValueOperation::path_ray_depth:
    // SurfacePoint owns the evaluation-context adjustment from Cycles:
    // background/light emission and shadow states have already added
    // the effective extra bounce. Re-applying it here is invalid.
    value = cast<float>(point.ray_depth);
    break;
  case compiler::ValueOperation::path_diffuse_depth:
    value = cast<float>(point.diffuse_depth);
    break;
  case compiler::ValueOperation::path_glossy_depth:
    value = cast<float>(point.glossy_depth);
    break;
  case compiler::ValueOperation::path_transparent_depth:
    value = cast<float>(point.transparent_depth);
    break;
  case compiler::ValueOperation::path_transmission_depth:
    value = cast<float>(point.transmission_depth);
    break;
  case compiler::ValueOperation::path_portal_depth:
    // Portal closures are not in the admitted surface program yet, so
    // portal_bounce == 0 is a reachable-state invariant, not an alias
    // for any other counter.
    value = 0.0f;
    break;
  default:
    std::abort();
  }
  write_surface_value_scalar(locals, instruction, std::move(value));
}

void emit_fresnel_family(const SurfacePoint &point,
                         const SurfaceValueLocalsView &locals,
                         Var<luisa::uint4> instruction,
                         const compiler::SurfaceValueStaticVariant &variant,
                         SurfaceValueOperandReader &operands) noexcept {
  require_operand_count(variant, operand::fresnel::count);
  require_result_bank(variant, compiler::SurfaceValueBank::scalar);
  if (variant.instruction.operation != compiler::ValueOperation::fresnel) {
    std::abort();
  }
  auto eta = max(operands.scalar(operand::fresnel::ior), 1.0e-5f);
  const auto normal = context_normal(point, instruction, variant, operands,
                                     operand::fresnel::normal);
  eta = select(eta, 1.0f / eta, point.back_facing);
  write_surface_value_scalar(
      locals, instruction,
      fresnel_dielectric_cos(dot(point.incoming, normal), eta));
}

void emit_layer_weight_family(
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
  require_operand_count(variant, operand::layer_weight::count);
  require_result_bank(variant, compiler::SurfaceValueBank::scalar);
  auto blend = operands.scalar(operand::layer_weight::blend);
  const auto normal = context_normal(point, instruction, variant, operands,
                                     operand::layer_weight::normal);
  Float value = 0.0f;
  switch (variant.instruction.operation) {
  case compiler::ValueOperation::layer_weight_fresnel: {
    auto eta = max(1.0f - blend, 1.0e-5f);
    eta = select(1.0f / eta, eta, point.back_facing);
    value = fresnel_dielectric_cos(dot(point.incoming, normal), eta);
    break;
  }
  case compiler::ValueOperation::layer_weight_facing: {
    value = abs(dot(point.incoming, normal));
    // Match Cycles' exact identity branch. Besides avoiding a useless
    // pow(x, 1), this preserves its exceptional-value semantics.
    $if(blend != 0.5f) {
      blend = clamp(blend, 0.0f, 1.0f - 1.0e-5f);
      blend = select(0.5f / (1.0f - blend), 2.0f * blend, blend < 0.5f);
      value = pow(value, blend);
    };
    value = 1.0f - value;
    break;
  }
  default:
    std::abort();
  }
  write_surface_value_scalar(locals, instruction, std::move(value));
}

} // namespace

void emit_direct_surface_context_family(
    compiler::SurfaceSvmValueOpcode family, const SurfacePoint &point,
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
  switch (family) {
  case compiler::SurfaceSvmValueOpcode::tangent:
    emit_tangent_family(point, locals, instruction, variant);
    return;
  case compiler::SurfaceSvmValueOpcode::object_info:
    emit_object_info_family(point, locals, instruction, variant);
    return;
  case compiler::SurfaceSvmValueOpcode::particle_info:
    emit_particle_info_family(point, locals, instruction, variant);
    return;
  case compiler::SurfaceSvmValueOpcode::hair_info:
    emit_hair_info_family(point, locals, instruction, variant);
    return;
  case compiler::SurfaceSvmValueOpcode::light_path:
    emit_light_path_family(point, locals, instruction, variant);
    return;
  case compiler::SurfaceSvmValueOpcode::fresnel:
    emit_fresnel_family(point, locals, instruction, variant, operands);
    return;
  case compiler::SurfaceSvmValueOpcode::layer_weight:
    emit_layer_weight_family(point, locals, instruction, variant, operands);
    return;
  default:
    std::abort();
  }
}

} // namespace psycles::luisa_backend::detail
