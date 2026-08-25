#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <psycles/compiler/surface_program.h>

namespace psycles::compiler {

// Physical storage for a strongly typed value. Parameters remain references
// into the material block; only computed values occupy interpreter-local
// slots. No device-side type tag is required.
enum class SurfaceValueStorageClass : std::uint8_t {
  inactive,
  parameter,
  local_slot
};

enum class SurfaceValueBank : std::uint8_t { scalar, vector, unsigned_integer };

struct SurfaceValueLocation {
  SurfaceValueStorageClass storage{SurfaceValueStorageClass::inactive};
  SurfaceValueBank bank{SurfaceValueBank::scalar};
  std::uint32_t index{};
};

// A deterministic allocation for one topologically closed value domain.
// `instructions` excludes parameters, which are direct material-data reads.
//
// The allocation contract is read-before-write: an evaluator must load all
// operands of an instruction before writing its result. Under that contract,
// the result may reuse a same-bank operand slot whose final use is the current
// instruction. Since every value has one interval in the fixed topological
// order, greedy expiration colors this interval graph with exactly its peak
// number of simultaneously live values.
struct SurfaceValueStoragePlan {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceValueLocation> locations;
  std::vector<ValueExpressionId> instructions;
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t active_values{};
  std::uint32_t parameter_values{};

  [[nodiscard]] bool compatible(const SurfaceProgram &program) const noexcept;

  // Logical SoA payload. A backend may impose stronger alignment, but this
  // metric intentionally excludes type tags and float4 padding.
  [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

// Compact device-program address. The high bit selects the immutable material
// parameter block; computed values use a typed local bank. The next two bits
// encode the bank and the low 29 bits encode its index. 0xffffffff remains
// invalid so accidental sentinel use cannot alias valid storage.
class SurfaceValueAddress {

private:
  std::uint32_t _value{invalid_value};

public:
  static constexpr std::uint32_t invalid_value = ~std::uint32_t{0u};
  static constexpr std::uint32_t parameter_bit = 1u << 31u;
  static constexpr std::uint32_t bank_shift = 29u;
  static constexpr std::uint32_t bank_mask = 0x3u << bank_shift;
  static constexpr std::uint32_t index_mask = (1u << bank_shift) - 1u;

  SurfaceValueAddress() noexcept = default;
  explicit constexpr SurfaceValueAddress(std::uint32_t value) noexcept
      : _value{value} {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    return _value != invalid_value;
  }
  [[nodiscard]] constexpr bool parameter() const noexcept {
    return (_value & parameter_bit) != 0u;
  }
  [[nodiscard]] constexpr SurfaceValueBank bank() const noexcept {
    return static_cast<SurfaceValueBank>((_value & bank_mask) >> bank_shift);
  }
  [[nodiscard]] constexpr std::uint32_t index() const noexcept {
    return _value & index_mask;
  }
  [[nodiscard]] constexpr std::uint32_t encoded() const noexcept {
    return _value;
  }

  auto operator<=>(const SurfaceValueAddress &) const noexcept = default;
};

// Closure-tree topology becomes immutable scene data after the typed value
// stream has been scheduled. A leaf can feed physical population, emission,
// or both; endpoint bits are kept in the instruction so one traversal
// preserves the source closure order for every consumer.
enum class SurfaceClosureEndpoint : std::uint32_t {
  physical = 1u << 0u,
  emission = 1u << 1u
};

using SurfaceClosureEndpointMask = std::uint32_t;

[[nodiscard]] constexpr SurfaceClosureEndpointMask
surface_closure_endpoint_bit(SurfaceClosureEndpoint endpoint) noexcept {
  return static_cast<SurfaceClosureEndpointMask>(endpoint);
}

inline constexpr SurfaceClosureEndpointMask all_surface_closure_endpoints =
    surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical) |
    surface_closure_endpoint_bit(SurfaceClosureEndpoint::emission);

// Semantic operand layouts for closure leaves. Inactive Principled features
// retain invalid addresses; the shared handler supplies their Cycles defaults.
// Named indices are part of the bytecode contract and avoid magic offsets in
// either the host verifier or the Luisa interpreter.
namespace surface_closure_operand {

struct diffuse {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t count = 3u;
};

struct translucent {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t count = 2u;
};

struct principled {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t diffuse_roughness = 3u;
  static constexpr std::size_t subsurface_weight = 4u;
  static constexpr std::size_t subsurface_radius = 5u;
  static constexpr std::size_t subsurface_scale = 6u;
  static constexpr std::size_t subsurface_ior = 7u;
  static constexpr std::size_t subsurface_anisotropy = 8u;
  static constexpr std::size_t transmission_weight = 9u;
  static constexpr std::size_t metallic = 10u;
  static constexpr std::size_t ior = 11u;
  static constexpr std::size_t specular_ior_level = 12u;
  static constexpr std::size_t specular_tint = 13u;
  static constexpr std::size_t alpha = 14u;
  static constexpr std::size_t thin_wall = 15u;
  static constexpr std::size_t sheen_weight = 16u;
  static constexpr std::size_t sheen_roughness = 17u;
  static constexpr std::size_t sheen_tint = 18u;
  static constexpr std::size_t coat_weight = 19u;
  static constexpr std::size_t coat_roughness = 20u;
  static constexpr std::size_t coat_ior = 21u;
  static constexpr std::size_t coat_tint = 22u;
  static constexpr std::size_t coat_normal = 23u;
  static constexpr std::size_t emission_color = 24u;
  static constexpr std::size_t emission_strength = 25u;
  static constexpr std::size_t count = 26u;
};

struct glossy {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t count = 3u;
};

struct glass {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t ior = 3u;
  static constexpr std::size_t count = 4u;
};

struct emission {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t strength = 1u;
  static constexpr std::size_t count = 2u;
};

struct transparent {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t count = 1u;
};

struct subsurface {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t radius = 3u;
  static constexpr std::size_t scale = 4u;
  static constexpr std::size_t ior = 5u;
  static constexpr std::size_t anisotropy = 6u;
  static constexpr std::size_t count = 7u;
};

using refraction = glass;

} // namespace surface_closure_operand

[[nodiscard]] constexpr std::size_t
surface_closure_operand_count(ClosureOperation operation) noexcept {
  switch (operation) {
  case ClosureOperation::null_closure:
  case ClosureOperation::add:
  case ClosureOperation::mix:
    return 0u;
  case ClosureOperation::diffuse:
    return surface_closure_operand::diffuse::count;
  case ClosureOperation::translucent:
    return surface_closure_operand::translucent::count;
  case ClosureOperation::principled:
    return surface_closure_operand::principled::count;
  case ClosureOperation::glossy:
    return surface_closure_operand::glossy::count;
  case ClosureOperation::glass:
    return surface_closure_operand::glass::count;
  case ClosureOperation::emission:
    return surface_closure_operand::emission::count;
  case ClosureOperation::transparent:
    return surface_closure_operand::transparent::count;
  case ClosureOperation::subsurface:
    return surface_closure_operand::subsurface::count;
  case ClosureOperation::refraction:
    return surface_closure_operand::refraction::count;
  }
  return 0u;
}

// A Mix term multiplies the current leaf weight by either factor or
// (1 - factor). Address and polarity are separate because the high address
// bit already selects material-parameter storage.
struct SurfaceClosureMixTerm {
  std::uint32_t address{SurfaceValueAddress::invalid_value};
  std::uint32_t flags{};
};

inline constexpr std::uint32_t surface_closure_mix_complement = 1u << 0u;
inline constexpr std::uint32_t surface_closure_mix_flags_mask =
    surface_closure_mix_complement;

// The hot closure stream is one uint4. Operand arity is fixed by opcode;
// feature masks are parallel data so ordinary non-Principled leaves do not
// carry an extra word. Mix paths are flattened per leaf, avoiding a dynamic
// traversal stack while preserving exact DFS source order.
struct SurfaceClosureBytecodeInstruction {
  std::uint32_t control{};
  std::uint32_t operand_begin{};
  std::uint32_t mix_term_begin{};
  std::uint32_t mix_term_count{};
};

inline constexpr std::uint32_t surface_closure_opcode_mask = 0xffu;
inline constexpr std::uint32_t surface_closure_endpoint_shift = 8u;
inline constexpr std::uint32_t surface_closure_endpoint_mask =
    0x3u << surface_closure_endpoint_shift;
inline constexpr std::uint32_t surface_closure_bssrdf_method_shift = 10u;
inline constexpr std::uint32_t surface_closure_bssrdf_method_mask =
    0x3u << surface_closure_bssrdf_method_shift;
inline constexpr std::uint32_t surface_closure_normal_uses_bump = 1u << 12u;
inline constexpr std::uint32_t surface_closure_coat_normal_linked = 1u << 13u;
inline constexpr std::uint32_t surface_closure_preserve_ggx_energy = 1u << 14u;
inline constexpr std::uint32_t surface_closure_beckmann = 1u << 15u;
inline constexpr std::uint32_t surface_closure_control_mask =
    surface_closure_opcode_mask | surface_closure_endpoint_mask |
    surface_closure_bssrdf_method_mask | surface_closure_normal_uses_bump |
    surface_closure_coat_normal_linked |
    surface_closure_preserve_ggx_energy | surface_closure_beckmann;

// Host-selected semantic handler identity. Endpoint membership and Bump
// provenance are data/control-flow properties outside raw closure setup; the
// remaining bits select C++-stage types or algorithms and therefore form the
// finite Luisa AST variant key.
inline constexpr std::uint32_t surface_closure_static_variant_mask =
    surface_closure_opcode_mask | surface_closure_bssrdf_method_mask |
    surface_closure_coat_normal_linked |
    surface_closure_preserve_ggx_energy | surface_closure_beckmann;

// Emission projection never observes BSSRDF, microfacet-energy, or Beckmann
// configuration. Coat-normal linkage alone changes its Principled layer
// algorithm, so the projected interpreter has a strictly smaller semantic
// key without conflating any observed behavior.
inline constexpr std::uint32_t surface_closure_emission_static_variant_mask =
    surface_closure_opcode_mask | surface_closure_coat_normal_linked;

[[nodiscard]] constexpr std::uint32_t make_surface_closure_control(
    const ClosureInstruction &instruction,
    SurfaceClosureEndpointMask endpoints) noexcept {
  return static_cast<std::uint32_t>(instruction.operation) |
         ((endpoints & 0x3u) << surface_closure_endpoint_shift) |
         (static_cast<std::uint32_t>(instruction.subsurface_method)
          << surface_closure_bssrdf_method_shift) |
         (instruction.normal_uses_bump ? surface_closure_normal_uses_bump
                                       : 0u) |
         (instruction.coat_normal_linked
              ? surface_closure_coat_normal_linked
              : 0u) |
         (instruction.preserve_ggx_energy
              ? surface_closure_preserve_ggx_energy
              : 0u) |
         (instruction.beckmann ? surface_closure_beckmann : 0u);
}

[[nodiscard]] constexpr ClosureOperation surface_closure_operation(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return static_cast<ClosureOperation>(instruction.control &
                                       surface_closure_opcode_mask);
}

[[nodiscard]] constexpr SurfaceClosureEndpointMask surface_closure_endpoints(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return (instruction.control & surface_closure_endpoint_mask) >>
         surface_closure_endpoint_shift;
}

[[nodiscard]] constexpr BssrdfMethod surface_closure_bssrdf_method(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return static_cast<BssrdfMethod>(
      (instruction.control & surface_closure_bssrdf_method_mask) >>
      surface_closure_bssrdf_method_shift);
}

// One exact flattened closure program. `principled_features` is parallel to
// `instructions`; aggregate masks allow the JIT to omit handlers unused by the
// complete scene without specializing on individual material topologies.
struct SurfaceClosureProgramImage {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceClosureBytecodeInstruction> instructions;
  std::vector<PrincipledClosureFeatureMask> principled_features;
  std::vector<std::uint32_t> operands;
  std::vector<SurfaceClosureMixTerm> mix_terms;
  std::uint32_t maximum_mix_depth{};
  std::uint32_t used_operations{};
  PrincipledClosureFeatureMask used_principled_features{};
};

// The hot stream is deliberately 16 bytes. Operand arity is an opcode
// invariant, and uncommon immutable fields live in a side table so ordinary
// arithmetic does not fetch two uint64 values and a static-table descriptor.
// The high 14 bits form an opcode-owned immediate, analogous to the packed
// fields of one Cycles SVM node. An operation may erase a static field from
// its host evaluator key only after this immediate represents that field
// exactly and the serialized-image validator proves the two copies agree.
struct SurfaceValueBytecodeInstruction {
  // Packed as [opcode-owned immediate:14 | result bank:2 |
  // operand count:8 | opcode:8].
  // ValueOperation is a closed uint8_t enum and every operation has a fixed
  // arity. Keeping those facts in the stream makes a serialized image
  // independently verifiable without enlarging the hot 16-byte record.
  std::uint32_t control{};
  std::uint32_t result{};
  std::uint32_t operand_begin{};
  std::uint32_t metadata_index{~std::uint32_t{0u}};
};

inline constexpr std::uint32_t surface_value_opcode_mask = 0xffu;
inline constexpr std::uint32_t surface_value_operand_count_shift = 8u;
inline constexpr std::uint32_t surface_value_operand_count_mask =
    0xffu << surface_value_operand_count_shift;
inline constexpr std::uint32_t surface_value_result_bank_shift = 16u;
inline constexpr std::uint32_t surface_value_result_bank_mask =
    0x3u << surface_value_result_bank_shift;
inline constexpr std::uint32_t surface_value_svm_immediate_shift = 18u;
inline constexpr std::uint32_t surface_value_svm_immediate_value_mask =
    (1u << 14u) - 1u;
inline constexpr std::uint32_t surface_value_svm_immediate_mask =
    surface_value_svm_immediate_value_mask
    << surface_value_svm_immediate_shift;

// Operation-local layouts inside the unshifted 14-bit immediate.
inline constexpr std::uint32_t
    surface_value_noise_normalize_immediate_bit = 1u << 0u;
inline constexpr std::uint32_t surface_value_mix_operation_mask = 0x1fu;
inline constexpr std::uint32_t surface_value_mix_factor_clamp_bit = 1u << 5u;
inline constexpr std::uint32_t surface_value_mix_result_clamp_bit = 1u << 6u;
inline constexpr std::uint32_t surface_value_mapping_type_mask = 0x3u;
inline constexpr std::uint32_t surface_value_mapping_axes_shift = 2u;
inline constexpr std::uint32_t surface_value_mapping_axes_mask =
    0x3fu << surface_value_mapping_axes_shift;
inline constexpr std::uint32_t surface_value_image_extension_mask = 0xffu;
inline constexpr std::uint32_t surface_value_image_srgb_bit = 1u << 8u;
inline constexpr std::uint32_t surface_value_image_unassociate_alpha_bit =
    1u << 9u;
inline constexpr std::uint32_t surface_value_image_interpolation_shift = 10u;
inline constexpr std::uint32_t surface_value_image_interpolation_mask =
    0x3u << surface_value_image_interpolation_shift;
inline constexpr std::uint32_t surface_value_image_projection_shift = 12u;
inline constexpr std::uint32_t surface_value_image_projection_mask =
    0x3u << surface_value_image_projection_shift;
inline constexpr std::uint32_t surface_value_image_configuration_mask =
    surface_value_image_extension_mask | surface_value_image_srgb_bit |
    surface_value_image_unassociate_alpha_bit |
    surface_value_image_interpolation_mask |
    surface_value_image_projection_mask;
inline constexpr std::uint32_t surface_value_control_mask =
    surface_value_opcode_mask | surface_value_operand_count_mask |
    surface_value_result_bank_mask | surface_value_svm_immediate_mask;

[[nodiscard]] constexpr bool surface_value_operation_uses_mapping_immediate(
    ValueOperation operation) noexcept {
  return operation == ValueOperation::mapping;
}

[[nodiscard]] constexpr bool surface_value_operation_uses_image_immediate(
    ValueOperation operation) noexcept {
  return operation == ValueOperation::image_color ||
         operation == ValueOperation::image_alpha ||
         operation == ValueOperation::environment_color ||
         operation == ValueOperation::environment_alpha;
}

[[nodiscard]] constexpr bool
surface_value_operation_uses_svm_immediate(ValueOperation operation) noexcept {
  return operation == ValueOperation::noise_factor ||
         operation == ValueOperation::noise_color ||
         operation == ValueOperation::mix ||
         surface_value_operation_uses_mapping_immediate(operation) ||
         surface_value_operation_uses_image_immediate(operation);
}

[[nodiscard]] constexpr bool surface_value_operation_uses_noise_normalize(
    ValueOperation operation) noexcept {
  return operation == ValueOperation::noise_factor ||
         operation == ValueOperation::noise_color;
}

[[nodiscard]] constexpr std::uint64_t
surface_value_svm_static_u0_mask(ValueOperation operation) noexcept {
  return operation == ValueOperation::mix ||
                 surface_value_operation_uses_mapping_immediate(operation)
             ? ~std::uint64_t{0u}
             : 0u;
}

[[nodiscard]] constexpr std::uint64_t
surface_value_svm_static_u1_mask(ValueOperation operation) noexcept {
  if (surface_value_operation_uses_noise_normalize(operation)) {
    return 1u;
  }
  return operation == ValueOperation::mix ||
                 surface_value_operation_uses_mapping_immediate(operation) ||
                 surface_value_operation_uses_image_immediate(operation)
             ? ~std::uint64_t{0u}
             : 0u;
}

// The device immediate is the complete immutable semantic record, while the
// host evaluator key retains only distinctions that change the recorded AST
// shape. Image BOX projection is a separate family: like Cycles'
// NODE_TEX_IMAGE_BOX, it evaluates object-space normal weights and may issue
// three texture samples, whereas FLAT/SPHERE/TUBE share the single-sample
// image handler. This is an exact quotient by execution shape, not a scene or
// material specialization; projection itself remains in the immediate.
[[nodiscard]] constexpr std::uint64_t
surface_value_svm_evaluator_static_u0(ValueOperation operation,
                                      std::uint64_t static_u0) noexcept {
  return static_u0 & ~surface_value_svm_static_u0_mask(operation);
}

[[nodiscard]] constexpr std::uint64_t
surface_value_svm_evaluator_static_u1(ValueOperation operation,
                                      std::uint64_t static_u1) noexcept {
  const auto image = operation == ValueOperation::image_color ||
                     operation == ValueOperation::image_alpha;
  if (image) {
    const auto projection = (static_u1 & surface_value_image_projection_mask) >>
                            surface_value_image_projection_shift;
    return projection == 1u
               ? std::uint64_t{1u} << surface_value_image_projection_shift
               : 0u;
  }
  return static_u1 & ~surface_value_svm_static_u1_mask(operation);
}

[[nodiscard]] constexpr bool surface_value_svm_static_fields_valid(
    ValueOperation operation, std::uint64_t static_u0,
    std::uint64_t static_u1) noexcept {
  if (operation == ValueOperation::mix) {
    return static_u0 <=
               static_cast<std::uint64_t>(BlendOperation::value) &&
           static_u1 <= 0x3u;
  }
  if (surface_value_operation_uses_mapping_immediate(operation)) {
    return static_u0 <= static_cast<std::uint64_t>(MappingVectorType::normal) &&
           static_u1 <= 0x3fu;
  }
  if (surface_value_operation_uses_image_immediate(operation)) {
    if (static_u0 != 0u ||
        (static_u1 & ~static_cast<std::uint64_t>(
                         surface_value_image_configuration_mask)) != 0u ||
        (static_u1 & surface_value_image_extension_mask) > 3u) {
      return false;
    }
    const auto projection = (static_u1 & surface_value_image_projection_mask) >>
                            surface_value_image_projection_shift;
    const auto environment = operation == ValueOperation::environment_color ||
                             operation == ValueOperation::environment_alpha;
    return !environment ||
           ((static_u1 & surface_value_image_extension_mask) == 0u &&
            projection <= 1u);
  }
  return true;
}

[[nodiscard]] constexpr std::uint32_t make_surface_value_svm_immediate(
    ValueOperation operation, std::uint64_t static_u0,
    std::uint64_t static_u1) noexcept {
  if (surface_value_operation_uses_noise_normalize(operation)) {
    return (static_u1 & 1u) != 0u
               ? surface_value_noise_normalize_immediate_bit
               : 0u;
  }
  if (operation == ValueOperation::mix) {
    return static_cast<std::uint32_t>(static_u0) |
           ((static_u1 & 1u) != 0u
                ? surface_value_mix_factor_clamp_bit
                : 0u) |
           ((static_u1 & 2u) != 0u
                ? surface_value_mix_result_clamp_bit
                : 0u);
  }
  if (surface_value_operation_uses_mapping_immediate(operation)) {
    return static_cast<std::uint32_t>(static_u0) |
           (static_cast<std::uint32_t>(static_u1)
            << surface_value_mapping_axes_shift);
  }
  if (surface_value_operation_uses_image_immediate(operation)) {
    return static_cast<std::uint32_t>(static_u1);
  }
  return 0u;
}

[[nodiscard]] constexpr std::uint32_t make_surface_value_control(
    ValueOperation operation, std::uint8_t operand_count,
    SurfaceValueBank result_bank, std::uint32_t svm_immediate) noexcept {
  return static_cast<std::uint32_t>(operation) |
         (static_cast<std::uint32_t>(operand_count)
          << surface_value_operand_count_shift) |
         (static_cast<std::uint32_t>(result_bank)
          << surface_value_result_bank_shift) |
         ((svm_immediate & surface_value_svm_immediate_value_mask)
          << surface_value_svm_immediate_shift);
}

[[nodiscard]] constexpr ValueOperation surface_value_operation(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return static_cast<ValueOperation>(instruction.control &
                                     surface_value_opcode_mask);
}

[[nodiscard]] constexpr std::uint32_t surface_value_operand_count(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return (instruction.control & surface_value_operand_count_mask) >>
         surface_value_operand_count_shift;
}

[[nodiscard]] constexpr SurfaceValueBank surface_value_result_bank(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return static_cast<SurfaceValueBank>(
      (instruction.control & surface_value_result_bank_mask) >>
      surface_value_result_bank_shift);
}

[[nodiscard]] constexpr bool surface_value_noise_normalize(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return (instruction.control &
          (surface_value_noise_normalize_immediate_bit
           << surface_value_svm_immediate_shift)) != 0u;
}

[[nodiscard]] constexpr std::uint16_t surface_value_svm_immediate(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return static_cast<std::uint16_t>(
      (instruction.control & surface_value_svm_immediate_mask) >>
      surface_value_svm_immediate_shift);
}

// Ordered largest-to-smallest alignment. A metadata record exists only when
// at least one field differs from its canonical zero/invalid default.
struct SurfaceValueBytecodeMetadata {
  std::uint64_t static_u0{};
  std::uint64_t static_u1{};
  float static_f0{};
  float static_f1{};
  std::uint32_t parameter{~std::uint32_t{0u}};
  std::uint32_t static_table_begin{};
  std::uint32_t static_table_count{};
  std::uint32_t reserved{};
};

struct SurfaceValueProgramImage {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceValueBytecodeInstruction> instructions;
  std::vector<std::uint32_t> operands;
  std::vector<SurfaceValueBytecodeMetadata> metadata;
  std::vector<float> static_data;
  // Host-side bridge used while closure records are lowered. Inactive values
  // retain SurfaceValueAddress::invalid_value and are never device operands.
  std::vector<std::uint32_t> value_addresses;
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
};

// One entry per runtime surface tag. All offsets in the aggregate streams are
// absolute, while typed local addresses remain relative to an invocation's
// small scalar/vector/uint64 banks. Eight uint32 fields keep the descriptor a
// naturally aligned 32-byte device record and leave named extension fields for
// the subsequent closure-program lowering.
struct SurfaceValueProgramDescriptor {
  std::uint32_t instruction_begin{};
  std::uint32_t instruction_count{};
  std::uint32_t scalar_slots{};
  std::uint32_t vector_slots{};
  std::uint32_t unsigned_integer_slots{};
  std::uint32_t closure_begin{};
  std::uint32_t closure_count{};
  std::uint32_t flags{};
};

// Scene-wide immutable image consumed by one shared device evaluator. The
// builder is transactional: any malformed source program rejects the complete
// image, so a runtime tag can never observe a partially relocated stream.
struct SurfaceValueSceneImage {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceValueProgramDescriptor> programs;
  std::vector<SurfaceValueBytecodeInstruction> instructions;
  std::vector<std::uint32_t> operands;
  std::vector<SurfaceValueBytecodeMetadata> metadata;
  std::vector<float> static_data;
  std::vector<SurfaceClosureBytecodeInstruction> closure_instructions;
  std::vector<PrincipledClosureFeatureMask> closure_principled_features;
  std::vector<std::uint32_t> closure_operands;
  std::vector<SurfaceClosureMixTerm> closure_mix_terms;
  std::uint32_t maximum_closure_mix_depth{};
  std::uint32_t used_closure_operations{};
  PrincipledClosureFeatureMask used_principled_closure_features{};
};

// One AST body per exact semantic instruction configuration. Operands are
// renumbered to [0, arity), and their original socket types are retained so a
// shared evaluator can load typed addresses before invoking the existing
// Luisa node implementation. Source-node identity and static-table payloads
// are deliberately absent: they are provenance/runtime data, not executable
// semantics. Static-table shape remains part of the variant contract.
struct SurfaceValueStaticVariant {
  ValueInstruction instruction;
  std::vector<contract::SocketType> operand_types;
  // Sorted exact device-immediate image of this host evaluator equivalence
  // class. Dynamic handler switches are generated from this minimal domain,
  // not from every mode known to the frontend.
  std::vector<std::uint16_t> svm_immediates;
};

struct SurfaceValueExecutionInput {
  const SurfaceProgram *program{};
  const SurfaceValueStoragePlan *storage{};
  // Optional for value-only programs. When present, closure bytecode is
  // lowered from the exact value-address image and endpoint projection
  // produced by this input; callers cannot accidentally pair a physical or
  // emission closure stream with a different typed slot plan.
  const SurfaceClosurePlan *closure_plan{};
  SurfaceClosureEndpointMask closure_endpoints{all_surface_closure_endpoints};
};

// `instruction_variants` is parallel to `values.instructions`. Semantic
// interning is exact and bit-preserving for every control field (including
// NaN payloads and signed zero); it never relies on a collision-prone hash
// equivalence. Authored static-table payloads remain in `values.static_data`
// and are addressed by the per-instruction metadata stream.
struct SurfaceValueExecutableScene {
  bool valid{};
  std::string diagnostic;
  SurfaceValueSceneImage values;
  std::vector<SurfaceValueStaticVariant> variants;
  std::vector<std::uint32_t> instruction_variants;
};

// Exact finite-strata execution plan for authored Bump. Root programs may
// contain Bump instructions. Each such instruction names a topologically
// closed height subprogram evaluated at the X/Y offset points. Nested Bump
// dependencies share height subprograms, and `maximum_bump_depth` records the
// number of non-recursive device-callable strata required by the scene.
struct SurfaceValueBumpExecutableScene {
  bool valid{};
  std::string diagnostic;
  SurfaceValueExecutableScene executable;
  std::uint32_t root_program_count{};
  std::uint32_t maximum_bump_depth{};
  // Parallel to executable.values.instructions. Non-Bump instructions retain
  // SurfaceValueAddress::invalid_value.
  std::vector<std::uint32_t> bump_height_programs;
  // Parallel to executable.values.programs. Root entries are invalid; each
  // height subprogram names the typed address returned at stream completion.
  std::vector<std::uint32_t> program_outputs;
};

// `active` must be transitively closed over ValueInstruction operands.
// `outputs` names values consumed after the stream (normally closure roots),
// and must be a subset of `active`.
[[nodiscard]] SurfaceValueStoragePlan
plan_surface_value_storage(const SurfaceProgram &program,
                           const std::vector<bool> &active,
                           const std::vector<bool> &outputs);

// Lowers the proven storage plan without changing graph semantics. Every
// original closure/output remains a typed address into this image; parameters
// remain late-bound material data and are never baked into the stream.
[[nodiscard]] SurfaceValueProgramImage lower_surface_value_program(
    const SurfaceProgram &program,
    const SurfaceValueStoragePlan &storage);

// Flattens the reachable closure tree over an already-lowered preparation
// value image. Add contributes no weight term; a Mix contributes factor or
// (1 - factor) exactly when both of its closure-plan branches are reachable.
// Leaves are emitted in the same depth-first source order as GraphSurface.
[[nodiscard]] SurfaceClosureProgramImage lower_surface_closure_program(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan,
    const SurfaceValueDependencyPlan &dependencies,
    std::span<const std::uint32_t> value_addresses,
    SurfaceClosureEndpointMask endpoints = all_surface_closure_endpoints);

// Concatenates topology programs in runtime-tag order and rebases every
// operand, metadata, and static-table reference to the scene-wide streams.
// Local typed addresses and material ParameterId references are intentionally
// unchanged.
[[nodiscard]] SurfaceValueSceneImage build_surface_value_scene_image(
    std::span<const SurfaceValueProgramImage> programs);

// Aggregates parallel value and closure programs. Closure operands keep their
// invocation-local typed addresses, while all stream offsets are rebased into
// the scene image and published through each value-program descriptor.
[[nodiscard]] SurfaceValueSceneImage build_surface_execution_scene_image(
    std::span<const SurfaceValueProgramImage> value_programs,
    std::span<const SurfaceClosureProgramImage> closure_programs);

// Builds the aggregate scene image and interns immutable instruction
// configurations for a scene-pruned shared Luisa evaluator. Inputs and output
// descriptors remain in runtime surface-tag order.
[[nodiscard]] SurfaceValueExecutableScene
build_surface_value_executable_scene(
    std::span<const SurfaceValueExecutionInput> inputs);

[[nodiscard]] SurfaceValueBumpExecutableScene
build_surface_value_bump_executable_scene(
    std::span<const SurfaceValueExecutionInput> root_inputs);

} // namespace psycles::compiler
