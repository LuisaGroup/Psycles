#pragma once

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
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

// Cycles' SVM stack is typed by physical load/store width rather than by the
// nominal Blender socket spelling. Keep that quotient explicit here: Boolean,
// Integer, and Float are one scalar execution type; Float2/Float3/Color/
// Spectrum/Point/Vector/Normal are one float3 execution type; UInt64 is the
// remaining integer-resource type. An evaluator may be shared exactly when
// its result and every operand have the same bank and all non-type semantics
// agree.
[[nodiscard]] constexpr bool classify_surface_value_type(
    contract::SocketType type, SurfaceValueBank &bank) noexcept {
  using contract::SocketType;
  switch (type) {
  case SocketType::boolean:
  case SocketType::integer:
  case SocketType::floating:
    bank = SurfaceValueBank::scalar;
    return true;
  case SocketType::float2:
  case SocketType::float3:
  case SocketType::color:
  case SocketType::spectrum:
  case SocketType::point:
  case SocketType::vector:
  case SocketType::normal:
    bank = SurfaceValueBank::vector;
    return true;
  case SocketType::unsigned_integer:
    bank = SurfaceValueBank::unsigned_integer;
    return true;
  case SocketType::transform:
  case SocketType::string:
  case SocketType::closure:
  case SocketType::volume_closure:
    return false;
  }
  return false;
}

[[nodiscard]] constexpr contract::SocketType
canonical_surface_value_type(SurfaceValueBank bank) noexcept {
  using contract::SocketType;
  switch (bank) {
  case SurfaceValueBank::scalar:
    return SocketType::floating;
  case SurfaceValueBank::vector:
    return SocketType::vector;
  case SurfaceValueBank::unsigned_integer:
    return SocketType::unsigned_integer;
  }
  return SocketType::floating;
}

struct SurfaceValueLocation {
  SurfaceValueStorageClass storage{SurfaceValueStorageClass::inactive};
  SurfaceValueBank bank{SurfaceValueBank::scalar};
  std::uint32_t index{};
};

// Resource constraint for typed interval coloring. The unconstrained default
// preserves the compiler-only API, while a concrete interpreter supplies the
// exact bank capacities of its ABI. Feasibility is component-wise; total byte
// size cannot substitute for this contract because the banks are independent.
struct SurfaceValueStorageCapacity {
  std::uint32_t scalar_slots{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t vector_slots{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t unsigned_integer_slots{
      std::numeric_limits<std::uint32_t>::max()};
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
  // Pure same-bank passthroughs are representatives of an existing SSA
  // value, not device instructions. Their public ValueExpressionId remains
  // valid and maps to the representative's exact address.
  std::uint32_t alias_values{};

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

// Operand records preserve the complete storage class and typed bank, but the
// compact SVM has deliberately small address domains: material ParameterId
// and local-slot indices must fit 13 bits. Two such addresses occupy one
// uint32 word. 0xffff is unambiguously invalid because bank value three is not
// a SurfaceValueBank; it is also the canonical padding lane of an odd-arity
// record. A scene outside this compact domain is rejected transactionally;
// callers may then select the established expanded evaluator rather than
// executing a partially encoded compact image.
class SurfaceValueOperandAddress {

private:
  std::uint16_t _value{invalid_value};

public:
  static constexpr std::uint16_t invalid_value =
      static_cast<std::uint16_t>(0xffffu);
  static constexpr std::uint16_t parameter_bit = 1u << 15u;
  static constexpr std::uint16_t bank_shift = 13u;
  static constexpr std::uint16_t bank_mask = 0x3u << bank_shift;
  static constexpr std::uint16_t index_mask = (1u << bank_shift) - 1u;

  SurfaceValueOperandAddress() noexcept = default;
  explicit constexpr SurfaceValueOperandAddress(std::uint16_t value) noexcept
      : _value{value} {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    return _value != invalid_value &&
           static_cast<std::uint32_t>(bank()) <=
               static_cast<std::uint32_t>(
                   SurfaceValueBank::unsigned_integer);
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
  [[nodiscard]] constexpr std::uint16_t encoded() const noexcept {
    return _value;
  }
  [[nodiscard]] constexpr SurfaceValueAddress expanded() const noexcept {
    return valid()
               ? SurfaceValueAddress{
                     (parameter() ? SurfaceValueAddress::parameter_bit : 0u) |
                     (static_cast<std::uint32_t>(bank())
                      << SurfaceValueAddress::bank_shift) |
                     index()}
               : SurfaceValueAddress{};
  }

  auto operator<=>(const SurfaceValueOperandAddress &) const noexcept =
      default;
};

[[nodiscard]] constexpr bool encode_surface_value_operand_address(
    SurfaceValueAddress address,
    SurfaceValueOperandAddress &operand) noexcept {
  if (!address.valid() ||
      static_cast<std::uint32_t>(address.bank()) >
          static_cast<std::uint32_t>(
              SurfaceValueBank::unsigned_integer) ||
      address.index() > SurfaceValueOperandAddress::index_mask) {
    return false;
  }
  operand = SurfaceValueOperandAddress{static_cast<std::uint16_t>(
      (address.parameter() ? SurfaceValueOperandAddress::parameter_bit : 0u) |
      (static_cast<std::uint32_t>(address.bank())
       << SurfaceValueOperandAddress::bank_shift) |
      address.index())};
  return operand.valid() && operand.expanded() == address;
}

inline constexpr std::uint32_t surface_value_operands_per_word = 2u;
inline constexpr std::uint32_t surface_value_operand_lane_bits = 16u;
inline constexpr std::uint32_t surface_value_inline_operand_capacity = 2u;
inline constexpr std::uint32_t surface_value_invalid_operand_word =
    static_cast<std::uint32_t>(SurfaceValueOperandAddress::invalid_value) |
    (static_cast<std::uint32_t>(SurfaceValueOperandAddress::invalid_value)
     << surface_value_operand_lane_bits);

// ValueOperation is a dense closed enum. Computing the maximum from every
// member makes the compact stream's arity bound follow the semantic contract
// instead of relying on whichever node currently happens to have most inputs.
[[nodiscard]] consteval std::size_t
maximum_surface_value_operand_count() noexcept {
  auto maximum = std::size_t{};
  for (auto opcode = std::uint32_t{};
       opcode <= static_cast<std::uint32_t>(ValueOperation::nishita_sky);
       ++opcode) {
    const auto count = value_operation_operand_count(
        static_cast<ValueOperation>(opcode));
    maximum = count > maximum ? count : maximum;
  }
  return maximum;
}

inline constexpr std::size_t surface_value_max_operand_count =
    maximum_surface_value_operand_count();
static_assert(surface_value_max_operand_count == value_operand::brick::count);

[[nodiscard]] constexpr std::uint32_t surface_value_operand_word_count(
    std::size_t operand_count) noexcept {
  return static_cast<std::uint32_t>(
      (operand_count + surface_value_operands_per_word - 1u) /
      surface_value_operands_per_word);
}

[[nodiscard]] constexpr SurfaceValueOperandAddress
surface_value_operand_from_word(std::uint32_t word,
                                std::size_t lane) noexcept {
  if (lane >= surface_value_operands_per_word) {
    return {};
  }
  return SurfaceValueOperandAddress{static_cast<std::uint16_t>(
      word >> (surface_value_operand_lane_bits * lane))};
}

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
  static constexpr std::size_t anisotropic = 26u;
  static constexpr std::size_t anisotropic_rotation = 27u;
  static constexpr std::size_t tangent = 28u;
  static constexpr std::size_t thin_film_thickness = 29u;
  static constexpr std::size_t thin_film_ior = 30u;
  static constexpr std::size_t count = 31u;
};

struct glossy {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t anisotropy = 3u;
  static constexpr std::size_t rotation = 4u;
  static constexpr std::size_t tangent = 5u;
  static constexpr std::size_t count = 6u;
};

struct metallic {
  static constexpr std::size_t base_ior = 0u;
  static constexpr std::size_t edge_tint_k = 1u;
  static constexpr std::size_t normal = 2u;
  static constexpr std::size_t roughness = 3u;
  static constexpr std::size_t anisotropy = 4u;
  static constexpr std::size_t rotation = 5u;
  static constexpr std::size_t tangent = 6u;
  static constexpr std::size_t thin_film_thickness = 7u;
  static constexpr std::size_t thin_film_ior = 8u;
  static constexpr std::size_t count = 9u;
};

struct sheen {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t count = 3u;
};

struct hair {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t offset = 1u;
  static constexpr std::size_t roughness_u = 2u;
  static constexpr std::size_t roughness_v = 3u;
  static constexpr std::size_t tangent = 4u;
  static constexpr std::size_t count = 5u;
};

struct glass {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t ior = 3u;
  static constexpr std::size_t thin_film_thickness = 4u;
  static constexpr std::size_t thin_film_ior = 5u;
  static constexpr std::size_t count = 6u;
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

struct refraction {
  static constexpr std::size_t color = 0u;
  static constexpr std::size_t normal = 1u;
  static constexpr std::size_t roughness = 2u;
  static constexpr std::size_t ior = 3u;
  static constexpr std::size_t count = 4u;
};

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
  case ClosureOperation::metallic_f82:
  case ClosureOperation::metallic_conductor:
    return surface_closure_operand::metallic::count;
  case ClosureOperation::sheen_microfiber:
  case ClosureOperation::sheen_ashikhmin:
    return surface_closure_operand::sheen::count;
  case ClosureOperation::hair_reflection:
  case ClosureOperation::hair_transmission:
    return surface_closure_operand::hair::count;
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

// The closure stream is a topologically ordered weight program, matching the
// representation used by Cycles after `transform_multi_closure`. AddClosure
// contributes no instruction and forwards its incoming weight to both
// children. A live Mix contributes exactly one instruction and defines the
// weights of its retained children:
//
//   W(root) = 1
//   W(Add.a) = W(Add.b) = W(Add)
//   W(Mix.a) = W(Mix) * (1 - factor)
//   W(Mix.b) = W(Mix) * factor
//
// Every leaf reads its unique incoming weight. The host computes exact
// definition-to-last-use intervals for these SSA weights and colors them into
// a small scalar bank under the read-before-write contract. Thus one Mix is
// one device instruction, no traversal marker or restoration state exists,
// and execution is a single forward pass over immutable scene data.
enum class SurfaceClosureInstructionKind : std::uint32_t {
  leaf = 0u,
  mix_both = 1u,
  mix_left = 2u,
  mix_right = 3u,
};

// The hot closure stream remains one uint4. Payload fields have a semantic
// interpretation selected by `kind`:
//
// leaf:      payload0 = operand begin, payload1 = incoming weight slot,
//            payload2 = zero
// mix_both:  payload0 = factor address, payload1 = incoming weight slot,
//            payload2 = packed (left slot, right slot)
// mix_left:  payload0 = factor address, payload1 = incoming weight slot,
//            payload2 = left slot
// mix_right: payload0 = factor address, payload1 = incoming weight slot,
//            payload2 = right slot
//
// `surface_closure_root_weight_slot` denotes the implicit constant one. Mix
// result slots are 16-bit because the compact runtime is deliberately bounded;
// 0xffff remains an unambiguous sentinel.
struct SurfaceClosureBytecodeInstruction {
  std::uint32_t control{};
  std::uint32_t payload0{};
  std::uint32_t payload1{};
  std::uint32_t payload2{};
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
inline constexpr std::uint32_t surface_closure_instruction_kind_shift = 16u;
inline constexpr std::uint32_t surface_closure_instruction_kind_mask =
    0x3u << surface_closure_instruction_kind_shift;
inline constexpr std::uint32_t surface_closure_microfacet_anisotropy =
    1u << 18u;
inline constexpr std::uint32_t surface_closure_thin_film = 1u << 19u;
inline constexpr std::uint32_t surface_closure_hair_tangent_linked = 1u << 20u;
inline constexpr std::uint32_t surface_closure_control_mask =
    surface_closure_opcode_mask | surface_closure_endpoint_mask |
    surface_closure_bssrdf_method_mask | surface_closure_normal_uses_bump |
    surface_closure_coat_normal_linked |
    surface_closure_preserve_ggx_energy | surface_closure_beckmann |
    surface_closure_instruction_kind_mask |
    surface_closure_microfacet_anisotropy | surface_closure_thin_film |
    surface_closure_hair_tangent_linked;

[[nodiscard]] constexpr std::uint32_t make_surface_closure_instruction_kind(
    SurfaceClosureInstructionKind kind) noexcept {
  return static_cast<std::uint32_t>(kind)
         << surface_closure_instruction_kind_shift;
}

[[nodiscard]] constexpr SurfaceClosureInstructionKind
surface_closure_instruction_kind(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return static_cast<SurfaceClosureInstructionKind>(
      (instruction.control & surface_closure_instruction_kind_mask) >>
      surface_closure_instruction_kind_shift);
}

[[nodiscard]] constexpr bool surface_closure_is_leaf(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return surface_closure_instruction_kind(instruction) ==
         SurfaceClosureInstructionKind::leaf;
}

[[nodiscard]] constexpr std::uint32_t surface_closure_leaf_operand_begin(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return instruction.payload0;
}

inline constexpr std::uint32_t surface_closure_root_weight_slot =
    ~std::uint32_t{0u};
inline constexpr std::uint32_t surface_closure_weight_slot_mask = 0xffffu;
inline constexpr std::uint32_t surface_closure_invalid_packed_weight_slot =
    0xffffu;

[[nodiscard]] constexpr std::uint32_t surface_closure_leaf_weight_slot(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return instruction.payload1;
}

[[nodiscard]] constexpr std::uint32_t surface_closure_mix_factor_address(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return instruction.payload0;
}

[[nodiscard]] constexpr std::uint32_t surface_closure_mix_parent_weight_slot(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return instruction.payload1;
}

[[nodiscard]] constexpr std::uint32_t surface_closure_mix_left_weight_slot(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return instruction.payload2 & surface_closure_weight_slot_mask;
}

[[nodiscard]] constexpr std::uint32_t surface_closure_mix_unary_weight_slot(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return instruction.payload2 & surface_closure_weight_slot_mask;
}

[[nodiscard]] constexpr std::uint32_t surface_closure_mix_right_weight_slot(
    const SurfaceClosureBytecodeInstruction &instruction) noexcept {
  return instruction.payload2 >> 16u;
}

// Host-selected semantic handler identity. Endpoint membership and Bump
// provenance are data/control-flow properties outside raw closure setup; the
// remaining bits select C++-stage types or algorithms and therefore form the
// finite Luisa AST variant key.
inline constexpr std::uint32_t surface_closure_static_variant_mask =
    surface_closure_opcode_mask | surface_closure_bssrdf_method_mask |
    surface_closure_coat_normal_linked |
    surface_closure_preserve_ggx_energy | surface_closure_beckmann |
    surface_closure_microfacet_anisotropy | surface_closure_thin_film |
    surface_closure_hair_tangent_linked;

// Emission projection never observes BSSRDF, microfacet-energy, or Beckmann
// configuration. Coat-normal linkage alone changes its Principled layer
// algorithm, so the projected interpreter has a strictly smaller semantic
// key without conflating any observed behavior.
inline constexpr std::uint32_t surface_closure_emission_static_variant_mask =
    surface_closure_opcode_mask | surface_closure_coat_normal_linked;

[[nodiscard]] constexpr std::uint32_t make_surface_closure_control(
    const ClosureInstruction &instruction,
    SurfaceClosureEndpointMask endpoints,
    bool microfacet_anisotropy = false,
    bool thin_film = false) noexcept {
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
         (instruction.beckmann ? surface_closure_beckmann : 0u) |
         (microfacet_anisotropy
              ? surface_closure_microfacet_anisotropy
              : 0u) |
         (thin_film ? surface_closure_thin_film : 0u) |
         (instruction.hair_tangent_linked
              ? surface_closure_hair_tangent_linked
              : 0u);
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

// One exact topological closure-weight program. `principled_features` is
// parallel to `instructions` and zero on weight instructions; aggregate masks
// allow the JIT to omit leaf handlers unused by the complete scene without
// specializing on individual material topologies.
struct SurfaceClosureProgramImage {
  bool valid{};
  std::string diagnostic;
  std::vector<SurfaceClosureBytecodeInstruction> instructions;
  std::vector<PrincipledClosureFeatureMask> principled_features;
  std::vector<std::uint32_t> operands;
  std::uint32_t maximum_mix_depth{};
  std::uint32_t mix_slots{};
  std::uint32_t used_operations{};
  PrincipledClosureFeatureMask used_principled_features{};
};

// The hot stream is deliberately 16 bytes. Operand arity is a total function
// of the closed opcode enum and is therefore not redundantly serialized.
// Arity-zero through arity-two instructions embed their one packed operand
// word in `operand_payload`; larger instructions store a word offset into the
// packed overflow stream. Uncommon immutable fields live in a side table so
// ordinary arithmetic does not fetch two uint64 values and a static-table
// descriptor. The high 14 bits form an opcode-owned immediate, analogous to
// the packed fields of one Cycles SVM node. An operation may erase a static
// field from its host evaluator key only after this immediate represents that
// field exactly and the serialized-image validator proves the two copies
// agree.
struct SurfaceValueBytecodeInstruction {
  // Packed as [opcode-owned immediate:14 | result bank:2 |
  // reserved-zero:8 | opcode:8].
  std::uint32_t control{};
  std::uint32_t result{};
  std::uint32_t operand_payload{surface_value_invalid_operand_word};
  std::uint32_t metadata_index{~std::uint32_t{0u}};
};

inline constexpr std::uint32_t surface_value_opcode_mask = 0xffu;
inline constexpr std::uint32_t surface_value_result_bank_shift = 16u;
inline constexpr std::uint32_t surface_value_result_bank_mask =
    0x3u << surface_value_result_bank_shift;
inline constexpr std::uint32_t surface_value_svm_immediate_shift = 18u;
inline constexpr std::uint32_t surface_value_svm_immediate_value_mask =
    (1u << 14u) - 1u;
inline constexpr std::uint32_t surface_value_svm_immediate_mask =
    surface_value_svm_immediate_value_mask << surface_value_svm_immediate_shift;

// Reserved internal opcode separating the automatic-normal prefix from the
// endpoint root in one transaction stream. It is not a ValueOperation: the
// record consumes no operands, writes no typed slot, and commits `result` as
// ShaderData::N before execution continues. Keeping the boundary in the data
// stream gives the interpreter one loop independently of backend inlining and
// loop-unrolling choices.
inline constexpr std::uint32_t surface_value_surface_normal_transition_control =
    surface_value_opcode_mask;
static_assert(static_cast<std::uint32_t>(ValueOperation::nishita_sky) <
              surface_value_surface_normal_transition_control);

[[nodiscard]] constexpr bool is_surface_value_surface_normal_transition(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return instruction.control == surface_value_surface_normal_transition_control;
}

// Operation-local layouts inside the unshifted 14-bit immediate.
inline constexpr std::uint32_t surface_value_noise_normalize_immediate_bit =
    1u << 0u;
inline constexpr std::uint32_t surface_value_uv_named_immediate_bit = 1u;
inline constexpr std::uint32_t surface_value_noise_dimensions_shift = 1u;
inline constexpr std::uint32_t surface_value_noise_dimensions_mask =
    0x7u << surface_value_noise_dimensions_shift;
inline constexpr std::uint32_t surface_value_noise_type_shift = 4u;
inline constexpr std::uint32_t surface_value_noise_type_mask =
    0x7u << surface_value_noise_type_shift;
inline constexpr std::uint32_t surface_value_wave_type_mask = 0x1u;
inline constexpr std::uint32_t surface_value_wave_bands_direction_shift = 1u;
inline constexpr std::uint32_t surface_value_wave_bands_direction_mask =
    0x3u << surface_value_wave_bands_direction_shift;
inline constexpr std::uint32_t surface_value_wave_rings_direction_shift = 3u;
inline constexpr std::uint32_t surface_value_wave_rings_direction_mask =
    0x3u << surface_value_wave_rings_direction_shift;
inline constexpr std::uint32_t surface_value_wave_profile_shift = 5u;
inline constexpr std::uint32_t surface_value_wave_profile_mask =
    0x3u << surface_value_wave_profile_shift;
inline constexpr std::uint32_t surface_value_mix_operation_mask = 0x1fu;
inline constexpr std::uint32_t surface_value_mix_factor_clamp_bit = 1u << 5u;
inline constexpr std::uint32_t surface_value_mix_result_clamp_bit = 1u << 6u;
inline constexpr std::uint32_t surface_value_clamp_mode_mask = 0x1u;
inline constexpr std::uint32_t surface_value_map_range_interpolation_mask =
    0x3u;
inline constexpr std::uint32_t surface_value_map_range_clamp_bit = 1u << 2u;
inline constexpr std::uint32_t surface_value_map_range_configuration_mask =
    surface_value_map_range_interpolation_mask |
    surface_value_map_range_clamp_bit;
static_assert(static_cast<std::uint32_t>(ClampMode::minmax) == 0u);
static_assert(static_cast<std::uint32_t>(ClampMode::range) == 1u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::linear) == 0u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::stepped) == 1u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::smoothstep) ==
              2u);
static_assert(static_cast<std::uint32_t>(MapRangeInterpolation::smootherstep) ==
              3u);
static_assert((surface_value_map_range_interpolation_mask &
               surface_value_map_range_clamp_bit) == 0u);
static_assert((surface_value_map_range_configuration_mask &
               ~surface_value_svm_immediate_value_mask) == 0u);

[[nodiscard]] constexpr std::uint32_t encode_surface_value_clamp_immediate(
    ClampMode mode) noexcept {
  return static_cast<std::uint32_t>(mode);
}

[[nodiscard]] constexpr ClampMode decode_surface_value_clamp_immediate(
    std::uint32_t immediate) noexcept {
  return static_cast<ClampMode>(immediate & surface_value_clamp_mode_mask);
}

[[nodiscard]] constexpr std::uint32_t encode_surface_value_map_range_immediate(
    MapRangeInterpolation interpolation, bool clamp_result) noexcept {
  return static_cast<std::uint32_t>(interpolation) |
         (clamp_result ? surface_value_map_range_clamp_bit : 0u);
}

[[nodiscard]] constexpr MapRangeInterpolation
decode_surface_value_map_range_interpolation(
    std::uint32_t immediate) noexcept {
  return static_cast<MapRangeInterpolation>(
      immediate & surface_value_map_range_interpolation_mask);
}

[[nodiscard]] constexpr bool decode_surface_value_map_range_clamp(
    std::uint32_t immediate) noexcept {
  return (immediate & surface_value_map_range_clamp_bit) != 0u;
}

// Exhaust the finite semantic domains at compile time. This proves both
// decode(encode(c)) == c and injectivity, which are the obligations required
// before the corresponding host evaluator fields may be quotiented away.
[[nodiscard]] constexpr bool surface_value_clamp_immediate_contract_holds()
    noexcept {
  constexpr auto mode_count = static_cast<std::uint32_t>(ClampMode::range) + 1u;
  for (auto mode = 0u; mode < mode_count; ++mode) {
    const auto semantic_mode = static_cast<ClampMode>(mode);
    const auto encoded = encode_surface_value_clamp_immediate(semantic_mode);
    if ((encoded & ~surface_value_clamp_mode_mask) != 0u ||
        decode_surface_value_clamp_immediate(encoded) != semantic_mode) {
      return false;
    }
    for (auto other = 0u; other < mode_count; ++other) {
      if (mode != other &&
          encoded == encode_surface_value_clamp_immediate(
                         static_cast<ClampMode>(other))) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr bool surface_value_map_range_immediate_contract_holds()
    noexcept {
  for (auto interpolation = 0u;
       interpolation < map_range_interpolation_count;
       ++interpolation) {
    for (auto clamp = 0u; clamp < 2u; ++clamp) {
      const auto semantic_interpolation =
          static_cast<MapRangeInterpolation>(interpolation);
      const auto encoded = encode_surface_value_map_range_immediate(
          semantic_interpolation, clamp != 0u);
      if ((encoded & ~surface_value_map_range_configuration_mask) != 0u ||
          decode_surface_value_map_range_interpolation(encoded) !=
              semantic_interpolation ||
          decode_surface_value_map_range_clamp(encoded) != (clamp != 0u)) {
        return false;
      }
      for (auto other_interpolation = 0u;
           other_interpolation < map_range_interpolation_count;
           ++other_interpolation) {
        for (auto other_clamp = 0u; other_clamp < 2u; ++other_clamp) {
          if ((interpolation != other_interpolation || clamp != other_clamp) &&
              encoded == encode_surface_value_map_range_immediate(
                             static_cast<MapRangeInterpolation>(
                                 other_interpolation),
                             other_clamp != 0u)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

static_assert(surface_value_clamp_immediate_contract_holds());
static_assert(surface_value_map_range_immediate_contract_holds());
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
inline constexpr std::uint32_t surface_value_image_interpolation_mode_count = 4u;
inline constexpr std::uint32_t
    surface_value_image_interpolation_family_count = 3u;
inline constexpr std::uint32_t surface_value_image_extension_mode_count = 4u;
inline constexpr std::uint32_t surface_value_image_sampling_key_count =
    surface_value_image_interpolation_family_count *
    surface_value_image_extension_mode_count;

// Cycles gives Cubic and Smart the same texture-filter implementation. This is
// the sole quotient used by Image Texture sampling; extension modes remain
// distinct. Keeping the quotient in the bytecode contract prevents graph
// dispatch and callable specialization from independently reinterpreting the
// same immediate bits.
[[nodiscard]] constexpr std::uint32_t
canonical_surface_value_image_interpolation(
    std::uint32_t interpolation) noexcept {
  return interpolation < 2u ? interpolation : 2u;
}

[[nodiscard]] constexpr std::uint32_t make_surface_value_image_sampling_key(
    std::uint32_t interpolation, std::uint32_t extension) noexcept {
  return canonical_surface_value_image_interpolation(interpolation) *
             surface_value_image_extension_mode_count +
         extension;
}

// Exhaust the complete authored domain. Boundedness makes the key a safe array
// index, surjectivity proves that no callable slot is dead, and the final
// biconditional proves that equality of keys loses exactly the Cubic/Smart
// distinction and no other sampling semantics.
[[nodiscard]] constexpr bool
surface_value_image_sampling_quotient_contract_holds() noexcept {
  std::array<bool, surface_value_image_sampling_key_count> reached{};
  for (auto interpolation = 0u;
       interpolation < surface_value_image_interpolation_mode_count;
       ++interpolation) {
    for (auto extension = 0u;
         extension < surface_value_image_extension_mode_count;
         ++extension) {
      const auto key = make_surface_value_image_sampling_key(
          interpolation, extension);
      if (key >= surface_value_image_sampling_key_count) {
        return false;
      }
      reached[key] = true;
      for (auto other_interpolation = 0u;
           other_interpolation < surface_value_image_interpolation_mode_count;
           ++other_interpolation) {
        for (auto other_extension = 0u;
             other_extension < surface_value_image_extension_mode_count;
             ++other_extension) {
          const auto same_canonical_pair =
              canonical_surface_value_image_interpolation(interpolation) ==
                  canonical_surface_value_image_interpolation(
                      other_interpolation) &&
              extension == other_extension;
          if ((key == make_surface_value_image_sampling_key(
                          other_interpolation, other_extension)) !=
              same_canonical_pair) {
            return false;
          }
        }
      }
    }
  }
  for (const auto value : reached) {
    if (!value) {
      return false;
    }
  }
  return true;
}

static_assert((surface_value_image_configuration_mask &
               ~surface_value_svm_immediate_value_mask) == 0u);
static_assert(surface_value_image_sampling_quotient_contract_holds());
inline constexpr std::uint32_t surface_value_control_mask =
    surface_value_opcode_mask | surface_value_result_bank_mask |
    surface_value_svm_immediate_mask;

// Primary interpreter dispatch is derived from the instruction itself. The
// opcode and result bank select the typed handler. Image BOX is the sole
// current sub-opcode execution family because it records a normal-weighted
// multi-sample AST instead of the regular one-sample image AST. Bit 18 is in a
// separate handler-key namespace; it does not consume another bytecode bit.
inline constexpr std::uint32_t surface_value_handler_image_box_bit = 1u << 18u;
static_assert(
    (surface_value_handler_image_box_bit &
     (surface_value_opcode_mask | surface_value_result_bank_mask)) == 0u);

// Graph-expanded Bump operations retain the immutable contract of the source
// evaluator. Canonicalizing only for metadata/immediate validation keeps their
// device opcodes distinct while proving that no semantic field is dropped.
[[nodiscard]] constexpr ValueOperation
surface_value_semantic_base_operation(ValueOperation operation) noexcept {
  switch (operation) {
  case ValueOperation::bump_samples:
    return ValueOperation::bump;
  case ValueOperation::sampled_surface_position:
    return ValueOperation::surface_position;
  case ValueOperation::sampled_uv:
    return ValueOperation::uv;
  case ValueOperation::sampled_generated:
    return ValueOperation::generated;
  case ValueOperation::sampled_object_position:
    return ValueOperation::object_position;
  case ValueOperation::sampled_object_position_with_transform:
    return ValueOperation::object_position_with_transform;
  case ValueOperation::sampled_pointiness:
    return ValueOperation::pointiness;
  case ValueOperation::sampled_attribute_color:
    return ValueOperation::attribute_color;
  case ValueOperation::sampled_attribute_factor:
    return ValueOperation::attribute_factor;
  case ValueOperation::sampled_attribute_alpha:
    return ValueOperation::attribute_alpha;
  case ValueOperation::sampled_normal_map:
    return ValueOperation::normal_map;
  default:
    return operation;
  }
}

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
  operation = surface_value_semantic_base_operation(operation);
  return operation == ValueOperation::noise_factor ||
         operation == ValueOperation::noise_color ||
         operation == ValueOperation::math ||
         operation == ValueOperation::vector_math_value ||
         operation == ValueOperation::vector_math_vector ||
         operation == ValueOperation::clamp_range ||
         operation == ValueOperation::map_range_float ||
         operation == ValueOperation::map_range_vector ||
         operation == ValueOperation::mix ||
         operation == ValueOperation::fresnel ||
         operation == ValueOperation::layer_weight_fresnel ||
         operation == ValueOperation::layer_weight_facing ||
         operation == ValueOperation::uv ||
         operation == ValueOperation::normal_map ||
         operation == ValueOperation::bump ||
         operation == ValueOperation::wave_color ||
         operation == ValueOperation::wave_factor ||
         operation == ValueOperation::gradient ||
         operation == ValueOperation::color_ramp ||
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
  operation = surface_value_semantic_base_operation(operation);
  return operation == ValueOperation::math ||
             operation == ValueOperation::vector_math_value ||
             operation == ValueOperation::vector_math_vector ||
             operation == ValueOperation::clamp_range ||
             operation == ValueOperation::map_range_float ||
             operation == ValueOperation::map_range_vector ||
             operation == ValueOperation::mix ||
             operation == ValueOperation::fresnel ||
             operation == ValueOperation::layer_weight_fresnel ||
             operation == ValueOperation::layer_weight_facing ||
             operation == ValueOperation::uv ||
             operation == ValueOperation::normal_map ||
             operation == ValueOperation::bump ||
             operation == ValueOperation::noise_factor ||
             operation == ValueOperation::noise_color ||
             operation == ValueOperation::wave_color ||
             operation == ValueOperation::wave_factor ||
             operation == ValueOperation::gradient ||
             operation == ValueOperation::color_ramp ||
             surface_value_operation_uses_mapping_immediate(operation)
             ? ~std::uint64_t{0u}
             : 0u;
}

[[nodiscard]] constexpr std::uint64_t
surface_value_svm_static_u1_mask(ValueOperation operation) noexcept {
  operation = surface_value_semantic_base_operation(operation);
  if (surface_value_operation_uses_noise_normalize(operation)) {
    return ~std::uint64_t{0u};
  }
  return operation == ValueOperation::mix ||
                 operation == ValueOperation::map_range_float ||
                 operation == ValueOperation::map_range_vector ||
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
  operation = surface_value_semantic_base_operation(operation);
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
  operation = surface_value_semantic_base_operation(operation);
  if (operation == ValueOperation::mix) {
    return static_u0 <=
               static_cast<std::uint64_t>(BlendOperation::value) &&
           static_u1 <= 0x3u;
  }
  if (operation == ValueOperation::math) {
    return static_u0 < math_operation_count && static_u1 == 0u;
  }
  if (operation == ValueOperation::vector_math_value ||
      operation == ValueOperation::vector_math_vector) {
    return static_u0 < vector_math_operation_count && static_u1 == 0u;
  }
  if (operation == ValueOperation::clamp_range) {
    return static_u0 <= static_cast<std::uint64_t>(ClampMode::range) &&
           static_u1 == 0u;
  }
  if (operation == ValueOperation::map_range_float ||
      operation == ValueOperation::map_range_vector) {
    return static_u0 < map_range_interpolation_count && static_u1 <= 1u;
  }
  if (operation == ValueOperation::fresnel ||
      operation == ValueOperation::layer_weight_fresnel ||
      operation == ValueOperation::layer_weight_facing) {
    return static_u0 <= 1u && static_u1 == 0u;
  }
  if (operation == ValueOperation::normal_map) {
    return (static_u0 & ~(normal_map_space_mask | normal_map_named_tangent |
                         normal_map_displaced_base | normal_map_direct_x)) ==
               0u &&
           decode_normal_map_space(static_u0) <= NormalMapSpace::blender_world &&
           static_u1 == 0u;
  }
  if (operation == ValueOperation::uv) {
    return static_u0 <= 1u && static_u1 == 0u;
  }
  if (operation == ValueOperation::bump) {
    return static_u0 <= 0x7u && static_u1 == 0u;
  }
  if (surface_value_operation_uses_noise_normalize(operation)) {
    const auto type = (static_u1 >> 8u) & 0xffu;
    return static_u0 >= 1u && static_u0 <= 4u && type <= 4u &&
           (static_u1 & ~std::uint64_t{0x701u}) == 0u;
  }
  if (operation == ValueOperation::wave_color ||
      operation == ValueOperation::wave_factor) {
    const auto type = static_u0 & 0xffu;
    const auto bands_direction = (static_u0 >> 8u) & 0xffu;
    const auto rings_direction = (static_u0 >> 16u) & 0xffu;
    const auto profile = (static_u0 >> 24u) & 0xffu;
    return type <= 1u && bands_direction <= 3u &&
           rings_direction <= 3u && profile <= 2u &&
           (static_u0 >> 32u) == 0u && static_u1 == 0u;
  }
  if (operation == ValueOperation::gradient) {
    return static_u0 <= 6u && static_u1 == 0u;
  }
  if (operation == ValueOperation::color_ramp) {
    return static_u0 <= 3u && static_u1 <= 1u;
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
  operation = surface_value_semantic_base_operation(operation);
  if (surface_value_operation_uses_noise_normalize(operation)) {
    return ((static_u1 & 1u) != 0u
                ? surface_value_noise_normalize_immediate_bit
                : 0u) |
           (static_cast<std::uint32_t>(static_u0)
            << surface_value_noise_dimensions_shift) |
           (static_cast<std::uint32_t>((static_u1 >> 8u) & 0xffu)
            << surface_value_noise_type_shift);
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
  if (operation == ValueOperation::math) {
    return static_cast<std::uint32_t>(static_u0);
  }
  if (operation == ValueOperation::vector_math_value ||
      operation == ValueOperation::vector_math_vector) {
    return static_cast<std::uint32_t>(static_u0);
  }
  if (operation == ValueOperation::clamp_range) {
    return encode_surface_value_clamp_immediate(
        static_cast<ClampMode>(static_u0));
  }
  if (operation == ValueOperation::map_range_float ||
      operation == ValueOperation::map_range_vector) {
    return encode_surface_value_map_range_immediate(
        static_cast<MapRangeInterpolation>(static_u0), static_u1 != 0u);
  }
  if (operation == ValueOperation::fresnel ||
      operation == ValueOperation::layer_weight_fresnel ||
      operation == ValueOperation::layer_weight_facing ||
      operation == ValueOperation::uv ||
      operation == ValueOperation::normal_map ||
      operation == ValueOperation::bump ||
      operation == ValueOperation::gradient ||
      operation == ValueOperation::color_ramp) {
    return static_cast<std::uint32_t>(static_u0);
  }
  if (operation == ValueOperation::wave_color ||
      operation == ValueOperation::wave_factor) {
    const auto type = static_cast<std::uint32_t>(static_u0 & 0xffu);
    const auto direction = static_cast<std::uint32_t>(
        (static_u0 >> (type == 0u ? 8u : 16u)) & 0xffu);
    return type |
           (direction <<
            (type == 0u ? surface_value_wave_bands_direction_shift
                        : surface_value_wave_rings_direction_shift)) |
           (static_cast<std::uint32_t>((static_u0 >> 24u) & 0xffu)
            << surface_value_wave_profile_shift);
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

[[nodiscard]] constexpr bool surface_value_operation_uses_image_box_family(
    ValueOperation operation) noexcept {
  return operation == ValueOperation::image_color ||
         operation == ValueOperation::image_alpha;
}

// This coarse instruction-local projection identifies the common Cycles-style
// opcode/execution-family branch without consulting operands or metadata.
// Distinct exact evaluator variants may intentionally map to the same primary
// key; the interpreter refines such a fiber with the complete semantic
// discriminator. Thus this projection only moves the first dispatch level and
// never weakens exact evaluator interning.
[[nodiscard]] constexpr std::uint32_t make_surface_value_handler_key(
    ValueOperation operation,
    SurfaceValueBank result_bank,
    std::uint32_t svm_immediate) noexcept {
  auto key = static_cast<std::uint32_t>(operation) |
             (static_cast<std::uint32_t>(result_bank)
              << surface_value_result_bank_shift);
  if (surface_value_operation_uses_image_box_family(operation)) {
    const auto projection =
        (svm_immediate & surface_value_image_projection_mask) >>
        surface_value_image_projection_shift;
    if (projection == 1u) {
      key |= surface_value_handler_image_box_bit;
    }
  }
  return key;
}

[[nodiscard]] constexpr std::uint32_t make_surface_value_control(
    ValueOperation operation, SurfaceValueBank result_bank,
    std::uint32_t svm_immediate) noexcept {
  return static_cast<std::uint32_t>(operation) |
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
  return is_surface_value_surface_normal_transition(instruction)
             ? 0u
             : static_cast<std::uint32_t>(value_operation_operand_count(
                   surface_value_operation(instruction)));
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

[[nodiscard]] constexpr std::uint32_t surface_value_handler_key(
    const SurfaceValueBytecodeInstruction &instruction) noexcept {
  return make_surface_value_handler_key(
      surface_value_operation(instruction),
      surface_value_result_bank(instruction),
      surface_value_svm_immediate(instruction));
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
  std::uint32_t flags{};
};

// Exact storage-class abstraction for one operand position of an interned
// evaluator variant. The scene builder joins the concrete class observed at
// every instruction mapped to that variant over the two-point domain
// {local, parameter}. A singleton remains directly routable; observing both
// classes yields `dynamic`. There is deliberately no public bottom value:
// every published variant is used by at least one validated instruction.
enum class SurfaceValueOperandRoute : std::uint8_t {
  local,
  parameter,
  dynamic,
};

inline constexpr std::uint32_t
    surface_value_program_automatic_normal_uses_undisplaced_geometry = 1u << 0u;
inline constexpr std::uint32_t surface_value_program_flag_mask =
    surface_value_program_automatic_normal_uses_undisplaced_geometry;

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
  std::uint32_t maximum_closure_mix_depth{};
  std::uint32_t maximum_closure_mix_slots{};
  std::uint32_t used_closure_operations{};
  PrincipledClosureFeatureMask used_principled_closure_features{};
};

// One AST body per exact execution-equivalence class. Operands are renumbered
// to [0, arity), and nominal socket types are quotiented to their
// scalar/float3/uint64 execution banks, exactly matching the runtime load and
// projection semantics. Finite semantic fields covered by an opcode's SVM
// immediate are data in the instruction stream, not host/JIT identity.
// Source-node identity and static-table payloads are likewise absent: they are
// provenance/runtime data, not executable semantics. Static-table shape and
// every unrepresented semantic field remain part of the variant contract.
struct SurfaceValueStaticVariant {
  ValueInstruction instruction;
  std::vector<contract::SocketType> operand_types;
  // Sorted exact device-immediate image of this host evaluator equivalence
  // class. Dynamic handler switches are generated from this minimal domain,
  // not from every mode known to the frontend.
  std::vector<std::uint16_t> svm_immediates;
  // Parallel to operand_types. This is a scene-wide, exact storage-class
  // join: a direct route is recorded only when every bytecode instruction
  // using this evaluator reads that operand position from the same class.
  std::vector<SurfaceValueOperandRoute> operand_routes;
};

struct SurfaceValueExecutionInput {
  const SurfaceProgram *program{};
  const SurfaceValueStoragePlan *storage{};
  // Optional automatic-normal prefix. The compiler emits
  // `surface_normal_storage; transition(surface_normal_output); storage` as
  // one transaction. Prefix and root slot plans may overlap: the transition
  // consumes the prefix output before the first root instruction executes.
  const SurfaceValueStoragePlan *surface_normal_storage{};
  ValueExpressionId surface_normal_output{};
  bool surface_normal_uses_undisplaced_geometry{};
  // Optional for value-only programs. When present, closure bytecode is
  // lowered from the exact value-address image and endpoint projection
  // produced by this input; callers cannot accidentally pair a physical or
  // emission closure stream with a different typed slot plan.
  const SurfaceClosurePlan *closure_plan{};
  SurfaceClosureEndpointMask closure_endpoints{all_surface_closure_endpoints};
};

// `instruction_variants` is parallel to `values.instructions`. Execution
// interning compares the complete projected key, never a hash alone. The
// projection is exact and bit-preserving for every field not represented by
// typed instruction data (including NaN payloads and signed zero). Authored
// modes, flags, and static-table payloads remain in the bytecode image and are
// addressed by each instruction's immediate and metadata record.
struct SurfaceValueExecutableScene {
  bool valid{};
  std::string diagnostic;
  SurfaceValueSceneImage values;
  std::vector<SurfaceValueStaticVariant> variants;
  std::vector<std::uint32_t> instruction_variants;
};

// `active` must be transitively closed over ValueInstruction operands.
// `outputs` names values consumed after the stream (normally closure roots),
// and must be a subset of `active`.
[[nodiscard]] SurfaceValueStoragePlan
plan_surface_value_storage(const SurfaceProgram &program,
                           const std::vector<bool> &active,
                           const std::vector<bool> &outputs,
                           SurfaceValueStorageCapacity capacity = {});

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

} // namespace psycles::compiler
