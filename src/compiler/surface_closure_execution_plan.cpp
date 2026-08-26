#include <psycles/compiler/surface_execution_plan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

[[nodiscard]] SurfaceClosureProgramImage reject(std::string diagnostic) {
  SurfaceClosureProgramImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] bool closure_active(
    const SurfaceValueDependencyPlan &dependencies,
    ClosureExpressionId id,
    SurfaceClosureEndpointMask selected_endpoints) noexcept {
  return id.valid() &&
         id.value < dependencies.physical_closures.size() &&
         (((selected_endpoints & surface_closure_endpoint_bit(
                                     SurfaceClosureEndpoint::physical)) != 0u &&
           dependencies.physical_closures[id.value]) ||
          ((selected_endpoints & surface_closure_endpoint_bit(
                                     SurfaceClosureEndpoint::emission)) != 0u &&
           dependencies.emission_closures[id.value]));
}

[[nodiscard]] SurfaceClosureEndpointMask closure_endpoints(
    const SurfaceValueDependencyPlan &dependencies,
    ClosureExpressionId id,
    SurfaceClosureEndpointMask selected_endpoints) noexcept {
  auto result = SurfaceClosureEndpointMask{};
  if (dependencies.physical_closures[id.value]) {
    result |= surface_closure_endpoint_bit(
        SurfaceClosureEndpoint::physical);
  }
  if (dependencies.emission_closures[id.value]) {
    result |= surface_closure_endpoint_bit(
        SurfaceClosureEndpoint::emission);
  }
  return result & selected_endpoints;
}

[[nodiscard]] bool value_active(
    const SurfaceValueDependencyPlan &dependencies,
    ValueExpressionId id,
    SurfaceClosureEndpointMask selected_endpoints) noexcept {
  if (!id.valid() || id.value >= dependencies.preparation.size()) {
    return false;
  }
  return ((selected_endpoints & surface_closure_endpoint_bit(
                                    SurfaceClosureEndpoint::physical)) != 0u &&
          dependencies.physical[id.value]) ||
         ((selected_endpoints & surface_closure_endpoint_bit(
                                    SurfaceClosureEndpoint::emission)) != 0u &&
          dependencies.emission[id.value]);
}

inline constexpr auto emission_principled_features =
    principled_closure_feature_bit(PrincipledClosureFeature::alpha) |
    principled_closure_feature_bit(PrincipledClosureFeature::sheen) |
    principled_closure_feature_bit(PrincipledClosureFeature::coat) |
    principled_closure_feature_bit(PrincipledClosureFeature::emission);

[[nodiscard]] std::vector<ValueExpressionId> closure_operands(
    const ClosureInstruction &closure) {
  switch (closure.operation) {
  case ClosureOperation::null_closure:
  case ClosureOperation::add:
  case ClosureOperation::mix:
    return {};
  case ClosureOperation::diffuse:
    return {closure.color, closure.normal, closure.roughness};
  case ClosureOperation::translucent:
    return {closure.color, closure.normal};
  case ClosureOperation::principled:
    return {closure.color,
            closure.normal,
            closure.roughness,
            closure.diffuse_roughness,
            closure.subsurface_weight,
            closure.subsurface_radius,
            closure.subsurface_scale,
            closure.subsurface_ior,
            closure.subsurface_anisotropy,
            closure.transmission_weight,
            closure.metallic,
            closure.ior,
            closure.specular_ior_level,
            closure.specular_tint,
            closure.alpha,
            closure.thin_wall,
            closure.sheen_weight,
            closure.sheen_roughness,
            closure.sheen_tint,
            closure.coat_weight,
            closure.coat_roughness,
            closure.coat_ior,
            closure.coat_tint,
            closure.coat_normal,
            closure.emission_color,
            closure.emission_strength};
  case ClosureOperation::glossy:
    return {closure.color, closure.normal, closure.roughness};
  case ClosureOperation::glass:
  case ClosureOperation::refraction:
    return {closure.color, closure.normal, closure.roughness, closure.ior};
  case ClosureOperation::emission:
    return {closure.color, closure.strength};
  case ClosureOperation::transparent:
    return {closure.color};
  case ClosureOperation::subsurface:
    return {closure.color,
            closure.normal,
            closure.roughness,
            closure.subsurface_radius,
            closure.subsurface_scale,
            closure.subsurface_ior,
            closure.subsurface_anisotropy};
  }
  return {};
}

[[nodiscard]] bool valid_child(const SurfaceProgram &program,
                               ClosureExpressionId parent,
                               ClosureExpressionId child) noexcept {
  // SurfaceProgram's closure stream is topological. Requiring a strict
  // predecessor proves termination of the iterative DFS and rejects cycles
  // without a recursion-depth or visited-state heuristic.
  return child.valid() &&
         child.value < program.closure_instructions().size() &&
         child.value < parent.value;
}

enum class ClosureLoweringActionKind : std::uint8_t {
  visit,
  emit_mix_right,
  finish_mix,
};

struct ClosureLoweringAction {
  ClosureLoweringActionKind kind{ClosureLoweringActionKind::visit};
  ClosureExpressionId id{};
  std::uint32_t mix_begin{};
  std::uint32_t mix_slot{};
  std::uint32_t mix_depth{};
  // Continuation demand: the enclosing region evaluates another closure with
  // this region's entry weight after the current subtree completes.
  bool restore_after{};
};

[[nodiscard]] bool closure_leaf_operation(
    ClosureOperation operation) noexcept {
  switch (operation) {
  case ClosureOperation::diffuse:
  case ClosureOperation::translucent:
  case ClosureOperation::principled:
  case ClosureOperation::glossy:
  case ClosureOperation::glass:
  case ClosureOperation::emission:
  case ClosureOperation::transparent:
  case ClosureOperation::subsurface:
  case ClosureOperation::refraction:
    return true;
  case ClosureOperation::null_closure:
  case ClosureOperation::add:
  case ClosureOperation::mix:
    return false;
  }
  return false;
}

[[nodiscard]] SurfaceValueBank closure_operand_bank(
    ClosureOperation operation, std::size_t operand) noexcept {
  switch (operation) {
  case ClosureOperation::diffuse:
  case ClosureOperation::glossy:
    return operand == surface_closure_operand::diffuse::color ||
                   operand == surface_closure_operand::diffuse::normal
               ? SurfaceValueBank::vector
               : SurfaceValueBank::scalar;
  case ClosureOperation::translucent:
    return SurfaceValueBank::vector;
  case ClosureOperation::principled:
    switch (operand) {
    case surface_closure_operand::principled::color:
    case surface_closure_operand::principled::normal:
    case surface_closure_operand::principled::subsurface_radius:
    case surface_closure_operand::principled::specular_tint:
    case surface_closure_operand::principled::sheen_tint:
    case surface_closure_operand::principled::coat_tint:
    case surface_closure_operand::principled::coat_normal:
    case surface_closure_operand::principled::emission_color:
      return SurfaceValueBank::vector;
    default:
      return SurfaceValueBank::scalar;
    }
  case ClosureOperation::glass:
  case ClosureOperation::refraction:
    return operand == surface_closure_operand::glass::color ||
                   operand == surface_closure_operand::glass::normal
               ? SurfaceValueBank::vector
               : SurfaceValueBank::scalar;
  case ClosureOperation::emission:
    return operand == surface_closure_operand::emission::color
               ? SurfaceValueBank::vector
               : SurfaceValueBank::scalar;
  case ClosureOperation::transparent:
    return SurfaceValueBank::vector;
  case ClosureOperation::subsurface:
    return operand == surface_closure_operand::subsurface::color ||
                   operand == surface_closure_operand::subsurface::normal ||
                   operand == surface_closure_operand::subsurface::radius
               ? SurfaceValueBank::vector
               : SurfaceValueBank::scalar;
  case ClosureOperation::null_closure:
  case ClosureOperation::add:
  case ClosureOperation::mix:
    return SurfaceValueBank::scalar;
  }
  return SurfaceValueBank::scalar;
}

[[nodiscard]] bool address_fits_value_program(
    std::uint32_t encoded, SurfaceValueBank expected,
    const SurfaceValueProgramImage &values, bool allow_invalid) noexcept {
  const auto address = SurfaceValueAddress{encoded};
  if (!address.valid()) {
    return allow_invalid;
  }
  if (address.bank() != expected) {
    return false;
  }
  if (address.parameter()) {
    return true;
  }
  switch (address.bank()) {
  case SurfaceValueBank::scalar:
    return address.index() < values.scalar_slots;
  case SurfaceValueBank::vector:
    return address.index() < values.vector_slots;
  case SurfaceValueBank::unsigned_integer:
    return address.index() < values.unsigned_integer_slots;
  }
  return false;
}

[[nodiscard]] constexpr PrincipledClosureFeatureMask
all_principled_closure_features() noexcept {
  return principled_closure_feature_bit(PrincipledClosureFeature::alpha) |
         principled_closure_feature_bit(PrincipledClosureFeature::sheen) |
         principled_closure_feature_bit(PrincipledClosureFeature::coat) |
         principled_closure_feature_bit(PrincipledClosureFeature::metallic) |
         principled_closure_feature_bit(
             PrincipledClosureFeature::thick_transmission) |
         principled_closure_feature_bit(
             PrincipledClosureFeature::thin_transmission) |
         principled_closure_feature_bit(PrincipledClosureFeature::dielectric) |
         principled_closure_feature_bit(
             PrincipledClosureFeature::thick_subsurface) |
         principled_closure_feature_bit(
             PrincipledClosureFeature::thin_subsurface) |
         principled_closure_feature_bit(PrincipledClosureFeature::diffuse) |
         principled_closure_feature_bit(PrincipledClosureFeature::emission);
}

[[nodiscard]] std::string validate_surface_closure_program_image(
    const SurfaceClosureProgramImage &closures,
    const SurfaceValueProgramImage &values) {
  if (!closures.valid) {
    return closures.diagnostic.empty()
               ? "the closure program is invalid"
               : closures.diagnostic;
  }
  if (closures.principled_features.size() !=
      closures.instructions.size()) {
    return "the Principled feature stream is not parallel to closures";
  }

  struct OpenMixRegion {
    std::size_t right{};
    std::size_t end{};
    std::uint32_t mix_slot{};
    bool executing_right{};
    bool restores_parent{};
  };

  auto expected_operations = std::uint32_t{};
  auto expected_features = PrincipledClosureFeatureMask{};
  auto expected_maximum_mix_depth = std::uint32_t{};
  auto maximum_live_mix_slots = std::uint32_t{};
  auto live_mix_slot_count = std::uint32_t{};
  std::vector<bool> live_mix_slots(closures.mix_slots, false);
  std::vector<OpenMixRegion> open_regions;

  // A non-restoring Mix is legal only in tail position, so its logical end is
  // exactly the enclosing branch boundary. Closing such regions before the
  // boundary instruction models an implicit end marker without consuming a
  // device instruction. Multiple right-nested tail regions may share a
  // boundary and therefore close in strict LIFO order here.
  const auto close_tail_regions = [&](std::size_t boundary) {
    while (!open_regions.empty() &&
           open_regions.back().executing_right &&
           !open_regions.back().restores_parent &&
           open_regions.back().end == boundary) {
      const auto slot = open_regions.back().mix_slot;
      if (slot >= live_mix_slots.size() || !live_mix_slots[slot] ||
          live_mix_slot_count == 0u) {
        return false;
      }
      live_mix_slots[slot] = false;
      --live_mix_slot_count;
      open_regions.pop_back();
    }
    return true;
  };

  for (auto instruction_index = std::size_t{0u};
       instruction_index < closures.instructions.size();) {
    if (!close_tail_regions(instruction_index)) {
      return "an implicit tail Mix consumes a frame slot that is not live";
    }
    const auto enclosing_end = open_regions.empty()
                                   ? closures.instructions.size()
                                   : (open_regions.back().executing_right
                                          ? open_regions.back().end
                                          : open_regions.back().right);
    if (instruction_index > enclosing_end) {
      return "closure instruction " + std::to_string(instruction_index) +
             " crosses enclosing branch end " +
             std::to_string(enclosing_end) + " at structured depth " +
             std::to_string(open_regions.size());
    }

    const auto &instruction = closures.instructions[instruction_index];
    if ((instruction.control & ~surface_closure_control_mask) != 0u) {
      return "a closure control word contains undefined bits";
    }
    const auto kind = surface_closure_instruction_kind(instruction);
    if (!open_regions.empty() && instruction_index == enclosing_end) {
      const auto expected_kind = open_regions.back().executing_right
                                     ? SurfaceClosureInstructionKind::mix_end
                                     : SurfaceClosureInstructionKind::mix_right;
      if (kind != expected_kind) {
        return "closure instruction " + std::to_string(instruction_index) +
               " does not close its enclosing Mix branch at structured depth " +
               std::to_string(open_regions.size());
      }
    }
    if (kind == SurfaceClosureInstructionKind::mix_begin) {
      if (instruction.control != make_surface_closure_instruction_kind(kind)) {
        return "a Mix-begin instruction contains leaf semantic bits";
      }
      if (!address_fits_value_program(
              surface_closure_mix_factor_address(instruction),
              SurfaceValueBank::scalar, values, false)) {
        return "a Mix-begin factor has the wrong type or exceeds its bank";
      }
      const auto slot = surface_closure_mix_frame_slot(instruction);
      if (slot >= closures.mix_slots || live_mix_slots[slot]) {
        return "a Mix-begin frame slot is out of range or already live";
      }
      const auto right_offset =
          surface_closure_mix_right_offset(instruction);
      if (right_offset == 0u ||
          right_offset > enclosing_end - instruction_index) {
        return "a Mix-begin has an invalid right-region offset";
      }
      const auto right = instruction_index + right_offset;
      if (right >= enclosing_end) {
        return "a Mix-begin marker is outside its enclosing branch";
      }
      const auto &marker = closures.instructions[right];
      if (surface_closure_instruction_kind(marker) !=
              SurfaceClosureInstructionKind::mix_right ||
          marker.control != make_surface_closure_instruction_kind(
                                SurfaceClosureInstructionKind::mix_right) ||
          surface_closure_right_frame_slot(marker) != slot ||
          (marker.payload2 & ~surface_closure_mix_right_flags_mask) != 0u) {
        return "a Mix-begin does not target its exact right marker";
      }
      const auto end_offset = surface_closure_mix_end_offset(marker);
      if (end_offset == 0u || end_offset > enclosing_end - right) {
        return "a Mix-right has an invalid region-end offset";
      }
      const auto end = right + end_offset;
      const auto restores_parent =
          surface_closure_mix_restores_parent(marker);
      if (restores_parent) {
        if (end >= enclosing_end) {
          return "a restoring Mix-end marker is outside its enclosing branch";
        }
        const auto &end_marker = closures.instructions[end];
        if (surface_closure_instruction_kind(end_marker) !=
                SurfaceClosureInstructionKind::mix_end ||
            end_marker.control != make_surface_closure_instruction_kind(
                                      SurfaceClosureInstructionKind::mix_end) ||
            surface_closure_end_frame_slot(end_marker) != slot ||
            end_marker.payload1 != 0u || end_marker.payload2 != 0u) {
          return "a Mix-right does not target its exact end marker";
        }
      } else if (end != enclosing_end) {
        return "a non-restoring Mix does not consume its enclosing branch tail";
      }

      live_mix_slots[slot] = true;
      ++live_mix_slot_count;
      maximum_live_mix_slots =
          std::max(maximum_live_mix_slots, live_mix_slot_count);
      open_regions.emplace_back(OpenMixRegion{
          .right = right,
          .end = end,
          .mix_slot = slot,
          .restores_parent = restores_parent});
      expected_maximum_mix_depth = std::max(
          expected_maximum_mix_depth,
          static_cast<std::uint32_t>(open_regions.size()));
      ++instruction_index;
      continue;
    }
    if (kind == SurfaceClosureInstructionKind::mix_right) {
      if (open_regions.empty() ||
          open_regions.back().executing_right ||
          open_regions.back().right != instruction_index ||
          open_regions.back().mix_slot !=
              surface_closure_right_frame_slot(instruction) ||
          open_regions.back().restores_parent !=
              surface_closure_mix_restores_parent(instruction) ||
          (instruction.payload2 &
           ~surface_closure_mix_right_flags_mask) != 0u) {
        return "a Mix-right marker is unpaired or improperly nested";
      }
      const auto slot = open_regions.back().mix_slot;
      if (!live_mix_slots[slot] || live_mix_slot_count == 0u) {
        return "a Mix-right reads a frame slot that is not live";
      }
      open_regions.back().executing_right = true;
      ++instruction_index;
      continue;
    }
    if (kind == SurfaceClosureInstructionKind::mix_end) {
      if (open_regions.empty() ||
          !open_regions.back().executing_right ||
          !open_regions.back().restores_parent ||
          open_regions.back().end != instruction_index ||
          open_regions.back().mix_slot !=
              surface_closure_end_frame_slot(instruction)) {
        return "a Mix-end marker is unpaired or improperly nested";
      }
      const auto slot = open_regions.back().mix_slot;
      if (!live_mix_slots[slot] || live_mix_slot_count == 0u) {
        return "a Mix-end consumes a frame slot that is not live";
      }
      live_mix_slots[slot] = false;
      --live_mix_slot_count;
      open_regions.pop_back();
      ++instruction_index;
      continue;
    }
    if (kind != SurfaceClosureInstructionKind::leaf) {
      return "the closure stream contains an unknown instruction kind";
    }
    if (instruction.payload1 != 0u || instruction.payload2 != 0u) {
      return "a closure leaf contains nonzero control payload";
    }
    const auto operation = surface_closure_operation(instruction);
    if (!closure_leaf_operation(operation)) {
      return "the closure stream contains a non-leaf or unknown opcode";
    }
    const auto endpoints = surface_closure_endpoints(instruction);
    constexpr auto endpoint_mask =
        surface_closure_endpoint_bit(SurfaceClosureEndpoint::physical) |
        surface_closure_endpoint_bit(SurfaceClosureEndpoint::emission);
    if (endpoints == 0u || (endpoints & ~endpoint_mask) != 0u) {
      return "a closure leaf has an invalid endpoint mask";
    }
    if (static_cast<std::uint32_t>(
            surface_closure_bssrdf_method(instruction)) >
        static_cast<std::uint32_t>(BssrdfMethod::random_walk_skin)) {
      return "a closure leaf has an unknown BSSRDF method";
    }

    const auto operand_count = surface_closure_operand_count(operation);
    const auto operand_begin =
        surface_closure_leaf_operand_begin(instruction);
    if (operand_begin > closures.operands.size() ||
        operand_count >
            closures.operands.size() - operand_begin) {
      return "a closure leaf exceeds the operand stream";
    }
    for (auto operand = std::size_t{0u}; operand < operand_count;
         ++operand) {
      if (!address_fits_value_program(
              closures.operands[operand_begin + operand],
              closure_operand_bank(operation, operand), values, true)) {
        return "a closure operand has the wrong type or exceeds its bank";
      }
    }
    expected_operations |= 1u << static_cast<std::uint32_t>(operation);

    const auto features = closures.principled_features[instruction_index];
    if ((features & ~all_principled_closure_features()) != 0u ||
        (operation != ClosureOperation::principled && features != 0u)) {
      return "a closure leaf has an invalid Principled feature mask";
    }
    expected_features |= features;
    ++instruction_index;
  }
  if (!close_tail_regions(closures.instructions.size())) {
    return "an implicit tail Mix consumes a frame slot that is not live";
  }
  if (!open_regions.empty() || live_mix_slot_count != 0u) {
    return "the closure stream ends inside an open Mix region";
  }
  if (closures.maximum_mix_depth != expected_maximum_mix_depth) {
    return "the maximum closure Mix depth is inconsistent";
  }
  if (closures.mix_slots != maximum_live_mix_slots) {
    return "the closure Mix-frame allocation is not exact";
  }
  if (closures.used_operations != expected_operations) {
    return "the used closure-operation mask is inconsistent";
  }
  if (closures.used_principled_features != expected_features) {
    return "the used Principled feature mask is inconsistent";
  }
  return {};
}

[[nodiscard]] bool add_scene_extent(
    std::size_t &total, std::size_t count) noexcept {
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  if (total > limit || count > limit - total) {
    return false;
  }
  total += count;
  return true;
}

[[nodiscard]] SurfaceValueSceneImage reject_scene(
    std::string diagnostic) {
  SurfaceValueSceneImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

} // namespace

SurfaceClosureProgramImage lower_surface_closure_program(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan,
    const SurfaceValueDependencyPlan &dependencies,
    std::span<const std::uint32_t> value_addresses,
    SurfaceClosureEndpointMask selected_endpoints) {
  static_assert(
      std::is_trivially_copyable_v<SurfaceClosureBytecodeInstruction>);
  static_assert(sizeof(SurfaceClosureBytecodeInstruction) == 16u);
  static_assert(static_cast<std::uint32_t>(ClosureOperation::refraction) <
                32u);
  static_assert(static_cast<std::uint32_t>(BssrdfMethod::random_walk_skin) <
                4u);

  if (!closure_plan.compatible(program)) {
    return reject("the closure reachability plan is incompatible");
  }
  if (!dependencies.compatible(program)) {
    return reject("the closure dependency plan is incompatible");
  }
  if (selected_endpoints == 0u ||
      (selected_endpoints & ~all_surface_closure_endpoints) != 0u) {
    return reject("the closure endpoint projection is empty or invalid");
  }
  if (value_addresses.size() != program.value_instructions().size()) {
    return reject("the typed value-address image has the wrong size");
  }

  SurfaceClosureProgramImage result;
  const auto root = program.root();
  if (!root.valid()) {
    result.valid = true;
    return result;
  }
  if (root.value >= program.closure_instructions().size()) {
    return reject("the closure root is outside the instruction stream");
  }

  const auto encoded_address = [&](ValueExpressionId value,
                                   bool required,
                                   std::uint32_t &encoded) -> bool {
    encoded = SurfaceValueAddress::invalid_value;
    if (!value.valid()) {
      return !required;
    }
    if (value.value >= value_addresses.size()) {
      return false;
    }
    if (!value_active(dependencies, value, selected_endpoints)) {
      return !required;
    }
    const auto address = SurfaceValueAddress{value_addresses[value.value]};
    if (!address.valid() ||
        static_cast<std::uint32_t>(address.bank()) >
            static_cast<std::uint32_t>(
                SurfaceValueBank::unsigned_integer)) {
      return false;
    }
    encoded = address.encoded();
    return true;
  };

  std::vector<bool> occupied_mix_slots;
  auto live_mix_slots = std::uint32_t{};
  const auto allocate_mix_slot = [&]() noexcept {
    for (auto slot = std::size_t{};
         slot < occupied_mix_slots.size(); ++slot) {
      if (!occupied_mix_slots[slot]) {
        occupied_mix_slots[slot] = true;
        ++live_mix_slots;
        return static_cast<std::uint32_t>(slot);
      }
    }
    occupied_mix_slots.emplace_back(true);
    ++live_mix_slots;
    return static_cast<std::uint32_t>(occupied_mix_slots.size() - 1u);
  };
  const auto release_mix_slot = [&](std::uint32_t slot) noexcept {
    if (slot >= occupied_mix_slots.size() ||
        !occupied_mix_slots[slot] || live_mix_slots == 0u) {
      return false;
    }
    occupied_mix_slots[slot] = false;
    --live_mix_slots;
    return true;
  };
  const auto append_instruction = [&](
                                      SurfaceClosureBytecodeInstruction value,
                                      PrincipledClosureFeatureMask features = 0u)
      noexcept {
    if (result.instructions.size() >=
        std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    result.instructions.emplace_back(value);
    result.principled_features.emplace_back(features);
    return true;
  };

  std::vector<ClosureLoweringAction> pending;
  pending.emplace_back(ClosureLoweringAction{
      .kind = ClosureLoweringActionKind::visit,
      .id = root});
  while (!pending.empty()) {
    const auto current = pending.back();
    pending.pop_back();
    if (current.kind == ClosureLoweringActionKind::emit_mix_right) {
      if (current.mix_begin >= result.instructions.size() ||
          result.instructions.size() >=
              std::numeric_limits<std::uint32_t>::max()) {
        return reject("a closure Mix marker exceeds 32-bit offsets");
      }
      const auto marker = static_cast<std::uint32_t>(
          result.instructions.size());
      auto &begin = result.instructions[current.mix_begin];
      if (surface_closure_instruction_kind(begin) !=
              SurfaceClosureInstructionKind::mix_begin ||
          marker <= current.mix_begin) {
        return reject("the closure Mix lowering stack is inconsistent");
      }
      begin.payload2 = marker - current.mix_begin;
      if (!append_instruction(SurfaceClosureBytecodeInstruction{
              .control = make_surface_closure_instruction_kind(
                  SurfaceClosureInstructionKind::mix_right),
              .payload0 = current.mix_slot})) {
        return reject("a closure Mix marker exceeds 32-bit offsets");
      }
      continue;
    }
    if (current.kind == ClosureLoweringActionKind::finish_mix) {
      if (current.mix_begin >= result.instructions.size()) {
        return reject("a closure Mix region has no begin instruction");
      }
      const auto &begin = result.instructions[current.mix_begin];
      const auto marker_index = static_cast<std::size_t>(current.mix_begin) +
                                begin.payload2;
      if (marker_index >= result.instructions.size() ||
          result.instructions.size() >
              std::numeric_limits<std::uint32_t>::max()) {
        return reject("a closure Mix region has no right marker");
      }
      auto &marker = result.instructions[marker_index];
      if (surface_closure_instruction_kind(marker) !=
              SurfaceClosureInstructionKind::mix_right ||
          surface_closure_right_frame_slot(marker) != current.mix_slot) {
        return reject("a closure Mix right region is inconsistent");
      }
      marker.payload1 = static_cast<std::uint32_t>(
          result.instructions.size() - marker_index);
      if (current.restore_after) {
        marker.payload2 = surface_closure_mix_right_restores_parent;
        if (!append_instruction(SurfaceClosureBytecodeInstruction{
                .control = make_surface_closure_instruction_kind(
                    SurfaceClosureInstructionKind::mix_end),
                .payload0 = current.mix_slot})) {
          return reject("a closure Mix end exceeds 32-bit device offsets");
        }
      }
      if (!release_mix_slot(current.mix_slot)) {
        return reject("a closure Mix frame interval is not properly nested");
      }
      continue;
    }

    if (!closure_active(dependencies, current.id, selected_endpoints) ||
        !closure_plan.entry(current.id).reachable) {
      continue;
    }

    const auto &closure =
        program.closure_instructions()[current.id.value];
    if (closure.operation == ClosureOperation::add ||
        closure.operation == ClosureOperation::mix) {
      if (!valid_child(program, current.id, closure.a) ||
          !valid_child(program, current.id, closure.b)) {
        return reject(
            "a closure-tree edge is not a strict topological predecessor");
      }
      const auto a_reachable =
          closure_plan.entry(closure.a).reachable;
      const auto b_reachable =
          closure_plan.entry(closure.b).reachable;
      const auto a_active =
          a_reachable &&
          closure_active(dependencies, closure.a, selected_endpoints);
      const auto b_active =
          b_reachable &&
          closure_active(dependencies, closure.b, selected_endpoints);
      if (closure.operation == ClosureOperation::mix &&
          a_reachable && b_reachable && (a_active || b_active)) {
        std::uint32_t factor_address{};
        if (!encoded_address(
                closure.factor, true, factor_address)) {
          return reject(
              "a live closure Mix has no typed factor address");
        }
        if (occupied_mix_slots.size() >=
            std::numeric_limits<std::uint32_t>::max()) {
          return reject("a closure Mix exceeds 32-bit frame slots");
        }
        const auto mix_slot = allocate_mix_slot();
        result.mix_slots = std::max(result.mix_slots, live_mix_slots);
        const auto begin = static_cast<std::uint32_t>(
            result.instructions.size());
        if (!append_instruction(SurfaceClosureBytecodeInstruction{
                .control = make_surface_closure_instruction_kind(
                    SurfaceClosureInstructionKind::mix_begin),
                .payload0 = factor_address,
                .payload1 = mix_slot})) {
          return reject("a closure Mix exceeds 32-bit device offsets");
        }
        const auto child_depth = current.mix_depth + 1u;
        result.maximum_mix_depth =
            std::max(result.maximum_mix_depth, child_depth);

        pending.emplace_back(ClosureLoweringAction{
            .kind = ClosureLoweringActionKind::finish_mix,
            .mix_begin = begin,
            .mix_slot = mix_slot,
            .mix_depth = current.mix_depth,
            .restore_after = current.restore_after});
        if (b_active) {
          pending.emplace_back(ClosureLoweringAction{
              .kind = ClosureLoweringActionKind::visit,
              .id = closure.b,
              .mix_depth = child_depth,
              .restore_after = false});
        }
        pending.emplace_back(ClosureLoweringAction{
            .kind = ClosureLoweringActionKind::emit_mix_right,
            .mix_begin = begin,
            .mix_slot = mix_slot,
            .mix_depth = current.mix_depth});
        if (a_active) {
          pending.emplace_back(ClosureLoweringAction{
              .kind = ClosureLoweringActionKind::visit,
              .id = closure.a,
              .mix_depth = child_depth,
              .restore_after = false});
        }
        continue;
      }

      // LIFO insertion is reversed so the resulting leaf stream is the same
      // left-to-right depth-first order as GraphSurface::for_each_closure.
      if (b_active) {
        pending.emplace_back(ClosureLoweringAction{
            .kind = ClosureLoweringActionKind::visit,
            .id = closure.b,
            .mix_depth = current.mix_depth,
            .restore_after = current.restore_after});
      }
      if (a_active) {
        pending.emplace_back(ClosureLoweringAction{
            .kind = ClosureLoweringActionKind::visit,
            .id = closure.a,
            .mix_depth = current.mix_depth,
            .restore_after = b_active || current.restore_after});
      }
      continue;
    }
    if (closure.operation == ClosureOperation::null_closure) {
      continue;
    }

    const auto endpoints =
        closure_endpoints(dependencies, current.id, selected_endpoints);
    if (endpoints == 0u) {
      continue;
    }
    const auto operands = closure_operands(closure);
    if (operands.size() !=
        surface_closure_operand_count(closure.operation)) {
      return reject("a closure opcode has an inconsistent operand layout");
    }
    if (operands.size() >
            std::numeric_limits<std::uint32_t>::max() -
                result.operands.size()) {
      return reject("the closure program exceeds 32-bit device offsets");
    }

    const auto operand_begin =
        static_cast<std::uint32_t>(result.operands.size());
    for (const auto operand : operands) {
      std::uint32_t address{};
      const auto required =
          value_active(dependencies, operand, selected_endpoints);
      if (!encoded_address(operand, required, address)) {
        return reject("a live closure operand has no typed value address");
      }
      result.operands.emplace_back(address);
    }

    auto features = closure.operation == ClosureOperation::principled
        ? closure_plan.entry(current.id).principled_features
        : PrincipledClosureFeatureMask{};
    if ((endpoints & surface_closure_endpoint_bit(
                         SurfaceClosureEndpoint::physical)) == 0u) {
      features &= emission_principled_features;
    }
    if (!append_instruction(SurfaceClosureBytecodeInstruction{
            .control = make_surface_closure_control(closure, endpoints),
            .payload0 = operand_begin},
          features)) {
      return reject("the closure program exceeds 32-bit device offsets");
    }
    result.used_operations |=
        1u << static_cast<std::uint32_t>(closure.operation);
    result.used_principled_features |= features;
  }

  if (live_mix_slots != 0u ||
      std::any_of(occupied_mix_slots.begin(),
                  occupied_mix_slots.end(),
                  [](bool occupied) noexcept { return occupied; })) {
    return reject("the closure Mix lowering left a live frame interval");
  }
  result.valid = true;
  SurfaceValueProgramImage validation_values;
  validation_values.valid = true;
  for (const auto encoded : value_addresses) {
    const auto address = SurfaceValueAddress{encoded};
    if (!address.valid() || address.parameter()) {
      continue;
    }
    const auto extent = address.index() + 1u;
    switch (address.bank()) {
    case SurfaceValueBank::scalar:
      validation_values.scalar_slots =
          std::max(validation_values.scalar_slots, extent);
      break;
    case SurfaceValueBank::vector:
      validation_values.vector_slots =
          std::max(validation_values.vector_slots, extent);
      break;
    case SurfaceValueBank::unsigned_integer:
      validation_values.unsigned_integer_slots =
          std::max(validation_values.unsigned_integer_slots, extent);
      break;
    }
  }
  if (const auto diagnostic =
          validate_surface_closure_program_image(result, validation_values);
      !diagnostic.empty()) {
    return reject("internal closure verifier: " + diagnostic);
  }
  return result;
}

SurfaceValueSceneImage build_surface_execution_scene_image(
    std::span<const SurfaceValueProgramImage> value_programs,
    std::span<const SurfaceClosureProgramImage> closure_programs) {
  if (value_programs.size() != closure_programs.size()) {
    return reject_scene(
        "value and closure programs do not form a bijection");
  }

  auto closure_instruction_count = std::size_t{0u};
  auto closure_operand_count = std::size_t{0u};
  for (auto program_index = std::size_t{0u};
       program_index < value_programs.size(); ++program_index) {
    const auto diagnostic = validate_surface_closure_program_image(
        closure_programs[program_index], value_programs[program_index]);
    if (!diagnostic.empty()) {
      return reject_scene(
          "closure program " + std::to_string(program_index) + ": " +
          diagnostic);
    }
    if (!add_scene_extent(closure_instruction_count,
                          closure_programs[program_index]
                              .instructions.size()) ||
        !add_scene_extent(closure_operand_count,
                          closure_programs[program_index].operands.size())) {
      return reject_scene(
          "the aggregate closure program exceeds 32-bit device offsets");
    }
  }

  auto result = build_surface_value_scene_image(value_programs);
  if (!result.valid) {
    return result;
  }
  result.closure_instructions.reserve(closure_instruction_count);
  result.closure_principled_features.reserve(closure_instruction_count);
  result.closure_operands.reserve(closure_operand_count);

  for (auto program_index = std::size_t{0u};
       program_index < closure_programs.size(); ++program_index) {
    const auto &program = closure_programs[program_index];
    auto &descriptor = result.programs[program_index];
    descriptor.closure_begin = static_cast<std::uint32_t>(
        result.closure_instructions.size());
    descriptor.closure_count =
        static_cast<std::uint32_t>(program.instructions.size());
    const auto operand_begin =
        static_cast<std::uint32_t>(result.closure_operands.size());
    for (auto instruction : program.instructions) {
      if (surface_closure_is_leaf(instruction)) {
        instruction.payload0 += operand_begin;
      }
      result.closure_instructions.emplace_back(instruction);
    }
    result.closure_principled_features.insert(
        result.closure_principled_features.end(),
        program.principled_features.begin(),
        program.principled_features.end());
    result.closure_operands.insert(
        result.closure_operands.end(), program.operands.begin(),
        program.operands.end());
    result.maximum_closure_mix_depth = std::max(
        result.maximum_closure_mix_depth, program.maximum_mix_depth);
    result.maximum_closure_mix_slots = std::max(
        result.maximum_closure_mix_slots, program.mix_slots);
    result.used_closure_operations |= program.used_operations;
    result.used_principled_closure_features |=
        program.used_principled_features;
  }
  return result;
}

} // namespace psycles::compiler
