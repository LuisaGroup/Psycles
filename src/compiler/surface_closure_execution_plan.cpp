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

struct PendingClosure {
  ClosureExpressionId id;
  std::vector<SurfaceClosureMixTerm> mix_path;
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

  auto expected_operations = std::uint32_t{};
  auto expected_features = PrincipledClosureFeatureMask{};
  auto expected_maximum_mix_depth = std::uint32_t{};
  for (auto instruction_index = std::size_t{0u};
       instruction_index < closures.instructions.size();
       ++instruction_index) {
    const auto &instruction = closures.instructions[instruction_index];
    if ((instruction.control & ~surface_closure_control_mask) != 0u) {
      return "a closure control word contains undefined bits";
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
    if (instruction.operand_begin > closures.operands.size() ||
        operand_count >
            closures.operands.size() - instruction.operand_begin) {
      return "a closure leaf exceeds the operand stream";
    }
    for (auto operand = std::size_t{0u}; operand < operand_count;
         ++operand) {
      if (!address_fits_value_program(
              closures.operands[instruction.operand_begin + operand],
              closure_operand_bank(operation, operand), values, true)) {
        return "a closure operand has the wrong type or exceeds its bank";
      }
    }
    if (instruction.mix_term_begin > closures.mix_terms.size() ||
        instruction.mix_term_count >
            closures.mix_terms.size() - instruction.mix_term_begin) {
      return "a closure leaf exceeds the Mix-term stream";
    }
    expected_maximum_mix_depth = std::max(
        expected_maximum_mix_depth, instruction.mix_term_count);
    expected_operations |= 1u << static_cast<std::uint32_t>(operation);

    const auto features = closures.principled_features[instruction_index];
    if ((features & ~all_principled_closure_features()) != 0u ||
        (operation != ClosureOperation::principled && features != 0u)) {
      return "a closure leaf has an invalid Principled feature mask";
    }
    expected_features |= features;
  }
  for (const auto &term : closures.mix_terms) {
    if (!address_fits_value_program(
            term.address, SurfaceValueBank::scalar, values, false) ||
        (term.flags & ~surface_closure_mix_flags_mask) != 0u) {
      return "a closure Mix term is invalid";
    }
  }
  if (closures.maximum_mix_depth != expected_maximum_mix_depth) {
    return "the maximum closure Mix depth is inconsistent";
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
  static_assert(std::is_trivially_copyable_v<SurfaceClosureMixTerm>);
  static_assert(sizeof(SurfaceClosureBytecodeInstruction) == 16u);
  static_assert(sizeof(SurfaceClosureMixTerm) == 8u);
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

  std::vector<PendingClosure> pending;
  pending.emplace_back(PendingClosure{.id = root, .mix_path = {}});
  while (!pending.empty()) {
    auto current = std::move(pending.back());
    pending.pop_back();
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

      auto a_path = current.mix_path;
      auto b_path = std::move(current.mix_path);
      if (closure.operation == ClosureOperation::mix &&
          a_reachable && b_reachable && (a_active || b_active)) {
        std::uint32_t factor_address{};
        if (!encoded_address(
                closure.factor, true, factor_address)) {
          return reject(
              "a live closure Mix has no typed factor address");
        }
        a_path.emplace_back(SurfaceClosureMixTerm{
            .address = factor_address,
            .flags = surface_closure_mix_complement});
        b_path.emplace_back(SurfaceClosureMixTerm{
            .address = factor_address});
      }

      // LIFO insertion is reversed so the resulting leaf stream is the same
      // left-to-right depth-first order as GraphSurface::for_each_closure.
      if (b_active) {
        pending.emplace_back(PendingClosure{
            .id = closure.b,
            .mix_path = std::move(b_path)});
      }
      if (a_active) {
        pending.emplace_back(PendingClosure{
            .id = closure.a,
            .mix_path = std::move(a_path)});
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
    if (result.instructions.size() >=
            std::numeric_limits<std::uint32_t>::max() ||
        operands.size() >
            std::numeric_limits<std::uint32_t>::max() -
                result.operands.size() ||
        current.mix_path.size() >
            std::numeric_limits<std::uint32_t>::max() -
                result.mix_terms.size()) {
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
    const auto mix_term_begin =
        static_cast<std::uint32_t>(result.mix_terms.size());
    result.mix_terms.insert(result.mix_terms.end(),
                            current.mix_path.begin(),
                            current.mix_path.end());
    result.maximum_mix_depth = std::max(
        result.maximum_mix_depth,
        static_cast<std::uint32_t>(current.mix_path.size()));

    auto features = closure.operation == ClosureOperation::principled
        ? closure_plan.entry(current.id).principled_features
        : PrincipledClosureFeatureMask{};
    if ((endpoints & surface_closure_endpoint_bit(
                         SurfaceClosureEndpoint::physical)) == 0u) {
      features &= emission_principled_features;
    }
    result.instructions.emplace_back(SurfaceClosureBytecodeInstruction{
        .control = make_surface_closure_control(closure, endpoints),
        .operand_begin = operand_begin,
        .mix_term_begin = mix_term_begin,
        .mix_term_count =
            static_cast<std::uint32_t>(current.mix_path.size())});
    result.principled_features.emplace_back(features);
    result.used_operations |=
        1u << static_cast<std::uint32_t>(closure.operation);
    result.used_principled_features |= features;
  }

  if (result.principled_features.size() != result.instructions.size()) {
    return reject(
        "the Principled feature stream is not parallel to closures");
  }
  for (const auto &term : result.mix_terms) {
    if (term.address == SurfaceValueAddress::invalid_value ||
        (term.flags & ~surface_closure_mix_flags_mask) != 0u) {
      return reject("a flattened closure Mix term is invalid");
    }
  }
  result.valid = true;
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
  auto closure_mix_term_count = std::size_t{0u};
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
                          closure_programs[program_index].operands.size()) ||
        !add_scene_extent(closure_mix_term_count,
                          closure_programs[program_index]
                              .mix_terms.size())) {
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
  result.closure_mix_terms.reserve(closure_mix_term_count);

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
    const auto mix_term_begin =
        static_cast<std::uint32_t>(result.closure_mix_terms.size());
    for (auto instruction : program.instructions) {
      instruction.operand_begin += operand_begin;
      instruction.mix_term_begin += mix_term_begin;
      result.closure_instructions.emplace_back(instruction);
    }
    result.closure_principled_features.insert(
        result.closure_principled_features.end(),
        program.principled_features.begin(),
        program.principled_features.end());
    result.closure_operands.insert(
        result.closure_operands.end(), program.operands.begin(),
        program.operands.end());
    result.closure_mix_terms.insert(
        result.closure_mix_terms.end(), program.mix_terms.begin(),
        program.mix_terms.end());
    result.maximum_closure_mix_depth = std::max(
        result.maximum_closure_mix_depth, program.maximum_mix_depth);
    result.used_closure_operations |= program.used_operations;
    result.used_principled_closure_features |=
        program.used_principled_features;
  }
  return result;
}

} // namespace psycles::compiler
