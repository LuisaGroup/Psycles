#include <psycles/compiler/surface_execution_plan.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <utility>

namespace psycles::compiler {
namespace {

[[nodiscard]] bool
decode_operand(const SurfaceValueSceneImage &image,
               const SurfaceValueBytecodeInstruction &instruction,
               std::size_t operand_index,
               SurfaceValueOperandAddress &operand) noexcept {
  const auto operand_count = surface_value_operand_count(instruction);
  if (operand_index >= operand_count) {
    return false;
  }
  const auto word_index = operand_index / surface_value_operands_per_word;
  const auto lane = operand_index % surface_value_operands_per_word;
  auto word = instruction.operand_payload;
  if (operand_count > surface_value_inline_operand_capacity) {
    if (instruction.operand_payload >= image.operands.size() ||
        word_index >= image.operands.size() - instruction.operand_payload) {
      return false;
    }
    word = image.operands[instruction.operand_payload + word_index];
  }
  operand = surface_value_operand_from_word(word, lane);
  return operand.valid();
}

} // namespace

SurfaceValueDefinitionLiveness analyze_surface_value_definition_liveness(
    const SurfaceValueSceneImage &image,
    const SurfaceValueProgramDescriptor &program) {
  SurfaceValueDefinitionLiveness result;
  const auto reject = [&](std::string diagnostic) {
    result.diagnostic = std::move(diagnostic);
    return result;
  };
  if (!image.valid) {
    return reject("the surface value scene image is invalid");
  }
  if (program.instruction_begin > image.instructions.size() ||
      program.instruction_count >
          image.instructions.size() - program.instruction_begin ||
      program.closure_begin > image.closure_instructions.size() ||
      program.closure_count >
          image.closure_instructions.size() - program.closure_begin) {
    return reject("the program descriptor exceeds the scene image");
  }

  result.last_use_offsets.assign(program.instruction_count,
                                 surface_value_definition_no_use);
  std::map<std::uint32_t, std::uint32_t> active_definitions;
  const auto record_local_use = [&](std::uint32_t encoded_address,
                                    std::uint32_t use_offset,
                                    bool invalid_is_optional = false) noexcept {
    const auto address = SurfaceValueAddress{encoded_address};
    if (!address.valid()) {
      return invalid_is_optional;
    }
    if (address.parameter()) {
      return true;
    }
    const auto definition = active_definitions.find(address.encoded());
    if (definition == active_definitions.end() ||
        definition->second >= result.last_use_offsets.size()) {
      return false;
    }
    result.last_use_offsets[definition->second] = use_offset;
    return true;
  };

  for (auto offset = std::uint32_t{}; offset < program.instruction_count;
       ++offset) {
    const auto &instruction =
        image.instructions[program.instruction_begin + offset];
    if (is_surface_value_surface_normal_transition(instruction)) {
      if (!record_local_use(instruction.result, offset)) {
        return reject("a normal commit reads an undefined local value");
      }
      active_definitions.clear();
      continue;
    }
    const auto operand_count = surface_value_operand_count(instruction);
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      auto operand = SurfaceValueOperandAddress{};
      if (!decode_operand(image, instruction, operand_index, operand)) {
        return reject("an ordinary instruction has an invalid operand");
      }
      if (!operand.parameter() &&
          !record_local_use(operand.expanded().encoded(), offset)) {
        return reject("an ordinary instruction reads an undefined local "
                      "value");
      }
    }
    const auto definition = SurfaceValueAddress{instruction.result};
    if (!definition.valid() || definition.parameter() ||
        definition.bank() != surface_value_result_bank(instruction)) {
      return reject("an ordinary instruction has an invalid local result");
    }
    // Operands are read before the result is written. insert_or_assign is
    // therefore the definition-epoch transition for an in-place colored slot.
    active_definitions.insert_or_assign(definition.encoded(), offset);
  }

  const auto terminal_use = program.instruction_count;
  for (auto closure_offset = std::uint32_t{};
       closure_offset < program.closure_count; ++closure_offset) {
    const auto &instruction =
        image.closure_instructions[program.closure_begin + closure_offset];
    if (!surface_closure_is_leaf(instruction)) {
      if (!record_local_use(surface_closure_mix_factor_address(instruction),
                            terminal_use)) {
        return reject("a closure Mix reads an undefined local factor");
      }
      continue;
    }
    const auto operation = surface_closure_operation(instruction);
    const auto operand_begin = surface_closure_leaf_operand_begin(instruction);
    const auto operand_count = surface_closure_operand_count(operation);
    if (operand_begin > image.closure_operands.size() ||
        operand_count > image.closure_operands.size() - operand_begin) {
      return reject("a closure leaf exceeds the operand stream");
    }
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      if (!record_local_use(
              image.closure_operands[operand_begin + operand_index],
              terminal_use, true)) {
        return reject("a closure leaf reads an undefined local value");
      }
    }
  }

  result.valid = true;
  return result;
}

SurfaceValueForwardingPlan plan_surface_value_forwarding(
    const SurfaceValueSceneImage &image,
    const SurfaceValueProgramDescriptor &program) {
  SurfaceValueForwardingPlan result;
  const auto reject = [&](std::string diagnostic) {
    result.diagnostic = std::move(diagnostic);
    return result;
  };
  const auto liveness =
      analyze_surface_value_definition_liveness(image, program);
  if (!liveness.valid) {
    return reject("definition liveness: " + liveness.diagnostic);
  }
  if (liveness.last_use_offsets.size() != program.instruction_count) {
    return reject("definition liveness is not parallel to the program");
  }
  result.successor_operand_masks.assign(program.instruction_count, 0u);

  // Let D_i be the definition produced by ordinary instruction i and U_i(j)
  // the definition read by successor operand j. Forwarding is legal exactly
  // when {j | U_i(j) = D_i} is non-empty and last_use(D_i) = i + 1. The first
  // equality supplies every substituted use; the second proves that omitting
  // D_i's bank write cannot affect any instruction or closure after the
  // successor. This is a local implementation of that relation, not an
  // opcode-specific peephole.
  for (auto source_offset = std::uint32_t{};
       source_offset + 1u < program.instruction_count; ++source_offset) {
    const auto target_offset = source_offset + 1u;
    const auto &source =
        image.instructions[program.instruction_begin + source_offset];
    const auto &target =
        image.instructions[program.instruction_begin + target_offset];
    if (is_surface_value_surface_normal_transition(source) ||
        is_surface_value_surface_normal_transition(target)) {
      continue;
    }
    const auto definition = SurfaceValueAddress{source.result};
    if (!definition.valid() || definition.parameter() ||
        definition.bank() != surface_value_result_bank(source)) {
      return reject("an ordinary source has an invalid local result");
    }

    auto operand_mask = std::uint32_t{};
    const auto operand_count = surface_value_operand_count(target);
    if (operand_count > surface_value_max_operand_count) {
      return reject("a successor exceeds the forwarding mask domain");
    }
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      auto operand = SurfaceValueOperandAddress{};
      if (!decode_operand(image, target, operand_index, operand)) {
        return reject("a successor has an invalid operand");
      }
      if (!operand.parameter() && operand.expanded() == definition) {
        operand_mask |= std::uint32_t{1u} << operand_index;
      }
    }
    if (liveness.last_use_offsets[source_offset] == target_offset) {
      if (operand_mask == 0u) {
        return reject("definition liveness names a successor without a use");
      }
      result.successor_operand_masks[source_offset] = operand_mask;
    }
  }

  result.valid = true;
  return result;
}

SurfaceValueRegionPlan plan_surface_value_regions(
    const SurfaceValueSceneImage &image,
    const SurfaceValueProgramDescriptor &program) {
  SurfaceValueRegionPlan result;
  const auto reject = [&](std::string diagnostic) {
    result.diagnostic = std::move(diagnostic);
    return result;
  };
  const auto liveness =
      analyze_surface_value_definition_liveness(image, program);
  if (!liveness.valid) {
    return reject("definition liveness: " + liveness.diagnostic);
  }
  const auto forwarding = plan_surface_value_forwarding(image, program);
  if (!forwarding.valid) {
    return reject("adjacent forwarding: " + forwarding.diagnostic);
  }
  if (liveness.last_use_offsets.size() != program.instruction_count ||
      forwarding.successor_operand_masks.size() !=
          program.instruction_count) {
    return reject("the prerequisite analyses are not parallel to the program");
  }

  result.successor_operand_masks = forwarding.successor_operand_masks;
  result.instruction_region_indices.assign(program.instruction_count,
                                           surface_value_no_region);

  // The legal-edge graph is a subgraph of the instruction path. Its maximal
  // connected components are therefore the unique non-overlapping partition
  // that realizes every forwarding edge. Normal commits have no definition,
  // clear the colored namespace, and cannot belong to such a component.
  for (auto offset = std::uint32_t{}; offset < program.instruction_count;) {
    const auto &instruction =
        image.instructions[program.instruction_begin + offset];
    if (is_surface_value_surface_normal_transition(instruction)) {
      ++offset;
      continue;
    }
    const auto region_index = static_cast<std::uint32_t>(result.regions.size());
    const auto begin = offset;
    auto end = offset;
    while (end + 1u < program.instruction_count &&
           forwarding.successor_operand_masks[end] != 0u) {
      const auto &successor =
          image.instructions[program.instruction_begin + end + 1u];
      if (is_surface_value_surface_normal_transition(successor)) {
        return reject("adjacent forwarding crosses a normal commit");
      }
      ++end;
    }
    result.regions.emplace_back(SurfaceValueRegionDescriptor{
        .instruction_begin_offset = begin,
        .instruction_count = end - begin + 1u,
        .live_input_definition_offsets = {},
        .live_output_definition_offsets = {},
        .operand_sources = std::vector<std::vector<
            SurfaceValueRegionOperandSource>>(end - begin + 1u)});
    for (auto member = begin; member <= end; ++member) {
      result.instruction_region_indices[member] = region_index;
    }
    offset = end + 1u;
  }

  // Replay the exact definition epochs. An operand whose active definition
  // predates its region is a true live-in. A definition whose final semantic
  // use follows its region is a true live-out. The liveness analysis already
  // proves every local read has an active epoch, but this replay intentionally
  // checks that invariant again so a future analysis change fails closed.
  std::map<std::uint32_t, std::uint32_t> active_definitions;
  const auto find_or_append = [](auto &values, std::uint32_t value) {
    const auto found = std::find(values.begin(), values.end(), value);
    if (found != values.end()) {
      return static_cast<std::uint32_t>(found - values.begin());
    }
    values.emplace_back(value);
    return static_cast<std::uint32_t>(values.size() - 1u);
  };
  for (auto offset = std::uint32_t{}; offset < program.instruction_count;
       ++offset) {
    const auto &instruction =
        image.instructions[program.instruction_begin + offset];
    if (is_surface_value_surface_normal_transition(instruction)) {
      active_definitions.clear();
      continue;
    }
    const auto region_index = result.instruction_region_indices[offset];
    if (region_index == surface_value_no_region ||
        region_index >= result.regions.size()) {
      return reject("an ordinary instruction has no region");
    }
    auto &region = result.regions[region_index];
    const auto region_end = region.instruction_begin_offset +
                            region.instruction_count - 1u;
    auto &operand_sources =
        region.operand_sources[offset - region.instruction_begin_offset];
    const auto operand_count = surface_value_operand_count(instruction);
    operand_sources.reserve(operand_count);
    for (auto operand_index = std::size_t{}; operand_index < operand_count;
         ++operand_index) {
      auto operand = SurfaceValueOperandAddress{};
      if (!decode_operand(image, instruction, operand_index, operand)) {
        return reject("an ordinary instruction has an invalid operand");
      }
      if (operand.parameter()) {
        operand_sources.emplace_back(SurfaceValueRegionOperandSource{
            .kind = SurfaceValueRegionOperandSourceKind::parameter,
            .index = operand.index()});
        continue;
      }
      const auto definition = active_definitions.find(
          operand.expanded().encoded());
      if (definition == active_definitions.end()) {
        return reject("an ordinary instruction reads an undefined epoch");
      }
      if (definition->second < region.instruction_begin_offset) {
        const auto input_index = find_or_append(
            region.live_input_definition_offsets, definition->second);
        operand_sources.emplace_back(SurfaceValueRegionOperandSource{
            .kind = SurfaceValueRegionOperandSourceKind::live_input,
            .index = input_index});
      } else {
        if (definition->second >= offset) {
          return reject("a region operand does not precede its use");
        }
        operand_sources.emplace_back(SurfaceValueRegionOperandSource{
            .kind = SurfaceValueRegionOperandSourceKind::instruction_result,
            .index = definition->second -
                     region.instruction_begin_offset});
      }
    }

    const auto definition = SurfaceValueAddress{instruction.result};
    if (!definition.valid() || definition.parameter()) {
      return reject("an ordinary instruction has an invalid local result");
    }
    active_definitions.insert_or_assign(definition.encoded(), offset);
    const auto final_use = liveness.last_use_offsets[offset];
    if (final_use != surface_value_definition_no_use &&
        final_use > region_end) {
      region.live_output_definition_offsets.emplace_back(offset);
    }
  }

  // The symbolic data-flow graph and the forwarding masks are two independent
  // projections of the same epoch relation. Requiring exact agreement here
  // prevents either representation from becoming a permissive approximation.
  for (const auto &region : result.regions) {
    for (auto source = std::uint32_t{};
         source + 1u < region.instruction_count; ++source) {
      const auto source_offset = region.instruction_begin_offset + source;
      const auto mask = result.successor_operand_masks[source_offset];
      if (mask == 0u) {
        return reject("a maximal region contains a non-forwarding edge");
      }
      const auto &target_sources = region.operand_sources[source + 1u];
      auto symbolic_mask = std::uint32_t{};
      for (auto operand = std::size_t{}; operand < target_sources.size();
           ++operand) {
        const auto &source_ref = target_sources[operand];
        if (source_ref.kind ==
                SurfaceValueRegionOperandSourceKind::instruction_result &&
            source_ref.index == source) {
          symbolic_mask |= std::uint32_t{1u} << operand;
        }
      }
      if (symbolic_mask != mask) {
        return reject("region data flow disagrees with its forwarding mask");
      }
    }
  }

  result.valid = true;
  return result;
}

SurfaceValueRegionSpecializationPlan
plan_surface_value_region_specializations(
    const SurfaceValueExecutableScene &scene,
    std::uint32_t handler_site_budget) {
  SurfaceValueRegionSpecializationPlan result;
  result.handler_site_budget = handler_site_budget;
  const auto reject = [&](std::string diagnostic) {
    result.diagnostic = std::move(diagnostic);
    result.specializations.clear();
    result.instruction_specialization_indices.clear();
    result.selected_handler_sites = 0u;
    result.eliminated_bank_accesses = 0u;
    return result;
  };
  const auto &image = scene.values;
  if (!scene.valid || !image.valid) {
    return reject("the executable surface scene is invalid");
  }
  if (scene.instruction_variants.size() != image.instructions.size()) {
    return reject("the instruction-variant stream is not parallel");
  }
  result.instruction_specialization_indices.assign(
      image.instructions.size(), surface_value_no_region);
  if (handler_site_budget == 0u) {
    result.valid = true;
    return result;
  }

  struct Candidate {
    SurfaceValueRegionShape shape;
    std::vector<std::uint32_t> instruction_begins;
    std::uint64_t eliminated_bank_accesses{};
  };
  std::map<SurfaceValueRegionShape, Candidate> candidates;
  for (const auto &program : image.programs) {
    const auto regions = plan_surface_value_regions(image, program);
    if (!regions.valid) {
      return reject("region planning: " + regions.diagnostic);
    }
    for (const auto &region : regions.regions) {
      if (region.instruction_count < 2u ||
          region.instruction_begin_offset >= program.instruction_count ||
          region.instruction_count >
              program.instruction_count - region.instruction_begin_offset ||
          region.operand_sources.size() != region.instruction_count) {
        if (region.instruction_count < 2u) {
          continue;
        }
        return reject("a candidate region has inconsistent extents");
      }
      SurfaceValueRegionShape shape;
      shape.variant_indices.reserve(region.instruction_count);
      shape.operand_sources = region.operand_sources;
      for (auto &sources : shape.operand_sources) {
        for (auto &source : sources) {
          if (source.kind ==
              SurfaceValueRegionOperandSourceKind::parameter) {
            source.index = 0u;
          }
        }
      }
      shape.live_input_banks.reserve(
          region.live_input_definition_offsets.size());
      for (const auto definition_offset :
           region.live_input_definition_offsets) {
        if (definition_offset >= region.instruction_begin_offset ||
            definition_offset >= program.instruction_count) {
          return reject("a candidate live input is not before its region");
        }
        const auto &definition = image.instructions[
            program.instruction_begin + definition_offset];
        if (is_surface_value_surface_normal_transition(definition)) {
          return reject("a candidate live input is a normal commit");
        }
        shape.live_input_banks.emplace_back(
            surface_value_result_bank(definition));
      }
      shape.live_output_instruction_offsets.reserve(
          region.live_output_definition_offsets.size());
      for (const auto definition_offset :
           region.live_output_definition_offsets) {
        if (definition_offset < region.instruction_begin_offset ||
            definition_offset >= region.instruction_begin_offset +
                                     region.instruction_count) {
          return reject("a candidate live output is outside its region");
        }
        shape.live_output_instruction_offsets.emplace_back(
            definition_offset - region.instruction_begin_offset);
      }

      std::uint64_t occurrence_accesses = 0u;
      for (auto member = std::uint32_t{};
           member < region.instruction_count; ++member) {
        const auto instruction_index = program.instruction_begin +
                                       region.instruction_begin_offset + member;
        if (instruction_index >= scene.instruction_variants.size() ||
            is_surface_value_surface_normal_transition(
                image.instructions[instruction_index])) {
          return reject("a candidate contains an invalid ordinary instruction");
        }
        const auto variant = scene.instruction_variants[instruction_index];
        if (variant >= scene.variants.size()) {
          return reject("a candidate names an invalid static variant");
        }
        shape.variant_indices.emplace_back(variant);
        if (member + 1u < region.instruction_count) {
          const auto mask = regions.successor_operand_masks[
              region.instruction_begin_offset + member];
          if (mask == 0u) {
            return reject("a candidate contains a non-forwarding edge");
          }
          occurrence_accesses +=
              1u + static_cast<std::uint64_t>(std::popcount(mask));
        }
      }

      const auto instruction_begin =
          program.instruction_begin + region.instruction_begin_offset;
      auto [candidate, inserted] = candidates.try_emplace(
          shape,
          Candidate{.shape = shape,
                    .instruction_begins = {},
                    .eliminated_bank_accesses = 0u});
      auto &entry = candidate->second;
      entry.instruction_begins.emplace_back(instruction_begin);
      if (entry.eliminated_bank_accesses >
          std::numeric_limits<std::uint64_t>::max() - occurrence_accesses) {
        return reject("candidate benefit overflows uint64");
      }
      entry.eliminated_bank_accesses += occurrence_accesses;
      static_cast<void>(inserted);
    }
  }

  struct KnapsackState {
    std::uint64_t benefit{};
    std::uint32_t cost{};
    std::vector<std::size_t> selected;
  };
  std::vector<const Candidate *> ordered_candidates;
  ordered_candidates.reserve(candidates.size());
  std::uint64_t total_cost = 0u;
  for (const auto &[shape, candidate] : candidates) {
    static_cast<void>(shape);
    ordered_candidates.emplace_back(&candidate);
    total_cost += candidate.shape.variant_indices.size();
  }
  const auto effective_budget = static_cast<std::uint32_t>(std::min<
      std::uint64_t>(handler_site_budget, total_cost));
  std::vector<KnapsackState> states(effective_budget + 1u);
  const auto better = [](const KnapsackState &lhs,
                         const KnapsackState &rhs) noexcept {
    if (lhs.benefit != rhs.benefit) {
      return lhs.benefit > rhs.benefit;
    }
    if (lhs.cost != rhs.cost) {
      return lhs.cost < rhs.cost;
    }
    return lhs.selected < rhs.selected;
  };
  for (auto candidate_index = std::size_t{};
       candidate_index < ordered_candidates.size(); ++candidate_index) {
    const auto *candidate = ordered_candidates[candidate_index];
    const auto cost = static_cast<std::uint32_t>(
        candidate->shape.variant_indices.size());
    for (auto capacity = effective_budget; capacity >= cost; --capacity) {
      auto proposal = states[capacity - cost];
      if (proposal.benefit >
          std::numeric_limits<std::uint64_t>::max() -
              candidate->eliminated_bank_accesses) {
        return reject("knapsack benefit overflows uint64");
      }
      proposal.benefit += candidate->eliminated_bank_accesses;
      proposal.cost += cost;
      proposal.selected.emplace_back(candidate_index);
      if (better(proposal, states[capacity])) {
        states[capacity] = std::move(proposal);
      }
    }
  }
  auto optimum = KnapsackState{};
  for (const auto &state : states) {
    if (better(state, optimum)) {
      optimum = state;
    }
  }

  result.specializations.reserve(optimum.selected.size());
  // The current maximal-region construction is a partition, but make that
  // theorem an independently checked lowering precondition. The public side
  // stream marks only beginnings; this full owner map prevents a future
  // candidate-construction change from silently selecting overlapping
  // interiors.
  std::vector<std::uint32_t> selected_instruction_owners(
      image.instructions.size(), surface_value_no_region);
  for (const auto candidate_index : optimum.selected) {
    const auto &candidate = *ordered_candidates[candidate_index];
    if (candidate.instruction_begins.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      return reject("candidate occurrence count exceeds uint32");
    }
    const auto specialization_index =
        static_cast<std::uint32_t>(result.specializations.size());
    result.specializations.emplace_back(SurfaceValueRegionSpecialization{
        .shape = candidate.shape,
        .static_occurrences = static_cast<std::uint32_t>(
            candidate.instruction_begins.size()),
        .eliminated_bank_accesses = candidate.eliminated_bank_accesses,
        .handler_site_cost = static_cast<std::uint32_t>(
            candidate.shape.variant_indices.size())});
    for (const auto instruction_begin : candidate.instruction_begins) {
      if (instruction_begin >=
          result.instruction_specialization_indices.size()) {
        return reject("selected region beginning is outside the instruction "
                      "stream");
      }
      const auto instruction_count = static_cast<std::uint32_t>(
          candidate.shape.variant_indices.size());
      if (instruction_count > image.instructions.size() - instruction_begin) {
        return reject("selected region extent is outside the instruction "
                      "stream");
      }
      for (auto member = std::uint32_t{}; member < instruction_count; ++member) {
        auto &owner = selected_instruction_owners[instruction_begin + member];
        if (owner != surface_value_no_region) {
          return reject("selected region occurrences overlap");
        }
        owner = specialization_index;
      }
      if (result.instruction_specialization_indices[instruction_begin] !=
          surface_value_no_region) {
        return reject("selected region occurrences overlap");
      }
      result.instruction_specialization_indices[instruction_begin] =
          specialization_index;
    }
  }
  result.selected_handler_sites = optimum.cost;
  result.eliminated_bank_accesses = optimum.benefit;
  result.valid = true;
  return result;
}

} // namespace psycles::compiler
