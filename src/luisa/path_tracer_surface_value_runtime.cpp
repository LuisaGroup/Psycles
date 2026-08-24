#include "path_tracer_surface_values.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <luisa/core/logging.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::vector<bool> dependency_mask(
    const compiler::SurfaceProgram &program,
    compiler::ValueExpressionId root) {
    std::vector<bool> active(
        program.value_instructions().size(), false);
    std::vector<compiler::ValueExpressionId> pending;
    if (root.valid()) {
        pending.emplace_back(root);
    }
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (!id.valid() ||
            id.value >= active.size() ||
            active[id.value]) {
            continue;
        }
        active[id.value] = true;
        for (const auto operand :
             program.value_instructions()[id.value].operands) {
            if (operand.valid()) {
                pending.emplace_back(operand);
            }
        }
    }
    return active;
}

[[nodiscard]] bool fits_runtime_capacity(
    const compiler::SurfaceValueProgramDescriptor &program) noexcept {
    return program.scalar_slots <= SurfaceValueRuntime::scalar_capacity &&
           program.vector_slots <= SurfaceValueRuntime::vector_capacity &&
           program.unsigned_integer_slots <=
               SurfaceValueRuntime::unsigned_integer_capacity;
}

template<typename T>
void provide_dummy_if_empty(luisa::vector<T> &values, T dummy) {
    if (values.empty()) {
        values.emplace_back(std::move(dummy));
    }
}

} // namespace

std::unique_ptr<SurfaceValueRuntime> build_surface_value_runtime(
    luisa::compute::Device &device,
    std::span<const std::shared_ptr<const compiler::SurfaceProgram>> programs,
    std::span<const compiler::SurfaceClosurePlan> closure_plans,
    std::span<const std::uint32_t> bssrdf_bump_tags,
    std::string &diagnostic) {
    diagnostic.clear();
    if (programs.empty() || programs.size() != closure_plans.size()) {
        diagnostic = "surface topology programs and closure plans do not form a "
                     "non-empty bijection";
        return nullptr;
    }
    if (programs.size() >
        std::numeric_limits<std::uint32_t>::max() /
            SurfaceValueRuntime::programs_per_topology) {
        diagnostic = "surface topology count exceeds compact device program ids";
        return nullptr;
    }

    std::vector<bool> bssrdf_topologies(programs.size(), false);
    auto bssrdf_topology_count = std::size_t{0u};
    for (const auto tag : bssrdf_bump_tags) {
        if (tag >= programs.size()) {
            diagnostic =
                "BSSRDF-bump topology tag exceeds compact surface programs";
            return nullptr;
        }
        if (!bssrdf_topologies[tag]) {
            bssrdf_topologies[tag] = true;
            ++bssrdf_topology_count;
        }
    }

    auto runtime = std::make_unique<SurfaceValueRuntime>();
    runtime->topologies.reserve(programs.size());
    std::vector<compiler::SurfaceValueStoragePlan> root_storage;
    root_storage.reserve(
        programs.size() * SurfaceValueRuntime::programs_per_topology);

    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        const auto &program_ptr = programs[topology];
        if (!program_ptr || !closure_plans[topology].compatible(*program_ptr)) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " has no compatible closure plan";
            return nullptr;
        }
        const auto &program = *program_ptr;
        auto normal_active =
            dependency_mask(program, program.surface_normal_root());
        auto normal_outputs =
            std::vector<bool>(normal_active.size(), false);
        if (program.surface_normal_root().valid()) {
            if (program.surface_normal_root().value >= normal_outputs.size()) {
                diagnostic = "surface topology " + std::to_string(topology) +
                             " has an invalid automatic-normal root";
                return nullptr;
            }
            normal_outputs[program.surface_normal_root().value] = true;
        }
        auto normal_storage = compiler::plan_surface_value_storage(
            program, normal_active, normal_outputs);
        if (!normal_storage.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " automatic-normal plan: " +
                         normal_storage.diagnostic;
            return nullptr;
        }
        auto normal_image = compiler::lower_surface_value_program(
            program, normal_storage);
        if (!normal_image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " automatic-normal lowering: " +
                         normal_image.diagnostic;
            return nullptr;
        }

        const auto dependencies =
            compiler::analyze_surface_value_dependencies(
                program, closure_plans[topology]);
        auto preparation_storage =
            compiler::plan_surface_value_storage(
                program,
                dependencies.preparation,
                dependencies.preparation_outputs);
        if (!preparation_storage.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " preparation plan: " +
                         preparation_storage.diagnostic;
            return nullptr;
        }
        auto preparation_image =
            compiler::lower_surface_value_program(
                program, preparation_storage);
        if (!preparation_image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " preparation lowering: " +
                         preparation_image.diagnostic;
            return nullptr;
        }
    auto emission_storage = compiler::plan_surface_value_storage(
        program, dependencies.emission, dependencies.emission_outputs);
    if (!emission_storage.valid) {
      diagnostic = "surface topology " + std::to_string(topology) +
                   " emission plan: " + emission_storage.diagnostic;
      return nullptr;
    }
    const auto emission_image =
        compiler::lower_surface_value_program(program, emission_storage);
    if (!emission_image.valid) {
      diagnostic = "surface topology " + std::to_string(topology) +
                   " emission lowering: " + emission_image.diagnostic;
      return nullptr;
    }

        auto normal_output =
            compiler::SurfaceValueAddress::invalid_value;
        if (program.surface_normal_root().valid()) {
            normal_output = normal_image.value_addresses[
                program.surface_normal_root().value];
            if (normal_output ==
                compiler::SurfaceValueAddress::invalid_value) {
                diagnostic = "surface topology " +
                             std::to_string(topology) +
                             " automatic-normal root has no typed address";
                return nullptr;
            }
        }
        auto uses_undisplaced_geometry = false;
        for (auto index = std::size_t{0u};
             index < normal_active.size(); ++index) {
            const auto &instruction =
                program.value_instructions()[index];
            uses_undisplaced_geometry |=
                normal_active[index] &&
                instruction.operation ==
                    compiler::ValueOperation::bump &&
                (instruction.static_u0 & 4u) != 0u;
        }
        runtime->topologies.emplace_back(
            SurfaceValueRuntimeTopology{
                .program = program_ptr,
                .preparation_addresses =
                    std::move(preparation_image.value_addresses),
                .normal_output_address = normal_output,
                .automatic_bump_uses_undisplaced_geometry =
                    uses_undisplaced_geometry});
        runtime->normal_output_addresses.emplace_back(normal_output);
        runtime->topology_flags.emplace_back(
            uses_undisplaced_geometry ? surface_value_runtime_topology_flag(
                                            SurfaceValueRuntimeTopologyFlag::
                                                automatic_bump_uses_undisplaced_geometry)
                                      : 0u);
        if (dependencies.emission_observes_shading_normal() &&
            program.surface_normal_root().valid()) {
            runtime->topology_flags.back() |=
                surface_value_runtime_topology_flag(
                    SurfaceValueRuntimeTopologyFlag::
                        emission_uses_automatic_normal);
        }
        root_storage.emplace_back(std::move(normal_storage));
        root_storage.emplace_back(std::move(preparation_storage));
    root_storage.emplace_back(std::move(emission_storage));
    }

    std::vector<compiler::SurfaceValueExecutionInput> roots;
    roots.reserve(root_storage.size());
    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        roots.emplace_back(compiler::SurfaceValueExecutionInput{
            .program = programs[topology].get(),
            .storage = &root_storage[
                topology * SurfaceValueRuntime::programs_per_topology +
                SurfaceValueRuntime::normal_program_offset]});
        roots.emplace_back(compiler::SurfaceValueExecutionInput{
            .program = programs[topology].get(),
            .storage = &root_storage[
                topology * SurfaceValueRuntime::programs_per_topology +
                SurfaceValueRuntime::preparation_program_offset],
            .closure_plan = &closure_plans[topology]});
    roots.emplace_back(compiler::SurfaceValueExecutionInput{
        .program = programs[topology].get(),
        .storage =
            &root_storage[topology *
                              SurfaceValueRuntime::programs_per_topology +
                          SurfaceValueRuntime::emission_program_offset],
        .closure_plan = &closure_plans[topology],
        .closure_endpoints = compiler::surface_closure_endpoint_bit(
            compiler::SurfaceClosureEndpoint::emission)});
    }
    runtime->executable =
        compiler::build_surface_value_bump_executable_scene(roots);
    if (!runtime->executable.valid) {
        diagnostic = runtime->executable.diagnostic;
        return nullptr;
    }
    if (runtime->executable.root_program_count != roots.size()) {
        diagnostic = "compact surface root program count changed during Bump stratification";
        return nullptr;
    }
    const auto &image = runtime->executable.executable.values;
    runtime->used_principled_closure_features =
        image.used_principled_closure_features;
    runtime->closure_static_variants.reserve(
        image.closure_instructions.size());
    for (const auto &instruction : image.closure_instructions) {
        const auto key = instruction.control &
                         compiler::surface_closure_static_variant_mask;
        if (std::find(
                runtime->closure_static_variants.begin(),
                runtime->closure_static_variants.end(),
                key) == runtime->closure_static_variants.end()) {
            runtime->closure_static_variants.emplace_back(key);
        }
    }
    std::sort(
        runtime->closure_static_variants.begin(),
        runtime->closure_static_variants.end());
  if (runtime->executable.executable.instruction_variants.size() !=
          image.instructions.size() ||
      runtime->executable.bump_height_programs.size() !=
          image.instructions.size() ||
      image.closure_principled_features.size() !=
          image.closure_instructions.size()) {
    diagnostic = "compact surface semantic side streams are not parallel";
    return nullptr;
  }
  const auto append_unique = [](auto &values, auto value) noexcept {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
      values.emplace_back(value);
    }
  };
  const auto valid_program_range = [&image](std::uint32_t program) noexcept {
    if (program >= image.programs.size()) {
      return false;
    }
    const auto &range = image.programs[program];
    return range.instruction_begin <= image.instructions.size() &&
           range.instruction_count <=
               image.instructions.size() - range.instruction_begin &&
           range.closure_begin <= image.closure_instructions.size() &&
           range.closure_count <=
               image.closure_instructions.size() - range.closure_begin;
  };
  const auto collect_direct_values =
      [&](std::uint32_t program, std::vector<std::uint32_t> &variants,
          std::vector<std::uint32_t> &pending_height_programs) noexcept {
        if (!valid_program_range(program)) {
          return false;
        }
        const auto &range = image.programs[program];
        for (auto offset = std::uint32_t{0u}; offset < range.instruction_count;
             ++offset) {
          const auto instruction = range.instruction_begin + offset;
          append_unique(
              variants,
              runtime->executable.executable.instruction_variants[instruction]);
          const auto height_program =
              runtime->executable.bump_height_programs[instruction];
          if (height_program != compiler::SurfaceValueAddress::invalid_value) {
            pending_height_programs.emplace_back(height_program);
          }
        }
        return true;
      };
  const auto close_height_domain =
      [&](std::vector<std::uint32_t> pending,
          std::vector<std::uint32_t> &variants) noexcept {
        std::vector<bool> visited(image.programs.size(), false);
        while (!pending.empty()) {
          const auto program = pending.back();
          pending.pop_back();
          if (!valid_program_range(program)) {
            return false;
          }
          if (visited[program]) {
            continue;
          }
          visited[program] = true;
          const auto &range = image.programs[program];
          // Bump height programs are scalar value subroutines. A
          // closure range here would violate the finite-strata call
          // graph and make endpoint-local reachability ill-defined.
          if (range.closure_count != 0u) {
            return false;
          }
          for (auto offset = std::uint32_t{0u};
               offset < range.instruction_count; ++offset) {
            const auto instruction = range.instruction_begin + offset;
            append_unique(variants, runtime->executable.executable
                                        .instruction_variants[instruction]);
            const auto child =
                runtime->executable.bump_height_programs[instruction];
            if (child != compiler::SurfaceValueAddress::invalid_value) {
              pending.emplace_back(child);
            }
          }
        }
        return true;
      };

  std::vector<std::uint32_t> pending_preparation_height_programs;
  std::vector<std::uint32_t> pending_emission_height_programs;
  std::vector<std::uint32_t> pending_bssrdf_height_programs;
  pending_preparation_height_programs.reserve(programs.size() * 2u);
  pending_emission_height_programs.reserve(programs.size() * 2u);
  pending_bssrdf_height_programs.reserve(bssrdf_topology_count * 2u);
  for (auto topology = std::size_t{0u}; topology < programs.size();
       ++topology) {
    const auto base = static_cast<std::uint32_t>(
        topology * SurfaceValueRuntime::programs_per_topology);
    const auto normal_program =
        base + SurfaceValueRuntime::normal_program_offset;
    const auto preparation_program =
        base + SurfaceValueRuntime::preparation_program_offset;
    const auto emission_program =
        base + SurfaceValueRuntime::emission_program_offset;
    if (!collect_direct_values(normal_program,
                               runtime->normal_value_static_variants,
                               pending_preparation_height_programs) ||
        !collect_direct_values(preparation_program,
                               runtime->preparation_value_static_variants,
                               pending_preparation_height_programs) ||
        !collect_direct_values(emission_program,
                               runtime->emission_value_static_variants,
                               pending_emission_height_programs)) {
      diagnostic = "surface root program range exceeds its stream";
      return nullptr;
    }
    if ((runtime->topology_flags[topology] &
         surface_value_runtime_topology_flag(
             SurfaceValueRuntimeTopologyFlag::
                 emission_uses_automatic_normal)) != 0u &&
        !collect_direct_values(normal_program,
                               runtime->emission_normal_value_static_variants,
                               pending_emission_height_programs)) {
      diagnostic = "emission automatic-normal program range exceeds its stream";
      return nullptr;
    }
    if (bssrdf_topologies[topology] &&
        (!collect_direct_values(normal_program,
                                runtime->bssrdf_normal_value_static_variants,
                                pending_bssrdf_height_programs) ||
         !collect_direct_values(preparation_program,
                                runtime->bssrdf_value_static_variants,
                                pending_bssrdf_height_programs))) {
      diagnostic = "BSSRDF topology program range exceeds its stream";
      return nullptr;
    }
    const auto &range = image.programs[emission_program];
    for (auto offset = std::uint32_t{0u}; offset < range.closure_count;
         ++offset) {
      const auto closure = range.closure_begin + offset;
      const auto &instruction = image.closure_instructions[closure];
      const auto endpoints = compiler::surface_closure_endpoints(instruction);
      const auto operation = compiler::surface_closure_operation(instruction);
      if (endpoints != compiler::surface_closure_endpoint_bit(
                           compiler::SurfaceClosureEndpoint::emission) ||
          (operation != compiler::ClosureOperation::emission &&
           operation != compiler::ClosureOperation::principled)) {
        diagnostic = "emission projection retained a non-emission closure";
        return nullptr;
      }
      append_unique(runtime->emission_closure_static_variants,
                    instruction.control &
                        compiler::surface_closure_emission_static_variant_mask);
      runtime->emission_principled_closure_features |=
          image.closure_principled_features[closure];
    }
    if (bssrdf_topologies[topology]) {
      const auto &bssrdf_range = image.programs[preparation_program];
      for (auto offset = std::uint32_t{0u};
           offset < bssrdf_range.closure_count; ++offset) {
        const auto closure = bssrdf_range.closure_begin + offset;
        const auto &instruction = image.closure_instructions[closure];
        if (compiler::surface_closure_endpoints(instruction) == 0u) {
          diagnostic = "BSSRDF topology retained an endpoint-free closure";
          return nullptr;
        }
        append_unique(runtime->bssrdf_closure_static_variants,
                      instruction.control &
                          compiler::surface_closure_static_variant_mask);
        runtime->bssrdf_principled_closure_features |=
            image.closure_principled_features[closure];
      }
    }
  }
  if (!close_height_domain(std::move(pending_preparation_height_programs),
                           runtime->height_value_static_variants) ||
      !close_height_domain(std::move(pending_emission_height_programs),
                           runtime->emission_height_value_static_variants) ||
      !close_height_domain(std::move(pending_bssrdf_height_programs),
                           runtime->bssrdf_height_value_static_variants)) {
    diagnostic = "compact surface Bump-height call graph is malformed";
    return nullptr;
  }
  const auto sort_variants = [](auto &variants) noexcept {
    std::sort(variants.begin(), variants.end());
  };
  sort_variants(runtime->preparation_value_static_variants);
  sort_variants(runtime->normal_value_static_variants);
  sort_variants(runtime->height_value_static_variants);
  sort_variants(runtime->emission_value_static_variants);
  sort_variants(runtime->emission_normal_value_static_variants);
  sort_variants(runtime->emission_height_value_static_variants);
  sort_variants(runtime->bssrdf_value_static_variants);
  sort_variants(runtime->bssrdf_normal_value_static_variants);
  sort_variants(runtime->bssrdf_height_value_static_variants);
  std::sort(runtime->emission_closure_static_variants.begin(),
            runtime->emission_closure_static_variants.end());
  std::sort(runtime->bssrdf_closure_static_variants.begin(),
            runtime->bssrdf_closure_static_variants.end());

  for (auto index = std::size_t{0u}; index < image.programs.size(); ++index) {
        if (!fits_runtime_capacity(image.programs[index])) {
            diagnostic = "compact surface program " + std::to_string(index) +
                         " requires typed slots beyond the 8 scalar / 12 vector / 1 "
                         "uint64 validation capacity";
            return nullptr;
        }
        runtime->program_ranges.emplace_back(luisa::make_uint4(
            image.programs[index].instruction_begin,
            image.programs[index].instruction_count,
            image.programs[index].closure_begin,
            image.programs[index].closure_count));
    }
    runtime->instructions.reserve(image.instructions.size());
    for (const auto &instruction : image.instructions) {
        runtime->instructions.emplace_back(luisa::make_uint4(
            instruction.control,
            instruction.result,
            instruction.operand_begin,
            instruction.metadata_index));
    }
    runtime->operands.assign(
        image.operands.begin(), image.operands.end());
    runtime->instruction_variants.assign(
        runtime->executable.executable.instruction_variants.begin(),
        runtime->executable.executable.instruction_variants.end());
    runtime->metadata_parameters.reserve(image.metadata.size());
    runtime->metadata_static_ranges.reserve(image.metadata.size());
    for (const auto &metadata : image.metadata) {
        runtime->metadata_parameters.emplace_back(metadata.parameter);
        runtime->metadata_static_ranges.emplace_back(luisa::make_uint2(
            metadata.static_table_begin,
            metadata.static_table_count));
    }
    runtime->static_data.assign(
        image.static_data.begin(), image.static_data.end());
    runtime->closure_instructions.reserve(
        image.closure_instructions.size());
    for (const auto &instruction : image.closure_instructions) {
        runtime->closure_instructions.emplace_back(luisa::make_uint4(
            instruction.control,
            instruction.operand_begin,
            instruction.mix_term_begin,
            instruction.mix_term_count));
    }
    runtime->closure_operands.assign(
        image.closure_operands.begin(), image.closure_operands.end());
    runtime->closure_mix_terms.reserve(
        image.closure_mix_terms.size());
    for (const auto &term : image.closure_mix_terms) {
        runtime->closure_mix_terms.emplace_back(
            luisa::make_uint2(term.address, term.flags));
    }
    runtime->bump_height_programs.assign(
        runtime->executable.bump_height_programs.begin(),
        runtime->executable.bump_height_programs.end());
    runtime->program_outputs.assign(
        runtime->executable.program_outputs.begin(),
        runtime->executable.program_outputs.end());

    provide_dummy_if_empty(
        runtime->program_ranges, luisa::make_uint4(0u));
    provide_dummy_if_empty(
        runtime->instructions, luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->operands, 0u);
    provide_dummy_if_empty(runtime->instruction_variants, 0u);
    provide_dummy_if_empty(
        runtime->metadata_parameters,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(
        runtime->metadata_static_ranges, luisa::make_uint2(0u));
    provide_dummy_if_empty(runtime->static_data, 0.0f);
    provide_dummy_if_empty(
        runtime->closure_instructions, luisa::make_uint4(0u));
    provide_dummy_if_empty(
        runtime->closure_operands,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(
        runtime->closure_mix_terms, luisa::make_uint2(0u));
    provide_dummy_if_empty(
        runtime->bump_height_programs,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(
        runtime->program_outputs,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(
        runtime->normal_output_addresses,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(runtime->topology_flags, 0u);

    auto maximum_instruction_count = std::uint32_t{0u};
    auto maximum_scalar_slots = std::uint32_t{0u};
    auto maximum_vector_slots = std::uint32_t{0u};
    auto maximum_unsigned_integer_slots = std::uint32_t{0u};
    for (const auto &program : image.programs) {
        maximum_instruction_count = std::max(
            maximum_instruction_count, program.instruction_count);
        maximum_scalar_slots = std::max(
            maximum_scalar_slots, program.scalar_slots);
        maximum_vector_slots = std::max(
            maximum_vector_slots, program.vector_slots);
        maximum_unsigned_integer_slots = std::max(
            maximum_unsigned_integer_slots,
            program.unsigned_integer_slots);
    }
    LUISA_INFO(
        "Built compact surface runtime: {} root programs, {} total programs, "
        "{} instructions, {} operands, {} metadata records, {} static floats, "
      "{} semantic variants (population {}/{}/{}, emission {}/{}/{}, "
      "BSSRDF {} tags {}/{}/{}), {} closure instructions "
      "(population/emission/BSSRDF variants {}/{}/{}), {} Bump evaluator "
        "strata, maximum program "
        "length {}, typed slots {}/{}/{}.",
      runtime->executable.root_program_count, image.programs.size(),
      image.instructions.size(), image.operands.size(), image.metadata.size(),
      image.static_data.size(), runtime->executable.executable.variants.size(),
      runtime->preparation_value_static_variants.size(),
      runtime->normal_value_static_variants.size(),
      runtime->height_value_static_variants.size(),
      runtime->emission_value_static_variants.size(),
      runtime->emission_normal_value_static_variants.size(),
      runtime->emission_height_value_static_variants.size(),
      bssrdf_topology_count,
      runtime->bssrdf_value_static_variants.size(),
      runtime->bssrdf_normal_value_static_variants.size(),
      runtime->bssrdf_height_value_static_variants.size(),
      image.closure_instructions.size(),
      runtime->closure_static_variants.size(),
      runtime->emission_closure_static_variants.size(),
      runtime->bssrdf_closure_static_variants.size(),
      runtime->executable.maximum_bump_depth, maximum_instruction_count,
        maximum_scalar_slots, maximum_vector_slots,
        maximum_unsigned_integer_slots);

    runtime->program_buffer =
        device.create_buffer<luisa::uint4>(runtime->program_ranges.size());
    runtime->instruction_buffer =
        device.create_buffer<luisa::uint4>(runtime->instructions.size());
    runtime->operand_buffer =
        device.create_buffer<luisa::uint>(runtime->operands.size());
    runtime->instruction_variant_buffer =
        device.create_buffer<luisa::uint>(
            runtime->instruction_variants.size());
    runtime->metadata_parameter_buffer =
        device.create_buffer<luisa::uint>(
            runtime->metadata_parameters.size());
    runtime->metadata_static_range_buffer =
        device.create_buffer<luisa::uint2>(
            runtime->metadata_static_ranges.size());
    runtime->static_data_buffer =
        device.create_buffer<float>(runtime->static_data.size());
    runtime->closure_instruction_buffer =
        device.create_buffer<luisa::uint4>(
            runtime->closure_instructions.size());
    runtime->closure_operand_buffer =
        device.create_buffer<luisa::uint>(
            runtime->closure_operands.size());
    runtime->closure_mix_term_buffer =
        device.create_buffer<luisa::uint2>(
            runtime->closure_mix_terms.size());
    runtime->bump_height_program_buffer =
        device.create_buffer<luisa::uint>(
            runtime->bump_height_programs.size());
    runtime->program_output_buffer =
        device.create_buffer<luisa::uint>(runtime->program_outputs.size());
    runtime->normal_output_address_buffer =
        device.create_buffer<luisa::uint>(
            runtime->normal_output_addresses.size());
    runtime->topology_flag_buffer =
        device.create_buffer<luisa::uint>(
            runtime->topology_flags.size());
    runtime->device_view = device.create_bindless_array(
        surface_value_runtime_buffer_slot_count);
    const auto bind = [&](SurfaceValueRuntimeBufferSlot slot,
                          const auto &buffer) noexcept {
        runtime->device_view.emplace_on_update(
            surface_value_runtime_buffer_slot(slot), buffer);
    };
    bind(SurfaceValueRuntimeBufferSlot::program,
         runtime->program_buffer);
    bind(SurfaceValueRuntimeBufferSlot::instruction,
         runtime->instruction_buffer);
    bind(SurfaceValueRuntimeBufferSlot::operand,
         runtime->operand_buffer);
    bind(SurfaceValueRuntimeBufferSlot::instruction_variant,
         runtime->instruction_variant_buffer);
    bind(SurfaceValueRuntimeBufferSlot::metadata_parameter,
         runtime->metadata_parameter_buffer);
    bind(SurfaceValueRuntimeBufferSlot::metadata_static_range,
         runtime->metadata_static_range_buffer);
    bind(SurfaceValueRuntimeBufferSlot::static_data,
         runtime->static_data_buffer);
    bind(SurfaceValueRuntimeBufferSlot::closure_instruction,
         runtime->closure_instruction_buffer);
    bind(SurfaceValueRuntimeBufferSlot::closure_operand,
         runtime->closure_operand_buffer);
    bind(SurfaceValueRuntimeBufferSlot::closure_mix_term,
         runtime->closure_mix_term_buffer);
    bind(SurfaceValueRuntimeBufferSlot::bump_height_program,
         runtime->bump_height_program_buffer);
    bind(SurfaceValueRuntimeBufferSlot::program_output,
         runtime->program_output_buffer);
    bind(SurfaceValueRuntimeBufferSlot::normal_output_address,
         runtime->normal_output_address_buffer);
    bind(SurfaceValueRuntimeBufferSlot::topology_flag,
         runtime->topology_flag_buffer);
    return runtime;
}

void upload_surface_value_runtime(
    Stream &stream,
    SurfaceValueRuntime &runtime) noexcept {
    stream << runtime.program_buffer.copy_from(
                  luisa::span{runtime.program_ranges})
           << runtime.instruction_buffer.copy_from(
                  luisa::span{runtime.instructions})
           << runtime.operand_buffer.copy_from(
                  luisa::span{runtime.operands})
           << runtime.instruction_variant_buffer.copy_from(
                  luisa::span{runtime.instruction_variants})
           << runtime.metadata_parameter_buffer.copy_from(
                  luisa::span{runtime.metadata_parameters})
           << runtime.metadata_static_range_buffer.copy_from(
                  luisa::span{runtime.metadata_static_ranges})
           << runtime.static_data_buffer.copy_from(
                  luisa::span{runtime.static_data})
           << runtime.closure_instruction_buffer.copy_from(
                  luisa::span{runtime.closure_instructions})
           << runtime.closure_operand_buffer.copy_from(
                  luisa::span{runtime.closure_operands})
           << runtime.closure_mix_term_buffer.copy_from(
                  luisa::span{runtime.closure_mix_terms})
           << runtime.bump_height_program_buffer.copy_from(
                  luisa::span{runtime.bump_height_programs})
           << runtime.program_output_buffer.copy_from(
                  luisa::span{runtime.program_outputs})
           << runtime.normal_output_address_buffer.copy_from(
                  luisa::span{runtime.normal_output_addresses})
           << runtime.topology_flag_buffer.copy_from(
                  luisa::span{runtime.topology_flags})
           << runtime.device_view.update();
}

} // namespace psycles::luisa_backend::detail
