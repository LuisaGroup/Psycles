#include "path_tracer_internal.h"
#include "sample_dispatch_partition.h"

#include <psycles/luisa/volume_guiding.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>

namespace psycles::luisa_backend::detail {

PathDiagnosticBufferLayout path_diagnostic_buffer_layout(
    const LuisaPathTracerOptions &options,
    std::size_t surface_value_topology_count) noexcept {
  PathDiagnosticBufferLayout layout;
  layout.path_trace_slot_count =
      options.path_trace ? path_trace_schema::slot_count : 0u;
  layout.surface_closure_count_histogram_base = layout.path_trace_slot_count;
  layout.surface_closure_count_histogram_slot_count =
      options.surface_closure_count_histogram
          ? luisa_surface_closure_count_histogram_bin_count
          : 0u;
  layout.surface_program_execution_histogram_base =
      layout.surface_closure_count_histogram_base +
      layout.surface_closure_count_histogram_slot_count;
  if (options.surface_program_execution_histogram &&
      surface_value_topology_count != 0u) {
    if (surface_value_topology_count >
        std::numeric_limits<std::size_t>::max() /
            surface_program_execution_histogram_shards_per_topology) {
      std::abort();
    }
    layout.surface_program_execution_histogram_slot_count =
        surface_value_topology_count *
        surface_program_execution_histogram_shards_per_topology;
  }
  if (layout.surface_program_execution_histogram_base >
      std::numeric_limits<std::size_t>::max() -
          layout.surface_program_execution_histogram_slot_count) {
    std::abort();
  }
  layout.allocation_slot_count = std::max<std::size_t>(
      layout.surface_program_execution_histogram_base +
          layout.surface_program_execution_histogram_slot_count,
      1u);
  if (layout.allocation_slot_count >
      std::numeric_limits<std::uint32_t>::max()) {
    std::abort();
  }
  return layout;
}

std::size_t LuisaRenderSession::pixel_count() const noexcept {
  return static_cast<std::size_t>(_window.width) *
         static_cast<std::size_t>(_window.height);
}

std::size_t
LuisaRenderSession::surface_program_execution_histogram_topology_count()
    const noexcept {
  if (!_options.surface_program_execution_histogram ||
      !_scene->populate_surface_once || !_scene->surface_values) {
    return 0u;
  }
  return _scene->surface_values->topologies.size();
}

void LuisaRenderSession::deliver_path_trace() {
  if (_path_trace_delivered || !_options.path_trace ||
      !_options.path_trace->sink) {
    return;
  }
  luisa::vector<luisa::float4> slots(path_trace_schema::slot_count);
  _stream << _path_trace.view(0u, path_trace_schema::slot_count)
                 .copy_to(luisa::span{slots})
          << synchronize();
  LuisaPathTrace trace{.pixel_x = _options.path_trace->pixel_x,
                       .pixel_y = _options.path_trace->pixel_y,
                       .sample = _options.path_trace->sample};
  for (auto index = std::size_t{0u}; index < slots.size(); ++index) {
    trace.slots[index] = {slots[index].x, slots[index].y, slots[index].z,
                          slots[index].w};
  }
  _options.path_trace->sink->write(trace);
  _path_trace_delivered = true;
}

void LuisaRenderSession::deliver_surface_closure_count_histogram() {
  if (!_options.surface_closure_count_histogram ||
      !_options.surface_closure_count_histogram->sink) {
    return;
  }
  luisa::vector<luisa::float4> bins(
      luisa_surface_closure_count_histogram_bin_count);
  const auto layout = path_diagnostic_buffer_layout(
      _options, surface_program_execution_histogram_topology_count());
  _stream << _path_trace
                 .view(layout.surface_closure_count_histogram_base, bins.size())
                 .copy_to(luisa::span{bins})
          << synchronize();

  constexpr auto largest_consecutive_float_integer = 16777216.0f;
  LuisaSurfaceClosureCountHistogram histogram;
  histogram.exact = true;
  for (auto bin = std::size_t{0u}; bin < bins.size(); ++bin) {
    const std::array lanes{bins[bin].x, bins[bin].y, bins[bin].z, bins[bin].w};
    for (const auto value : lanes) {
      const auto lane_exact = std::isfinite(value) && value >= 0.0f &&
                              value < largest_consecutive_float_integer &&
                              std::trunc(value) == value;
      histogram.exact &= lane_exact;
      // Only exact lanes are converted. Besides making an invalid
      // result unusable by construction, this avoids an out-of-range
      // float-to-uint conversion if a corrupted device counter is a
      // large but finite value.
      if (lane_exact) {
        histogram.counts[bin] += static_cast<std::uint64_t>(value);
      }
    }
  }
  _options.surface_closure_count_histogram->sink->write(histogram);
}

void LuisaRenderSession::deliver_surface_program_execution_histogram() {
  if (!_options.surface_program_execution_histogram ||
      !_options.surface_program_execution_histogram->sink) {
    return;
  }

  LuisaSurfaceProgramExecutionHistogram histogram;
  const auto topology_count =
      surface_program_execution_histogram_topology_count();
  if (topology_count == 0u || !_scene->surface_values) {
    _options.surface_program_execution_histogram->sink->write(histogram);
    return;
  }
  const auto layout = path_diagnostic_buffer_layout(_options, topology_count);
  luisa::vector<luisa::float4> shards(
      layout.surface_program_execution_histogram_slot_count);
  _stream << _path_trace
                 .view(layout.surface_program_execution_histogram_base,
                       shards.size())
                 .copy_to(luisa::span{shards})
          << synchronize();

  constexpr auto largest_consecutive_float_integer = 16777216.0f;
  histogram.exact = true;
  histogram.topology_surface_populations.assign(topology_count, 0u);
  for (auto topology = std::size_t{}; topology < topology_count; ++topology) {
    auto &count = histogram.topology_surface_populations[topology];
    const auto begin =
        topology * surface_program_execution_histogram_shards_per_topology;
    const auto end =
        begin + surface_program_execution_histogram_shards_per_topology;
    for (auto shard = begin; shard < end; ++shard) {
      const std::array lanes{shards[shard].x, shards[shard].y, shards[shard].z,
                             shards[shard].w};
      for (const auto value : lanes) {
        const auto lane_exact = std::isfinite(value) && value >= 0.0f &&
                                value < largest_consecutive_float_integer &&
                                std::trunc(value) == value;
        histogram.exact &= lane_exact;
        if (lane_exact) {
          count += static_cast<std::uint64_t>(value);
        }
      }
    }
  }

  const auto checked_add = [&](std::uint64_t &destination,
                               std::uint64_t value) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - destination) {
      histogram.exact = false;
      return;
    }
    destination += value;
  };
  const auto checked_weighted_add = [&](std::uint64_t &destination,
                                        std::uint64_t count,
                                        std::uint64_t weight) noexcept {
    if (count != 0u &&
        weight > std::numeric_limits<std::uint64_t>::max() / count) {
      histogram.exact = false;
      return;
    }
    checked_add(destination, count * weight);
  };

  // These are exact counts in the static unified-PC projection weighted by the
  // measured number of preparation invocations per topology. Structured guard
  // bodies are deliberately not mislabeled as dynamically observed PC visits:
  // the current diagnostic records topology populations, not branch outcomes.
  using ValueKey = std::array<std::uint32_t, 5u>;
  std::map<ValueKey, std::uint64_t> value_counts;
  using ValueTransitionKey = std::array<std::uint32_t, 10u>;
  std::map<ValueTransitionKey, std::uint64_t> value_transition_counts;
  std::map<std::uint32_t, std::uint64_t> closure_leaf_counts;

  const auto &runtime = *_scene->surface_values;
  const auto &image = runtime.svm_scene;
  if (!image.valid ||
      runtime.svm_instruction_variants.size() != image.instructions.size() ||
      image.programs.size() != runtime.topologies.size() *
                                   SurfaceValueRuntime::programs_per_topology) {
    histogram.exact = false;
  }

  for (auto topology = std::size_t{}; topology < topology_count; ++topology) {
    const auto populations = histogram.topology_surface_populations[topology];
    if (populations == 0u) {
      continue;
    }
    const auto program_index =
        topology * SurfaceValueRuntime::programs_per_topology +
        SurfaceValueRuntime::preparation_program_offset;
    if (program_index >= image.programs.size()) {
      histogram.exact = false;
      continue;
    }
    const auto &program = image.programs[program_index];
    if (program.instruction_begin > image.instructions.size() ||
        program.instruction_count >
            image.instructions.size() - program.instruction_begin) {
      histogram.exact = false;
      continue;
    }

    const auto decode_operand =
        [&](const compiler::SurfaceValueBytecodeInstruction &instruction,
            std::size_t operand_index,
            compiler::SurfaceValueOperandAddress &operand) noexcept {
          const auto operand_count =
              compiler::surface_value_operand_count(instruction);
          if (operand_index >= operand_count) {
            return false;
          }
          const auto word_index =
              operand_index / compiler::surface_value_operands_per_word;
          const auto lane =
              operand_index % compiler::surface_value_operands_per_word;
          auto word = instruction.operand_payload;
          if (operand_count > compiler::surface_value_inline_operand_capacity) {
            if (instruction.operand_payload >= image.value_operands.size() ||
                word_index >=
                    image.value_operands.size() - instruction.operand_payload) {
              return false;
            }
            word =
                image.value_operands[instruction.operand_payload + word_index];
          }
          operand = compiler::surface_value_operand_from_word(word, lane);
          return operand.valid();
        };

    struct UniqueParameterAddresses {
      std::set<std::uint32_t> scalars;
      std::set<std::uint32_t> vectors;
      std::set<std::uint32_t> unsigned_integers;
    } unique_parameter_addresses;
    struct ParameterUseInterval {
      std::uint32_t references{};
      std::uint32_t dynamic_references{};
      std::uint32_t first_instruction{};
      std::uint32_t last_instruction{};
    };
    using ParameterAddress = std::pair<std::uint32_t, std::uint32_t>;
    std::map<ParameterAddress, ParameterUseInterval> parameter_intervals;
    struct PreviousValueInstruction {
      std::uint32_t variant{};
      std::uint32_t handler_key{};
      std::uint32_t operation{};
      std::uint32_t result_bank{};
      std::uint32_t result_address{};
    };
    std::optional<PreviousValueInstruction> previous_value_instruction;

    const auto flush_parameter_intervals = [&]() noexcept {
      for (const auto &[address, interval] : parameter_intervals) {
        if (address.first >= histogram.parameter_reuse_bins.size() ||
            interval.references == 0u ||
            interval.dynamic_references > interval.references ||
            interval.first_instruction > interval.last_instruction) {
          histogram.exact = false;
          continue;
        }
        const auto bin =
            std::min<std::size_t>(interval.references,
                                  luisa_surface_parameter_reuse_bin_count) -
            1u;
        auto &destination = histogram.parameter_reuse_bins[address.first][bin];
        checked_add(destination.unique_values, populations);
        checked_weighted_add(destination.references, populations,
                             interval.references);
        checked_weighted_add(destination.dynamic_references, populations,
                             interval.dynamic_references);
        checked_weighted_add(
            destination.instruction_span, populations,
            static_cast<std::uint64_t>(interval.last_instruction) -
                interval.first_instruction + 1u);
      }
      parameter_intervals.clear();
    };

    for (auto offset = std::uint32_t{}; offset < program.instruction_count;
         ++offset) {
      const auto instruction_index = program.instruction_begin + offset;
      const auto &record = image.instructions[instruction_index];
      const auto kind = compiler::surface_svm_bytecode_kind(record);
      if (kind != compiler::SurfaceSvmBytecodeKind::value) {
        previous_value_instruction.reset();
        switch (kind) {
        case compiler::SurfaceSvmBytecodeKind::set_normal:
          flush_parameter_intervals();
          checked_add(histogram.surface_normal_transition_executions,
                      populations);
          break;
        case compiler::SurfaceSvmBytecodeKind::closure_leaf: {
          checked_add(histogram.closure_instruction_visits, populations);
          checked_add(histogram.closure_instruction_kind_visits[0u],
                      populations);
          const auto control = compiler::surface_svm_closure_control(record);
          checked_add(
              closure_leaf_counts
                  [control & compiler::surface_closure_static_variant_mask],
              populations);
          break;
        }
        case compiler::SurfaceSvmBytecodeKind::mix_closure:
          checked_add(histogram.closure_instruction_visits, populations);
          checked_add(histogram.closure_instruction_kind_visits[1u],
                      populations);
          break;
        case compiler::SurfaceSvmBytecodeKind::add_closure_weight:
          checked_add(histogram.closure_instruction_visits, populations);
          checked_add(histogram.closure_instruction_kind_visits[2u],
                      populations);
          break;
        case compiler::SurfaceSvmBytecodeKind::jump_if_one:
        case compiler::SurfaceSvmBytecodeKind::jump_if_zero:
          checked_add(histogram.closure_instruction_visits, populations);
          checked_add(histogram.closure_instruction_kind_visits[3u],
                      populations);
          break;
        case compiler::SurfaceSvmBytecodeKind::end:
          break;
        case compiler::SurfaceSvmBytecodeKind::invalid:
          histogram.exact = false;
          break;
        case compiler::SurfaceSvmBytecodeKind::value:
          std::abort();
        }
        continue;
      }

      checked_add(histogram.value_instruction_executions, populations);
      if (instruction_index >= runtime.svm_instruction_variants.size()) {
        histogram.exact = false;
        continue;
      }
      const auto variant = runtime.svm_instruction_variants[instruction_index];
      if (variant >= runtime.value_variants.size()) {
        histogram.exact = false;
        continue;
      }
      const auto instruction = compiler::surface_svm_value_instruction(record);
      const auto &static_variant = runtime.value_variants[variant];
      const auto operand_count =
          compiler::surface_value_operand_count(instruction);
      const auto immediate = compiler::surface_value_svm_immediate(instruction);
      if (compiler::surface_value_operation(instruction) !=
              static_variant.instruction.operation ||
          operand_count != static_variant.operand_routes.size() ||
          std::find(static_variant.svm_immediates.begin(),
                    static_variant.svm_immediates.end(),
                    immediate) == static_variant.svm_immediates.end()) {
        histogram.exact = false;
        previous_value_instruction.reset();
        continue;
      }

      auto direct_operand_mask = std::uint32_t{};
      auto dynamic_direct_operand_mask = std::uint32_t{};
      for (auto operand_index = std::size_t{}; operand_index < operand_count;
           ++operand_index) {
        auto operand = compiler::SurfaceValueOperandAddress{};
        if (!decode_operand(instruction, operand_index, operand)) {
          histogram.exact = false;
          break;
        }
        const auto directly_depends_on_previous =
            previous_value_instruction.has_value() &&
            operand.expanded().encoded() ==
                previous_value_instruction->result_address;
        if (directly_depends_on_previous) {
          if (operand_index >= 32u) {
            histogram.exact = false;
            continue;
          }
          const auto operand_bit = std::uint32_t{1u} << operand_index;
          direct_operand_mask |= operand_bit;
          if (static_variant.operand_routes[operand_index] ==
              compiler::SurfaceValueOperandRoute::dynamic) {
            dynamic_direct_operand_mask |= operand_bit;
          }
        }

        const auto concrete_parameter = operand.parameter();
        if (concrete_parameter) {
          switch (operand.bank()) {
          case compiler::SurfaceValueBank::scalar:
            unique_parameter_addresses.scalars.emplace(operand.index());
            break;
          case compiler::SurfaceValueBank::vector:
            unique_parameter_addresses.vectors.emplace(operand.index());
            break;
          case compiler::SurfaceValueBank::unsigned_integer:
            unique_parameter_addresses.unsigned_integers.emplace(
                operand.index());
            break;
          }
          const auto key = ParameterAddress{
              static_cast<std::uint32_t>(operand.bank()), operand.index()};
          auto [iter, inserted] = parameter_intervals.try_emplace(
              key, ParameterUseInterval{.first_instruction = offset,
                                        .last_instruction = offset});
          auto &interval = iter->second;
          if (!inserted) {
            interval.last_instruction = offset;
          }
          if (interval.references ==
              std::numeric_limits<std::uint32_t>::max()) {
            histogram.exact = false;
          } else {
            ++interval.references;
          }
          if (static_variant.operand_routes[operand_index] ==
              compiler::SurfaceValueOperandRoute::dynamic) {
            if (interval.dynamic_references ==
                std::numeric_limits<std::uint32_t>::max()) {
              histogram.exact = false;
            } else {
              ++interval.dynamic_references;
            }
          }
        }

        switch (static_variant.operand_routes[operand_index]) {
        case compiler::SurfaceValueOperandRoute::local:
          if (concrete_parameter) {
            histogram.exact = false;
          } else {
            checked_add(histogram.value_operand_executions.direct_local,
                        populations);
          }
          break;
        case compiler::SurfaceValueOperandRoute::parameter:
          if (!concrete_parameter) {
            histogram.exact = false;
          } else {
            checked_add(histogram.value_operand_executions.direct_parameter,
                        populations);
          }
          break;
        case compiler::SurfaceValueOperandRoute::dynamic:
          checked_add(concrete_parameter
                          ? histogram.value_operand_executions.dynamic_parameter
                          : histogram.value_operand_executions.dynamic_local,
                      populations);
          break;
        }
      }

      const auto handler_key = compiler::surface_value_handler_key(instruction);
      const auto operation = static_cast<std::uint32_t>(
          compiler::surface_value_operation(instruction));
      const auto result_bank = static_cast<std::uint32_t>(
          compiler::surface_value_result_bank(instruction));
      checked_add(value_counts[ValueKey{variant, handler_key, operation,
                                        result_bank, immediate}],
                  populations);
      if (previous_value_instruction) {
        checked_add(value_transition_counts[ValueTransitionKey{
                        previous_value_instruction->variant,
                        previous_value_instruction->handler_key,
                        previous_value_instruction->operation,
                        previous_value_instruction->result_bank, variant,
                        handler_key, operation, direct_operand_mask,
                        dynamic_direct_operand_mask, 0u}],
                    populations);
      }
      previous_value_instruction =
          PreviousValueInstruction{.variant = variant,
                                   .handler_key = handler_key,
                                   .operation = operation,
                                   .result_bank = result_bank,
                                   .result_address = instruction.result};
    }

    flush_parameter_intervals();
    const auto accumulate_unique_parameters =
        [&](const std::set<std::uint32_t> &addresses,
            std::uint64_t &destination) noexcept {
          for ([[maybe_unused]] const auto address : addresses) {
            checked_add(destination, populations);
          }
        };
    accumulate_unique_parameters(unique_parameter_addresses.scalars,
                                 histogram.unique_parameter_values.scalar);
    accumulate_unique_parameters(unique_parameter_addresses.vectors,
                                 histogram.unique_parameter_values.vector);
    accumulate_unique_parameters(
        unique_parameter_addresses.unsigned_integers,
        histogram.unique_parameter_values.unsigned_integer);
  }

  histogram.value_handlers.reserve(value_counts.size());
  for (const auto &[key, executions] : value_counts) {
    histogram.value_handlers.emplace_back(
        LuisaSurfaceValueHandlerExecutionCount{.variant_index = key[0u],
                                               .handler_key = key[1u],
                                               .operation = key[2u],
                                               .result_bank = key[3u],
                                               .svm_immediate = key[4u],
                                               .executions = executions});
  }
  histogram.value_handler_transitions.reserve(value_transition_counts.size());
  for (const auto &[key, executions] : value_transition_counts) {
    histogram.value_handler_transitions.emplace_back(
        LuisaSurfaceValueHandlerTransitionExecutionCount{
            .source_variant_index = key[0u],
            .source_handler_key = key[1u],
            .source_operation = key[2u],
            .source_result_bank = key[3u],
            .target_variant_index = key[4u],
            .target_handler_key = key[5u],
            .target_operation = key[6u],
            .direct_operand_mask = key[7u],
            .dynamic_direct_operand_mask = key[8u],
            .direct_dependency = key[7u] != 0u,
            .source_last_used_by_target = false,
            .executions = executions});
  }
  histogram.closure_leaf_variants.reserve(closure_leaf_counts.size());
  for (const auto &[static_variant, visits] : closure_leaf_counts) {
    histogram.closure_leaf_variants.emplace_back(
        LuisaSurfaceClosureLeafVisitCount{
            .static_variant = static_variant,
            .operation = static_variant & compiler::surface_closure_opcode_mask,
            .visits = visits});
  }
  _options.surface_program_execution_histogram->sink->write(histogram);
}

void LuisaRenderSession::prepare_sobol_table(std::uint32_t total_samples) {
  const auto sequence_size =
      tabulated_sobol::sequence_size_for_samples(total_samples);
  if (!_sobol_table || _sobol_sequence_size != sequence_size) {
    const auto generated = tabulated_sobol::generate_table(sequence_size);
    luisa::vector<luisa::float4> table;
    table.reserve(generated.size());
    for (const auto sample : generated) {
      table.emplace_back(
          luisa::make_float4(sample.x, sample.y, sample.z, sample.w));
    }
    _sobol_table = _scene->device.create_buffer<luisa::float4>(table.size());
    _stream << _sobol_table.copy_from(luisa::span{table}) << synchronize();
    _sobol_sequence_size = sequence_size;
  }
  _kernel_parameters.sobol_sequence_size = sequence_size;
}

bool LuisaRenderSession::write_passes(contract::OutputSink &output) {
  const auto count = pixel_count();
  luisa::vector<luisa::float4> combined(count);
  luisa::vector<luisa::float4> normal(count);
  luisa::vector<luisa::float4> albedo(count);
  luisa::vector<luisa::float4> light_passes(count * light_pass_buffer_count);
  luisa::vector<luisa::uint> samples(count);
  _stream << _combined.copy_to(luisa::span{combined})
          << _normal.copy_to(luisa::span{normal})
          << _albedo.copy_to(luisa::span{albedo})
          << _light_passes.copy_to(luisa::span{light_passes})
          << _sample_count.copy_to(luisa::span{samples}) << synchronize();

  if (!std::all_of(samples.begin(), samples.end(),
                   [&](const auto count) noexcept {
                     return count == _rendered_samples;
                   })) {
    return false;
  }

  output.begin(_settings);
  for (const auto &pass : _settings.passes) {
    if (!supported_pass(pass.kind)) {
      continue;
    }
    const auto channels = pass_channels(pass);
    std::vector<float> pixels(count * static_cast<std::size_t>(channels));
    for (std::size_t i = 0u; i < count; ++i) {
      const auto denominator = static_cast<float>(std::max(samples[i], 1u));
      const auto exposure = _settings.integrator.film_exposure;
      const auto light_pass_base = i * light_pass_buffer_count;
      const auto read_light_pass =
          [&](LightPassBuffer kind) noexcept -> const luisa::float4 & {
        return light_passes[light_pass_base + light_pass_index(kind)];
      };
      const auto divided_light_pass = [&](LightPassBuffer kind,
                                          const luisa::float4 &color) noexcept {
        const auto divided =
            safe_divide_even_color(read_light_pass(kind), color) * exposure;
        return luisa::make_float4(divided, 1.0f);
      };
      luisa::float4 value{};
      switch (pass.kind) {
      case PassKind::combined:
        value = luisa::make_float4(
            luisa::make_float3(combined[i]) * (exposure / denominator),
            std::clamp(1.0f - combined[i].w / denominator, 0.0f, 1.0f));
        break;
      case PassKind::normal:
      case PassKind::denoising_normal:
        value = normal[i] / denominator;
        break;
      case PassKind::albedo:
      case PassKind::denoising_albedo:
        value = albedo[i] / denominator;
        break;
      case PassKind::glossy_color:
        value = read_light_pass(LightPassBuffer::glossy_color) / denominator;
        break;
      case PassKind::transmission_color:
        value =
            read_light_pass(LightPassBuffer::transmission_color) / denominator;
        break;
      case PassKind::emission:
        value = read_light_pass(LightPassBuffer::emission) *
                (exposure / denominator);
        break;
      case PassKind::environment:
        value = read_light_pass(LightPassBuffer::environment) *
                (exposure / denominator);
        break;
      case PassKind::diffuse_direct:
        value = divided_light_pass(LightPassBuffer::diffuse_direct, albedo[i]);
        break;
      case PassKind::diffuse_indirect:
        value =
            divided_light_pass(LightPassBuffer::diffuse_indirect, albedo[i]);
        break;
      case PassKind::glossy_direct:
        value =
            divided_light_pass(LightPassBuffer::glossy_direct,
                               read_light_pass(LightPassBuffer::glossy_color));
        break;
      case PassKind::glossy_indirect:
        value =
            divided_light_pass(LightPassBuffer::glossy_indirect,
                               read_light_pass(LightPassBuffer::glossy_color));
        break;
      case PassKind::transmission_direct:
        value = divided_light_pass(
            LightPassBuffer::transmission_direct,
            read_light_pass(LightPassBuffer::transmission_color));
        break;
      case PassKind::transmission_indirect:
        value = divided_light_pass(
            LightPassBuffer::transmission_indirect,
            read_light_pass(LightPassBuffer::transmission_color));
        break;
      case PassKind::volume_direct:
        value = read_light_pass(LightPassBuffer::volume_direct) *
                (exposure / denominator);
        break;
      case PassKind::volume_indirect:
        value = read_light_pass(LightPassBuffer::volume_indirect) *
                (exposure / denominator);
        break;
      case PassKind::sample_count:
        value = luisa::make_float4(static_cast<float>(samples[i]));
        break;
      default:
        break;
      }
      const std::array source{value.x, value.y, value.z, value.w};
      for (std::uint32_t channel = 0u; channel < channels; ++channel) {
        pixels[i * channels + channel] =
            source[std::min<std::uint32_t>(channel, 3u)];
      }
    }
    output.write(PassTile{.pass = pass,
                          .window = _window,
                          .full_extent = _settings.full_extent,
                          .pixels = std::span<const float>{pixels}});
  }
  output.end(_cancelled.load());
  return true;
}

LuisaRenderSession::LuisaRenderSession(std::shared_ptr<LuisaSceneData> scene,
                                       LuisaPathTracerOptions options,
                                       const RenderSettings &settings)
    : _scene{std::move(scene)}, _options{options},
      _stream{_scene->device.create_stream()} {
  initialize(settings);
}

void LuisaRenderSession::reset(const RenderSettings &settings) {
  _cancelled.store(false);
  initialize(settings);
}

bool LuisaRenderSession::render_samples(const SampleRange &samples,
                                        contract::OutputSink &output) {
  if (_cancelled.load() || samples.count == 0u) {
    return false;
  }
  const auto sample_begin = static_cast<std::uint64_t>(samples.first) +
                            static_cast<std::uint64_t>(samples.offset);
  const auto sample_end =
      sample_begin + static_cast<std::uint64_t>(samples.count);
  if (samples.total == 0u ||
      sample_end > static_cast<std::uint64_t>(samples.total) ||
      (_total_aa_samples != 0u && _total_aa_samples != samples.total)) {
    return false;
  }
  _total_aa_samples = samples.total;
  prepare_sobol_table(samples.total);
  auto dispatches = _volume_guiding_filter
                        ? SampleDispatchPartition::make_volume_guided(
                              static_cast<std::uint32_t>(sample_begin),
                              samples.count, _options.max_samples_per_dispatch,
                              _rendered_samples, samples.total)
                        : SampleDispatchPartition::make(
                              static_cast<std::uint32_t>(sample_begin),
                              samples.count, _options.max_samples_per_dispatch);
  if (!dispatches) {
    return false;
  }
  while (const auto batch = dispatches->next()) {
    if (_cancelled.load()) {
      return false;
    }
    auto row_dispatches = PixelRowDispatchPartition::make(
        _window.width, _window.height, batch->count,
        _options.max_pixel_samples_per_dispatch);
    if (!row_dispatches) {
      return false;
    }
    while (const auto rows = row_dispatches->next()) {
      auto parameters = _kernel_parameters;
      parameters.window_y += rows->first_row;
      const auto pixel_offset = static_cast<std::size_t>(rows->first_pixel);
      const auto pixel_count = static_cast<std::size_t>(rows->pixel_count);
      const auto light_offset = pixel_offset * light_pass_buffer_count;
      const auto light_count = pixel_count * light_pass_buffer_count;
      const auto volume_raw_offset =
          pixel_offset * volume_guiding::raw_pixel_stride;
      const auto volume_raw_count =
          pixel_count * volume_guiding::raw_pixel_stride;
      const auto volume_denoised_offset =
          pixel_offset * volume_guiding::denoised_pixel_stride;
      const auto volume_denoised_count =
          pixel_count * volume_guiding::denoised_pixel_stride;
      _render_executor.dispatch(
          _stream,
          PathKernelDispatch{
              .combined = _combined.view(pixel_offset, pixel_count),
              .normal = _normal.view(pixel_offset, pixel_count),
              .albedo = _albedo.view(pixel_offset, pixel_count),
              .light_passes = _light_passes.view(light_offset, light_count),
              .sample_count = _sample_count.view(pixel_offset, pixel_count),
              .volume_guiding_raw =
                  _volume_guiding_raw.view(volume_raw_offset, volume_raw_count),
              .volume_guiding_denoised = _volume_guiding_denoised.view(
                  volume_denoised_offset, volume_denoised_count),
              .path_trace = _path_trace.view(),
              .sample_first = batch->first,
              .samples = batch->count,
              .sobol_table = _sobol_table.view(),
              .filter_table = _pixel_filter_table.view(),
              .parameters = parameters,
              .width = _window.width,
              .height = rows->row_count,
              .pixel_count = rows->pixel_count});
      _stream << synchronize();
    }
    _rendered_samples += batch->count;
    if (batch->filter_volume_guiding) {
      _volume_guiding_filter->dispatch(
          _stream, _volume_guiding_raw, _sample_count,
          _volume_guiding_intermediate, _volume_guiding_denoised,
          std::max(_window.width, 1u), std::max(_window.height, 1u));
      _stream << synchronize();
    }
    if (_options.path_trace && !_path_trace_delivered &&
        static_cast<std::uint64_t>(_options.path_trace->sample) >=
            static_cast<std::uint64_t>(batch->first) &&
        static_cast<std::uint64_t>(_options.path_trace->sample) <
            static_cast<std::uint64_t>(batch->first) +
                static_cast<std::uint64_t>(batch->count)) {
      deliver_path_trace();
    }
  }
  if (_cancelled.load()) {
    return false;
  }
  if (!write_passes(output)) {
    return false;
  }
  deliver_surface_closure_count_histogram();
  deliver_surface_program_execution_histogram();
  return true;
}

void LuisaRenderSession::cancel() noexcept { _cancelled.store(true); }

} // namespace psycles::luisa_backend::detail
