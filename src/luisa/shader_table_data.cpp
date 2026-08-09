#include "shader_table_data.h"

#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>

namespace psycles::luisa_backend::detail {
namespace {

struct ShaderTableLayout {
  std::uint32_t source_width{};
  std::uint32_t device_width{};
  std::uint32_t minimum_count{};
  bool discard_coordinate{};

  auto operator<=>(const ShaderTableLayout &) const noexcept = default;
};

[[nodiscard]] std::optional<ShaderTableLayout>
table_layout(const compiler::SurfaceProgram &program,
             compiler::ParameterId parameter, std::string &diagnostic) {
  std::optional<ShaderTableLayout> result;
  for (const auto &instruction : program.value_instructions()) {
    if (instruction.parameter != parameter) {
      continue;
    }
    std::optional<ShaderTableLayout> candidate;
    if (instruction.operation == compiler::ValueOperation::color_ramp) {
      const auto sampled = (instruction.static_u0 & 2u) != 0u;
      candidate = ShaderTableLayout{.source_width = 5u,
                                    .device_width = sampled ? 4u : 5u,
                                    .minimum_count = sampled ? 2u : 0u,
                                    .discard_coordinate = sampled};
    } else if (instruction.operation == compiler::ValueOperation::rgb_curve) {
      const auto sampled = (instruction.static_u0 & 1u) != 0u;
      candidate = ShaderTableLayout{.source_width = 4u,
                                    .device_width = sampled ? 3u : 4u,
                                    .minimum_count = sampled ? 2u : 0u,
                                    .discard_coordinate = sampled};
    } else {
      continue;
    }
    if (result && *result != *candidate) {
      diagnostic = "one runtime table parameter has inconsistent shader "
                   "layouts";
      return std::nullopt;
    }
    result = *candidate;
  }
  if (!result) {
    diagnostic = "runtime string parameter is not owned by a supported shader "
                 "table instruction";
  }
  return result;
}

[[nodiscard]] std::optional<std::vector<float>>
decode_float_table(const std::string &encoded, std::string &diagnostic) {
  std::vector<float> result;
  const auto separator = [](char c) noexcept {
    return c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\n' ||
           c == '\r';
  };
  const char *cursor = encoded.c_str();
  const char *const end = cursor + encoded.size();
  while (cursor < end) {
    while (cursor < end && separator(*cursor)) {
      ++cursor;
    }
    if (cursor == end) {
      break;
    }
    errno = 0;
    char *next = nullptr;
    const auto value = std::strtof(cursor, &next);
    if (next == cursor || errno == ERANGE || !std::isfinite(value)) {
      diagnostic =
          "shader table contains an invalid finite float at byte " +
          std::to_string(static_cast<std::size_t>(cursor - encoded.c_str()));
      return std::nullopt;
    }
    result.emplace_back(value);
    cursor = next;
    if (cursor < end && !separator(*cursor)) {
      diagnostic =
          "shader table contains an invalid separator at byte " +
          std::to_string(static_cast<std::size_t>(cursor - encoded.c_str()));
      return std::nullopt;
    }
  }
  return result;
}

} // namespace

ShaderTableStagingResult
stage_shader_table(const compiler::SurfaceProgram &program,
                   const compiler::ParameterDesc &parameter,
                   const contract::SocketValue &value,
                   std::uint32_t descriptor_index) {
  ShaderTableStagingResult result;
  if (parameter.type != contract::SocketType::string ||
      value.type != contract::SocketType::string || !value.well_typed()) {
    result.diagnostic = "shader table parameter is not a well-typed string";
    return result;
  }
  auto layout = table_layout(program, parameter.id, result.diagnostic);
  if (!layout) {
    return result;
  }
  auto decoded =
      decode_float_table(std::get<std::string>(value.value), result.diagnostic);
  if (!decoded) {
    return result;
  }
  if (decoded->size() % layout->source_width != 0u) {
    result.diagnostic = "shader table float count is not divisible by its " +
                        std::to_string(layout->source_width) +
                        "-component source layout";
    return result;
  }
  const auto count = decoded->size() / layout->source_width;
  if (count < layout->minimum_count) {
    result.diagnostic = "sampled shader table requires at least " +
                        std::to_string(layout->minimum_count) + " entries";
    return result;
  }
  result.table = PendingShaderTable{.descriptor_index = descriptor_index,
                                    .element_width = layout->device_width,
                                    .values = {}};
  result.table.values.reserve(count * layout->device_width);
  for (std::size_t element = 0u; element < count; ++element) {
    const auto source = element * layout->source_width;
    const auto first = source + (layout->discard_coordinate ? 1u : 0u);
    for (std::uint32_t component = 0u; component < layout->device_width;
         ++component) {
      result.table.values.emplace_back((*decoded)[first + component]);
    }
  }
  result.valid = true;
  return result;
}

bool finalize_shader_tables(const std::vector<PendingShaderTable> &tables,
                            luisa::vector<float> &scalar_parameters,
                            luisa::vector<luisa::float3> &vector_parameters,
                            std::string &diagnostic) {
  const auto uint_max =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
  auto final_size = scalar_parameters.size();
  for (const auto &table : tables) {
    if (table.element_width == 0u ||
        table.values.size() % table.element_width != 0u ||
        table.descriptor_index >= vector_parameters.size()) {
      diagnostic = "invalid staged shader table descriptor";
      return false;
    }
    if (final_size > uint_max || table.values.size() > uint_max - final_size ||
        table.values.size() / table.element_width > uint_max) {
      diagnostic = "shader table storage exceeds the uint32 device ABI";
      return false;
    }
    final_size += table.values.size();
  }

  for (const auto &table : tables) {
    const auto offset = static_cast<std::uint32_t>(scalar_parameters.size());
    const auto count =
        static_cast<std::uint32_t>(table.values.size() / table.element_width);
    vector_parameters[table.descriptor_index] = luisa::make_float3(
        std::bit_cast<float>(offset), std::bit_cast<float>(count),
        std::bit_cast<float>(table.element_width));
    scalar_parameters.insert(scalar_parameters.end(), table.values.begin(),
                             table.values.end());
  }
  return true;
}

} // namespace psycles::luisa_backend::detail
