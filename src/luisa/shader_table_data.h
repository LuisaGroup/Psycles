#pragma once

#include <psycles/compiler/surface_program.h>
#include <psycles/contract/shader_graph.h>

#include <cstdint>
#include <string>
#include <vector>

#include <luisa/core/mathematics.h>
#include <luisa/core/stl/vector.h>

namespace psycles::luisa_backend::detail {

// A variable-length material table is staged separately from its fixed-size
// descriptor. The descriptor lives at parameter_block + ParameterId; payload
// values are appended after all fixed-size material blocks so every material
// sharing the same SurfaceProgram keeps an identical ABI.
struct PendingShaderTable {
  std::uint32_t descriptor_index{};
  std::uint32_t element_width{};
  std::vector<float> values;
};

struct ShaderTableStagingResult {
  bool valid{};
  PendingShaderTable table;
  std::string diagnostic;
};

[[nodiscard]] ShaderTableStagingResult
stage_shader_table(const compiler::SurfaceProgram &program,
                   const compiler::ParameterDesc &parameter,
                   const contract::SocketValue &value,
                   std::uint32_t descriptor_index);

// Encodes {absolute scalar offset, element count, element width} as exact
// uint32 bit patterns in the vector-parameter slot, then appends payloads to
// the scalar buffer. Returns false without partially appending a failing
// table.
[[nodiscard]] bool
finalize_shader_tables(const std::vector<PendingShaderTable> &tables,
                       luisa::vector<float> &scalar_parameters,
                       luisa::vector<luisa::float3> &vector_parameters,
                       std::string &diagnostic);

} // namespace psycles::luisa_backend::detail
