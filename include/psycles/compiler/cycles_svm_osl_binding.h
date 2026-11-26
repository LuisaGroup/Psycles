#pragma once

#include <psycles/compiler/cycles_svm_node_types.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace psycles::compiler::cycles_svm {

enum class OslParamKind : std::uint8_t {
  floating = 0u,
  integer = 1u,
  color = 2u,
  vector = 3u,
  point = 4u,
  normal = 5u,
  closure = 6u,
  string = 7u,
};

struct OslParamBinding {
  std::string name;
  OslParamKind kind{OslParamKind::floating};
  bool is_output{};
  bool is_linked{};
  bool has_derivatives{};
  bool use_oso_init{};
  SVMStackOffset stack_offset{SVM_STACK_INVALID};
  std::array<float, 3> default_float3{};
  std::int32_t default_int{};
  std::string default_string;

  bool operator==(const OslParamBinding &) const noexcept = default;
};

struct OslAttrBinding {
  std::string name;
  std::uint64_t attr_id{};
  std::int32_t attr_i32{};

  bool operator==(const OslAttrBinding &) const noexcept = default;
};

struct OslScriptBinding {
  std::string script_name;
  std::string oso_bytecode;
  bool has_closure_output{};
  bool outputs_have_derivatives{};
  std::vector<OslParamBinding> params;
  std::vector<OslAttrBinding> attributes;

  bool operator==(const OslScriptBinding &) const noexcept = default;
};

} // namespace psycles::compiler::cycles_svm
