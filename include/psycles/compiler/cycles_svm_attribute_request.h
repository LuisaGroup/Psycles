#pragma once

#include <psycles/compiler/cycles_svm_types.h>

#include <compare>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace psycles::compiler::cycles_svm {

// Symbolic counterpart of Cycles 5.2.1 AttributeRequest. Keeping requests
// symbolic is required: an out-of-band geometry dependency must not consume a
// ShaderManager named-attribute id before SVM bytecode compilation would.
struct AttributeRequest {
  AttributeStandard standard{ATTR_STD_NONE};
  std::string name;

  auto operator<=>(const AttributeRequest &) const = default;
};

// Set semantics and add/add_standard transitions match Cycles'
// AttributeRequestSet. Insertion order is retained while the graph is being
// inspected; canonical_requests() provides an order-independent shader image.
class AttributeRequestSet final {
private:
  std::vector<AttributeRequest> _requests;

public:
  void add(std::string_view name);
  void add(AttributeStandard standard);
  void add_standard(std::string_view name);

  [[nodiscard]] std::span<const AttributeRequest> requests() const noexcept {
    return _requests;
  }
  [[nodiscard]] std::vector<AttributeRequest> canonical_requests() const;
};

[[nodiscard]] std::string_view
attribute_standard_name(AttributeStandard standard) noexcept;
[[nodiscard]] AttributeStandard
attribute_standard_from_name(std::string_view name) noexcept;

} // namespace psycles::compiler::cycles_svm
