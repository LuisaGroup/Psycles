// Cycles fractal_voronoi_x_fx has one F2, one Smooth F1 and one shared F1
// fallback. Inspect production DSL loop families; no host shader evaluator.
#include <psycles/luisa/cycles_voronoi.h>
#include <luisa/luisa-compute.h>

#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

namespace {
using namespace luisa::compute;
namespace voronoi = psycles::luisa_backend::cycles_voronoi;

std::optional<unsigned> literal(const Expression *expression) {
  if (expression->tag() != Expression::Tag::LITERAL) { return {}; }
  return luisa::visit([](auto value) -> std::optional<unsigned> {
    if constexpr (std::is_integral_v<decltype(value)>) { return value; }
    return {};
  }, static_cast<const LiteralExpr *>(expression)->value());
}

bool is_case(const Statement *statement) {
  return statement->tag() == Statement::Tag::SWITCH_CASE ||
         statement->tag() == Statement::Tag::SWITCH_CASE_GROUP;
}

const ScopeStmt *fractal_default(const ScopeStmt *body) {
  const ScopeStmt *result = nullptr;
  unsigned matches = 0u;
  traverse_expressions<false>(body, [](auto) {}, [&](const Statement *s) {
    if (s->tag() != Statement::Tag::SWITCH) { return; }
    const auto sw = static_cast<const SwitchStmt *>(s);
    std::set<unsigned> labels;
    const ScopeStmt *fallback = nullptr;
    for (auto entry : sw->body()->statements()) {
      if (is_case(entry)) {
        for (auto expr : static_cast<const SwitchCaseStmt *>(entry)->expressions()) {
          if (auto value = literal(expr)) { labels.insert(*value); }
        }
      } else if (entry->tag() == Statement::Tag::SWITCH_DEFAULT) {
        fallback = static_cast<const SwitchDefaultStmt *>(entry)->body();
      }
    }
    // The original node's Edge/Radius/default partition, not its metric switch.
    if (labels == std::set<unsigned>{3u, 4u} && fallback != nullptr) {
      result = fallback;
      ++matches;
    }
  }, [](auto) {});
  return matches == 1u ? result : nullptr;
}

bool check_dimension(Function function, unsigned dimension, bool extra) {
  const auto body = fractal_default(function.body());
  if (body == nullptr) { return false; }
  // Resolve single-assignment scalar temporaries only, so this does not
  // mistake a mutable loop induction variable for a constant bound.
  std::map<unsigned, std::vector<const Expression *>> definitions;
  traverse_expressions<false>(function.body(), [](auto) {}, [&](const Statement *s) {
    if (s->tag() == Statement::Tag::ASSIGN) {
      const auto assignment = static_cast<const AssignStmt *>(s);
      if (assignment->lhs()->tag() == Expression::Tag::REF) {
        definitions[static_cast<const RefExpr *>(assignment->lhs())->variable().uid()]
            .push_back(assignment->rhs());
      }
    }
  }, [](auto) {});
  const auto resolve = [&](const Expression *expr) {
    std::set<unsigned> visited;
    while (expr->tag() == Expression::Tag::REF) {
      const auto id = static_cast<const RefExpr *>(expr)->variable().uid();
      const auto it = definitions.find(id);
      if (!visited.insert(id).second || it == definitions.end() || it->second.size() != 1u) { break; }
      expr = it->second.front();
    }
    return expr;
  };
  unsigned unit_radius = 0u, smooth_radius = 0u, other_loops = 0u;
  traverse_expressions<false>(body, [](auto) {}, [&](const Statement *s) {
    if (s->tag() != Statement::Tag::FOR) { return; }
    const auto condition = resolve(static_cast<const ForStmt *>(s)->condition());
    std::optional<unsigned> upper;
    if (condition->tag() == Expression::Tag::BINARY) {
      const auto comparison = static_cast<const BinaryExpr *>(condition);
      if (comparison->op() == BinaryOp::LESS) { upper = literal(resolve(comparison->rhs())); }
    }
    if (upper == 3u) { ++unit_radius; }
    else if (upper == 5u) { ++smooth_radius; }
    else { ++other_loops; }
  }, [](auto) {});
  const auto active = dimension == 1u || extra;
  // Source-derived loop families: F1+F2, Smooth F1, and the octave loop.
  const auto pass = unit_radius == (active ? 2u * dimension : 0u) &&
                    smooth_radius == (active ? dimension : 0u) &&
                    other_loops == static_cast<unsigned>(active);
  std::cout << "dimension=" << dimension << " extra=" << extra
            << " radius1_loops=" << unit_radius << " radius2_loops=" << smooth_radius
            << " octave_loops=" << other_loops << " " << (pass ? "PASS" : "FAIL") << '\n';
  return pass;
}

unsigned check_mask(bool extra) {
  Kernel1D kernel = [&](BufferUInt selectors, BufferFloat4 values, BufferFloat4 output) {
    const auto p = values.read(0u);
    const auto q = values.read(1u);
    const auto r = values.read(2u);
    const auto value = voronoi::evaluate_runtime(extra, selectors.read(0u),
        selectors.read(1u), selectors.read(2u), selectors.read(3u) != 0u,
        p.xyz(), p.w, q.x, q.y, q.z, q.w, r.x, r.y, r.z);
    output.write(0u, value.color_distance);
    output.write(1u, value.position);
    output.write(2u, make_float4(value.radius));
  };
  unsigned passed = 0u;
  std::set<unsigned> dimensions;
  traverse_expressions<false>(kernel.function()->body(), [](auto) {}, [&](const Statement *s) {
    if (!is_case(s)) { return; }
    const auto branch = static_cast<const SwitchCaseStmt *>(s);
    const auto dimension = literal(branch->expression());
    if (!dimension || *dimension < 1u || *dimension > 4u || !dimensions.insert(*dimension).second) { return; }
    std::vector<Function> callees;
    traverse_expressions<true>(branch->body(), [&](const Expression *expr) {
      if (expr->tag() == Expression::Tag::CALL) {
        const auto call = static_cast<const CallExpr *>(expr);
        if (call->is_custom()) { callees.push_back(call->custom()); }
      }
    }, [](auto) {}, [](auto) {});
    if (callees.size() == 1u && check_dimension(callees.front(), *dimension, extra)) { ++passed; }
  }, [](auto) {});
  return dimensions.size() == 4u ? passed : 0u;
}
} // namespace

int main() {
  const auto passed = check_mask(false) + check_mask(true);
  std::cout << "Cycles Voronoi octave structure: " << passed << "/8\n";
  return passed == 8u ? 0 : 1;
}
