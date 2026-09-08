// Cycles 5.2.1 svm/closure.h rejects caustics before decoding these payloads.
// This records production DSL and checks lexical control; no host shader runs.
#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <type_traits>
#include <vector>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

namespace {
using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;

class Globals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  bool _reflective;
  bool _refractive;

 public:
  Globals(bool reflective, bool refractive) noexcept
      : _reflective{reflective}, _refractive{refractive} {}
  Bool caustics_reflective() const noexcept override { return _reflective; }
  Bool caustics_refractive() const noexcept override { return _refractive; }
};

bool refers_to(const Expression* expression, const std::set<unsigned>& ids) {
  auto found = false;
  traverse_subexpressions(
      expression,
      [&](const Expression* child) {
        if (child->tag() == Expression::Tag::REF &&
            ids.contains(
                static_cast<const RefExpr*>(child)->variable().uid())) {
          found = true;
        }
      },
      [](auto) {});
  return found;
}

bool has_label(const SwitchCaseStmt* statement, unsigned label) {
  for (auto expression : statement->expressions()) {
    if (expression->tag() != Expression::Tag::LITERAL) {
      continue;
    }
    const auto value = luisa::visit(
        [](auto x) -> unsigned {
          if constexpr (std::is_integral_v<decltype(x)>) {
            return x;
          }
          return ~0u;
        },
        static_cast<const LiteralExpr*>(expression)->value());
    if (value == label) {
      return true;
    }
  }
  return false;
}

bool check(bool physical_pool, bool reflective, bool refractive) {
  unsigned visibility_id = ~0u;
  const Expression* stack_expression = nullptr;
  Kernel1D kernel = [&](BufferUInt words, BufferUInt4 output, UInt visibility) {
    visibility_id =
        static_cast<const RefExpr*>(visibility.expression())->variable().uid();
    const auto identity = make_float4x4(1.0f);
    svm::ClosurePool pool{4u};
    svm::ShaderData sd{make_float3(0.0f),
                       make_float3(0.0f, 0.0f, 1.0f),
                       make_float3(0.0f, 0.0f, 1.0f),
                       make_float3(0.0f, 0.0f, 1.0f),
                       svm::primitive_triangle,
                       0u,
                       0u,
                       0u,
                       0u,
                       0.2f,
                       0.3f,
                       0u,
                       0.5f,
                       1.0f,
                       0.1f,
                       0.1f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       make_float3(1.0f, 0.0f, 0.0f),
                       make_float3(0.0f, 1.0f, 0.0f),
                       identity,
                       identity,
                       0u,
                       physical_pool ? &pool : nullptr};
    const Globals kg{reflective, refractive};
    svm::detail::Stack stack{32u};
    stack_expression = stack.expression();
    UInt offset = 0u;
    Bool supported = true;
    svm::detail::Cursor cursor{words, offset};
    const svm::PathState state{visibility, 0u};
    svm::detail::node_closure_bsdf(
        kg, cursor, stack, make_float3(1.0f), SHADER_TYPE_SURFACE, 0u,
        svm::kernel_feature_node_bsdf, sd, state,
        svm::detail::EvaluationTransition{&supported});
    output.write(0u, make_uint4(offset, sd.flag, pool.count(),
                                supported.cast<unsigned>()));
  };
  const auto body = kernel.function()->body();
  std::vector<const AssignStmt*> assignments;
  traverse_expressions<false>(
      body, [](auto) {},
      [&](const Statement* statement) {
        if (statement->tag() == Statement::Tag::ASSIGN) {
          assignments.push_back(static_cast<const AssignStmt*>(statement));
        }
      },
      [](auto) {});
  // Follow recorded scalar temporaries, not variable names or source lines.
  // This is data dependence only: assignments in a guarded region do not
  // make all subsequent conditions count as visibility guards.
  std::set<unsigned> visibility_dependent{visibility_id};
  for (auto changed = true; changed;) {
    changed = false;
    for (auto assignment : assignments) {
      if (assignment->lhs()->tag() == Expression::Tag::REF &&
          refers_to(assignment->rhs(), visibility_dependent)) {
        changed |= visibility_dependent
                       .emplace(static_cast<const RefExpr*>(assignment->lhs())
                                    ->variable()
                                    .uid())
                       .second;
      }
    }
  }
  constexpr std::array<unsigned, 4u> families{
      CLOSURE_BSDF_PHYSICAL_CONDUCTOR, CLOSURE_BSDF_MICROFACET_GGX_ID,
      CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
      CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID};
  auto passed = true;
  for (auto label : families) {
    unsigned reads = 0u;
    unsigned unguarded = 0u;
    traverse_expressions<false>(
        body, [](auto) {},
        [&](const Statement* statement) {
          if (statement->tag() != Statement::Tag::SWITCH_CASE &&
              statement->tag() != Statement::Tag::SWITCH_CASE_GROUP) {
            return;
          }
          const auto branch = static_cast<const SwitchCaseStmt*>(statement);
          // These four native families have shared labels; the separate PC-only
          // skip switch deliberately has individual labels and no payload
          // reads.
          if (branch->expressions().size() < 2u || !has_label(branch, label)) {
            return;
          }
          std::set<const Statement*> guarded_scopes;
          unsigned guard_depth = 0u;
          traverse_expressions<true>(
              branch->body(),
              [&](const Expression* expression) {
                if (expression->tag() == Expression::Tag::ACCESS &&
                    static_cast<const AccessExpr*>(expression)->range() ==
                        stack_expression) {
                  reads++;
                  unguarded += guard_depth == 0u;
                }
              },
              [&](const Statement* child) {
                if (child->tag() == Statement::Tag::IF) {
                  const auto condition = static_cast<const IfStmt*>(child);
                  if (refers_to(condition->condition(), visibility_dependent)) {
                    guarded_scopes.emplace(condition->true_branch());
                  }
                }
                guard_depth += guarded_scopes.contains(child);
              },
              [&](const Statement* child) {
                guard_depth -= guarded_scopes.contains(child);
              });
        },
        [](auto) {});
    const auto valid =
        physical_pool ? reads != 0u && unguarded == 0u : reads == 0u;
    if (!valid) {
      passed = false;
      std::cerr << "Closure guard mismatch: type=" << label
                << " pool=" << physical_pool << " reflective=" << reflective
                << " refractive=" << refractive << " reads=" << reads
                << " outside_visibility_guard=" << unguarded << '\n';
    }
  }
  return passed;
}
}  // namespace

int main() {
  auto passed = 0u;
  for (auto physical_pool : {false, true}) {
    for (auto reflective : {false, true}) {
      for (auto refractive : {false, true}) {
        passed += check(physical_pool, reflective, refractive);
      }
    }
  }
  std::cout << "Closure input guards: " << passed
            << "/8 configurations passed (four families each)\n";
  return passed != 8u;
}
