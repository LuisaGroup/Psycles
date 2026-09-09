// Inspect the production host-recorded emission operation. Zero RGB does not
// substitute for control over an atomic film side effect.
#include "cycles_surface_emission_test_support.h"

#include <iostream>
#include <set>
#include <vector>

namespace {
using namespace luisa::compute;
namespace f = psycles::test_support::surface_emission;

bool depends(const Expression *expression, const std::set<unsigned> &ids) {
  bool found = false;
  traverse_subexpressions(
      expression,
      [&](const Expression *e) {
        if (e->tag() == Expression::Tag::REF) {
          found |=
              ids.contains(static_cast<const RefExpr *>(e)->variable().uid());
        }
      },
      [](auto) {});
  return found;
}

bool check(bool nee, bool trace) {
  const auto config = f::config(nee, trace);
  std::set<unsigned> flags, exits;
  Kernel1D kernel = [&](BufferFloat4 film, BufferFloat4 diagnostic,
                        BufferUInt count, BufferFloat unused, UInt sd_flag,
                        Bool exit, UInt path, UInt depth, Float3 emission,
                        UInt sampling) {
    flags.insert(
        static_cast<const RefExpr *>(sd_flag.expression())->variable().uid());
    exits.insert(
        static_cast<const RefExpr *>(exit.expression())->variable().uid());
    f::record(config, f::PathFilmAccumulation::atomic, film, diagnostic, count,
              unused, sd_flag, exit, path, depth, emission, sampling, 0, 0);
  };
  const auto body = kernel.function()->body();
  std::vector<const AssignStmt *> assignments;
  traverse_expressions<false>(
      body, [](auto) {},
      [&](const Statement *s) {
        if (s->tag() == Statement::Tag::ASSIGN) {
          assignments.push_back(static_cast<const AssignStmt *>(s));
        }
      },
      [](auto) {});
  for (auto *ids : {&flags, &exits}) {
    for (bool changed = true; changed;) {
      changed = false;
      for (auto a : assignments) {
        if (a->lhs()->tag() == Expression::Tag::REF &&
            depends(a->rhs(), *ids)) {
          changed |=
              ids->insert(
                     static_cast<const RefExpr *>(a->lhs())->variable().uid())
                  .second;
        }
      }
    }
  }
  std::set<const Statement *> guards;
  const auto mis_hash =
      config.light_transport.forward_light_weight.function().hash();
  unsigned depth = 0, atomics = 0, mis_calls = 0, unguarded = 0;
  traverse_expressions<true>(
      body,
      [&](const Expression *e) {
        if (e->tag() == Expression::Tag::CALL) {
          const auto c = static_cast<const CallExpr *>(e);
          if (!c->is_custom() && c->op() == CallOp::ATOMIC_FETCH_ADD) {
            ++atomics;
            unguarded += depth == 0;
          }
          if (c->is_custom() && c->custom().hash() == mis_hash) {
            ++mis_calls;
            unguarded += depth == 0;
          }
        }
      },
      [&](const Statement *s) {
        if (s->tag() == Statement::Tag::IF) {
          const auto branch = static_cast<const IfStmt *>(s);
          if (depends(branch->condition(), flags) &&
              depends(branch->condition(), exits)) {
            guards.insert(branch->true_branch());
          }
        }
        depth += guards.contains(s);
      },
      [&](const Statement *s) { depth -= guards.contains(s); });
  std::cout << "nee=" << nee << " trace=" << trace << " atomics=" << atomics
            << " MIS calls=" << mis_calls
            << " outside emission/exit guard=" << unguarded << '\n';
  return atomics != 0 && mis_calls == unsigned(nee) && unguarded == 0;
}
} // namespace

int main() {
  unsigned passed = 0;
  for (bool nee : {false, true}) {
    for (bool trace : {false, true}) {
      passed += check(nee, trace);
    }
  }
  std::cout << "Cycles surface emission control: " << passed << "/4\n";
  return passed == 4 ? 0 : 1;
}
