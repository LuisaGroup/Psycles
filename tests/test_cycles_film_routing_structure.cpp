// Inspect actual production destination expressions. This is an AST data-
// dependence check, not a CPU evaluator of shader or film arithmetic.
#include "cycles_film_routing_test_support.h"

#include <iostream>
#include <fstream>
#include <set>
#include <vector>

namespace {
using namespace luisa::compute;
namespace d = psycles::test_support::film_routing;
namespace f = psycles::test_support::film_routing_fixture;
bool depends(const Expression *expression, const std::set<uint64_t> &ids) {
  bool found = false;
  traverse_subexpressions(expression,
      [&](const Expression *e) { found |= ids.contains(e->hash()); }, [](auto) {});
  return found;
}
// The native writer has two surface-lobe calls plus one common final call.
// Its maximum observed writes therefore also bounds the unreplicated AST
// sites in this all-pass specialization. Zero-valued native writes count.
std::array<unsigned, f::operations> native_write_slots() {
  std::array<unsigned, f::operations> result{};
  std::ifstream input{PSYCLES_FILM_ROUTING_ORACLE};
  for (unsigned i = 0; i < f::cases; ++i) {
    char tag{};
    unsigned index{}, state{};
    input >> tag >> index;
    for (unsigned j = 0; j < 3; ++j) { input >> state; }
    float value{};
    for (unsigned j = 0; j < 6 + f::film_lanes; ++j) { input >> value; }
    unsigned total = 0, count{};
    for (unsigned j = 0; j < f::film_lanes; ++j) {
      input >> count;
      if (j >= 4) { total += count; }
    }
    if (!input || tag != 'F' || index != i) { return {}; }
    auto &maximum = result[f::inputs[i].operation];
    maximum = std::max(maximum, total);
  }
  return result;
}

bool check(unsigned operation, bool detached, unsigned native_slots) {
  const auto config = psycles::test_support::surface_emission::config(false, false);
  const auto evaluator = d::make_direct_light_task_evaluator(config);
  std::set<uint64_t> depths;
  unsigned film_id{};
  Kernel1D kernel = [&](BufferFloat4 combined, BufferFloat4 film, BufferFloat4 weights,
                        BufferUInt state, BufferFloat unused, UInt flags, UInt depth,
                        Float3 value, Float3 dw, Float3 gw, Float3 bd, Float3 bg, Float3 bs) {
    depths.insert(depth.expression()->hash());
    film_id = static_cast<const RefExpr *>(film.expression())->variable().uid();
    d::record(config, evaluator, operation, d::PathFilmAccumulation::atomic, detached,
               combined, film, weights, state, unused, flags, depth, value, dw, gw, bd, bg, bs);
  };
  const auto body = kernel.function()->body();
  std::vector<const AssignStmt *> assignments;
  traverse_expressions<false>(body, [](auto) {}, [&](const Statement *s) {
    if (s->tag() == Statement::Tag::ASSIGN) {
      assignments.push_back(static_cast<const AssignStmt *>(s));
    }
  }, [](auto) {});
  for (bool changed = true; changed;) {
    changed = false;
    for (auto a : assignments) {
      if (depends(a->rhs(), depths)) { changed |= depths.insert(a->lhs()->hash()).second; }
    }
  }
  unsigned atomics = 0, selected = 0;
  traverse_expressions<true>(body, [&](const Expression *e) {
    if (e->tag() != Expression::Tag::CALL) { return; }
    const auto c = static_cast<const CallExpr *>(e);
    if (c->is_custom() || c->op() != CallOp::ATOMIC_FETCH_ADD) { return; }
    const auto args = c->arguments();
    if (args[0]->tag() != Expression::Tag::REF ||
        static_cast<const RefExpr *>(args[0])->variable().uid() != film_id) { return; }
    ++atomics;
    selected += depends(args[1], depths);
  }, [](auto) {}, [](auto) {});
  std::cout << "operation=" << operation << " detached=" << detached
            << " film atomics=" << atomics << " bounce-selected destinations=" << selected
            << " original writer slots=" << native_slots << '\n';
  return atomics != 0 && selected == atomics && atomics == native_slots;
}
} // namespace
int main() {
  unsigned passed = 0;
  const auto slots = native_write_slots();
  for (unsigned op = 0; op < f::operations; ++op) { passed += check(op, false, slots[op]); }
  passed += check(f::surface_nee, true, slots[f::surface_nee]);
  std::cout << "Native film address selection: " << passed << "/5\n";
  return passed == 5 ? 0 : 1;
}
