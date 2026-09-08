// Structural oracle: Cycles 5.2.1 kernel/closure/bsdf.h shared-label groups.
// Records production Luisa DSL only; never evaluates a shader on the host.
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>

#include <array>
#include <bit>
#include <iostream>
#include <map>
#include <set>

#include "cycles_svm_bsdf.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

namespace {
using namespace luisa::compute;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace bsdf = svm::detail;
namespace closure = psycles::luisa_backend::cycles_closure;
using Mask = bsdf::ClosureTypeMask;
constexpr Mask bit(unsigned tag) { return Mask{1u} << tag; }
constexpr std::array<unsigned, 3u> ggx{closure::type_microfacet_ggx,
                                       closure::type_microfacet_ggx_refraction,
                                       closure::type_microfacet_ggx_glass};
constexpr std::array<unsigned, 3u> beckmann{
    closure::type_microfacet_beckmann,
    closure::type_microfacet_beckmann_refraction,
    closure::type_microfacet_beckmann_glass};
constexpr auto thin = closure::type_thin_glass_transmission;
constexpr auto all_microfacet_closures =
    bit(ggx[0]) | bit(ggx[1]) | bit(ggx[2]) | bit(beckmann[0]) |
    bit(beckmann[1]) | bit(beckmann[2]) | bit(thin);

bool check(unsigned mode, Mask mask) {
  Kernel1D kernel = [&](BufferFloat output, UInt index, Float3 wo) {
    svm::ClosurePool pool{1u};
    const auto identity = make_float4x4(1.0f);
    svm::ShaderData sd{make_float3(0.0f),
                       make_float3(0.0f, 0.0f, 1.0f),
                       make_float3(0.0f, 0.0f, 1.0f),
                       make_float3(0.0f, 0.0f, 1.0f),
                       0u,
                       0u,
                       0u,
                       0u,
                       0u,
                       0.0f,
                       0.0f,
                       0u,
                       0.0f,
                       1.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       make_float3(1.0f, 0.0f, 0.0f),
                       make_float3(0.0f, 1.0f, 0.0f),
                       identity,
                       identity,
                       0u,
                       &pool};
    psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
    switch (mode) {
      case 0:
        output.write(0u, bsdf::bsdf_sample(kg, sd, index, wo, mask).pdf);
        break;
      case 1:
        output.write(0u, bsdf::bsdf_eval(kg, sd, index, wo, mask).pdf);
        break;
      case 2:
        output.write(0u, bsdf::bsdf_roughness_eta(pool, index, wo, mask).eta);
        break;
      case 3:
        output.write(0u,
                     bsdf::bsdf_label(kg, pool, index, wo, mask).cast<float>());
        break;
      case 4:
        bsdf::bsdf_blur(pool, index, 0.7f, mask);
        break;
    }
  };
  const SwitchStmt* dispatch = nullptr;
  traverse_expressions<false>(
      kernel.function()->body(), [](auto) {},
      [&](const Statement* s) {
        if (dispatch == nullptr && s->tag() == Statement::Tag::SWITCH) {
          dispatch = static_cast<const SwitchStmt*>(s);
        }
      },
      [](auto) {});
  if (dispatch == nullptr) {
    return false;
  }
  std::map<unsigned, const ScopeStmt*> bodies;
  for (auto s : dispatch->body()->statements()) {
    if (s->tag() != Statement::Tag::SWITCH_CASE &&
        s->tag() != Statement::Tag::SWITCH_CASE_GROUP) {
      continue;
    }
    auto c = static_cast<const SwitchCaseStmt*>(s);
    for (auto expression : c->expressions()) {
      auto literal = static_cast<const LiteralExpr*>(expression);
      auto tag = luisa::get<unsigned>(literal->value());
      if ((mask & bit(tag)) == 0u || !bodies.emplace(tag, c->body()).second) {
        return false;
      }
    }
  }
  if (bodies.size() != static_cast<size_t>(std::popcount(mask))) {
    return false;
  }
  std::map<unsigned, const ScopeStmt*> groups;
  for (auto [tag, body] : bodies) {
    auto group = 0u;
    if (mode <= 1u) {
      group = (bit(tag) & (bit(ggx[0]) | bit(ggx[1]) | bit(ggx[2]))) ? 1u
              : tag == thin                                          ? 2u
                                                                     : 3u;
    } else if (mode == 3u) {
      group = tag == thin ? 1u : 2u;
    }
    auto [iter, inserted] = groups.emplace(group, body);
    if (!inserted && iter->second != body) {
      return false;
    }
  }
  std::set<const ScopeStmt*> distinct_bodies;
  for (auto [group, body] : groups) {
    if (!distinct_bodies.emplace(body).second) {
      return false;
    }
  }
  return true;
}
}  // namespace

int main() {
  const std::array<Mask, 7u> masks{0u,
                                   all_microfacet_closures,
                                   bit(ggx[1]),
                                   bit(ggx[2]) | bit(beckmann[1]),
                                   bit(thin),
                                   bit(ggx[0]) | bit(ggx[2]),
                                   bit(beckmann[0]) | bit(beckmann[2])};
  auto failures = 0u;
  for (auto mode = 0u; mode < 5u; mode++) {
    for (auto mask : masks) {
      if (!check(mode, mask)) {
        std::cerr << "Cycles shared BSDF case mismatch: mode=" << mode
                  << " mask=" << mask << '\n';
        failures++;
      }
    }
  }
  std::cout << "BSDF shared-case structure: " << (35u - failures)
            << "/35 masks/domains passed\n";
  return failures != 0u;
}
