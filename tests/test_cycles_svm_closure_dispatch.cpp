// Structural oracle: Cycles 5.2.1 kernel/svm/closure.h closure-type switch.
// Records the production DSL; it does not execute a host shader evaluator.
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <type_traits>
#include <vector>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

namespace {
using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
using Groups = std::vector<std::vector<std::uint32_t>>;

Groups original_groups(std::uint32_t features) {
  Groups result{
      {CLOSURE_BSDF_PRINCIPLED_ID},
      {CLOSURE_BSDF_DIFFUSE_ID},
      {CLOSURE_BSDF_TRANSLUCENT_ID},
      {CLOSURE_BSDF_TRANSPARENT_ID},
      {CLOSURE_BSDF_PHYSICAL_CONDUCTOR, CLOSURE_BSDF_F82_CONDUCTOR},
      {CLOSURE_BSDF_RAY_PORTAL_ID},
      {CLOSURE_BSDF_MICROFACET_GGX_ID, CLOSURE_BSDF_MICROFACET_BECKMANN_ID,
       CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID, CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID},
      {CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
       CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID},
      {CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
       CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID,
       CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID},
      {CLOSURE_BSDF_ASHIKHMIN_VELVET_ID},
      {CLOSURE_BSDF_SHEEN_ID},
      {CLOSURE_BSDF_GLOSSY_TOON_ID, CLOSURE_BSDF_DIFFUSE_TOON_ID}};
  if ((features & svm::kernel_feature_hair) != 0u) {
    if ((features & svm::kernel_feature_node_principled_hair) != 0u) {
      result.push_back(
          {CLOSURE_BSDF_HAIR_CHIANG_ID, CLOSURE_BSDF_HAIR_HUANG_ID});
    }
    result.push_back(
        {CLOSURE_BSDF_HAIR_REFLECTION_ID, CLOSURE_BSDF_HAIR_TRANSMISSION_ID});
  }
  if ((features & svm::kernel_feature_subsurface) != 0u) {
    result.push_back({CLOSURE_BSSRDF_BURLEY_ID, CLOSURE_BSSRDF_RANDOM_WALK_ID,
                      CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID,
                      CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID});
  }
  return result;
}

bool matches_partition(const SwitchStmt* statement, const Groups& groups) {
  std::map<std::uint32_t, const ScopeStmt*> labels;
  for (auto entry : statement->body()->statements()) {
    if (entry->tag() != Statement::Tag::SWITCH_CASE &&
        entry->tag() != Statement::Tag::SWITCH_CASE_GROUP) {
      continue;
    }
    auto body = static_cast<const SwitchCaseStmt*>(entry);
    for (auto expression : body->expressions()) {
      const auto literal = static_cast<const LiteralExpr*>(expression);
      const auto value = luisa::visit(
          [](auto x) -> std::uint32_t {
            if constexpr (std::is_integral_v<decltype(x)>) {
              return x;
            }
            return ~0u;
          },
          literal->value());
      if (!labels.emplace(value, body->body()).second) {
        return false;
      }
    }
  }
  std::set<const ScopeStmt*> distinct_bodies;
  for (const auto& group : groups) {
    const ScopeStmt* expected = nullptr;
    for (auto value : group) {
      auto iter = labels.find(value);
      if (iter == labels.end()) {
        return false;
      }
      if (expected != nullptr && expected != iter->second) {
        return false;
      }
      expected = iter->second;
      labels.erase(iter);
    }
    if (!distinct_bodies.emplace(expected).second) {
      return false;
    }
  }
  return labels.empty();
}

bool check(ShaderType domain, std::uint32_t features, std::uint32_t node_mask) {
  Kernel1D kernel = [&](BufferUInt words, BufferUInt4 output) {
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
                       &pool};
    const psycles::test_support::DefaultCyclesSvmKernelGlobals kg;
    svm::detail::Stack stack;
    UInt offset = 0u;
    Bool supported = true;
    svm::detail::Cursor cursor{words, offset};
    const svm::PathState state{svm::path_ray_visibility_camera, 0u};
    svm::detail::node_closure_bsdf(
        kg, cursor, stack, make_float3(1.0f), domain, features, node_mask, sd,
        state, svm::detail::EvaluationTransition{&supported});
    output.write(
        0u, make_uint4(offset, sd.flag, pool.count(), supported.cast<uint>()));
  };
  const auto groups = original_groups(features);
  auto found = 0u;
  traverse_expressions<false>(
      kernel.function()->body(), [](auto) {},
      [&](const Statement* statement) {
        if (statement->tag() == Statement::Tag::SWITCH &&
            matches_partition(static_cast<const SwitchStmt*>(statement),
                              groups)) {
          found++;
        }
      },
      [](auto) {});
  const auto active = domain == SHADER_TYPE_SURFACE &&
                      (node_mask & svm::kernel_feature_node_bsdf) != 0u;
  return found == static_cast<unsigned>(active);
}
}  // namespace

int main() {
  constexpr auto hair = svm::kernel_feature_hair;
  constexpr auto principled_hair = svm::kernel_feature_node_principled_hair;
  constexpr auto subsurface = svm::kernel_feature_subsurface;
  constexpr std::array<std::uint32_t, 6u> features{
      0u,
      hair,
      principled_hair,
      hair | principled_hair,
      subsurface,
      hair | principled_hair | subsurface};
  constexpr std::array<std::uint32_t, 4u> node_masks{
      0u, svm::kernel_feature_node_emission, svm::kernel_feature_node_bsdf,
      svm::kernel_feature_node_emission | svm::kernel_feature_node_bsdf};
  auto failed = 0u;
  for (auto domain :
       {SHADER_TYPE_SURFACE, SHADER_TYPE_VOLUME, SHADER_TYPE_DISPLACEMENT}) {
    for (auto feature : features) {
      for (auto node_mask : node_masks) {
        if (!check(domain, feature, node_mask)) {
          failed++;
          std::cerr << "Closure dispatch mismatch: domain=" << domain
                    << " features=" << feature << " node_mask=" << node_mask
                    << '\n';
        }
      }
    }
  }
  std::cout << "Closure dispatch structure: " << (72u - failed)
            << "/72 passed\n";
  return failed != 0u;
}
