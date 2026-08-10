#include <psycles/compiler/surface_program.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

class ValueDependencyMask {

private:
  const SurfaceProgram &_program;
  std::vector<bool> _mask;

public:
  explicit ValueDependencyMask(const SurfaceProgram &program)
      : _program{program}, _mask(program.value_instructions().size(), false) {}

  void include(ValueExpressionId root) noexcept {
    const auto &instructions = _program.value_instructions();
    std::vector<ValueExpressionId> pending;
    pending.emplace_back(root);
    while (!pending.empty()) {
      const auto id = pending.back();
      pending.pop_back();
      if (!id.valid() || id.value >= instructions.size() || _mask[id.value]) {
        continue;
      }
      _mask[id.value] = true;
      for (const auto dependency : instructions[id.value].operands) {
        if (dependency.valid()) {
          pending.emplace_back(dependency);
        }
      }
    }
  }

  [[nodiscard]] std::vector<bool> finish() && noexcept {
    return std::move(_mask);
  }
};

[[nodiscard]] bool has_feature(const SurfaceClosurePlanEntry &entry,
                               PrincipledClosureFeature feature) noexcept {
  return (entry.principled_features &
          principled_closure_feature_bit(feature)) != 0u;
}

void include_principled_layer_normal_inputs(
    ValueDependencyMask &dependencies,
    const ClosureInstruction &closure) noexcept {
  dependencies.include(closure.normal);
  dependencies.include(closure.coat_weight);
  if (closure.coat_normal_linked) {
    dependencies.include(closure.coat_normal);
  }
}

void include_principled_coat_inputs(
    ValueDependencyMask &dependencies,
    const ClosureInstruction &closure) noexcept {
  dependencies.include(closure.coat_weight);
  dependencies.include(closure.coat_roughness);
  dependencies.include(closure.coat_ior);
  dependencies.include(closure.coat_tint);
  if (closure.coat_normal_linked) {
    dependencies.include(closure.coat_normal);
  }
}

[[nodiscard]] bool
include_physical_leaf(ValueDependencyMask &dependencies,
                      const ClosureInstruction &closure,
                      const SurfaceClosurePlanEntry &entry) noexcept {
  switch (closure.operation) {
  case ClosureOperation::null_closure:
  case ClosureOperation::emission:
  case ClosureOperation::add:
  case ClosureOperation::mix:
    return false;
  case ClosureOperation::translucent:
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    return true;
  case ClosureOperation::diffuse:
  case ClosureOperation::glossy:
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    return true;
  case ClosureOperation::glass:
  case ClosureOperation::refraction:
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    dependencies.include(closure.ior);
    return true;
  case ClosureOperation::transparent:
    dependencies.include(closure.color);
    return true;
  case ClosureOperation::subsurface:
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    dependencies.include(closure.subsurface_radius);
    dependencies.include(closure.subsurface_scale);
    dependencies.include(closure.subsurface_ior);
    dependencies.include(closure.subsurface_anisotropy);
    return true;
  case ClosureOperation::principled:
    break;
  }

  auto present = false;
  if (has_feature(entry, PrincipledClosureFeature::alpha)) {
    dependencies.include(closure.alpha);
    present = true;
  }
  if (has_feature(entry, PrincipledClosureFeature::sheen)) {
    dependencies.include(closure.sheen_weight);
    dependencies.include(closure.sheen_roughness);
    dependencies.include(closure.sheen_tint);
    include_principled_layer_normal_inputs(dependencies, closure);
    present = true;
  }
  if (has_feature(entry, PrincipledClosureFeature::coat)) {
    include_principled_coat_inputs(dependencies, closure);
    present = true;
  }
  if (has_feature(entry, PrincipledClosureFeature::metallic)) {
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    dependencies.include(closure.metallic);
    dependencies.include(closure.specular_tint);
    present = true;
  }
  const auto transmission =
      has_feature(entry, PrincipledClosureFeature::thick_transmission) ||
      has_feature(entry, PrincipledClosureFeature::thin_transmission);
  if (transmission) {
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    dependencies.include(closure.transmission_weight);
    dependencies.include(closure.ior);
    dependencies.include(closure.specular_tint);
    dependencies.include(closure.thin_wall);
    present = true;
  }
  if (has_feature(entry, PrincipledClosureFeature::dielectric)) {
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    dependencies.include(closure.ior);
    dependencies.include(closure.specular_ior_level);
    dependencies.include(closure.specular_tint);
    present = true;
  }
  const auto subsurface =
      has_feature(entry, PrincipledClosureFeature::thick_subsurface) ||
      has_feature(entry, PrincipledClosureFeature::thin_subsurface);
  if (subsurface) {
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    dependencies.include(closure.subsurface_weight);
    dependencies.include(closure.subsurface_radius);
    dependencies.include(closure.subsurface_scale);
    dependencies.include(closure.subsurface_ior);
    dependencies.include(closure.subsurface_anisotropy);
    // Non-skin BSSRDF setup uses adjusted_ior(), while a linked Thin Wall
    // keeps both the thin and thick physical families reachable.
    dependencies.include(closure.ior);
    dependencies.include(closure.specular_ior_level);
    dependencies.include(closure.thin_wall);
    present = true;
  }
  if (has_feature(entry, PrincipledClosureFeature::diffuse)) {
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.diffuse_roughness);
    // Cycles reduces the diffuse allocation by the authored subsurface
    // weight even when that weight is below the BSSRDF allocation cutoff.
    dependencies.include(closure.subsurface_weight);
    present = true;
  }
  return present;
}

[[nodiscard]] bool
include_emission_leaf(ValueDependencyMask &dependencies,
                      const ClosureInstruction &closure,
                      const SurfaceClosurePlanEntry &entry) noexcept {
  if (closure.operation == ClosureOperation::emission) {
    dependencies.include(closure.color);
    dependencies.include(closure.strength);
    return true;
  }
  if (closure.operation != ClosureOperation::principled ||
      !has_feature(entry, PrincipledClosureFeature::emission)) {
    return false;
  }

  dependencies.include(closure.emission_color);
  dependencies.include(closure.emission_strength);
  if (has_feature(entry, PrincipledClosureFeature::alpha)) {
    dependencies.include(closure.alpha);
  }
  // Principled emission is below Sheen and Coat in Cycles' ordered layer
  // stack. Their exact attenuation inputs are therefore emission
  // dependencies even though the corresponding scattering result is not.
  if (has_feature(entry, PrincipledClosureFeature::sheen)) {
    dependencies.include(closure.sheen_weight);
    dependencies.include(closure.sheen_roughness);
    dependencies.include(closure.sheen_tint);
    include_principled_layer_normal_inputs(dependencies, closure);
  }
  if (has_feature(entry, PrincipledClosureFeature::coat)) {
    include_principled_coat_inputs(dependencies, closure);
  }
  return true;
}

template <typename LeafFunction>
[[nodiscard]] bool include_closure_subtree(
    const SurfaceProgram &program, const SurfaceClosurePlan &plan,
    ValueDependencyMask &dependencies, ClosureExpressionId id,
    LeafFunction &&include_leaf) noexcept {
  if (!id.valid() || id.value >= program.closure_instructions().size() ||
      !plan.entry(id).reachable) {
    return false;
  }
  const auto &closure = program.closure_instructions()[id.value];
  switch (closure.operation) {
  case ClosureOperation::add: {
    const auto a = include_closure_subtree(program, plan, dependencies,
                                           closure.a, include_leaf);
    const auto b = include_closure_subtree(program, plan, dependencies,
                                           closure.b, include_leaf);
    return a || b;
  }
  case ClosureOperation::mix: {
    const auto a_reachable =
        closure.a.valid() && plan.entry(closure.a).reachable;
    const auto b_reachable =
        closure.b.valid() && plan.entry(closure.b).reachable;
    const auto a = include_closure_subtree(program, plan, dependencies,
                                           closure.a, include_leaf);
    const auto b = include_closure_subtree(program, plan, dependencies,
                                           closure.b, include_leaf);
    // GraphSurface bypasses the factor entirely when the closure plan
    // proves one branch unreachable. If both survive, every contribution
    // below this node carries the exact runtime factor.
    if (a_reachable && b_reachable && (a || b)) {
      dependencies.include(closure.factor);
    }
    return a || b;
  }
  case ClosureOperation::null_closure:
  case ClosureOperation::diffuse:
  case ClosureOperation::translucent:
  case ClosureOperation::principled:
  case ClosureOperation::glossy:
  case ClosureOperation::glass:
  case ClosureOperation::emission:
  case ClosureOperation::transparent:
  case ClosureOperation::subsurface:
  case ClosureOperation::refraction:
    return include_leaf(dependencies, closure, plan.entry(id));
  }
  return false;
}

[[nodiscard]] std::vector<bool> merge_masks(const std::vector<bool> &a,
                                            const std::vector<bool> &b) {
  std::vector<bool> result(a.size(), false);
  for (auto i = std::size_t{0u}; i < result.size(); ++i) {
    result[i] = a[i] || b[i];
  }
  return result;
}

} // namespace

bool SurfaceValueDependencyPlan::compatible(
    const SurfaceProgram &program) const noexcept {
  const auto size = program.value_instructions().size();
  return physical.size() == size && emission.size() == size &&
         preparation.size() == size;
}

SurfaceValueDependencyPlan analyze_surface_value_dependencies(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan) noexcept {
  if (!closure_plan.compatible(program)) {
    const auto size = program.value_instructions().size();
    return {.physical = std::vector<bool>(size, true),
            .emission = std::vector<bool>(size, true),
            .preparation = std::vector<bool>(size, true)};
  }

  ValueDependencyMask physical{program};
  static_cast<void>(include_closure_subtree(
      program, closure_plan, physical, program.root(), include_physical_leaf));
  auto physical_mask = std::move(physical).finish();

  ValueDependencyMask emission{program};
  static_cast<void>(include_closure_subtree(
      program, closure_plan, emission, program.root(), include_emission_leaf));
  auto emission_mask = std::move(emission).finish();

  auto preparation_mask = merge_masks(physical_mask, emission_mask);
  return {.physical = std::move(physical_mask),
          .emission = std::move(emission_mask),
          .preparation = std::move(preparation_mask)};
}

} // namespace psycles::compiler
