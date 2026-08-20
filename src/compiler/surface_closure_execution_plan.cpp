#include <psycles/compiler/surface_execution_plan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

[[nodiscard]] SurfaceClosureProgramImage reject(std::string diagnostic) {
  SurfaceClosureProgramImage result;
  result.diagnostic = std::move(diagnostic);
  return result;
}

[[nodiscard]] bool closure_active(
    const SurfaceValueDependencyPlan &dependencies,
    ClosureExpressionId id) noexcept {
  return id.valid() &&
         id.value < dependencies.physical_closures.size() &&
         (dependencies.physical_closures[id.value] ||
          dependencies.emission_closures[id.value]);
}

[[nodiscard]] SurfaceClosureEndpointMask closure_endpoints(
    const SurfaceValueDependencyPlan &dependencies,
    ClosureExpressionId id) noexcept {
  auto result = SurfaceClosureEndpointMask{};
  if (dependencies.physical_closures[id.value]) {
    result |= surface_closure_endpoint_bit(
        SurfaceClosureEndpoint::physical);
  }
  if (dependencies.emission_closures[id.value]) {
    result |= surface_closure_endpoint_bit(
        SurfaceClosureEndpoint::emission);
  }
  return result;
}

[[nodiscard]] std::vector<ValueExpressionId> closure_operands(
    const ClosureInstruction &closure) {
  switch (closure.operation) {
  case ClosureOperation::null_closure:
  case ClosureOperation::add:
  case ClosureOperation::mix:
    return {};
  case ClosureOperation::diffuse:
    return {closure.color, closure.normal, closure.roughness};
  case ClosureOperation::translucent:
    return {closure.color, closure.normal};
  case ClosureOperation::principled:
    return {closure.color,
            closure.normal,
            closure.roughness,
            closure.diffuse_roughness,
            closure.subsurface_weight,
            closure.subsurface_radius,
            closure.subsurface_scale,
            closure.subsurface_ior,
            closure.subsurface_anisotropy,
            closure.transmission_weight,
            closure.metallic,
            closure.ior,
            closure.specular_ior_level,
            closure.specular_tint,
            closure.alpha,
            closure.thin_wall,
            closure.sheen_weight,
            closure.sheen_roughness,
            closure.sheen_tint,
            closure.coat_weight,
            closure.coat_roughness,
            closure.coat_ior,
            closure.coat_tint,
            closure.coat_normal,
            closure.emission_color,
            closure.emission_strength};
  case ClosureOperation::glossy:
    return {closure.color, closure.normal, closure.roughness};
  case ClosureOperation::glass:
  case ClosureOperation::refraction:
    return {closure.color, closure.normal, closure.roughness, closure.ior};
  case ClosureOperation::emission:
    return {closure.color, closure.strength};
  case ClosureOperation::transparent:
    return {closure.color};
  case ClosureOperation::subsurface:
    return {closure.color,
            closure.normal,
            closure.roughness,
            closure.subsurface_radius,
            closure.subsurface_scale,
            closure.subsurface_ior,
            closure.subsurface_anisotropy};
  }
  return {};
}

[[nodiscard]] bool valid_child(const SurfaceProgram &program,
                               ClosureExpressionId parent,
                               ClosureExpressionId child) noexcept {
  // SurfaceProgram's closure stream is topological. Requiring a strict
  // predecessor proves termination of the iterative DFS and rejects cycles
  // without a recursion-depth or visited-state heuristic.
  return child.valid() &&
         child.value < program.closure_instructions().size() &&
         child.value < parent.value;
}

struct PendingClosure {
  ClosureExpressionId id;
  std::vector<SurfaceClosureMixTerm> mix_path;
};

} // namespace

SurfaceClosureProgramImage lower_surface_closure_program(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan,
    const SurfaceValueDependencyPlan &dependencies,
    std::span<const std::uint32_t> value_addresses) {
  static_assert(
      std::is_trivially_copyable_v<SurfaceClosureBytecodeInstruction>);
  static_assert(std::is_trivially_copyable_v<SurfaceClosureMixTerm>);
  static_assert(sizeof(SurfaceClosureBytecodeInstruction) == 16u);
  static_assert(sizeof(SurfaceClosureMixTerm) == 8u);
  static_assert(static_cast<std::uint32_t>(ClosureOperation::refraction) <
                32u);
  static_assert(static_cast<std::uint32_t>(BssrdfMethod::random_walk_skin) <
                4u);

  if (!closure_plan.compatible(program)) {
    return reject("the closure reachability plan is incompatible");
  }
  if (!dependencies.compatible(program)) {
    return reject("the closure dependency plan is incompatible");
  }
  if (value_addresses.size() != program.value_instructions().size()) {
    return reject("the typed value-address image has the wrong size");
  }

  SurfaceClosureProgramImage result;
  const auto root = program.root();
  if (!root.valid()) {
    result.valid = true;
    return result;
  }
  if (root.value >= program.closure_instructions().size()) {
    return reject("the closure root is outside the instruction stream");
  }

  const auto encoded_address = [&](ValueExpressionId value,
                                   bool required,
                                   std::uint32_t &encoded) -> bool {
    encoded = SurfaceValueAddress::invalid_value;
    if (!value.valid()) {
      return !required;
    }
    if (value.value >= value_addresses.size()) {
      return false;
    }
    if (!dependencies.preparation[value.value]) {
      return !required;
    }
    const auto address = SurfaceValueAddress{value_addresses[value.value]};
    if (!address.valid() ||
        static_cast<std::uint32_t>(address.bank()) >
            static_cast<std::uint32_t>(
                SurfaceValueBank::unsigned_integer)) {
      return false;
    }
    encoded = address.encoded();
    return true;
  };

  std::vector<PendingClosure> pending;
  pending.emplace_back(PendingClosure{.id = root, .mix_path = {}});
  while (!pending.empty()) {
    auto current = std::move(pending.back());
    pending.pop_back();
    if (!closure_active(dependencies, current.id) ||
        !closure_plan.entry(current.id).reachable) {
      continue;
    }

    const auto &closure =
        program.closure_instructions()[current.id.value];
    if (closure.operation == ClosureOperation::add ||
        closure.operation == ClosureOperation::mix) {
      if (!valid_child(program, current.id, closure.a) ||
          !valid_child(program, current.id, closure.b)) {
        return reject(
            "a closure-tree edge is not a strict topological predecessor");
      }
      const auto a_reachable =
          closure_plan.entry(closure.a).reachable;
      const auto b_reachable =
          closure_plan.entry(closure.b).reachable;
      const auto a_active =
          a_reachable && closure_active(dependencies, closure.a);
      const auto b_active =
          b_reachable && closure_active(dependencies, closure.b);

      auto a_path = current.mix_path;
      auto b_path = std::move(current.mix_path);
      if (closure.operation == ClosureOperation::mix &&
          a_reachable && b_reachable && (a_active || b_active)) {
        std::uint32_t factor_address{};
        if (!encoded_address(
                closure.factor, true, factor_address)) {
          return reject(
              "a live closure Mix has no typed factor address");
        }
        a_path.emplace_back(SurfaceClosureMixTerm{
            .address = factor_address,
            .flags = surface_closure_mix_complement});
        b_path.emplace_back(SurfaceClosureMixTerm{
            .address = factor_address});
      }

      // LIFO insertion is reversed so the resulting leaf stream is the same
      // left-to-right depth-first order as GraphSurface::for_each_closure.
      if (b_active) {
        pending.emplace_back(PendingClosure{
            .id = closure.b,
            .mix_path = std::move(b_path)});
      }
      if (a_active) {
        pending.emplace_back(PendingClosure{
            .id = closure.a,
            .mix_path = std::move(a_path)});
      }
      continue;
    }
    if (closure.operation == ClosureOperation::null_closure) {
      continue;
    }

    const auto endpoints = closure_endpoints(dependencies, current.id);
    if (endpoints == 0u) {
      continue;
    }
    const auto operands = closure_operands(closure);
    if (operands.size() !=
        surface_closure_operand_count(closure.operation)) {
      return reject("a closure opcode has an inconsistent operand layout");
    }
    if (result.instructions.size() >=
            std::numeric_limits<std::uint32_t>::max() ||
        operands.size() >
            std::numeric_limits<std::uint32_t>::max() -
                result.operands.size() ||
        current.mix_path.size() >
            std::numeric_limits<std::uint32_t>::max() -
                result.mix_terms.size()) {
      return reject("the closure program exceeds 32-bit device offsets");
    }

    const auto operand_begin =
        static_cast<std::uint32_t>(result.operands.size());
    for (const auto operand : operands) {
      std::uint32_t address{};
      const auto required =
          operand.valid() && operand.value < dependencies.preparation.size() &&
          dependencies.preparation[operand.value];
      if (!encoded_address(operand, required, address)) {
        return reject("a live closure operand has no typed value address");
      }
      result.operands.emplace_back(address);
    }
    const auto mix_term_begin =
        static_cast<std::uint32_t>(result.mix_terms.size());
    result.mix_terms.insert(result.mix_terms.end(),
                            current.mix_path.begin(),
                            current.mix_path.end());
    result.maximum_mix_depth = std::max(
        result.maximum_mix_depth,
        static_cast<std::uint32_t>(current.mix_path.size()));

    const auto features =
        closure.operation == ClosureOperation::principled
            ? closure_plan.entry(current.id).principled_features
            : PrincipledClosureFeatureMask{};
    result.instructions.emplace_back(SurfaceClosureBytecodeInstruction{
        .control = make_surface_closure_control(closure, endpoints),
        .operand_begin = operand_begin,
        .mix_term_begin = mix_term_begin,
        .mix_term_count =
            static_cast<std::uint32_t>(current.mix_path.size())});
    result.principled_features.emplace_back(features);
    result.used_operations |=
        1u << static_cast<std::uint32_t>(closure.operation);
    result.used_principled_features |= features;
  }

  if (result.principled_features.size() != result.instructions.size()) {
    return reject(
        "the Principled feature stream is not parallel to closures");
  }
  for (const auto &term : result.mix_terms) {
    if (term.address == SurfaceValueAddress::invalid_value ||
        (term.flags & ~surface_closure_mix_flags_mask) != 0u) {
      return reject("a flattened closure Mix term is invalid");
    }
  }
  result.valid = true;
  return result;
}

} // namespace psycles::compiler
