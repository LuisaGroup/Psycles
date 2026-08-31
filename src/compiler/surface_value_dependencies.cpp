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
  std::vector<bool> _outputs;

public:
  explicit ValueDependencyMask(const SurfaceProgram &program)
      : _program{program}, _mask(program.value_instructions().size(), false),
        _outputs(program.value_instructions().size(), false) {}

  void include(ValueExpressionId root) noexcept {
    const auto &instructions = _program.value_instructions();
    if (!root.valid() || root.value >= instructions.size()) {
      return;
    }
    _outputs[root.value] = true;
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

  struct Result {
    std::vector<bool> active;
    std::vector<bool> outputs;
  };

  [[nodiscard]] Result finish() && noexcept {
    return {.active = std::move(_mask),
            .outputs = std::move(_outputs)};
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
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    return true;
  case ClosureOperation::sheen_microfiber:
  case ClosureOperation::sheen_ashikhmin:
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    return true;
  case ClosureOperation::hair_reflection:
  case ClosureOperation::hair_transmission:
    dependencies.include(closure.color);
    dependencies.include(closure.hair_offset);
    dependencies.include(closure.roughness);
    dependencies.include(closure.diffuse_roughness);
    if (closure.hair_tangent_linked) {
      dependencies.include(closure.tangent);
    }
    return true;
  case ClosureOperation::glossy:
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    if (entry.microfacet_anisotropy) {
      dependencies.include(closure.microfacet_anisotropy);
      dependencies.include(closure.microfacet_rotation);
      dependencies.include(closure.tangent);
    }
    return true;
  case ClosureOperation::metallic_f82:
  case ClosureOperation::metallic_conductor:
    dependencies.include(closure.metallic_base_ior);
    dependencies.include(closure.metallic_edge_tint_k);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    if (entry.microfacet_anisotropy) {
      dependencies.include(closure.microfacet_anisotropy);
      dependencies.include(closure.microfacet_rotation);
      dependencies.include(closure.tangent);
    }
    if (entry.thin_film) {
      dependencies.include(closure.thin_film_thickness);
      dependencies.include(closure.thin_film_ior);
    }
    return true;
  case ClosureOperation::glass:
    dependencies.include(closure.color);
    dependencies.include(closure.normal);
    dependencies.include(closure.roughness);
    dependencies.include(closure.ior);
    if (entry.thin_film) {
      dependencies.include(closure.thin_film_thickness);
      dependencies.include(closure.thin_film_ior);
    }
    return true;
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
  if (entry.microfacet_anisotropy) {
    dependencies.include(closure.microfacet_anisotropy);
    dependencies.include(closure.microfacet_rotation);
    dependencies.include(closure.tangent);
  }
  if (entry.thin_film) {
    dependencies.include(closure.thin_film_thickness);
    dependencies.include(closure.thin_film_ior);
  }
  const auto thin_subsurface =
      has_feature(entry, PrincipledClosureFeature::thin_subsurface);
  const auto subsurface =
      has_feature(entry, PrincipledClosureFeature::thick_subsurface) ||
      thin_subsurface;
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
    // Cycles' Thin Wall subsurface split uses Diffuse Roughness to select
    // both the reflection and transmission closure families. This read is
    // independent of whether the lower standalone diffuse lobe survives:
    // Subsurface Weight == 1 can eliminate that lobe while the thin
    // subsurface closures still observe the value.
    if (thin_subsurface) {
      dependencies.include(closure.diffuse_roughness);
    }
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
    ValueDependencyMask &dependencies, std::vector<bool> &closure_mask,
    ClosureExpressionId id, LeafFunction &&include_leaf) noexcept {
  if (!id.valid() || id.value >= program.closure_instructions().size() ||
      !plan.entry(id).reachable) {
    return false;
  }
  const auto &closure = program.closure_instructions()[id.value];
  auto included = false;
  switch (closure.operation) {
  case ClosureOperation::add: {
    const auto a = include_closure_subtree(
        program, plan, dependencies, closure_mask, closure.a, include_leaf);
    const auto b = include_closure_subtree(
        program, plan, dependencies, closure_mask, closure.b, include_leaf);
    included = a || b;
    break;
  }
  case ClosureOperation::mix: {
    const auto a_reachable =
        closure.a.valid() && plan.entry(closure.a).reachable;
    const auto b_reachable =
        closure.b.valid() && plan.entry(closure.b).reachable;
    const auto a = include_closure_subtree(
        program, plan, dependencies, closure_mask, closure.a, include_leaf);
    const auto b = include_closure_subtree(
        program, plan, dependencies, closure_mask, closure.b, include_leaf);
    // GraphSurface bypasses the factor entirely when the closure plan
    // proves one branch unreachable. If both survive, every contribution
    // below this node carries the exact runtime factor.
    if (a_reachable && b_reachable && (a || b)) {
      dependencies.include(closure.factor);
    }
    included = a || b;
    break;
  }
  case ClosureOperation::null_closure:
  case ClosureOperation::diffuse:
  case ClosureOperation::translucent:
  case ClosureOperation::principled:
  case ClosureOperation::glossy:
  case ClosureOperation::metallic_f82:
  case ClosureOperation::metallic_conductor:
  case ClosureOperation::sheen_microfiber:
  case ClosureOperation::sheen_ashikhmin:
  case ClosureOperation::hair_reflection:
  case ClosureOperation::hair_transmission:
  case ClosureOperation::glass:
  case ClosureOperation::emission:
  case ClosureOperation::transparent:
  case ClosureOperation::subsurface:
  case ClosureOperation::refraction:
    included = include_leaf(dependencies, closure, plan.entry(id));
    break;
  }
  closure_mask[id.value] = included;
  return included;
}

[[nodiscard]] std::vector<bool> merge_masks(const std::vector<bool> &a,
                                            const std::vector<bool> &b) {
  std::vector<bool> result(a.size(), false);
  for (auto i = std::size_t{0u}; i < result.size(); ++i) {
    result[i] = a[i] || b[i];
  }
  return result;
}

[[nodiscard]] bool
active_values_observe_shading_normal(const SurfaceProgram &program,
                                     const std::vector<bool> &active) noexcept {
  const auto &instructions = program.value_instructions();
  for (auto index = std::size_t{0u}; index < instructions.size(); ++index) {
    if (active[index] &&
        value_instruction_observes_shading_normal(instructions[index])) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool emission_closures_observe_shading_normal(
    const SurfaceProgram &program, const SurfaceClosurePlan &plan,
    const std::vector<bool> &active) noexcept {
  const auto &closures = program.closure_instructions();
  for (auto index = std::size_t{0u}; index < closures.size(); ++index) {
    if (!active[index] ||
        closures[index].operation != ClosureOperation::principled) {
      continue;
    }
    const auto &entry =
        plan.entry(ClosureExpressionId{static_cast<std::uint32_t>(index)});
    // Alpha and direct emission are independent of ShaderData::N. Sheen and
    // Coat attenuation each use the current normal as an unlinked/degenerate
    // fallback, so even linked normals retain a formal dependence.
    if (has_feature(entry, PrincipledClosureFeature::sheen) ||
        has_feature(entry, PrincipledClosureFeature::coat)) {
      return true;
    }
  }
  return false;
}

} // namespace

bool value_instruction_observes_shading_normal(
    const ValueInstruction &instruction) noexcept {
  switch (instruction.operation) {
  case ValueOperation::shading_normal:
  case ValueOperation::reflection:
  case ValueOperation::normal_map:
  case ValueOperation::sampled_normal_map:
    return true;
  case ValueOperation::fresnel:
  case ValueOperation::layer_weight_fresnel:
  case ValueOperation::layer_weight_facing:
    // static_u0 is the exact "Normal socket linked" bit for these nodes.
    return instruction.static_u0 == 0u;
  case ValueOperation::ambient_occlusion:
    return (instruction.static_u0 &
            ambient_occlusion_normal_linked) == 0u;
  case ValueOperation::displacement:
    return (instruction.static_u0 &
            displacement_normal_linked) == 0u;
  case ValueOperation::image_color:
  case ValueOperation::image_alpha:
    // Box projection computes axis weights from the current ShaderData normal.
    return ((instruction.static_u1 >> 12u) & 0x3u) == 1u;
  case ValueOperation::bump:
  case ValueOperation::bump_samples: {
    const auto normal_linked = (instruction.static_u0 & 2u) != 0u;
    const auto object_space = (instruction.static_u0 & 4u) != 0u;
    // An unlinked Bump consumes the current normal as its input. Object-space
    // Bump additionally uses it as the exact zero-vector transform fallback.
    return !normal_linked || object_space;
  }
  case ValueOperation::parameter:
  case ValueOperation::passthrough:
  case ValueOperation::scalar_to_color:
  case ValueOperation::scalar_to_boolean:
  case ValueOperation::color_to_scalar:
  case ValueOperation::vector_to_scalar:
  case ValueOperation::add:
  case ValueOperation::subtract:
  case ValueOperation::multiply:
  case ValueOperation::divide:
  case ValueOperation::minimum:
  case ValueOperation::maximum:
  case ValueOperation::power:
  case ValueOperation::math:
  case ValueOperation::light_falloff:
  case ValueOperation::absolute:
  case ValueOperation::clamp01:
  case ValueOperation::bump_offset_zero:
  case ValueOperation::bump_filter_width:
  case ValueOperation::clamp_range:
  case ValueOperation::map_range_float:
  case ValueOperation::map_range_vector:
  case ValueOperation::vector_math_value:
  case ValueOperation::vector_math_vector:
  case ValueOperation::mix_float:
  case ValueOperation::mix_vector:
  case ValueOperation::mix:
  case ValueOperation::multiply_color:
  case ValueOperation::hue_saturation:
  case ValueOperation::invert:
  case ValueOperation::gamma:
  case ValueOperation::brightness_contrast:
  case ValueOperation::blackbody:
  case ValueOperation::wavelength:
  case ValueOperation::surface_position:
  case ValueOperation::sampled_surface_position:
  case ValueOperation::geometric_normal:
  case ValueOperation::incoming:
  case ValueOperation::tangent:
  case ValueOperation::uv:
  case ValueOperation::sampled_uv:
  case ValueOperation::generated:
  case ValueOperation::sampled_generated:
  case ValueOperation::object_position:
  case ValueOperation::sampled_object_position:
  case ValueOperation::object_position_with_transform:
  case ValueOperation::sampled_object_position_with_transform:
  case ValueOperation::object_location:
  case ValueOperation::object_random:
  case ValueOperation::particle_index:
  case ValueOperation::particle_random:
  case ValueOperation::back_facing:
  case ValueOperation::pointiness:
  case ValueOperation::sampled_pointiness:
  case ValueOperation::random_per_island:
  case ValueOperation::curve_is_strand:
  case ValueOperation::curve_intercept:
  case ValueOperation::curve_length:
  case ValueOperation::curve_thickness:
  case ValueOperation::curve_tangent_normal:
  case ValueOperation::curve_random:
  case ValueOperation::path_is_camera:
  case ValueOperation::path_is_shadow:
  case ValueOperation::path_is_diffuse:
  case ValueOperation::path_is_glossy:
  case ValueOperation::path_is_singular:
  case ValueOperation::path_is_reflection:
  case ValueOperation::path_is_transmission:
  case ValueOperation::path_is_volume_scatter:
  case ValueOperation::path_ray_length:
  case ValueOperation::path_ray_depth:
  case ValueOperation::path_diffuse_depth:
  case ValueOperation::path_glossy_depth:
  case ValueOperation::path_transparent_depth:
  case ValueOperation::path_transmission_depth:
  case ValueOperation::path_portal_depth:
  case ValueOperation::mapping:
  case ValueOperation::environment_color:
  case ValueOperation::environment_alpha:
  case ValueOperation::attribute_color:
  case ValueOperation::sampled_attribute_color:
  case ValueOperation::attribute_factor:
  case ValueOperation::sampled_attribute_factor:
  case ValueOperation::attribute_alpha:
  case ValueOperation::sampled_attribute_alpha:
  case ValueOperation::noise_factor:
  case ValueOperation::noise_color:
  case ValueOperation::white_noise_value:
  case ValueOperation::white_noise_color:
  case ValueOperation::checker_color:
  case ValueOperation::checker_factor:
  case ValueOperation::brick_color:
  case ValueOperation::brick_factor:
  case ValueOperation::magic_color:
  case ValueOperation::magic_factor:
  case ValueOperation::wave_color:
  case ValueOperation::wave_factor:
  case ValueOperation::voronoi_distance:
  case ValueOperation::voronoi_color:
  case ValueOperation::voronoi_position:
  case ValueOperation::voronoi_w:
  case ValueOperation::voronoi_radius:
  case ValueOperation::gradient:
  case ValueOperation::color_ramp:
  case ValueOperation::rgb_curve:
  case ValueOperation::separate_r:
  case ValueOperation::separate_g:
  case ValueOperation::separate_b:
  case ValueOperation::combine_color:
  case ValueOperation::hosek_wilkie_sky:
  case ValueOperation::nishita_sky:
    return false;
  }
  std::abort();
}

bool SurfaceValueDependencyPlan::compatible(
    const SurfaceProgram &program) const noexcept {
  const auto size = program.value_instructions().size();
  const auto closure_size = program.closure_instructions().size();
  return physical.size() == size && emission.size() == size &&
         preparation.size() == size &&
         physical_outputs.size() == size &&
         emission_outputs.size() == size &&
         preparation_outputs.size() == size &&
         physical_closures.size() == closure_size &&
         emission_closures.size() == closure_size;
}

SurfaceValueDependencyPlan analyze_surface_value_dependencies(
    const SurfaceProgram &program,
    const SurfaceClosurePlan &closure_plan) noexcept {
  if (!closure_plan.compatible(program)) {
    const auto size = program.value_instructions().size();
    const auto closure_size = program.closure_instructions().size();
    return {.physical = std::vector<bool>(size, true),
            .emission = std::vector<bool>(size, true),
            .preparation = std::vector<bool>(size, true),
            .physical_outputs = std::vector<bool>(size, true),
            .emission_outputs = std::vector<bool>(size, true),
            .preparation_outputs = std::vector<bool>(size, true),
            .physical_closures = std::vector<bool>(closure_size, true),
            .emission_closures = std::vector<bool>(closure_size, true),
            .emission_values_observe_shading_normal = true,
            .emission_closures_observe_shading_normal = true};
  }

  ValueDependencyMask physical{program};
  std::vector<bool> physical_closures(program.closure_instructions().size(),
                                      false);
  static_cast<void>(include_closure_subtree(program, closure_plan, physical,
                                            physical_closures, program.root(),
                                            include_physical_leaf));
  auto physical_result = std::move(physical).finish();

  ValueDependencyMask emission{program};
  std::vector<bool> emission_closures(program.closure_instructions().size(),
                                      false);
  static_cast<void>(include_closure_subtree(program, closure_plan, emission,
                                            emission_closures, program.root(),
                                            include_emission_leaf));
  auto emission_result = std::move(emission).finish();

  auto preparation_mask = merge_masks(
      physical_result.active, emission_result.active);
  auto preparation_outputs = merge_masks(
      physical_result.outputs, emission_result.outputs);
  const auto emission_values_observe_normal =
      active_values_observe_shading_normal(program, emission_result.active);
  const auto emission_closures_observe_normal =
      emission_closures_observe_shading_normal(program, closure_plan,
                                               emission_closures);
  return {.physical = std::move(physical_result.active),
          .emission = std::move(emission_result.active),
          .preparation = std::move(preparation_mask),
          .physical_outputs = std::move(physical_result.outputs),
          .emission_outputs = std::move(emission_result.outputs),
          .preparation_outputs = std::move(preparation_outputs),
          .physical_closures = std::move(physical_closures),
          .emission_closures = std::move(emission_closures),
          .emission_values_observe_shading_normal =
              emission_values_observe_normal,
          .emission_closures_observe_shading_normal =
              emission_closures_observe_normal};
}

} // namespace psycles::compiler
