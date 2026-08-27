#include "surface_program_builder.h"

#include <psycles/contract/scene.h>

#include <cmath>
#include <utility>
#include <vector>

namespace psycles::compiler {
namespace {

struct EmissionProof {
    bool may_emit{};
    bool cycles_constant{true};
};

[[nodiscard]] bool is_unconnected_parameter(
    const std::vector<ValueInstruction> &values,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    return expression.valid() &&
           expression.value < values.size() &&
           values[expression.value].operation ==
               ValueOperation::parameter &&
           values[expression.value].parameter.valid() &&
           values[expression.value].source_node == owner;
}

// This transfer function is the closed-form counterpart of Cycles'
// output_estimate_emission relation. Non-emitting closure leaves are the
// additive identity and do not force shader evaluation. Emission inputs and
// Mix weights are provably constant only when they are direct, unconnected
// parameters; a linked value remains deferred even if a later optimizer could
// prove it numerically uniform. That preserves Cycles' conservative scheduling
// without evaluating or baking a single material value on the host.
[[nodiscard]] EmissionEvaluationMode analyze_emission_evaluation(
    const std::vector<ValueInstruction> &values,
    const std::vector<ClosureInstruction> &closures,
    ClosureExpressionId root) noexcept {
    if (!root.valid() || root.value >= closures.size()) {
        return EmissionEvaluationMode::none;
    }
    std::vector<EmissionProof> proofs;
    proofs.reserve(closures.size());
    const auto dependency = [&](ClosureExpressionId id) noexcept {
        return id.valid() && id.value < proofs.size()
                   ? proofs[id.value]
                   : EmissionProof{.may_emit = true,
                         .cycles_constant = false};
    };
    for (const auto &closure : closures) {
        EmissionProof proof;
        switch (closure.operation) {
            case ClosureOperation::null_closure:
            case ClosureOperation::diffuse:
            case ClosureOperation::translucent:
            case ClosureOperation::glossy:
            case ClosureOperation::metallic_f82:
            case ClosureOperation::metallic_conductor:
            case ClosureOperation::sheen_microfiber:
            case ClosureOperation::sheen_ashikhmin:
            case ClosureOperation::hair_reflection:
            case ClosureOperation::hair_transmission:
            case ClosureOperation::glass:
            case ClosureOperation::refraction:
            case ClosureOperation::transparent:
            case ClosureOperation::subsurface:
                break;
            case ClosureOperation::principled:
                // Cycles never exposes Principled through the constant
                // emission path: alpha, sheen and coat can all affect its
                // emitted radiance. Material-specific zero emission is
                // refined separately by estimate_surface_emission().
                proof.may_emit = true;
                proof.cycles_constant = false;
                break;
            case ClosureOperation::emission:
                proof.may_emit = true;
                proof.cycles_constant =
                    is_unconnected_parameter(
                        values, closure.color, closure.source_node) &&
                    is_unconnected_parameter(
                        values, closure.strength, closure.source_node);
                break;
            case ClosureOperation::add: {
                const auto a = dependency(closure.a);
                const auto b = dependency(closure.b);
                proof.may_emit = a.may_emit || b.may_emit;
                proof.cycles_constant =
                    a.cycles_constant && b.cycles_constant;
                break;
            }
            case ClosureOperation::mix: {
                const auto a = dependency(closure.a);
                const auto b = dependency(closure.b);
                proof.may_emit = a.may_emit || b.may_emit;
                proof.cycles_constant =
                    a.cycles_constant && b.cycles_constant &&
                    is_unconnected_parameter(
                        values, closure.factor, closure.source_node);
                break;
            }
        }
        proofs.emplace_back(proof);
    }
    const auto result = proofs[root.value];
    if (!result.may_emit) {
        return EmissionEvaluationMode::none;
    }
    return result.cycles_constant
               ? EmissionEvaluationMode::constant
               : EmissionEvaluationMode::deferred;
}

}// namespace

SurfaceProgram::SurfaceProgram(
    std::uint64_t structure_signature,
    std::vector<ParameterDesc> parameters,
    std::vector<ValueInstruction> value_instructions,
    std::vector<ClosureInstruction> closure_instructions,
    ClosureExpressionId root,
    std::vector<VolumeInstruction> volume_instructions,
    VolumeExpressionId volume_root,
    ValueExpressionId surface_normal_root,
    ValueExpressionId displacement_root) noexcept
    : _structure_signature{structure_signature},
      _parameters{std::move(parameters)},
      _value_instructions{std::move(value_instructions)},
      _closure_instructions{std::move(closure_instructions)},
      _volume_instructions{std::move(volume_instructions)},
      _root{root},
      _volume_root{volume_root},
      _surface_normal_root{surface_normal_root},
      _displacement_root{displacement_root},
      _emission_evaluation{analyze_emission_evaluation(
          _value_instructions,
          _closure_instructions,
          _root)} {
  // SurfaceProgram is also constructible by tooling/tests without going
  // through SurfaceProgramBuilder. Keep exact operand arity an invariant at
  // that public boundary as well.
  for (const auto &instruction : _value_instructions) {
    if (instruction.operands.size() !=
        value_operation_operand_count(instruction.operation)) {
      std::abort();
    }
  }
}

SurfaceParameterBlock::SurfaceParameterBlock(
    const SurfaceProgram &program) {
    _values.reserve(program.parameters().size());
    for (const auto &parameter : program.parameters()) {
        _values.emplace_back(parameter.default_value);
    }
}

const contract::SocketValue *SurfaceParameterBlock::find(
    ParameterId id) const noexcept {
    if (!id.valid() ||
        static_cast<std::size_t>(id.value) >= _values.size()) {
        return nullptr;
    }
    return &_values[id.value];
}

bool SurfaceParameterBlock::set(
    const SurfaceProgram &program,
    ParameterId id,
    contract::SocketValue value) {
    if (!id.valid() ||
        static_cast<std::size_t>(id.value) >= _values.size() ||
        static_cast<std::size_t>(id.value) >=
            program.parameters().size() ||
        value.type != program.parameters()[id.value].type ||
        !value.well_typed()) {
        return false;
    }
    _values[id.value] = std::move(value);
    return true;
}

namespace {

[[nodiscard]] const contract::SocketValue *direct_parameter_value(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    if (!expression.valid() ||
        expression.value >= program.value_instructions().size()) {
        return nullptr;
    }
    const auto &instruction =
        program.value_instructions()[expression.value];
    if (instruction.operation != ValueOperation::parameter ||
        instruction.source_node != owner ||
        !instruction.parameter.valid()) {
        return nullptr;
    }
    return parameters.find(instruction.parameter);
}

[[nodiscard]] const contract::SocketValue *direct_parameter_value(
    const SurfaceProgram &program,
    const SurfaceParameterBlock *parameters,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    return parameters == nullptr
               ? nullptr
               : direct_parameter_value(
                     program, *parameters, expression, owner);
}

[[nodiscard]] const float *direct_float(
    const SurfaceProgram &program,
    const SurfaceParameterBlock *parameters,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    const auto *value = direct_parameter_value(
        program, parameters, expression, owner);
    return value == nullptr
               ? nullptr
               : std::get_if<float>(&value->value);
}

[[nodiscard]] const bool *direct_bool(
    const SurfaceProgram &program,
    const SurfaceParameterBlock *parameters,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    const auto *value = direct_parameter_value(
        program, parameters, expression, owner);
    return value == nullptr
               ? nullptr
               : std::get_if<bool>(&value->value);
}

[[nodiscard]] const Vec3f *direct_color(
    const SurfaceProgram &program,
    const SurfaceParameterBlock *parameters,
    ValueExpressionId expression,
    contract::NodeId owner) noexcept {
    const auto *value = direct_parameter_value(
        program, parameters, expression, owner);
    return value == nullptr
               ? nullptr
               : std::get_if<Vec3f>(&value->value);
}

[[nodiscard]] bool may_have_positive_component(
    const Vec3f *value) noexcept {
    return value == nullptr ||
           !std::isfinite(value->x) ||
           !std::isfinite(value->y) ||
           !std::isfinite(value->z) ||
           value->x > 0.0f ||
           value->y > 0.0f ||
           value->z > 0.0f;
}

[[nodiscard]] bool unknown_float(const float *value) noexcept {
    return value == nullptr || !std::isfinite(*value);
}

[[nodiscard]] bool thin_film_possible(
    const SurfaceProgram &program,
    const SurfaceParameterBlock *parameters,
    const ClosureInstruction &closure) noexcept {
    constexpr auto thickness_cutoff = 0.1f;
    const auto *thickness = direct_float(
        program,
        parameters,
        closure.thin_film_thickness,
        closure.source_node);
    return unknown_float(thickness) || *thickness > thickness_cutoff;
}

[[nodiscard]] PrincipledClosureFeatureMask principled_feature_mask(
    const SurfaceProgram &program,
    const SurfaceParameterBlock *parameters,
    const ClosureInstruction &closure,
    bool thin_film) noexcept {
    constexpr auto cutoff = 1.0e-5f;
    const auto feature = [](PrincipledClosureFeature value) noexcept {
        return principled_closure_feature_bit(value);
    };
    auto result = PrincipledClosureFeatureMask{};

    const auto *alpha = direct_float(
        program, parameters, closure.alpha, closure.source_node);
    const auto opaque =
        !unknown_float(alpha) && *alpha >= 1.0f;
    if (!opaque) {
        result |= feature(PrincipledClosureFeature::alpha);
    }
    // Alpha is the first Principled layer. A direct non-positive value proves
    // that every lower physical lobe has zero weight, independently of the
    // closure-tree mix weight and caustics policy.
    if (!unknown_float(alpha) && *alpha <= 0.0f) {
        return result;
    }

    // Principled emission is multiplied by the closure's lower layer
    // weight. Alpha above proved the only host-visible zero lower weight;
    // sheen and coat remain device expressions and may only attenuate it.
    // A direct zero in either multiplicand proves the emission unreachable.
    // Linked and non-finite inputs deliberately remain conservative.
    const auto *emission_color = direct_color(
        program,
        parameters,
        closure.emission_color,
        closure.source_node);
    const auto *emission_strength = direct_float(
        program,
        parameters,
        closure.emission_strength,
        closure.source_node);
    const auto color_proven_zero =
        emission_color != nullptr &&
        std::isfinite(emission_color->x) &&
        std::isfinite(emission_color->y) &&
        std::isfinite(emission_color->z) &&
        emission_color->x == 0.0f &&
        emission_color->y == 0.0f &&
        emission_color->z == 0.0f;
    const auto strength_proven_zero =
        emission_strength != nullptr &&
        std::isfinite(*emission_strength) &&
        *emission_strength == 0.0f;
    if (!color_proven_zero && !strength_proven_zero) {
        result |= feature(PrincipledClosureFeature::emission);
    }

    const auto *sheen_weight = direct_float(
        program, parameters, closure.sheen_weight, closure.source_node);
    const auto *sheen_tint = direct_color(
        program, parameters, closure.sheen_tint, closure.source_node);
    if ((unknown_float(sheen_weight) || *sheen_weight > cutoff) &&
        may_have_positive_component(sheen_tint)) {
        result |= feature(PrincipledClosureFeature::sheen);
    }

    const auto *coat_weight = direct_float(
        program, parameters, closure.coat_weight, closure.source_node);
    if (unknown_float(coat_weight) || *coat_weight > cutoff) {
        result |= feature(PrincipledClosureFeature::coat);
    }

    const auto *metallic = direct_float(
        program, parameters, closure.metallic, closure.source_node);
    const auto metallic_possible =
        unknown_float(metallic) || *metallic > cutoff;
    if (metallic_possible) {
        result |= feature(PrincipledClosureFeature::metallic);
    }
    const auto metallic_saturates_lower =
        !unknown_float(metallic) && *metallic >= 1.0f;

    const auto *transmission = direct_float(
        program,
        parameters,
        closure.transmission_weight,
        closure.source_node);
    const auto transmission_possible =
        !metallic_saturates_lower &&
        (unknown_float(transmission) || *transmission > cutoff);
    const auto *thin_wall = direct_bool(
        program, parameters, closure.thin_wall, closure.source_node);
    const auto may_be_thick = thin_wall == nullptr || !*thin_wall;
    const auto may_be_thin = thin_wall == nullptr || *thin_wall;
    if (transmission_possible && may_be_thick) {
        result |= feature(
            PrincipledClosureFeature::thick_transmission);
    }
    if (transmission_possible && may_be_thin) {
        result |= feature(
            PrincipledClosureFeature::thin_transmission);
    }
    const auto transmission_saturates_lower =
        transmission_possible &&
        !unknown_float(transmission) && *transmission >= 1.0f;
    const auto lower_possible =
        !metallic_saturates_lower &&
        !transmission_saturates_lower;
    if (!lower_possible) {
        return result;
    }

    const auto *ior = direct_float(
        program, parameters, closure.ior, closure.source_node);
    const auto *specular_level = direct_float(
        program,
        parameters,
        closure.specular_ior_level,
        closure.source_node);
    // adjusted_ior() is exactly one when a direct Specular IOR Level is
    // non-positive, or when the unadjusted default level observes a direct
    // unit IOR. Other combinations remain conservative rather than
    // duplicating floating-point Fresnel algebra on the host.
    const auto dielectric_proven_unit =
        (!unknown_float(specular_level) &&
         *specular_level <= 0.0f) ||
        (!unknown_float(specular_level) &&
         *specular_level == 0.5f &&
         !unknown_float(ior) && *ior == 1.0f);
    // An active film reflects even when the substrate has unit adjusted IOR.
    // Cycles therefore retains this lobe for eta != 1 OR film thickness above
    // the cutoff; applying only the eta proof would erase real interference.
    if (!dielectric_proven_unit || thin_film) {
        result |= feature(PrincipledClosureFeature::dielectric);
    }

    const auto *subsurface_weight = direct_float(
        program,
        parameters,
        closure.subsurface_weight,
        closure.source_node);
    const auto *base_color = direct_color(
        program, parameters, closure.color, closure.source_node);
    // Scale and radius do not prove the family unreachable: thick BSSRDF
    // setup produces Cycles' diffuse fallback for inactive radius channels,
    // while Thin Wall scattering does not consume scale/radius at all.
    const auto subsurface_possible =
        (unknown_float(subsurface_weight) ||
         *subsurface_weight > cutoff) &&
        may_have_positive_component(base_color);
    if (subsurface_possible && may_be_thick) {
        result |= feature(
            PrincipledClosureFeature::thick_subsurface);
    }
    if (subsurface_possible && may_be_thin) {
        result |= feature(
            PrincipledClosureFeature::thin_subsurface);
    }

    const auto subsurface_saturates_diffuse =
        !unknown_float(subsurface_weight) &&
        *subsurface_weight >= 1.0f;
    if (!subsurface_saturates_diffuse &&
        may_have_positive_component(base_color)) {
        result |= feature(PrincipledClosureFeature::diffuse);
    }
    return result;
}

[[nodiscard]] SurfaceClosurePlan build_surface_closure_plan(
    const SurfaceProgram &program,
    const SurfaceParameterBlock *parameters) noexcept {
    std::vector<SurfaceClosurePlanEntry> entries(
        program.closure_instructions().size());
    if (!program.root().valid() ||
        program.root().value >= entries.size()) {
        return SurfaceClosurePlan{std::move(entries)};
    }
    const auto visit = [&](auto &&self,
                           ClosureExpressionId id) noexcept -> void {
        if (!id.valid() || id.value >= entries.size()) {
            return;
        }
        auto &entry = entries[id.value];
        if (entry.reachable) {
            return;
        }
        entry.reachable = true;
        const auto &closure =
            program.closure_instructions()[id.value];
        switch (closure.operation) {
            case ClosureOperation::add:
                self(self, closure.a);
                self(self, closure.b);
                return;
            case ClosureOperation::mix: {
                const auto *factor = direct_float(
                    program,
                    parameters,
                    closure.factor,
                    closure.source_node);
                if (unknown_float(factor) || *factor < 1.0f) {
                    self(self, closure.a);
                }
                if (unknown_float(factor) || *factor > 0.0f) {
                    self(self, closure.b);
                }
                return;
            }
            case ClosureOperation::principled:
                entry.thin_film = thin_film_possible(
                    program, parameters, closure);
                entry.principled_features = principled_feature_mask(
                    program, parameters, closure, entry.thin_film);
                if ((entry.principled_features &
                     (principled_closure_feature_bit(
                          PrincipledClosureFeature::metallic) |
                      principled_closure_feature_bit(
                          PrincipledClosureFeature::dielectric))) != 0u) {
                    const auto *anisotropic = direct_float(
                        program,
                        parameters,
                        closure.microfacet_anisotropy,
                        closure.source_node);
                    // PrincipledBsdfNode::has_nonzero_weight() retains a
                    // linked/non-finite value and direct values at Cycles'
                    // closure cutoff. Runtime setup saturates the value and
                    // therefore a direct negative finite value is inert.
                    entry.microfacet_anisotropy =
                        unknown_float(anisotropic) ||
                        *anisotropic >= 1.0e-5f;
                }
                return;
            case ClosureOperation::null_closure:
            case ClosureOperation::diffuse:
            case ClosureOperation::translucent:
            case ClosureOperation::sheen_microfiber:
            case ClosureOperation::sheen_ashikhmin:
            case ClosureOperation::hair_reflection:
            case ClosureOperation::hair_transmission:
                return;
            case ClosureOperation::glossy: {
                const auto *anisotropy = direct_float(
                    program,
                    parameters,
                    closure.microfacet_anisotropy,
                    closure.source_node);
                // Keep this in sync with GlossyBsdfNode::is_isotropic() and
                // svm_node_closure_bsdf: linked/non-finite values remain
                // dynamic; direct values in the closed threshold interval
                // are observationally isotropic.
                entry.microfacet_anisotropy =
                    unknown_float(anisotropy) ||
                    std::abs(*anisotropy) > 1.0e-4f;
                return;
            }
            case ClosureOperation::metallic_f82:
            case ClosureOperation::metallic_conductor: {
                entry.thin_film = thin_film_possible(
                    program, parameters, closure);
                const auto *anisotropy = direct_float(
                    program,
                    parameters,
                    closure.microfacet_anisotropy,
                    closure.source_node);
                // MetallicBsdfNode::is_isotropic() is a topology
                // specialization at |a| <= 1e-4. Runtime SVM setup then
                // saturates the surviving authored value to [0, 1].
                entry.microfacet_anisotropy =
                    unknown_float(anisotropy) ||
                    std::abs(*anisotropy) > 1.0e-4f;
                return;
            }
            case ClosureOperation::glass:
                entry.thin_film = thin_film_possible(
                    program, parameters, closure);
                return;
            case ClosureOperation::emission:
            case ClosureOperation::transparent:
            case ClosureOperation::subsurface:
            case ClosureOperation::refraction:
                return;
        }
    };
    visit(visit, program.root());
    return SurfaceClosurePlan{std::move(entries)};
}

[[nodiscard]] bool principled_has_surface_bssrdf(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters,
    const ClosureInstruction &closure) noexcept {
    constexpr auto closure_weight_cutoff = 1.0e-5f;
    const auto *thin_wall_value = direct_parameter_value(
        program,
        parameters,
        closure.thin_wall,
        closure.source_node);
    const auto *thin_wall = thin_wall_value != nullptr
                                ? std::get_if<bool>(
                                      &thin_wall_value->value)
                                : nullptr;
    // Cycles' PrincipledBsdfNode::is_thin_wall() is true only for an
    // unlinked direct true input. A linked Thin Wall remains conservative.
    if (thin_wall != nullptr && *thin_wall) {
        return false;
    }

    const auto *weight_value = direct_parameter_value(
        program,
        parameters,
        closure.subsurface_weight,
        closure.source_node);
    const auto *weight = weight_value != nullptr
                             ? std::get_if<float>(&weight_value->value)
                             : nullptr;
    const auto *scale_value = direct_parameter_value(
        program,
        parameters,
        closure.subsurface_scale,
        closure.source_node);
    const auto *scale = scale_value != nullptr
                            ? std::get_if<float>(&scale_value->value)
                            : nullptr;
    // A linked value is conservatively possible, exactly like Cycles'
    // ShaderNode input-link test. Direct literals use the same cutoff and
    // exact-zero scale relation as PrincipledBsdfNode.
    return (weight == nullptr || *weight > closure_weight_cutoff) &&
           (scale == nullptr || *scale != 0.0f);
}

[[nodiscard]] Vec3f add(Vec3f a, Vec3f b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3f multiply(Vec3f value, float scale) noexcept {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale};
}

}// namespace

bool SurfaceClosurePlan::compatible(
    const SurfaceProgram &program) const noexcept {
    return _entries.size() ==
           program.closure_instructions().size();
}

const SurfaceClosurePlanEntry &SurfaceClosurePlan::entry(
    ClosureExpressionId id) const noexcept {
    if (!id.valid() || id.value >= _entries.size()) {
        std::abort();
    }
    return _entries[id.value];
}

void SurfaceClosurePlan::merge(
    const SurfaceClosurePlan &other) noexcept {
    if (_entries.empty()) {
        _entries = other._entries;
        return;
    }
    if (_entries.size() != other._entries.size()) {
        std::abort();
    }
    for (std::size_t index = 0u;
         index < _entries.size();
         ++index) {
        _entries[index].reachable |=
            other._entries[index].reachable;
        _entries[index].principled_features |=
            other._entries[index].principled_features;
        _entries[index].microfacet_anisotropy |=
            other._entries[index].microfacet_anisotropy;
        _entries[index].thin_film |=
            other._entries[index].thin_film;
    }
}

SurfaceClosurePlan conservative_surface_closure_plan(
    const SurfaceProgram &program) noexcept {
    return build_surface_closure_plan(program, nullptr);
}

SurfaceClosurePlan analyze_surface_closure_plan(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters) noexcept {
    return build_surface_closure_plan(program, &parameters);
}

Vec3f estimate_surface_emission(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters) {
    if (!program.root().valid()) {
        return {};
    }
    std::vector<Vec3f> estimates;
    estimates.reserve(program.closure_instructions().size());
    const auto dependency = [&](ClosureExpressionId id) noexcept {
        // A malformed dependency must not silently remove an emitter from
        // the scene. Valid compiled programs always take the first branch.
        return id.valid() && id.value < estimates.size()
                   ? estimates[id.value]
                   : Vec3f{1.0f, 1.0f, 1.0f};
    };
    for (const auto &closure : program.closure_instructions()) {
        Vec3f estimate{};
        switch (closure.operation) {
            case ClosureOperation::principled:
            case ClosureOperation::emission: {
                const auto color_expression =
                    closure.operation == ClosureOperation::principled
                        ? closure.emission_color
                        : closure.color;
                const auto strength_expression =
                    closure.operation == ClosureOperation::principled
                        ? closure.emission_strength
                        : closure.strength;
                auto color = Vec3f{1.0f, 1.0f, 1.0f};
                if (const auto *value = direct_parameter_value(
                        program,
                        parameters,
                        color_expression,
                        closure.source_node)) {
                    if (const auto *literal =
                            std::get_if<Vec3f>(&value->value)) {
                        color = *literal;
                    }
                }
                auto strength = 1.0f;
                if (const auto *value = direct_parameter_value(
                        program,
                        parameters,
                        strength_expression,
                        closure.source_node)) {
                    if (const auto *literal =
                            std::get_if<float>(&value->value)) {
                        strength = *literal;
                    }
                }
                estimate = multiply(color, strength);
                break;
            }
            case ClosureOperation::add:
                estimate = add(
                    dependency(closure.a),
                    dependency(closure.b));
                break;
            case ClosureOperation::mix: {
                const auto a = dependency(closure.a);
                const auto b = dependency(closure.b);
                const auto *value = direct_parameter_value(
                    program,
                    parameters,
                    closure.factor,
                    closure.source_node);
                const auto *factor =
                    value != nullptr
                        ? std::get_if<float>(&value->value)
                        : nullptr;
                // Cycles intentionally does not try to estimate a linked
                // factor: both branches remain possible and are summed.
                estimate = factor != nullptr
                               ? add(
                                     multiply(a, 1.0f - *factor),
                                     multiply(b, *factor))
                               : add(a, b);
                break;
            }
            case ClosureOperation::null_closure:
            case ClosureOperation::diffuse:
            case ClosureOperation::translucent:
            case ClosureOperation::glossy:
            case ClosureOperation::metallic_f82:
            case ClosureOperation::metallic_conductor:
            case ClosureOperation::sheen_microfiber:
            case ClosureOperation::sheen_ashikhmin:
            case ClosureOperation::hair_reflection:
            case ClosureOperation::hair_transmission:
            case ClosureOperation::glass:
            case ClosureOperation::refraction:
            case ClosureOperation::transparent:
            case ClosureOperation::subsurface:
                break;
        }
        estimates.emplace_back(estimate);
    }
    return program.root().value < estimates.size()
               ? estimates[program.root().value]
               : Vec3f{};
}

bool cycles_surface_has_bssrdf(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters) noexcept {
    for (const auto &closure : program.closure_instructions()) {
        if (closure.operation == ClosureOperation::subsurface) {
            return true;
        }
        if (closure.operation == ClosureOperation::principled &&
            principled_has_surface_bssrdf(
                program, parameters, closure)) {
            return true;
        }
    }
    return false;
}

bool cycles_surface_has_bssrdf_bump(
    const SurfaceProgram &program,
    const SurfaceParameterBlock &parameters,
    contract::DisplacementMethod displacement_method) noexcept {
    auto has_bssrdf = false;
    auto bssrdf_normal_uses_bump = false;
    for (const auto &closure : program.closure_instructions()) {
        const auto closure_has_bssrdf =
            closure.operation == ClosureOperation::subsurface ||
            (closure.operation == ClosureOperation::principled &&
             principled_has_surface_bssrdf(
                 program, parameters, closure));
        has_bssrdf |= closure_has_bssrdf;
        bssrdf_normal_uses_bump |=
            closure_has_bssrdf && closure.normal_uses_bump;
    }
    if (!has_bssrdf) {
        return false;
    }
    const auto displacement_uses_bump =
        program.surface_normal_root().valid() &&
        contract::uses_displacement_bump(displacement_method);
    return bssrdf_normal_uses_bump || displacement_uses_bump;
}

SurfaceProgramCompilation compile_surface_program(
    const ShaderProgram &shader) {
    return detail::SurfaceProgramBuilder{shader}.build();
}

SurfaceParameterBinding bind_surface_parameters(
    const SurfaceProgram &program,
    const ShaderProgram &shader) {
    SurfaceParameterBinding result;
    if (program.structure_signature() !=
        shader.analysis().structure_signature) {
        result.diagnostics.emplace_back(
            SurfaceProgramDiagnostic{
                .code =
                    SurfaceProgramDiagnosticCode::structure_mismatch,
                .message =
                    "shader structure does not match the reusable "
                    "surface program",
                .node = std::nullopt,
                .socket = {}});
        return result;
    }

    SurfaceParameterBlock block{program};
    for (const auto &parameter : program.parameters()) {
        const auto *node =
            shader.graph().find(parameter.node);
        if (node == nullptr) {
            result.diagnostics.emplace_back(
                SurfaceProgramDiagnostic{
                    .code =
                        SurfaceProgramDiagnosticCode::missing_input,
                    .message =
                        "parameter binding references a missing node",
                    .node = parameter.node,
                    .socket = parameter.socket});
            continue;
        }
        const contract::SocketValue *value = nullptr;
        if (parameter.source == ParameterSource::input) {
            const auto input = node->inputs.find(parameter.socket);
            if (input != node->inputs.end() &&
                !input->second.source &&
                input->second.value) {
                value = &*input->second.value;
            }
        } else {
            const auto property =
                node->properties.find(parameter.socket);
            if (property != node->properties.end()) {
                value = &property->second;
            }
        }
        if (value == nullptr) {
            const auto source_name =
                parameter.source == ParameterSource::input
                    ? "input"
                    : "property";
            result.diagnostics.emplace_back(
                SurfaceProgramDiagnostic{
                    .code =
                        SurfaceProgramDiagnosticCode::missing_input,
                    .message =
                        detail::node_prefix(parameter.node) +
                        "runtime parameter " + source_name + " '" +
                        parameter.socket +
                        "' is missing" +
                        (parameter.source == ParameterSource::input
                             ? " or connected"
                             : ""),
                    .node = parameter.node,
                    .socket = parameter.socket});
            continue;
        }
        if (!block.set(
                program,
                parameter.id,
                *value)) {
            const auto source_name =
                parameter.source == ParameterSource::input
                    ? "input"
                    : "property";
            result.diagnostics.emplace_back(
                SurfaceProgramDiagnostic{
                    .code =
                        SurfaceProgramDiagnosticCode::type_mismatch,
                    .message =
                        detail::node_prefix(parameter.node) +
                        "runtime parameter " + source_name + " '" +
                        parameter.socket +
                        "' changed type",
                    .node = parameter.node,
                    .socket = parameter.socket});
        }
    }

    if (result.diagnostics.empty()) {
        result.parameters = std::move(block);
    }
    return result;
}

}// namespace psycles::compiler
