#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

enum class NormalTopology {
    unlinked,
    geometry,
    normal_map
};

[[nodiscard]] ShaderGraph make_principled_graph(
    float subsurface_weight,
    NormalTopology normal_topology = NormalTopology::unlinked,
    bool thin_wall = false,
    bool automatic_displacement_bump = false) {
    ShaderGraph graph;
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Principled BSSRDF metadata");
    require(
        graph.set_input(
            principled,
            "SubsurfaceWeight",
            SocketValue::floating(subsurface_weight)) &&
            graph.set_input(
                principled,
                "ThinWall",
                SocketValue::boolean(thin_wall)),
        "failed to configure Principled BSSRDF metadata");
    if (normal_topology != NormalTopology::unlinked) {
        const auto source = graph.add_node(
            normal_topology == NormalTopology::geometry
                ? node_type::geometry
                : node_type::normal_map,
            "Principled normal source");
        require(
            graph.connect(
                {.node = source, .socket = "Normal"},
                principled,
                "Normal"),
            "failed to connect Principled normal source");
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    if (automatic_displacement_bump) {
        const auto bump = graph.add_node(
            node_type::bump,
            "Normalized automatic displacement bump");
        graph.set_root(
            ShaderDomain::surface_normal,
            OutputRef{.node = bump, .socket = "Normal"});
    }
    return graph;
}

[[nodiscard]] ShaderGraph make_standalone_bssrdf_graph(
    NormalTopology normal_topology) {
    ShaderGraph graph;
    const auto subsurface = graph.add_node(
        node_type::subsurface_scattering,
        "Standalone BSSRDF metadata");
    if (normal_topology != NormalTopology::unlinked) {
        const auto source = graph.add_node(
            normal_topology == NormalTopology::geometry
                ? node_type::geometry
                : node_type::normal_map,
            "Standalone normal source");
        require(
            graph.connect(
                {.node = source, .socket = "Normal"},
                subsurface,
                "Normal"),
            "failed to connect standalone BSSRDF normal source");
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = subsurface, .socket = "Closure"});
    return graph;
}

void test_cycles_surface_bssrdf_metadata() {
    ShaderCompiler compiler{make_core_node_registry()};
    const auto lower = [&](ShaderGraph graph) {
        const auto shader = compiler.compile(graph);
        require(shader.ok(), "BSSRDF metadata graph failed to compile");
        const auto surface = compile_surface_program(*shader.program);
        require(surface.ok(), "BSSRDF metadata graph failed to lower");
        return std::pair{
            surface.program,
            SurfaceParameterBlock{*surface.program}};
    };

    const auto [zero_program, zero_parameters] =
        lower(make_principled_graph(0.0f));
    require(
        !cycles_surface_has_bssrdf(*zero_program, zero_parameters),
        "direct zero Principled subsurface blocked static transforms");
    require(
        !cycles_surface_has_bssrdf_bump(
            *zero_program,
            zero_parameters,
            DisplacementMethod::bump),
        "zero Principled subsurface acquired BSSRDF bump metadata");

    const auto [positive_program, positive_parameters] =
        lower(make_principled_graph(0.25f));
    require(
        cycles_surface_has_bssrdf(*positive_program, positive_parameters),
        "positive Principled subsurface lost Cycles geometry metadata");
    require(
        !cycles_surface_has_bssrdf_bump(
            *positive_program,
            positive_parameters,
            DisplacementMethod::bump),
        "unlinked Principled Normal incorrectly enabled BSSRDF bump");

    const auto [geometry_program, geometry_parameters] = lower(
        make_principled_graph(
            0.25f, NormalTopology::geometry));
    require(
        cycles_surface_has_bssrdf(*geometry_program, geometry_parameters) &&
            !cycles_surface_has_bssrdf_bump(
                *geometry_program,
                geometry_parameters,
                DisplacementMethod::bump),
        "direct Geometry Normal did not preserve Cycles' no-bump proof");

    const auto [normal_map_program, normal_map_parameters] = lower(
        make_principled_graph(
            0.25f, NormalTopology::normal_map));
    require(
        cycles_surface_has_bssrdf_bump(
            *normal_map_program,
            normal_map_parameters,
            DisplacementMethod::bump),
        "linked Principled Normal lost Cycles BSSRDF bump metadata");

    ShaderGraph linked_zero_graph;
    const auto linked_zero = linked_zero_graph.add_node(
        node_type::constant_float,
        "Linked zero weight");
    const auto linked_principled = linked_zero_graph.add_node(
        node_type::principled_bsdf,
        "Linked Principled subsurface");
    require(
        linked_zero_graph.set_input(
            linked_zero,
            "Value",
            SocketValue::floating(0.0f)) &&
            linked_zero_graph.connect(
                {.node = linked_zero, .socket = "Value"},
                linked_principled,
                "SubsurfaceWeight"),
        "failed to link Principled subsurface weight");
    linked_zero_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = linked_principled, .socket = "Closure"});
    const auto [linked_program, linked_parameters] =
        lower(linked_zero_graph);
    require(
        cycles_surface_has_bssrdf(
            *linked_program, linked_parameters),
        "linked zero subsurface was incorrectly host-evaluated");

    const auto [explicit_program, explicit_parameters] =
        lower(make_standalone_bssrdf_graph(
            NormalTopology::unlinked));
    require(
        cycles_surface_has_bssrdf(
            *explicit_program, explicit_parameters),
        "explicit Subsurface closure lost Cycles geometry metadata");
    require(
        !cycles_surface_has_bssrdf_bump(
            *explicit_program,
            explicit_parameters,
            DisplacementMethod::bump),
        "unlinked standalone BSSRDF Normal incorrectly enabled bump");

    const auto [explicit_bump_program, explicit_bump_parameters] =
        lower(make_standalone_bssrdf_graph(
            NormalTopology::normal_map));
    require(
        cycles_surface_has_bssrdf_bump(
            *explicit_bump_program,
            explicit_bump_parameters,
            DisplacementMethod::bump),
        "standalone BSSRDF linked Normal lost bump metadata");

    const auto [displacement_program, displacement_parameters] = lower(
        make_principled_graph(
            0.25f,
            NormalTopology::unlinked,
            false,
            true));
    require(
        displacement_program->surface_normal_root().valid() &&
            cycles_surface_has_bssrdf_bump(
                *displacement_program,
                displacement_parameters,
                DisplacementMethod::bump) &&
            cycles_surface_has_bssrdf_bump(
                *displacement_program,
                displacement_parameters,
                DisplacementMethod::both) &&
            !cycles_surface_has_bssrdf_bump(
                *displacement_program,
                displacement_parameters,
                DisplacementMethod::displacement),
        "automatic displacement bump policy diverged from Cycles");

    const auto thin_graph = make_principled_graph(
        0.25f, NormalTopology::normal_map, true);
    const auto thin_shader = compiler.compile(thin_graph);
    require(thin_shader.ok(), "Thin Wall metadata graph failed to compile");
    require(
        thin_shader.program->analysis().structure_signature ==
            normal_map_program->structure_signature(),
        "direct Thin Wall parameter changed reusable topology");
    const auto thin_parameters = bind_surface_parameters(
        *normal_map_program, *thin_shader.program);
    require(
        thin_parameters.ok() &&
            !cycles_surface_has_bssrdf(
                *normal_map_program,
                *thin_parameters.parameters) &&
            !cycles_surface_has_bssrdf_bump(
                *normal_map_program,
                *thin_parameters.parameters,
                DisplacementMethod::bump),
        "direct Thin Wall did not disable real Principled BSSRDF metadata");

    ShaderGraph mixed_graph;
    const auto mixed_normal = mixed_graph.add_node(
        node_type::normal_map, "Zero-weight bumped normal");
    const auto mixed_principled = mixed_graph.add_node(
        node_type::principled_bsdf, "Zero-weight bumped Principled");
    const auto mixed_subsurface = mixed_graph.add_node(
        node_type::subsurface_scattering, "Unbumped standalone BSSRDF");
    const auto mixed_root = mixed_graph.add_node(
        node_type::add_closure, "Mixed BSSRDF root");
    require(
        mixed_graph.connect(
            {.node = mixed_normal, .socket = "Normal"},
            mixed_principled,
            "Normal") &&
            mixed_graph.connect(
                {.node = mixed_principled, .socket = "Closure"},
                mixed_root,
                "A") &&
            mixed_graph.connect(
                {.node = mixed_subsurface, .socket = "Closure"},
                mixed_root,
                "B"),
        "failed to configure mixed BSSRDF bump proof graph");
    mixed_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = mixed_root, .socket = "Closure"});
    const auto [mixed_program, mixed_parameters] = lower(mixed_graph);
    require(
        cycles_surface_has_bssrdf(*mixed_program, mixed_parameters) &&
            !cycles_surface_has_bssrdf_bump(
                *mixed_program,
                mixed_parameters,
                DisplacementMethod::bump),
        "zero-weight bumped closure contaminated another BSSRDF's proof");
}

[[nodiscard]] ShaderGraph make_closure_plan_principled(
    float alpha, float sheen, float coat, float metallic, float transmission,
    float subsurface, bool thin_wall, float subsurface_scale = 1.0f,
    psycles::Vec3f emission_color = {}, float emission_strength = 1.0f,
    float specular_ior_level = 0.5f) {
    ShaderGraph graph;
    const auto principled =
        graph.add_node(node_type::principled_bsdf, "Closure-plan Principled");
    require(
        graph.set_input(principled, "BaseColor",
                        SocketValue::color({0.4f, 0.3f, 0.2f})) &&
            graph.set_input(principled, "Alpha", SocketValue::floating(alpha)) &&
            graph.set_input(principled, "SheenWeight",
                            SocketValue::floating(sheen)) &&
            graph.set_input(principled, "CoatWeight",
                            SocketValue::floating(coat)) &&
            graph.set_input(principled, "Metallic",
                            SocketValue::floating(metallic)) &&
            graph.set_input(principled, "TransmissionWeight",
                            SocketValue::floating(transmission)) &&
            graph.set_input(principled, "SubsurfaceWeight",
                            SocketValue::floating(subsurface)) &&
            graph.set_input(principled, "SubsurfaceScale",
                            SocketValue::floating(subsurface_scale)) &&
            graph.set_input(principled, "ThinWall",
                            SocketValue::boolean(thin_wall)) &&
            graph.set_input(principled, "EmissionColor",
                            SocketValue::color(emission_color)) &&
            graph.set_input(principled, "EmissionStrength",
                            SocketValue::floating(emission_strength)) &&
            graph.set_input(principled, "IOR", SocketValue::floating(1.45f)) &&
            graph.set_input(principled, "SpecularIORLevel",
                            SocketValue::floating(specular_ior_level)),
        "failed to configure closure-plan Principled");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

void test_surface_closure_plan() {
    ShaderCompiler compiler{make_core_node_registry()};
    const auto compile = [&](ShaderGraph graph) {
        const auto shader = compiler.compile(graph);
        require(shader.ok(), "closure-plan graph failed to compile");
        const auto lowered = compile_surface_program(*shader.program);
        require(lowered.ok(), "closure-plan graph failed to lower");
        return std::pair{shader.program, lowered.program};
    };
    const auto feature = [](const SurfaceClosurePlanEntry &entry,
                            PrincipledClosureFeature value) noexcept {
        return (entry.principled_features &
                principled_closure_feature_bit(value)) != 0u;
    };
    const auto principled_entry =
        [](const SurfaceProgram &program,
           const SurfaceClosurePlan &plan) -> const SurfaceClosurePlanEntry & {
        for (std::size_t index = 0u; index < program.closure_instructions().size();
             ++index) {
            if (program.closure_instructions()[index].operation ==
                ClosureOperation::principled) {
                return plan.entry(
                    ClosureExpressionId{static_cast<std::uint32_t>(index)});
            }
        }
        throw std::runtime_error{"closure-plan graph has no Principled leaf"};
    };
    const auto principled_instruction =
        [](const SurfaceProgram &program) -> const ClosureInstruction & {
        for (const auto &instruction : program.closure_instructions()) {
            if (instruction.operation == ClosureOperation::principled) {
                return instruction;
            }
        }
        throw std::runtime_error{
            "value-dependency graph has no Principled leaf"};
    };
    const auto active = [](const std::vector<bool> &mask,
                           ValueExpressionId id) noexcept {
        return id.valid() && id.value < mask.size() && mask[id.value];
    };
    const auto closure_active = [](const std::vector<bool> &mask,
                                   ClosureExpressionId id) noexcept {
        return id.valid() && id.value < mask.size() && mask[id.value];
    };
    const auto require_topology_closed = [&](const SurfaceProgram &program,
                                             const std::vector<bool> &mask,
                                             const std::string &domain) {
        require(mask.size() == program.value_instructions().size(),
                domain + " dependency mask has the wrong size");
        for (auto index = std::size_t{0u}; index < mask.size(); ++index) {
            if (!mask[index]) {
                continue;
            }
            for (const auto operand :
                 program.value_instructions()[index].operands) {
                require(!operand.valid() || active(mask, operand),
                        domain +
                            " dependency mask is not transitively closed");
            }
        }
    };

    const auto [base_shader, base_program] = compile(
        make_closure_plan_principled(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false));
    const auto base_binding =
        bind_surface_parameters(*base_program, *base_shader);
    require(base_binding.ok(), "failed to bind base closure plan");
    auto union_plan =
        analyze_surface_closure_plan(*base_program, *base_binding.parameters);
    const auto &base = principled_entry(*base_program, union_plan);
    require(base.reachable && !feature(base, PrincipledClosureFeature::alpha) &&
                !feature(base, PrincipledClosureFeature::sheen) &&
                !feature(base, PrincipledClosureFeature::coat) &&
                !feature(base, PrincipledClosureFeature::metallic) &&
                !feature(base, PrincipledClosureFeature::thick_transmission) &&
                !feature(base, PrincipledClosureFeature::thin_transmission) &&
                feature(base, PrincipledClosureFeature::dielectric) &&
                !feature(base, PrincipledClosureFeature::thick_subsurface) &&
                !feature(base, PrincipledClosureFeature::thin_subsurface) &&
                feature(base, PrincipledClosureFeature::diffuse) &&
                !feature(base, PrincipledClosureFeature::emission),
            "direct zero Principled sockets did not remove physical lobe code");

    const auto base_dependencies = analyze_surface_value_dependencies(
        *base_program, union_plan);
    require(base_dependencies.compatible(*base_program),
            "surface value dependency plan is incompatible with its program");
    const auto require_outputs_closed = [&](const std::vector<bool> &active_mask,
                                             const std::vector<bool> &outputs,
                                             const std::string &domain) {
        require(outputs.size() == active_mask.size(),
                domain + " output mask has the wrong size");
        for (auto index = std::size_t{0u}; index < outputs.size(); ++index) {
            require(!outputs[index] || active_mask[index],
                    domain + " output is absent from its active value mask");
        }
    };
    require_outputs_closed(base_dependencies.physical,
                           base_dependencies.physical_outputs, "physical");
    require_outputs_closed(base_dependencies.emission,
                           base_dependencies.emission_outputs, "emission");
    require_outputs_closed(base_dependencies.preparation,
                           base_dependencies.preparation_outputs,
                           "preparation");
    auto output_union = base_dependencies.physical_outputs;
    for (auto index = std::size_t{0u}; index < output_union.size(); ++index) {
        output_union[index] = output_union[index] ||
                              base_dependencies.emission_outputs[index];
    }
    require(base_dependencies.preparation_outputs == output_union,
            "preparation outputs are not the physical/emission union");
    require(
        closure_active(base_dependencies.physical_closures,
                       base_program->root()) &&
            std::none_of(base_dependencies.emission_closures.begin(),
                         base_dependencies.emission_closures.end(),
                         [](bool value) noexcept { return value; }),
        "non-emissive Principled closure leaked across consumer domains");
    require_topology_closed(
        *base_program, base_dependencies.physical, "physical");
    require_topology_closed(
        *base_program, base_dependencies.emission, "emission");
    require_topology_closed(
        *base_program, base_dependencies.preparation, "preparation");
    const auto &base_closure = principled_instruction(*base_program);
    require(
        active(base_dependencies.physical_outputs, base_closure.color) &&
            active(base_dependencies.physical_outputs, base_closure.normal) &&
            active(base_dependencies.physical_outputs,
                   base_closure.roughness),
        "physical closure roots were not recorded as terminal outputs");
    require(
        active(base_dependencies.physical, base_closure.color) &&
            active(base_dependencies.physical, base_closure.normal) &&
            active(base_dependencies.physical, base_closure.roughness) &&
            active(base_dependencies.physical,
                   base_closure.diffuse_roughness) &&
            active(base_dependencies.physical,
                   base_closure.subsurface_weight) &&
            active(base_dependencies.physical, base_closure.ior) &&
            active(base_dependencies.physical,
                   base_closure.specular_ior_level) &&
            active(base_dependencies.physical,
                   base_closure.specular_tint),
        "physical dependency plan lost a reachable dielectric/diffuse input");
    require(
        !active(base_dependencies.physical, base_closure.alpha) &&
            !active(base_dependencies.physical, base_closure.sheen_weight) &&
            !active(base_dependencies.physical, base_closure.coat_weight) &&
            !active(base_dependencies.physical, base_closure.coat_tint) &&
            !active(base_dependencies.physical, base_closure.metallic) &&
            !active(base_dependencies.physical,
                    base_closure.transmission_weight) &&
            !active(base_dependencies.physical, base_closure.thin_wall) &&
            !active(base_dependencies.physical,
                    base_closure.emission_color) &&
            !active(base_dependencies.physical,
                    base_closure.emission_strength),
        "physical dependency plan retained a proven-unreachable socket family");
    require(
        std::none_of(base_dependencies.emission.begin(),
                     base_dependencies.emission.end(),
                     [](bool value) noexcept { return value; }) &&
            base_dependencies.preparation == base_dependencies.physical,
        "non-emissive preparation retained an emission-only value schedule");
    const auto base_storage = plan_surface_value_storage(
        *base_program, base_dependencies.preparation,
        base_dependencies.preparation_outputs);
    require(base_storage.compatible(*base_program),
            "base surface value storage plan is invalid: " +
                base_storage.diagnostic);

    const auto [diffuse_shader, diffuse_program] = compile(
        make_closure_plan_principled(
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, false, 1.0f, {}, 1.0f,
            0.0f));
    const auto diffuse_binding =
        bind_surface_parameters(*diffuse_program, *diffuse_shader);
    require(diffuse_binding.ok(), "failed to bind diffuse-only closure plan");
    const auto diffuse_plan = analyze_surface_closure_plan(
        *diffuse_program, *diffuse_binding.parameters);
    const auto &diffuse_entry =
        principled_entry(*diffuse_program, diffuse_plan);
    require(
        feature(diffuse_entry, PrincipledClosureFeature::diffuse) &&
            !feature(diffuse_entry, PrincipledClosureFeature::dielectric),
        "diffuse-only regression graph did not isolate the diffuse family");
    const auto diffuse_dependencies = analyze_surface_value_dependencies(
        *diffuse_program, diffuse_plan);
    const auto &diffuse_closure =
        principled_instruction(*diffuse_program);
    require(active(diffuse_dependencies.physical, diffuse_closure.normal),
            "diffuse-only value schedule lost its authored normal");

    const auto [thin_shader, thin_program] = compile(
        make_closure_plan_principled(0.6f, 0.2f, 0.3f, 0.4f, 0.5f, 0.25f, true));
    require(thin_program->structure_signature() ==
                base_program->structure_signature(),
            "closure-plan literals changed reusable topology");
    const auto thin_binding =
        bind_surface_parameters(*base_program, *thin_shader);
    require(thin_binding.ok(), "failed to bind thin closure plan");
    union_plan.merge(
        analyze_surface_closure_plan(*base_program, *thin_binding.parameters));
    const auto &combined = principled_entry(*base_program, union_plan);
    require(
        feature(combined, PrincipledClosureFeature::alpha) &&
            feature(combined, PrincipledClosureFeature::sheen) &&
            feature(combined, PrincipledClosureFeature::coat) &&
            feature(combined, PrincipledClosureFeature::metallic) &&
            !feature(combined, PrincipledClosureFeature::thick_transmission) &&
            feature(combined, PrincipledClosureFeature::thin_transmission) &&
            feature(combined, PrincipledClosureFeature::dielectric) &&
            !feature(combined, PrincipledClosureFeature::thick_subsurface) &&
            feature(combined, PrincipledClosureFeature::thin_subsurface) &&
            feature(combined, PrincipledClosureFeature::diffuse) &&
            !feature(combined, PrincipledClosureFeature::emission),
        "same-topology parameter union lost reachable thin Principled lobes");

    const auto [thick_shader, thick_program] = compile(
        make_closure_plan_principled(0.6f, 0.2f, 0.3f, 0.4f, 0.5f, 0.25f, false));
    require(thick_program->structure_signature() ==
                base_program->structure_signature(),
            "thick closure-plan literals changed reusable topology");
    const auto thick_binding =
        bind_surface_parameters(*base_program, *thick_shader);
    require(thick_binding.ok(), "failed to bind thick closure plan");
    union_plan.merge(
        analyze_surface_closure_plan(*base_program, *thick_binding.parameters));
    const auto &both = principled_entry(*base_program, union_plan);
    require(feature(both, PrincipledClosureFeature::thick_transmission) &&
                feature(both, PrincipledClosureFeature::thin_transmission) &&
                feature(both, PrincipledClosureFeature::thick_subsurface) &&
                feature(both, PrincipledClosureFeature::thin_subsurface),
            "same topology did not union mutually exclusive thick/thin lobes");

    const auto [emitting_shader, emitting_program] = compile(
        make_closure_plan_principled(
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            0.0f,
            false,
            1.0f,
            psycles::Vec3f{0.25f, 0.5f, 1.0f},
            2.0f));
    require(emitting_program->structure_signature() ==
                base_program->structure_signature(),
            "Principled emission literals changed reusable topology");
    const auto emitting_binding =
        bind_surface_parameters(*base_program, *emitting_shader);
    require(emitting_binding.ok(),
            "failed to bind emitting closure plan");
    auto emission_union = analyze_surface_closure_plan(
        *base_program, *base_binding.parameters);
    emission_union.merge(analyze_surface_closure_plan(
        *base_program, *emitting_binding.parameters));
    require(feature(principled_entry(*base_program, emission_union),
                    PrincipledClosureFeature::emission),
            "same-topology plans did not retain reachable Principled emission");
    const auto emission_dependencies =
        analyze_surface_value_dependencies(*base_program, emission_union);
    require(closure_active(emission_dependencies.emission_closures,
                           base_program->root()) &&
                active(emission_dependencies.emission,
                       base_closure.emission_color) &&
                active(emission_dependencies.emission,
                       base_closure.emission_strength),
            "reachable Principled emission lost its authored value roots");

    ShaderGraph linked_graph;
    const auto zero =
        linked_graph.add_node(node_type::constant_float, "Linked zero sheen");
    const auto linked_principled = linked_graph.add_node(
        node_type::principled_bsdf, "Linked closure-plan Principled");
    require(linked_graph.set_input(zero, "Value", SocketValue::floating(0.0f)) &&
                linked_graph.connect({.node = zero, .socket = "Value"},
                                     linked_principled, "SheenWeight"),
            "failed to configure linked closure plan");
    linked_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = linked_principled, .socket = "Closure"});
    const auto [linked_shader, linked_program] = compile(linked_graph);
    const auto linked_binding =
        bind_surface_parameters(*linked_program, *linked_shader);
    require(linked_binding.ok(), "failed to bind linked closure plan");
    const auto linked_plan =
        analyze_surface_closure_plan(*linked_program, *linked_binding.parameters);
    require(
        feature(principled_entry(*linked_program, linked_plan),
                PrincipledClosureFeature::sheen),
        "linked numerical zero was incorrectly host-folded from closure plan");
    const auto linked_dependencies =
        analyze_surface_value_dependencies(*linked_program, linked_plan);
    const auto &linked_closure = principled_instruction(*linked_program);
    require(active(linked_dependencies.physical,
                   linked_closure.sheen_weight),
            "linked Sheen input was incorrectly removed from value schedule");

    ShaderGraph linked_emission_graph;
    const auto linked_zero_color = linked_emission_graph.add_node(
        node_type::constant_color, "Linked zero emission color");
    const auto linked_emission_principled = linked_emission_graph.add_node(
        node_type::principled_bsdf, "Linked emission closure-plan Principled");
    require(
        linked_emission_graph.set_input(
            linked_zero_color,
            "Color",
            SocketValue::color({0.0f, 0.0f, 0.0f})) &&
            linked_emission_graph.set_input(
                linked_emission_principled,
                "EmissionStrength",
                SocketValue::floating(1.0f)) &&
            linked_emission_graph.connect(
                {.node = linked_zero_color, .socket = "Color"},
                linked_emission_principled,
                "EmissionColor"),
        "failed to configure linked emission closure plan");
    linked_emission_graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = linked_emission_principled,
            .socket = "Closure"});
    const auto [linked_emission_shader, linked_emission_program] =
        compile(linked_emission_graph);
    const auto linked_emission_binding = bind_surface_parameters(
        *linked_emission_program, *linked_emission_shader);
    require(linked_emission_binding.ok(),
            "failed to bind linked emission closure plan");
    require(
        feature(
            principled_entry(
                *linked_emission_program,
                analyze_surface_closure_plan(
                    *linked_emission_program,
                    *linked_emission_binding.parameters)),
            PrincipledClosureFeature::emission),
        "linked numerical zero was incorrectly removed from emission plan");

    const auto [thin_zero_scale_shader, thin_zero_scale_program] =
        compile(make_closure_plan_principled(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.25f,
                                             true, 0.0f));
    const auto thin_zero_scale_binding = bind_surface_parameters(
        *thin_zero_scale_program, *thin_zero_scale_shader);
    require(thin_zero_scale_binding.ok(),
            "failed to bind zero-scale thin-subsurface closure plan");
    require(feature(principled_entry(*thin_zero_scale_program,
                                     analyze_surface_closure_plan(
                                         *thin_zero_scale_program,
                                         *thin_zero_scale_binding.parameters)),
                    PrincipledClosureFeature::thin_subsurface),
            "subsurface scale incorrectly disabled Thin Wall scattering");

    const auto make_mix = [](float factor) {
        ShaderGraph graph;
        const auto diffuse =
            graph.add_node(node_type::diffuse_bsdf, "Closure-plan diffuse");
        const auto glossy =
            graph.add_node(node_type::glossy_bsdf, "Closure-plan glossy");
        const auto mix = graph.add_node(node_type::mix_closure, "Closure-plan mix");
        require(
            graph.connect({.node = diffuse, .socket = "Closure"}, mix, "A") &&
                graph.connect({.node = glossy, .socket = "Closure"}, mix, "B") &&
                graph.set_input(mix, "Factor", SocketValue::floating(factor)),
            "failed to configure closure-plan mix");
        graph.set_root(ShaderDomain::surface,
                       OutputRef{.node = mix, .socket = "Closure"});
        return graph;
    };

    ShaderGraph domain_mix_graph;
    const auto domain_diffuse = domain_mix_graph.add_node(
        node_type::diffuse_bsdf, "Physical domain leaf");
    const auto domain_emission = domain_mix_graph.add_node(
        node_type::emission, "Emission domain leaf");
    const auto domain_mix = domain_mix_graph.add_node(
        node_type::mix_closure, "Consumer-domain mix");
    require(
        domain_mix_graph.set_input(
            domain_emission, "Color", SocketValue::color({0.5f, 0.25f, 0.1f})) &&
            domain_mix_graph.set_input(
                domain_emission, "Strength", SocketValue::floating(2.0f)) &&
            domain_mix_graph.set_input(
                domain_mix, "Factor", SocketValue::floating(0.25f)) &&
            domain_mix_graph.connect(
                {.node = domain_diffuse, .socket = "Closure"},
                domain_mix,
                "A") &&
            domain_mix_graph.connect(
                {.node = domain_emission, .socket = "Closure"},
                domain_mix,
                "B"),
        "failed to configure consumer-domain closure mix");
    domain_mix_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = domain_mix, .socket = "Closure"});
    const auto [domain_mix_shader, domain_mix_program] =
        compile(domain_mix_graph);
    const auto domain_mix_binding = bind_surface_parameters(
        *domain_mix_program, *domain_mix_shader);
    require(domain_mix_binding.ok(),
            "failed to bind consumer-domain closure mix");
    const auto domain_mix_plan = analyze_surface_closure_plan(
        *domain_mix_program, *domain_mix_binding.parameters);
    const auto domain_dependencies = analyze_surface_value_dependencies(
        *domain_mix_program, domain_mix_plan);
    auto diffuse_id = ClosureExpressionId{};
    auto emission_id = ClosureExpressionId{};
    auto mix_id = ClosureExpressionId{};
    for (auto index = std::size_t{0u};
         index < domain_mix_program->closure_instructions().size(); ++index) {
        const auto id = ClosureExpressionId{
            static_cast<std::uint32_t>(index)};
        switch (domain_mix_program->closure_instructions()[index].operation) {
            case ClosureOperation::diffuse:
                diffuse_id = id;
                break;
            case ClosureOperation::emission:
                emission_id = id;
                break;
            case ClosureOperation::mix:
                mix_id = id;
                break;
            default:
                break;
        }
    }
    require(
        closure_active(domain_dependencies.physical_closures, mix_id) &&
            closure_active(domain_dependencies.physical_closures, diffuse_id) &&
            !closure_active(domain_dependencies.physical_closures, emission_id) &&
            closure_active(domain_dependencies.emission_closures, mix_id) &&
            !closure_active(domain_dependencies.emission_closures, diffuse_id) &&
            closure_active(domain_dependencies.emission_closures, emission_id),
        "physical/emission closure domains did not partition a mixed tree");
    const auto &domain_mix_instruction =
        domain_mix_program->closure_instructions()[mix_id.value];
    require(active(domain_dependencies.physical,
                   domain_mix_instruction.factor) &&
                active(domain_dependencies.emission,
                       domain_mix_instruction.factor),
            "a single live Mix branch lost its consumer-domain factor");

    const auto [mix_zero_shader, mix_program] = compile(make_mix(0.0f));
    const auto mix_zero_binding =
        bind_surface_parameters(*mix_program, *mix_zero_shader);
    require(mix_zero_binding.ok(), "failed to bind zero closure mix");
    auto mix_plan =
        analyze_surface_closure_plan(*mix_program, *mix_zero_binding.parameters);
    auto diffuse_reachable = false;
    auto glossy_reachable = false;
    for (std::size_t index = 0u;
         index < mix_program->closure_instructions().size(); ++index) {
        const auto &instruction = mix_program->closure_instructions()[index];
        const auto reachable =
            mix_plan.entry(ClosureExpressionId{static_cast<std::uint32_t>(index)})
                .reachable;
        diffuse_reachable |=
            instruction.operation == ClosureOperation::diffuse && reachable;
        glossy_reachable |=
            instruction.operation == ClosureOperation::glossy && reachable;
    }
    require(diffuse_reachable && !glossy_reachable,
            "direct zero Mix factor did not prune its B closure");
    const auto [mix_one_shader, mix_one_program] = compile(make_mix(1.0f));
    require(mix_one_program->structure_signature() ==
                mix_program->structure_signature(),
            "Mix factor literal changed reusable topology");
    const auto mix_one_binding =
        bind_surface_parameters(*mix_program, *mix_one_shader);
    require(mix_one_binding.ok(), "failed to bind one closure mix");
    mix_plan.merge(
        analyze_surface_closure_plan(*mix_program, *mix_one_binding.parameters));
    diffuse_reachable = false;
    glossy_reachable = false;
    for (std::size_t index = 0u;
         index < mix_program->closure_instructions().size(); ++index) {
        const auto &instruction = mix_program->closure_instructions()[index];
        const auto reachable =
            mix_plan.entry(ClosureExpressionId{static_cast<std::uint32_t>(index)})
                .reachable;
        diffuse_reachable |=
            instruction.operation == ClosureOperation::diffuse && reachable;
        glossy_reachable |=
            instruction.operation == ClosureOperation::glossy && reachable;
    }
    require(diffuse_reachable && glossy_reachable,
            "same-topology Mix plans did not union both selected branches");

    const auto [mix_nan_shader, mix_nan_program] =
        compile(make_mix(std::numeric_limits<float>::quiet_NaN()));
    const auto mix_nan_binding =
        bind_surface_parameters(*mix_nan_program, *mix_nan_shader);
    require(mix_nan_binding.ok(), "failed to bind NaN closure mix");
    const auto mix_nan_plan = analyze_surface_closure_plan(
        *mix_nan_program, *mix_nan_binding.parameters);
    diffuse_reachable = false;
    glossy_reachable = false;
    for (std::size_t index = 0u;
         index < mix_nan_program->closure_instructions().size(); ++index) {
        const auto &instruction = mix_nan_program->closure_instructions()[index];
        const auto reachable =
            mix_nan_plan
                .entry(ClosureExpressionId{static_cast<std::uint32_t>(index)})
                .reachable;
        diffuse_reachable |=
            instruction.operation == ClosureOperation::diffuse && reachable;
        glossy_reachable |=
            instruction.operation == ClosureOperation::glossy && reachable;
    }
    require(diffuse_reachable && glossy_reachable,
            "non-finite Mix factor was incorrectly treated as a proof");
}

void test_surface_value_storage_plan() {
    const auto make_parameter = [](std::uint32_t index) {
        return ParameterDesc{
            .id = ParameterId{index},
            .node = NodeId{index + 1u},
            .socket = "Value",
            .type = SocketType::floating,
            .default_value = SocketValue::floating(0.0f),
            .source = ParameterSource::input};
    };
    const auto make_parameter_value = [](std::uint32_t index) {
        return ValueInstruction{
            .operation = ValueOperation::parameter,
            .source_node = NodeId{index + 1u},
            .result_type = SocketType::floating,
            .parameter = ParameterId{index}};
    };

    std::vector<ValueInstruction> values;
    values.emplace_back(make_parameter_value(0u));
    values.emplace_back(make_parameter_value(1u));
    values.emplace_back(ValueInstruction{
        .operation = ValueOperation::add,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::binary>({
            {value_operand::binary::a, ValueExpressionId{0u}},
            {value_operand::binary::b, ValueExpressionId{1u}}})});
    values.emplace_back(ValueInstruction{
        .operation = ValueOperation::absolute,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::unary>({
            {value_operand::unary::input, ValueExpressionId{2u}}})});
    values.emplace_back(ValueInstruction{
        .operation = ValueOperation::add,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::binary>({
            {value_operand::binary::a, ValueExpressionId{3u}},
            {value_operand::binary::b, ValueExpressionId{1u}}})});
    const SurfaceProgram program{
        1u,
        {make_parameter(0u), make_parameter(1u)},
        std::move(values),
        {},
        {}};
    const std::vector<bool> active(5u, true);
    auto outputs = std::vector<bool>(5u, false);
    outputs[4u] = true;
    const auto plan = plan_surface_value_storage(program, active, outputs);
    require(plan.compatible(program),
            "typed value storage plan failed: " + plan.diagnostic);
    require(plan.active_values == 5u && plan.parameter_values == 2u &&
                plan.instructions.size() == 3u,
            "typed value storage plan did not elide parameter instructions");
    require(plan.scalar_slots == 1u && plan.vector_slots == 0u &&
                plan.unsigned_integer_slots == 0u &&
                plan.payload_bytes() == sizeof(float),
            "read-before-write liveness did not reuse a dying scalar slot");
    require(plan.locations[0u].storage ==
                    SurfaceValueStorageClass::parameter &&
                plan.locations[1u].storage ==
                    SurfaceValueStorageClass::parameter,
            "parameter expressions were copied into local slots");
    for (auto index = std::size_t{2u}; index < 5u; ++index) {
        require(plan.locations[index].storage ==
                        SurfaceValueStorageClass::local_slot &&
                    plan.locations[index].bank == SurfaceValueBank::scalar &&
                    plan.locations[index].index == 0u,
                "a linear scalar chain did not share its single live slot");
    }
    const auto image = lower_surface_value_program(program, plan);
    require(image.valid,
            "typed value program lowering failed: " + image.diagnostic);
    require(image.instructions.size() == 3u && image.operands.size() == 5u &&
                image.metadata.empty() && image.static_data.empty() &&
                image.value_addresses.size() == 5u,
            "compact value program has an unexpected stream extent");
    for (auto index = std::size_t{0u}; index < 2u; ++index) {
        const auto address = SurfaceValueAddress{image.value_addresses[index]};
        require(address.valid() && address.parameter() &&
                    address.bank() == SurfaceValueBank::scalar &&
                    address.index() == index,
                "compact value program lost a parameter address");
    }
    for (auto index = std::size_t{2u}; index < 5u; ++index) {
        const auto address = SurfaceValueAddress{image.value_addresses[index]};
        require(address.valid() && !address.parameter() &&
                    address.bank() == SurfaceValueBank::scalar &&
                    address.index() == 0u,
                "compact value program lost a reused local address");
    }
    require(image.instructions[0u].operand_begin == 0u &&
                image.instructions[1u].operand_begin == 2u &&
                image.instructions[2u].operand_begin == 3u &&
                surface_value_operation(image.instructions[0u]) ==
                    ValueOperation::add &&
                surface_value_operand_count(image.instructions[0u]) == 2u &&
                surface_value_result_bank(image.instructions[0u]) ==
                    SurfaceValueBank::scalar &&
                surface_value_operation(image.instructions[1u]) ==
                    ValueOperation::absolute &&
                surface_value_operand_count(image.instructions[1u]) == 1u,
            "compact value program changed topological instruction order");

    // Deliberately place a shallow independent branch before a two-value
    // branch. Source order needs three scalar slots: the shallow result stays
    // live while both deep operands are formed. Cycles-style Sethi-Ullman
    // scheduling evaluates the high-pressure branch first and needs two.
    std::vector<ValueInstruction> pressure_values;
    pressure_values.emplace_back(make_parameter_value(0u));
    pressure_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::absolute,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::unary>({
            {value_operand::unary::input, ValueExpressionId{0u}}})});
    pressure_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::clamp01,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::unary>({
            {value_operand::unary::input, ValueExpressionId{0u}}})});
    pressure_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::add,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::binary>({
            {value_operand::binary::a, ValueExpressionId{0u}},
            {value_operand::binary::b, ValueExpressionId{0u}}})});
    pressure_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::add,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::binary>({
            {value_operand::binary::a, ValueExpressionId{2u}},
            {value_operand::binary::b, ValueExpressionId{3u}}})});
    pressure_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::add,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::binary>({
            {value_operand::binary::a, ValueExpressionId{4u}},
            {value_operand::binary::b, ValueExpressionId{1u}}})});
    const SurfaceProgram pressure_program{
        2u, {make_parameter(0u)}, std::move(pressure_values), {}, {}};
    const auto pressure_active =
        std::vector<bool>(pressure_program.value_instructions().size(), true);
    auto pressure_outputs =
        std::vector<bool>(pressure_program.value_instructions().size(), false);
    pressure_outputs[5u] = true;
    const auto pressure_plan = plan_surface_value_storage(
        pressure_program, pressure_active, pressure_outputs);
    require(pressure_plan.compatible(pressure_program),
            "Sethi-Ullman value plan failed: " + pressure_plan.diagnostic);
    require(pressure_plan.instructions ==
                std::vector<ValueExpressionId>{ValueExpressionId{2u},
                                               ValueExpressionId{3u},
                                               ValueExpressionId{4u},
                                               ValueExpressionId{1u},
                                               ValueExpressionId{5u}},
            "value scheduler did not prioritize the high-pressure branch");
    require(pressure_plan.scalar_slots == 2u,
            "Sethi-Ullman scheduling did not remove the avoidable live slot");
    const auto pressure_image =
        lower_surface_value_program(pressure_program, pressure_plan);
    require(pressure_image.valid &&
                pressure_image.instructions.size() == 5u &&
                surface_value_operation(pressure_image.instructions[0u]) ==
                    ValueOperation::clamp01 &&
                surface_value_operation(pressure_image.instructions[3u]) ==
                    ValueOperation::absolute,
            "compact bytecode did not preserve the scheduled dependency order");

    // Cycles has one scalar stack, whereas the Psycles interpreter has typed
    // banks. This DAG is the minimal counterexample where scalar-stack
    // Sethi-Ullman order would keep two vectors live (28 B total) while the
    // authored topological order needs one vector and one scalar (16 B).
    // The planner must retain the lower-pressure legal order.
    const SurfaceProgram typed_pressure_program{
        3u,
        {},
        {ValueInstruction{.operation = ValueOperation::surface_position,
                          .result_type = SocketType::point},
         ValueInstruction{
             .operation = ValueOperation::vector_to_scalar,
             .result_type = SocketType::floating,
             .operands = make_value_operands<value_operand::unary>({
                 {value_operand::unary::input, ValueExpressionId{0u}}})},
         ValueInstruction{
             .operation = ValueOperation::passthrough,
             .result_type = SocketType::vector,
             .operands = make_value_operands<value_operand::unary>({
                 {value_operand::unary::input, ValueExpressionId{0u}}})},
         ValueInstruction{
             .operation = ValueOperation::absolute,
             .result_type = SocketType::floating,
             .operands = make_value_operands<value_operand::unary>({
                 {value_operand::unary::input, ValueExpressionId{1u}}})}},
        {},
        {}};
    const auto typed_pressure_active = std::vector<bool>(4u, true);
    const auto typed_pressure_outputs =
        std::vector<bool>{false, false, true, true};
    const auto typed_pressure_plan = plan_surface_value_storage(
        typed_pressure_program,
        typed_pressure_active,
        typed_pressure_outputs);
    require(typed_pressure_plan.compatible(typed_pressure_program),
            "typed-bank pressure plan failed: " +
                typed_pressure_plan.diagnostic);
    require(typed_pressure_plan.instructions ==
                std::vector<ValueExpressionId>{ValueExpressionId{0u},
                                               ValueExpressionId{1u},
                                               ValueExpressionId{2u},
                                               ValueExpressionId{3u}} &&
                typed_pressure_plan.scalar_slots == 1u &&
                typed_pressure_plan.vector_slots == 1u &&
                typed_pressure_plan.payload_bytes() == 16u,
            "typed-bank no-regression selection retained a worse schedule");

    std::vector<float> transform(16u, 0.0f);
    for (auto diagonal = std::size_t{0u}; diagonal < 4u; ++diagonal) {
        transform[diagonal * 5u] = 1.0f;
    }
    const SurfaceProgram metadata_program{
        3u,
        {},
        {ValueInstruction{
            .operation = ValueOperation::object_position_with_transform,
            .result_type = SocketType::point,
            .static_u0 = 7u,
            .static_f0 = 0.25f,
            .static_f1 = -0.0f,
            .static_table = transform}},
        {},
        {}};
    const auto metadata_plan = plan_surface_value_storage(
        metadata_program, std::vector<bool>{true},
        std::vector<bool>{true});
    const auto metadata_image =
        lower_surface_value_program(metadata_program, metadata_plan);
    require(metadata_image.valid && metadata_image.instructions.size() == 1u &&
                metadata_image.metadata.size() == 1u &&
                metadata_image.static_data == transform &&
                metadata_image.metadata[0u].static_u0 == 7u &&
                metadata_image.metadata[0u].static_f0 == 0.25f &&
                std::bit_cast<std::uint32_t>(
                    metadata_image.metadata[0u].static_f1) == 0x80000000u &&
                metadata_image.metadata[0u].static_table_begin == 0u &&
                metadata_image.metadata[0u].static_table_count == 16u,
            "compact value program lost immutable instruction metadata");

    const std::vector scene_programs{
        image, metadata_image, image, metadata_image};
    const auto scene_image =
        build_surface_value_scene_image(scene_programs);
    require(scene_image.valid && scene_image.programs.size() == 4u &&
                scene_image.instructions.size() == 8u &&
                scene_image.operands.size() == 10u &&
                scene_image.metadata.size() == 2u &&
                scene_image.static_data.size() == 32u,
            "scene value-program aggregation changed a stream extent");
    require(scene_image.programs[0u].instruction_begin == 0u &&
                scene_image.programs[0u].instruction_count == 3u &&
                scene_image.programs[1u].instruction_begin == 3u &&
                scene_image.programs[2u].instruction_begin == 4u &&
                scene_image.programs[3u].instruction_begin == 7u,
            "scene value-program descriptors do not preserve tag order");
    require(scene_image.instructions[4u].operand_begin == 5u &&
                scene_image.instructions[7u].metadata_index == 1u &&
                scene_image.metadata[1u].static_table_begin == 16u,
            "scene value-program aggregation did not rebase a global stream");
    require(scene_image.programs[0u].scalar_slots == 1u &&
                scene_image.programs[1u].vector_slots == 1u,
            "scene value-program descriptors lost typed slot bounds");

    const std::vector execution_inputs{
        SurfaceValueExecutionInput{.program = &program, .storage = &plan},
        SurfaceValueExecutionInput{.program = &metadata_program,
                                   .storage = &metadata_plan},
        SurfaceValueExecutionInput{.program = &program, .storage = &plan},
        SurfaceValueExecutionInput{.program = &metadata_program,
                                   .storage = &metadata_plan}};
    const auto executable_scene =
        build_surface_value_executable_scene(execution_inputs);
    require(executable_scene.valid &&
                executable_scene.variants.size() == 3u &&
                executable_scene.instruction_variants ==
                    std::vector<std::uint32_t>{0u, 1u, 0u, 2u,
                                               0u, 1u, 0u, 2u},
            "exact immutable variants were not interned in first-use order");
    require(executable_scene.variants[0u].operand_types ==
                    std::vector<SocketType>{SocketType::floating,
                                            SocketType::floating} &&
                executable_scene.variants[0u].instruction.operands ==
                    std::vector<ValueExpressionId>{ValueExpressionId{0u},
                                                   ValueExpressionId{1u}},
            "an immutable variant lost its typed normalized operands");

    const auto make_passthrough_program =
        [](std::uint32_t tag, SocketType type, SocketValue value) {
        return SurfaceProgram{
            tag,
            {ParameterDesc{
                .id = ParameterId{0u},
                .node = NodeId{tag + 1u},
                .socket = "Value",
                .type = type,
                .default_value = std::move(value),
                .source = ParameterSource::input}},
            {ValueInstruction{
                 .operation = ValueOperation::parameter,
                 .source_node = NodeId{tag + 1u},
                 .result_type = type,
                 .parameter = ParameterId{0u}},
             ValueInstruction{
                 .operation = ValueOperation::passthrough,
                 .source_node = NodeId{tag + 1u},
                 .result_type = type,
                 .operands =
                     make_value_operands<value_operand::unary>({
                         {value_operand::unary::input,
                          ValueExpressionId{0u}}})}},
            {},
            {}};
    };
    const auto float_passthrough = make_passthrough_program(
        30u, SocketType::floating, SocketValue::floating(0.25f));
    const auto boolean_passthrough = make_passthrough_program(
        31u, SocketType::boolean, SocketValue::boolean(true));
    const auto color_passthrough = make_passthrough_program(
        32u, SocketType::color,
        SocketValue::color({0.1f, 0.2f, 0.3f}));
    const auto normal_passthrough = make_passthrough_program(
        33u, SocketType::normal,
        SocketValue::normal({0.0f, 0.0f, 1.0f}));
    const auto uint_passthrough = make_passthrough_program(
        34u, SocketType::unsigned_integer,
        SocketValue::unsigned_integer(7u));
    const auto make_passthrough_plan = [](const SurfaceProgram &source) {
        return plan_surface_value_storage(
            source, std::vector<bool>{true, true},
            std::vector<bool>{false, true});
    };
    const auto float_passthrough_plan =
        make_passthrough_plan(float_passthrough);
    const auto boolean_passthrough_plan =
        make_passthrough_plan(boolean_passthrough);
    const auto color_passthrough_plan =
        make_passthrough_plan(color_passthrough);
    const auto normal_passthrough_plan =
        make_passthrough_plan(normal_passthrough);
    const auto uint_passthrough_plan =
        make_passthrough_plan(uint_passthrough);
    const std::vector passthrough_inputs{
        SurfaceValueExecutionInput{.program = &float_passthrough,
                                   .storage = &float_passthrough_plan},
        SurfaceValueExecutionInput{.program = &boolean_passthrough,
                                   .storage = &boolean_passthrough_plan},
        SurfaceValueExecutionInput{.program = &color_passthrough,
                                   .storage = &color_passthrough_plan},
        SurfaceValueExecutionInput{.program = &normal_passthrough,
                                   .storage = &normal_passthrough_plan},
        SurfaceValueExecutionInput{.program = &uint_passthrough,
                                   .storage = &uint_passthrough_plan}};
    const auto passthrough_scene =
        build_surface_value_executable_scene(passthrough_inputs);
    require(
        passthrough_scene.valid &&
            passthrough_scene.variants.size() == 3u &&
            passthrough_scene.instruction_variants ==
                std::vector<std::uint32_t>{0u, 0u, 1u, 1u, 2u} &&
            passthrough_scene.variants[0u].instruction.result_type ==
                SocketType::floating &&
            passthrough_scene.variants[0u].operand_types ==
                std::vector<SocketType>{SocketType::floating} &&
            passthrough_scene.variants[1u].instruction.result_type ==
                SocketType::vector &&
            passthrough_scene.variants[1u].operand_types ==
                std::vector<SocketType>{SocketType::vector} &&
            passthrough_scene.variants[2u].instruction.result_type ==
                SocketType::unsigned_integer &&
            passthrough_scene.variants[2u].operand_types ==
                std::vector<SocketType>{SocketType::unsigned_integer},
        "nominal socket spellings did not quotient to the three exact SVM "
        "execution banks");

    auto positive_zero_values = metadata_program.value_instructions();
    positive_zero_values.front().static_f1 = 0.0f;
    const SurfaceProgram positive_zero_program{
        4u, {}, std::move(positive_zero_values), {}, {}};
    const auto positive_zero_plan = plan_surface_value_storage(
        positive_zero_program, std::vector<bool>{true},
        std::vector<bool>{true});
    const std::vector zero_inputs{
        SurfaceValueExecutionInput{.program = &metadata_program,
                                   .storage = &metadata_plan},
        SurfaceValueExecutionInput{.program = &positive_zero_program,
                                   .storage = &positive_zero_plan}};
    const auto signed_zero_scene =
        build_surface_value_executable_scene(zero_inputs);
    require(signed_zero_scene.valid &&
                signed_zero_scene.variants.size() == 2u,
            "exact immutable-variant interning merged signed zero");

    auto translated_transform = transform;
    translated_transform[12u] = 3.5f;
    translated_transform[13u] = -2.25f;
    auto translated_values = metadata_program.value_instructions();
    translated_values.front().static_table = translated_transform;
    const SurfaceProgram translated_program{
        5u, {}, std::move(translated_values), {}, {}};
    const auto translated_plan = plan_surface_value_storage(
        translated_program, std::vector<bool>{true},
        std::vector<bool>{true});
    const std::vector transform_inputs{
        SurfaceValueExecutionInput{.program = &metadata_program,
                                   .storage = &metadata_plan},
        SurfaceValueExecutionInput{.program = &translated_program,
                                   .storage = &translated_plan}};
    const auto transform_scene =
        build_surface_value_executable_scene(transform_inputs);
    require(transform_scene.valid &&
                transform_scene.variants.size() == 1u &&
                transform_scene.instruction_variants ==
                    std::vector<std::uint32_t>{0u, 0u},
            "equal-shape static tables did not share one semantic evaluator");
    require(transform_scene.values.metadata.size() == 2u &&
                transform_scene.values.metadata[0u].static_table_begin == 0u &&
                transform_scene.values.metadata[0u].static_table_count == 16u &&
                transform_scene.values.metadata[1u].static_table_begin == 16u &&
                transform_scene.values.metadata[1u].static_table_count == 16u &&
                transform_scene.values.static_data ==
                    [&] {
                        auto data = transform;
                        data.insert(data.end(), translated_transform.begin(),
                                    translated_transform.end());
                        return data;
                    }(),
            "semantic interning lost distinct static-table bytecode payloads");
    require(
        transform_scene.variants[0u].instruction.static_table.size() == 16u &&
            std::all_of(
                transform_scene.variants[0u].instruction.static_table.begin(),
                transform_scene.variants[0u].instruction.static_table.end(),
                [](float value) noexcept { return value == 0.0f; }),
        "a semantic evaluator retained an authored static-table payload");

    const auto make_noise_program = [&](std::uint32_t tag,
                                        std::uint64_t dimensions,
                                        std::uint64_t type,
                                        bool normalize,
                                        bool color) {
        std::vector<ParameterDesc> parameters;
        std::vector<ValueInstruction> noise_values;
        parameters.reserve(8u);
        noise_values.reserve(10u);
        noise_values.emplace_back(ValueInstruction{
            .operation = ValueOperation::surface_position,
            .result_type = SocketType::point});
        for (auto index = std::uint32_t{0u}; index < 8u; ++index) {
            parameters.emplace_back(make_parameter(index));
            noise_values.emplace_back(make_parameter_value(index));
        }
        noise_values.emplace_back(ValueInstruction{
            .operation = color ? ValueOperation::noise_color
                               : ValueOperation::noise_factor,
            .result_type = color ? SocketType::color : SocketType::floating,
            .operands = make_value_operands<value_operand::noise>({
                {value_operand::noise::vector, ValueExpressionId{0u}},
                {value_operand::noise::scale, ValueExpressionId{1u}},
                {value_operand::noise::detail, ValueExpressionId{2u}},
                {value_operand::noise::roughness, ValueExpressionId{3u}},
                {value_operand::noise::lacunarity, ValueExpressionId{4u}},
                {value_operand::noise::distortion, ValueExpressionId{5u}},
                {value_operand::noise::w, ValueExpressionId{6u}},
                {value_operand::noise::offset, ValueExpressionId{7u}},
                {value_operand::noise::gain, ValueExpressionId{8u}}}),
            .static_u0 = dimensions,
            .static_u1 = (type << 8u) | (normalize ? 1u : 0u)});
        return SurfaceProgram{
            tag, std::move(parameters), std::move(noise_values), {}, {}};
    };
    const auto make_noise_plan = [](const SurfaceProgram &noise_program) {
        auto outputs = std::vector<bool>(10u, false);
        outputs.back() = true;
        return plan_surface_value_storage(
            noise_program, std::vector<bool>(10u, true), outputs);
    };
    constexpr auto noise_fbm = std::uint64_t{1u};
    constexpr auto noise_multifractal = std::uint64_t{0u};
    const auto normalized_noise =
        make_noise_program(20u, 3u, noise_fbm, true, false);
    const auto raw_noise =
        make_noise_program(21u, 3u, noise_fbm, false, false);
    const auto noise_2d =
        make_noise_program(22u, 2u, noise_fbm, false, false);
    const auto noise_multifractal_3d =
        make_noise_program(23u, 3u, noise_multifractal, false, false);
    const auto noise_color_3d =
        make_noise_program(24u, 3u, noise_fbm, true, true);
    const auto normalized_noise_plan = make_noise_plan(normalized_noise);
    const auto raw_noise_plan = make_noise_plan(raw_noise);
    const auto noise_2d_plan = make_noise_plan(noise_2d);
    const auto noise_multifractal_3d_plan =
        make_noise_plan(noise_multifractal_3d);
    const auto noise_color_3d_plan = make_noise_plan(noise_color_3d);
    const auto normalized_noise_image = lower_surface_value_program(
        normalized_noise, normalized_noise_plan);
    const auto raw_noise_image =
        lower_surface_value_program(raw_noise, raw_noise_plan);
    require(normalized_noise_image.valid && raw_noise_image.valid &&
                normalized_noise_image.instructions.size() == 2u &&
                raw_noise_image.instructions.size() == 2u &&
                surface_value_noise_normalize(
                    normalized_noise_image.instructions.back()) &&
                !surface_value_noise_normalize(
                    raw_noise_image.instructions.back()) &&
                (normalized_noise_image.metadata.back().static_u1 & 1u) != 0u &&
                (raw_noise_image.metadata.back().static_u1 & 1u) == 0u,
            "Noise Normalize was not preserved as validated bytecode data");

    const std::vector noise_inputs{
        SurfaceValueExecutionInput{.program = &normalized_noise,
                                   .storage = &normalized_noise_plan},
        SurfaceValueExecutionInput{.program = &raw_noise,
                                   .storage = &raw_noise_plan},
        SurfaceValueExecutionInput{.program = &noise_2d,
                                   .storage = &noise_2d_plan},
        SurfaceValueExecutionInput{.program = &noise_multifractal_3d,
                                   .storage = &noise_multifractal_3d_plan},
        SurfaceValueExecutionInput{.program = &noise_color_3d,
                                   .storage = &noise_color_3d_plan}};
    const auto noise_scene =
        build_surface_value_executable_scene(noise_inputs);
    require(noise_scene.valid && noise_scene.variants.size() == 5u &&
                noise_scene.instruction_variants ==
                    std::vector<std::uint32_t>{0u, 1u, 0u, 1u, 0u,
                                               2u, 0u, 3u, 0u, 4u},
            "Noise semantic interning merged more than the Normalize "
            "equivalence class");
    require(noise_scene.variants[1u].instruction.static_u0 == 3u &&
                noise_scene.variants[1u].instruction.static_u1 ==
                    (noise_fbm << 8u) &&
                noise_scene.variants[1u].svm_immediates ==
                    std::vector<std::uint16_t>{
                        0u,
                        surface_value_noise_normalize_immediate_bit} &&
                noise_scene.variants[1u].instruction.operation ==
                    ValueOperation::noise_factor,
            "the shared Noise evaluator retained a baked Normalize value");

    auto mismatched_noise_image = raw_noise_image;
    mismatched_noise_image.instructions.back().control |=
        surface_value_noise_normalize_immediate_bit
        << surface_value_svm_immediate_shift;
    const std::vector mismatched_noise_images{mismatched_noise_image};
    const auto mismatched_noise_scene =
        build_surface_value_scene_image(mismatched_noise_images);
    require(!mismatched_noise_scene.valid &&
                mismatched_noise_scene.diagnostic.find(
                    "immediate disagrees with immutable metadata") !=
                    std::string::npos,
            "serialized Noise accepted inconsistent Normalize data");

    auto foreign_normalize_image = metadata_image;
    foreign_normalize_image.instructions.front().control |=
        surface_value_noise_normalize_immediate_bit
        << surface_value_svm_immediate_shift;
    const std::vector foreign_normalize_images{foreign_normalize_image};
    const auto foreign_normalize_scene =
        build_surface_value_scene_image(foreign_normalize_images);
    require(!foreign_normalize_scene.valid &&
                foreign_normalize_scene.diagnostic.find(
                    "without an immediate contract") != std::string::npos,
            "serialized non-Noise opcode accepted a Noise-owned control bit");

    const auto make_mix_program = [](
                                      std::uint32_t tag,
                                      BlendOperation operation,
                                      bool clamp_factor,
                                      bool clamp_result) {
        std::vector<ParameterDesc> parameters{
            ParameterDesc{
                .id = ParameterId{0u},
                .node = NodeId{1u},
                .socket = "A",
                .type = SocketType::color,
                .default_value = SocketValue::color({0.0f, 0.0f, 0.0f}),
                .source = ParameterSource::input},
            ParameterDesc{
                .id = ParameterId{1u},
                .node = NodeId{2u},
                .socket = "B",
                .type = SocketType::color,
                .default_value = SocketValue::color({1.0f, 1.0f, 1.0f}),
                .source = ParameterSource::input},
            ParameterDesc{
                .id = ParameterId{2u},
                .node = NodeId{3u},
                .socket = "Factor",
                .type = SocketType::floating,
                .default_value = SocketValue::floating(0.5f),
                .source = ParameterSource::input}};
        std::vector<ValueInstruction> mix_values{
            ValueInstruction{
                .operation = ValueOperation::parameter,
                .result_type = SocketType::color,
                .parameter = ParameterId{0u}},
            ValueInstruction{
                .operation = ValueOperation::parameter,
                .result_type = SocketType::color,
                .parameter = ParameterId{1u}},
            ValueInstruction{
                .operation = ValueOperation::parameter,
                .result_type = SocketType::floating,
                .parameter = ParameterId{2u}},
            ValueInstruction{
                .operation = ValueOperation::mix,
                .result_type = SocketType::color,
                .operands = make_value_operands<value_operand::mix>({
                    {value_operand::mix::a, ValueExpressionId{0u}},
                    {value_operand::mix::b, ValueExpressionId{1u}},
                    {value_operand::mix::factor, ValueExpressionId{2u}}}),
                .static_u0 = static_cast<std::uint64_t>(operation),
                .static_u1 =
                    (clamp_factor ? 1u : 0u) |
                    (clamp_result ? 2u : 0u)}};
        return SurfaceProgram{
            tag, std::move(parameters), std::move(mix_values), {}, {}};
    };
    const auto make_mix_plan = [](const SurfaceProgram &mix_program) {
        auto outputs = std::vector<bool>(4u, false);
        outputs.back() = true;
        return plan_surface_value_storage(
            mix_program, std::vector<bool>(4u, true), outputs);
    };
    const auto plain_mix = make_mix_program(
        30u, BlendOperation::mix, false, false);
    const auto clamped_multiply = make_mix_program(
        31u, BlendOperation::multiply, true, false);
    const auto clamped_overlay = make_mix_program(
        32u, BlendOperation::overlay, false, true);
    const auto clamped_value = make_mix_program(
        33u, BlendOperation::value, true, true);
    const auto plain_mix_plan = make_mix_plan(plain_mix);
    const auto clamped_multiply_plan = make_mix_plan(clamped_multiply);
    const auto clamped_overlay_plan = make_mix_plan(clamped_overlay);
    const auto clamped_value_plan = make_mix_plan(clamped_value);
    const auto plain_mix_image =
        lower_surface_value_program(plain_mix, plain_mix_plan);
    const auto clamped_multiply_image = lower_surface_value_program(
        clamped_multiply, clamped_multiply_plan);
    const auto clamped_overlay_image = lower_surface_value_program(
        clamped_overlay, clamped_overlay_plan);
    const auto clamped_value_image = lower_surface_value_program(
        clamped_value, clamped_value_plan);
    require(
        plain_mix_image.valid && clamped_multiply_image.valid &&
            clamped_overlay_image.valid && clamped_value_image.valid &&
            surface_value_svm_immediate(
                plain_mix_image.instructions.front()) == 0u &&
            surface_value_svm_immediate(
                clamped_multiply_image.instructions.front()) ==
                (static_cast<std::uint32_t>(BlendOperation::multiply) |
                 surface_value_mix_factor_clamp_bit) &&
            surface_value_svm_immediate(
                clamped_overlay_image.instructions.front()) ==
                (static_cast<std::uint32_t>(BlendOperation::overlay) |
                 surface_value_mix_result_clamp_bit) &&
            surface_value_svm_immediate(
                clamped_value_image.instructions.front()) ==
                (static_cast<std::uint32_t>(BlendOperation::value) |
                 surface_value_mix_factor_clamp_bit |
                 surface_value_mix_result_clamp_bit),
        "Mix properties were not preserved by the opcode-owned immediate");

    const std::vector mix_inputs{
        SurfaceValueExecutionInput{.program = &plain_mix,
                                   .storage = &plain_mix_plan},
        SurfaceValueExecutionInput{.program = &clamped_multiply,
                                   .storage = &clamped_multiply_plan},
        SurfaceValueExecutionInput{.program = &clamped_overlay,
                                   .storage = &clamped_overlay_plan},
        SurfaceValueExecutionInput{.program = &clamped_value,
                                   .storage = &clamped_value_plan}};
    const auto mix_scene =
        build_surface_value_executable_scene(mix_inputs);
    const std::vector<std::uint16_t> expected_mix_immediates{
        0u,
        static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(BlendOperation::multiply) |
            surface_value_mix_factor_clamp_bit),
        static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(BlendOperation::overlay) |
            surface_value_mix_result_clamp_bit),
        static_cast<std::uint16_t>(
            static_cast<std::uint32_t>(BlendOperation::value) |
            surface_value_mix_factor_clamp_bit |
            surface_value_mix_result_clamp_bit)};
    require(
        mix_scene.valid && mix_scene.variants.size() == 1u &&
            mix_scene.instruction_variants ==
                std::vector<std::uint32_t>{0u, 0u, 0u, 0u} &&
            mix_scene.variants.front().instruction.static_u0 == 0u &&
            mix_scene.variants.front().instruction.static_u1 == 0u &&
            mix_scene.variants.front().svm_immediates ==
                expected_mix_immediates,
        "Mix SVM interning did not form the exact immediate quotient");

    auto mismatched_mix_image = plain_mix_image;
    mismatched_mix_image.instructions.front().control |=
        surface_value_mix_factor_clamp_bit
        << surface_value_svm_immediate_shift;
    const auto mismatched_mix_scene = build_surface_value_scene_image(
        std::vector{mismatched_mix_image});
    require(
        !mismatched_mix_scene.valid &&
            mismatched_mix_scene.diagnostic.find(
                "immediate disagrees with immutable metadata") !=
                std::string::npos,
        "serialized Mix accepted an immediate/metadata disagreement");

    const auto invalid_mix = make_mix_program(
        34u,
        static_cast<BlendOperation>(
            static_cast<std::uint32_t>(BlendOperation::value) + 1u),
        false,
        false);
    const auto invalid_mix_plan = make_mix_plan(invalid_mix);
    const auto invalid_mix_image =
        lower_surface_value_program(invalid_mix, invalid_mix_plan);
    require(
        !invalid_mix_image.valid &&
            invalid_mix_image.diagnostic.find("immediate contract") !=
                std::string::npos,
        "Mix lowering truncated an unrepresentable operation");

  const auto make_mapping_program =
      [](std::uint32_t tag, MappingVectorType type, std::uint64_t axes) {
        std::vector<ParameterDesc> parameters;
        std::vector<ValueInstruction> mapping_values;
        parameters.reserve(value_operand::mapping::count);
        mapping_values.reserve(value_operand::mapping::count + 1u);
        for (auto index = std::uint32_t{0u};
             index < value_operand::mapping::count; ++index) {
          parameters.emplace_back(ParameterDesc{
              .id = ParameterId{index},
              .node = NodeId{index + 1u},
              .socket = "Vector" + std::to_string(index),
              .type = SocketType::vector,
              .default_value =
                  SocketValue::vector(index == value_operand::mapping::scale
                                          ? psycles::Vec3f{1.0f, 1.0f, 1.0f}
                                          : psycles::Vec3f{}),
              .source = ParameterSource::input});
          mapping_values.emplace_back(
              ValueInstruction{.operation = ValueOperation::parameter,
                               .result_type = SocketType::vector,
                               .parameter = ParameterId{index}});
        }
        mapping_values.emplace_back(ValueInstruction{
            .operation = ValueOperation::mapping,
            .result_type = SocketType::vector,
            .operands = make_value_operands<value_operand::mapping>(
                {{value_operand::mapping::vector, ValueExpressionId{0u}},
                 {value_operand::mapping::location, ValueExpressionId{1u}},
                 {value_operand::mapping::rotation, ValueExpressionId{2u}},
                 {value_operand::mapping::scale, ValueExpressionId{3u}}}),
            .static_u0 = static_cast<std::uint64_t>(type),
            .static_u1 = axes});
        return SurfaceProgram{
            tag, std::move(parameters), std::move(mapping_values), {}, {}};
      };
  const auto make_mapping_plan = [](const SurfaceProgram &mapping_program) {
    const auto count = mapping_program.value_instructions().size();
    auto outputs = std::vector<bool>(count, false);
    outputs.back() = true;
    return plan_surface_value_storage(mapping_program,
                                      std::vector<bool>(count, true), outputs);
  };
  constexpr auto remap_yzx = std::uint64_t{0x39u};
  constexpr auto remap_zxy = std::uint64_t{0x1eu};
  const auto point_mapping =
      make_mapping_program(40u, MappingVectorType::point, 0u);
  const auto texture_mapping =
      make_mapping_program(41u, MappingVectorType::texture, remap_yzx);
  const auto normal_mapping =
      make_mapping_program(42u, MappingVectorType::normal, remap_zxy);
  const auto point_mapping_plan = make_mapping_plan(point_mapping);
  const auto texture_mapping_plan = make_mapping_plan(texture_mapping);
  const auto normal_mapping_plan = make_mapping_plan(normal_mapping);
  const auto point_mapping_image =
      lower_surface_value_program(point_mapping, point_mapping_plan);
  const auto texture_mapping_image =
      lower_surface_value_program(texture_mapping, texture_mapping_plan);
  const auto normal_mapping_image =
      lower_surface_value_program(normal_mapping, normal_mapping_plan);
  const auto texture_mapping_immediate = static_cast<std::uint16_t>(
      static_cast<std::uint32_t>(MappingVectorType::texture) |
      (remap_yzx << surface_value_mapping_axes_shift));
  const auto normal_mapping_immediate = static_cast<std::uint16_t>(
      static_cast<std::uint32_t>(MappingVectorType::normal) |
      (remap_zxy << surface_value_mapping_axes_shift));
  require(point_mapping_image.valid && texture_mapping_image.valid &&
              normal_mapping_image.valid &&
              surface_value_svm_immediate(
                  point_mapping_image.instructions.front()) == 0u &&
              surface_value_svm_immediate(
                  texture_mapping_image.instructions.front()) ==
                  texture_mapping_immediate &&
              surface_value_svm_immediate(
                  normal_mapping_image.instructions.front()) ==
                  normal_mapping_immediate,
          "Mapping type/axis semantics were not preserved by bytecode");
  const std::vector mapping_inputs{
      SurfaceValueExecutionInput{.program = &point_mapping,
                                 .storage = &point_mapping_plan},
      SurfaceValueExecutionInput{.program = &texture_mapping,
                                 .storage = &texture_mapping_plan},
      SurfaceValueExecutionInput{.program = &normal_mapping,
                                 .storage = &normal_mapping_plan}};
  const auto mapping_scene =
      build_surface_value_executable_scene(mapping_inputs);
  require(mapping_scene.valid && mapping_scene.variants.size() == 1u &&
              mapping_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u, 0u} &&
              mapping_scene.variants.front().instruction.static_u0 == 0u &&
              mapping_scene.variants.front().instruction.static_u1 == 0u &&
              mapping_scene.variants.front().svm_immediates ==
                  std::vector<std::uint16_t>{0u, normal_mapping_immediate,
                                             texture_mapping_immediate},
          "Mapping SVM interning did not form the exact immediate quotient");

  auto mismatched_mapping_image = point_mapping_image;
  mismatched_mapping_image.instructions.front().control |=
      1u << surface_value_svm_immediate_shift;
  const auto mismatched_mapping_scene =
      build_surface_value_scene_image(std::vector{mismatched_mapping_image});
  require(!mismatched_mapping_scene.valid &&
              mismatched_mapping_scene.diagnostic.find(
                  "immediate disagrees with immutable metadata") !=
                  std::string::npos,
          "serialized Mapping accepted an immediate/metadata disagreement");
  const auto invalid_mapping_type =
      make_mapping_program(43u, static_cast<MappingVectorType>(4u), 0u);
  const auto invalid_mapping_axes =
      make_mapping_program(44u, MappingVectorType::point, 0x40u);
  require(
      !lower_surface_value_program(invalid_mapping_type,
                                   make_mapping_plan(invalid_mapping_type))
              .valid &&
          !lower_surface_value_program(invalid_mapping_axes,
                                       make_mapping_plan(invalid_mapping_axes))
               .valid,
      "Mapping lowering truncated an unrepresentable immediate");

  const auto make_image_program = [](std::uint32_t tag,
                                     ValueOperation operation,
                                     std::uint64_t configuration) {
    const auto environment = operation == ValueOperation::environment_color ||
                             operation == ValueOperation::environment_alpha;
    const auto color = operation == ValueOperation::image_color ||
                       operation == ValueOperation::environment_color;
    std::vector<ParameterDesc> parameters{
        ParameterDesc{.id = ParameterId{0u},
                      .node = NodeId{1u},
                      .socket = "Vector",
                      .type = SocketType::vector,
                      .default_value = SocketValue::vector({0.0f, 0.0f, 0.0f}),
                      .source = ParameterSource::input},
        ParameterDesc{.id = ParameterId{1u},
                      .node = NodeId{2u},
                      .socket = "Image",
                      .type = SocketType::unsigned_integer,
                      .default_value = SocketValue::unsigned_integer(0u),
                      .source = ParameterSource::input}};
    std::vector<ValueInstruction> image_values{
        ValueInstruction{.operation = ValueOperation::parameter,
                         .result_type = SocketType::vector,
                         .parameter = ParameterId{0u}},
        ValueInstruction{.operation = ValueOperation::parameter,
                         .result_type = SocketType::unsigned_integer,
                         .parameter = ParameterId{1u}}};
    std::vector<ValueExpressionId> operands;
    if (environment) {
      operands = make_value_operands<value_operand::environment_texture>(
          {{value_operand::environment_texture::vector, ValueExpressionId{0u}},
           {value_operand::environment_texture::image, ValueExpressionId{1u}}});
    } else {
      parameters.emplace_back(
          ParameterDesc{.id = ParameterId{2u},
                        .node = NodeId{3u},
                        .socket = "ProjectionBlend",
                        .type = SocketType::floating,
                        .default_value = SocketValue::floating(0.0f),
                        .source = ParameterSource::input});
      image_values.emplace_back(
          ValueInstruction{.operation = ValueOperation::parameter,
                           .result_type = SocketType::floating,
                           .parameter = ParameterId{2u}});
      operands = make_value_operands<value_operand::image_texture>(
          {{value_operand::image_texture::vector, ValueExpressionId{0u}},
           {value_operand::image_texture::image, ValueExpressionId{1u}},
           {value_operand::image_texture::projection_blend,
            ValueExpressionId{2u}}});
    }
    image_values.emplace_back(ValueInstruction{
        .operation = operation,
        .result_type = color ? SocketType::color : SocketType::floating,
        .operands = std::move(operands),
        .static_u1 = configuration});
    return SurfaceProgram{
        tag, std::move(parameters), std::move(image_values), {}, {}};
  };
  const auto make_image_plan = [](const SurfaceProgram &image_program) {
    const auto count = image_program.value_instructions().size();
    auto outputs = std::vector<bool>(count, false);
    outputs.back() = true;
    return plan_surface_value_storage(image_program,
                                      std::vector<bool>(count, true), outputs);
  };
  constexpr auto linear_srgb = surface_value_image_srgb_bit |
                               (1u << surface_value_image_interpolation_shift);
  constexpr auto cubic_box_extend =
      2u | surface_value_image_unassociate_alpha_bit |
      (2u << surface_value_image_interpolation_shift) |
      (1u << surface_value_image_projection_shift);
  constexpr auto nearest_sphere_clip =
      1u | (2u << surface_value_image_projection_shift);
  constexpr auto linear_mirrorball =
      (1u << surface_value_image_interpolation_shift) |
      (1u << surface_value_image_projection_shift);
  const auto image_flat =
      make_image_program(50u, ValueOperation::image_color, linear_srgb);
  const auto image_box =
      make_image_program(51u, ValueOperation::image_color, cubic_box_extend);
  const auto image_sphere =
      make_image_program(52u, ValueOperation::image_color, nearest_sphere_clip);
  const auto alpha_flat =
      make_image_program(53u, ValueOperation::image_alpha, linear_srgb);
  const auto alpha_box =
      make_image_program(54u, ValueOperation::image_alpha, cubic_box_extend);
  const auto environment_mirrorball = make_image_program(
      55u, ValueOperation::environment_color, linear_mirrorball);
  const auto image_flat_plan = make_image_plan(image_flat);
  const auto image_box_plan = make_image_plan(image_box);
  const auto image_sphere_plan = make_image_plan(image_sphere);
  const auto alpha_flat_plan = make_image_plan(alpha_flat);
  const auto alpha_box_plan = make_image_plan(alpha_box);
  const auto environment_mirrorball_plan =
      make_image_plan(environment_mirrorball);
  const std::vector image_inputs{
      SurfaceValueExecutionInput{.program = &image_flat,
                                 .storage = &image_flat_plan},
      SurfaceValueExecutionInput{.program = &image_box,
                                 .storage = &image_box_plan},
      SurfaceValueExecutionInput{.program = &image_sphere,
                                 .storage = &image_sphere_plan},
      SurfaceValueExecutionInput{.program = &alpha_flat,
                                 .storage = &alpha_flat_plan},
      SurfaceValueExecutionInput{.program = &alpha_box,
                                 .storage = &alpha_box_plan},
      SurfaceValueExecutionInput{.program = &environment_mirrorball,
                                 .storage = &environment_mirrorball_plan}};
  const auto image_scene = build_surface_value_executable_scene(image_inputs);
  require(image_scene.valid && image_scene.variants.size() == 5u &&
              image_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 1u, 0u, 2u, 3u, 4u} &&
              image_scene.variants[0u].instruction.static_u1 == 0u &&
              image_scene.variants[0u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(linear_srgb),
                      static_cast<std::uint16_t>(nearest_sphere_clip)} &&
              image_scene.variants[1u].instruction.static_u1 ==
                  (1u << surface_value_image_projection_shift) &&
              image_scene.variants[1u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(cubic_box_extend)} &&
              image_scene.variants[2u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(linear_srgb)} &&
              image_scene.variants[3u].instruction.static_u1 ==
                  (1u << surface_value_image_projection_shift) &&
              image_scene.variants[3u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(cubic_box_extend)} &&
              image_scene.variants[4u].svm_immediates ==
                  std::vector<std::uint16_t>{
                      static_cast<std::uint16_t>(linear_mirrorball)},
          "Image SVM interning merged outside the exact evaluator quotient");

  auto mismatched_image =
      lower_surface_value_program(image_flat, image_flat_plan);
  require(mismatched_image.valid,
          "valid Image configuration failed bytecode lowering");
  mismatched_image.instructions.front().control ^=
      1u << surface_value_svm_immediate_shift;
  const auto mismatched_image_scene =
      build_surface_value_scene_image(std::vector{mismatched_image});
  require(!mismatched_image_scene.valid &&
              mismatched_image_scene.diagnostic.find(
                  "immediate disagrees with immutable metadata") !=
                  std::string::npos,
          "serialized Image accepted an immediate/metadata disagreement");
  const auto invalid_image_extension =
      make_image_program(56u, ValueOperation::image_color, 4u);
  const auto invalid_image_reserved =
      make_image_program(57u, ValueOperation::image_color, 1u << 14u);
  const auto invalid_environment_projection =
      make_image_program(58u, ValueOperation::environment_color,
                         2u << surface_value_image_projection_shift);
  require(
      !lower_surface_value_program(invalid_image_extension,
                                   make_image_plan(invalid_image_extension))
              .valid &&
          !lower_surface_value_program(invalid_image_reserved,
                                       make_image_plan(invalid_image_reserved))
               .valid &&
          !lower_surface_value_program(
               invalid_environment_projection,
               make_image_plan(invalid_environment_projection))
               .valid,
      "Image lowering accepted a configuration outside its exact ABI");

  const auto make_table_program = [&](std::uint32_t table_parameter,
                                      std::uint64_t interpolation = 0u) {
    std::vector<ValueInstruction> table_values;
    table_values.emplace_back(make_parameter_value(0u));
    table_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::color_ramp,
        .result_type = SocketType::color,
        .parameter = ParameterId{table_parameter},
        .operands = make_value_operands<value_operand::color_ramp>(
            {{value_operand::color_ramp::factor, ValueExpressionId{0u}}}),
        .static_u0 = interpolation});
    return SurfaceProgram{10u + table_parameter,
                          {make_parameter(0u)},
                          std::move(table_values),
                          {},
                          {}};
  };
  const auto table_program_a = make_table_program(3u);
  const auto table_program_b = make_table_program(7u, 3u);
  const auto table_plan_a =
      plan_surface_value_storage(table_program_a, std::vector<bool>(2u, true),
                                 std::vector<bool>{false, true});
  const auto table_plan_b =
      plan_surface_value_storage(table_program_b, std::vector<bool>(2u, true),
                                 std::vector<bool>{false, true});
  const std::vector table_inputs{
      SurfaceValueExecutionInput{.program = &table_program_a,
                                 .storage = &table_plan_a},
      SurfaceValueExecutionInput{.program = &table_program_b,
                                 .storage = &table_plan_b}};
  const auto table_scene = build_surface_value_executable_scene(table_inputs);
  require(table_scene.valid && table_scene.variants.size() == 1u &&
              table_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u} &&
              table_scene.variants.front().instruction.static_u0 == 0u &&
              table_scene.variants.front().svm_immediates ==
                  std::vector<std::uint16_t>{0u, 3u} &&
              table_scene.values.metadata.size() == 2u &&
              table_scene.values.metadata[0u].parameter == 3u &&
              table_scene.values.metadata[1u].parameter == 7u,
          "Color Ramp record data changed the evaluator body or lost its "
          "late-bound shader-table address");

  const auto make_gradient_program = [](std::uint32_t tag,
                                        std::uint64_t mode) {
    return SurfaceProgram{
        tag,
        {ParameterDesc{
            .id = ParameterId{0u},
            .node = NodeId{tag + 1u},
            .socket = "Vector",
            .type = SocketType::vector,
            .default_value = SocketValue::vector({0.0f, 0.0f, 0.0f}),
            .source = ParameterSource::input}},
        {ValueInstruction{.operation = ValueOperation::parameter,
                          .result_type = SocketType::vector,
                          .parameter = ParameterId{0u}},
         ValueInstruction{
             .operation = ValueOperation::gradient,
             .result_type = SocketType::floating,
             .operands = make_value_operands<value_operand::gradient>({
                 {value_operand::gradient::vector,
                  ValueExpressionId{0u}}}),
             .static_u0 = mode}},
        {},
        {}};
  };
  const auto gradient_linear = make_gradient_program(70u, 0u);
  const auto gradient_radial = make_gradient_program(71u, 4u);
  const auto gradient_spherical = make_gradient_program(72u, 6u);
  const auto make_gradient_plan = [](const SurfaceProgram &source) {
    return plan_surface_value_storage(
        source, std::vector<bool>{true, true},
        std::vector<bool>{false, true});
  };
  const auto gradient_linear_plan = make_gradient_plan(gradient_linear);
  const auto gradient_radial_plan = make_gradient_plan(gradient_radial);
  const auto gradient_spherical_plan = make_gradient_plan(gradient_spherical);
  const std::vector gradient_inputs{
      SurfaceValueExecutionInput{.program = &gradient_linear,
                                 .storage = &gradient_linear_plan},
      SurfaceValueExecutionInput{.program = &gradient_radial,
                                 .storage = &gradient_radial_plan},
      SurfaceValueExecutionInput{.program = &gradient_spherical,
                                 .storage = &gradient_spherical_plan}};
  const auto gradient_scene =
      build_surface_value_executable_scene(gradient_inputs);
  require(gradient_scene.valid && gradient_scene.variants.size() == 1u &&
              gradient_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u, 0u} &&
              gradient_scene.variants.front().instruction.static_u0 == 0u &&
              gradient_scene.variants.front().svm_immediates ==
                  std::vector<std::uint16_t>{0u, 4u, 6u},
          "Gradient modes did not quotient to one typed SVM handler");
  const auto invalid_gradient = make_gradient_program(73u, 7u);
  require(!lower_surface_value_program(
               invalid_gradient, make_gradient_plan(invalid_gradient))
               .valid,
          "Gradient lowering truncated an invalid record mode");

  const auto make_optional_normal_program = [](
                                                std::uint32_t tag,
                                                ValueOperation operation,
                                                std::uint64_t normal_linked) {
    return SurfaceProgram{
        tag,
        {ParameterDesc{.id = ParameterId{0u},
                       .node = NodeId{tag + 1u},
                       .socket = "Value",
                       .type = SocketType::floating,
                       .default_value = SocketValue::floating(0.5f),
                       .source = ParameterSource::input},
         ParameterDesc{
             .id = ParameterId{1u},
             .node = NodeId{tag + 1u},
             .socket = "Normal",
             .type = SocketType::normal,
             .default_value = SocketValue::normal({0.0f, 0.0f, 1.0f}),
             .source = ParameterSource::input}},
        {ValueInstruction{.operation = ValueOperation::parameter,
                          .result_type = SocketType::floating,
                          .parameter = ParameterId{0u}},
         ValueInstruction{.operation = ValueOperation::parameter,
                          .result_type = SocketType::normal,
                          .parameter = ParameterId{1u}},
         ValueInstruction{
             .operation = operation,
             .result_type = SocketType::floating,
             .operands = std::vector<ValueExpressionId>{
                 ValueExpressionId{0u}, ValueExpressionId{1u}},
             .static_u0 = normal_linked}},
        {},
        {}};
  };
  const auto fresnel_default = make_optional_normal_program(
      74u, ValueOperation::fresnel, 0u);
  const auto fresnel_linked = make_optional_normal_program(
      75u, ValueOperation::fresnel, 1u);
  const auto make_optional_normal_plan = [](const SurfaceProgram &source) {
    return plan_surface_value_storage(
        source, std::vector<bool>{true, true, true},
        std::vector<bool>{false, false, true});
  };
  const auto fresnel_default_plan =
      make_optional_normal_plan(fresnel_default);
  const auto fresnel_linked_plan =
      make_optional_normal_plan(fresnel_linked);
  const std::vector fresnel_inputs{
      SurfaceValueExecutionInput{.program = &fresnel_default,
                                 .storage = &fresnel_default_plan},
      SurfaceValueExecutionInput{.program = &fresnel_linked,
                                 .storage = &fresnel_linked_plan}};
  const auto fresnel_scene =
      build_surface_value_executable_scene(fresnel_inputs);
  require(fresnel_scene.valid && fresnel_scene.variants.size() == 1u &&
              fresnel_scene.instruction_variants ==
                  std::vector<std::uint32_t>{0u, 0u} &&
              fresnel_scene.variants.front().instruction.static_u0 == 0u &&
              fresnel_scene.variants.front().svm_immediates ==
                  std::vector<std::uint16_t>{0u, 1u},
          "optional Fresnel normals did not remain typed SVM record data");
  const auto invalid_fresnel = make_optional_normal_program(
      76u, ValueOperation::fresnel, 2u);
  require(!lower_surface_value_program(
               invalid_fresnel,
               make_optional_normal_plan(invalid_fresnel))
               .valid,
          "Fresnel lowering accepted an invalid linked-normal flag");

    std::vector<ParameterDesc> bump_parameters;
    for (auto index = 0u; index < 4u; ++index) {
        bump_parameters.emplace_back(make_parameter(index));
    }
    bump_parameters.emplace_back(ParameterDesc{
        .id = ParameterId{4u},
        .node = NodeId{5u},
        .socket = "Normal",
        .type = SocketType::normal,
        .default_value = SocketValue::normal({0.0f, 0.0f, 1.0f}),
        .source = ParameterSource::input});
    std::vector<ValueInstruction> bump_values;
    for (auto index = 0u; index < 4u; ++index) {
        bump_values.emplace_back(make_parameter_value(index));
    }
    bump_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::parameter,
        .source_node = NodeId{5u},
        .result_type = SocketType::normal,
        .parameter = ParameterId{4u}});
    const auto make_bump_instruction =
        [](ValueExpressionId height) {
            return ValueInstruction{
                .operation = ValueOperation::bump,
                .result_type = SocketType::normal,
                .operands = make_value_operands<value_operand::bump>({
                    {value_operand::bump::height, height},
                    {value_operand::bump::strength,
                     ValueExpressionId{1u}},
                    {value_operand::bump::distance,
                     ValueExpressionId{2u}},
                    {value_operand::bump::filter_width,
                     ValueExpressionId{3u}},
                    {value_operand::bump::normal,
                     ValueExpressionId{4u}}}),
                .static_u0 = 2u};
        };
    bump_values.emplace_back(
        make_bump_instruction(ValueExpressionId{0u}));
    const SurfaceProgram bump_program{
        20u, bump_parameters, bump_values, {}, {}};
    const auto bump_plan = plan_surface_value_storage(
        bump_program, std::vector<bool>(6u, true),
        std::vector<bool>{false, false, false, false, false, true});
    const auto bump_scene = build_surface_value_bump_executable_scene(
        std::vector{SurfaceValueExecutionInput{
            .program = &bump_program, .storage = &bump_plan}});
    require(bump_scene.valid && bump_scene.root_program_count == 1u &&
                bump_scene.maximum_bump_depth == 1u &&
                bump_scene.executable.values.programs.size() == 2u &&
                bump_scene.executable.values.instructions.size() == 1u &&
                bump_scene.bump_height_programs ==
                    std::vector<std::uint32_t>{1u} &&
                SurfaceValueAddress{
                    bump_scene.program_outputs[1u]}.parameter() &&
                SurfaceValueAddress{
                    bump_scene.program_outputs[1u]}.index() == 0u,
            "Bump height did not lower to an exact typed subprogram");

    auto nested_values = bump_values;
    nested_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::vector_to_scalar,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::unary>({
            {value_operand::unary::input, ValueExpressionId{5u}}})});
    nested_values.emplace_back(
        make_bump_instruction(ValueExpressionId{6u}));
    const SurfaceProgram nested_bump_program{
        21u, bump_parameters, std::move(nested_values), {}, {}};
    const auto nested_bump_plan = plan_surface_value_storage(
        nested_bump_program, std::vector<bool>(8u, true),
        std::vector<bool>{false, false, false, false, false, false,
                          false, true});
    const auto nested_bump_scene =
        build_surface_value_bump_executable_scene(
            std::vector{SurfaceValueExecutionInput{
                .program = &nested_bump_program,
                .storage = &nested_bump_plan}});
    const auto invalid_address = SurfaceValueAddress::invalid_value;
    require(nested_bump_scene.valid &&
                nested_bump_scene.maximum_bump_depth == 2u &&
                nested_bump_scene.executable.values.programs.size() == 3u &&
                nested_bump_scene.executable.values.instructions.size() == 5u &&
                nested_bump_scene.bump_height_programs ==
                    std::vector<std::uint32_t>{
                        1u, invalid_address, 2u, 1u, invalid_address} &&
                SurfaceValueAddress{
                    nested_bump_scene.program_outputs[1u]}.parameter() &&
                !SurfaceValueAddress{
                    nested_bump_scene.program_outputs[2u]}.parameter() &&
                SurfaceValueAddress{
                    nested_bump_scene.program_outputs[2u]}.bank() ==
                    SurfaceValueBank::scalar,
            "nested Bump did not lower to a shared finite-strata evaluator "
            "DAG");

    auto malformed_image = image;
    malformed_image.instructions.front().control |= 1u << 31u;
    const auto malformed_scene = build_surface_value_scene_image(
        std::vector{malformed_image, metadata_image});
    require(!malformed_scene.valid && malformed_scene.programs.empty() &&
                malformed_scene.instructions.empty() &&
                malformed_scene.diagnostic.find(
                    "without an immediate contract") !=
                    std::string::npos,
            "scene aggregation accepted or partially committed a malformed "
            "instruction stream");
    auto malformed_arity = image;
    malformed_arity.instructions.front().control ^=
        1u << surface_value_operand_count_shift;
    const auto malformed_arity_scene =
        build_surface_value_scene_image(
            std::vector{malformed_arity});
    require(!malformed_arity_scene.valid &&
                malformed_arity_scene.diagnostic.find("arity") !=
                    std::string::npos,
            "scene aggregation accepted an opcode/arity disagreement");

    std::vector<ValueInstruction> invalid_values;
    invalid_values.emplace_back(ValueInstruction{
        .operation = ValueOperation::add,
        .result_type = SocketType::floating,
        .operands = make_value_operands<value_operand::binary>({
            {value_operand::binary::a, ValueExpressionId{1u}},
            {value_operand::binary::b, ValueExpressionId{1u}}})});
    invalid_values.emplace_back(make_parameter_value(0u));
    const SurfaceProgram invalid_program{
        2u,
        {make_parameter(0u)},
        std::move(invalid_values),
        {},
        {}};
    const auto invalid_plan = plan_surface_value_storage(
        invalid_program, std::vector<bool>(2u, true),
        std::vector<bool>{true, false});
    require(!invalid_plan.valid &&
                invalid_plan.diagnostic.find("topological") !=
                    std::string::npos,
            "storage planning accepted a forward value dependency");
}

}// namespace

int main() {
    try {
        test_cycles_surface_bssrdf_metadata();
        test_surface_closure_plan();
        test_surface_value_storage_plan();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Surface-program metadata test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
