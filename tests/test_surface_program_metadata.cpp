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

    auto malformed_image = image;
    malformed_image.instructions.front().control |= 1u << 31u;
    const auto malformed_scene = build_surface_value_scene_image(
        std::vector{malformed_image, metadata_image});
    require(!malformed_scene.valid && malformed_scene.programs.empty() &&
                malformed_scene.instructions.empty() &&
                malformed_scene.diagnostic.find("control word") !=
                    std::string::npos,
            "scene aggregation accepted or partially committed a malformed "
            "instruction stream");

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
