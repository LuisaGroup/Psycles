#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_reachability.h>
#include <psycles/luisa/surface_closure_sampling.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::make_surface_point;
using psycles::test_support::ParameterShaderServices;

struct ReachabilityBasis {
    std::uint32_t bit{};
    SurfaceClosureReachability result{};
};

[[nodiscard]] constexpr std::uint32_t
operation_bit(ClosureOperation operation) noexcept {
    return std::uint32_t{1u} << static_cast<std::uint32_t>(operation);
}

[[nodiscard]] SurfaceClosureReachability
kinds(std::initializer_list<SurfaceClosureKind> values) noexcept {
    SurfaceClosureReachability result;
    for (const auto value : values) {
        result.kinds |= surface_closure_kind_bit(value);
    }
    return result;
}

[[nodiscard]] SurfaceClosureReachability
principled_lobe(SurfaceClosureLobe lobe) noexcept {
    return {.kinds = surface_closure_kind_bit(SurfaceClosureKind::principled),
            .principled_lobes = surface_closure_lobe_bit(lobe)};
}

[[nodiscard]] SurfaceClosureReachability
anisotropic_kind(SurfaceClosureKind kind) noexcept {
    return {.kinds = surface_closure_kind_bit(kind),
            .anisotropic_microfacet_kinds = surface_closure_kind_bit(kind)};
}

[[nodiscard]] SurfaceClosureReachability
thin_film_kind(SurfaceClosureKind kind) noexcept {
    return {.kinds = surface_closure_kind_bit(kind),
            .thin_film_kinds = surface_closure_kind_bit(kind)};
}

[[nodiscard]] SurfaceClosureReachability
thin_film_principled_lobe(SurfaceClosureLobe lobe) noexcept {
    return {.kinds = surface_closure_kind_bit(SurfaceClosureKind::principled),
            .principled_lobes = surface_closure_lobe_bit(lobe),
            .thin_film_principled_lobes =
                surface_closure_lobe_bit(lobe)};
}

[[nodiscard]] bool verify_reachability_lattice() noexcept {
    constexpr auto meet_lhs = SurfaceClosureReachability{
        .kinds = surface_closure_kind_bit(SurfaceClosureKind::diffuse) |
                 surface_closure_kind_bit(SurfaceClosureKind::principled),
        .principled_lobes =
            surface_closure_lobe_bit(SurfaceClosureLobe::sheen) |
            surface_closure_lobe_bit(SurfaceClosureLobe::coat) |
            surface_closure_lobe_bit(SurfaceClosureLobe::metallic),
        .anisotropic_microfacet_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::principled),
        .thin_film_principled_lobes =
            surface_closure_lobe_bit(SurfaceClosureLobe::metallic)};
    constexpr auto meet_rhs = SurfaceClosureReachability{
        .kinds = surface_closure_kind_bit(SurfaceClosureKind::principled) |
                 surface_closure_kind_bit(SurfaceClosureKind::glass),
        .principled_lobes =
            surface_closure_lobe_bit(SurfaceClosureLobe::coat) |
            surface_closure_lobe_bit(SurfaceClosureLobe::metallic),
        .anisotropic_microfacet_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::principled),
        .thin_film_principled_lobes =
            surface_closure_lobe_bit(SurfaceClosureLobe::metallic)};
    constexpr auto meet_expected = SurfaceClosureReachability{
        .kinds = surface_closure_kind_bit(SurfaceClosureKind::principled),
        .principled_lobes =
            surface_closure_lobe_bit(SurfaceClosureLobe::coat) |
            surface_closure_lobe_bit(SurfaceClosureLobe::metallic),
        .anisotropic_microfacet_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::principled),
        .thin_film_principled_lobes =
            surface_closure_lobe_bit(SurfaceClosureLobe::metallic)};
    static_assert((meet_lhs & meet_rhs) == meet_expected);
    static_assert((meet_rhs & meet_lhs) == meet_expected);
    static_assert((meet_lhs & meet_lhs) == meet_lhs);
    static_assert((meet_lhs & all_surface_closure_reachability) == meet_lhs);

    // The reduced product invariant forbids a lobe bit when Principled is not
    // reachable. Meet must restore that invariant even for a malformed input,
    // so later family specialization cannot observe a dangling lobe.
    constexpr auto malformed = SurfaceClosureReachability{
        .kinds = surface_closure_kind_bit(SurfaceClosureKind::diffuse),
        .principled_lobes = all_surface_closure_lobes,
        .anisotropic_microfacet_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::principled) |
            surface_closure_kind_bit(SurfaceClosureKind::glossy),
        .thin_film_kinds =
            surface_closure_kind_bit(SurfaceClosureKind::glass),
        .thin_film_principled_lobes = all_surface_closure_lobes};
    constexpr auto normalized =
        malformed & all_surface_closure_reachability;
    static_assert(normalized == SurfaceClosureReachability{
                                    .kinds = surface_closure_kind_bit(
                                        SurfaceClosureKind::diffuse),
                                    .principled_lobes = 0u,
                                    .anisotropic_microfacet_kinds = 0u,
                                    .thin_film_kinds = 0u,
                                    .thin_film_principled_lobes = 0u});
    static_assert((malformed | SurfaceClosureReachability{}) == normalized);

    const std::array operation_basis{
        ReachabilityBasis{operation_bit(ClosureOperation::null_closure), {}},
        ReachabilityBasis{operation_bit(ClosureOperation::diffuse),
                          kinds({SurfaceClosureKind::diffuse})},
        ReachabilityBasis{operation_bit(ClosureOperation::translucent),
                          kinds({SurfaceClosureKind::translucent})},
        ReachabilityBasis{operation_bit(ClosureOperation::principled), {}},
        ReachabilityBasis{operation_bit(ClosureOperation::glossy),
                          kinds({SurfaceClosureKind::glossy})},
        ReachabilityBasis{operation_bit(ClosureOperation::metallic_f82),
                          kinds({SurfaceClosureKind::metallic_f82})},
        ReachabilityBasis{
            operation_bit(ClosureOperation::metallic_conductor),
            kinds({SurfaceClosureKind::metallic_conductor})},
        ReachabilityBasis{
            operation_bit(ClosureOperation::sheen_microfiber),
            kinds({SurfaceClosureKind::sheen_microfiber})},
        ReachabilityBasis{
            operation_bit(ClosureOperation::sheen_ashikhmin),
            kinds({SurfaceClosureKind::sheen_ashikhmin})},
        ReachabilityBasis{
            operation_bit(ClosureOperation::hair_reflection),
            kinds({SurfaceClosureKind::hair_reflection})},
        ReachabilityBasis{
            operation_bit(ClosureOperation::hair_transmission),
            kinds({SurfaceClosureKind::hair_transmission})},
        ReachabilityBasis{operation_bit(ClosureOperation::glass),
                          kinds({SurfaceClosureKind::glass})},
        ReachabilityBasis{operation_bit(ClosureOperation::emission), {}},
        ReachabilityBasis{operation_bit(ClosureOperation::transparent),
                          kinds({SurfaceClosureKind::transparent})},
        ReachabilityBasis{
            operation_bit(ClosureOperation::subsurface),
            kinds({SurfaceClosureKind::bssrdf, SurfaceClosureKind::diffuse})},
        ReachabilityBasis{operation_bit(ClosureOperation::add), {}},
        ReachabilityBasis{operation_bit(ClosureOperation::mix), {}},
        ReachabilityBasis{operation_bit(ClosureOperation::refraction),
                          kinds({SurfaceClosureKind::refraction})}};
    static_assert(
        operation_basis.size() ==
        static_cast<std::size_t>(ClosureOperation::refraction) + 1u);
    const std::array feature_basis{
        ReachabilityBasis{
            principled_closure_feature_bit(PrincipledClosureFeature::alpha),
            kinds({SurfaceClosureKind::transparent})},
        ReachabilityBasis{
            principled_closure_feature_bit(PrincipledClosureFeature::sheen),
            principled_lobe(SurfaceClosureLobe::sheen)},
        ReachabilityBasis{
            principled_closure_feature_bit(PrincipledClosureFeature::coat),
            principled_lobe(SurfaceClosureLobe::coat)},
        ReachabilityBasis{
            principled_closure_feature_bit(PrincipledClosureFeature::metallic),
            principled_lobe(SurfaceClosureLobe::metallic)},
        ReachabilityBasis{principled_closure_feature_bit(
                              PrincipledClosureFeature::thick_transmission),
                          kinds({SurfaceClosureKind::glass})},
        ReachabilityBasis{principled_closure_feature_bit(
                              PrincipledClosureFeature::thin_transmission),
                          kinds({SurfaceClosureKind::glossy,
                                 SurfaceClosureKind::thin_glass_transmission,
                                 SurfaceClosureKind::transparent})},
        ReachabilityBasis{
            principled_closure_feature_bit(PrincipledClosureFeature::dielectric),
            principled_lobe(SurfaceClosureLobe::dielectric)},
        ReachabilityBasis{
            principled_closure_feature_bit(
                PrincipledClosureFeature::thick_subsurface),
            kinds({SurfaceClosureKind::bssrdf, SurfaceClosureKind::diffuse})},
        ReachabilityBasis{
            principled_closure_feature_bit(
                PrincipledClosureFeature::thin_subsurface),
            kinds({SurfaceClosureKind::diffuse, SurfaceClosureKind::translucent,
                   SurfaceClosureKind::rough_translucent})},
        ReachabilityBasis{
            principled_closure_feature_bit(PrincipledClosureFeature::diffuse),
            kinds({SurfaceClosureKind::diffuse})},
        ReachabilityBasis{
            principled_closure_feature_bit(PrincipledClosureFeature::emission),
            {}}};

    const auto make_subsets = [](const auto &basis) {
        const auto subset_count = std::size_t{1u} << basis.size();
        std::vector<std::uint32_t> masks(subset_count);
        std::vector<SurfaceClosureReachability> expected(subset_count);
        for (auto subset = std::size_t{0u}; subset < subset_count; ++subset) {
            for (auto bit = std::size_t{0u}; bit < basis.size(); ++bit) {
                if ((subset & (std::size_t{1u} << bit)) != 0u) {
                    masks[subset] |= basis[bit].bit;
                    expected[subset] |= basis[bit].result;
                }
            }
        }
        return std::pair{std::move(masks), std::move(expected)};
    };
    const auto [operation_masks, operation_expected] =
        make_subsets(operation_basis);
    const auto [feature_masks, feature_expected] = make_subsets(feature_basis);

    // Exhaust the complete recognized powerset product. This verifies the
    // transfer as a union homomorphism rather than sampling selected material
    // cases: every possible scene-wide union of known opcodes and feature bits
    // must equal the join of its singleton images.
    for (auto operation_subset = std::size_t{0u};
         operation_subset < operation_masks.size(); ++operation_subset) {
        for (auto feature_subset = std::size_t{0u};
             feature_subset < feature_masks.size(); ++feature_subset) {
            const auto expected = operation_expected[operation_subset] |
                                  feature_expected[feature_subset];
            const auto actual = reachable_surface_closures(
                operation_masks[operation_subset],
                feature_masks[feature_subset],
                0u,
                0u);
            if (actual != expected) {
                std::cerr << "closure reachability is not additive at op "
                          << operation_subset << ", feature " << feature_subset << '\n';
                return false;
            }
        }
    }

    constexpr auto unknown_bit = std::uint32_t{1u} << 31u;
    if (reachable_surface_closures(unknown_bit, 0u, 0u, 0u) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(0u, unknown_bit, 0u, 0u) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(0u, 0u, unknown_bit, 0u) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(0u, 0u, 0u, unknown_bit) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(0u, 0u, 0u, 0u, unknown_bit, 0u) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(0u, 0u, 0u, 0u, 0u, unknown_bit) !=
            all_surface_closure_reachability) {
        std::cerr << "unknown closure metadata did not map to lattice top\n";
        return false;
    }

    constexpr auto glossy_operation = operation_bit(ClosureOperation::glossy);
    constexpr auto metallic_f82_operation =
        operation_bit(ClosureOperation::metallic_f82);
    constexpr auto metallic_conductor_operation =
        operation_bit(ClosureOperation::metallic_conductor);
    constexpr auto principled_operation =
        operation_bit(ClosureOperation::principled);
    constexpr auto diffuse_operation = operation_bit(ClosureOperation::diffuse);
    constexpr auto metallic_feature =
        principled_closure_feature_bit(PrincipledClosureFeature::metallic);
    constexpr auto dielectric_feature =
        principled_closure_feature_bit(PrincipledClosureFeature::dielectric);
    constexpr auto diffuse_feature =
        principled_closure_feature_bit(PrincipledClosureFeature::diffuse);
    if (reachable_surface_closures(
            glossy_operation, 0u, glossy_operation, 0u) !=
            anisotropic_kind(SurfaceClosureKind::glossy) ||
        reachable_surface_closures(
            metallic_f82_operation, 0u, metallic_f82_operation, 0u) !=
            anisotropic_kind(SurfaceClosureKind::metallic_f82) ||
        reachable_surface_closures(
            metallic_conductor_operation,
            0u,
            metallic_conductor_operation,
            0u) !=
            anisotropic_kind(SurfaceClosureKind::metallic_conductor) ||
        reachable_surface_closures(principled_operation,
                                   metallic_feature,
                                   principled_operation,
                                   metallic_feature) !=
            (principled_lobe(SurfaceClosureLobe::metallic) |
             anisotropic_kind(SurfaceClosureKind::principled)) ||
        reachable_surface_closures(principled_operation,
                                   dielectric_feature,
                                   principled_operation,
                                   dielectric_feature) !=
            (principled_lobe(SurfaceClosureLobe::dielectric) |
             anisotropic_kind(SurfaceClosureKind::principled)) ||
        // An anisotropic Principled node which has only a diffuse lobe cannot
        // create an anisotropic physical closure. A separate isotropic node's
        // metallic feature must not be spuriously correlated with it.
        reachable_surface_closures(principled_operation,
                                   diffuse_feature | metallic_feature,
                                   principled_operation,
                                   diffuse_feature) !=
            (kinds({SurfaceClosureKind::diffuse}) |
             principled_lobe(SurfaceClosureLobe::metallic)) ||
        reachable_surface_closures(diffuse_operation,
                                   0u,
                                   diffuse_operation,
                                   0u) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(glossy_operation,
                                   0u,
                                   principled_operation,
                                   0u) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(principled_operation,
                                   0u,
                                   principled_operation,
                                   metallic_feature) !=
            all_surface_closure_reachability) {
        std::cerr << "anisotropic closure transfer violated its reduced product\n";
        return false;
    }

    constexpr auto glass_operation = operation_bit(ClosureOperation::glass);
    constexpr auto thick_transmission_feature =
        principled_closure_feature_bit(
            PrincipledClosureFeature::thick_transmission);
    if (reachable_surface_closures(
            glass_operation, 0u, 0u, 0u, glass_operation, 0u) !=
            thin_film_kind(SurfaceClosureKind::glass) ||
        reachable_surface_closures(
            metallic_f82_operation,
            0u,
            0u,
            0u,
            metallic_f82_operation,
            0u) !=
            thin_film_kind(SurfaceClosureKind::metallic_f82) ||
        reachable_surface_closures(
            metallic_conductor_operation,
            0u,
            0u,
            0u,
            metallic_conductor_operation,
            0u) !=
            thin_film_kind(SurfaceClosureKind::metallic_conductor) ||
        reachable_surface_closures(
            principled_operation,
            metallic_feature,
            0u,
            0u,
            principled_operation,
            metallic_feature) !=
            thin_film_principled_lobe(SurfaceClosureLobe::metallic) ||
        reachable_surface_closures(
            principled_operation,
            dielectric_feature,
            0u,
            0u,
            principled_operation,
            dielectric_feature) !=
            thin_film_principled_lobe(SurfaceClosureLobe::dielectric) ||
        reachable_surface_closures(
            principled_operation,
            thick_transmission_feature,
            0u,
            0u,
            principled_operation,
            thick_transmission_feature) !=
            thin_film_kind(SurfaceClosureKind::glass) ||
        reachable_surface_closures(
            glass_operation, 0u, 0u, 0u, principled_operation, 0u) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(
            principled_operation,
            metallic_feature,
            0u,
            0u,
            principled_operation,
            dielectric_feature) !=
            all_surface_closure_reachability ||
        reachable_surface_closures(
            principled_operation,
            0u,
            0u,
            0u,
            principled_operation,
            metallic_feature) !=
            all_surface_closure_reachability) {
        std::cerr << "Thin Film closure transfer violated its reduced product\n";
        return false;
    }
    return true;
}

struct AstFootprint {
    std::size_t functions{};
    std::unordered_set<const Statement *> statements;
    std::unordered_set<const Expression *> expressions;
    std::unordered_set<const void *> builders;
};

void collect_ast(const Function &function, AstFootprint &result) noexcept {
    result.functions += 1u;
    traverse_expressions<true>(
        function.body(),
        [&result](const Expression *expression) noexcept {
            result.expressions.emplace(expression);
        },
        [&result](const Statement *statement) noexcept {
            result.statements.emplace(statement);
        },
        [](const Statement *) noexcept {});
    for (const auto &callable : function.custom_callables()) {
        if (result.builders.emplace(callable.get()).second) {
            collect_ast(callable->function(), result);
        }
    }
}

template <typename Kernel>
[[nodiscard]] AstFootprint ast_footprint(const Kernel &kernel) noexcept {
    AstFootprint result;
    result.builders.emplace(kernel.function().get());
    collect_ast(kernel.function()->function(), result);
    return result;
}

[[nodiscard]] auto make_probe_kernel(SurfaceClosureReachability reachability) {
    return Kernel1D{[reachability](
                        BufferFloat4 parameters, BufferUInt closure_kinds,
                        BufferUInt closure_lobes, BufferFloat4 output) noexcept {
        const auto invocation = dispatch_x();
        ParameterShaderServices services{parameters, 0.73f};
        const auto point = make_surface_point();
        const SurfaceClosurePoint closure_point{point};
        auto closure_record = SurfaceClosureRecord::zero();
        closure_record.kind = closure_kinds.read(invocation);
        closure_record.lobe = closure_lobes.read(invocation);
        closure_record.weight = make_float3(0.7f, 0.5f, 0.3f);
        closure_record.allocation_weight = 0.6f;
        closure_record.sample_weight = 0.55f;
        closure_record.setup_valid = true;
        closure_record.color = make_float3(0.8f, 0.6f, 0.4f);
        closure_record.normal = make_float3(0.0f, 0.0f, 1.0f);
        closure_record.roughness = 0.37f;
        closure_record.microfacet_tangent =
            make_float3(0.6f, 0.8f, 0.0f);
        closure_record.microfacet_alpha_x =
            closure_record.roughness * closure_record.roughness;
        closure_record.microfacet_alpha_y =
            closure_record.microfacet_alpha_x;
        closure_record.diffuse_roughness = 0.21f;
        closure_record.ior = 1.45f;
        closure_record.specular_tint = make_float3(0.9f, 0.8f, 0.7f);
        closure_record.evaluation_scale = make_float3(1.0f);
        closure_record.fresnel_f0 = make_float3(0.04f);
        closure_record.fresnel_f90 = make_float3(1.0f);
        closure_record.reflection_tint = make_float3(1.0f);
        closure_record.transmission_tint = make_float3(0.8f);
        closure_record.preserve_ggx_energy = true;
        const auto closure =
            static_cast<SurfaceClosurePhysicalRecord>(closure_record);
        constexpr auto all_lobes = static_cast<std::uint32_t>(
            event_diffuse | event_glossy | event_transmission | event_transparent);
        const auto query = SurfaceQuery{
            .lobe_mask = all_lobes,
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto policy =
            make_surface_closure_evaluation_policy(false, Expr<std::uint32_t>{0u});
        const auto contribution = surface_closure_evaluation_contribution(
            services, closure_point, point.shading_normal, closure, point.incoming,
            normalize(make_float3(0.23f, -0.17f, 0.958f)), query, policy, false,
            reachability);
        const auto sample = surface_closure_conditional_sample(
            services, closure_point, point.shading_normal, closure, point.incoming,
            closure.normal, make_float2(0.31f, 0.73f), 0.47f, query, reachability);
        const auto base = invocation * 5u;
        output.write(base, make_float4(contribution.f, contribution.weighted_pdf));
        output.write(base + 1u, make_float4(contribution.diffuse_f,
                                            contribution.total_sample_weight));
        output.write(base + 2u,
                     make_float4(contribution.glossy_f,
                                 contribution.weighted_roughness_squared));
        output.write(base + 3u, make_float4(sample.direction,
                                            select(0.0f, 1.0f, sample.valid)));
        output.write(base + 4u, make_float4(sample.roughness, sample.eta,
                                            cast<float>(sample.properties)));
    }};
}

[[nodiscard]] bool bit_exact(luisa::float4 lhs, luisa::float4 rhs) noexcept {
    static_assert(sizeof(luisa::float4) == 4u * sizeof(std::uint32_t));
    return std::bit_cast<std::array<std::uint32_t, 4u>>(lhs) ==
           std::bit_cast<std::array<std::uint32_t, 4u>>(rhs);
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    if (!verify_reachability_lattice()) {
        return EXIT_FAILURE;
    }

    auto diffuse_probe = make_probe_kernel(kinds({SurfaceClosureKind::diffuse}));
    auto top_probe = make_probe_kernel(all_surface_closure_reachability);
    const auto isotropic_glossy_reachability =
        kinds({SurfaceClosureKind::glossy});
    const auto anisotropic_glossy_reachability =
        isotropic_glossy_reachability |
        anisotropic_kind(SurfaceClosureKind::glossy);
    auto isotropic_glossy_probe =
        make_probe_kernel(isotropic_glossy_reachability);
    auto anisotropic_glossy_probe =
        make_probe_kernel(anisotropic_glossy_reachability);
    const auto no_film_metallic_reachability =
        principled_lobe(SurfaceClosureLobe::metallic);
    const auto film_metallic_reachability =
        thin_film_principled_lobe(SurfaceClosureLobe::metallic);
    auto no_film_metallic_probe =
        make_probe_kernel(no_film_metallic_reachability);
    auto film_metallic_probe = make_probe_kernel(film_metallic_reachability);
    const auto standalone_f82_reachability =
        kinds({SurfaceClosureKind::metallic_f82});
    const auto standalone_conductor_reachability =
        kinds({SurfaceClosureKind::metallic_conductor});
    const auto standalone_microfiber_reachability =
        kinds({SurfaceClosureKind::sheen_microfiber});
    const auto standalone_ashikhmin_reachability =
        kinds({SurfaceClosureKind::sheen_ashikhmin});
    auto standalone_f82_probe =
        make_probe_kernel(standalone_f82_reachability);
    auto standalone_conductor_probe =
        make_probe_kernel(standalone_conductor_reachability);
    auto standalone_microfiber_probe =
        make_probe_kernel(standalone_microfiber_reachability);
    auto standalone_ashikhmin_probe =
        make_probe_kernel(standalone_ashikhmin_reachability);
    const auto diffuse_ast = ast_footprint(diffuse_probe);
    const auto top_ast = ast_footprint(top_probe);
    const auto isotropic_glossy_ast = ast_footprint(isotropic_glossy_probe);
    const auto anisotropic_glossy_ast = ast_footprint(anisotropic_glossy_probe);
    const auto no_film_metallic_ast = ast_footprint(no_film_metallic_probe);
    const auto film_metallic_ast = ast_footprint(film_metallic_probe);
    const auto standalone_f82_ast = ast_footprint(standalone_f82_probe);
    const auto standalone_conductor_ast =
        ast_footprint(standalone_conductor_probe);
    const auto standalone_microfiber_ast =
        ast_footprint(standalone_microfiber_probe);
    const auto standalone_ashikhmin_ast =
        ast_footprint(standalone_ashikhmin_probe);
    if (diffuse_probe.function()->function().hash() ==
            top_probe.function()->function().hash() ||
        top_ast.expressions.size() <= diffuse_ast.expressions.size() + 100u ||
        top_ast.statements.size() <= diffuse_ast.statements.size() + 20u) {
        std::cerr << "closure specialization did not remove algorithm ASTs: "
                  << diffuse_ast.expressions.size() << "/"
                  << diffuse_ast.statements.size() << " versus "
                  << top_ast.expressions.size() << "/" << top_ast.statements.size()
                  << " on " << backend << '\n';
        return EXIT_FAILURE;
    }
    if (no_film_metallic_probe.function()->function().hash() ==
            film_metallic_probe.function()->function().hash() ||
        film_metallic_ast.expressions.size() <=
            no_film_metallic_ast.expressions.size() + 100u ||
        film_metallic_ast.statements.size() <=
            no_film_metallic_ast.statements.size()) {
        std::cerr << "Thin Film proof did not remove interference ASTs: "
                  << no_film_metallic_ast.expressions.size() << "/"
                  << no_film_metallic_ast.statements.size() << " versus "
                  << film_metallic_ast.expressions.size() << "/"
                  << film_metallic_ast.statements.size() << " on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    if (isotropic_glossy_probe.function()->function().hash() ==
            anisotropic_glossy_probe.function()->function().hash() ||
        anisotropic_glossy_ast.expressions.size() <=
            isotropic_glossy_ast.expressions.size() ||
        anisotropic_glossy_ast.statements.size() <=
            isotropic_glossy_ast.statements.size()) {
        std::cerr << "isotropic proof did not remove anisotropic microfacet ASTs: "
                  << isotropic_glossy_ast.expressions.size() << "/"
                  << isotropic_glossy_ast.statements.size() << " versus "
                  << anisotropic_glossy_ast.expressions.size() << "/"
                  << anisotropic_glossy_ast.statements.size() << " on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    if (standalone_f82_probe.function()->function().hash() ==
            standalone_conductor_probe.function()->function().hash() ||
        standalone_conductor_ast.expressions.size() <=
            standalone_f82_ast.expressions.size()) {
        std::cerr << "standalone Metallic Fresnel tags did not select "
                     "distinct static algebra: F82 "
                  << standalone_f82_ast.expressions.size() << "/"
                  << standalone_f82_ast.statements.size() << " versus conductor "
                  << standalone_conductor_ast.expressions.size() << "/"
                  << standalone_conductor_ast.statements.size() << " on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    if (standalone_microfiber_probe.function()->function().hash() ==
            standalone_ashikhmin_probe.function()->function().hash() ||
        standalone_microfiber_ast.expressions.size() ==
            standalone_ashikhmin_ast.expressions.size()) {
        std::cerr << "standalone Sheen distribution tags did not select "
                     "distinct static algebra: Microfiber "
                  << standalone_microfiber_ast.expressions.size() << "/"
                  << standalone_microfiber_ast.statements.size()
                  << " versus Ashikhmin "
                  << standalone_ashikhmin_ast.expressions.size() << "/"
                  << standalone_ashikhmin_ast.statements.size() << " on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    constexpr auto scene_operations =
        operation_bit(ClosureOperation::diffuse) |
        operation_bit(ClosureOperation::translucent) |
        operation_bit(ClosureOperation::glossy) |
        operation_bit(ClosureOperation::metallic_f82) |
        operation_bit(ClosureOperation::metallic_conductor) |
        operation_bit(ClosureOperation::sheen_microfiber) |
        operation_bit(ClosureOperation::sheen_ashikhmin) |
        operation_bit(ClosureOperation::hair_reflection) |
        operation_bit(ClosureOperation::hair_transmission) |
        operation_bit(ClosureOperation::glass) |
        operation_bit(ClosureOperation::transparent) |
        operation_bit(ClosureOperation::subsurface) |
        operation_bit(ClosureOperation::refraction);
    constexpr auto scene_features =
        principled_closure_feature_bit(PrincipledClosureFeature::sheen) |
        principled_closure_feature_bit(PrincipledClosureFeature::coat) |
        principled_closure_feature_bit(PrincipledClosureFeature::metallic) |
        principled_closure_feature_bit(PrincipledClosureFeature::dielectric);
    const auto scene_reachability =
        reachable_surface_closures(scene_operations,
                                   scene_features,
                                   0u,
                                   0u);
    auto scene_probe = make_probe_kernel(scene_reachability);

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    constexpr std::array closure_kinds{
        static_cast<std::uint32_t>(SurfaceClosureKind::diffuse),
        static_cast<std::uint32_t>(SurfaceClosureKind::translucent),
        static_cast<std::uint32_t>(SurfaceClosureKind::glossy),
        static_cast<std::uint32_t>(SurfaceClosureKind::metallic_f82),
        static_cast<std::uint32_t>(SurfaceClosureKind::metallic_conductor),
        static_cast<std::uint32_t>(SurfaceClosureKind::sheen_microfiber),
        static_cast<std::uint32_t>(SurfaceClosureKind::sheen_ashikhmin),
        static_cast<std::uint32_t>(SurfaceClosureKind::hair_reflection),
        static_cast<std::uint32_t>(SurfaceClosureKind::hair_transmission),
        static_cast<std::uint32_t>(SurfaceClosureKind::glass),
        static_cast<std::uint32_t>(SurfaceClosureKind::transparent),
        static_cast<std::uint32_t>(SurfaceClosureKind::bssrdf),
        static_cast<std::uint32_t>(SurfaceClosureKind::refraction),
        static_cast<std::uint32_t>(SurfaceClosureKind::principled),
        static_cast<std::uint32_t>(SurfaceClosureKind::principled),
        static_cast<std::uint32_t>(SurfaceClosureKind::principled),
        static_cast<std::uint32_t>(SurfaceClosureKind::principled)};
    constexpr std::array closure_lobes{
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::none),
        static_cast<std::uint32_t>(SurfaceClosureLobe::sheen),
        static_cast<std::uint32_t>(SurfaceClosureLobe::coat),
        static_cast<std::uint32_t>(SurfaceClosureLobe::metallic),
        static_cast<std::uint32_t>(SurfaceClosureLobe::dielectric)};
    static_assert(closure_kinds.size() == closure_lobes.size());
    constexpr auto records_per_closure = 5u;
    auto parameters = device.create_buffer<luisa::float4>(1u);
    auto closure_kind_buffer =
        device.create_buffer<std::uint32_t>(closure_kinds.size());
    auto closure_lobe_buffer =
        device.create_buffer<std::uint32_t>(closure_lobes.size());
    auto specialized_output = device.create_buffer<luisa::float4>(
        closure_kinds.size() * records_per_closure);
    auto top_output = device.create_buffer<luisa::float4>(closure_kinds.size() *
                                                          records_per_closure);
    auto specialized_shader = device.compile(scene_probe);
    auto top_shader = device.compile(top_probe);
    constexpr std::array parameter_data{luisa::float4{0.0f}};
    std::array<luisa::float4, closure_kinds.size() * records_per_closure>
        specialized_result{};
    std::array<luisa::float4, closure_kinds.size() * records_per_closure>
        top_result{};
    stream << parameters.copy_from(luisa::span{parameter_data})
           << closure_kind_buffer.copy_from(luisa::span{closure_kinds})
           << closure_lobe_buffer.copy_from(luisa::span{closure_lobes})
           << specialized_shader(parameters, closure_kind_buffer,
                                 closure_lobe_buffer, specialized_output)
                  .dispatch(static_cast<std::uint32_t>(closure_kinds.size()))
           << top_shader(parameters, closure_kind_buffer, closure_lobe_buffer,
                         top_output)
                  .dispatch(static_cast<std::uint32_t>(closure_kinds.size()))
           << specialized_output.copy_to(luisa::span{specialized_result})
           << top_output.copy_to(luisa::span{top_result}) << synchronize();
    for (auto index = std::size_t{0u}; index < specialized_result.size();
         ++index) {
        if (!bit_exact(specialized_result[index], top_result[index])) {
            const auto specialized_bits =
                std::bit_cast<std::array<std::uint32_t, 4u>>(
                    specialized_result[index]);
            const auto top_bits =
                std::bit_cast<std::array<std::uint32_t, 4u>>(
                    top_result[index]);
            std::cerr << "closure specialization changed reachable result " << index
                      << " on " << backend << ": specialized {"
                      << specialized_result[index].x << ", "
                      << specialized_result[index].y << ", "
                      << specialized_result[index].z << ", "
                      << specialized_result[index].w << "}, top {"
                      << top_result[index].x << ", "
                      << top_result[index].y << ", "
                      << top_result[index].z << ", "
                      << top_result[index].w << "}; bits specialized {"
                      << specialized_bits[0u] << ", "
                      << specialized_bits[1u] << ", "
                      << specialized_bits[2u] << ", "
                      << specialized_bits[3u] << "}, top {"
                      << top_bits[0u] << ", "
                      << top_bits[1u] << ", "
                      << top_bits[2u] << ", "
                      << top_bits[3u] << "}\n";
            return EXIT_FAILURE;
        }
    }
    std::cout << "closure reachability passed on " << backend
              << ": AST expressions " << top_ast.expressions.size() << " -> "
              << diffuse_ast.expressions.size() << ", statements "
              << top_ast.statements.size() << " -> "
              << diffuse_ast.statements.size()
              << "; isotropic/anisotropic Glossy expressions "
              << isotropic_glossy_ast.expressions.size() << "/"
              << anisotropic_glossy_ast.expressions.size()
              << ", statements "
              << isotropic_glossy_ast.statements.size() << "/"
              << anisotropic_glossy_ast.statements.size()
              << "; no-film/film metallic expressions "
              << no_film_metallic_ast.expressions.size() << "/"
              << film_metallic_ast.expressions.size()
              << ", statements "
              << no_film_metallic_ast.statements.size() << "/"
              << film_metallic_ast.statements.size() << '\n';
    return EXIT_SUCCESS;
}
