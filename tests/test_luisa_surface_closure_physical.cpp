#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_physical_blocks.h>
#include <psycles/luisa/surface_closure_sampling.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::ParameterShaderServices;

template<typename T>
concept HasAllocationState = requires(T x) {
    x.allocation_weight;
    x.setup_valid;
};
template<typename T>
concept HasAovInputs = requires(T x) {
    x.albedo;
    x.reflection_albedo;
    x.transmission_albedo;
};
template<typename T>
concept HasSetupClassification = requires(T x) {
    x.specular_ior_level;
    x.preserve_ggx_energy;
    x.beckmann;
    x.bssrdf_method;
};
template<typename T>
concept HasAuthoringMixtureState = requires(T x) {
    x.diffuse_roughness;
    x.metallic;
};
template<typename T>
concept HasGeneralState = requires(T x) {
    x.payload.microfacet_tangent;
    x.payload.microfacet_alpha_x;
    x.payload.microfacet_alpha_y;
    x.payload.specular_tint;
    x.payload.evaluation_scale;
};
template<typename T>
concept HasHairState = requires(T x) {
    x.payload.tangent;
    x.payload.roughness_u;
    x.payload.roughness_v;
    x.payload.offset;
};
template<typename T>
concept HasDielectricState = requires(T x) {
    x.payload.fresnel_f0;
    x.payload.fresnel_f90;
    x.payload.reflection_tint;
    x.payload.transmission_tint;
};
template<typename T>
concept HasBssrdfState = requires(T x) {
    x.payload.radius;
    x.payload.albedo;
    x.payload.bssrdf_ior;
    x.payload.roughness;
    x.payload.anisotropy;
};
template<typename T>
concept HasThinFilmState = requires(T x) {
    x.payload.thin_film_thickness;
    x.payload.thin_film_ior;
};

static_assert(surface_closure_physical_block_count == 2u);
static_assert(!HasAllocationState<SurfaceClosurePhysicalRecord>);
static_assert(!HasAovInputs<SurfaceClosurePhysicalRecord>);
static_assert(!HasSetupClassification<SurfaceClosurePhysicalRecord>);
static_assert(!HasAuthoringMixtureState<SurfaceClosurePhysicalRecord>);

static_assert(!HasGeneralState<SurfaceClosurePhysicalCommonOnlyRecord>);
static_assert(!HasHairState<SurfaceClosurePhysicalCommonOnlyRecord>);
static_assert(!HasDielectricState<SurfaceClosurePhysicalCommonOnlyRecord>);
static_assert(!HasBssrdfState<SurfaceClosurePhysicalCommonOnlyRecord>);
static_assert(!HasThinFilmState<SurfaceClosurePhysicalCommonOnlyRecord>);

static_assert(HasGeneralState<SurfaceClosurePhysicalGeneralRecord>);
static_assert(!HasHairState<SurfaceClosurePhysicalGeneralRecord>);
static_assert(!HasDielectricState<SurfaceClosurePhysicalGeneralRecord>);
static_assert(!HasBssrdfState<SurfaceClosurePhysicalGeneralRecord>);
static_assert(HasThinFilmState<SurfaceClosurePhysicalGeneralRecord>);

static_assert(HasHairState<SurfaceClosurePhysicalHairRecord>);
static_assert(!HasGeneralState<SurfaceClosurePhysicalHairRecord>);
static_assert(!HasDielectricState<SurfaceClosurePhysicalHairRecord>);
static_assert(!HasBssrdfState<SurfaceClosurePhysicalHairRecord>);
static_assert(!HasThinFilmState<SurfaceClosurePhysicalHairRecord>);

static_assert(HasDielectricState<SurfaceClosurePhysicalDielectricRecord>);
static_assert(!HasGeneralState<SurfaceClosurePhysicalDielectricRecord>);
static_assert(!HasHairState<SurfaceClosurePhysicalDielectricRecord>);
static_assert(!HasBssrdfState<SurfaceClosurePhysicalDielectricRecord>);
static_assert(HasThinFilmState<SurfaceClosurePhysicalDielectricRecord>);

static_assert(HasBssrdfState<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(!HasGeneralState<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(!HasHairState<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(!HasDielectricState<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(!HasThinFilmState<SurfaceClosurePhysicalBssrdfRecord>);

enum class PayloadClass : std::uint32_t {
    common,
    general,
    hair,
    dielectric,
    bssrdf,
};

struct ClosureCase {
    std::uint32_t type;
    cycles_closure::MicrofacetFresnel fresnel;
    PayloadClass payload;
    SurfaceBssrdfMethod bssrdf_method;
    bool beckmann;
    bool preserve_energy;
};

using Fresnel = cycles_closure::MicrofacetFresnel;
constexpr auto common = PayloadClass::common;
constexpr auto general = PayloadClass::general;
constexpr auto hair = PayloadClass::hair;
constexpr auto dielectric = PayloadClass::dielectric;
constexpr auto bssrdf = PayloadClass::bssrdf;
constexpr auto random_walk = SurfaceBssrdfMethod::random_walk;

constexpr std::array closure_cases{
    ClosureCase{cycles_closure::type_oren_nayar, Fresnel::none,
                common, random_walk, false, false},
    ClosureCase{cycles_closure::type_translucent, Fresnel::none,
                common, random_walk, false, false},
    ClosureCase{cycles_closure::type_sheen, Fresnel::none,
                general, random_walk, false, false},
    ClosureCase{cycles_closure::type_microfacet_ggx,
                Fresnel::generalized_schlick,
                general, random_walk, false, false},
    ClosureCase{cycles_closure::type_microfacet_ggx, Fresnel::f82_tint,
                general, random_walk, false, false},
    ClosureCase{cycles_closure::type_microfacet_ggx,
                Fresnel::generalized_schlick,
                general, random_walk, false, false},
    ClosureCase{cycles_closure::type_microfacet_beckmann, Fresnel::none,
                general, random_walk, true, false},
    ClosureCase{cycles_closure::type_microfacet_ggx, Fresnel::f82_tint,
                general, random_walk, false, false},
    ClosureCase{cycles_closure::type_microfacet_ggx, Fresnel::conductor,
                general, random_walk, false, false},
    ClosureCase{cycles_closure::type_sheen, Fresnel::none,
                general, random_walk, false, false},
    ClosureCase{cycles_closure::type_ashikhmin_velvet, Fresnel::none,
                common, random_walk, false, false},
    ClosureCase{cycles_closure::type_hair_reflection, Fresnel::none,
                hair, random_walk, false, false},
    ClosureCase{cycles_closure::type_hair_transmission, Fresnel::none,
                hair, random_walk, false, false},
    ClosureCase{cycles_closure::type_microfacet_beckmann_glass,
                Fresnel::generalized_schlick,
                dielectric, random_walk, true, false},
    ClosureCase{cycles_closure::type_microfacet_ggx_glass,
                Fresnel::generalized_schlick,
                dielectric, random_walk, false, true},
    ClosureCase{cycles_closure::type_microfacet_beckmann_refraction,
                Fresnel::none, dielectric, random_walk, true, false},
    ClosureCase{cycles_closure::type_bssrdf_burley, Fresnel::none,
                bssrdf, SurfaceBssrdfMethod::burley, false, false},
    ClosureCase{cycles_closure::type_bssrdf_random_walk, Fresnel::none,
                bssrdf, random_walk, false, false},
    ClosureCase{cycles_closure::type_bssrdf_random_walk_legacy,
                Fresnel::none, bssrdf,
                SurfaceBssrdfMethod::random_walk_legacy, false, false},
    ClosureCase{cycles_closure::type_bssrdf_random_walk_skin,
                Fresnel::none, bssrdf,
                SurfaceBssrdfMethod::random_walk_skin, false, false},
    ClosureCase{cycles_closure::type_rough_translucent, Fresnel::none,
                common, random_walk, false, false},
    ClosureCase{cycles_closure::type_thin_glass_transmission,
                Fresnel::none, general, random_walk, false, false},
    ClosureCase{cycles_closure::type_transparent, Fresnel::none,
                common, random_walk, false, false}};
constexpr auto case_count =
    static_cast<std::uint32_t>(closure_cases.size());

namespace result_row {
constexpr std::uint32_t identity = 0u;
constexpr std::uint32_t common_projection = 1u;
constexpr std::uint32_t family_projection = 2u;
constexpr std::uint32_t canonical_payload = 3u;
constexpr std::uint32_t evaluation = 4u;
constexpr std::uint32_t sample = 5u;
constexpr std::uint32_t selection = 6u;
constexpr std::uint32_t payload_reads = 7u;
constexpr std::uint32_t invalid_setup = 8u;
constexpr std::uint32_t derived_identity = 9u;
constexpr std::uint32_t count = 10u;
}// namespace result_row

constexpr float abs_tolerance = 1.0e-5f;
constexpr float rel_tolerance =
    8.0f * std::numeric_limits<float>::epsilon();

[[nodiscard]] Bool close(Float a, Float b) noexcept {
    return abs(a - b) <= abs_tolerance +
                              rel_tolerance * max(abs(a), abs(b));
}
[[nodiscard]] Bool close(Float2 a, Float2 b) noexcept {
    return all(abs(a - b) <= make_float2(abs_tolerance) +
                                  rel_tolerance * max(abs(a), abs(b)));
}
[[nodiscard]] Bool close(Float3 a, Float3 b) noexcept {
    return all(abs(a - b) <= make_float3(abs_tolerance) +
                                  rel_tolerance * max(abs(a), abs(b)));
}

[[nodiscard]] Bool same_common(
    const SurfaceClosurePhysicalCommonRecord &a,
    const SurfaceClosurePhysicalCommonRecord &b) noexcept {
    return (a.closure_type == b.closure_type) &
           (a.microfacet_fresnel == b.microfacet_fresnel) &
           all(a.weight == b.weight) &
           (a.sample_weight == b.sample_weight) &
           all(a.color_or_evaluation_scale == b.color_or_evaluation_scale) &
           all(a.normal == b.normal) & (a.roughness == b.roughness);
}

[[nodiscard]] Bool same_evaluation(
    const Var<SurfaceClosureEvaluationContributionCall> &a,
    const Var<SurfaceClosureEvaluationContributionCall> &b) noexcept {
    return close(a.f, b.f) & close(a.diffuse_f, b.diffuse_f) &
           close(a.glossy_f, b.glossy_f) &
           close(a.total_sample_weight, b.total_sample_weight) &
           close(a.weighted_pdf, b.weighted_pdf) &
           close(a.weighted_roughness_squared,
                 b.weighted_roughness_squared) &
           (a.events == b.events);
}

[[nodiscard]] Bool same_sample(
    const Var<SurfaceClosureConditionalSampleCall> &a,
    const Var<SurfaceClosureConditionalSampleCall> &b) noexcept {
    return all(abs(a.direction - b.direction) <= make_float3(5.0e-4f)) &
           close(a.roughness, b.roughness) &
           close(a.singular_evaluation, b.singular_evaluation) &
           close(a.singular_pdf, b.singular_pdf) & close(a.eta, b.eta) &
           (a.properties == b.properties) & (a.valid == b.valid) &
           (a.bssrdf_method == b.bssrdf_method) &
           close(a.bssrdf_radius, b.bssrdf_radius) &
           close(a.bssrdf_albedo, b.bssrdf_albedo) &
           close(a.bssrdf_normal, b.bssrdf_normal) &
           close(a.bssrdf_ior, b.bssrdf_ior) &
           close(a.bssrdf_roughness, b.bssrdf_roughness) &
           close(a.bssrdf_anisotropy, b.bssrdf_anisotropy);
}

[[nodiscard]] SurfaceClosureRecord make_source(UInt case_index) noexcept {
    auto result = SurfaceClosureRecord::zero();
    for (auto i = 0u; i < case_count; ++i) {
        const auto selected = case_index == i;
        const auto &c = closure_cases[i];
        result.closure_type = select(
            result.closure_type, c.type, selected);
        result.microfacet_fresnel = select(
            result.microfacet_fresnel,
            static_cast<std::uint32_t>(c.fresnel),
            selected);
        result.bssrdf_method = select(
            result.bssrdf_method,
            static_cast<std::uint32_t>(c.bssrdf_method), selected);
        result.beckmann = select(result.beckmann, c.beckmann, selected);
        result.preserve_ggx_energy = select(
            result.preserve_ggx_energy, c.preserve_energy, selected);
    }
    result.weight = make_float3(0.37f, 0.29f, 0.41f);
    result.allocation_weight = 0.73f;
    result.sample_weight = 0.61f;
    result.setup_valid = true;
    result.color = make_float3(0.62f, 0.48f, 0.31f);
    result.normal = normalize(make_float3(0.13f, -0.09f, 0.98f));
    result.roughness = 0.36f;
    result.microfacet_tangent = normalize(make_float3(0.93f, 0.31f, 0.07f));
    result.microfacet_alpha_x = 0.08f;
    result.microfacet_alpha_y = 0.31f;
    result.ior = 1.43f;
    result.thin_film_thickness = 385.0f;
    result.thin_film_ior = 1.32f;
    result.specular_tint = make_float3(0.84f, 0.71f, 0.59f);
    result.sheen_transform_a = 0.83f;
    result.sheen_transform_b = 0.17f;
    result.evaluation_scale = make_float3(0.91f, 0.87f, 0.79f);
    result.fresnel_f0 = make_float3(0.04f, 0.05f, 0.06f);
    result.fresnel_f90 = make_float3(1.0f);
    result.reflection_tint = make_float3(0.92f, 0.81f, 0.73f);
    result.transmission_tint = make_float3(0.67f, 0.76f, 0.88f);
    result.bssrdf_radius = make_float3(0.8f, 0.5f, 0.3f);
    result.bssrdf_albedo = make_float3(0.7f, 0.4f, 0.2f);
    result.bssrdf_ior = 1.4f;
    result.bssrdf_roughness = 0.32f;
    result.bssrdf_anisotropy = 0.15f;
    return result;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Kernel1D test = [](BufferFloat4 parameters, BufferFloat4 output) noexcept {
        const UInt case_index{dispatch_x()};
        const auto source = make_source(case_index);
        const auto physical = static_cast<SurfaceClosurePhysicalRecord>(source);
        const auto packed = pack_surface_closure_physical(physical);
        const auto decoded_common = unpack_surface_closure_physical_common(
            Expr<luisa::float4x4>{packed.block_0.expression()});
        const auto direct_common =
            project_surface_closure_physical_common(physical);

        UInt expected_type = closure_cases[0u].type;
        UInt expected_fresnel = static_cast<std::uint32_t>(
            closure_cases[0u].fresnel);
        UInt expected_payload = static_cast<std::uint32_t>(
            closure_cases[0u].payload);
        UInt expected_method = static_cast<std::uint32_t>(
            closure_cases[0u].bssrdf_method);
        Bool expected_beckmann = closure_cases[0u].beckmann;
        for (auto i = 1u; i < case_count; ++i) {
            const auto selected = case_index == i;
            expected_type = select(expected_type, closure_cases[i].type, selected);
            expected_fresnel = select(
                expected_fresnel,
                static_cast<std::uint32_t>(closure_cases[i].fresnel), selected);
            expected_payload = select(
                expected_payload,
                static_cast<std::uint32_t>(closure_cases[i].payload), selected);
            expected_method = select(
                expected_method,
                static_cast<std::uint32_t>(closure_cases[i].bssrdf_method),
                selected);
            expected_beckmann = select(
                expected_beckmann, closure_cases[i].beckmann, selected);
        }

        const auto identity = packed.block_0[0u].bitcast<luisa::uint4>();
        const auto identity_ok =
            (identity.x == expected_type) & (identity.y == expected_fresnel) &
            (identity.z == 0u) & (identity.w == 0u) &
            (packed.block_0[1u].w == 0.0f) &
            (decoded_common.closure_type == expected_type) &
            (decoded_common.microfacet_fresnel == expected_fresnel);
        output.write(case_index * result_row::count + result_row::identity,
                     make_float4(select(0.0f, 1.0f, identity_ok)));
        output.write(
            case_index * result_row::count + result_row::common_projection,
            make_float4(select(
                0.0f, 1.0f, same_common(decoded_common, direct_common))));

        const auto uses_general = surface_closure_uses_general_payload(
            decoded_common.closure_type);
        const auto uses_hair = surface_closure_uses_hair_payload(
            decoded_common.closure_type);
        const auto uses_dielectric = surface_closure_uses_dielectric_payload(
            decoded_common.closure_type);
        const auto uses_bssrdf = surface_closure_uses_bssrdf_payload(
            decoded_common.closure_type);
        Bool family_equal = false;
        $if(uses_general) {
            const auto a = project_surface_closure_physical_general(physical);
            const auto b = unpack_surface_closure_physical_general(
                decoded_common,
                Expr<luisa::float4x4>{packed.block_1.expression()});
            family_equal = same_common(a.common, b.common) &
                (a.payload.thin_film_thickness == b.payload.thin_film_thickness) &
                (a.payload.thin_film_ior == b.payload.thin_film_ior) &
                (a.payload.ior == b.payload.ior) &
                all(a.payload.specular_tint == b.payload.specular_tint) &
                (a.payload.sheen_transform_a == b.payload.sheen_transform_a) &
                (a.payload.sheen_transform_b == b.payload.sheen_transform_b) &
                all(a.payload.evaluation_scale == b.payload.evaluation_scale) &
                all(a.payload.microfacet_tangent == b.payload.microfacet_tangent) &
                (a.payload.microfacet_alpha_x == b.payload.microfacet_alpha_x) &
                (a.payload.microfacet_alpha_y == b.payload.microfacet_alpha_y);
        }
        $elif(uses_hair) {
            const auto a = project_surface_closure_physical_hair(physical);
            const auto b = unpack_surface_closure_physical_hair(
                decoded_common,
                Expr<luisa::float4x4>{packed.block_1.expression()});
            family_equal = same_common(a.common, b.common) &
                all(a.payload.tangent == b.payload.tangent) &
                (a.payload.roughness_u == b.payload.roughness_u) &
                (a.payload.roughness_v == b.payload.roughness_v) &
                (a.payload.offset == b.payload.offset);
        }
        $elif(uses_dielectric) {
            const auto a = project_surface_closure_physical_dielectric(physical);
            const auto b = unpack_surface_closure_physical_dielectric(
                decoded_common,
                Expr<luisa::float4x4>{packed.block_1.expression()});
            family_equal = same_common(a.common, b.common) &
                (a.payload.ior == b.payload.ior) &
                (a.payload.thin_film_thickness == b.payload.thin_film_thickness) &
                (a.payload.thin_film_ior == b.payload.thin_film_ior) &
                all(a.payload.fresnel_f0 == b.payload.fresnel_f0) &
                all(a.payload.fresnel_f90 == b.payload.fresnel_f90) &
                all(a.payload.reflection_tint == b.payload.reflection_tint) &
                all(a.payload.transmission_tint == b.payload.transmission_tint);
        }
        $elif(uses_bssrdf) {
            const auto a = project_surface_closure_physical_bssrdf(physical);
            const auto b = unpack_surface_closure_physical_bssrdf(
                decoded_common,
                Expr<luisa::float4x4>{packed.block_1.expression()});
            family_equal = same_common(a.common, b.common) &
                all(a.payload.radius == b.payload.radius) &
                all(a.payload.albedo == b.payload.albedo) &
                (a.payload.bssrdf_ior == b.payload.bssrdf_ior) &
                (a.payload.roughness == b.payload.roughness) &
                (a.payload.anisotropy == b.payload.anisotropy);
        }
        $else {
            family_equal = expected_payload ==
                static_cast<std::uint32_t>(PayloadClass::common);
        };
        const auto family_count =
            cast<std::uint32_t>(uses_general) +
            cast<std::uint32_t>(uses_hair) +
            cast<std::uint32_t>(uses_dielectric) +
            cast<std::uint32_t>(uses_bssrdf);
        UInt actual_payload = static_cast<std::uint32_t>(PayloadClass::common);
        actual_payload = select(actual_payload,
            static_cast<std::uint32_t>(PayloadClass::general), uses_general);
        actual_payload = select(actual_payload,
            static_cast<std::uint32_t>(PayloadClass::hair), uses_hair);
        actual_payload = select(actual_payload,
            static_cast<std::uint32_t>(PayloadClass::dielectric), uses_dielectric);
        actual_payload = select(actual_payload,
            static_cast<std::uint32_t>(PayloadClass::bssrdf), uses_bssrdf);
        family_equal &= (family_count <= 1u) &
                        (actual_payload == expected_payload);
        output.write(
            case_index * result_row::count + result_row::family_projection,
            make_float4(select(0.0f, 1.0f, family_equal)));

        // The physical blocks are a canonical tagged sum, not merely a
        // sufficiently large byte container. Lanes outside the selected
        // payload family are representation-independent zero; otherwise two
        // observationally equal closures could retain different hidden state.
        Bool payload_canonical = true;
        $if(uses_general) {
            const auto film_payload =
                cycles_closure::fresnel_uses_thin_film_payload(
                    decoded_common.microfacet_fresnel);
            payload_canonical &= film_payload |
                ((packed.block_1[0u].w == 0.0f) &
                 (packed.block_1[1u].w == 0.0f));
        }
        $elif(uses_hair) {
            payload_canonical &= all(packed.block_1[0u] == make_float4(0.0f)) &
                all(packed.block_1[1u] == make_float4(0.0f)) &
                (packed.block_1[2u].y == 0.0f) &
                (packed.block_1[2u].z == 0.0f);
        }
        $elif(uses_dielectric) {
            payload_canonical &= packed.block_1[3u].w == 0.0f;
        }
        $elif(uses_bssrdf) {
            payload_canonical &=
                all(packed.block_1[2u].yzw() == make_float3(0.0f)) &
                all(packed.block_1[3u] == make_float4(0.0f));
        }
        $else {
            payload_canonical &=
                all(packed.block_1[0u] == make_float4(0.0f)) &
                all(packed.block_1[1u] == make_float4(0.0f)) &
                all(packed.block_1[2u] == make_float4(0.0f)) &
                all(packed.block_1[3u] == make_float4(0.0f));
        };
        output.write(
            case_index * result_row::count + result_row::canonical_payload,
            make_float4(select(0.0f, 1.0f, payload_canonical)));

        auto point_value = make_surface_point();
        point_value.incoming = normalize(make_float3(0.21f, -0.16f, 0.96f));
        const SurfaceClosurePoint point{point_value};
        ParameterShaderServices services{parameters, 1.0f};
        const auto incoming = make_surface_closure_sampling_incoming(point);
        const auto outgoing = normalize(make_float3(-0.18f, 0.24f, 0.95f));
        const auto query = SurfaceQuery{
            .lobe_mask = static_cast<std::uint32_t>(
                event_diffuse | event_glossy | event_transmission |
                event_transparent),
            .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto policy = SurfaceClosureEvaluationPolicy{
            .diffuse_included = true,
            .glossy_included = true,
            .glass_included = true,
            .transmission_included = true,
            .preserve_pdf = true};
        const auto direct_eval = surface_closure_evaluation_contribution(
            services, point, point_value.shading_normal, physical,
            incoming, outgoing, query, policy, false);
        UInt payload_reads = 0u;
        const auto load_payload = [&] {
            payload_reads += 1u;
            return Float4x4{packed.block_1};
        };
        const auto retained_eval =
            surface_closure_evaluation_contribution_from_physical_common(
                services, point, point_value.shading_normal, decoded_common,
                load_payload, incoming, outgoing, query, policy, false);
        output.write(case_index * result_row::count + result_row::evaluation,
            make_float4(select(
                0.0f, 1.0f, same_evaluation(direct_eval, retained_eval))));

        const auto direct_sample = surface_closure_conditional_sample(
            services, point, point_value.shading_normal, physical,
            incoming, physical.normal, make_float2(0.37f, 0.64f),
            0.42f, query);
        const auto retained_sample =
            surface_closure_conditional_sample_from_physical_common(
                services, point, point_value.shading_normal, decoded_common,
                load_payload, incoming, decoded_common.normal,
                make_float2(0.37f, 0.64f), 0.42f, query);
        output.write(case_index * result_row::count + result_row::sample,
            make_float4(select(
                0.0f, 1.0f, same_sample(direct_sample, retained_sample))));

        Bool selections_equal = true;
        constexpr std::array masks{
            std::uint32_t{0u},
            static_cast<std::uint32_t>(event_diffuse),
            static_cast<std::uint32_t>(event_glossy),
            static_cast<std::uint32_t>(event_transmission),
            static_cast<std::uint32_t>(event_transparent),
            static_cast<std::uint32_t>(event_diffuse | event_transmission),
            static_cast<std::uint32_t>(event_glossy | event_transmission),
            static_cast<std::uint32_t>(
                event_diffuse | event_glossy | event_transmission |
                event_transparent)};
        for (const auto mask : masks) {
            const auto context = SurfaceClosureSelectionContext{
                .lobe_mask = mask, .glossy_filter_roughness = 0.07f};
            const auto a = surface_closure_selection(
                context, physical, true);
            const auto b = surface_closure_selection(
                context, decoded_common, true);
            selections_equal &= close(a.weight, b.weight) &
                (a.runtime_flags == b.runtime_flags) &
                (a.closure_type == b.closure_type) &
                close(a.closure_sample_weight, b.closure_sample_weight) &
                all(a.glossy_normal == b.glossy_normal);
        }
        output.write(case_index * result_row::count + result_row::selection,
                     make_float4(select(0.0f, 1.0f, selections_equal)));

        const auto expected_reads =
            select(0u, 2u, uses_general | uses_hair | uses_dielectric) +
            select(0u, 1u, uses_bssrdf);
        output.write(case_index * result_row::count + result_row::payload_reads,
            make_float4(select(
                0.0f, 1.0f, payload_reads == expected_reads)));

        auto invalid = physical;
        invalid.closure_type = cycles_closure::type_none;
        invalid.microfacet_fresnel = static_cast<std::uint32_t>(
            Fresnel::none);
        const auto invalid_blocks = pack_surface_closure_physical(invalid);
        const auto invalid_common = unpack_surface_closure_physical_common(
            Expr<luisa::float4x4>{invalid_blocks.block_0.expression()});
        const auto invalid_selection = surface_closure_selection(
            SurfaceClosureSelectionContext{
                .lobe_mask = static_cast<std::uint32_t>(
                    event_diffuse | event_glossy | event_transmission |
                    event_transparent),
                .glossy_filter_roughness = 0.07f},
            invalid_common, true);
        const auto invalid_ok =
            (invalid_common.closure_type == cycles_closure::type_none) &
            (invalid_common.microfacet_fresnel ==
             static_cast<std::uint32_t>(Fresnel::none)) &
            !surface_closure_uses_general_payload(invalid_common.closure_type) &
            !surface_closure_uses_hair_payload(invalid_common.closure_type) &
            !surface_closure_uses_dielectric_payload(invalid_common.closure_type) &
            !surface_closure_uses_bssrdf_payload(invalid_common.closure_type) &
            (invalid_selection.weight == 0.0f) &
            (invalid_selection.runtime_flags == 0u);
        output.write(case_index * result_row::count + result_row::invalid_setup,
                     make_float4(select(0.0f, 1.0f, invalid_ok)));

        UInt derived_method = static_cast<std::uint32_t>(random_walk);
        derived_method = select(derived_method,
            static_cast<std::uint32_t>(SurfaceBssrdfMethod::burley),
            decoded_common.closure_type == cycles_closure::type_bssrdf_burley);
        derived_method = select(derived_method,
            static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk_legacy),
            decoded_common.closure_type ==
                cycles_closure::type_bssrdf_random_walk_legacy);
        derived_method = select(derived_method,
            static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk_skin),
            decoded_common.closure_type ==
                cycles_closure::type_bssrdf_random_walk_skin);
        const auto distribution_ok =
            !cycles_closure::is_reflection_microfacet(
                decoded_common.closure_type) |
            (cycles_closure::is_beckmann_microfacet(
                 decoded_common.closure_type) == expected_beckmann);
        const auto method_ok = !uses_bssrdf |
                               (derived_method == expected_method);
        output.write(case_index * result_row::count +
                         result_row::derived_identity,
            make_float4(select(
                0.0f, 1.0f, distribution_ok & method_ok)));
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto parameters = device.create_buffer<luisa::float4>(1u);
    auto output = device.create_buffer<luisa::float4>(
        case_count * result_row::count);
    auto shader = compile_named_kernel(
        device, "surface closure Cycles ABI", test);
    const std::array parameter_data{luisa::float4{1.0f}};
    std::array<luisa::float4, case_count * result_row::count> actual{};
    stream << parameters.copy_from(luisa::span{parameter_data})
           << shader(parameters, output).dispatch(case_count)
           << output.copy_to(luisa::span{actual}) << synchronize();
    for (auto c = 0u; c < case_count; ++c) {
        for (auto r = 0u; r < result_row::count; ++r) {
            const auto value = actual[c * result_row::count + r];
            if (!(value.x == 1.0f && value.y == 1.0f &&
                  value.z == 1.0f && value.w == 1.0f)) {
                std::cerr << "Physical closure Cycles ABI mismatch on "
                          << backend << std::setprecision(9) << ", case "
                          << c << ", row " << r << ": got {" << value.x
                          << ", " << value.y << ", " << value.z << ", "
                          << value.w << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
