#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface.h> through the Psycles::luisa target."
#endif

#include <cstdint>
#include <optional>

#include <psycles/contract/cycles_abi.h>
#include <psycles/contract/surface.h>

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend {

namespace cycles_volume_phase {
struct Closure;
}// namespace cycles_volume_phase

using luisa::compute::Bool;
using luisa::compute::Expr;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4;
using luisa::compute::make_float2;
using luisa::compute::make_float3;
using luisa::compute::make_float4;

using contract::event_diffuse;
using contract::event_glossy;
using contract::event_none;
using contract::event_reflection;
using contract::event_singular;
using contract::event_subsurface;
using contract::event_transmission;
using contract::event_transparent;
using contract::SurfaceEvent;
using contract::TransportMode;
using luisa::compute::UInt;
using luisa::compute::ULong;

struct SurfacePoint {
    Float3 position;
    Float3 object_position;
    Float3 object_location;
    Float3 generated;
    Float3 geometric_normal;
    Float3 shading_normal;
    Float3 object_shading_normal;
    Float tangent_sign;
    Float3 dpdu;
    Float3 dpdv;
    Float3 dPdx;
    Float3 dPdy;
    Float3 object_dPdx;
    Float3 object_dPdy;
    Float3 incoming;
    Float2 uv;
    UInt geometry_index;
    Float2 barycentric;
    Float2 barycentric_dx;
    Float2 barycentric_dy;
    UInt instance_id;
    UInt primitive_id;
    UInt parameter_block;
    Float object_random;
    UInt particle_index;
    Float random_per_island;
    Bool triangle_smooth;
    Bool is_curve;
    UInt ray_visibility;
    UInt ray_events;
    UInt ray_depth;
    UInt diffuse_depth;
    UInt glossy_depth;
    UInt transparent_depth;
    UInt transmission_depth;
    Float ray_length;
    Float time;
    Bool use_bump_map_correction;
    Bool back_facing;
};

struct SurfaceQuery {
    UInt lobe_mask;
    UInt transport_mode;
    Float glossy_filter_roughness;
    Bool reflective_caustics{true};
    Bool refractive_caustics{true};
    Bool subsurface_exit{false};
    Float3 subsurface_normal{make_float3(0.0f, 0.0f, 1.0f)};
};

struct SurfaceLightQuery {
    SurfaceQuery surface;
    UInt shader_flags;
};

struct VolumeCoefficients {
    Float3 sigma_t;
    Float3 sigma_s;
    Float3 emission;
    Bool has_extinction;
    Bool has_scatter;
    Bool has_emission;

    [[nodiscard]] static VolumeCoefficients zero() noexcept {
        return {
            .sigma_t = make_float3(0.0f),
            .sigma_s = make_float3(0.0f),
            .emission = make_float3(0.0f),
            .has_extinction = false,
            .has_scatter = false,
            .has_emission = false};
    }
};

class VolumePhaseCollector {

  public:
    virtual ~VolumePhaseCollector() noexcept = default;
    virtual void add(
        const cycles_volume_phase::Closure &phase,
        Float3 weight) noexcept = 0;
};

enum class SurfaceBssrdfMethod : std::uint32_t {
    burley,
    random_walk,
    random_walk_legacy,
    random_walk_skin
};

// Cycles kernel/types.h::MAX_CLOSURE.
inline constexpr std::uint32_t
    maximum_surface_closure_capacity = 64u;

struct SurfacePopulationQuery {
    Expr<bool> emission_reflective_caustics;
    Expr<bool> reflective_caustics;
    Expr<bool> refractive_caustics;
    Expr<float> glossy_filter_roughness;
    Expr<bool> include_runtime_flags;
    Expr<bool> include_aov;
};

struct SurfaceEvaluation {
    Float3 f;
    Float pdf;
    Float3 diffuse_f;
    Float3 glossy_f;
    Float diffuse_pdf;
    Float average_roughness_squared;
    UInt events;
    std::optional<Float4> spectral_f{};
    std::optional<Float4> spectral_diffuse_f{};
    std::optional<Float4> spectral_glossy_f{};
    std::optional<Bool> spectral_terminated_secondary{};

    [[nodiscard]] static SurfaceEvaluation zero(bool spectral = false) noexcept {
        SurfaceEvaluation out{
            .f = make_float3(0.0f),
            .pdf = 0.0f,
            .diffuse_f = make_float3(0.0f),
            .glossy_f = make_float3(0.0f),
            .diffuse_pdf = 0.0f,
            .average_roughness_squared = 0.0f,
            .events = static_cast<std::uint32_t>(event_none)};
        if (spectral) {
            out.spectral_f.emplace(make_float4(0.0f));
            out.spectral_diffuse_f.emplace(make_float4(0.0f));
            out.spectral_glossy_f.emplace(make_float4(0.0f));
            out.spectral_terminated_secondary.emplace(false);
        }
        return out;
    }
};

struct SurfaceSample {
    SurfaceEvaluation evaluation;
    Float3 wi;
    Float eta;
    Float2 roughness;
    UInt runtime_flags;
    UInt bssrdf_method;
    Float3 bssrdf_radius;
    Float3 bssrdf_albedo;
    Float3 bssrdf_normal;
    Float bssrdf_ior;
    Float bssrdf_roughness;
    Float bssrdf_anisotropy;
    Bool valid;

    [[nodiscard]] static SurfaceSample zero(bool spectral = false) noexcept {
        return {
            .evaluation = SurfaceEvaluation::zero(spectral),
            .wi = make_float3(0.0f, 0.0f, 1.0f),
            .eta = 1.0f,
            .roughness = make_float2(0.0f),
            .runtime_flags = 0u,
            .bssrdf_method = static_cast<std::uint32_t>(
                SurfaceBssrdfMethod::random_walk),
            .bssrdf_radius = make_float3(0.0f),
            .bssrdf_albedo = make_float3(0.0f),
            .bssrdf_normal = make_float3(0.0f, 0.0f, 1.0f),
            .bssrdf_ior = 1.4f,
            .bssrdf_roughness = 1.0f,
            .bssrdf_anisotropy = 0.0f,
            .valid = false};
    }
};

struct SurfaceClosureTrace {
    UInt count;
    UInt runtime_flags;
    UInt index;
    UInt type;
    Float sample_weight;
    Float3 weight;
    Float3 normal;
    Bool valid;

    [[nodiscard]] static SurfaceClosureTrace zero(
        Expr<std::uint32_t> requested_index = 0u) noexcept {
        return {
            .count = 0u,
            .runtime_flags = 0u,
            .index = requested_index,
            .type = 0u,
            .sample_weight = 0.0f,
            .weight = make_float3(0.0f),
            .normal = make_float3(0.0f, 0.0f, 1.0f),
            .valid = false};
    }
};

struct SurfaceSampleTrace {
    SurfaceSample sample;
    UInt closure_index;
    UInt closure_type;
    Float closure_sample_weight;
    Float selection_rescaled;
    Float3 closure_weight;
    Float3 closure_normal;
    Bool closure_valid;

    [[nodiscard]] static SurfaceSampleTrace zero(bool spectral = false) noexcept {
        return {
            .sample = SurfaceSample::zero(spectral),
            .closure_index = 0u,
            .closure_type = 0u,
            .closure_sample_weight = 0.0f,
            .selection_rescaled = 0.0f,
            .closure_weight = make_float3(0.0f),
            .closure_normal = make_float3(0.0f, 0.0f, 1.0f),
            .closure_valid = false};
    }
};

struct SurfaceAov {
    Float3 albedo;
    Float3 glossy_albedo;
    Float3 transmission_albedo;
    Float2 roughness;
    Float3 normal;
    Float3 transparency;
};

struct SurfacePreparation {
    Float3 emission;
    Float3 shading_normal;
    UInt runtime_flags;
    SurfaceAov aov;
    std::optional<Float4> spectral_emission{};

    [[nodiscard]] static SurfacePreparation zero(
        const SurfacePoint &point, bool spectral = false) noexcept {
        SurfacePreparation out{
            .emission = make_float3(0.0f),
            .shading_normal = point.shading_normal,
            .runtime_flags = 0u,
            .aov = {
                .albedo = make_float3(0.0f),
                .glossy_albedo = make_float3(0.0f),
                .transmission_albedo = make_float3(0.0f),
                .roughness = make_float2(0.0f),
                .normal = point.shading_normal,
                .transparency = make_float3(0.0f)}};
        if (spectral) {
            out.spectral_emission.emplace(make_float4(0.0f));
        }
        return out;
    }
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfacePoint)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceLightQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::VolumeCoefficients)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceEvaluation)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceSample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceClosureTrace)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceSampleTrace)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceAov)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfacePreparation)
