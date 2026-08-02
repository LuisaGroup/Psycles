#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface.h> through the Psycles::luisa target."
#endif

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <psycles/contract/cycles_abi.h>
#include <psycles/contract/surface.h>

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/polymorphic.h>
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
using luisa::compute::Polymorphic;
using luisa::compute::UInt;
using luisa::compute::ULong;
using luisa::compute::make_float2;
using luisa::compute::make_float3;
using luisa::compute::make_float4;

using contract::SurfaceEvent;
using contract::TransportMode;
using contract::event_diffuse;
using contract::event_glossy;
using contract::event_none;
using contract::event_reflection;
using contract::event_singular;
using contract::event_subsurface;
using contract::event_transmission;
using contract::event_transparent;

struct SurfaceCapabilities {
    std::uint64_t features{};
    bool may_emit{false};
    // Cycles' SD_HAS_CONSTANT_EMISSION scheduling property. This remains
    // true for a non-emitting graph (whose constant value is zero) and false
    // only when full shading-point evaluation must be deferred.
    bool emission_is_constant{true};
    bool may_be_transparent{false};
    bool may_have_subsurface{false};
    bool may_have_volume{false};
};

struct SurfacePoint {
    Float3 position;
    Float3 object_position;
    Float3 object_location;
    Float3 generated;
    Float3 geometric_normal;
    Float3 shading_normal;
    // Cycles tangent-space Normal Map is constructed in object space from
    // Blender's evaluated MikkTSpace attributes and only then transformed as
    // a normal. The interpolated shading normal remains raw here: the Normal
    // Map node owns Cycles' normalization-before-frame contract. Keeping
    // these values explicit avoids an invalid world-space shortcut under
    // non-uniform instance transforms.
    Float3 object_shading_normal;
    Float3 object_tangent;
    Float tangent_sign;
    Float3 normal_to_world_x;
    Float3 normal_to_world_y;
    Float3 normal_to_world_z;
    Float3 dpdu;
    Float3 dpdv;
    Float3 dPdx;
    Float3 dPdy;
    Float3 object_dPdx;
    Float3 object_dPdy;
    Float3 generated_dx;
    Float3 generated_dy;
    Float3 incoming;
    Float2 uv;
    Float2 uv_dx;
    Float2 uv_dy;
    UInt geometry_index;
    Float2 barycentric;
    Float2 barycentric_dx;
    Float2 barycentric_dy;
    UInt instance_id;
    UInt primitive_id;
    // Runtime parameter storage base for the material bound to this hit.
    // Materials with an identical Cycles graph structure share one
    // GraphSurface AST and differ only by this block.
    UInt parameter_block;
    Float object_random;
    UInt particle_index;
    Float random_per_island;
    UInt ray_visibility;
    UInt ray_events;
    UInt ray_depth;
    UInt diffuse_depth;
    UInt glossy_depth;
    UInt transparent_depth;
    UInt transmission_depth;
    Float ray_length;
    Float time;
    // Cycles SD_USE_BUMP_MAP_CORRECTION is material-owned runtime state.
    // Keep it on the shading point because one GraphSurface AST may be
    // shared by materials with different policies.
    Bool use_bump_map_correction;
    Bool back_facing;
};

struct SurfaceQuery {
    UInt lobe_mask;
    UInt transport_mode;
    // Cycles Filter Glossy widens microfacet alpha after closure setup. Zero
    // leaves the material closure unchanged.
    Float glossy_filter_roughness;
};

// Cycles sampled-light visibility is deliberately separate from the path
// lobe mask. Shader exclude flags remove closure contributions, but every
// otherwise eligible closure remains in the one-sample-model PDF.
struct SurfaceLightQuery {
    SurfaceQuery surface;
    UInt shader_flags;
};

struct ShaderAttribute {
    Float4 value;
    Bool found;

    [[nodiscard]] static ShaderAttribute missing() noexcept {
        return {
            .value = make_float4(0.0f),
            .found = false};
    }
};

// Cycles evaluates a volume graph in two distinct contexts. Shadow and
// extinction-only queries suppress emission, while full volume shading keeps
// it. Object-space grids additionally scale coefficients by the instance's
// volume-density correction.
struct VolumeQuery {
    Float object_density;
    Bool evaluate_emission;
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

// Host-stage sink for raw phase closures. A GraphSurface invokes this while
// Luisa records the shader AST; implementations may retain closures in device
// local storage, merge them across stacked media, or expose them to a focused
// diagnostic kernel. The closure parameters are never averaged or pre-baked.
class VolumePhaseCollector {

  public:
    virtual ~VolumePhaseCollector() noexcept = default;
    virtual void add(
        const cycles_volume_phase::Closure &phase,
        Float3 weight) noexcept = 0;
};

struct SurfaceEvaluation {
    Float3 f;
    Float pdf;
    Float3 diffuse_f;
    Float3 glossy_f;
    Float diffuse_pdf;
    UInt events;

    [[nodiscard]] static SurfaceEvaluation zero() noexcept {
        return {
            .f = make_float3(0.0f),
            .pdf = 0.0f,
            .diffuse_f = make_float3(0.0f),
            .glossy_f = make_float3(0.0f),
            .diffuse_pdf = 0.0f,
            .events = static_cast<std::uint32_t>(event_none)};
    }
};

struct SurfaceSample {
    SurfaceEvaluation evaluation;
    Float3 wi;
    Float eta;
    Float2 roughness;
    UInt runtime_flags;
    Bool valid;

    [[nodiscard]] static SurfaceSample zero() noexcept {
        return {
            .evaluation = SurfaceEvaluation::zero(),
            .wi = make_float3(0.0f, 0.0f, 1.0f),
            .eta = 1.0f,
            .roughness = make_float2(0.0f),
            .runtime_flags = 0u,
            .valid = false};
    }
};

// Diagnostic view of one entry in the post-shader closure array. `count` is
// repeated so a single runtime-indexed callable can expose the fixed-capacity
// trace without returning a backend-specific aggregate array.
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

// Trace-only extension of SurfaceSample. Normal render kernels call `sample`
// and retain the compact production ABI; differential kernels call
// `sample_trace` to observe the exact closure selection without reimplementing
// it in the integrator.
struct SurfaceSampleTrace {
    SurfaceSample sample;
    UInt closure_index;
    UInt closure_type;
    Float closure_sample_weight;
    Float selection_rescaled;
    Float3 closure_weight;
    Float3 closure_normal;
    Bool closure_valid;

    [[nodiscard]] static SurfaceSampleTrace zero() noexcept {
        return {
            .sample = SurfaceSample::zero(),
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

class SurfaceParameterServices {

public:
    virtual ~SurfaceParameterServices() noexcept = default;

    [[nodiscard]] virtual Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept = 0;

    [[nodiscard]] virtual Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept = 0;
};

class ShaderServices : public SurfaceParameterServices {

public:
    ~ShaderServices() noexcept override = default;

    [[nodiscard]] virtual Float4 texture_2d(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        Expr<luisa::float2> d_uv_dx,
        Expr<luisa::float2> d_uv_dy,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept = 0;

    [[nodiscard]] virtual ShaderAttribute attribute(
        Expr<std::uint64_t> attribute_id,
        const SurfacePoint &point) const noexcept = 0;

    // Versioned Cycles compatibility data. The index addresses the
    // contiguous Blender BSDF table buffer; interpolation and table shape
    // remain explicit in the Luisa shader so all backends execute the same
    // semantics.
    [[nodiscard]] virtual Float cycles_bsdf_data(
        Expr<std::uint32_t> index) const noexcept = 0;

    [[nodiscard]] virtual Float3 xyz_to_rgb(
        Expr<luisa::float3> xyz) const noexcept = 0;

    [[nodiscard]] virtual Float3 rec709_to_rgb(
        Expr<luisa::float3> rec709) const noexcept = 0;

    // Cycles Nishita is a precomputed 512x128 spectral LUT, not an analytic
    // color approximation. The host/JIT-stage sky index identifies the LUT
    // associated with this material parameter block; direction-dependent
    // sampling and the solar disc remain Luisa expressions.
    [[nodiscard]] virtual Float3 nishita_sky(
        Expr<std::uint32_t> block,
        std::uint32_t sky_index,
        Expr<luisa::float3> direction,
        Expr<float> sun_elevation,
        Expr<float> sun_rotation,
        Expr<float> angular_diameter,
        Expr<float> sun_intensity) const noexcept = 0;
};

class Surface {

public:
    virtual ~Surface() noexcept = default;

    [[nodiscard]] virtual SurfaceCapabilities capabilities() const noexcept = 0;

    // Runtime ShaderData flags produced by closure setup. This is separate
    // from sampling so integrators can apply Cycles' pre-NEE capability gate
    // without selecting a BSDF closure early.
    [[nodiscard]] virtual UInt runtime_flags(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<float>) const noexcept {
        return 0u;
    }

    [[nodiscard]] virtual SurfaceEvaluation evaluate(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept = 0;

    [[nodiscard]] virtual SurfaceEvaluation evaluate_light(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept = 0;

    [[nodiscard]] virtual SurfaceSample sample(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept = 0;

    [[nodiscard]] virtual SurfaceClosureTrace closure_trace(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<std::uint32_t> requested_index) const noexcept {
        return SurfaceClosureTrace::zero(requested_index);
    }

    [[nodiscard]] virtual SurfaceSampleTrace sample_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceSampleTrace::zero();
        result.sample = sample(
            services,
            point,
            u_lobe,
            u_direction,
            query);
        return result;
    }

    // Device-side evaluation of Cycles' constant-emission proof. The narrow
    // services type makes shading-point, texture, attribute and BSDF-table
    // access unavailable while Luisa records this AST.
    [[nodiscard]] virtual Float3 constant_emission(
        const SurfaceParameterServices &,
        Expr<std::uint32_t>) const noexcept {
        return make_float3(0.0f);
    }

    [[nodiscard]] virtual Float3 emission(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<luisa::float3>,
        Expr<bool>) const noexcept {
        return make_float3(0.0f);
    }

    [[nodiscard]] virtual Float3 transparent_extinction(
        const ShaderServices &,
        const SurfacePoint &) const noexcept {
        return make_float3(0.0f);
    }

    // Combined volume evaluation is the host-stage semantic boundary.
    // Passing a collector emits raw phase closures from the same traced graph
    // that produced the coefficients, so the path kernel never evaluates a
    // volume shader twice.
    [[nodiscard]] virtual VolumeCoefficients evaluate_volume(
        const ShaderServices &,
        const SurfacePoint &,
        const VolumeQuery &,
        VolumePhaseCollector *) const noexcept {
        return VolumeCoefficients::zero();
    }

    // Final Cycles sd->N after shader bump evaluation. This is distinct from
    // closure-specific Normal inputs and is consumed by shadow-terminator
    // geometry offset.
    [[nodiscard]] virtual Float3 shading_normal(
        const ShaderServices &,
        const SurfacePoint &point) const noexcept {
        return point.shading_normal;
    }

    [[nodiscard]] virtual SurfaceAov aov(
        const ShaderServices &,
        const SurfacePoint &point) const noexcept {
        return {
            .albedo = make_float3(0.0f),
            .glossy_albedo = make_float3(0.0f),
            .transmission_albedo = make_float3(0.0f),
            .roughness = make_float2(0.0f),
            .normal = point.shading_normal,
            .transparency = make_float3(0.0f)};
    }
};

class SurfaceDispatch {

private:
    // This runtime-tag dispatch is a semantic boundary, not a temporary
    // fallback. Scene-specific fusion belongs to a later compiler/IR pass.
    Polymorphic<Surface> _surfaces;

public:
    [[nodiscard]] bool empty() const noexcept { return _surfaces.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return _surfaces.size(); }
    [[nodiscard]] const Surface *implementation(std::size_t index) const noexcept {
        return _surfaces.impl(index);
    }
    [[nodiscard]] SurfaceCapabilities capabilities(std::size_t index) const noexcept {
        return implementation(index)->capabilities();
    }

    template<typename Implementation, typename... Args>
        requires std::derived_from<Implementation, Surface>
    [[nodiscard]] std::uint32_t create(Args &&...args) noexcept {
        return _surfaces.template create<Implementation>(
            std::forward<Args>(args)...);
    }

    template<typename Tag, typename Function>
    void dispatch(Tag &&tag, Function &&function) const noexcept {
        _surfaces.dispatch(
            std::forward<Tag>(tag),
            std::forward<Function>(function));
    }

    [[nodiscard]] SurfaceEvaluation evaluate(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceEvaluation::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->evaluate(
                services, point, outgoing, query);
        });
        return result;
    }

    [[nodiscard]] UInt runtime_flags(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> glossy_filter_roughness) const noexcept {
        UInt result = 0u;
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->runtime_flags(
                services,
                point,
                glossy_filter_roughness);
        });
        return result;
    }

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept {
        auto result = SurfaceEvaluation::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->evaluate_light(
                services, point, outgoing, query);
        });
        return result;
    }

    [[nodiscard]] SurfaceSample sample(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceSample::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->sample(
                services, point, u_lobe, u_direction, query);
        });
        return result;
    }

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<std::uint32_t> requested_index) const noexcept {
        auto result =
            SurfaceClosureTrace::zero(requested_index);
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->closure_trace(
                services,
                point,
                requested_index);
        });
        return result;
    }

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceSampleTrace::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->sample_trace(
                services,
                point,
                u_lobe,
                u_direction,
                query);
        });
        return result;
    }

    [[nodiscard]] Float3 emission(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> wo,
        Expr<bool> reflective_caustics) const noexcept {
        Float3 result = make_float3(0.0f);
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->emission(
                services,
                point,
                wo,
                reflective_caustics);
        });
        return result;
    }

    [[nodiscard]] Float3 constant_emission(
        Expr<std::uint32_t> tag,
        const SurfaceParameterServices &services,
        Expr<std::uint32_t> parameter_block) const noexcept {
        Float3 result = make_float3(0.0f);
        luisa::vector<luisa::uint> constant_tags;
        constant_tags.reserve(_surfaces.size());
        for (auto index = std::size_t{0u};
             index < _surfaces.size();
             ++index) {
            if (_surfaces.impl(index)
                    ->capabilities()
                    .emission_is_constant) {
                constant_tags.emplace_back(
                    static_cast<luisa::uint>(index));
            }
        }
        if (!constant_tags.empty()) {
            _surfaces.dispatch_group_with_default(
                tag,
                constant_tags,
                [&](const Surface *surface) noexcept {
                    result = surface->constant_emission(
                        services,
                        parameter_block);
                },
                []() noexcept {});
        }
        return result;
    }

    [[nodiscard]] Float3 transparent_extinction(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept {
        Float3 result = make_float3(0.0f);
        // Transparency is a structural property of the original closure
        // program. Restrict the JIT dispatch to programs which can actually
        // produce a Transparent closure: opaque programs retain their raw
        // closures for every other surface operation, but do not inflate the
        // ray-query callback with unreachable shader code.
        luisa::vector<luisa::uint> transparent_tags;
        transparent_tags.reserve(_surfaces.size());
        for (auto index = std::size_t{0u};
             index < _surfaces.size();
             ++index) {
            if (_surfaces.impl(index)
                    ->capabilities()
                    .may_be_transparent) {
                transparent_tags.emplace_back(
                    static_cast<luisa::uint>(index));
            }
        }
        if (!transparent_tags.empty()) {
            _surfaces.dispatch_group_with_default(
                tag,
                transparent_tags,
                [&](const Surface *surface) noexcept {
                    result =
                        surface->transparent_extinction(
                            services, point);
                },
                []() noexcept {});
        }
        return result;
    }

    [[nodiscard]] VolumeCoefficients evaluate_volume(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query,
        VolumePhaseCollector *collector) const noexcept {
        auto result = VolumeCoefficients::zero();
        luisa::vector<luisa::uint> volume_tags;
        volume_tags.reserve(_surfaces.size());
        for (auto index = std::size_t{0u};
             index < _surfaces.size();
             ++index) {
            if (_surfaces.impl(index)
                    ->capabilities()
                    .may_have_volume) {
                volume_tags.emplace_back(
                    static_cast<luisa::uint>(index));
            }
        }
        if (!volume_tags.empty()) {
            _surfaces.dispatch_group_with_default(
                tag,
                volume_tags,
                [&](const Surface *surface) noexcept {
                    result = surface->evaluate_volume(
                        services,
                        point,
                        query,
                        collector);
                },
                []() noexcept {});
        }
        return result;
    }

    [[nodiscard]] VolumeCoefficients volume_coefficients(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query) const noexcept {
        return evaluate_volume(
            tag,
            services,
            point,
            query,
            nullptr);
    }

    void volume_phases(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query,
        VolumePhaseCollector &collector) const noexcept {
        static_cast<void>(
            evaluate_volume(
                tag,
                services,
                point,
                query,
                &collector));
    }

    [[nodiscard]] Float3 shading_normal(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept {
        Float3 result = point.shading_normal;
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->shading_normal(services, point);
        });
        return result;
    }

    [[nodiscard]] SurfaceAov aov(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept {
        auto result = SurfaceAov{
            .albedo = make_float3(0.0f),
            .glossy_albedo = make_float3(0.0f),
            .transmission_albedo = make_float3(0.0f),
            .roughness = make_float2(0.0f),
            .normal = point.shading_normal,
            .transparency = make_float3(0.0f)};
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->aov(services, point);
        });
        return result;
    }
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfacePoint)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceLightQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::ShaderAttribute)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::VolumeQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::VolumeCoefficients)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceEvaluation)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceSample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceClosureTrace)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceSampleTrace)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceAov)
