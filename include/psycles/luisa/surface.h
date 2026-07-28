#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface.h> through the Psycles::luisa target."
#endif

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <psycles/contract/surface.h>

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/polymorphic.h>
#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend {

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
    bool may_be_transparent{false};
    bool may_have_subsurface{false};
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
    // a normal. Keeping these values explicit avoids an invalid world-space
    // shortcut under non-uniform instance transforms.
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
    Bool back_facing;
};

struct SurfaceQuery {
    UInt lobe_mask;
    UInt transport_mode;
};

struct SurfaceEvaluation {
    Float3 f;
    Float pdf;
    Float3 diffuse_f;
    Float diffuse_pdf;
    UInt events;

    [[nodiscard]] static SurfaceEvaluation zero() noexcept {
        return {
            .f = make_float3(0.0f),
            .pdf = 0.0f,
            .diffuse_f = make_float3(0.0f),
            .diffuse_pdf = 0.0f,
            .events = static_cast<std::uint32_t>(event_none)};
    }
};

struct SurfaceSample {
    SurfaceEvaluation evaluation;
    Float3 wi;
    Float eta;
    Float2 roughness;
    Bool valid;

    [[nodiscard]] static SurfaceSample zero() noexcept {
        return {
            .evaluation = SurfaceEvaluation::zero(),
            .wi = make_float3(0.0f, 0.0f, 1.0f),
            .eta = 1.0f,
            .roughness = make_float2(0.0f),
            .valid = false};
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

class ShaderServices {

public:
    virtual ~ShaderServices() noexcept = default;

    [[nodiscard]] virtual Float4 texture_2d(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        Expr<luisa::float2> d_uv_dx,
        Expr<luisa::float2> d_uv_dy,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept = 0;

    [[nodiscard]] virtual Float4 attribute(
        Expr<std::uint64_t> attribute_id,
        const SurfacePoint &point) const noexcept = 0;

    // The block and slot are host/JIT-stage constants. Implementations emit
    // runtime storage reads while the material graph is being traced.
    [[nodiscard]] virtual Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept = 0;

    [[nodiscard]] virtual Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept = 0;

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

    [[nodiscard]] virtual SurfaceEvaluation evaluate(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept = 0;

    [[nodiscard]] virtual SurfaceSample sample(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept = 0;

    [[nodiscard]] virtual Float3 emission(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<luisa::float3>) const noexcept {
        return make_float3(0.0f);
    }

    [[nodiscard]] virtual Float3 transparent_extinction(
        const ShaderServices &,
        const SurfacePoint &) const noexcept {
        return make_float3(0.0f);
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

    [[nodiscard]] Float3 emission(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> wo) const noexcept {
        Float3 result = make_float3(0.0f);
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->emission(services, point, wo);
        });
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
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceEvaluation)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceSample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceAov)
