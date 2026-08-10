#include "path_tracer_surface_closure_setup.h"

#include "principled_base_component.h"

namespace psycles::luisa_backend::detail {
namespace {

using namespace luisa::compute;

class CyclesTableShaderServices final : public ShaderServices {

private:
    const BufferFloat &_cycles_bsdf_tables;

public:
    explicit CyclesTableShaderServices(
        const BufferFloat &cycles_bsdf_tables) noexcept
        : _cycles_bsdf_tables{cycles_bsdf_tables} {}

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t>,
        Expr<std::uint32_t>) const noexcept override {
        return 0.0f;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t>,
        Expr<std::uint32_t>) const noexcept override {
        return make_float3(0.0f);
    }

    [[nodiscard]] ULong parameter_uint64(
        Expr<std::uint32_t>,
        Expr<std::uint32_t>) const noexcept override {
        return 0u;
    }

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<luisa::ulong>,
        const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t> index) const noexcept override {
        return _cycles_bsdf_tables->read(index);
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

[[nodiscard]] PrincipledDielectricSetupCallable
make_principled_dielectric_setup_callable(
    bool preserve_ggx_energy) noexcept {
    PrincipledDielectricSetupCallable callable =
        [preserve_ggx_energy](
            BufferFloat cycles_bsdf_tables,
            Var<PrincipledDielectricSetupInputCall> packed_input,
            Bool reflective_caustics) noexcept {
            const CyclesTableShaderServices services{
                cycles_bsdf_tables};
            const auto populated = populate_principled_dielectric(
                {.lower_weight = packed_input.lower_weight,
                 .normal = packed_input.normal,
                 .incoming = packed_input.incoming,
                 .surface_shading_normal =
                     packed_input.surface_shading_normal,
                 .surface_geometric_normal =
                     packed_input.surface_geometric_normal,
                 .roughness = packed_input.roughness,
                 .ior = packed_input.ior,
                 .specular_ior_level =
                     packed_input.specular_ior_level,
                 .specular_tint = packed_input.specular_tint,
                 .use_bump_map_correction =
                     packed_input.use_bump_map_correction,
                 .preserve_ggx_energy =
                     preserve_ggx_energy});
            const auto setup = setup_principled_dielectric(
                services, populated, reflective_caustics);
            Var<PrincipledDielectricSetupCall> result;
            result.weight = setup.weight;
            result.allocation_weight =
                setup.allocation_weight;
            result.sample_weight = setup.sample_weight;
            result.albedo = setup.albedo;
            result.normal = setup.normal;
            result.color = setup.color;
            result.ior = setup.ior;
            result.evaluation_scale =
                setup.evaluation_scale;
            result.lower_weight = setup.lower_weight;
            return result;
        };
    callable.set_name(
        preserve_ggx_energy
            ? "principled_dielectric_setup_energy"
            : "principled_dielectric_setup");
    return callable;
}

}// namespace

SurfaceClosureSetupCallables
make_surface_closure_setup_callables() noexcept {
    return {
        .principled_dielectric =
            make_principled_dielectric_setup_callable(false),
        .principled_dielectric_preserve_energy =
            make_principled_dielectric_setup_callable(true)};
}

CallableSurfaceClosureSetupProvider::
    CallableSurfaceClosureSetupProvider(
        const BufferFloat &cycles_bsdf_tables,
        const SurfaceClosureSetupCallables &callables) noexcept
    : _cycles_bsdf_tables{cycles_bsdf_tables},
      _callables{callables} {}

PrincipledDielectricSetupResult
CallableSurfaceClosureSetupProvider::principled_dielectric(
    const PrincipledDielectricSetupInput &input,
    Expr<bool> reflective_caustics) const noexcept {
    const auto &callable = input.preserve_ggx_energy
                               ? _callables
                                     .principled_dielectric_preserve_energy
                               : _callables.principled_dielectric;
    Var<PrincipledDielectricSetupInputCall> packed_input;
    packed_input.lower_weight = input.lower_weight;
    packed_input.normal = input.normal;
    packed_input.incoming = input.incoming;
    packed_input.surface_shading_normal =
        input.surface_shading_normal;
    packed_input.surface_geometric_normal =
        input.surface_geometric_normal;
    packed_input.roughness = input.roughness;
    packed_input.ior = input.ior;
    packed_input.specular_ior_level = input.specular_ior_level;
    packed_input.specular_tint = input.specular_tint;
    packed_input.use_bump_map_correction =
        input.use_bump_map_correction;
    const auto result = callable(
        _cycles_bsdf_tables, packed_input, reflective_caustics);
    return {
        .weight = result.weight,
        .allocation_weight = result.allocation_weight,
        .sample_weight = result.sample_weight,
        .albedo = result.albedo,
        .normal = result.normal,
        .color = result.color,
        .ior = result.ior,
        .evaluation_scale = result.evaluation_scale,
        .lower_weight = result.lower_weight};
}

}// namespace psycles::luisa_backend::detail
