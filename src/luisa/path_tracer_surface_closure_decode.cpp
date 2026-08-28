#include "path_tracer_surface_closure_decode.h"

#include "graph_surface_internal.h"

#include <psycles/compiler/surface_execution_plan.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {


[[nodiscard]] Float read_closure_scalar_or_from(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot closure_operand_slot,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    std::size_t operand_index,
    Float fallback) noexcept {
    const auto address =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            closure_operand_slot)
            .read(
                instruction.y +
                static_cast<std::uint32_t>(operand_index));
    Float result = fallback;
    $if(address != compiler::SurfaceValueAddress::invalid_value) {
        result = read_scalar_dynamic(
            services, point, locals, address);
    };
    return result;
}

[[nodiscard]] Float3 read_closure_vector_or_from(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot closure_operand_slot,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    std::size_t operand_index,
    Float3 fallback) noexcept {
    const auto address =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            closure_operand_slot)
            .read(
                instruction.y +
                static_cast<std::uint32_t>(operand_index));
    Float3 result = fallback;
    $if(address != compiler::SurfaceValueAddress::invalid_value) {
        result = read_vector_dynamic(
            services, point, locals, address);
    };
    return result;
}


[[nodiscard]] TracedClosure decode_surface_closure_impl(
    std::uint32_t static_variant,
    compiler::PrincipledClosureFeatureMask principled_features,
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot closure_operand_slot,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    Float mix_weight) noexcept {
    // Operand provenance is an explicit part of the decoder contract. These
    // scoped adapters keep every closure-family case uniform while forcing all
    // reads through the selected unified stream rather than mutable or ambient
    // buffer state.
    const auto read_closure_scalar_or =
        [closure_operand_slot](
            const SurfaceValueRuntime &value_runtime,
            const ShaderServices &value_services,
            const SurfacePoint &value_point,
            const SurfaceValueLocalsView &value_locals,
            Var<luisa::uint4> value_instruction,
            std::size_t operand_index,
            Float fallback) noexcept {
            return read_closure_scalar_or_from(
                value_runtime, closure_operand_slot, value_services,
                value_point, value_locals, value_instruction, operand_index,
                std::move(fallback));
        };
    const auto read_closure_vector_or =
        [closure_operand_slot](
            const SurfaceValueRuntime &value_runtime,
            const ShaderServices &value_services,
            const SurfacePoint &value_point,
            const SurfaceValueLocalsView &value_locals,
            Var<luisa::uint4> value_instruction,
            std::size_t operand_index,
            Float3 fallback) noexcept {
            return read_closure_vector_or_from(
                value_runtime, closure_operand_slot, value_services,
                value_point, value_locals, value_instruction, operand_index,
                std::move(fallback));
        };
    namespace operand = compiler::surface_closure_operand;
    const auto operation = static_cast<compiler::ClosureOperation>(
        static_variant & compiler::surface_closure_opcode_mask);
    const auto bssrdf_method = static_cast<compiler::BssrdfMethod>(
        (static_variant & compiler::surface_closure_bssrdf_method_mask) >>
        compiler::surface_closure_bssrdf_method_shift);
    const auto coat_normal_linked =
        (static_variant &
         compiler::surface_closure_coat_normal_linked) != 0u;
    const auto preserve_ggx_energy =
        (static_variant &
         compiler::surface_closure_preserve_ggx_energy) != 0u;
    const auto beckmann =
        (static_variant & compiler::surface_closure_beckmann) != 0u;
    const auto anisotropy_enabled =
        (static_variant &
         compiler::surface_closure_microfacet_anisotropy) != 0u;
    const auto thin_film_enabled =
        (static_variant & compiler::surface_closure_thin_film) != 0u;
    const auto hair_tangent_linked =
        (static_variant &
         compiler::surface_closure_hair_tangent_linked) != 0u;

    switch (operation) {
        case compiler::ClosureOperation::diffuse:
        case compiler::ClosureOperation::glossy: {
            const auto glossy =
                operation == compiler::ClosureOperation::glossy;
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::diffuse::color,
                make_float3(0.0f));
            const auto roughness = read_closure_scalar_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::diffuse::roughness,
                0.0f);
            return TracedClosure{
                .operation = operation,
                .weight = bsdf_allocated_weight(color * mix_weight),
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::diffuse::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = roughness,
                .diffuse_roughness = roughness,
                .metallic = glossy ? 1.0f : 0.0f,
                .ior = 1.5f,
                .specular_ior_level = 0.5f,
                .specular_tint = make_float3(1.0f),
                .anisotropy_enabled = glossy && anisotropy_enabled,
                .anisotropy = glossy && anisotropy_enabled
                                  ? read_closure_scalar_or(
                                        runtime,
                                        services,
                                        point,
                                        locals,
                                        instruction,
                                        operand::glossy::anisotropy,
                                        0.0f)
                                  : Float{0.0f},
                .anisotropic_rotation = glossy && anisotropy_enabled
                                            ? read_closure_scalar_or(
                                                  runtime,
                                                  services,
                                                  point,
                                                  locals,
                                                  instruction,
                                                  operand::glossy::rotation,
                                                  0.0f)
                                            : Float{0.0f},
                .tangent = glossy && anisotropy_enabled
                               ? read_closure_vector_or(
                                     runtime,
                                     services,
                                     point,
                                     locals,
                                     instruction,
                                     operand::glossy::tangent,
                                     make_float3(0.0f))
                               : make_float3(0.0f),
                .preserve_ggx_energy =
                    glossy && preserve_ggx_energy,
                .beckmann = glossy && beckmann};
        }
        case compiler::ClosureOperation::metallic_f82:
        case compiler::ClosureOperation::metallic_conductor: {
            const auto f82 =
                operation == compiler::ClosureOperation::metallic_f82;
            const auto clamp_parameter = [f82](Float3 value) noexcept {
                value = max(value, make_float3(0.0f));
                return f82 ? min(value, make_float3(1.0f)) : value;
            };
            const auto vector = [&](std::size_t index,
                                    luisa::float3 fallback) noexcept {
                return read_closure_vector_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    index,
                    Float3{fallback});
            };
            const auto scalar = [&](std::size_t index,
                                    float fallback) noexcept {
                return read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    index,
                    Float{fallback});
            };
            return TracedClosure{
                .operation = operation,
                .weight = make_float3(mix_weight),
                .color = clamp_parameter(vector(
                    operand::metallic::base_ior,
                    luisa::make_float3(0.0f))),
                .normal = safe_normalize(
                    vector(
                        operand::metallic::normal,
                        luisa::make_float3(0.0f)),
                    point.shading_normal),
                .roughness = scalar(
                    operand::metallic::roughness, 0.0f),
                .metallic = 1.0f,
                .ior = 1.0f,
                .specular_tint = clamp_parameter(vector(
                    operand::metallic::edge_tint_k,
                    luisa::make_float3(0.0f))),
                .anisotropy_enabled = anisotropy_enabled,
                .anisotropy = anisotropy_enabled
                                  ? scalar(
                                        operand::metallic::anisotropy,
                                        0.0f)
                                  : Float{0.0f},
                .anisotropic_rotation = anisotropy_enabled
                                            ? scalar(
                                                  operand::metallic::rotation,
                                                  0.0f)
                                            : Float{0.0f},
                .tangent = anisotropy_enabled
                               ? vector(
                                     operand::metallic::tangent,
                                     luisa::make_float3(0.0f))
                               : make_float3(0.0f),
                .thin_film_enabled = thin_film_enabled,
                .thin_film_thickness = thin_film_enabled
                                           ? max(
                                                 scalar(
                                                     operand::metallic::thin_film_thickness,
                                                     0.0f),
                                                 1.0e-5f)
                                           : Float{0.0f},
                .thin_film_ior = thin_film_enabled
                                     ? max(
                                           scalar(
                                               operand::metallic::thin_film_ior,
                                               1.33f),
                                           1.0e-5f)
                                     : Float{0.0f},
                .preserve_ggx_energy = preserve_ggx_energy,
                .beckmann = beckmann};
        }
        case compiler::ClosureOperation::sheen_microfiber:
        case compiler::ClosureOperation::sheen_ashikhmin: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::sheen::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = bsdf_allocated_weight(color * mix_weight),
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::sheen::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::sheen::roughness,
                    0.5f),
                .diffuse_roughness = 0.0f,
                .metallic = 0.0f,
                .ior = 1.0f,
                .specular_ior_level = 0.0f,
                .specular_tint = make_float3(1.0f)};
        }
        case compiler::ClosureOperation::hair_reflection:
        case compiler::ClosureOperation::hair_transmission: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::hair::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = bsdf_allocated_weight(color * mix_weight),
                .color = color,
                .normal = point.shading_normal,
                .roughness = read_closure_scalar_or(
                    runtime, services, point, locals, instruction,
                    operand::hair::roughness_u, 0.1f),
                .diffuse_roughness = read_closure_scalar_or(
                    runtime, services, point, locals, instruction,
                    operand::hair::roughness_v, 1.0f),
                .metallic = 0.0f,
                .ior = 1.0f,
                .specular_ior_level = 0.0f,
                .specular_tint = make_float3(1.0f),
                .tangent = hair_tangent_linked
                               ? read_closure_vector_or(
                                     runtime,
                                     services,
                                     point,
                                     locals,
                                     instruction,
                                     operand::hair::tangent,
                                     make_float3(0.0f))
                               : make_float3(0.0f),
                .hair_tangent_linked = hair_tangent_linked,
                .hair_offset = read_closure_scalar_or(
                    runtime, services, point, locals, instruction,
                    operand::hair::offset, 0.0f)};
        }
        case compiler::ClosureOperation::translucent: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::translucent::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = bsdf_allocated_weight(color * mix_weight),
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::translucent::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = 0.0f,
                .diffuse_roughness = 0.0f,
                .metallic = 0.0f,
                .ior = 1.0f,
                .specular_ior_level = 0.0f,
                .specular_tint = make_float3(1.0f)};
        }
        case compiler::ClosureOperation::principled: {
            const auto scalar = [&](std::size_t index, float fallback) noexcept {
                return read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    index,
                    Float{fallback});
            };
            const auto vector = [&](std::size_t index,
                                    luisa::float3 fallback) noexcept {
                return read_closure_vector_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    index,
                    Float3{fallback});
            };
            const auto emission_color = vector(
                operand::principled::emission_color,
                luisa::make_float3(0.0f));
            const auto emission_strength = scalar(
                operand::principled::emission_strength, 0.0f);
            return TracedClosure{
                .operation = operation,
                .principled_features = principled_features,
                .weight = make_float3(mix_weight),
                .color = vector(
                    operand::principled::color,
                    luisa::make_float3(0.0f)),
                .normal = safe_normalize(
                    vector(
                        operand::principled::normal,
                        luisa::make_float3(0.0f)),
                    point.shading_normal),
                .roughness = scalar(
                    operand::principled::roughness, 0.0f),
                .diffuse_roughness = scalar(
                    operand::principled::diffuse_roughness, 0.0f),
                .subsurface_weight = scalar(
                    operand::principled::subsurface_weight, 0.0f),
                .subsurface_radius = vector(
                    operand::principled::subsurface_radius,
                    luisa::make_float3(0.0f)),
                .subsurface_scale = scalar(
                    operand::principled::subsurface_scale, 0.0f),
                .subsurface_method = bssrdf_method,
                .subsurface_ior = scalar(
                    operand::principled::subsurface_ior, 1.4f),
                .subsurface_anisotropy = scalar(
                    operand::principled::subsurface_anisotropy, 0.0f),
                .transmission_weight = scalar(
                    operand::principled::transmission_weight, 0.0f),
                .metallic = clamp(
                    scalar(operand::principled::metallic, 0.0f),
                    0.0f,
                    1.0f),
                .ior = max(
                    scalar(operand::principled::ior, 1.0f),
                    1.0e-5f),
                .specular_ior_level = max(
                    scalar(
                        operand::principled::specular_ior_level,
                        0.0f),
                    0.0f),
                .specular_tint = max(
                    vector(
                        operand::principled::specular_tint,
                        luisa::make_float3(1.0f)),
                    make_float3(0.0f)),
                .anisotropy_enabled = anisotropy_enabled,
                .anisotropy = anisotropy_enabled
                                  ? scalar(
                                        operand::principled::anisotropic,
                                        0.0f)
                                  : Float{0.0f},
                .anisotropic_rotation = anisotropy_enabled
                                            ? scalar(
                                                  operand::principled::anisotropic_rotation,
                                                  0.0f)
                                            : Float{0.0f},
                .tangent = anisotropy_enabled
                               ? vector(
                                     operand::principled::tangent,
                                     luisa::make_float3(0.0f))
                               : make_float3(0.0f),
                .alpha = scalar(operand::principled::alpha, 1.0f),
                .thin_wall =
                    scalar(operand::principled::thin_wall, 0.0f) != 0.0f,
                .sheen_weight = scalar(
                    operand::principled::sheen_weight, 0.0f),
                .sheen_roughness = scalar(
                    operand::principled::sheen_roughness, 0.5f),
                .sheen_tint = vector(
                    operand::principled::sheen_tint,
                    luisa::make_float3(1.0f)),
                .coat_weight = scalar(
                    operand::principled::coat_weight, 0.0f),
                .coat_roughness = scalar(
                    operand::principled::coat_roughness, 0.03f),
                .coat_ior = scalar(
                    operand::principled::coat_ior, 1.5f),
                .coat_tint = vector(
                    operand::principled::coat_tint,
                    luisa::make_float3(1.0f)),
                .coat_normal = vector(
                    operand::principled::coat_normal,
                    luisa::make_float3(0.0f)),
                .coat_normal_linked = coat_normal_linked,
                .emission = emission_color * emission_strength,
                .thin_film_enabled = thin_film_enabled,
                .thin_film_thickness = thin_film_enabled
                    ? scalar(operand::principled::thin_film_thickness, 0.0f)
                    : Float{0.0f},
                .thin_film_ior = thin_film_enabled
                    ? max(scalar(operand::principled::thin_film_ior, 1.33f),
                          1.0e-5f)
                    : Float{0.0f},
                .preserve_ggx_energy = preserve_ggx_energy,
                .beckmann = false};
        }
        case compiler::ClosureOperation::glass:
        case compiler::ClosureOperation::refraction: {
            const auto glass =
                operation == compiler::ClosureOperation::glass;
            const auto color = max(
                read_closure_vector_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::glass::color,
                    make_float3(0.0f)),
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = glass ? make_float3(mix_weight)
                                : color * mix_weight,
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::glass::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::glass::roughness,
                    0.0f),
                .diffuse_roughness = 0.0f,
                .metallic = 0.0f,
                .ior = max(
                    read_closure_scalar_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::glass::ior,
                        1.0f),
                    1.0e-5f),
                .specular_ior_level = 0.5f,
                .specular_tint = make_float3(1.0f),
                .thin_film_enabled = glass && thin_film_enabled,
                .thin_film_thickness = glass && thin_film_enabled
                    ? read_closure_scalar_or(
                          runtime,
                          services,
                          point,
                          locals,
                          instruction,
                          operand::glass::thin_film_thickness,
                          0.0f)
                    : Float{0.0f},
                .thin_film_ior = glass && thin_film_enabled
                    ? max(read_closure_scalar_or(
                              runtime,
                              services,
                              point,
                              locals,
                              instruction,
                              operand::glass::thin_film_ior,
                              1.33f),
                          1.0e-5f)
                    : Float{0.0f},
                .preserve_ggx_energy =
                    glass && preserve_ggx_energy,
                .beckmann = beckmann};
        }
        case compiler::ClosureOperation::emission: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::emission::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = color *
                          read_closure_scalar_or(
                              runtime,
                              services,
                              point,
                              locals,
                              instruction,
                              operand::emission::strength,
                              0.0f) *
                          mix_weight,
                .color = color,
                .normal = make_float3(0.0f, 0.0f, 1.0f),
                .roughness = 0.0f,
                .metallic = 0.0f,
                .ior = 1.0f};
        }
        case compiler::ClosureOperation::transparent: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::transparent::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = color * mix_weight,
                .color = color,
                .normal = point.shading_normal,
                .roughness = 0.0f,
                .metallic = 0.0f,
                .ior = 1.0f};
        }
        case compiler::ClosureOperation::subsurface: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::subsurface::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = color * mix_weight,
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::subsurface::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::subsurface::roughness,
                    0.0f),
                .subsurface_radius = max(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::subsurface::radius,
                        make_float3(0.0f)),
                    make_float3(0.0f)),
                .subsurface_scale = max(
                    read_closure_scalar_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::subsurface::scale,
                        0.0f),
                    0.0f),
                .subsurface_method = bssrdf_method,
                .subsurface_ior = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::subsurface::ior,
                    1.4f),
                .subsurface_anisotropy = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::subsurface::anisotropy,
                    0.0f),
                .ior = 1.0f};
        }
        case compiler::ClosureOperation::null_closure:
        case compiler::ClosureOperation::add:
        case compiler::ClosureOperation::mix:
            break;
    }
    std::abort();
}


} // namespace

TracedClosure decode_surface_closure(
    std::uint32_t static_variant,
    compiler::PrincipledClosureFeatureMask principled_features,
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot closure_operand_slot,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    Float mix_weight) noexcept {
    return decode_surface_closure_impl(
        static_variant, principled_features, runtime, closure_operand_slot,
        services, point, locals, instruction, std::move(mix_weight));
}


} // namespace psycles::luisa_backend::detail
