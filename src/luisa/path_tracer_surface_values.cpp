#include "path_tracer_surface_values.h"
#include "path_tracer_surface_value_program.h"

#include "graph_surface_internal.h"
#include "path_tracer_attribute_lookup.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surface_closure_setup.h"
#include "path_tracer_surface_execution_domain.h"
#include "path_tracer_surfaces.h"
#include "path_tracer_texture_sampling.h"
#include "principled_layer_component.h"
#include "surface_bump.h"
#include "surface_preparation_accumulator.h"

#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/cycles_closure.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/dsl/local.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {


[[nodiscard]] Float read_closure_scalar_or(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    std::size_t operand_index,
    Float fallback) noexcept {
    const auto address =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            SurfaceValueRuntimeBufferSlot::closure_operand)
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

[[nodiscard]] Float3 read_closure_vector_or(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    std::size_t operand_index,
    Float3 fallback) noexcept {
    const auto address =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            SurfaceValueRuntimeBufferSlot::closure_operand)
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

[[nodiscard]] TracedClosure decode_surface_closure(
    std::uint32_t static_variant,
    compiler::PrincipledClosureFeatureMask scene_features,
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    Float mix_weight) noexcept {
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
                .preserve_ggx_energy =
                    glossy && preserve_ggx_energy,
                .beckmann = glossy && beckmann};
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
                .principled_features = scene_features,
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

template<bool FilterLeafPrefix, typename Visitor>
void traverse_surface_closure_program(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt instruction_begin,
    UInt instruction_end,
    UInt leaf_begin,
    Visitor &&visit_leaf) noexcept {
    // The host lowerer proves a strict topological weight program and exact
    // read-before-write interval coloring. Slot zero is a harmless dummy for
    // Mix-free scenes; the root sentinel is projected to that safe index but
    // semantically reads the implicit constant one.
    luisa::compute::Local<float> weights{
        std::max(runtime.maximum_closure_mix_slots, 1u)};
    const auto read_weight = [&](UInt slot) noexcept {
        const auto stored =
            slot != compiler::surface_closure_root_weight_slot;
        const auto safe_slot = select(0u, slot, stored);
        return select(1.0f, weights.read(safe_slot), stored);
    };
    UInt instruction_index = instruction_begin;
    $while(instruction_index < instruction_end) {
        Var<luisa::uint4> instruction =
            surface_value_runtime_buffer<luisa::uint4>(
                runtime,
                SurfaceValueRuntimeBufferSlot::closure_instruction)
                .read(instruction_index);
        const auto kind =
            (instruction.x &
             compiler::surface_closure_instruction_kind_mask) >>
            compiler::surface_closure_instruction_kind_shift;
        $if(kind != static_cast<std::uint32_t>(
                        compiler::SurfaceClosureInstructionKind::leaf)) {
            const auto parent_weight = read_weight(instruction.z);
            const auto factor = clamp(
                read_scalar_dynamic(
                    services, point, locals, instruction.y),
                0.0f,
                1.0f);
            const auto output =
                instruction.w &
                compiler::surface_closure_weight_slot_mask;
            $if(kind == static_cast<std::uint32_t>(
                            compiler::SurfaceClosureInstructionKind::mix_both)) {
                weights.write(output, parent_weight * (1.0f - factor));
                weights.write(instruction.w >> 16u, parent_weight * factor);
            }
            $elif(kind == static_cast<std::uint32_t>(
                              compiler::SurfaceClosureInstructionKind::mix_left)) {
                weights.write(output, parent_weight * (1.0f - factor));
            }
            $else {
                weights.write(output, parent_weight * factor);
            };
        }
        $else {
            const auto weight = read_weight(instruction.z);
            // Preserve the former `!(weight <= 0)` behavior for NaN exactly:
            // a NaN factor is not silently converted into an unreachable leaf.
            $if(!(weight <= 0.0f)) {
                if constexpr (FilterLeafPrefix) {
                    $if(instruction_index >= leaf_begin) {
                        visit_leaf(instruction, weight, instruction_index);
                    };
                } else {
                    visit_leaf(instruction, weight, instruction_index);
                }
            };
        };
        instruction_index += 1u;
    };
}

template<bool FilterLeafPrefix = false, typename Visitor>
void emit_surface_closure_program(
    const SurfaceValueRuntime &runtime,
    const SurfaceClosureProgramDomainView &domain,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt instruction_begin,
    UInt instruction_end,
    UInt leaf_begin,
    Visitor &&visit) noexcept {
    traverse_surface_closure_program<FilterLeafPrefix>(
        runtime,
        services,
        point,
        locals,
        instruction_begin,
        instruction_end,
        leaf_begin,
        [&](Var<luisa::uint4> instruction,
            Float mix_weight,
            UInt instruction_index) noexcept {
        const auto static_variant =
            instruction.x &
            compiler::surface_closure_static_variant_mask;
        const auto endpoints =
            (instruction.x &
             compiler::surface_closure_endpoint_mask) >>
            compiler::surface_closure_endpoint_shift;
        luisa::compute::detail::SwitchStmtBuilder{static_variant} % [&] {
            for (const auto variant :
                 domain.static_variants) {
                luisa::compute::detail::SwitchCaseStmtBuilder{variant} %
                    [&, variant] {
                        const auto closure = decode_surface_closure(
                            variant,
                            domain.principled_features,
                            runtime,
                            services,
                            point,
                            locals,
                            instruction,
                            mix_weight);
                        visit(closure, endpoints, instruction_index);
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid compact surface closure variant");
            };
        };
        });
}

void accumulate_surface_emission(
    compiler::PrincipledClosureFeatureMask principled_features,
    const PrincipledLayerComponent &principled_layers,
    const TracedClosure &raw,
    UInt endpoints,
    Expr<bool> reflective_caustics,
    Float3 &emission) noexcept {
    const auto emission_endpoint =
        (endpoints & compiler::surface_closure_endpoint_bit(
                         compiler::SurfaceClosureEndpoint::emission)) != 0u;
    if (raw.operation == compiler::ClosureOperation::emission) {
        $if(emission_endpoint) { emission += raw.weight; };
    } else if (raw.operation == compiler::ClosureOperation::principled &&
               (principled_features &
                compiler::principled_closure_feature_bit(
                    compiler::PrincipledClosureFeature::emission)) != 0u) {
        const auto contribution =
            principled_layers
                .evaluate_emission(raw, raw.principled_features,
                                   reflective_caustics)
                .radiance;
        $if(emission_endpoint) { emission += contribution; };
    }
}

[[nodiscard]] SurfaceClosureRecord merged_transparent_closure(
    const SurfacePoint &point,
    Float3 weight,
    Float sample_weight_value) noexcept {
    return canonical_surface_closure(TracedClosure{
        .operation = compiler::ClosureOperation::transparent,
        .weight = weight,
        .allocation_weight = sample_weight_value,
        .sample_weight = sample_weight_value,
        .setup_valid = true,
        .albedo = weight,
        .color = make_float3(1.0f),
        .normal = point.shading_normal,
        .roughness = 0.0f,
        .ior = 1.0f,
        .evaluation_scale = make_float3(1.0f)});
}

template<typename PhysicalClosureSink>
[[nodiscard]] SurfacePopulation execute_surface_closure_program(
    const SurfaceValueRuntime &runtime,
    SurfaceClosureProgramDomain domain,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt program,
    Expr<bool> emission_reflective_caustics,
    Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics,
    PhysicalClosureSink &&emit_physical) noexcept {
    const auto range = surface_value_runtime_buffer<luisa::uint4>(
                           runtime,
                           SurfaceValueRuntimeBufferSlot::program)
                           .read(program);
    const auto closure_begin = range.z;
    const auto closure_end = range.z + range.w;
    const auto domain_view = surface_closure_program_domain(runtime, domain);
    const PrincipledLayerComponent principled_layers{services, point};
    Float3 emission = make_float3(0.0f);
    Float3 transparent_weight = make_float3(0.0f);
    Float transparent_sample_weight = 0.0f;
    Bool transparent_pending = false;
    UInt replay_begin = closure_end;

    emit_surface_closure_program(
        runtime,
        domain_view,
        services,
        point,
        locals,
        closure_begin,
        closure_end,
        closure_begin,
        [&](const TracedClosure &raw,
            UInt endpoints,
            UInt instruction_index) noexcept {
            accumulate_surface_emission(
                domain_view.principled_features,
                principled_layers,
                raw,
                endpoints,
                emission_reflective_caustics,
                emission);

            const auto physical_endpoint =
                (endpoints & compiler::surface_closure_endpoint_bit(
                                 compiler::SurfaceClosureEndpoint::physical)) !=
                0u;
            $if(physical_endpoint) {
                expand_physical_surface_closure(
                    services,
                    point,
                    raw,
                    reflective_caustics,
                    refractive_caustics,
                    [&](const TracedClosure &physical) noexcept {
                        if (physical.operation ==
                            compiler::ClosureOperation::transparent) {
                            transparent_weight += physical.weight;
                            transparent_sample_weight +=
                                physical.sample_weight;
                            const auto allocated =
                                physical.sample_weight >=
                                cycles_closure::closure_weight_cutoff;
                            $if(!transparent_pending & allocated) {
                                replay_begin = instruction_index;
                            };
                            transparent_pending |= allocated;
                            return;
                        }
                        $if(!transparent_pending) {
                            emit_physical(
                                canonical_surface_closure(physical));
                        };
                    });
            };
        });

    $if(transparent_pending) {
        emit_physical(merged_transparent_closure(
            point,
            transparent_weight,
            transparent_sample_weight));
    };

    // The first pass has already evaluated values and found the exact raw
    // instruction containing the first allocated transparent output. Replaying
    // only from that leaf reconstructs the non-transparent suffix in source
    // order without retaining a per-thread physical closure arena. Pure setup
    // is deterministic, so the retained sequence and left-fold order are
    // identical to the expanded route.
    Bool reached_first_transparent = false;
    $if(transparent_pending) {
        emit_surface_closure_program<true>(
            runtime,
            domain_view,
            services,
            point,
            locals,
            closure_begin,
            closure_end,
            replay_begin,
            [&](const TracedClosure &raw,
                UInt endpoints,
                UInt) noexcept {
                const auto physical_endpoint =
                    (endpoints & compiler::surface_closure_endpoint_bit(
                                     compiler::SurfaceClosureEndpoint::physical)) !=
                    0u;
                $if(physical_endpoint) {
                    expand_physical_surface_closure(
                        services,
                        point,
                        raw,
                        reflective_caustics,
                        refractive_caustics,
                        [&](const TracedClosure &physical) noexcept {
                            if (physical.operation ==
                                compiler::ClosureOperation::transparent) {
                                reached_first_transparent |=
                                    physical.sample_weight >=
                                    cycles_closure::closure_weight_cutoff;
                                return;
                            }
                            $if(reached_first_transparent) {
                                emit_physical(
                                    canonical_surface_closure(physical));
                            };
                        });
                };
            });
    };
    return {
        .emission = std::move(emission),
        .shading_normal = point.shading_normal};
}

[[nodiscard]] Float3 evaluate_surface_emission_variant(
    std::uint32_t static_variant, const SurfaceValueRuntime &runtime,
    const ShaderServices &services, const SurfacePoint &point,
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    Float mix_weight, Expr<bool> reflective_caustics) noexcept {
  namespace operand = compiler::surface_closure_operand;
  const auto operation = static_cast<compiler::ClosureOperation>(
      static_variant & compiler::surface_closure_opcode_mask);
  if (operation == compiler::ClosureOperation::emission) {
    const auto color =
        read_closure_vector_or(runtime, services, point, locals, instruction,
                               operand::emission::color, make_float3(0.0f));
    return color *
           read_closure_scalar_or(runtime, services, point, locals, instruction,
                                  operand::emission::strength, 0.0f) *
           mix_weight;
  }
  if (operation == compiler::ClosureOperation::principled) {
    const auto scalar = [&](std::size_t index, float fallback) noexcept {
      return read_closure_scalar_or(runtime, services, point, locals,
                                    instruction, index, Float{fallback});
    };
    const auto vector = [&](std::size_t index,
                            luisa::float3 fallback) noexcept {
      return read_closure_vector_or(runtime, services, point, locals,
                                    instruction, index, Float3{fallback});
    };
    const auto features = runtime.emission_principled_closure_features;
    const auto enabled =
        [features](compiler::PrincipledClosureFeature feature) noexcept {
          return (features &
                  compiler::principled_closure_feature_bit(feature)) != 0u;
        };

    TracedClosure raw;
    raw.operation = compiler::ClosureOperation::principled;
    raw.principled_features = features;
    raw.weight = make_float3(mix_weight);
    raw.normal = point.shading_normal;
    raw.alpha = 1.0f;
    raw.sheen_weight = 0.0f;
    raw.sheen_roughness = 0.5f;
    raw.sheen_tint = make_float3(1.0f);
    raw.coat_weight = 0.0f;
    raw.coat_roughness = 0.03f;
    raw.coat_ior = 1.5f;
    raw.coat_tint = make_float3(1.0f);
    raw.coat_normal = make_float3(0.0f);
    raw.coat_normal_linked =
        (static_variant & compiler::surface_closure_coat_normal_linked) != 0u;
    if (enabled(compiler::PrincipledClosureFeature::alpha)) {
      raw.alpha = scalar(operand::principled::alpha, 1.0f);
    }
    if (enabled(compiler::PrincipledClosureFeature::sheen)) {
      raw.normal = safe_normalize(
          vector(operand::principled::normal, luisa::make_float3(0.0f)),
          point.shading_normal);
      raw.sheen_weight = scalar(operand::principled::sheen_weight, 0.0f);
      raw.sheen_roughness = scalar(operand::principled::sheen_roughness, 0.5f);
      raw.sheen_tint =
          vector(operand::principled::sheen_tint, luisa::make_float3(1.0f));
      raw.coat_weight = scalar(operand::principled::coat_weight, 0.0f);
    }
    if (enabled(compiler::PrincipledClosureFeature::coat)) {
      raw.coat_weight = scalar(operand::principled::coat_weight, 0.0f);
      raw.coat_roughness = scalar(operand::principled::coat_roughness, 0.03f);
      raw.coat_ior = scalar(operand::principled::coat_ior, 1.5f);
      raw.coat_tint =
          vector(operand::principled::coat_tint, luisa::make_float3(1.0f));
    }
    if (raw.coat_normal_linked &&
        (enabled(compiler::PrincipledClosureFeature::sheen) ||
         enabled(compiler::PrincipledClosureFeature::coat))) {
      raw.coat_normal =
          vector(operand::principled::coat_normal, luisa::make_float3(0.0f));
    }
    raw.emission =
        vector(operand::principled::emission_color, luisa::make_float3(0.0f)) *
        scalar(operand::principled::emission_strength, 0.0f);
    return PrincipledLayerComponent{services, point}
        .evaluate_emission(raw, features, reflective_caustics)
        .radiance;
  }
  std::abort();
}

[[nodiscard]] Float3 execute_surface_emission_program(
    const SurfaceValueRuntime &runtime, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    UInt program,
    Expr<bool> reflective_caustics) noexcept {
  const auto range = surface_value_runtime_buffer<luisa::uint4>(
                         runtime, SurfaceValueRuntimeBufferSlot::program)
                         .read(program);
  const auto closure_begin = range.z;
  const auto closure_end = range.z + range.w;
  Float3 emission = make_float3(0.0f);
  traverse_surface_closure_program<false>(
      runtime,
      services,
      point,
      locals,
      closure_begin,
      closure_end,
      closure_begin,
      [&](Var<luisa::uint4> instruction,
          Float mix_weight,
          UInt) noexcept {
    const auto static_variant =
        instruction.x & compiler::surface_closure_emission_static_variant_mask;
    luisa::compute::detail::SwitchStmtBuilder{static_variant} % [&] {
      for (const auto variant : runtime.emission_closure_static_variants) {
        luisa::compute::detail::SwitchCaseStmtBuilder{variant} % [&, variant] {
          emission += evaluate_surface_emission_variant(
              variant, runtime, services, point, locals, instruction,
              mix_weight, reflective_caustics);
        };
      }
      luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
        luisa::compute::dsl::unreachable(
            "invalid compact emission closure variant");
      };
    };
      });
  return emission;
}

[[nodiscard]] SurfacePreparation prepare_surface_closure_program(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt program,
    const SurfacePreparationQuery &query,
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureAovCallable &aov_operation) noexcept {
    SurfacePreparationAccumulator accumulator{
        point,
        maximum_surface_closure_capacity,
        query.glossy_filter_roughness,
        query.include_runtime_flags,
        query.include_aov,
        identity,
        aov_operation};
    const auto population = execute_surface_closure_program(
        runtime,
        SurfaceClosureProgramDomain::population,
        services,
        point,
        locals,
        program,
        query.emission_reflective_caustics,
        query.reflective_caustics,
        query.refractive_caustics,
        [&](const SurfaceClosureRecord &closure) noexcept {
            accumulator.add(closure);
        });
    accumulator.finish();
    return accumulator.preparation(population.emission);
}


// Execute the exact dependency-union value schedule once and expose its
// surface-local typed banks only to a continuation recorded in the same
// lexical/device control-flow region. Locals never cross the population
// boundary, and both preparation-only and populate-once routes therefore
// share automatic-normal, Bump, and value-numbering semantics.
template<typename Continuation>
void emit_compact_surface_values(
    const SurfaceValueRuntime &runtime,
    SurfaceValueProgramDomain domain,
    const SurfaceValueProgramCallable &value_program,
    UInt surface_tag,
    SurfacePoint point,
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables,
    Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap,
    SurfaceValueBankDefinition bank_definition,
    Continuation &&continuation) noexcept {
    const auto domain_view = surface_value_program_domain(runtime, domain);
    $if(surface_tag <
        static_cast<luisa::uint>(runtime.topologies.size())) {
        SurfaceValueLocals locals;
        if (bank_definition == SurfaceValueBankDefinition::full_bank) {
            locals.define_all();
        }
        const auto locals_view = locals.view();
        auto packed_point = pack_surface_point(point);
        point.shading_normal = value_program(
            scalar_parameters,
            vector_parameters,
            cycles_bsdf_tables,
            textures,
            geometry_heap,
            surface_tag,
            packed_point,
            locals_view.scalars.storage,
            locals_view.vectors.storage,
            locals_view.unsigned_integers.storage);

        const auto program =
            surface_tag * SurfaceValueRuntime::programs_per_topology +
            domain_view.program_offset;
        continuation(point, locals_view, program);
    };
}


} // namespace

namespace {

class CompactSurfacePopulationProgramImpl final
    : public SurfacePopulationProgram {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    SurfaceValueProgramCallable _value_program;

  public:
    CompactSurfacePopulationProgramImpl(
        std::shared_ptr<LuisaSceneData> scene,
        SurfaceValueProgramCallable value_program) noexcept
        : _scene{std::move(scene)},
          _value_program{std::move(value_program)} {}

    [[nodiscard]] SurfacePopulation populate(
        Expr<std::uint32_t> surface_tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) const noexcept override {
        SurfacePopulation result{
            .emission = make_float3(0.0f),
            .shading_normal = point.shading_normal};

        // begin/finish form one transaction even for an invalid runtime tag.
        // A valid compact program refines begin() with the automatic-normal
        // result before emitting its first retained physical closure.
        collector.begin(point.shading_normal);
        emit_compact_surface_values(
            *_scene->surface_values,
            SurfaceValueProgramDomain::preparation,
            _value_program,
            UInt{surface_tag},
            point,
            Expr<Buffer<float>>{_scene->scalar_parameter_buffer},
            Expr<Buffer<luisa::float3>>{
                _scene->vector_parameter_buffer},
            Expr<Buffer<float>>{_scene->cycles_bsdf_table_buffer},
            Expr<BindlessArray>{_scene->texture_heap},
            Expr<BindlessArray>{_scene->heap},
            SurfaceValueBankDefinition::full_bank,
            [&](const SurfacePoint &evaluated_point,
                const SurfaceValueLocalsView &locals,
                UInt preparation_program) noexcept {
                collector.begin(evaluated_point.shading_normal);
                const auto population =
                    execute_surface_closure_program(
                        *_scene->surface_values,
                        SurfaceClosureProgramDomain::population,
                        services,
                        evaluated_point,
                        locals,
                        preparation_program,
                        query.emission_reflective_caustics,
                        query.reflective_caustics,
                        query.refractive_caustics,
                        [&](const SurfaceClosureRecord &closure) noexcept {
                            collector.add(closure);
                        });
                result.emission = population.emission;
                result.shading_normal = population.shading_normal;
            });
        collector.finish();
        return result;
    }
};

} // namespace

std::shared_ptr<const SurfacePopulationProgram>
make_compact_surface_population_program(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene->surface_values ||
        scene->surface_values->topologies.size() !=
            scene->surfaces.size()) {
        std::abort();
    }
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(
            scene->attribute_binding_slot,
            scene->attribute_range_slot);
    auto value_program = make_surface_value_program_callable(
        scene,
        texture_sampling,
        attribute_lookup,
        SurfaceValueProgramDomain::preparation);
    return std::make_shared<
        CompactSurfacePopulationProgramImpl>(
        scene, std::move(value_program));
}

SurfaceEmissionCallable make_compact_surface_emission_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene->surface_values ||
        scene->surface_values->topologies.size() !=
            scene->surfaces.size()) {
        std::abort();
    }
    const auto *runtime = scene->surface_values.get();
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(
            scene->attribute_binding_slot,
            scene->attribute_range_slot);
    auto value_program = make_surface_value_program_callable(
        scene,
        texture_sampling,
        attribute_lookup,
        SurfaceValueProgramDomain::emission);

    SurfaceEmissionCallable emission =
        [scene,
         runtime,
         value_program = std::move(value_program),
         texture_sampling,
         attribute_lookup](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float3,
            Bool reflective_caustics) noexcept {
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
        CallableSurfaceAttributeLookupProvider attribute_provider{
            geometry_heap, attribute_lookup};
            BufferShaderServices services{
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space,
                nullptr,
                &texture_provider,
                &attribute_provider};
            const auto point = unpack_surface_point(packed_point);
            Float3 result = make_float3(0.0f);
            emit_compact_surface_values(
                *runtime,
                SurfaceValueProgramDomain::emission,
                value_program,
                surface_tag,
                point,
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                SurfaceValueBankDefinition::program_prefix,
                [&](const SurfacePoint &evaluated_point,
                    const SurfaceValueLocalsView &locals,
                    UInt emission_program) noexcept {
                    result = execute_surface_emission_program(
                        *runtime,
                        services,
                        evaluated_point,
                        locals,
                        emission_program,
                        reflective_caustics);
                });
            return result;
        };
    emission.set_name("surface_emission_compact_values");
    return emission;
}

SurfacePreparationCallable
make_compact_surface_preparation_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene->surface_values ||
        scene->surface_values->topologies.size() !=
            scene->surfaces.size()) {
        std::abort();
    }
    const auto closure_setup =
        make_surface_closure_setup_callables();
    const auto closure_identity =
        make_surface_closure_identity_callable();
    const auto closure_aov =
        make_surface_closure_aov_callable();
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(
            scene->attribute_binding_slot,
            scene->attribute_range_slot);

    auto value_program = make_surface_value_program_callable(
        scene,
        texture_sampling,
        attribute_lookup,
        SurfaceValueProgramDomain::preparation);

    SurfacePreparationCallable preparation =
        [scene,
         value_program = std::move(value_program),
         closure_setup,
         closure_identity,
         closure_aov,
         texture_sampling,
         attribute_lookup](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Var<SurfacePreparationQueryCall> packed_query) noexcept {
            CallableSurfaceClosureSetupProvider setup_provider{
                cycles_bsdf_tables, closure_setup};
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space,
                &setup_provider,
                &texture_provider,
                &attribute_provider};
            auto point = unpack_surface_point(packed_point);
            Var<SurfacePreparationCall> result =
                pack_surface_preparation(
                    SurfacePreparation::zero(point));
            emit_compact_surface_values(
                *scene->surface_values,
                SurfaceValueProgramDomain::preparation,
                value_program,
                surface_tag,
                point,
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                SurfaceValueBankDefinition::program_prefix,
                [&](const SurfacePoint &evaluated_point,
                    const SurfaceValueLocalsView &locals,
                    UInt preparation_program) noexcept {
                    const auto query =
                        unpack_surface_preparation_query(packed_query);
                    result = pack_surface_preparation(
                        prepare_surface_closure_program(
                            *scene->surface_values,
                            services,
                            evaluated_point,
                            locals,
                            preparation_program,
                            query,
                            closure_identity,
                            closure_aov));
                });
            return result;
        };
    preparation.set_name("surface_prepare_compact_values");
    return preparation;
}

SurfaceBssrdfNormalCallable make_compact_surface_bssrdf_normal_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene->surface_values ||
        scene->surface_values->topologies.size() != scene->surfaces.size()) {
        std::abort();
    }
    if (scene->surface_bssrdf_bump_tags.empty()) {
        // The tag set is the conservative host/JIT image of every material
        // whose exact binding can set has_bssrdf_bump. An empty image proves
        // the device predicate is false for the whole scene, so recording the
        // value/closure interpreters here would be dead specialization.
        SurfaceBssrdfNormalCallable identity = [](
            BufferFloat,
            BufferFloat3,
            BufferFloat,
            BindlessVar,
            BindlessVar,
            UInt,
            Var<SurfacePointCall> packed_point,
            Bool,
            Bool,
            Bool) noexcept {
            return unpack_surface_point(packed_point).shading_normal;
        };
        identity.set_name("surface_bssrdf_normal_scene_identity");
        return identity;
    }
    const auto closure_setup = make_surface_closure_setup_callables();
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup = make_surface_attribute_lookup_callable(
        scene->attribute_binding_slot, scene->attribute_range_slot);
    auto value_program = make_surface_value_program_callable(
        scene,
        texture_sampling,
        attribute_lookup,
        SurfaceValueProgramDomain::bssrdf);

    SurfaceBssrdfNormalCallable bssrdf_normal =
        [scene, value_program = std::move(value_program), closure_setup,
         texture_sampling, attribute_lookup](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point, Bool has_bssrdf_bump,
            Bool reflective_caustics, Bool refractive_caustics) noexcept {
            CallableSurfaceClosureSetupProvider setup_provider{cycles_bsdf_tables,
                                                               closure_setup};
            CallableTexture2DSamplingProvider texture_provider{textures,
                                                               texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{scalar_parameters,
                                          vector_parameters,
                                          cycles_bsdf_tables,
                                          textures,
                                          geometry_heap,
                                          scene->attribute_binding_slot,
                                          scene->attribute_range_slot,
                                          scene->nishita_texture_bindings,
                                          scene->shader_color_space,
                                          &setup_provider,
                                          &texture_provider,
                                          &attribute_provider};
            const auto point = unpack_surface_point(packed_point);
            Float3 result = point.shading_normal;
            $if (has_bssrdf_bump) {
                $if (surface_tag < static_cast<luisa::uint>(
                                       scene->surface_values->topologies.size())) {
                    emit_compact_surface_values(
                        *scene->surface_values,
                        SurfaceValueProgramDomain::bssrdf,
                        value_program,
                        surface_tag,
                        point,
                        scalar_parameters,
                        vector_parameters,
                        cycles_bsdf_tables, textures, geometry_heap,
                        SurfaceValueBankDefinition::program_prefix,
                        [&](const SurfacePoint &evaluated_point,
                            const SurfaceValueLocalsView &locals,
                            UInt preparation_program) noexcept {
                            SurfaceBssrdfNormalAccumulator accumulator{
                                evaluated_point.shading_normal,
                                scene->volume_metadata.closure_allocation_budget};
                            static_cast<void>(execute_surface_closure_program(
                                *scene->surface_values,
                                SurfaceClosureProgramDomain::bssrdf,
                                services, evaluated_point, locals,
                                preparation_program, reflective_caustics,
                                reflective_caustics, refractive_caustics,
                                [&](const SurfaceClosureRecord &closure) noexcept {
                                    accumulator.add(closure.kind, closure.weight,
                                                    closure.allocation_weight,
                                                    closure.normal);
                                }));
                            result = accumulator.result();
                        });
                };
            };
            return result;
        };
    bssrdf_normal.set_name("surface_bssrdf_normal_compact_values");
    return bssrdf_normal;
}

} // namespace psycles::luisa_backend::detail
