#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

void GraphSurfaceImplementation::for_each_closure(
    const TracedValues &values,
    const ClosureVisitor &function) const noexcept {
    auto visit =
        [&](auto &&self,
            compiler::ClosureExpressionId id,
            Float mix_weight) noexcept -> void {
        if (!id.valid() ||
            !_closure_plan.entry(id).reachable) {
            return;
        }
        const auto &closure =
            _program->closure_instructions()[id.value];
        switch (closure.operation) {
            case compiler::ClosureOperation::null_closure:
                return;
            case compiler::ClosureOperation::add:
                self(self, closure.a, mix_weight);
                self(self, closure.b, mix_weight);
                return;
            case compiler::ClosureOperation::mix: {
                const auto a_reachable =
                    closure.a.valid() &&
                    _closure_plan.entry(closure.a).reachable;
                const auto b_reachable =
                    closure.b.valid() &&
                    _closure_plan.entry(closure.b).reachable;
                if (!a_reachable && !b_reachable) {
                    return;
                }
                if (a_reachable && !b_reachable) {
                    self(self, closure.a, mix_weight);
                    return;
                }
                if (!a_reachable && b_reachable) {
                    self(self, closure.b, mix_weight);
                    return;
                }
                auto factor = clamp(
                    scalar(closure.factor, values),
                    0.0f,
                    1.0f);
                self(
                    self,
                    closure.a,
                    mix_weight * (1.0f - factor));
                self(
                    self,
                    closure.b,
                    mix_weight * factor);
                return;
            }
            case compiler::ClosureOperation::translucent: {
                auto color = vector(
                    closure.color, values);
                function(TracedClosure{
                    .operation =
                        compiler::ClosureOperation::translucent,
                    .weight = bsdf_allocated_weight(
                        color * mix_weight),
                    .color = color,
                    .normal = safe_normalize(
                        vector(closure.normal, values),
                        values.shading_normal),
                    .roughness = 0.0f,
                    .diffuse_roughness = 0.0f,
                    .metallic = 0.0f,
                    .ior = 1.0f,
                    .specular_ior_level = 0.0f,
                    .specular_tint = make_float3(1.0f)});
                return;
            }
            case compiler::ClosureOperation::diffuse:
            case compiler::ClosureOperation::principled:
            case compiler::ClosureOperation::glossy: {
                auto color = vector(
                    closure.color, values);
                auto metallic =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? clamp(
                              scalar(
                                  closure.metallic, values),
                              0.0f,
                              1.0f)
                        : closure.operation ==
                                  compiler::ClosureOperation::glossy
                              ? Float{1.0f}
                              : Float{0.0f};
                auto ior =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? max(
                              scalar(closure.ior, values),
                              1.0e-5f)
                        : Float{1.5f};
                auto diffuse_roughness =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(
                              closure.diffuse_roughness,
                              values)
                        : scalar(
                              closure.roughness, values);
                auto specular_ior_level =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? max(
                              scalar(
                                  closure.specular_ior_level,
                                  values),
                              0.0f)
                        : Float{0.5f};
                auto subsurface_weight =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? clamp(
                              scalar(
                                  closure.subsurface_weight,
                                  values),
                              0.0f,
                              1.0f)
                        : Float{0.0f};
                auto subsurface_radius =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? max(
                              vector(
                                  closure.subsurface_radius,
                                  values),
                              make_float3(0.0f))
                        : make_float3(0.0f);
                auto subsurface_scale =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? max(
                              scalar(
                                  closure.subsurface_scale,
                                  values),
                              0.0f)
                        : Float{0.0f};
                auto subsurface_ior =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.subsurface_ior, values)
                        : Float{1.4f};
                auto subsurface_anisotropy =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(
                              closure.subsurface_anisotropy,
                              values)
                        : Float{0.0f};
                auto transmission_weight =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(
                              closure.transmission_weight,
                              values)
                        : Float{0.0f};
                auto specular_tint =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? max(
                              vector(
                                  closure.specular_tint,
                                  values),
                              make_float3(0.0f))
                        : make_float3(1.0f);
                auto alpha =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.alpha, values)
                        : Float{1.0f};
                auto thin_wall =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.thin_wall, values) != 0.0f
                        : Bool{false};
                auto sheen_weight =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.sheen_weight, values)
                        : Float{0.0f};
                auto sheen_roughness =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.sheen_roughness, values)
                        : Float{0.5f};
                auto sheen_tint =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? vector(closure.sheen_tint, values)
                        : make_float3(1.0f);
                auto coat_weight =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.coat_weight, values)
                        : Float{0.0f};
                auto coat_roughness =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.coat_roughness, values)
                        : Float{0.03f};
                auto coat_ior =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? scalar(closure.coat_ior, values)
                        : Float{1.5f};
                auto coat_tint =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? vector(closure.coat_tint, values)
                        : make_float3(1.0f);
                auto coat_normal =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? vector(closure.coat_normal, values)
                        : make_float3(0.0f);
                auto emission =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? vector(closure.emission_color, values) *
                              scalar(closure.emission_strength, values)
                        : make_float3(0.0f);
                function(TracedClosure{
                    .operation = closure.operation,
                    .principled_features =
                        closure.operation ==
                                compiler::ClosureOperation::principled
                            ? _closure_plan.entry(id)
                                  .principled_features
                            : compiler::PrincipledClosureFeatureMask{},
                    .weight =
                        closure.operation ==
                                compiler::ClosureOperation::principled
                            ? make_float3(mix_weight)
                            : bsdf_allocated_weight(
                                  color * mix_weight),
                    .color = color,
                    .normal = safe_normalize(
                        vector(closure.normal, values),
                        values.shading_normal),
                    .roughness = scalar(
                        closure.roughness, values),
                    .diffuse_roughness =
                        diffuse_roughness,
                    .subsurface_weight =
                        subsurface_weight,
                    .subsurface_radius =
                        subsurface_radius,
                    .subsurface_scale =
                        subsurface_scale,
                    .subsurface_method = closure.subsurface_method,
                    .subsurface_ior = subsurface_ior,
                    .subsurface_anisotropy =
                        subsurface_anisotropy,
                    .transmission_weight =
                        transmission_weight,
                    .metallic = metallic,
                    .ior = ior,
                    .specular_ior_level =
                        specular_ior_level,
                    .specular_tint = specular_tint,
                    .alpha = alpha,
                    .thin_wall = thin_wall,
                    .sheen_weight = sheen_weight,
                    .sheen_roughness = sheen_roughness,
                    .sheen_tint = sheen_tint,
                    .coat_weight = coat_weight,
                    .coat_roughness = coat_roughness,
                    .coat_ior = coat_ior,
                    .coat_tint = coat_tint,
                    .coat_normal = coat_normal,
                    .coat_normal_linked =
                        closure.coat_normal_linked,
                    .emission = emission,
                    .preserve_ggx_energy =
                        closure.preserve_ggx_energy,
                    .beckmann =
                        closure.operation ==
                                compiler::ClosureOperation::glossy &&
                        closure.beckmann});
                return;
            }
            case compiler::ClosureOperation::subsurface: {
                const auto color = vector(closure.color, values);
                function(TracedClosure{
                    .operation = compiler::ClosureOperation::subsurface,
                    // Standalone SVM BSSRDF allocates the authored Color
                    // through the closure-tree mix weight; its transport
                    // albedo remains the unscaled Color socket.
                    .weight = color * mix_weight,
                    .color = color,
                    .normal = safe_normalize(
                        vector(closure.normal, values),
                        values.shading_normal),
                    .roughness = scalar(closure.roughness, values),
                    .subsurface_radius = max(
                        vector(closure.subsurface_radius, values),
                        make_float3(0.0f)),
                    .subsurface_scale = max(
                        scalar(closure.subsurface_scale, values),
                        0.0f),
                    .subsurface_method = closure.subsurface_method,
                    .subsurface_ior = scalar(
                        closure.subsurface_ior, values),
                    .subsurface_anisotropy = scalar(
                        closure.subsurface_anisotropy, values),
                    .ior = 1.0f});
                return;
            }
            case compiler::ClosureOperation::glass:
            case compiler::ClosureOperation::refraction: {
                const auto glass = closure.operation ==
                                   compiler::ClosureOperation::glass;
                auto color = max(
                    vector(closure.color, values),
                    make_float3(0.0f));
                function(TracedClosure{
                    .operation = closure.operation,
                    // Cycles allocates Glass with the closure mix weight and
                    // treats Color as a Fresnel tint. Standalone Refraction
                    // is a pure-transmission closure whose Color is its
                    // ordinary allocation weight.
                    .weight = glass
                                  ? make_float3(mix_weight)
                                  : color * mix_weight,
                    .color = color,
                    .normal = safe_normalize(
                        vector(closure.normal, values),
                        values.shading_normal),
                    .roughness = scalar(closure.roughness, values),
                    .diffuse_roughness = 0.0f,
                    .metallic = 0.0f,
                    .ior = max(scalar(closure.ior, values), 1.0e-5f),
                    .specular_ior_level = 0.5f,
                    .specular_tint = make_float3(1.0f),
                    .preserve_ggx_energy =
                        glass && closure.preserve_ggx_energy,
                    .beckmann = closure.beckmann});
                return;
            }
            case compiler::ClosureOperation::emission: {
                auto color = vector(
                    closure.color, values);
                function(TracedClosure{
                    .operation =
                        compiler::ClosureOperation::emission,
                    .weight =
                        color *
                        scalar(closure.strength, values) *
                        mix_weight,
                    .color = color,
                    .normal = make_float3(0.0f, 0.0f, 1.0f),
                    .roughness = 0.0f,
                    .metallic = 0.0f,
                    .ior = 1.0f});
                return;
            }
            case compiler::ClosureOperation::transparent: {
                auto color = vector(
                    closure.color, values);
                function(TracedClosure{
                    .operation =
                        compiler::ClosureOperation::transparent,
                    .weight = color * mix_weight,
                    .color = color,
                    // Cycles initializes ShaderClosure::N from ShaderData::N
                    // even though transparent sampling does not consume it.
                    .normal = values.shading_normal,
                    .roughness = 0.0f,
                    .metallic = 0.0f,
                    .ior = 1.0f});
                return;
            }
        }
    };
    visit(visit, _program->root(), 1.0f);
}
void GraphSurfaceImplementation::for_each_volume(
    const TracedValues &values,
    const VolumeVisitor &function) const noexcept {
    auto visit =
        [&](auto &&self,
            compiler::VolumeExpressionId id,
            Float mix_weight) noexcept -> void {
        const auto &volume =
            _program->volume_instructions()[id.value];
        switch (volume.operation) {
            case compiler::VolumeOperation::null_volume:
                return;
            case compiler::VolumeOperation::add:
                self(self, volume.a, mix_weight);
                self(self, volume.b, mix_weight);
                return;
            case compiler::VolumeOperation::mix: {
                const auto factor = clamp(
                    scalar(volume.factor, values),
                    0.0f,
                    1.0f);
                self(
                    self,
                    volume.a,
                    mix_weight * (1.0f - factor));
                self(
                    self,
                    volume.b,
                    mix_weight * factor);
                return;
            }
            case compiler::VolumeOperation::absorption:
            case compiler::VolumeOperation::scatter:
            case compiler::VolumeOperation::coefficients:
            case compiler::VolumeOperation::emission:
            case compiler::VolumeOperation::principled:
                function(volume, mix_weight);
                return;
        }
    };
    visit(visit, _program->volume_root(), 1.0f);
}

}// namespace psycles::luisa_backend::detail
