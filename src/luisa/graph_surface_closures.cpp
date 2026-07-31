#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

void GraphSurfaceImplementation::for_each_closure(
    const TracedValues &values,
    const ClosureVisitor &function) const noexcept {
    auto visit =
        [&](auto &&self,
            compiler::ClosureExpressionId id,
            Float mix_weight) noexcept -> void {
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
                auto specular_tint =
                    closure.operation ==
                            compiler::ClosureOperation::principled
                        ? max(
                              vector(
                                  closure.specular_tint,
                                  values),
                              make_float3(0.0f))
                        : make_float3(1.0f);
                function(TracedClosure{
                    .operation = closure.operation,
                    .weight =
                        closure.operation ==
                                compiler::ClosureOperation::diffuse
                            ? bsdf_allocated_weight(
                                  color * mix_weight)
                            : make_float3(mix_weight),
                    .color = color,
                    .normal = safe_normalize(
                        vector(closure.normal, values),
                        values.shading_normal),
                    .roughness = scalar(
                        closure.roughness, values),
                    .diffuse_roughness =
                        diffuse_roughness,
                    .metallic = metallic,
                    .ior = ior,
                    .specular_ior_level =
                        specular_ior_level,
                    .specular_tint = specular_tint,
                    .preserve_ggx_energy =
                        closure.preserve_ggx_energy});
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
            case compiler::VolumeOperation::principled:
                function(volume, mix_weight);
                return;
        }
    };
    visit(visit, _program->volume_root(), 1.0f);
}

}// namespace psycles::luisa_backend::detail
