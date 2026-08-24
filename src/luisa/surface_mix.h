#pragma once

#include <cstdint>
#include <span>

#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// One statically selected Cycles Mix operation. Keeping the formula in this
// common host-stage function lets the expanded graph and every generated SVM
// switch case record the same Luisa AST without textual duplication.
[[nodiscard]] Float3
evaluate_surface_mix_operation(const ShaderServices &services,
                               compiler::BlendOperation operation,
                               Float factor,
                               Float3 a,
                               Float3 b) noexcept;

// Host-specialized GraphSurface entry. This preserves the original expanded
// graph shape and is also the semantic reference for compact SVM execution.
[[nodiscard]] Float3 evaluate_surface_mix(const ShaderServices &services,
                                          compiler::BlendOperation operation,
                                          bool clamp_factor,
                                          bool clamp_result,
                                          Float factor,
                                          Float3 a,
                                          Float3 b) noexcept;

// Compact SVM entry. The instruction supplies one opcode-owned immediate;
// `immediate_domain` is the exact scene-reachable equivalence-class image used
// to prune the generated device switch.
[[nodiscard]] Float3
evaluate_surface_mix_svm(const ShaderServices &services,
                         UInt immediate,
                         std::span<const std::uint16_t> immediate_domain,
                         Float factor,
                         Float3 a,
                         Float3 b) noexcept;

} // namespace psycles::luisa_backend::detail
