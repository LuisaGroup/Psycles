/* SPDX-FileCopyrightText: 2026 Psycles Contributors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_svm_translator.h> through the Psycles::luisa target."
#endif

#include <psycles/compiler/cycles_svm_scene.h>
#include <psycles/luisa/cycles_svm.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace psycles::luisa_backend::cycles_svm {

// Pluggable backend lowering strategy for the unified Cycles SVM frontend
// representation (ShaderTableImage = SvmTableIr + linked word payload image).
enum class SvmBackendStrategy : std::uint8_t {
  auto_select = 0u,
  bytecode_vm = 1u,
  direct_jit = 2u,
};

struct SvmTranslationRequest {
  const KernelGlobals &kernel_globals;
  const compiler::cycles_svm::ShaderTableImage &table;
  luisa::compute::Expr<luisa::compute::Buffer<luisa::uint>> words;
  compiler::cycles_svm::ShaderType shader_type;
  std::uint32_t kernel_features{};
  std::uint32_t node_feature_mask{};
  const std::array<bool, compiler::cycles_svm::NODE_NUM> &node_types_used;
  const TransformState &transform_state;
  ShaderData &shader_data;
  const PathState &path_state;
  std::size_t stack_size{SVM_STACK_SIZE};
  compiler::cycles_svm::NoiseUsage noise_usage{};
  // Optional per-shader reachability filter in [0, table.shader_count). Empty
  // means all shaders in the table may reach this call site.
  std::span<const std::uint8_t> active_shader_mask{};
};

// Re-targetable SVM backend translator interface. Concrete translators lower
// the same unified frontend representation into either a true loop-based VM or
// flat, per-shader JIT-specialized device AST.
class SvmTranslator {
public:
  virtual ~SvmTranslator() noexcept = default;

  [[nodiscard]] virtual SvmBackendStrategy strategy() const noexcept = 0;

  virtual void translate(const SvmTranslationRequest &request) const noexcept = 0;
};

// Lowers ShaderTableImage into the loop-based bytecode interpreter
// (while (pc) switch (opcode) over the device word buffer).
class BytecodeVmTranslator final : public SvmTranslator {
public:
  [[nodiscard]] SvmBackendStrategy strategy() const noexcept override {
    return SvmBackendStrategy::bytecode_vm;
  }

  void translate(const SvmTranslationRequest &request) const noexcept override;
};

// Lowers ShaderTableImage::ir directly into loop-free per-shader Luisa AST with
// compile-time operand folding, constant-indexed SSA stack promotion, routine
// deduplication across equivalent shader IDs, and per-instruction static
// specialization.
class DirectJitTranslator final : public SvmTranslator {
public:
  [[nodiscard]] SvmBackendStrategy strategy() const noexcept override {
    return SvmBackendStrategy::direct_jit;
  }

  void translate(const SvmTranslationRequest &request) const noexcept override;
};

[[nodiscard]] SvmBackendStrategy resolve_svm_backend_strategy(
    const SvmTranslationRequest &request,
    SvmBackendStrategy preferred = SvmBackendStrategy::auto_select) noexcept;

[[nodiscard]] const SvmTranslator &svm_translator_for(
    SvmBackendStrategy strategy) noexcept;

void translate_svm_nodes_assume_valid(
    const SvmTranslationRequest &request,
    SvmBackendStrategy preferred = SvmBackendStrategy::auto_select) noexcept;

} // namespace psycles::luisa_backend::cycles_svm
