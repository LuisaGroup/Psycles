#pragma once

#include "path_tracer_surface_value_program.h"

#include <cstddef>
#include <vector>

namespace psycles::luisa_backend::detail {

// Transitional capability boundary for the Cycles-aligned direct SVM
// evaluator. A supported family reads its packed operands and writes the
// typed stack directly; it never constructs TracedValues, a
// SurfaceValueExpression, or a host ValueNode. Keeping this predicate total
// makes partial replacement explicit and testable while remaining a pure
// host/JIT decision.
[[nodiscard]] constexpr bool surface_value_family_has_direct_evaluator(
    compiler::SurfaceSvmValueOpcode opcode) noexcept {
    return opcode == compiler::SurfaceSvmValueOpcode::convert ||
           opcode == compiler::SurfaceSvmValueOpcode::math ||
           opcode == compiler::SurfaceSvmValueOpcode::mix_color ||
           opcode == compiler::SurfaceSvmValueOpcode::rgb_ramp ||
           opcode == compiler::SurfaceSvmValueOpcode::mapping ||
           opcode == compiler::SurfaceSvmValueOpcode::tex_image ||
           opcode == compiler::SurfaceSvmValueOpcode::tex_image_box;
}

// A host/JIT typed cursor over one static family subtype. Operand words are
// decoded exactly once. Bank and route checks are host invariants, so a typed
// read records only the required device load rather than a weak runtime union.
class SurfaceValueOperandReader {

  private:
    const compiler::SurfaceValueStaticVariant &_variant;
    const ShaderServices &_services;
    const SurfacePoint &_point;
    const SurfaceValueLocalsView &_locals;
    std::vector<UInt> _addresses;

  private:
    [[nodiscard]] compiler::SurfaceValueBank
    bank(std::size_t index) const noexcept;
    [[nodiscard]] compiler::SurfaceValueOperandRoute
    route(std::size_t index) const noexcept;
    [[nodiscard]] UInt address(std::size_t index) const noexcept;

  public:
    SurfaceValueOperandReader(
        const SurfaceValueRuntime &runtime,
        SurfaceValueRuntimeBufferSlot operand_slot,
        const ShaderServices &services, const SurfacePoint &point,
        const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
        const compiler::SurfaceValueStaticVariant &variant) noexcept;

    [[nodiscard]] Float scalar(std::size_t index) const noexcept;
    [[nodiscard]] Float3 vector(std::size_t index) const noexcept;
    [[nodiscard]] ULong unsigned_integer(std::size_t index) const noexcept;
};

void write_surface_value_scalar(const SurfaceValueLocalsView &locals,
                                Var<luisa::uint4> instruction,
                                Float value) noexcept;
void write_surface_value_vector(const SurfaceValueLocalsView &locals,
                                Var<luisa::uint4> instruction,
                                Float3 value) noexcept;

// Emits one statically selected family subtype. Returns false exactly when
// the family has not yet been migrated to the direct typed-stack evaluator.
// A true return means the result has already been written to instruction.y.
[[nodiscard]] bool emit_direct_surface_value_variant(
    const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept;

}// namespace psycles::luisa_backend::detail
