#include <psycles/compiler/cycles_svm_ir.h>

#include <algorithm>

namespace psycles::compiler::cycles_svm {

bool operator==(const SvmOpIr &lhs, const SvmOpIr &rhs) noexcept {
  return lhs.is_conditional == rhs.is_conditional &&
         lhs.jump_if_one == rhs.jump_if_one &&
         lhs.cond_stack_offset == rhs.cond_stack_offset &&
         lhs.instruction == rhs.instruction &&
         lhs.children == rhs.children;
}

bool operator==(const SvmShaderIr &lhs, const SvmShaderIr &rhs) noexcept {
  return lhs.routines == rhs.routines;
}

namespace {

void relocate_block(SvmBlockIr &block, std::uint32_t add_offset,
                    std::uint32_t subtract_offset) noexcept {
  for (auto &op : block) {
    if (op.is_conditional) {
      relocate_block(op.children, add_offset, subtract_offset);
    } else {
      op.instruction.payload_offset =
          op.instruction.payload_offset - subtract_offset + add_offset;
    }
  }
}

[[nodiscard]] bool same_linked_op(std::span<const std::uint32_t> words,
                                  const SvmOpIr &lhs,
                                  const SvmOpIr &rhs) noexcept {
  if (lhs.is_conditional != rhs.is_conditional) {
    return false;
  }
  if (lhs.is_conditional) {
    return lhs.jump_if_one == rhs.jump_if_one &&
           lhs.cond_stack_offset == rhs.cond_stack_offset &&
           same_linked_svm_block(words, lhs.children, rhs.children);
  }
  const auto &a = lhs.instruction;
  const auto &b = rhs.instruction;
  if (a.opcode != b.opcode || a.payload_word_count != b.payload_word_count ||
      a.specialization != b.specialization) {
    return false;
  }
  const auto a_end =
      static_cast<std::size_t>(a.payload_offset) + a.payload_word_count;
  const auto b_end =
      static_cast<std::size_t>(b.payload_offset) + b.payload_word_count;
  if (a_end > words.size() || b_end > words.size()) {
    return a.payload_offset == b.payload_offset;
  }
  return std::equal(words.begin() + static_cast<std::ptrdiff_t>(a.payload_offset),
                    words.begin() + static_cast<std::ptrdiff_t>(a_end),
                    words.begin() + static_cast<std::ptrdiff_t>(b.payload_offset));
}

} // namespace

void relocate_svm_shader_ir(SvmShaderIr &ir, std::uint32_t add_offset,
                            std::uint32_t subtract_offset) noexcept {
  for (auto &routine : ir.routines) {
    relocate_block(routine, add_offset, subtract_offset);
  }
}

bool same_linked_svm_block(std::span<const std::uint32_t> words,
                           const SvmBlockIr &lhs,
                           const SvmBlockIr &rhs) noexcept {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0u; i < lhs.size(); ++i) {
    if (!same_linked_op(words, lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

} // namespace psycles::compiler::cycles_svm
