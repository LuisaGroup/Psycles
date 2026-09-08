#pragma once

#include <psycles/compiler/cycles_svm_types.h>
#include <luisa/luisa-compute.h>

namespace psycles::test_support {
using namespace luisa::compute;

// Structural AST inspection only; no reference shader evaluator.
struct InterpreterShape final : StmtVisitor {
  std::uint32_t loop_depth{};
  std::uint32_t switch_depth{};
  std::uint32_t loops{};
  std::uint32_t primary_switches{};
  std::uint32_t returns{};
  std::uint32_t primary_switch_returns{};
  std::uint32_t true_bool_initializers{};
  std::uint32_t primary_loop_true_bool_initializers{};
  std::uint32_t materialized_stack_lane_initializers{};

  void visit(const BreakStmt *) override {}
  void visit(const ContinueStmt *) override {}
  void visit(const ReturnStmt *) override {
    ++returns;
    primary_switch_returns += loop_depth == 1u && switch_depth == 1u;
  }
  void visit(const ScopeStmt *stmt) override {
    for (const auto *statement : stmt->statements()) {
      statement->accept(*this);
    }
  }
  void visit(const IfStmt *stmt) override {
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
  }
  void visit(const LoopStmt *stmt) override {
    ++loops;
    ++loop_depth;
    stmt->body()->accept(*this);
    --loop_depth;
  }
  void visit(const ExprStmt *) override {}
  void visit(const SwitchStmt *stmt) override {
    if (loop_depth == 1u && switch_depth == 0u) {
      ++primary_switches;
    }
    ++switch_depth;
    stmt->body()->accept(*this);
    --switch_depth;
  }
  void visit(const SwitchCaseStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const SwitchDefaultStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const AssignStmt *stmt) override {
    const auto *rhs = stmt->rhs();
    if (stmt->lhs()->type() == Type::of<bool>() &&
        rhs->tag() == Expression::Tag::LITERAL &&
        luisa::get<bool>(
            static_cast<const LiteralExpr *>(rhs)->value().to_variant())) {
      ++true_bool_initializers;
      primary_loop_true_bool_initializers += loop_depth == 1u;
    }
    const auto *stack_type = Type::array(Type::of<float>(), SVM_STACK_SIZE);
    auto touches_stack = false;
    traverse_subexpressions(
        stmt->lhs(),
        [&](const Expression *expression) noexcept {
          touches_stack = touches_stack ||
                          (expression->tag() == Expression::Tag::REF &&
                           expression->type() == stack_type);
        },
        [](const Expression *) noexcept {});
    const auto is_fresh_lifetime_seed =
        rhs->tag() == Expression::Tag::CALL &&
        static_cast<const CallExpr *>(rhs)->op() == CallOp::UNDEFINED;
    materialized_stack_lane_initializers +=
        touches_stack && !is_fresh_lifetime_seed;
  }
  void visit(const ForStmt *stmt) override { stmt->body()->accept(*this); }
  void visit(const CommentStmt *) override {}
  void visit(const RayQueryStmt *stmt) override {
    stmt->on_triangle_candidate()->accept(*this);
    stmt->on_procedural_candidate()->accept(*this);
  }
  void visit(const SuspendStmt *) override {}
  void visit(const AutoDiffStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const PrintStmt *) override {}
  void visit(const DebugBreakStmt *) override {}
};

} // namespace psycles::test_support
