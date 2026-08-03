#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_visitor.h> through the Psycles::luisa target."
#endif

#include <cstddef>

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend {

// OOP host/JIT-stage boundary for operations over one runtime material.
// GraphSurface records the material graph once, add() retains only expression
// handles, and visit() emits the operation before that dispatch branch ends.
// No closure value crosses to the C++ host and no device-local closure array is
// introduced by this class.
class SurfaceClosureExpressionVisitor : public SurfaceClosureCollector {

  private:
    std::size_t _capacity;
    luisa::vector<SurfaceClosureExpression> _closures;
    const luisa::compute::Expression *_shading_normal{};

  protected:
    [[nodiscard]] std::size_t capacity() const noexcept;

    virtual void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression> &closures) noexcept = 0;

  public:
    explicit SurfaceClosureExpressionVisitor(
        std::size_t capacity) noexcept;

    SurfaceClosureExpressionVisitor(
        const SurfaceClosureExpressionVisitor &) = delete;
    SurfaceClosureExpressionVisitor(
        SurfaceClosureExpressionVisitor &&) = delete;
    SurfaceClosureExpressionVisitor &operator=(
        const SurfaceClosureExpressionVisitor &) = delete;
    SurfaceClosureExpressionVisitor &operator=(
        SurfaceClosureExpressionVisitor &&) = delete;

    void begin(
        Expr<luisa::float3> shading_normal) noexcept final;
    void add(
        const SurfaceClosureRecord &closure) noexcept final;
    void finish() noexcept final;
};

}// namespace psycles::luisa_backend
