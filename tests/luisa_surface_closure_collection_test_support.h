#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::test_support {

struct CollectedClosureTrace {
    luisa::compute::UInt count;
    luisa_backend::SurfaceClosureRecord closure;
    luisa::compute::Bool valid;
    luisa::compute::Float3 shading_normal;
};

// Multistage diagnostic consumer of the collection boundary. add() retains
// only raw AST expression handles; runtime-indexed selection is deliberately
// emitted later by finish(), while still inside the material dispatch branch.
class RequestedClosureCollector final
    : public luisa_backend::SurfaceClosureCollector {

  private:
    luisa::compute::UInt _requested;
    luisa::vector<luisa_backend::SurfaceClosureExpression> _closures;
    luisa::compute::UInt _count{0u};
    luisa_backend::SurfaceClosureRecord _selected{
        luisa_backend::SurfaceClosureRecord::zero()};
    luisa::compute::Bool _valid{false};
    luisa::compute::Float3 _shading_normal{
        luisa::compute::make_float3(0.0f, 0.0f, 1.0f)};

  public:
    explicit RequestedClosureCollector(
        luisa::compute::UInt requested) noexcept;

    void begin(
        luisa::compute::Expr<luisa::float3>
            shading_normal) noexcept override;
    void add(
        const luisa_backend::SurfaceClosureRecord
            &closure) noexcept override;
    void finish() noexcept override;

    [[nodiscard]] CollectedClosureTrace result() const noexcept;
};

} // namespace psycles::test_support
