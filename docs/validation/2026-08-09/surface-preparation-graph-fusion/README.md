# Surface preparation graph fusion

Date: 2026-08-09

## Scope

The production surface-hit path previously invoked the same `GraphSurface`
program independently for emission, runtime closure flags, and camera AOVs.
Each invocation replayed the complete topologically ordered typed value graph,
including shared texture and parameter inputs.

The new `SurfacePreparation` boundary evaluates that typed graph once and
derives the three pre-sampling projections from the resulting values. It does
not store, bake, or reinterpret Blender material closures. The original raw
closure graph is still expanded at the current `SurfacePoint`, and BSDF
sampling/evaluation continue to consume the normal strongly typed Luisa DSL
closure implementation.

## Structural invariant

Let `T(P)` be the Luisa AST schedule obtained by walking a valid
`SurfaceProgram` value DAG `P` in its compiler-provided topological order.
Every `ValueExpressionId` is materialized exactly once in `T(P)`. The old
preparation path recorded `T(P)` independently for emission, runtime flags,
and AOVs. The fused path records one `T(P)` and projects all three results from
the same traced-value set.

Closure retention remains ordered and capacity-limited by the existing
`SurfaceClosureExpressionVisitor` contract. Runtime flags and AOVs share that
one retained-closure reduction. Compile-time-disabled reduction types contain
no Luisa DSL state, so a standalone legacy visitor does not accidentally
record the other projection.

## Code-shape result

The structural regression constructs a graph where one constant color feeds
both a diffuse closure and an emission closure. It first verifies that both
closure inputs reference the same `ValueExpressionId`, then records split and
fused kernels before backend optimization.

| Measurement | Split calls | Fused preparation | Change |
| --- | ---: | ---: | ---: |
| Shader-service parameter recordings | 12 | 4 | -66.7% |
| Unoptimized XIR instructions | 4,666 | 3,534 | -24.3% |

The exact 3:1 recording ratio is asserted by the test, so this result does not
depend on backend common-subexpression elimination. Metrics can be reproduced
with:

```sh
PSYCLES_REPORT_GRAPH_FUSION=1 build/psycles_luisa_compile_tests
```

## Semantic verification

`test_luisa_surface_closure_collection` compares fused preparation against
the former split emission, runtime-flag, and AOV paths on device. It checks
every returned field and separately checks the disabled flags/AOV contract.

| Backend | Result | Elapsed time |
| --- | --- | ---: |
| fallback | pass | 27.88 s |
| HIP | pass | 34.50 s |
| Vulkan XIR/SPIR-V | pass | 40.93 s |

Additional checks:

- `psycles.luisa_ast`: pass
- 32-thread complete project build: pass
- fallback, HIP, and Vulkan camera sampling regressions: pass

The timings above are regression-test wall times and are not renderer
throughput measurements. Scene-level cold compilation, kernel size, and
render throughput are measured separately so runtime claims are not inferred
from unit-test compilation time.
