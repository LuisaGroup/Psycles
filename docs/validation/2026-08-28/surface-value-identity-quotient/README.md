# Surface-value identity quotient

## Outcome

Accepted. Pure same-physical-bank `ValueOperation::passthrough` values are now
contracted as SSA aliases before scheduling and typed-slot coloring. The
source `SurfaceProgram`, nominal socket types, and every public
`ValueExpressionId` remain intact for diagnostics and graph round-tripping;
only the compact device program is quotiented. Already serialized legacy
Passthrough records remain valid and executable.

On the official Blender 5.2 Barbershop export, this removes 1,115 of 6,274
preparation instructions (17.77%) and one semantic evaluator variant without
increasing the 8/9/0 peak typed-slot vector or the 140 B maximum program
payload. Three HIP candidate traces put the `shade_surface` median at
26.7645 ns/work-item versus 27.0598 for the parent trace, a 1.09% directional
improvement. The surface code object shrinks by 128 B. This is intentionally a
small runtime gain: identities were cheap individually, but eliminating them
is a structural prerequisite for a compact Cycles-style SVM.

## Reference identity

- Parent Psycles baseline: `7aa37b082f06082a46be5d36155000182bb7281a`.
- LuisaCompute: `ca29d37d3458bd44c4d7bbc55f69c7bf047f83ae`.
- Cycles source reference: Blender 5.2 release at
  `9e2066aef7ef7e20c142ad7bd3303138a4304c93`.
- Device: AMD Radeon RX 9070 XT (`gfx1201`), HIP backend.
- Scene: official Blender 5.2 Barbershop export, 640x480, 64 fixed samples,
  Tabulated Sobol, adaptive sampling disabled, staged wavefront scheduler.

The export reports the same unavailable-image warnings in both builds. This
is therefore a parent-versus-candidate semantic and performance comparison,
not a new Barbershop-versus-Cycles texture-fidelity claim.

## Formal model

Let the value program be a finite strict-topological DAG `G = (V, E)`. Each
value has a nominal Blender socket type and a physical execution bank

```text
B = {scalar, vector, unsigned_integer}.
```

The evaluator contract for Passthrough is exactly

```text
eval(passthrough(x)) = eval(x).
```

Define `rep(v)` in source order. Parameters and ordinary operations represent
themselves. For `p = passthrough(x)`, set `rep(p) = rep(x)` only when
`bank(p) = bank(rep(x))`; otherwise planning rejects the invalid physical
copy. Strict topology makes this definition well-founded. The quotient keeps
one executable instruction per representative and rewrites every operand and
terminal output use to its representative.

For any evaluation environment, induction over source order gives

```text
eval(v) = eval(rep(v)).
```

The base cases represent themselves. The only contracted inductive case is
the evaluator identity above. Ordinary instructions retain their operation,
immutable metadata, parameter bindings, operand order, and the values of all
operands by the induction hypothesis. Consequently every closure input and
surface-normal transaction observes the same value. Publishing
`location(v) = location(rep(v))` preserves all original IDs without a device
copy or a weak runtime type.

## Liveness and constrained scheduling

Use counts are accumulated on representatives. Alias nodes neither read nor
write local storage. The allocator retains its read-before-write contract: it
first validates and consumes all operand locations, releases same-bank final
uses, and only then assigns the result. Greedy coloring is therefore exact
for the interval graph induced by a fixed schedule.

The compact interpreter has a component-wise capacity vector, not one scalar
byte budget:

```text
C = (8 scalar, 12 vector, 1 unsigned_integer).
```

A 20 B schedule can be infeasible when it needs two scalar slots, while a
28 B schedule with one scalar and two vector slots is legal. Selection hence
filters by `pressure <= C` component-wise before minimizing exact typed bytes.

Three independently legal schedules are considered:

1. Sethi-Ullman order computed directly on the quotient DAG.
2. Strict authored topological order with aliases removed.
3. The previous uncontracted Sethi-Ullman order projected by deleting aliases.

Projection is legal because deleting `p = id(x)` preserves `x` before every
surviving consumer and does not change the relative order of any surviving
pair. It cannot increase same-bank pressure: before the deleted copy, `x`
occupies the old slot; afterward its representative replaces `p`; at the copy
boundary read-before-write permits the dying source slot to be reused. Alias
chains and multiple consumers only merge additional equal-bank intervals.
The authored candidate is the corresponding projection of the old authored
schedule. Thus, if either pre-quotient schedule fit the ABI, at least its
projection still fits. This is the no-regression witness; the direct quotient
schedule remains available when it is better.

## Permanent regressions

The host tests cover the proof obligations separately:

- Float, Boolean, Color, Normal, and UInt64 identities produce zero device
  instructions, retain all source IDs, and share the representative's exact
  storage address.
- Contracted programs generate neither dead semantic variants nor bytecode
  records.
- Automatic-normal and endpoint transactions remain ordered when their value
  prefixes reduce to zero instructions.
- Explicit legacy uncoalesced plans still serialize and pass the bytecode
  verifier, so optimization does not silently remove format compatibility.
- A typed-pressure counterexample proves that capacity feasibility is checked
  component-wise before total payload size; an impossible capacity is rejected
  transactionally.

The focused runtime matrix passed all 15 tests:

```text
psycles.luisa_surface_closure_collection_{fallback,hip,vk}
psycles.luisa_surface_closure_reachability_{fallback,hip,vk}
psycles.luisa_surface_population_{fallback,hip,vk}
psycles.luisa_compact_surface_preparation_{fallback,hip,vk}
psycles.luisa_compact_surface_tail_{fallback,hip,vk}
```

The Vulkan cases use the repository's strict native XIR-to-SPIR-V route. The
full suite ran 314 tests: 308 passed, and the same six pre-existing numerical
fixtures failed (`stacked_volume_fallback`, `homogeneous_volume_fallback`,
`area_light_forward_vk`, `volume_path_fallback`, `volume_path_vk`, and
`volume_triangle_fallback`). None exercises Passthrough contraction, and the
failure set is unchanged from the parent baseline.

## Barbershop static census

| Metric | Parent | Identity quotient | Change |
|---|---:|---:|---:|
| Preparation runtime instructions | 6,274 | 5,159 | -1,115 (-17.77%) |
| Semantic value variants | 61 | 60 | -1 |
| Maximum scalar/vector/uint slots | 8 / 9 / 0 | 8 / 9 / 0 | unchanged |
| Maximum typed payload | 140 B | 140 B | unchanged |
| Sum of topology typed payloads | 8,324 B | 8,300 B | -24 B |
| HIP surface code object | 340,312 B | 340,184 B | -128 B (-0.038%) |
| Coroutine frame | 177 fields / 864 B | 177 fields / 864 B | unchanged |

The complete runtime contains 380 preparation/emission programs and 8,870
ordinary value instructions after contraction. No material value is baked,
and graph topology remains data rather than expanded Luisa AST.

## HIP measurement

Every trace renders the same 640x480 image at 64 fixed samples. Kernel time is
normalized by the actual dispatch work sum.

| Build / run | Calls | Work-items | GPU duration (ms) | ns/work-item |
|---|---:|---:|---:|---:|
| Parent | 293 | 53,660,864 | 1,452.051699 | 27.0597900 |
| Quotient 1 | 293 | 53,660,800 | 1,435.739747 | 26.7558394 |
| Quotient 2 | 293 | 53,660,864 | 1,436.205405 | 26.7644853 |
| Quotient 3 | 293 | 53,660,864 | 1,440.563778 | 26.8457060 |
| Candidate median | | | | **26.7644853 (-1.091%)** |

All three candidate samples are below the parent sample; the range is
26.7558-26.8457 ns/work-item. Because only one preserved parent trace is
available, this is directional evidence rather than a confidence interval.
Render-only wall time was 2.54230 s for the parent and 2.52027/2.52529 s for
the two candidate runs with preserved logs (mean change -0.768%). Cold JIT
state was not matched, so no shader-compilation speedup is claimed here.

## Numerical and visual comparison

The full 46-channel EXR comparison reports mean error `3.97126e-10`, RMS
`1.20709e-8`, and maximum absolute error `3.05176e-5` in one GlossDir value.
Only 21 of 307,200 pixels exceed `1e-6` in any channel. All 15 named film
passes have zero invalid pixels; the detailed result is
[`all-pass-report.json`](all-pass-report.json).

| Pass | Relative RMSE | Maximum absolute error | Mean absolute error |
|---|---:|---:|---:|
| Combined | `2.137e-8` | `9.537e-7` | `1.346e-10` |
| Normal | `1.749e-8` | `8.959e-7` | `1.412e-9` |
| Diffuse Color | `2.126e-8` | `1.788e-7` | `8.106e-10` |

The following triptychs show the parent on the left, the quotient build in the
center, and an explicitly amplified absolute difference on the right:

- [Combined](triptychs/combined.png)
- [Normal](triptychs/normal.png)
- [Diffuse Color](triptychs/diffcol.png)

Native-resolution visual inspection found no observable geometry, material,
texture, lighting, orientation, or structural difference between the left and
center panels. The right panels require approximately `2.42e8`, `1.51e7`, and
`3.02e7` amplification respectively; their sparse sample- and edge-shaped
residuals are consistent with floating-point atomic arrival order. No exact
hash test or tolerance was weakened.

## Reproduction

```sh
cmake --build build -j32 --target \
  psycles_render_blender_scene \
  psycles_inspect_blender_material \
  psycles_surface_program_metadata_tests \
  psycles_surface_svm_record_immediate_tests

ctest --test-dir build --output-on-failure -j3 \
  -R '^psycles\.luisa_(surface_closure_(collection|reachability)|surface_population|compact_surface_(preparation|tail))_(fallback|hip|vk)$'

build/bin/psycles_inspect_blender_material BARBERSHOP_5_2_EXPORT '*'

rocprofv3 --kernel-trace --scratch-memory-trace --stats \
  -f rocpd -d PROFILE_DIR -o trace_results -- \
  build/bin/psycles_render_blender_scene \
  BARBERSHOP_5_2_EXPORT PROFILE_DIR/out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 131072 0 0 1
```

The visual report was generated with `tools/compare_cycles.py` using
`--allow-unverified-build-identity` because both inputs are Psycles A/B
artifacts, not Blender/Cycles outputs. Exact parent and candidate identities
are recorded above, and that override is not used for a Cycles compatibility
claim. Machine-readable static, performance, comparison, and test summaries
are in [`metrics.json`](metrics.json).
