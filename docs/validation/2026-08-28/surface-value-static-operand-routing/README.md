# Surface-value static operand routing

## Outcome

Accepted. The compact surface-value SVM now specializes an operand's storage
route only when a complete-scene finite-domain analysis proves that every
instruction mapped to the same semantic evaluator variant uses the same
route. On the official Blender 5.2 Barbershop export this removes 500 final
LLVM branches, reduces the HIP surface object by 4.66%, reduces fixed private
storage by 200 B, and improves normalized `shade_surface` time by 0.943%.
Matched render-only wall time improves by 0.662%.

This is a representation change, not an opcode/material special case. It does
not alter the bytecode ABI, duplicate handlers for route masks, bake material
values, or add a runtime mode. The original dynamic read remains exactly for
the operand coordinates whose scene-wide route domain contains both local and
parameter sources.

## Reference identity

- Parent Psycles baseline: `19d6e4644ba1d3e996edbc2e43dd44c58f7b4bfe`.
- LuisaCompute: `eeda4b154fcf43e8709d1b42478e958677b9c6ae`.
- Cycles source reference: Blender 5.2 release at
  `9e2066aef7ef7e20c142ad7bd3303138a4304c93`.
- Device: AMD Radeon RX 9070 XT (`gfx1201`), HIP backend.
- Scene: official Blender 5.2 Barbershop export, 640x480, 64 fixed samples,
  Tabulated Sobol, adaptive sampling disabled, staged wavefront scheduler.

The export emits the same ten unavailable-image warnings in both A/B runs.
Consequently this document establishes candidate-versus-parent semantic and
performance equivalence; it does not use this A/B image as a new claim about
Barbershop-versus-Cycles texture fidelity.

## Formal model

Let `I` be the finite set of ordinary value-bytecode instructions in the
immutable scene. Exact semantic interning maps each instruction `i` to a
variant `v(i)`. For operand coordinate `j`, its concrete storage route is

```text
r(i, j) in R = {L, P},
```

where `L` denotes a typed local bank and `P` denotes the immutable material
parameter block. For each interned variant and operand coordinate the builder
computes

```text
D(v, j) = union { {r(i, j)} | i in I and v(i) = v }.
```

Each coordinate therefore lives in the finite lattice `P(R)` ordered by
subset inclusion. Published variants cannot contain bottom because every
variant has at least one validated instruction occurrence. Code generation is

```text
D(v, j) = {L}    => direct typed-local read
D(v, j) = {P}    => direct typed-parameter read
D(v, j) = {L, P} => original dynamic route test and its two arms.
```

The semantic variant key already preserves operation, result bank, operand
arity, every operand bank, evaluator-owned static configuration, and static
table shape. Thus route class is the only fact projected by this analysis.
For a singleton domain, the selected direct read equals the corresponding arm
of the old conditional for every occurrence. At top, the old conditional is
retained. Pointwise equality of every instruction follows, and induction over
the topologically scheduled bytecode proves equality of all typed-local bank
states and final closure inputs. The proof does not depend on material names,
parameter values, control-flow frequency, or a backend heuristic.

## Scene census

`psycles_inspect_blender_material` independently reconstructs route masks from
the emitted bytecode instead of reading the compiler's published
`operand_routes`. The Barbershop scene reports:

| Metric | Count |
|---|---:|
| Semantic value variants | 61 |
| `(variant, route-mask)` pairs | 83 |
| Route-polymorphic variants | 11 |
| Maximum masks for one variant | 7 |
| Operand references | 12,597 |
| Parameter references | 6,287 |
| Local references | 6,310 |
| Proven static references | 6,455 (51.24%) |
| Remaining dynamic references | 6,142 (48.76%) |
| Proven parameter references | 4,229 |
| Proven local references | 2,226 |

At instruction granularity there are 614 all-parameter, 2,785 all-local,
2,321 mixed-source, and 554 zero-arity instructions. The implementation does
not clone the 11 polymorphic semantic handlers into 83 route-mask handlers;
it specializes only individual operand reads inside the existing handler.

## Backend regression gate

The build used all 32 host threads. Nineteen focused tests passed sequentially:

```text
psycles.luisa_microfacet_anisotropy_{fallback,hip,vk}
psycles.luisa_surface_closure_collection_{fallback,hip,vk}
psycles.luisa_surface_population_{fallback,hip,vk}
psycles.luisa_compact_surface_preparation_{fallback,hip,vk}
psycles.luisa_surface_closure_physical_{fallback,hip,vk}
psycles.surface_program_metadata
psycles.surface_svm_math_immediate
psycles.surface_svm_vector_math_immediate
psycles.surface_svm_record_immediates
```

The Vulkan variants use the repository's strict native XIR-to-SPIR-V test
route. The metadata regression includes a semantic Add variant whose first
operand occurs through both storage classes and must remain dynamic, a second
operand that must become direct parameter, an always-local unary operand, and
nominal socket spellings that quotient to the same three execution banks.

## Cold generated structure

Both rows use the same one-sample Barbershop AST with the shader cache
disabled. `hip_kernel_final_5.ll` is the surface-stage final LLVM module;
object metadata comes from `llvm-readelf --notes`.

| Metric | Exact parent | Static routing | Change |
|---|---:|---:|---:|
| Coroutine frame | 177 fields / 864 B | 177 fields / 864 B | unchanged |
| Final LLVM | 57,588 lines / 3,199,282 B | 54,104 lines / 3,013,842 B | -6.05% lines / -5.80% bytes |
| HIP surface object | 357,752 B | 341,080 B | -16,672 B (-4.660%) |
| Fixed private storage | 3,296 B | 3,096 B | -200 B (-6.068%) |
| VGPR / SGPR | 256 / 107 | 256 / 107 | unchanged |
| Loads | 2,831 | 2,542 | -289 (-10.21%) |
| Phi nodes | 2,539 | 2,276 | -263 (-10.36%) |
| Branches | 2,097 | 1,597 | -500 (-23.84%) |
| Selects | 2,689 | 2,674 | -15 |
| Stores / calls / switches / allocas | 1,322 / 3,365 / 38 / 17 | 1,322 / 3,365 / 38 / 17 | unchanged |
| Surface kernel hash | `cf56d45a9da7ce66` | `15db21e1927b0475` | changed |

The total object shrinks even though LLVM repartitioned one former callable
into the main entry, so main-entry size alone is not used as a favorable
metric. Cold JIT timings were not cache-state matched and are deliberately not
claimed here.

## HIP measurement

Each `rocprofv3` trace rendered the same 640x480 image at 64 fixed samples.
Kernel time is normalized by the actual `grid_x * grid_y * grid_z` sum; the
32-work-item difference in candidate run 2 therefore cannot bias the result.

| Build / run | Calls | Work-items | GPU duration (ns) | ns/work-item |
|---|---:|---:|---:|---:|
| Exact parent 1 | 293 | 53,658,304 | 1,456,509,979 | 27.144167266 |
| Exact parent 2 | 293 | 53,658,304 | 1,459,611,970 | 27.201977349 |
| Static routing 1 | 293 | 53,658,304 | 1,443,906,366 | 26.909280733 |
| Static routing 2 | 293 | 53,658,336 | 1,444,719,840 | 26.924424939 |
| Exact-parent mean | | | | 27.173072308 |
| Static-routing mean | | | | 26.916852836 |
| Candidate change | | | | **-0.942917%** |

Render-only wall times are 2.53911 s and 2.54703 s for the parent, versus
2.52713 s and 2.52535 s for static routing. The means are 2.54307 s and
2.52624 s, a **0.661799%** improvement. The normalized surface stage is the
primary decision metric because that is the modified kernel.

## Numerical and visual comparison

The parent and candidate EXRs contain Combined, Normal, the three
albedo/color passes, eight direct/indirect/emission/environment light passes,
and both volume passes. All 15 compared passes contain zero invalid pixels;
the full machine-readable result is
[`all-pass-report.json`](all-pass-report.json).

| Pass | Relative RMSE | Maximum absolute error | Mean/luminance result |
|---|---:|---:|---:|
| Combined | `2.529e-8` | `1.907e-6` | luminance ratio exactly `1.0` |
| Normal | `1.774e-8` | `8.959e-7` | channel means identical |
| DiffCol | `4.838e-5` | `5.006e-3` | luminance ratio `0.9999999295` |

DiffCol's maximum is sparse: its mean absolute error is `4.702e-8`, while its
95th and 99th percentile pixel RMSE values are `8.603e-9` and `3.441e-8`.
The near-zero differences are consistent with changed floating-point atomic
arrival order after the control-flow simplification; no exact-hash tolerance
was weakened.

The following triptychs use the exact parent on the left, static routing in
the center, and an explicitly amplified absolute difference on the right:

- [Combined](triptychs/combined.png)
- [Normal](triptychs/normal.png)
- [Diffuse Color](triptychs/diffcol.png)

Visual inspection at native resolution found no observable structural,
material, geometry, lighting, or orientation difference between the left and
center panels. Difference scales are approximately `2.42e8`, `1.51e7`, and
`3.02e7`, respectively; only at those amplifications do sparse noise- and
edge-shaped residuals become visible.

## Reproduction

```sh
cmake --build build -j32 --target \
  psycles_render_blender_scene \
  psycles_inspect_blender_material \
  psycles_surface_program_metadata_tests

ctest --test-dir build --output-on-failure -j1 \
  -R '^(psycles\.surface_program_metadata|psycles\.surface_svm_(math_immediate|vector_math_immediate|record_immediates)|psycles\.luisa_(microfacet_anisotropy|surface_closure_collection|surface_population|compact_surface_preparation|surface_closure_physical)_(fallback|hip|vk))$'

build/bin/psycles_inspect_blender_material BARBERSHOP_5_2_EXPORT '*'

PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
rocprofv3 --kernel-trace -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene \
  BARBERSHOP_5_2_EXPORT PROFILE_DIR/out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 131072 0 0 1
```

The visual report was generated with `tools/compare_cycles.py` using
`--allow-unverified-build-identity` because both inputs are Psycles A/B
artifacts rather than Blender/Cycles build outputs. Their exact parent and
candidate identities are recorded above; the override is not used for a
Psycles-versus-Cycles compatibility claim.
