# Hit-weighted surface-handler transition census

## Outcome

The exact Barbershop census identifies interpreter dispatch as a broad
structural target rather than an isolated Bump or closure-node problem. Of
292,851,737 adjacent ordinary-handler executions, 234,028,046 (79.91%) are
also immediate producer-to-consumer edges. The hottest transitions span Mix,
typed conversion, Color Ramp, Math, Mapping, image lookup, Noise, and vector
operations. A case-by-case optimization of one opcode cannot cover this
mixture.

This measurement also corrects a tempting but invalid premise. Blender 5.2
Cycles does not replace the three finite-difference Bump samples with one
analytic dual evaluation. `ShaderGraph::refine_bump_nodes()` clones the Height
dependency graph for center, DX, and DY; `BumpNode::compile()` places the three
result offsets in `NODE_SET_BUMP`; and `svm_node_set_bump()` consumes them.
Cycles' separate `dual<T>` stack representation propagates ray differentials
for texture filtering and does not eliminate the required Bump samples.
Psycles must therefore preserve the already-verified expanded-DAG semantics.

The supported next experiment is a bounded, scene-specialized
superinstruction overlay: fuse common adjacent bytecode handlers while
preserving their exact handlers and evaluation order. It is not a depth/sample
loop unroll and it must be rejected unless cold HIP evidence improves code
size, compile time, and render time without changing results.

## Formal relation

For topology `t`, let `P[t]` be its immutable preparation program and `H[t]`
its exact surface-population count. Split `P[t]` at every explicit
surface-normal commit. For two adjacent ordinary instructions `a` and `b`
within one segment, define the exact transition class

```text
C(a, b) = (variant(a), handler_key(a), operation(a),
           variant(b), handler_key(b), operation(b), D(a, b))

D(a, b) = exists operand q of b: expanded_address(q) = result_address(a).
```

The reported count is

```text
E[c] = sum_t H[t] * multiplicity(c, P[t]).
```

The device records only the topology event. The host projects it through the
validated final bytecode, resets adjacency at a normal commit, rejects invalid
variant, operand, and address routes, and uses checked 64-bit sums. Thus
`exact=true` establishes the equation without adding a per-instruction device
counter or changing the shader AST. `D` is derived from encoded data flow, not
from authored-node identity or opcode heuristics.

A set of pair classes does not by itself define a legal fusion: adjacent hot
edges may overlap at their middle instruction. The raw top-K sums below are
therefore an oracle upper bound that deliberately ignores matching conflicts.
An implementation must select a deterministic finite dictionary and then mark
non-overlapping edges left-to-right in each immutable program. Its verifier
must prove that every marked head has an in-segment ordinary successor, the
successor has the encoded exact variant, and no instruction belongs to two
pairs. By induction over the resulting partition into singleton and paired
blocks, calling the same handlers in the same order preserves the original
program result and side effects.

## Barbershop evidence

The Blender 5.2 export was rendered on HIP at 320x240, 16 fixed samples, with
compact populate-once execution and the staged wavefront scheduler:

```sh
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export \
  /var/tmp/psycles-handler-transitions-20260828.03Zjc0/barbershop.exr \
  hip 320 240 16 16 - 0 0 0 0 16 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576 \
  - /var/tmp/psycles-handler-transitions-20260828.03Zjc0/barbershop-histogram.json
```

The histogram reports `exact=true`, 3,367,808 surface populations,
300,401,466 value-instruction executions, 2,229,611 normal commits, and 396
exact transition classes. Ordinary transitions total 292,851,737: 79.91% are
direct dependencies and 20.09% are independent adjacency.

| Raw hottest classes | Executions | Transition share |
| ---: | ---: | ---: |
| 1 | 23,936,213 | 8.17% |
| 2 | 36,238,467 | 12.37% |
| 4 | 59,466,662 | 20.31% |
| 8 | 94,561,871 | 32.29% |
| 16 | 138,104,348 | 47.16% |
| 32 | 185,671,486 | 63.40% |
| 64 | 243,871,847 | 83.27% |

The ten hottest exact pairs are all immediate data-flow edges:

| Source -> target | Executions | Share |
| --- | ---: | ---: |
| Mix -> Color to Scalar | 23,936,213 | 8.17% |
| Scalar to Color -> Mix | 12,302,254 | 4.20% |
| Color to Scalar -> Color Ramp | 11,845,387 | 4.04% |
| Color Ramp -> Mix | 11,382,808 | 3.89% |
| Color to Scalar -> Math | 11,207,014 | 3.83% |
| Math -> Scalar to Color | 9,923,453 | 3.39% |
| Mapping -> Image Color | 7,378,629 | 2.52% |
| Math -> Clamp | 6,586,113 | 2.25% |
| Vector Math Vector -> Vector Math Value | 6,425,787 | 2.19% |
| Subtract -> Multiply | 6,425,787 | 2.19% |

These are execution-weighted diagnostic counts, not a predicted speedup.
Fusing a pair can remove only interpreter fetch/dispatch/address overhead; it
does not remove either semantic operation. Code duplication, LLVM inlining
profitability, register pressure, cache behavior, and overlap determine the
real result and must be measured on cold HIP artifacts.

## Regression matrix

The per-sample dispatch regression now requires single and chunked execution
to produce the identical full transition vector. Its one-topology fixture also
proves that every transition count is a nonzero exact multiple of surface
events, at least one transition is a direct dependency, normal commits break
the relation, and transitions are strictly fewer than ordinary handler visits.

After a 32-thread build, the regression passed on fallback and HIP. It also
passed on strict native Vulkan with XIR-to-SPIR-V required and DXC disabled:

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_sample_dispatch_film_tests \
           psycles_render_blender_scene

build/bin/psycles_luisa_sample_dispatch_film_tests fallback
build/bin/psycles_luisa_sample_dispatch_film_tests hip
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
build/bin/psycles_luisa_sample_dispatch_film_tests vk
```

The Vulkan run selected the RX 9070 XT through RADV and completed without the
DXC route. This checkpoint is observational only, so it introduces no render
algorithm or image change and requires no new visual comparison.
