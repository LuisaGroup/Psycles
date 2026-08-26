# Graph-expanded Bump in the compact SVM stream

Commit `ff822727d0a3` removes the recursive Bump-height interpreter from the
production compact surface path. Each material now has one topological typed
SVM program: the height center, X sample, Y sample, and final Bump operation
are ordinary SSA values in that stream.

This follows the architecture of Blender 5.2 Cycles commit `fbe6228777e7`, not
its kernel text. Cycles clones the dependency subgraph of a Bump Height input
for center, DX, and DY in `ShaderGraph::refine_bump_nodes()`
(`intern/cycles/scene/shader_graph.cpp:946`), emits their stack offsets in one
`NODE_SET_BUMP` record (`scene/shader_nodes.cpp:7237`), and consumes those
precomputed samples in `svm_node_set_bump()`
(`kernel/svm/displace.h:20`). Psycles preserves its typed SoA bytecode and
Luisa-generated handlers while adopting the same single-stream data model.

## Formal model and proof obligations

Let the original value program be a strict DAG `G = (V, E)` in topological
order. A differential sample context is the pair `c = (dx, dy)` of scalar
coefficients multiplying the immutable ray differentials. Define

```text
shift(P, c).position        = P.position        + P.dPdx        * dx + P.dPdy        * dy
shift(P, c).object_position = P.object_position + P.object_dPdx * dx + P.object_dPdy * dy
shift(P, c).generated       = P.generated       + P.generated_dx * dx + P.generated_dy * dy
shift(P, c).uv              = P.uv              + P.uv_dx        * dx + P.uv_dy        * dy
shift(P, c).barycentric     = P.barycentric     + P.barycentric_dx * dx + P.barycentric_dy * dy
```

All other `SurfacePoint` fields, including the base normals and the
differentials themselves, are unchanged. This is the same field relation used
by the former recursive `SurfaceBumpEvaluationDomain` path.

For an ordinary value node `v`, `E(v, c)` recursively evaluates its operands
under `c`. A context-sensitive leaf evaluates against `shift(P, c)`; a pure
context-invariant leaf or operation is shared by exact value numbering. For a
Bump node `b`, the expansion is

```text
w  = max(E(filter_width, c), 0)
hc = E(height, c)
hx = E(height, c + (w, 0))
hy = E(height, c + (0, w))
N  = bump_samples(hc, hx, hy,
                  E(strength, c), E(distance, c), w, E(normal, c))
```

The context additions are explicit scalar SSA instructions. Therefore nested
Bump contexts compose instead of overwriting one another, and dependency,
liveness, storage coloring, and bytecode verification see the complete data
flow. The equivalence is exact over real arithmetic. IEEE results may differ
by the rounding of an affine reassociation such as
`dP * (outer + inner)` versus two consecutive shifts; this is a local
floating-point difference, not a structural change.

The implementation enforces the following proof obligations:

- Every source operand must precede its owner. Every emitted operand must
  precede its emitted instruction. Invalid, forward, or already-expanded
  input graphs fail closed with a diagnostic.
- Context-sensitive source operations have a distinct internal opcode whose
  first two operands are the explicit `dx` and `dy`. Static fields and source
  operands retain their named semantic layout.
- Exact GVN equality includes operation, source node, result type, parameter,
  all remapped operands, both integer fields, bitwise float fields, and every
  static-table float. A hash collision is never used as equality.
- Closure, volume, automatic-normal, and displacement endpoints are remapped
  through the root-context value map; no original ID leaks into the expanded
  program.
- `bump_samples` retains Bump's immediate/static contract, while sampled UV,
  attribute, and normal-map operations retain the complete original immediate
  contract. The device opcodes remain distinct from their semantic bases.
- Sampled leaves and `bump_samples` remain spatial sources in homogeneous
  volume analysis. This prevents an expanded spatial graph from being
  incorrectly constant-folded as a homogeneous volume.

Induction over the emitted topological sequence proves that every expanded
instruction computes `E(v, c)` once its operands are available. Applying the
Bump equation at each transformed Bump node establishes the result for the
root context. Endpoint remapping then establishes equivalence for all surface,
emission, BSSRDF, volume, normal, and displacement consumers.

## Permanent regressions

`psycles_surface_program_metadata_tests` now proves:

- a position-dependent Bump becomes one strict topological stream with one
  center source, two sampled-position sources, and no recursive `bump` opcode;
- invariant parameters are shared instead of cloned;
- closure fields, the automatic-normal root, and displacement root are
  remapped to expanded IDs;
- the normal dependency and typed storage plan remain valid;
- a nested position-dependent Bump contains an explicit context addition;
- a source graph with a forward dependency is rejected.

The compact device regression additionally checks that root-program count and
total-program count are equal, the height domain and every recursive target
are empty, and Combined, Normal, Albedo, all light passes, BSSRDF, and Bump
preparation retain their reference results.

Commands, all passing:

```sh
cmake --build build \
  --target psycles_surface_program_metadata_tests \
           psycles_luisa_compact_surface_preparation_tests \
           psycles_render_blender_scene \
  --parallel 32

./build/psycles_surface_program_metadata_tests
./build/bin/psycles_luisa_compact_surface_preparation_tests fallback
./build/bin/psycles_luisa_compact_surface_preparation_tests hip
LUISA_VULKAN_DISABLE_DXC=1 \
  ./build/bin/psycles_luisa_compact_surface_preparation_tests vk

ctest --test-dir build --output-on-failure -j1 \
  -R 'psycles\.(surface_program_metadata|luisa_bump_callable_(fallback|hip|vk))$'
```

The Vulkan command selected the RX 9070 XT, emitted five optimized native
SPIR-V modules of 129531, 19193, 6029, 12706, and 179208 words, and passed with
DXC disabled.

## Barbershop code-size and performance A/B

Both cold artifacts use the same exported scene, RX 9070 XT, 640x480 extent,
64 spp, wavefront-staged scheduler arguments, disabled Psycles shader cache,
fresh COMGR directories, and LLVM/ISA dumps:

```sh
PSYCLES_DISABLE_SHADER_CACHE=1 \
AMD_COMGR_CACHE_DIR=ARTIFACT/comgr \
LUISA_DUMP_HIP_ISA=ARTIFACT/isa \
LUISA_DUMP_LLVM_IR=1 \
build/bin/psycles_render_blender_scene \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export \
  ARTIFACT/out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 131072 0 0 1
```

The paired hot test rebuilt the exact baseline commit `2c77339d1feeb` in a
separate clean worktree, then ran the same command three times per binary with
the normal code-object cache enabled. The table reports every observation and
uses the median, not the best run.

| Metric | Recursive height callable | Expanded stream | Change |
|---|---:|---:|---:|
| Root programs | 378 | 378 | 0% |
| Total programs | 482 | 378 | -21.58% |
| Height programs / strata | 104 / 1 | 0 / 0 | removed |
| Bytecode instructions | 9,295 | 10,301 | +10.82% |
| Bytecode operands | 18,100 | 21,602 | +19.35% |
| Maximum program length | 136 | 230 | +69.12% |
| Typed scalar/vector slots | 6 / 9 | 8 / 9 | +2 / 0 |
| Final main LLVM | 4,054,798 B | 3,632,957 B | -10.40% |
| Final main LLVM lines | 65,354 | 59,741 | -8.59% |
| Main HIP code object | 430,032 B | 381,344 B | -11.32% |
| Main LLVM codegen | 1303.04 ms | 1294.62 ms | -0.65% |
| Main COMGR link | 2916.21 ms | 2591.24 ms | -11.14% |
| Cold shader JIT | 16.1319 s | 15.2505 s | -5.46% |
| Cold render-only | 3.25630 s | 3.08939 s | -5.13% |
| Hot render runs | 3.27470 / 3.25882 / 3.27110 s | 3.07612 / 3.09227 / 3.09567 s | |
| Hot render median | 3.27110 s | 3.09227 s | **-5.47%** |

The material bytecode grows because the sampled height DAG is now data in the
one stream; the generated program shrinks because it no longer records and
calls a second interpreter. That opposing movement is evidence that the
reduction is not caused by dropping material work. Codegen's 0.65% change is
within timing noise; the exact LLVM/ISA reductions, 11.1% COMGR reduction, and
paired 5.47% render median improvement are the supported claims. Full values
and artifact paths are in [metrics.json](metrics.json).

## Numerical and visual inspection

The before/after 15-pass report is
[barbershop-before-after.json](barbershop-before-after.json). Because separate
64 spp GPU executions have stochastic high-energy paths, the intrinsic floor
was measured by comparing two independent renders of the expanded binary in
[barbershop-repeat-floor.json](barbershop-repeat-floor.json):

| Pass | Before/after RMSE | Expanded repeat RMSE | Before/after mean ratio | Expanded repeat mean ratio |
|---|---:|---:|---:|---:|
| Combined | 0.017861 | 0.018191 | 1.05956 | 1.05083 |
| Normal | 0.002049 | 0.001817 | 1.000019 | 1.000017 |
| Diffuse Color | 0.003230 | 0.003105 | 1.000108 | 1.000060 |

The structural change therefore adds no error distinguishable from the
same-binary repeat floor at this sample count. Environment and both volume
passes are exactly equal. The ten unavailable optional image bindings are the
same before and after.

The following generated triptychs were opened at their original 1936x550
resolution and inspected:

- [Combined](triptychs/combined.png): camera, geometry silhouettes, floor and
  cabinet texture boundaries, and lighting structure coincide; the amplified
  difference is sparse path noise.
- [Normal](triptychs/normal.png): no normal island, orientation, or Bump edge
  moves. Differences are isolated subpixel/sample changes.
- [Diffuse Color](triptychs/diffcol.png): UV layout and every visible material
  boundary coincide, including the floor, brick/cabinet surfaces, and ceiling.

Their SHA-256 values are:

```text
f8a17aa9311b0ab95bb836894fd800d00d3c39435d3f3fb07f5adf1ff6c83f4f  combined.png
a2ead7be0aaff39fcb71edea6f07820266e6eb01384bafd51908c93071670e6b  normal.png
9c9fafb49ba2173b6dd74e9fbb0d4bdb2bf2017965061513945a9b1ba02ccc3e  diffcol.png
```
