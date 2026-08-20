# Single-evaluation surface population

## Question and model

This experiment removes the production path tracer's three independent
material-graph replays (preparation, next-event BSDF evaluation, and BSDF
sampling). Behind `PSYCLES_POPULATE_SURFACE_ONCE=1`, a surface hit now executes
the typed graph once, populates the original post-setup closures, and lets every
surface consumer read that same set. The legacy callables remain available as
an exact A/B route. No host material evaluation, closure baking, or Cycles
kernel translation is involved.

For a material value DAG `G = (V, E)` and consumer endpoint sets `P` (physical
closures) and `M` (emission), population evaluates the topologically closed
dependency union

```text
D = ancestors(P union M)
```

once. A device value is therefore evaluated at most once per hit. Closure
allocation retains source order and the same Cycles-compatible capacity and
weight-cutoff predicates. The populated set is constructed and consumed in
the shade-surface segment before any shadow or path continuation suspend, so
its local state is not part of the coroutine frame.

This is the dataflow used by Cycles 5.2: `svm_eval_nodes` executes one
sequential bytecode program, fills `ShaderData::closure[]`, and
`integrate_surface` consumes that set for passes, direct-light evaluation, and
sampling. Psycles keeps its own Luisa DSL implementations of every node and
closure; only the execution boundary is aligned.

## Regression coverage

`psycles_luisa_surface_population_tests` uses two materials: a layered
Principled graph whose one Image Texture feeds both Base Color and Emission,
and a Beckmann glass graph. Eight dynamic query scenarios compare the populated
and legacy routes across 32 `float4` records per invocation:

- emission, shading normal, runtime closure flags, and closure trace;
- Combined-related BSDF evaluation, sampled-light evaluation, and sampling;
- Normal, Albedo, glossy/transmission albedo, roughness, and transparency;
- selection identity and all BSSRDF sample fields.

The test also counts host/JIT graph recording and proves that the image node is
recorded exactly once even though it has both physical and emission consumers.
It passed on fallback, HIP, and strict native Vulkan XIR to SPIR-V. Existing
surface-program metadata and closure-collection regressions also passed on all
three backends.

Focused generated-test kernels show that the new boundary removes duplicated
shader AST independently of the full renderer:

| Backend | Legacy | Populate once | Reduction |
|---|---:|---:|---:|
| HIP code object | 113,856 B | 72,896 B | 36.0% |
| Vulkan SPIR-V | 133,486 words | 54,260 words | 59.4% |

## Real Barbershop HIP result

The real Blender 5.2 Barbershop export contains 1,055 geometries, 1,109
instances, and 564 deduplicated material topologies. Both measurements use the
same revision, RX 9070 XT (`gfx1201`), megakernel scheduler, 8x8 image, 256 spp,
and one sample per dispatch. The small launch is intentional: the present giant
megakernel's dynamic private stack faults the HIP driver at a 64x64 launch, on
both the legacy and populated routes; that launch-capacity defect is separate
and remains open.

```text
psycles_render_blender_scene <barbershop-export> <output.exr> hip \
  8 8 256 1 - 0 0 0 0 256 - 1 0 megakernel \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```

The populated side additionally sets `PSYCLES_POPULATE_SURFACE_ONCE=1`.

| Measurement | Legacy replay | Populate once | Result |
|---|---:|---:|---:|
| Render-only | 2.16454 s | 1.06493 s | **2.0326x faster** |
| Warm render repeat | - | 1.07902 s | stable |
| Main HIP code object | 15,861,328 B | 4,965,768 B | 68.7% smaller |
| Luisa to HIP LLVM | 30.7604 s | 34.7412 s | 1.129x slower |
| LLVM to AMDGPU code object | 93.9196 s | 498.008 s | **5.303x slower** |
| Full cold shader JIT | - | 541.736 s | compiler blocker |
| Fixed private segment | 36,512 B | 7,664 B | 79.0% smaller |
| SGPR / VGPR | 108 / 256 | 107 / 256 | effectively unchanged |

The runtime result validates the execution model: eliminating graph replay
more than doubles this surface-heavy render and drastically reduces machine
code and fixed private storage. The cold-compile result rejects the current
closure representation as a production endpoint. It stores every closure as
four unconditional `float4x4` blocks (256 B), whereas Cycles uses an
approximately 80 B tagged base slot and allocates extra typed payload only for
closure families that need it. Profiling the 498-second phase found 64.4% of
samples in one stripped AMD COMGR optimizer routine. This is consistent with
an optimization-complexity problem around the dynamically indexed, fully
flattened local arena, not with source-code volume: the input bitcode is already
66.5% smaller.

The next step is therefore a semantics-preserving closure ABI change: reduce
runtime flags and camera AOVs at population, retain only the physical projection
needed by later BSDF consumers, and use tagged family payloads rather than a
maximal record. The graph is still evaluated once and original closures remain
intact.

## Numerical and visual inspection

The complete 46-channel report is [report.json](report.json). The two populated
runs are byte-identical. Against legacy replay, Combined has RMSE
`2.3483e-7`, maximum absolute error `2.5779e-6`, and mean-luminance ratio
`1.00000012`. Normal has RMSE `5.1448e-9` and maximum absolute error
`5.9605e-8`. The largest all-channel error reported by `idiff` is
`1.0521e-5` in Glossy Indirect; it is non-structural floating-point scheduling
noise.

The triptychs below were opened and inspected at native nearest-neighbor scale.
Both left/right render panels have the same geometry, material, texture, and
lighting structure; only the deliberately amplified difference panels reveal
the numerical noise.

![Legacy replay, populated surface, and Combined difference](triptychs/combined.png)

![Legacy replay, populated surface, and Normal difference](triptychs/normal.png)
