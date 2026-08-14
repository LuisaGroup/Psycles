# HIP RayQuery traversal-plan validation

This checkpoint optimizes LuisaCompute's HIP RayQuery implementation without
embedding renderer-specific self-intersection, alpha-test, scene, or material
policy in the backend. Semantic eligibility and profitability are separate:
XIR first proves whether a query can execute synchronously, then HIP LLVM
lowering measures the exact projected callback ABI and chooses between two
native HIPRT traversal plans.

Machine: Radeon RX 9070 XT (`gfx1201`), ROCm 7.2.53211.

## Formal model

### Semantic eligibility

A lowered `RayQueryPipelineInst` whose traversal cannot escape its handlers is
eligible for one synchronous operation. Explicit `proceed`/dispatch control,
motion queries, true 3-D workgroups, and handlers that require reentrant
tracing retain their existing compatible path. This capability decision is
derived from the XIR and handler call graph, not from a renderer or backend
name.

The gfx12 synchronous traversal has two disjoint frontier representations:

1. the ordinary hardware frontier `H`, stored by `ds_bvh_stack`; and
2. at most one parent-link subtree continuation
   `S = (root, node, completed_child)`.

Before expanding a BVH8 node, the implementation switches from `H` to `S` when
the packed hardware entry count is greater than or equal to `capacity - 8`.
Equality must be rejected: filling the circular stack to its encoded capacity
raises its overflow sentinel before `pop1` can free a slot. The eight-entry
reserve is sufficient for one complete BVH8 expansion. While `S` is active,
descending clears `completed_child`; ascending follows the node's parent link
and supplies the completed child as the next skip token. Exhausting `root`
resumes the unchanged hardware frontier. A nested BLAS traversal saves the
TLAS continuation and restores it only after the BLAS root is exhausted.

Every reachable candidate consequently belongs to exactly one of `H` or `S`,
and a candidate transition removes it from that frontier before invoking user
code. A callback is neither replayed nor omitted. An unexpected hardware
overflow traps as a violated invariant instead of returning a false miss.

HIPRT's RTIP 3.1 triangle-pair traversal also received a state-machine fix. It
chooses the nearest unprocessed pair member, marks it processed before the
filter callback, and evaluates the farther member only after rejection or a
later resume. The generic synchronous path contracts the active interval with
HIPRT's monotone `contractRayMaxT` operation rather than relying on a private
traversal-object layout.

### Exact callback projection

The callback environment is compacted with an interprocedural least-fixed-
point demand analysis. For Callable argument `(f, i)`, a normal instruction use
seeds `live(f, i)`. A use that only forwards the value to local callee argument
`(g, j)` adds the equation `live(f, i) |= live(g, j)`. Backward propagation
from the seeds makes forwarding-only SCCs dead. Calls to external or
interposable functions and arguments with ABI attributes conservatively seed
liveness.

For query site `q`, let `D_s(q)` and `D_p(q)` be the demanded surface and
procedural handler arguments. Its stored environment is exactly their dynamic
candidate-kind union:

```text
E(q) = alloc_size(struct(fields(D_s(q) union D_p(q))))
E_module = max_q E(q)
```

Replacing all other formal operands with poison is valid because the fixed
point proves they are unobservable. The maximum is module-wide because the
current HIP module selects one RayQuery state ABI and symbol family.

### Profitability selection

The synchronous plan copies its callback product once per query and reloads it
at accepted candidates. That is profitable for small handlers but increases
private storage and live ranges sharply for a material-heavy renderer. The
backend therefore uses a conservative four-ABI-quantum budget:

```text
E_module <= 64 B  -> synchronous native gfx12 traversal
E_module >  64 B  -> resumable native HIPRT hardware-query traversal
```

The metric is exact; 64 B is an empirical profitability boundary rather than a
semantic theorem. It is intentionally conservative between the measured 48 B
cutout callback, where synchronous traversal is 1.34× faster, and the 192 B
Psycles callbacks, where unconditional synchronization was 1.56× slower. The
runtime suite also exercises a 56 B synchronous environment and several
72–480 B resumable environments.

Selection occurs after projection reaches its least fixed point but before
native linking and LLVM optimization. If the plan is rejected, codegen starts
again from the immutable XIR with the resumable hardware-query ABI. A fresh
module prevents partially projected IR from leaking across attempts, while
the early exit avoids optimizing code that will be discarded. Both plans use
HIPRT's native performance interfaces and the same user handlers; only their
execution ABI and value lifetimes differ.

## Performance

### Cutout microbenchmark

Benchmark: `example_path_tracing_cutout hip --offline --spp 64`, 1024×1024,
HIP register limit 176. FPS is render-only; higher is better.

| Configuration | Median FPS | Relative to old resumable |
|---|---:|---:|
| Old resumable HIP RayQuery | 333.310 | 1.000× |
| ABI-cost selection, 48 B callback | 444.668 | 1.334× |
| Ordinary no-RayQuery trace baseline | 916.774 | 2.751× |

The final clean-rebuild runs were 443.984, 442.996, 448.508, 444.834, 444.767,
434.419, and 444.668 FPS. An independent seven-run replicate measured a
447.302 FPS median, a 0.59% difference. The policy retains the synchronous win
for the compact callback and remains within 0.7% of the earlier unconditional
synchronous median of 447.626 FPS. The remaining
cutout-to-direct ratio is about 2.05× in time and still includes actual
candidate callback and alpha work.

Environment projection on this kernel changes private storage from 480 B to
336 B, removes all four VGPR spills, reduces SGPR spills from ten to two, and
reduces the allocated VGPR count from 176 to 173. A safe 12-entry traversal
stack measured about 429 FPS; 16 entries are retained for synchronous queries.
Ordinary static tracing continues to use eight entries, because raising every
kernel to 16 reduced the direct baseline to roughly 790 FPS.

### Psycles production A/B

Scene: Lone Monk transmission export, 640×480, 64 spp, staged wavefront,
one 64-sample dispatch, and 32-thread continuation blocks. The executor clamps
the logical frame capacity to the 307,200-pixel launch domain; the command's
32,768 persistent-worker argument is inactive for staged wavefront. These runs
use an identical Psycles source tree; only the LuisaCompute submodule changes.
Five warm render-only runs are summarized below.

| HIP RayQuery implementation | Warm median | Versus old baseline |
|---|---:|---:|
| Old resumable control | 1.49477 s | 1.000× |
| Unconditional synchronous plan | 2.32948 s | 1.558× time |
| Exact ABI-cost selection | 1.51818 s | 1.016× time |

The cost model removes 34.8% of the unconditional synchronous render time
(1.534× speedup) and finishes within 1.57% of the old control. Its five warm
runs were 1.51822, 1.51358, 1.51626, 1.51818, and 1.52098 seconds. A final
clean-cache-revision rebuild measured 1.52552 seconds over three warm runs,
0.48% above that median. Cold JIT measurements were 10.7724–11.2196 seconds:
exact projection rejected each 192 B callback before expensive optimization,
then generated the resumable native route.

The locally retained matched Cycles HIP golden uses the same source blend,
camera, frame, RX 9070 XT, 640×480 extent, 64 fixed samples, seed zero,
Tabulated Sobol, and disabled adaptive sampling/denoising. Its recorded
1.89428-second time, and the independent 1.901-second median, bracket
`bpy.ops.render.render(write_still=True)`: they include Cycles synchronization
and EXR output. They must not be divided by Psycles' narrower render-only
interval.

A fresh `rocprofv3` trace supplies a common GPU-work boundary. Summing Cycles'
semantic `kernel_gpu_*` render dispatches gives 869.255 ms; summing Psycles'
generated render dispatches gives 1349.752 ms. Scene construction/HIPRT build,
runtime buffer utilities, transfers, synchronization, and output are excluded
from both sums.

| Renderer | GPU render-kernel sum | Relative time | Relative throughput |
|---|---:|---:|---:|
| Cycles 5.3 HIP | 869.255 ms | 1.000× | 100.0% |
| Psycles HIP, ABI-cost selection | 1349.752 ms | 1.553× | 64.4% |

Thus this Lone Monk gate has not caught Cycles: Psycles still spends 55.3%
more GPU kernel time. Its two RayQuery continuations account for 1307.362 ms,
or 96.9% of its generated-kernel budget, which localizes the next work to
intersection/surface continuation cost rather than wrapper or output time.
This is not a cross-scene claim: the separately matched Barbershop render-only
gate remains about 2.96× slower than Cycles 5.2 HIP.

`rocprofv3` isolates the regression to the two RayQuery continuations. Times
are totals over 240 calls; resources are per dispatch.

| Continuation | Plan | Total | Average | Scratch | VGPR | LDS |
|---|---|---:|---:|---:|---:|---:|
| A | old resumable | 906.954 ms | 3.779 ms | 1888 B | 256 | 0 B |
| A | unconditional sync | 1221.311 ms | 5.089 ms | 2108 B | 256 | 4096 B |
| A | ABI-cost selection | 937.124 ms | 3.905 ms | 1916 B | 256 | 0 B |
| B | old resumable | 365.742 ms | 1.524 ms | 464 B | 184 | 0 B |
| B | unconditional sync | 878.989 ms | 3.662 ms | 656 B | 176 | 4096 B |
| B | ABI-cost selection | 370.238 ms | 1.543 ms | 464 B | 184 | 0 B |

Scene-building kernels and other renderer kernels were unchanged. The resource
restoration, especially continuation B's scratch and the removal of 4 KiB LDS,
confirms that callback materialization—not HIPRT construction or an unrelated
renderer phase—caused the production regression.

## Correctness and visual checks

The final HIP suite passes 1482 assertions in eight tests. Coverage includes
surface/procedural commits, `RayQueryAny` termination, paired-triangle resume,
coincident and reconstructed hits, bounded-stack BLAS/TLAS backtracking,
implicit Callable arguments forwarded through two levels, nested handler
tracing, and both sides of the ABI-cost decision. The deep traversal regression
uses 1024 overlapping BLAS triangles and 256 overlapping TLAS instances and
proves that every callback is observed exactly once. The same executable
passes its fallback baseline (42 assertions); HIP-specific cases are explicitly
skipped. The pure cost/pipeline suite passes 39 assertions in 12 tests.

The final cost-model EXR was compared with the exact old-resumable control for
Combined, Normal, Albedo, all light passes, transmission, and volume. Combined
has RMSE 0.000639180, relative RMSE 0.000409768, MAE 2.04565e-6, luminance mean
ratio 1.000000000, p95 pixel RMSE 3.85e-8, and p99 2.75e-7. Emission,
environment, transmission, and volume passes are bit-identical. The sparse
maximum Combined error is 0.415786; it is confined to stochastic geometry
edges and does not form a coherent feature.

[Lone Monk old-resumable / ABI-cost / amplified-difference triptych](triptychs/lone-monk-before-vs-cost-model.png)

The triptych was inspected at native resolution. The two renders have the same
geometry, silhouette, grass/roof detail, lighting, and occlusion. The amplified
difference contains sparse sample/edge noise only; there are no missing
surfaces, solidified cutouts, duplicated bands, or structured shadow changes.

The independently configured Vulkan canary remains strict native
XIR-to-SPIR-V: native/compatibility route guards and the full SPIR-V codegen
suite continue to pass. Explicit compatibility tests are the only cases that
load DXC.

## Reproduction

```bash
cmake --build build-tests --target \
  test_hip_llvm_pipeline test_hip_ray_query_pipeline \
  example_path_tracing example_path_tracing_cutout -j32

./build-tests/bin/test_hip_llvm_pipeline
./build-tests/bin/test_hip_ray_query_pipeline hip
./build-tests/bin/test_hip_ray_query_pipeline fallback
./build-tests/bin/example_path_tracing_cutout hip --offline --spp 64

./build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-lone-monk-transmission-dbdcb17/export \
  output.ppm hip 640 480 64 64 - 0 0 0 0 64 - 1 0 \
  wavefront-staged 32 32768 32 1 1 0 4 2 4096 131072 0 0 1

rocprofv3 --kernel-trace --memory-copy-trace --stats -f csv \
  -d /var/tmp/cycles-lone-monk-profile -- \
  /home/mike/Projects/blender-install-psycles-trace/blender \
  /home/mike/Downloads/lone-monk_cycles_and_exposure-node_demo.blend \
  --background --python-exit-code 1 \
  --python tools/render_cycles_golden.py -- cycles-lone-monk.exr \
  640 480 64 --cycles-device HIP --device-name 'Radeon RX 9070 XT'
```

Profiler CSVs, timing logs, EXRs, numerical reports, and all pass triptychs are
kept under `/var/tmp/psycles-hip-ray-query-scene-20260814` on the validation
machine. This checkpoint does not claim that the remaining 2.05×
cutout/direct gap is closed; the next backend step is candidate-handler and
frontier-level hardware-counter profiling.
